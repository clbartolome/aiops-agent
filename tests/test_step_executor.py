"""Tests for the real procedure Step Executor (Step 4).

Uses the same mocking boundary as the direct-agent tests
(`OpenAIChatCompletionsModel.get_response` mocked, `mcp_boundary` fakes only
the MCP transport): no real MCP server and no real LLM API is called
anywhere in this file.
"""
import asyncio
import json
import logging

import pytest
from agents import ModelResponse, OpenAIChatCompletionsModel, Usage
from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from unittest.mock import AsyncMock

from app.procedure.models import ProcedureDefinition, ProcedureInput, ProcedureStep, StepResult
from app.procedure.risk import ApprovalRequired, fingerprint_call
from app.procedure.runtime import CANCELLED, COMPLETED, WAITING_FOR_APPROVAL, ProcedureRuntime
from app.procedure.step_executor import make_step_executor


def answer(text):
    return ModelResponse(
        output=[ResponseOutputMessage(
            id="message", role="assistant", status="completed",
            content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
            type="message",
        )],
        usage=Usage(), response_id=None,
    )


def call(name, arguments):
    return ModelResponse(
        output=[ResponseFunctionToolCall(
            name=name, arguments=json.dumps(arguments), call_id=f"call_{name}",
            type="function_call",
        )],
        usage=Usage(), response_id=None,
    )


def step_result_json(outcome="SUCCESS", summary="done", data=None, error=None):
    return json.dumps({"outcome": outcome, "summary": summary, "data": data, "error": error})


def exposed_name(tools, server, original_name):
    matches = [tool.name for tool in tools
               if server in tool.name and tool.name.endswith(original_name)]
    assert len(matches) == 1
    return matches[0]


@pytest.fixture
def model(monkeypatch, mcp_boundary):
    # Fail immediately if any test accidentally reaches the HTTP client.
    monkeypatch.setattr(AsyncOpenAI, "request", AsyncMock(
        side_effect=AssertionError("Tests must not use the network")
    ))
    mock = AsyncMock()
    monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", mock)
    return mock


LIST_PODS_STEP = ProcedureStep(
    id="list-pods", title="List pods in the namespace",
    instruction="Retrieve all pods running in Namespace.", input_refs=["namespace"],
)

VERIFY_NAMESPACE_STEP = ProcedureStep(
    id="verify-namespace", title="Verify that the namespace exists",
    instruction="Verify Namespace exists. If it does not exist, stop the procedure.",
    input_refs=["namespace"],
)


# --- Successful read step -----------------------------------------------------

def test_successful_read_step_returns_success(model, config, mcp_boundary):
    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
        return answer(step_result_json(outcome="SUCCESS", summary="Namespace payments has 3 pods."))

    model.side_effect = respond
    executor = make_step_executor(config)
    result = asyncio.run(executor(LIST_PODS_STEP, {"namespace": "payments"}))

    assert isinstance(result, StepResult)
    assert result.outcome == "SUCCESS"
    assert "3 pods" in result.summary
    mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once_with(
        "get_pod_count", {"namespace": "payments"},
    )


# --- Recoverable tool error ---------------------------------------------------

def test_recoverable_tool_error_does_not_invalidate_the_step(model, config, mcp_boundary):
    def respond(**kwargs):
        if model.await_count == 1:
            # Incorrect arguments: rejected locally, the real MCP tool is never called.
            response = call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {})
            response.output[0].call_id = "attempt_1"
            return response
        if model.await_count == 2:
            response = call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
            response.output[0].call_id = "attempt_2"
            return response
        outputs = [item["output"] for item in kwargs["input"]
                   if isinstance(item, dict) and item.get("type") == "function_call_output"]
        assert any("could not be retrieved" in output for output in outputs)
        return answer(step_result_json(outcome="SUCCESS", summary="3 pods found after correcting the arguments."))

    model.side_effect = respond
    executor = make_step_executor(config)
    result = asyncio.run(executor(LIST_PODS_STEP, {"namespace": "payments"}))

    assert result.outcome == "SUCCESS"
    # Only the corrected call ever reaches the real MCP tool.
    mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once_with(
        "get_pod_count", {"namespace": "payments"},
    )


# --- No successful evidence ---------------------------------------------------

def test_no_successful_evidence_returns_failed(model, config, mcp_boundary):
    mcp_boundary.tool_results["openshift"] = RuntimeError("Query failed")

    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
        return answer(step_result_json(
            outcome="FAILED", summary="Pod information could not be retrieved.", error="tool_failed",
        ))

    model.side_effect = respond
    executor = make_step_executor(config)
    result = asyncio.run(executor(LIST_PODS_STEP, {"namespace": "payments"}))

    assert result.outcome == "FAILED"
    mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once()


# --- STOP outcome --------------------------------------------------------------

def test_stop_outcome_is_returned_when_the_step_explicitly_requires_it(model, config, mcp_boundary):
    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
        return answer(step_result_json(outcome="STOP", summary="Namespace payments does not exist."))

    model.side_effect = respond
    executor = make_step_executor(config)
    result = asyncio.run(executor(VERIFY_NAMESPACE_STEP, {"namespace": "payments"}))

    assert result.outcome == "STOP"
    assert "does not exist" in result.summary
    # STOP is an intentional procedural outcome, never reported as FAILED.
    assert result.error is None


# --- No hallucinated success ---------------------------------------------------

def test_no_hallucinated_success_is_rejected(model, config, mcp_boundary):
    mcp_boundary.tool_results["openshift"] = RuntimeError("Query failed")

    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
        return answer(step_result_json(outcome="SUCCESS", summary="There are 3 pods running."))

    model.side_effect = respond
    executor = make_step_executor(config)
    result = asyncio.run(executor(LIST_PODS_STEP, {"namespace": "payments"}))

    assert result.outcome == "FAILED"
    assert result.error == "no_live_mcp_evidence"
    assert "3 pods" not in result.summary


# --- Bounded scope -------------------------------------------------------------

def test_bounded_scope_does_not_perform_an_unrelated_write_operation(model, config, mcp_boundary):
    seen_tool_names = {}

    def respond(**kwargs):
        seen_tool_names["names"] = [tool.name for tool in kwargs["tools"]]
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
        return answer(step_result_json(outcome="SUCCESS", summary="3 pods listed."))

    model.side_effect = respond
    executor = make_step_executor(config)
    result = asyncio.run(executor(LIST_PODS_STEP, {"namespace": "payments"}))

    assert result.outcome == "SUCCESS"
    # The destructive tool is discoverable (normal Agents SDK tool-calling, no
    # registry/binder restricts it) but a read-only step never calls it.
    assert any("delete" in name for name in seen_tool_names["names"])
    mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once_with(
        "get_pod_count", {"namespace": "payments"},
    )


# --- Missing undeclared information --------------------------------------------

def test_missing_undeclared_information_returns_failed_without_asking(model, config, mcp_boundary):
    step = ProcedureStep(
        id="launch-job", title="Launch the job template",
        instruction="Launch the job template using a value that was never declared as an input.",
        input_refs=[],
    )
    model.return_value = answer(step_result_json(
        outcome="FAILED", summary="The step requires a value that was not provided.",
        error="missing_declared_input",
    ))
    executor = make_step_executor(config)
    result = asyncio.run(executor(step, {}))

    assert result.outcome == "FAILED"
    assert "not provided" in result.summary
    for session in mcp_boundary.sessions.values():
        session.call_tool.assert_not_awaited()
    # No local clarification tool is ever offered to the step executor:
    # pause/resume for missing input stays owned by LangGraph.
    assert all(tool.name != "request_user_input" for tool in model.call_args.kwargs["tools"])


# --- Protocol text ----------------------------------------------------------------

def test_protocol_text_causes_a_controlled_failed_result(model, config, mcp_boundary, caplog):
    caplog.set_level(logging.INFO, logger="app")
    text = '<|start|>assistant<|channel|>commentary to=functions.fake {"secret":"sensitive-test-payload"}'
    model.return_value = answer(text)
    executor = make_step_executor(config)
    result = asyncio.run(executor(LIST_PODS_STEP, {"namespace": "payments"}))

    assert result.outcome == "FAILED"
    assert result.error == "protocol_in_assistant_text"
    assert "<|" not in result.summary
    assert text not in caplog.text
    assert "sensitive-test-payload" not in caplog.text
    for session in mcp_boundary.sessions.values():
        session.call_tool.assert_not_awaited()


# --- Structured result ----------------------------------------------------------

def test_executor_output_validates_against_step_result_schema(model, config, mcp_boundary):
    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
        return answer(step_result_json(outcome="SUCCESS", summary="3 pods found.", data={"pod_count": 3}))

    model.side_effect = respond
    executor = make_step_executor(config)
    result = asyncio.run(executor(LIST_PODS_STEP, {"namespace": "payments"}))

    assert isinstance(result, StepResult)
    assert result.outcome == "SUCCESS"
    assert result.data == {"pod_count": 3}
    assert result.error is None


# --- LangGraph integration -------------------------------------------------------

def test_langgraph_integration_stops_sequencing_after_a_stop_outcome(model, config, mcp_boundary):
    """step1 SUCCESS, step2 SUCCESS, step3 STOP, step4 never executed."""
    definition = ProcedureDefinition(
        id="inspect-namespace-health", version=1, title="Inspect namespace health",
        risk="low", confirmation_required=False,
        inputs=[ProcedureInput(name="namespace", label="Namespace", required=True)],
        steps=[
            ProcedureStep(id="step1", title="Step 1", instruction="Check the namespace.", input_refs=["namespace"]),
            ProcedureStep(id="step2", title="Step 2", instruction="Check the pods.", input_refs=["namespace"]),
            ProcedureStep(id="step3", title="Step 3", instruction="Stop if the condition is met.", input_refs=["namespace"]),
            ProcedureStep(id="step4", title="Step 4", instruction="Must never run.", input_refs=["namespace"]),
        ],
    )
    outcomes = iter(["SUCCESS", "SUCCESS", "STOP"])

    def respond(**kwargs):
        if model.await_count % 2 == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
        return answer(step_result_json(outcome=next(outcomes), summary=f"turn {model.await_count}"))

    model.side_effect = respond
    runtime = ProcedureRuntime(step_executor=make_step_executor(config))
    first = asyncio.run(runtime.start(definition))
    outcome = asyncio.run(runtime.resume(first.run_id, {"namespace": "payments"}))

    assert outcome.status == COMPLETED
    assert set(outcome.step_results) == {"step1", "step2", "step3"}
    # 2 model turns per executed step (one tool call, one structured result);
    # step4 never gets a turn at all.
    assert model.await_count == 6


# --- Tool-risk classification, approval, scope, and duplicate protection -------
# Uses `risk_boundary` (not `mcp_boundary`): each connected server exposes one
# tool at each risk level (`get_status`=READ, `launch_job`=WRITE,
# `delete_<server>`=DESTRUCTIVE, `handle_event`=UNKNOWN by name/description).

LAUNCH_JOB_STEP = ProcedureStep(
    id="launch-job", title="Launch the job template",
    instruction="Launch the job template on the target host.", input_refs=[],
)

DELETE_FAILED_JOB_STEP = ProcedureStep(
    id="delete-failed-job", title="Delete the failed job",
    instruction="Delete the failed job run.", input_refs=[],
)

TRIGGER_EVENT_STEP = ProcedureStep(
    id="trigger-event", title="Trigger the incident handling event",
    instruction="Trigger the incident handling event for the alert.", input_refs=[],
)

INSPECT_PODS_STEP = ProcedureStep(
    id="inspect-pods", title="Inspect pods",
    instruction="Inspect the pods currently running.", input_refs=[],
)


def single_step_definition(step):
    return ProcedureDefinition(
        id="risk-procedure", version=1, title="Risk procedure",
        risk="medium", confirmation_required=False, inputs=[], steps=[step],
    )


def test_write_tool_requires_approval_and_executes_once_after_approval(model, config, risk_boundary):
    def respond(**kwargs):
        if model.await_count in (1, 2):
            return call(exposed_name(kwargs["tools"], "aap", "launch_job"), {"name": "job1"})
        return answer(step_result_json(outcome="SUCCESS", summary="Job launched."))

    model.side_effect = respond
    runtime = ProcedureRuntime(step_executor=make_step_executor(config))
    first = asyncio.run(runtime.start(single_step_definition(LAUNCH_JOB_STEP)))

    assert first.status == WAITING_FOR_APPROVAL
    assert "WRITE" in first.message
    assert "Launch a job" in first.message
    risk_boundary.sessions["aap"].call_tool.assert_not_awaited()

    second = asyncio.run(runtime.resume(first.run_id, True))

    assert second.status == COMPLETED
    risk_boundary.sessions["aap"].call_tool.assert_awaited_once_with("launch_job", {"name": "job1"})


def test_rejecting_tool_approval_cancels_without_executing(model, config, risk_boundary):
    model.side_effect = lambda **kwargs: call(exposed_name(kwargs["tools"], "aap", "launch_job"), {"name": "job1"})
    runtime = ProcedureRuntime(step_executor=make_step_executor(config))
    first = asyncio.run(runtime.start(single_step_definition(LAUNCH_JOB_STEP)))
    assert first.status == WAITING_FOR_APPROVAL

    second = asyncio.run(runtime.resume(first.run_id, False))

    assert second.status == CANCELLED
    risk_boundary.sessions["aap"].call_tool.assert_not_awaited()


def test_destructive_tool_always_requires_approval(model, config, risk_boundary):
    model.side_effect = lambda **kwargs: call(exposed_name(kwargs["tools"], "aap", "delete_aap"), {})
    executor = make_step_executor(config)

    with pytest.raises(ApprovalRequired) as excinfo:
        asyncio.run(executor(DELETE_FAILED_JOB_STEP, {}))

    assert excinfo.value.risk == "DESTRUCTIVE"
    risk_boundary.sessions["aap"].call_tool.assert_not_awaited()


def test_unknown_risk_tool_requires_approval(model, config, risk_boundary):
    model.side_effect = lambda **kwargs: call(exposed_name(kwargs["tools"], "aap", "handle_event"), {})
    executor = make_step_executor(config)

    with pytest.raises(ApprovalRequired) as excinfo:
        asyncio.run(executor(TRIGGER_EVENT_STEP, {}))

    assert excinfo.value.risk == "UNKNOWN"
    risk_boundary.sessions["aap"].call_tool.assert_not_awaited()


def test_out_of_scope_tool_call_is_rejected_and_never_executed(model, config, risk_boundary):
    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "aap", "delete_aap"), {})
        if model.await_count == 2:
            outputs = [item["output"] for item in kwargs["input"]
                       if isinstance(item, dict) and item.get("type") == "function_call_output"]
            assert any("outside the scope" in output for output in outputs)
            return call(exposed_name(kwargs["tools"], "aap", "get_status"), {})
        return answer(step_result_json(outcome="SUCCESS", summary="Pods inspected."))

    model.side_effect = respond
    executor = make_step_executor(config)
    result = asyncio.run(executor(INSPECT_PODS_STEP, {}))

    assert result.outcome == "SUCCESS"
    # Approval alone never makes an out-of-scope call valid: it is rejected
    # deterministically and the real destructive tool is never reached.
    risk_boundary.sessions["aap"].call_tool.assert_awaited_once_with("get_status", {})


def test_duplicate_successful_write_call_is_blocked(model, config, risk_boundary):
    fingerprint = fingerprint_call("launch_job", {"name": "job1"})

    def respond(**kwargs):
        if model.await_count in (1, 2):
            response = call(exposed_name(kwargs["tools"], "aap", "launch_job"), {"name": "job1"})
            response.output[0].call_id = f"attempt_{model.await_count}"
            return response
        outputs = [item["output"] for item in kwargs["input"]
                   if isinstance(item, dict) and item.get("type") == "function_call_output"]
        assert any("already completed" in output for output in outputs)
        return answer(step_result_json(outcome="SUCCESS", summary="Job already launched."))

    model.side_effect = respond
    executor = make_step_executor(config)
    result = asyncio.run(executor(LAUNCH_JOB_STEP, {}, approved_calls=frozenset({fingerprint})))

    assert result.outcome == "SUCCESS"
    # Proposed twice, but the provider/model retry is never allowed to
    # repeat the already-successful side effect.
    risk_boundary.sessions["aap"].call_tool.assert_awaited_once_with("launch_job", {"name": "job1"})


def test_approved_write_tool_that_fails_returns_failed(model, config, risk_boundary):
    risk_boundary.tool_results["aap"] = RuntimeError("Job launch failed")
    fingerprint = fingerprint_call("launch_job", {"name": "job1"})

    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "aap", "launch_job"), {"name": "job1"})
        return answer(step_result_json(
            outcome="FAILED", summary="The job could not be launched.", error="tool_failed",
        ))

    model.side_effect = respond
    executor = make_step_executor(config)
    result = asyncio.run(executor(LAUNCH_JOB_STEP, {}, approved_calls=frozenset({fingerprint})))

    assert result.outcome == "FAILED"
    risk_boundary.sessions["aap"].call_tool.assert_awaited_once_with("launch_job", {"name": "job1"})


def test_audit_record_logs_tool_attempts_and_approval_decisions(model, config, risk_boundary, caplog):
    caplog.set_level(logging.INFO, logger="app")

    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "aap", "get_status"), {})
        if model.await_count == 2:
            return call(exposed_name(kwargs["tools"], "aap", "delete_aap"), {})
        return answer(step_result_json(outcome="SUCCESS", summary="Pods inspected."))

    model.side_effect = respond
    executor = make_step_executor(config)
    result = asyncio.run(executor(INSPECT_PODS_STEP, {}))

    assert result.outcome == "SUCCESS"
    assert "tool=get_status risk=READ approval=not_required status=succeeded" in caplog.text
    assert "tool=delete_aap risk=DESTRUCTIVE approval=not_required status=blocked_scope" in caplog.text
    assert "tools_attempted=2 tools_succeeded=1 tools_failed=1" in caplog.text
    # Never raw argument values or model reasoning, only tool/risk/decision/status.
    assert "job1" not in caplog.text


# --- Direct-agent regression ----------------------------------------------------
# Covered by the unchanged `tests/test_agent.py` suite: `run_agent`/`OperationsAgent`
# are not imported or modified by this module, and all of its tests still pass.
