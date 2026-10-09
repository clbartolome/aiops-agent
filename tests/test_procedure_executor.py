"""The Step Executor's own agentic/MCP-level behavior.

Mirrors `tests/test_agent.py`'s mocking strategy exactly: `mcp_boundary`
provides fake-but-realistic MCP sessions (real SDK discovery, execution,
and cleanup), and `OpenAIChatCompletionsModel.get_response` is mocked so
no real model/network call ever happens. The operational phase and the
classification phase share this one mocked method; `respond()` tells them
apart by their distinct system instructions.
"""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from agents import ModelResponse, OpenAIChatCompletionsModel, Usage
from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from app.procedure.executor import (
    CLASSIFIER_INSTRUCTIONS, FAILED_NO_EVIDENCE_SUMMARY, STEP_EXECUTOR_INSTRUCTIONS, execute_step,
    format_previous_step_results,
)
from app.procedure.models import ProcedureInput, ProcedureStep, StepExecutionContext, StepResult

STEP = ProcedureStep(
    id="verify_that_the_namespace_exists", title="Verify that the namespace exists",
    instruction="Check that Namespace exists. If it does not exist, stop the procedure.",
)
STEP_2 = ProcedureStep(
    id="list_pods_in_the_namespace", title="List pods in the namespace",
    instruction="Retrieve all pods running in Namespace.",
)
DECLARED = [ProcedureInput(name="namespace", label="Namespace", required=True)]
COLLECTED = {"namespace": "payments"}


def answer(text):
    return ModelResponse(
        output=[ResponseOutputMessage(
            id="message", role="assistant", status="completed",
            content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
            type="message",
        )],
        usage=Usage(), response_id=None,
    )


def structured_answer(status, summary):
    return answer(json.dumps({"status": status, "summary": summary}))


def call(name, arguments, call_id="call_1"):
    return ModelResponse(
        output=[ResponseFunctionToolCall(
            name=name, arguments=json.dumps(arguments), call_id=call_id, type="function_call",
        )],
        usage=Usage(), response_id=None,
    )


def exposed_name(tools, server, original_name):
    matches = [tool.name for tool in tools if server in tool.name and tool.name.endswith(original_name)]
    assert len(matches) == 1
    return matches[0]


def is_operational(kwargs) -> bool:
    return kwargs["system_instructions"] == STEP_EXECUTOR_INSTRUCTIONS


def is_classifier(kwargs) -> bool:
    return kwargs["system_instructions"] == CLASSIFIER_INSTRUCTIONS


@pytest.fixture
def model(monkeypatch, mcp_boundary):
    monkeypatch.setattr(AsyncOpenAI, "request", AsyncMock(
        side_effect=AssertionError("Tests must not use the network")
    ))
    mock = AsyncMock()
    monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", mock)
    return mock


def test_resolved_inputs_reach_the_operational_prompt(model, config, mcp_boundary):
    """Resolved inputs are formatted as "Label = value" (e.g. "Namespace =
    payments"); the step title/instruction are included verbatim, and no
    future step's content is ever included.
    """
    captured = {}

    def respond(**kwargs):
        if is_operational(kwargs):
            captured["prompt"] = kwargs["input"]
            return answer("Namespace payments exists.")
        assert is_classifier(kwargs)
        return structured_answer("SUCCESS", "Namespace payments exists.")

    model.side_effect = respond
    asyncio.run(execute_step("run-1", STEP, DECLARED, COLLECTED, "inspect namespace payments", config))

    prompt = str(captured["prompt"])
    assert "Namespace = payments" in prompt
    assert STEP.title in prompt
    assert STEP.instruction in prompt
    assert "inspect namespace payments" in prompt
    for server in ("openshift", "aap", "itsm"):
        assert server not in repr(mcp_boundary.sessions[server].call_tool.await_args_list)


def test_successful_mcp_execution_yields_success(model, config, mcp_boundary):
    def respond(**kwargs):
        if is_operational(kwargs):
            if model.await_count == 1:
                return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
            mcp_boundary.sessions["openshift"].call_tool.assert_awaited_once()
            return answer("Namespace payments exists.")
        assert is_classifier(kwargs)
        return structured_answer("SUCCESS", "Namespace payments exists.")

    model.side_effect = respond
    result = asyncio.run(execute_step("run-1", STEP, DECLARED, COLLECTED, "", config))
    assert result.status == "SUCCESS"
    assert result.summary == "Namespace payments exists."


def test_stop_status_is_preserved_when_evidence_supports_it(model, config, mcp_boundary):
    def respond(**kwargs):
        if is_operational(kwargs):
            if model.await_count == 1:
                return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
            return answer("Namespace payments does not exist.")
        assert is_classifier(kwargs)
        return structured_answer("STOP", "Namespace payments does not exist.")

    model.side_effect = respond
    result = asyncio.run(execute_step("run-1", STEP, DECLARED, COLLECTED, "", config))
    assert result.status == "STOP"
    assert result.summary == "Namespace payments does not exist."


def test_failed_when_all_relevant_mcp_calls_fail(model, config, mcp_boundary, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="app")
    mcp_boundary.tool_results["openshift"] = RuntimeError("Query failed")

    def respond(**kwargs):
        if is_operational(kwargs):
            if model.await_count == 1:
                return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
            return answer("Unable to verify whether namespace payments exists.")
        assert is_classifier(kwargs)
        return structured_answer("FAILED", "Unable to verify whether namespace payments exists.")

    model.side_effect = respond
    result = asyncio.run(execute_step("run-1", STEP, DECLARED, COLLECTED, "", config))
    assert result.status == "FAILED"
    assert "mcp_attempted=1 mcp_succeeded=0 mcp_failed=1" in caplog.text


def test_recoverable_tool_error_still_allows_success(model, config, mcp_boundary):
    """A failed call followed by a corrected, successful call must not fail
    the step: multiple MCP calls within one step are expected and allowed.
    """
    def respond(**kwargs):
        if is_operational(kwargs):
            if model.await_count == 1:
                # First attempt: a bad/incomplete call that errors.
                response = call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"bogus": True})
                response.output[0].call_id = "attempt_1"
                return response
            if model.await_count == 2:
                # Corrected, successful retry.
                response = call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
                response.output[0].call_id = "attempt_2"
                return response
            return answer("Namespace payments exists.")
        assert is_classifier(kwargs)
        return structured_answer("SUCCESS", "Namespace payments exists.")

    model.side_effect = respond
    result = asyncio.run(execute_step("run-1", STEP, DECLARED, COLLECTED, "", config))
    assert result.status == "SUCCESS"


def test_no_hallucinated_success_without_mcp_evidence(model, config, mcp_boundary, caplog):
    """The model claims SUCCESS without ever calling a tool: this must be
    rejected and converted to FAILED, never trusted at face value.
    """
    import logging
    caplog.set_level(logging.INFO, logger="app")

    def respond(**kwargs):
        if is_operational(kwargs):
            # The model answers directly, with no tool call at all.
            return answer("Namespace payments exists.")
        assert is_classifier(kwargs)
        return structured_answer("SUCCESS", "Namespace payments exists.")

    model.side_effect = respond
    result = asyncio.run(execute_step("run-1", STEP, DECLARED, COLLECTED, "", config))
    assert result.status == "FAILED"
    assert result.summary == FAILED_NO_EVIDENCE_SUMMARY
    assert "evidence_rejected=true" in caplog.text
    for session in mcp_boundary.sessions.values():
        session.call_tool.assert_not_awaited()


def test_no_hallucinated_stop_without_mcp_evidence(model, config, mcp_boundary):
    """A self-reported STOP is equally rejected without live evidence."""
    def respond(**kwargs):
        if is_operational(kwargs):
            return answer("Namespace payments does not exist.")
        assert is_classifier(kwargs)
        return structured_answer("STOP", "Namespace payments does not exist.")

    model.side_effect = respond
    result = asyncio.run(execute_step("run-1", STEP, DECLARED, COLLECTED, "", config))
    assert result.status == "FAILED"


def test_classifier_never_sees_raw_mcp_payloads(model, config, mcp_boundary):
    """Only the step instruction and the operational phase's own concise
    free-text conclusion reach the classifier -- never raw MCP JSON.
    """
    captured = {}

    def respond(**kwargs):
        if is_operational(kwargs):
            if model.await_count == 1:
                return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"), {"namespace": "payments"})
            return answer("Namespace payments exists.")
        assert is_classifier(kwargs)
        captured["prompt"] = kwargs["input"]
        return structured_answer("SUCCESS", "Namespace payments exists.")

    model.side_effect = respond
    asyncio.run(execute_step("run-1", STEP, DECLARED, COLLECTED, "", config))
    prompt = str(captured["prompt"])
    assert "fake" not in prompt
    assert "pod_count" not in prompt
    assert STEP.instruction in prompt
    assert "Namespace payments exists." in prompt


def test_protocol_error_during_operational_phase_yields_failed(model, config, mcp_boundary):
    """The existing provider-protocol safety net (ProtocolTextError) must
    still apply here; a malformed protocol response must not crash the
    step or leak into the result, and must be reported as FAILED.
    """
    def respond(**kwargs):
        assert is_operational(kwargs)
        return answer('<|start|>assistant<|channel|>commentary to=functions.fake {}')

    model.side_effect = respond
    result = asyncio.run(execute_step("run-1", STEP, DECLARED, COLLECTED, "", config))
    assert result.status == "FAILED"
    assert "<|" not in result.summary


# --- previous_results plumbing (chaining step 1's result into step 2) ------

def test_format_previous_step_results_is_deterministic_and_compact():
    """A plain Python helper, not an LLM call: given the same inputs it
    always produces the same compact text.
    """
    previous = [StepExecutionContext(
        step=STEP, result=StepResult(status="SUCCESS", summary="Namespace payments exists."),
    )]

    text = format_previous_step_results(previous)

    assert text == (
        "Previous completed steps:\n\n"
        "- Verify that the namespace exists\n"
        "  Status: SUCCESS\n"
        "  Summary: Namespace payments exists."
    )


def test_previous_results_reach_the_step_2_prompt(model, config, mcp_boundary):
    """Step 2's prompt includes step 1's title/status/summary as compact
    context, alongside its own title/instruction/resolved inputs -- never
    future steps, never the full chat history.
    """
    captured = {}
    previous = [StepExecutionContext(
        step=STEP, result=StepResult(status="SUCCESS", summary="Namespace payments exists."),
    )]

    def respond(**kwargs):
        if is_operational(kwargs):
            if model.await_count == 1:
                captured["prompt"] = str(kwargs["input"])
                return call(exposed_name(kwargs["tools"], "openshift", "get_pod_count"),
                            {"namespace": "payments"})
            return answer("Found 8 pods in namespace payments.")
        assert is_classifier(kwargs)
        return structured_answer("SUCCESS", "Found 8 pods in namespace payments.")

    model.side_effect = respond
    result = asyncio.run(execute_step(
        "run-1", STEP_2, DECLARED, COLLECTED, "", config, previous_results=previous,
    ))

    assert result.status == "SUCCESS"
    prompt = captured["prompt"]
    assert STEP_2.title in prompt
    assert STEP_2.instruction in prompt
    assert "Namespace = payments" in prompt
    assert "Previous completed steps:" in prompt
    assert STEP.title in prompt
    assert "Status: SUCCESS" in prompt
    assert "Summary: Namespace payments exists." in prompt


def test_no_previous_results_omits_the_section_entirely(model, config, mcp_boundary):
    """Step 1 (no prior steps) must not render an empty "Previous completed
    steps:" section at all.
    """
    captured = {}

    def respond(**kwargs):
        if is_operational(kwargs):
            captured["prompt"] = str(kwargs["input"])
            return answer("Namespace payments exists.")
        assert is_classifier(kwargs)
        return structured_answer("SUCCESS", "Namespace payments exists.")

    model.side_effect = respond
    asyncio.run(execute_step("run-1", STEP, DECLARED, COLLECTED, "", config, previous_results=[]))

    assert "Previous completed steps" not in captured["prompt"]
