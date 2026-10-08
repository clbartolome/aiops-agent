"""End-to-end integration tests for the complete procedure lifecycle (Step 6).

Exercises the full internal stack through the actual application layers used
by the chat UI: `app.procedure.handle_message` -> KB retrieval/parsing ->
`app.procedure.runtime.ProcedureRuntime` (LangGraph) -> the real
`app.procedure.step_executor.make_step_executor` -> the Agents SDK -> MCP.

Only the external boundaries are faked:
  - the model, via mocking `OpenAIChatCompletionsModel.get_response`
    (never a real LLM call);
  - MCP transport, via `e2e_mcp_boundary` (never a real MCP server).

Uses the three realistic KB fixtures under `fixtures/procedures/`, loaded
from disk exactly as the real ITSM KB integration would return them.
"""
import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from agents import ModelResponse, OpenAIChatCompletionsModel, Usage
from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from app.procedure import cancel_active_run, handle_message, resume_procedure_run, start_procedure_run
from app.procedure.runtime import CANCELLED, COMPLETED, FAILED, ProcedureRuntime
from app.procedure.step_executor import make_step_executor
from app.sessions import Conversation

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "procedures"


def load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


def answer(text):
    return ModelResponse(
        output=[ResponseOutputMessage(
            id="message", role="assistant", status="completed",
            content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
            type="message",
        )],
        usage=Usage(), response_id=None,
    )


def call(name, arguments, call_id=None):
    return ModelResponse(
        output=[ResponseFunctionToolCall(
            name=name, arguments=json.dumps(arguments), call_id=call_id or f"call_{name}",
            type="function_call",
        )],
        usage=Usage(), response_id=None,
    )


def step_result_json(outcome="SUCCESS", summary="done", data=None, error=None):
    return json.dumps({"outcome": outcome, "summary": summary, "data": data, "error": error})


def exposed_name(tools, server, original_name):
    matches = [tool.name for tool in tools if server in tool.name and tool.name.endswith(original_name)]
    assert len(matches) == 1, f"expected exactly one match for {server}/{original_name}, got {matches}"
    return matches[0]


@pytest.fixture
def model(monkeypatch, e2e_mcp_boundary):
    # Fail immediately if any test accidentally reaches the HTTP client.
    monkeypatch.setattr(AsyncOpenAI, "request", AsyncMock(
        side_effect=AssertionError("Tests must not use the network")
    ))
    mock = AsyncMock()
    monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", mock)
    return mock


@pytest.fixture
def runtime(config):
    return ProcedureRuntime(step_executor=make_step_executor(config))


# =====================================================================
# Scenario 1: read-only happy path (Fixture A)
# =====================================================================

def test_read_only_happy_path_completes_all_steps(model, config, e2e_mcp_boundary, runtime):
    e2e_mcp_boundary.kb_results = [{
        "title": "Inspect namespace health", "content": load_fixture("inspect_namespace_health.md"),
    }]

    def respond(**kwargs):
        n = model.await_count
        if n == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_namespace_status"), {"namespace": "payments"})
        if n == 2:
            return answer(step_result_json(summary="Namespace payments exists."))
        if n == 3:
            return call(exposed_name(kwargs["tools"], "openshift", "list_pods"), {"namespace": "payments"})
        if n == 4:
            return answer(step_result_json(summary="1 pod running."))
        if n == 5:
            return call(exposed_name(kwargs["tools"], "openshift", "get_recent_events"), {"namespace": "payments"})
        return answer(step_result_json(summary="No abnormal events."))

    model.side_effect = respond
    conversation = Conversation()
    try:
        first = asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation, runtime=runtime,
        ))
        assert "Namespace" in first
        assert conversation.active_procedure_run_id is not None

        second = asyncio.run(resume_procedure_run("namespace=payments", config, conversation, runtime=runtime))

        assert "Procedure completed successfully." in second
        assert "✓ Verify that the namespace exists" in second
        assert "✓ List pods in the namespace" in second
        assert "✓ Review recent events" in second
        # Internal details are never exposed to the user.
        assert "thread_id" not in second and "interrupt" not in second.lower()
        # Active-procedure cleanup on terminal status.
        assert conversation.active_procedure_run_id is None

        assert e2e_mcp_boundary.calls["openshift"] == [
            ("get_namespace_status", {"namespace": "payments"}),
            ("list_pods", {"namespace": "payments"}),
            ("get_recent_events", {"namespace": "payments"}),
        ]
    finally:
        conversation.session.close()


def test_read_only_happy_path_observability_lifecycle_is_reconstructable(
    model, config, e2e_mcp_boundary, runtime, caplog,
):
    caplog.set_level(logging.INFO, logger="app")
    e2e_mcp_boundary.kb_results = [{
        "title": "Inspect namespace health", "content": load_fixture("inspect_namespace_health.md"),
    }]

    def respond(**kwargs):
        n = model.await_count
        if n % 2 == 1:
            names = ["get_namespace_status", "list_pods", "get_recent_events"]
            name = names[n // 2]
            return call(exposed_name(kwargs["tools"], "openshift", name), {"namespace": "payments"})
        return answer(step_result_json(summary="ok"))

    model.side_effect = respond
    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation, runtime=runtime,
        ))
        asyncio.run(resume_procedure_run("namespace=payments", config, conversation, runtime=runtime))
    finally:
        conversation.session.close()

    text = caplog.text
    # The run, procedure, and each step can be correlated end-to-end from the logs.
    assert "procedure_id=inspect-namespace-health" in text
    assert "status=created" in text
    assert "status=resumed" in text
    assert "step_id=verify_that_the_namespace_exists" in text and "status=starting" in text
    assert "status=SUCCESS" in text
    assert "status=COMPLETED" in text
    # No chain-of-thought, no raw argument values, and no secrets are ever logged.
    assert "payments" not in text
    assert config.model_api_key not in text


# =====================================================================
# Scenario 2: STOP (Fixture A, namespace does not exist)
# =====================================================================

def test_stop_outcome_halts_before_later_steps(model, config, e2e_mcp_boundary, runtime):
    e2e_mcp_boundary.kb_results = [{
        "title": "Inspect namespace health", "content": load_fixture("inspect_namespace_health.md"),
    }]

    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_namespace_status"), {"namespace": "ghost"})
        return answer(step_result_json(outcome="STOP", summary="Namespace ghost does not exist."))

    model.side_effect = respond
    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation, runtime=runtime,
        ))
        result = asyncio.run(resume_procedure_run("namespace=ghost", config, conversation, runtime=runtime))

        assert "Procedure stopped as instructed." in result
        assert "✓ Verify that the namespace exists" in result
        assert "Namespace ghost does not exist." in result
        assert "No later steps were executed." in result
        assert conversation.active_procedure_run_id is None
        # list_pods/get_recent_events never ran.
        assert e2e_mcp_boundary.calls["openshift"] == [("get_namespace_status", {"namespace": "ghost"})]
    finally:
        conversation.session.close()


# =====================================================================
# Scenario 3: multiple-input partial collection (Fixture B)
# =====================================================================

def test_multiple_input_partial_collection_asks_only_for_remaining_fields(
    model, config, e2e_mcp_boundary, runtime,
):
    e2e_mcp_boundary.kb_results = [{
        "title": "Check application pods", "content": load_fixture("check_application_pods.md"),
    }]

    def respond(**kwargs):
        n = model.await_count
        if n == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "list_pods"), {"namespace": "payments"})
        if n == 2:
            return answer(step_result_json(summary="1 pod running."))
        if n == 3:
            return call(exposed_name(kwargs["tools"], "openshift", "get_application_logs"), {
                "namespace": "payments", "application_name": "checkout",
            })
        return answer(step_result_json(summary="No errors."))

    model.side_effect = respond
    conversation = Conversation()
    try:
        first = asyncio.run(start_procedure_run(
            "/procedure check application pods", config, conversation, runtime=runtime,
        ))
        assert "Namespace" in first and "Application name" in first

        second = asyncio.run(resume_procedure_run("namespace=payments", config, conversation, runtime=runtime))
        # Still missing exactly the one remaining field; namespace is not re-requested.
        assert "Application name" in second
        assert "Namespace" not in second.replace("Application name", "")
        assert conversation.active_procedure_run_id is not None

        third = asyncio.run(resume_procedure_run(
            "application_name=checkout", config, conversation, runtime=runtime,
        ))
        assert "Procedure completed successfully." in third
        assert conversation.active_procedure_run_id is None
    finally:
        conversation.session.close()


# =====================================================================
# Scenario 4: natural multi-input reply via the constrained extractor
# =====================================================================

def test_natural_language_multi_field_reply_uses_the_extractor_fallback(
    model, config, e2e_mcp_boundary, runtime,
):
    e2e_mcp_boundary.kb_results = [{
        "title": "Check application pods", "content": load_fixture("check_application_pods.md"),
    }]

    def respond(**kwargs):
        n = model.await_count
        if n == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "list_pods"), {"namespace": "payments"})
        if n == 2:
            return answer(step_result_json(summary="1 pod running."))
        if n == 3:
            return call(exposed_name(kwargs["tools"], "openshift", "get_application_logs"), {
                "namespace": "payments", "application_name": "checkout",
            })
        return answer(step_result_json(summary="No errors."))

    model.side_effect = respond

    extractor_calls = []

    async def fake_extractor(message, field_names, cfg):
        extractor_calls.append((message, list(field_names)))
        # The extractor may only return values for the exact requested fields.
        assert set(field_names) <= {"namespace", "application_name"}
        return {"namespace": "payments", "application_name": "checkout"}

    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run(
            "/procedure check application pods", config, conversation, runtime=runtime,
        ))
        result = asyncio.run(resume_procedure_run(
            "namespace is payments and application is checkout", config, conversation,
            runtime=runtime, field_extractor=fake_extractor,
        ))

        assert "Procedure completed successfully." in result
        assert len(extractor_calls) == 1
        assert sorted(extractor_calls[0][1]) == ["application_name", "namespace"]
    finally:
        conversation.session.close()


def test_extractor_is_never_called_for_a_single_missing_field(model, config, e2e_mcp_boundary, runtime):
    """A single requested field is always resolved deterministically; the
    constrained extractor is reserved for multi-field natural-language replies.
    """
    e2e_mcp_boundary.kb_results = [{
        "title": "Inspect namespace health", "content": load_fixture("inspect_namespace_health.md"),
    }]
    model.side_effect = lambda **kwargs: answer(step_result_json(summary="ok"))

    extractor_calls = []

    async def fake_extractor(message, field_names, cfg):
        extractor_calls.append((message, list(field_names)))
        return {}

    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation, runtime=runtime,
        ))
        asyncio.run(resume_procedure_run(
            "the payments namespace please", config, conversation,
            runtime=runtime, field_extractor=fake_extractor,
        ))
        assert extractor_calls == []
    finally:
        conversation.session.close()


# =====================================================================
# Scenario 5: confirmation rejected (Fixture C)
# =====================================================================

def test_confirmation_rejected_cancels_without_executing_any_step(
    model, config, e2e_mcp_boundary, runtime,
):
    e2e_mcp_boundary.kb_results = [{"title": "Launch AAP job", "content": load_fixture("launch_aap_job.md")}]
    conversation = Conversation()
    try:
        first = asyncio.run(start_procedure_run(
            "/procedure launch aap job", config, conversation, runtime=runtime,
        ))
        assert "Job template name" in first

        second = asyncio.run(resume_procedure_run("job_template_name=deploy-app", config, conversation, runtime=runtime))
        assert "Procedure: Launch AAP job" in second
        assert "Risk: medium" in second
        assert "Steps:" in second
        assert "1. Find job template" in second
        assert "2. Review job template" in second
        assert "3. Launch job template" in second
        assert "4. Verify job" in second
        assert "Proceed?" in second
        # MCP tool names are never exposed in the confirmation prompt.
        assert "find_job_template" not in second and "launch_job_template" not in second

        third = asyncio.run(resume_procedure_run("no", config, conversation, runtime=runtime))
        assert "cancel" in third.lower()
        assert conversation.active_procedure_run_id is None
        # Only the KB lookup happened; no procedure step ever ran.
        assert e2e_mcp_boundary.calls.get("aap", []) == []
        assert e2e_mcp_boundary.calls.get("openshift", []) == []
    finally:
        conversation.session.close()


# =====================================================================
# Scenario 6: read-tool self-correction (Fixture A, step 2)
# =====================================================================

def test_read_tool_self_correction_recovers_within_the_bounded_loop(
    model, config, e2e_mcp_boundary, runtime,
):
    e2e_mcp_boundary.kb_results = [{
        "title": "Inspect namespace health", "content": load_fixture("inspect_namespace_health.md"),
    }]

    def respond(**kwargs):
        n = model.await_count
        if n == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_namespace_status"), {"namespace": "payments"})
        if n == 2:
            return answer(step_result_json(summary="Namespace exists."))
        if n == 3:
            # Missing the required argument: rejected locally, never reaches MCP.
            response = call(exposed_name(kwargs["tools"], "openshift", "list_pods"), {})
            response.output[0].call_id = "attempt_1"
            return response
        if n == 4:
            response = call(exposed_name(kwargs["tools"], "openshift", "list_pods"), {"namespace": "payments"})
            response.output[0].call_id = "attempt_2"
            return response
        if n == 5:
            return answer(step_result_json(summary="1 pod running."))
        if n == 6:
            return call(exposed_name(kwargs["tools"], "openshift", "get_recent_events"), {"namespace": "payments"})
        return answer(step_result_json(summary="No abnormal events."))

    model.side_effect = respond
    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation, runtime=runtime,
        ))
        result = asyncio.run(resume_procedure_run("namespace=payments", config, conversation, runtime=runtime))

        assert "Procedure completed successfully." in result
        # Only the corrected call ever reaches the real MCP tool.
        assert e2e_mcp_boundary.calls["openshift"] == [
            ("get_namespace_status", {"namespace": "payments"}),
            ("list_pods", {"namespace": "payments"}),
            ("get_recent_events", {"namespace": "payments"}),
        ]
    finally:
        conversation.session.close()


# =====================================================================
# Scenario 7: all MCP attempts fail -> FAILED, run cleared
# =====================================================================

def test_all_mcp_attempts_failing_produces_a_controlled_failed_result(
    model, config, e2e_mcp_boundary, runtime,
):
    e2e_mcp_boundary.kb_results = [{
        "title": "Inspect namespace health", "content": load_fixture("inspect_namespace_health.md"),
    }]
    e2e_mcp_boundary.tool_results["openshift"] = {"get_namespace_status": RuntimeError("Cluster unreachable")}

    def respond(**kwargs):
        if model.await_count == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_namespace_status"), {"namespace": "payments"})
        return answer(step_result_json(
            outcome="FAILED", summary="The namespace status could not be retrieved.", error="tool_failed",
        ))

    model.side_effect = respond
    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation, runtime=runtime,
        ))
        result = asyncio.run(resume_procedure_run("namespace=payments", config, conversation, runtime=runtime))

        assert "Procedure failed." in result
        assert "✗ Verify that the namespace exists" in result
        assert "No later steps were executed." in result
        assert conversation.active_procedure_run_id is None
        # list_pods/get_recent_events never ran after the failure.
        assert [name for name, _ in e2e_mcp_boundary.calls["openshift"]] == ["get_namespace_status"]
    finally:
        conversation.session.close()


# =====================================================================
# Scenario 8 & 9: risky tool approval (Fixture C) granted / rejected
# =====================================================================

def _launch_job_through_confirmation(model, config, conversation, runtime):
    asyncio.run(start_procedure_run("/procedure launch aap job", config, conversation, runtime=runtime))
    asyncio.run(resume_procedure_run("job_template_name=deploy-app", config, conversation, runtime=runtime))
    return asyncio.run(resume_procedure_run("yes", config, conversation, runtime=runtime))


def test_risky_write_tool_approval_granted_completes_the_procedure(
    model, config, e2e_mcp_boundary, runtime,
):
    e2e_mcp_boundary.kb_results = [{"title": "Launch AAP job", "content": load_fixture("launch_aap_job.md")}]

    def respond(**kwargs):
        n = model.await_count
        if n == 1:
            return call(exposed_name(kwargs["tools"], "aap", "find_job_template"), {"name": "deploy-app"})
        if n == 2:
            return answer(step_result_json(summary="Job template found."))
        if n == 3:
            return call(exposed_name(kwargs["tools"], "aap", "get_job_template_details"), {"name": "deploy-app"})
        if n == 4:
            return answer(step_result_json(summary="Reviewed: deploy.yml on prod inventory."))
        if n in (5, 6):
            # First proposal pauses for approval; the retry after approval
            # is the exact same call, now authorized.
            return call(exposed_name(kwargs["tools"], "aap", "launch_job_template"), {"name": "deploy-app"})
        if n == 7:
            return answer(step_result_json(summary="Job launched."))
        if n == 8:
            return call(exposed_name(kwargs["tools"], "aap", "get_job_status"), {"name": "deploy-app"})
        return answer(step_result_json(summary="Job completed successfully."))

    model.side_effect = respond
    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run("/procedure launch aap job", config, conversation, runtime=runtime))
        asyncio.run(resume_procedure_run("job_template_name=deploy-app", config, conversation, runtime=runtime))

        approval_prompt = asyncio.run(resume_procedure_run("yes", config, conversation, runtime=runtime))
        assert "PROCEDURE · APPROVAL REQUIRED" in approval_prompt
        assert "Launch job template" in approval_prompt
        assert "WRITE operation" in approval_prompt
        assert "Proceed?" in approval_prompt
        # The concrete MCP tool name is never shown outside debug mode.
        assert "launch_job_template" not in approval_prompt
        assert e2e_mcp_boundary.calls.get("aap", []) == [
            ("find_job_template", {"name": "deploy-app"}),
            ("get_job_template_details", {"name": "deploy-app"}),
        ]

        result = asyncio.run(resume_procedure_run("yes", config, conversation, runtime=runtime))
        assert "Procedure completed successfully." in result
        assert "✓ Launch job template" in result
        assert "✓ Verify job" in result
        assert conversation.active_procedure_run_id is None
        assert ("launch_job_template", {"name": "deploy-app"}) in e2e_mcp_boundary.calls["aap"]
    finally:
        conversation.session.close()


def test_risky_write_tool_approval_rejected_cancels_without_executing(
    model, config, e2e_mcp_boundary, runtime,
):
    e2e_mcp_boundary.kb_results = [{"title": "Launch AAP job", "content": load_fixture("launch_aap_job.md")}]

    def respond(**kwargs):
        n = model.await_count
        if n == 1:
            return call(exposed_name(kwargs["tools"], "aap", "find_job_template"), {"name": "deploy-app"})
        if n == 2:
            return answer(step_result_json(summary="Job template found."))
        if n == 3:
            return call(exposed_name(kwargs["tools"], "aap", "get_job_template_details"), {"name": "deploy-app"})
        if n == 4:
            return answer(step_result_json(summary="Reviewed."))
        return call(exposed_name(kwargs["tools"], "aap", "launch_job_template"), {"name": "deploy-app"})

    model.side_effect = respond
    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run("/procedure launch aap job", config, conversation, runtime=runtime))
        asyncio.run(resume_procedure_run("job_template_name=deploy-app", config, conversation, runtime=runtime))
        asyncio.run(resume_procedure_run("yes", config, conversation, runtime=runtime))  # confirmation

        result = asyncio.run(resume_procedure_run("no", config, conversation, runtime=runtime))  # approval

        assert "cancel" in result.lower()
        assert conversation.active_procedure_run_id is None
        # The write tool is never actually called.
        assert all(name != "launch_job_template" for name, _ in e2e_mcp_boundary.calls.get("aap", []))
    finally:
        conversation.session.close()


# =====================================================================
# Scenario 10: out-of-scope write tool rejected (Fixture A, read step)
# =====================================================================

def test_out_of_scope_write_tool_is_rejected_during_a_read_only_step(
    model, config, e2e_mcp_boundary, runtime,
):
    e2e_mcp_boundary.kb_results = [{
        "title": "Inspect namespace health", "content": load_fixture("inspect_namespace_health.md"),
    }]

    def respond(**kwargs):
        n = model.await_count
        if n == 1:
            return call(exposed_name(kwargs["tools"], "openshift", "get_namespace_status"), {"namespace": "payments"})
        if n == 2:
            return answer(step_result_json(summary="Namespace exists."))
        if n == 3:
            # Clearly unrelated to this read-only step; discoverable but out of scope.
            return call(exposed_name(kwargs["tools"], "aap", "launch_job_template"), {"name": "unrelated"})
        if n == 4:
            outputs = [item["output"] for item in kwargs["input"]
                       if isinstance(item, dict) and item.get("type") == "function_call_output"]
            assert any("outside the scope" in output for output in outputs)
            return call(exposed_name(kwargs["tools"], "openshift", "list_pods"), {"namespace": "payments"})
        if n == 5:
            return answer(step_result_json(summary="1 pod running."))
        if n == 6:
            return call(exposed_name(kwargs["tools"], "openshift", "get_recent_events"), {"namespace": "payments"})
        return answer(step_result_json(summary="No abnormal events."))

    model.side_effect = respond
    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation, runtime=runtime,
        ))
        result = asyncio.run(resume_procedure_run("namespace=payments", config, conversation, runtime=runtime))

        assert "Procedure completed successfully." in result
        # The out-of-scope write tool is never actually invoked.
        assert e2e_mcp_boundary.calls.get("aap", []) == []
        assert e2e_mcp_boundary.calls["openshift"] == [
            ("get_namespace_status", {"namespace": "payments"}),
            ("list_pods", {"namespace": "payments"}),
            ("get_recent_events", {"namespace": "payments"}),
        ]
    finally:
        conversation.session.close()


# =====================================================================
# Scenario 11: duplicate risky side effect blocked (Fixture C, launch step)
# =====================================================================

def test_duplicate_risky_side_effect_is_blocked_after_approval(
    model, config, e2e_mcp_boundary, runtime,
):
    e2e_mcp_boundary.kb_results = [{"title": "Launch AAP job", "content": load_fixture("launch_aap_job.md")}]

    def respond(**kwargs):
        n = model.await_count
        if n == 1:
            return call(exposed_name(kwargs["tools"], "aap", "find_job_template"), {"name": "deploy-app"})
        if n == 2:
            return answer(step_result_json(summary="Job template found."))
        if n == 3:
            return call(exposed_name(kwargs["tools"], "aap", "get_job_template_details"), {"name": "deploy-app"})
        if n == 4:
            return answer(step_result_json(summary="Reviewed."))
        if n == 5:
            # First proposal, before approval: pauses the run (no real call yet).
            return call(exposed_name(kwargs["tools"], "aap", "launch_job_template"), {"name": "deploy-app"})
        if n == 6:
            # Step retried after approval: this first proposal executes for real.
            response = call(exposed_name(kwargs["tools"], "aap", "launch_job_template"), {"name": "deploy-app"})
            response.output[0].call_id = "attempt_1"
            return response
        if n == 7:
            # A provider/model retry proposing the exact same already-succeeded call.
            response = call(exposed_name(kwargs["tools"], "aap", "launch_job_template"), {"name": "deploy-app"})
            response.output[0].call_id = "attempt_2"
            return response
        if n == 8:
            outputs = [item["output"] for item in kwargs["input"]
                       if isinstance(item, dict) and item.get("type") == "function_call_output"]
            assert any("already completed" in output for output in outputs)
            return answer(step_result_json(summary="Job already launched."))
        if n == 9:
            return call(exposed_name(kwargs["tools"], "aap", "get_job_status"), {"name": "deploy-app"})
        return answer(step_result_json(summary="Job completed successfully."))

    model.side_effect = respond
    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run("/procedure launch aap job", config, conversation, runtime=runtime))
        asyncio.run(resume_procedure_run("job_template_name=deploy-app", config, conversation, runtime=runtime))
        asyncio.run(resume_procedure_run("yes", config, conversation, runtime=runtime))  # confirmation
        result = asyncio.run(resume_procedure_run("yes", config, conversation, runtime=runtime))  # approval

        assert "Procedure completed successfully." in result
        # Proposed twice after approval, but the real side effect only happens once.
        launch_calls = [c for c in e2e_mcp_boundary.calls["aap"] if c[0] == "launch_job_template"]
        assert launch_calls == [("launch_job_template", {"name": "deploy-app"})]
    finally:
        conversation.session.close()


# =====================================================================
# Scenario 12: cancellation while waiting for input
# =====================================================================

def test_cancel_while_waiting_for_input_executes_no_steps(model, config, e2e_mcp_boundary, runtime):
    e2e_mcp_boundary.kb_results = [{
        "title": "Inspect namespace health", "content": load_fixture("inspect_namespace_health.md"),
    }]
    conversation = Conversation()
    try:
        first = asyncio.run(handle_message(
            "/procedure inspect namespace health", config, conversation, runtime=runtime,
        ))
        assert "Namespace" in first
        assert conversation.active_procedure_run_id is not None

        result = asyncio.run(handle_message("/cancel", config, conversation, runtime=runtime))

        assert "cancel" in result.lower()
        assert conversation.active_procedure_run_id is None
        # Only the KB lookup happened; no procedure step ever ran.
        assert e2e_mcp_boundary.calls.get("openshift", []) == []
    finally:
        conversation.session.close()


# =====================================================================
# Scenario 13: direct-chat regression after procedure completion
# =====================================================================

def test_direct_chat_falls_through_again_after_procedure_completion(
    model, config, e2e_mcp_boundary, runtime,
):
    e2e_mcp_boundary.kb_results = [{
        "title": "Inspect namespace health", "content": load_fixture("inspect_namespace_health.md"),
    }]

    def respond(**kwargs):
        n = model.await_count
        if n % 2 == 1:
            names = ["get_namespace_status", "list_pods", "get_recent_events"]
            return call(exposed_name(kwargs["tools"], "openshift", names[n // 2]), {"namespace": "payments"})
        return answer(step_result_json(summary="ok"))

    model.side_effect = respond
    conversation = Conversation()
    try:
        asyncio.run(handle_message(
            "/procedure inspect namespace health", config, conversation, runtime=runtime,
        ))
        completed = asyncio.run(handle_message(
            "namespace=payments", config, conversation, runtime=runtime,
        ))
        assert "Procedure completed successfully." in completed
        assert conversation.active_procedure_run_id is None

        # The next normal message is no longer routed into procedure mode:
        # `handle_message` returns None so the caller falls through to the
        # existing, unmodified `OperationsAgent` path.
        after = asyncio.run(handle_message("how many pods are running in payments?", config, conversation))
        assert after is None
        assert conversation.active_procedure_run_id is None
    finally:
        conversation.session.close()


# =====================================================================
# Scenario 14: two independent sessions share no state
# =====================================================================

def test_two_independent_sessions_do_not_share_procedure_state(
    model, config, e2e_mcp_boundary, runtime,
):
    e2e_mcp_boundary.kb_results = [{
        "title": "Inspect namespace health", "content": load_fixture("inspect_namespace_health.md"),
    }]

    # Each run interleaves model turns in `start`-then-`resume` order; since
    # the two runs are driven sequentially below (not concurrently), a
    # single flat response sequence covers both runs' turns in order.
    responses = iter([
        lambda kwargs: call(exposed_name(kwargs["tools"], "openshift", "get_namespace_status"), {"namespace": "payments"}),
        lambda kwargs: answer(step_result_json(summary="ok")),
        lambda kwargs: call(exposed_name(kwargs["tools"], "openshift", "list_pods"), {"namespace": "payments"}),
        lambda kwargs: answer(step_result_json(summary="ok")),
        lambda kwargs: call(exposed_name(kwargs["tools"], "openshift", "get_recent_events"), {"namespace": "payments"}),
        lambda kwargs: answer(step_result_json(summary="ok")),
        lambda kwargs: call(exposed_name(kwargs["tools"], "openshift", "get_namespace_status"), {"namespace": "billing"}),
        lambda kwargs: answer(step_result_json(summary="ok")),
        lambda kwargs: call(exposed_name(kwargs["tools"], "openshift", "list_pods"), {"namespace": "billing"}),
        lambda kwargs: answer(step_result_json(summary="ok")),
        lambda kwargs: call(exposed_name(kwargs["tools"], "openshift", "get_recent_events"), {"namespace": "billing"}),
        lambda kwargs: answer(step_result_json(summary="ok")),
    ])
    model.side_effect = lambda **kwargs: next(responses)(kwargs)

    conversation_a = Conversation()
    conversation_b = Conversation()
    try:
        first_a = asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation_a, runtime=runtime,
        ))
        first_b = asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation_b, runtime=runtime,
        ))
        assert conversation_a.active_procedure_run_id != conversation_b.active_procedure_run_id

        result_a = asyncio.run(resume_procedure_run("namespace=payments", config, conversation_a, runtime=runtime))
        result_b = asyncio.run(resume_procedure_run("namespace=billing", config, conversation_b, runtime=runtime))

        assert "Procedure completed successfully." in result_a
        assert "Procedure completed successfully." in result_b
        assert conversation_a.active_procedure_run_id is None
        assert conversation_b.active_procedure_run_id is None
        assert ("get_namespace_status", {"namespace": "payments"}) in e2e_mcp_boundary.calls["openshift"]
        assert ("get_namespace_status", {"namespace": "billing"}) in e2e_mcp_boundary.calls["openshift"]
    finally:
        conversation_a.session.close()
        conversation_b.session.close()
