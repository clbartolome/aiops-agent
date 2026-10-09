"""Orchestration of step chaining within the `/procedure` flow.

Mirrors the flow:

    READY -> execute procedure.steps[0]
    -> if SUCCESS, execute procedure.steps[1] with step 1's result as
       compact previous-step context
    -> store both results -> show each in chat -> stop
       (no step beyond steps[1] is executed yet)

The Step Executor itself (`app.procedure.execute_step`) is always mocked
here; its own agentic/MCP-level behavior (including `previous_results`
rendering) is covered separately in `tests/test_procedure_executor.py`. No
real model/LLM/MCP call is ever made in this file.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult, TextContent

import app.procedure as procedure_module
from app.procedure import handle_procedure, handle_procedure_input_reply
from app.procedure.models import ProcedureContext, StepResult

# A four-step procedure (mirrors the manual acceptance scenario): only
# steps 1 and 2 ("Verify that the namespace exists", "List pods in the
# namespace") must ever be invoked this iteration.
KB_FOUR_STEPS = """# Inspect namespace health

Inspect the current state of an OpenShift namespace.

## Procedure

**ID:** inspect-namespace-health
**Version:** 1
**Risk:** low
**Confirmation required:** no

## Required information

- **Namespace** — required — OpenShift namespace to inspect.

## Steps

### 1. Verify that the namespace exists

Check that **Namespace** exists.

### 2. List pods in the namespace

Retrieve all pods running in **Namespace**.

### 3. Review recent events

Retrieve recent events from **Namespace**.

### 4. Inspect deployments

Retrieve the deployments configured in **Namespace**.

## Success

The namespace exists and pod/event/deployment information was retrieved.
"""

STEP_1_TITLE = "Verify that the namespace exists"
STEP_1_ID = "verify_that_the_namespace_exists"
STEP_2_TITLE = "List pods in the namespace"
STEP_2_ID = "list_pods_in_the_namespace"
STEP_3_ID = "review_recent_events"

SUCCESS_1 = StepResult(status="SUCCESS", summary="Namespace payments exists.")
SUCCESS_2 = StepResult(status="SUCCESS", summary="Found 8 pods in namespace payments.")


def kb_result(*entries):
    return CallToolResult(content=[
        TextContent(type="text", text=json.dumps({"results": list(entries)}))
    ])


@pytest.fixture
def itsm_mcp(monkeypatch):
    calls = []
    call_tool = AsyncMock()

    async def connect(self):
        pass

    async def cleanup(self):
        pass

    async def traced_call_tool(self, tool_name, arguments, meta=None):
        calls.append((tool_name, arguments))
        return await call_tool(tool_name, arguments)

    monkeypatch.setattr(MCPServerStreamableHttp, "connect", connect)
    monkeypatch.setattr(MCPServerStreamableHttp, "cleanup", cleanup)
    monkeypatch.setattr(MCPServerStreamableHttp, "call_tool", traced_call_tool)
    return SimpleNamespace(calls=calls, call_tool=call_tool)


@pytest.fixture
def extractor(monkeypatch):
    mock = AsyncMock(return_value={})
    monkeypatch.setattr(procedure_module, "extract_procedure_inputs", mock)
    return mock


@pytest.fixture
def step_executor(monkeypatch):
    """Mock the Step Executor; defaults to SUCCESS for every step unless a
    test overrides `.return_value`/`.side_effect` (e.g. a list of results,
    one per expected call, to script step 1 then step 2 differently).
    """
    mock = AsyncMock(return_value=StepResult(status="SUCCESS", summary="Step completed."))
    monkeypatch.setattr(procedure_module, "execute_step", mock)
    return mock


def start_procedure(config, itsm_mcp, kb_content, title="Inspect namespace health"):
    itsm_mcp.call_tool.return_value = kb_result(
        {"id": 1, "title": title, "description": kb_content, "score": 0.9}
    )


def make_ready_context(run_id="run-1") -> ProcedureContext:
    from app.procedure.parser import parse_procedure

    procedure = parse_procedure(KB_FOUR_STEPS)
    return ProcedureContext(
        run_id=run_id, original_request="inspect namespace health for namespace payments",
        kb_title=procedure.title, kb_content=KB_FOUR_STEPS, procedure=procedure,
        inputs={"namespace": "payments"}, status="READY",
    )


# --- Step 1 SUCCESS executes step 2, in order -------------------------------

def test_step_1_success_executes_step_2_in_order(config, step_executor):
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2]
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert step_executor.await_count == 2
    first_step = step_executor.await_args_list[0].args[1]
    second_step = step_executor.await_args_list[1].args[1]
    assert first_step.id == STEP_1_ID
    assert second_step.id == STEP_2_ID
    assert status.state == "second_step_completed"


# --- Step 1 result is passed to step 2 as previous_results ------------------

def test_step_1_result_passed_to_step_2(config, step_executor):
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2]
    context = make_ready_context()

    asyncio.run(handle_procedure_input_reply("payments", context, config))

    second_call_kwargs = step_executor.await_args_list[1].kwargs
    previous = second_call_kwargs["previous_results"]
    assert len(previous) == 1
    assert previous[0].step.id == STEP_1_ID
    assert previous[0].result == SUCCESS_1

    # Step 1 itself was given no previous results.
    first_call_kwargs = step_executor.await_args_list[0].kwargs
    assert first_call_kwargs["previous_results"] == []


# --- Inputs are still passed alongside the previous result ------------------

def test_inputs_also_passed_to_step_2(config, step_executor):
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2]
    context = make_ready_context()

    asyncio.run(handle_procedure_input_reply("payments", context, config))

    _run_id, _step, _declared, collected, _original_request, _config = step_executor.await_args_list[1].args
    assert collected == {"namespace": "payments"}


# --- Step 1 STOP: step 2 never called ---------------------------------------

def test_step_1_stop_never_calls_step_2(config, step_executor):
    step_executor.return_value = StepResult(status="STOP", summary="Namespace payments does not exist.")
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    step_executor.assert_awaited_once()
    assert status.state == "stopped"
    assert STEP_2_ID not in status.context.step_results


# --- Step 1 FAILED: step 2 never called --------------------------------------

def test_step_1_failed_never_calls_step_2(config, step_executor):
    step_executor.return_value = StepResult(status="FAILED", summary="Unable to verify the namespace.")
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    step_executor.assert_awaited_once()
    assert status.state == "failed"
    assert STEP_2_ID not in status.context.step_results


# --- Step 2 STOP --------------------------------------------------------------

def test_step_2_stop_stops_the_procedure(config, step_executor):
    stop_2 = StepResult(status="STOP", summary="No pods found; stopping.")
    step_executor.side_effect = [SUCCESS_1, stop_2]
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert step_executor.await_count == 2
    assert status.state == "stopped"
    assert status.message == f"Step stopped the procedure: {STEP_2_TITLE}\n\nNo pods found; stopping."
    assert status.context.step_results[STEP_1_ID] == SUCCESS_1
    assert status.context.step_results[STEP_2_ID] == stop_2
    assert status.context.status == "STOPPED"


# --- Step 2 FAILED -------------------------------------------------------------

def test_step_2_failed_stops_the_procedure(config, step_executor):
    failed_2 = StepResult(status="FAILED", summary="Unable to list pods.")
    step_executor.side_effect = [SUCCESS_1, failed_2]
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert step_executor.await_count == 2
    assert status.state == "failed"
    assert status.message == f"Step failed: {STEP_2_TITLE}\n\nUnable to list pods."
    # Step 1 remains stored as SUCCESS; it is not retried or overwritten.
    assert status.context.step_results[STEP_1_ID] == SUCCESS_1
    assert status.context.step_results[STEP_1_ID].status == "SUCCESS"
    assert status.context.step_results[STEP_2_ID] == failed_2


# --- Step 3 is never executed, even across a four-step procedure -----------

def test_step_3_is_never_executed_when_both_steps_succeed(config, step_executor):
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2]
    context = make_ready_context()

    asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert step_executor.await_count == 2
    executed_ids = {call.args[1].id for call in step_executor.await_args_list}
    assert executed_ids == {STEP_1_ID, STEP_2_ID}
    assert STEP_3_ID not in executed_ids


def test_only_step_one_is_executed_when_a_four_step_procedure_starts_fresh(
    config, itsm_mcp, extractor, step_executor,
):
    start_procedure(config, itsm_mcp, KB_FOUR_STEPS)
    extractor.return_value = {"namespace": "payments"}
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2]

    status = asyncio.run(handle_procedure(
        "/procedure inspect namespace health for namespace payments", config,
    ))

    assert status.state == "second_step_completed"
    assert step_executor.await_count == 2
    executed_ids = [call.args[1].id for call in step_executor.await_args_list]
    assert executed_ids == [STEP_1_ID, STEP_2_ID]


# --- Both results preserved in context --------------------------------------

def test_both_step_results_are_preserved_in_context(config, step_executor):
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2]
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert status.context.step_results == {STEP_1_ID: SUCCESS_1, STEP_2_ID: SUCCESS_2}
    assert list(status.context.step_results) == [STEP_1_ID, STEP_2_ID]  # source order preserved
    assert status.context.current_step_index == 1


# --- Progress stage / sequential messages -----------------------------------

def test_running_step_progress_stages_for_both_steps(config, step_executor):
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2]
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert status.progress_stages == (f"Running: {STEP_1_TITLE}...", f"Running: {STEP_2_TITLE}...")
    assert status.messages == (
        "Inputs collected:\n\n- Namespace: payments\n\nProcedure is ready to execute.",
        f"Step completed: {STEP_1_TITLE}\n\nNamespace payments exists.",
        f"Step completed: {STEP_2_TITLE}\n\nFound 8 pods in namespace payments.",
    )
    # No hint of step 3 anywhere in the response.
    assert "Review recent events" not in str(status.messages) + str(status.progress_stages)


def test_full_sequence_from_a_fresh_procedure_request(config, itsm_mcp, extractor, step_executor):
    """The full acceptance-test sequence: KB found -> parsed -> ready ->
    step-1 result -> step-2 result, all within a single `/procedure`
    request/response.
    """
    start_procedure(config, itsm_mcp, KB_FOUR_STEPS)
    extractor.return_value = {"namespace": "payments"}
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2]

    status = asyncio.run(handle_procedure(
        "/procedure inspect namespace health for namespace payments", config,
    ))

    assert status.progress_stages == (
        "Parsing procedure", "Extracting inputs", f"Running: {STEP_1_TITLE}...", f"Running: {STEP_2_TITLE}...",
    )
    assert len(status.messages) == 5
    assert status.messages[0] == "KB found: Inspect namespace health"
    assert status.messages[2] == (
        "Inputs collected:\n\n- Namespace: payments\n\nProcedure is ready to execute."
    )
    assert status.messages[3] == f"Step completed: {STEP_1_TITLE}\n\nNamespace payments exists."
    assert status.messages[4] == f"Step completed: {STEP_2_TITLE}\n\nFound 8 pods in namespace payments."
    assert status.message == status.messages[4]


# --- Only-one-step cases still work (procedure with no second step) --------

def test_single_step_procedure_still_stops_after_step_1(config, extractor, step_executor):
    from app.procedure.parser import parse_procedure

    single_step_kb = """# Simple check

## Procedure

**ID:** simple-check
**Version:** 1
**Risk:** low
**Confirmation required:** no

## Steps

### 1. Verify that the namespace exists

Check that the namespace exists.

## Success

Done.
"""
    procedure = parse_procedure(single_step_kb)
    context = ProcedureContext(
        run_id="run-1", original_request="", kb_title=procedure.title, kb_content=single_step_kb,
        procedure=procedure, inputs={}, status="READY",
    )
    step_executor.return_value = SUCCESS_1

    status = asyncio.run(handle_procedure_input_reply("", context, config))

    step_executor.assert_awaited_once()
    assert status.state == "first_step_completed"
    assert status.context.status == "FIRST_STEP_COMPLETED"


# --- Unexpected executor failure handled safely (either step) --------------

def test_unexpected_step_2_failure_is_handled_safely(config, step_executor, caplog):
    step_executor.side_effect = [SUCCESS_1, RuntimeError("Authorization: Bearer test-openshift-token")]
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert status.state == "failed"
    assert status.context.step_results[STEP_1_ID] == SUCCESS_1
    assert status.context.step_results[STEP_2_ID].status == "FAILED"
    assert "test-openshift-token" not in caplog.text
    assert "test-openshift-token" not in status.message


# --- Logging makes the chaining decision visible ----------------------------

def test_next_step_logging_and_no_raw_summary_leakage(config, step_executor, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="app")
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2]
    context = make_ready_context()

    asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert f"Procedure run=run-1 next_step={STEP_2_ID}" in caplog.text
    assert "Procedure run=run-1 status=SECOND_STEP_COMPLETED" in caplog.text
    assert "Namespace payments exists." not in caplog.text
    assert "Found 8 pods" not in caplog.text
