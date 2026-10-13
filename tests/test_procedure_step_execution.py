"""Orchestration of the generic sequential step loop within the `/procedure`
flow.

Mirrors the flow:

    READY -> RUNNING -> execute procedure.steps in fixed order, one at a
    time, each step receiving every earlier successful step's result as
    compact previous-step context -> continue only while each step returns
    SUCCESS -> stop immediately on STOP or FAILED, preserving every result
    already stored -> COMPLETED once every step has succeeded

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

# A four-step procedure (mirrors the manual acceptance scenario).
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
STEP_3_TITLE = "Review recent events"
STEP_3_ID = "review_recent_events"
STEP_4_TITLE = "Inspect deployments"
STEP_4_ID = "inspect_deployments"
ALL_STEP_IDS = [STEP_1_ID, STEP_2_ID, STEP_3_ID, STEP_4_ID]

SUCCESS_1 = StepResult(status="SUCCESS", summary="Namespace payments exists.")
SUCCESS_2 = StepResult(status="SUCCESS", summary="Found 8 pods in namespace payments.")
SUCCESS_3 = StepResult(status="SUCCESS", summary="No warning events found.")
SUCCESS_4 = StepResult(status="SUCCESS", summary="All deployments are healthy.")
ALL_SUCCESS = [SUCCESS_1, SUCCESS_2, SUCCESS_3, SUCCESS_4]


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
    one per expected call, to script different steps differently).
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


def executed_step_ids(step_executor) -> list[str]:
    return [call.args[1].id for call in step_executor.await_args_list]


# --- All steps succeed -------------------------------------------------------

def test_all_four_steps_succeed_in_order(config, step_executor):
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert step_executor.await_count == 4
    assert executed_step_ids(step_executor) == ALL_STEP_IDS
    assert status.state == "completed"
    assert status.context.status == "COMPLETED"


def test_full_sequence_from_a_fresh_procedure_request(config, itsm_mcp, extractor, step_executor):
    """The full acceptance-test sequence: KB found -> parsed -> ready ->
    every step's result -> the final completion message, all within a
    single `/procedure` request/response.
    """
    start_procedure(config, itsm_mcp, KB_FOUR_STEPS)
    extractor.return_value = {"namespace": "payments"}
    step_executor.side_effect = list(ALL_SUCCESS)

    status = asyncio.run(handle_procedure(
        "/procedure inspect namespace health for namespace payments", config,
    ))

    assert status.progress_stages == (
        "Parsing procedure", "Extracting inputs",
        f"Running: {STEP_1_TITLE}...", f"Running: {STEP_2_TITLE}...",
        f"Running: {STEP_3_TITLE}...", f"Running: {STEP_4_TITLE}...", "Finishing up...",
    )
    assert len(status.messages) == 8
    assert status.messages[0] == "KB found: Inspect namespace health"
    assert status.messages[3] == f"Step completed: {STEP_1_TITLE}\n\nNamespace payments exists."
    assert status.messages[4] == f"Step completed: {STEP_2_TITLE}\n\nFound 8 pods in namespace payments."
    assert status.messages[5] == f"Step completed: {STEP_3_TITLE}\n\nNo warning events found."
    assert status.messages[6] == f"Step completed: {STEP_4_TITLE}\n\nAll deployments are healthy."
    assert status.messages[7] == "Procedure completed successfully."
    assert status.message == "Procedure completed successfully."
    assert status.state == "completed"


# --- Previous context grows ---------------------------------------------------

def test_previous_results_grow_with_each_step(config, step_executor):
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    asyncio.run(handle_procedure_input_reply("payments", context, config))

    previous_counts = [len(call.kwargs["previous_results"]) for call in step_executor.await_args_list]
    assert previous_counts == [0, 1, 2, 3]

    # Original execution order is preserved in step 4's previous results.
    step_4_previous = step_executor.await_args_list[3].kwargs["previous_results"]
    assert [item.step.id for item in step_4_previous] == [STEP_1_ID, STEP_2_ID, STEP_3_ID]
    assert [item.result for item in step_4_previous] == [SUCCESS_1, SUCCESS_2, SUCCESS_3]


def test_inputs_are_passed_to_every_step(config, step_executor):
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    asyncio.run(handle_procedure_input_reply("payments", context, config))

    for call in step_executor.await_args_list:
        _run_id, _step, _declared, collected, _original_request, _config = call.args
        assert collected == {"namespace": "payments"}


# --- STOP on step 1 ------------------------------------------------------------

def test_stop_on_step_1_never_calls_later_steps(config, step_executor):
    step_executor.return_value = StepResult(status="STOP", summary="Namespace payments does not exist.")
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    step_executor.assert_awaited_once()
    assert status.state == "stopped"
    assert status.message == f"Step stopped the procedure: {STEP_1_TITLE}\n\nNamespace payments does not exist."
    assert status.context.status == "STOPPED"
    assert set(status.context.step_results) == {STEP_1_ID}


# --- STOP in the middle ---------------------------------------------------------

def test_stop_in_the_middle_preserves_earlier_results(config, step_executor):
    stop_3 = StepResult(status="STOP", summary="No events endpoint available; stopping.")
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2, stop_3]
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert step_executor.await_count == 3
    assert executed_step_ids(step_executor) == [STEP_1_ID, STEP_2_ID, STEP_3_ID]
    assert status.state == "stopped"
    assert status.context.step_results[STEP_1_ID] == SUCCESS_1
    assert status.context.step_results[STEP_2_ID] == SUCCESS_2
    assert status.context.step_results[STEP_3_ID] == stop_3
    assert status.context.status == "STOPPED"


# --- FAILED in the middle --------------------------------------------------------

def test_failed_in_the_middle_stops_later_steps(config, step_executor):
    failed_2 = StepResult(status="FAILED", summary="Unable to list pods.")
    step_executor.side_effect = [SUCCESS_1, failed_2]
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert step_executor.await_count == 2
    assert executed_step_ids(step_executor) == [STEP_1_ID, STEP_2_ID]
    assert status.state == "failed"
    assert status.context.status == "FAILED"
    assert status.context.step_results[STEP_1_ID] == SUCCESS_1
    assert status.context.step_results[STEP_2_ID] == failed_2


# --- Final step FAILED -----------------------------------------------------------

def test_final_step_failed_keeps_all_earlier_results(config, step_executor):
    failed_4 = StepResult(status="FAILED", summary="Unable to inspect deployments.")
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2, SUCCESS_3, failed_4]
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert step_executor.await_count == 4
    assert status.state == "failed"
    assert status.context.status == "FAILED"
    assert status.context.step_results == {
        STEP_1_ID: SUCCESS_1, STEP_2_ID: SUCCESS_2, STEP_3_ID: SUCCESS_3, STEP_4_ID: failed_4,
    }
    assert status.message == f"Step failed: {STEP_4_TITLE}\n\nUnable to inspect deployments."


# --- Executor exception -----------------------------------------------------------

def test_unexpected_exception_during_step_3_fails_the_procedure(config, step_executor, caplog):
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2, RuntimeError("Authorization: Bearer test-openshift-token")]
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert step_executor.await_count == 3
    assert status.state == "failed"
    assert status.context.status == "FAILED"
    assert status.context.step_results[STEP_1_ID] == SUCCESS_1
    assert status.context.step_results[STEP_2_ID] == SUCCESS_2
    assert status.context.step_results[STEP_3_ID].status == "FAILED"
    assert "test-openshift-token" not in caplog.text
    assert "test-openshift-token" not in status.message


# --- Single-step procedure --------------------------------------------------------

def test_single_step_procedure_completes_normally(config, step_executor):
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
    assert status.state == "completed"
    assert status.context.status == "COMPLETED"
    assert status.messages[-1] == "Procedure completed successfully."


# --- UI order ----------------------------------------------------------------------

def test_running_step_progress_stages_for_every_step(config, step_executor):
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert status.progress_stages == (
        f"Running: {STEP_1_TITLE}...", f"Running: {STEP_2_TITLE}...",
        f"Running: {STEP_3_TITLE}...", f"Running: {STEP_4_TITLE}...", "Finishing up...",
    )
    assert status.messages == (
        "Inputs collected:\n\n- Namespace: payments\n\nProcedure is ready to execute.",
        f"Step completed: {STEP_1_TITLE}\n\nNamespace payments exists.",
        f"Step completed: {STEP_2_TITLE}\n\nFound 8 pods in namespace payments.",
        f"Step completed: {STEP_3_TITLE}\n\nNo warning events found.",
        f"Step completed: {STEP_4_TITLE}\n\nAll deployments are healthy.",
        "Procedure completed successfully.",
    )
    # progress_stages has exactly len(messages) - 1 entries.
    assert len(status.progress_stages) == len(status.messages) - 1


def test_completion_message_shown_exactly_once(config, step_executor):
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert status.messages.count("Procedure completed successfully.") == 1
    assert status.messages[-1] == "Procedure completed successfully."


# --- Results preserved in context --------------------------------------------------

def test_all_step_results_are_preserved_in_order(config, step_executor):
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert status.context.step_results == {
        STEP_1_ID: SUCCESS_1, STEP_2_ID: SUCCESS_2, STEP_3_ID: SUCCESS_3, STEP_4_ID: SUCCESS_4,
    }
    assert list(status.context.step_results) == ALL_STEP_IDS  # source order preserved
    assert status.context.current_step_index == 4


# --- Active-procedure cleanup / normal chat regression -----------------------------

def test_only_fixed_steps_execute_for_a_four_step_procedure_started_fresh(
    config, itsm_mcp, extractor, step_executor,
):
    start_procedure(config, itsm_mcp, KB_FOUR_STEPS)
    extractor.return_value = {"namespace": "payments"}
    step_executor.side_effect = list(ALL_SUCCESS)

    status = asyncio.run(handle_procedure(
        "/procedure inspect namespace health for namespace payments", config,
    ))

    assert status.state == "completed"
    assert step_executor.await_count == 4
    assert executed_step_ids(step_executor) == ALL_STEP_IDS


# --- Logging makes the chaining decision visible ----------------------------

def test_logging_reflects_the_full_sequential_run(config, step_executor, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="app")
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert "Procedure run=run-1 status=RUNNING steps=4" in caplog.text
    for index, step_id in enumerate(ALL_STEP_IDS):
        assert f"Procedure run=run-1 step_index={index} step={step_id} status=starting" in caplog.text
        assert f"Procedure run=run-1 step={step_id} status=SUCCESS" in caplog.text
    assert "Procedure run=run-1 status=COMPLETED" in caplog.text
    # Step summaries are never logged, only run id / status / counts.
    assert "Namespace payments exists." not in caplog.text
    assert "Found 8 pods" not in caplog.text


def test_logging_includes_the_stopping_step_id(config, step_executor, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="app")
    stop_2 = StepResult(status="STOP", summary="No pods found; stopping.")
    step_executor.side_effect = [SUCCESS_1, stop_2]
    context = make_ready_context()

    asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert f"Procedure run=run-1 status=STOPPED step={STEP_2_ID}" in caplog.text
