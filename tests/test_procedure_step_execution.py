"""Orchestration of first-step execution within the `/procedure` flow.

Mirrors the flow:

    READY -> execute procedure.steps[0] only -> store the result
    -> show it in chat -> stop (no further step is executed yet)

The Step Executor itself (`app.procedure.execute_step`) is always mocked
here; its own agentic/MCP-level behavior is covered separately in
`tests/test_procedure_executor.py`. No real model/LLM/MCP call is ever
made in this file.
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

# A four-step procedure (mirrors the manual acceptance scenario): only step 1
# ("Verify that the namespace exists") must ever be invoked this iteration.
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
    """Mock the Step Executor; tests set `.return_value`/`.side_effect`."""
    mock = AsyncMock(return_value=StepResult(status="SUCCESS", summary="Namespace payments exists."))
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


# --- Step 1 is selected deterministically; later steps are never invoked ----

def test_only_step_one_is_executed_for_a_four_step_procedure(config, itsm_mcp, extractor, step_executor):
    start_procedure(config, itsm_mcp, KB_FOUR_STEPS)
    extractor.return_value = {"namespace": "payments"}

    status = asyncio.run(handle_procedure(
        "/procedure inspect namespace health for namespace payments", config,
    ))

    assert status.state == "first_step_completed"
    step_executor.assert_awaited_once()
    executed_step = step_executor.await_args.args[1]
    assert executed_step.id == STEP_1_ID
    assert executed_step.title == STEP_1_TITLE


def test_resolved_inputs_are_passed_to_the_executor(config, itsm_mcp, extractor, step_executor):
    start_procedure(config, itsm_mcp, KB_FOUR_STEPS)
    extractor.return_value = {"namespace": "payments"}

    asyncio.run(handle_procedure("/procedure inspect namespace health for namespace payments", config))

    step_executor.assert_awaited_once()
    run_id, step, declared, collected, original_request, passed_config = step_executor.await_args.args
    assert collected == {"namespace": "payments"}
    assert {item.name for item in declared} == {"namespace"}
    assert original_request == "inspect namespace health for namespace payments"
    assert passed_config is config


# --- SUCCESS ------------------------------------------------------------------

def test_success_result_is_stored_and_shown(config, step_executor):
    step_executor.return_value = StepResult(status="SUCCESS", summary="Namespace payments exists.")
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert status.state == "first_step_completed"
    assert status.message == f"Step completed: {STEP_1_TITLE}\n\nNamespace payments exists."
    assert status.context.status == "FIRST_STEP_COMPLETED"
    assert status.context.step_results[STEP_1_ID] == StepResult(status="SUCCESS", summary="Namespace payments exists.")
    assert status.context.current_step_index == 0


# --- STOP -----------------------------------------------------------------

def test_stop_result_is_stored_and_shown(config, step_executor):
    step_executor.return_value = StepResult(status="STOP", summary="Namespace payments does not exist.")
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert status.state == "stopped"
    assert status.message == f"Step stopped the procedure: {STEP_1_TITLE}\n\nNamespace payments does not exist."
    assert status.context.status == "STOPPED"
    assert status.context.step_results[STEP_1_ID].status == "STOP"


# --- FAILED ---------------------------------------------------------------

def test_failed_result_is_stored_and_shown(config, step_executor):
    step_executor.return_value = StepResult(status="FAILED", summary="Unable to verify the namespace.")
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert status.state == "failed"
    assert status.message == f"Step failed: {STEP_1_TITLE}\n\nUnable to verify the namespace."
    assert status.context.status == "FAILED"
    assert status.context.step_results[STEP_1_ID].status == "FAILED"


def test_unexpected_executor_failure_is_handled_safely(config, step_executor, caplog):
    step_executor.side_effect = RuntimeError("Authorization: Bearer test-openshift-token")
    context = make_ready_context()

    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert status.state == "failed"
    assert status.context.status == "FAILED"
    assert "test-openshift-token" not in caplog.text
    assert "test-openshift-token" not in status.message


# --- Progress stage / sequential messages -----------------------------------

def test_running_step_progress_stage_uses_the_step_title(config, step_executor):
    context = make_ready_context()
    status = asyncio.run(handle_procedure_input_reply("payments", context, config))

    assert status.progress_stages == (f"Running: {STEP_1_TITLE}...",)
    assert status.messages == (
        "Inputs collected:\n\n- Namespace: payments\n\nProcedure is ready to execute.",
        f"Step completed: {STEP_1_TITLE}\n\nNamespace payments exists.",
    )


def test_full_sequence_from_a_fresh_procedure_request(config, itsm_mcp, extractor, step_executor):
    """The full acceptance-test sequence: KB found -> parsed -> ready ->
    step-1 result, all within a single `/procedure` request/response.
    """
    start_procedure(config, itsm_mcp, KB_FOUR_STEPS)
    extractor.return_value = {"namespace": "payments"}

    status = asyncio.run(handle_procedure(
        "/procedure inspect namespace health for namespace payments", config,
    ))

    assert status.progress_stages == ("Parsing procedure", "Extracting inputs", f"Running: {STEP_1_TITLE}...")
    assert len(status.messages) == 4
    assert status.messages[0] == "KB found: Inspect namespace health"
    assert status.messages[2] == (
        "Inputs collected:\n\n- Namespace: payments\n\nProcedure is ready to execute."
    )
    assert status.messages[3] == f"Step completed: {STEP_1_TITLE}\n\nNamespace payments exists."
    assert status.message == status.messages[3]


# --- Normal chat regression after step 1 completes -------------------------

def test_step_results_never_appear_in_logs_as_raw_values(config, step_executor, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="app")
    step_executor.return_value = StepResult(status="SUCCESS", summary="Namespace payments exists.")
    context = make_ready_context()

    asyncio.run(handle_procedure_input_reply("payments", context, config))

    # Only run id / status / counts are logged by the orchestration layer,
    # never the step summary text (the Step Executor's own logging, tested
    # separately, never logs it either).
    assert "Namespace payments exists." not in caplog.text
    assert "Procedure run=run-1 status=FIRST_STEP_COMPLETED" in caplog.text
