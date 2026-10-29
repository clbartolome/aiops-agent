"""Orchestration of the generic sequential step loop within the `/procedure`
flow, plus its live-progress delivery: one transient progress bubble per
phase, concise permanent chat messages (no "Step completed:" prefixes or
verbose parsed-step dumps), and a final natural-language summary once the
procedure reaches a terminal state -- all emitted as the backend actually
does the work, never collected and returned only once everything finishes
(see `app.procedure`'s module docstring).

Mirrors the flow:

    READY -> RUNNING -> execute procedure.steps in fixed order, one at a
    time, each step receiving every earlier successful step's result as
    compact previous-step context -> continue only while each step returns
    SUCCESS -> stop immediately on STOP or FAILED, preserving every result
    already stored -> terminal (COMPLETED/STOPPED/FAILED) -> generate and
    show one final summary

The Step Executor itself (`app.procedure.execute_step`) and the final
summarizer (`app.procedure.generate_procedure_summary`) are always mocked
here; the Step Executor's own agentic/MCP-level behavior is covered
separately in `tests/test_procedure_executor.py`. No real model/LLM/MCP
call is ever made in this file.
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
from tests.conftest import prepare_and_approve, drain_procedure_events, event_types, message_texts, progress_texts

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

DEFAULT_SUMMARY = "Procedure completed successfully."


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
def step_executor(monkeypatch, summarizer):
    """Mock the Step Executor; defaults to SUCCESS for every step unless a
    test overrides `.return_value`/`.side_effect` (e.g. a list of results,
    one per expected call, to script different steps differently). Also
    depends on `summarizer` (see `conftest.py`): every terminal run now
    generates a final summary too, which must never make a real model call.
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
        confirmed=True,
        run_id=run_id, original_request="inspect namespace health for namespace payments",
        kb_title=procedure.title, kb_content=KB_FOUR_STEPS, procedure=procedure,
        inputs={"namespace": "payments"}, status="READY",
    )


def executed_step_ids(step_executor) -> list[str]:
    return [call.args[1].id for call in step_executor.await_args_list]


def run_procedure(message, config):
    return asyncio.run(prepare_and_approve(message, config))


def run_reply(message, context, config):
    return asyncio.run(drain_procedure_events(handle_procedure_input_reply(message, context, config)))


# --- All steps succeed -------------------------------------------------------

def test_all_four_steps_succeed_in_order(config, step_executor):
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    _events, status = run_reply("payments", context, config)

    assert step_executor.await_count == 4
    assert executed_step_ids(step_executor) == ALL_STEP_IDS
    assert status.state == "completed"
    assert status.context.status == "COMPLETED"


def test_full_sequence_from_a_fresh_procedure_request(config, itsm_mcp, extractor, step_executor):
    """The full acceptance-test sequence: KB found -> loaded -> ready ->
    every step's plain result summary -> the final summary message, all
    within a single `/procedure` request, delivered live.
    """
    start_procedure(config, itsm_mcp, KB_FOUR_STEPS)
    extractor.return_value = {"namespace": "payments"}
    step_executor.side_effect = list(ALL_SUCCESS)

    events, status = run_procedure(
        "/procedure inspect namespace health for namespace payments", config,
    )

    assert progress_texts(events) == [
        "Searching KB", "Parsing procedure", "Extracting parameters",
        f"Running step 1: {STEP_1_TITLE}", f"Running step 2: {STEP_2_TITLE}",
        f"Running step 3: {STEP_3_TITLE}", f"Running step 4: {STEP_4_TITLE}", "Generating procedure summary",
    ]
    messages = message_texts(events)
    assert messages == [
        "KB found: Inspect namespace health",
        "Procedure found: Inspect namespace health\n\nSteps: 4\nRisk: low\n\nDo you want to run this procedure?",
        "Using Namespace: payments.",
        "Namespace payments exists.",
        "Found 8 pods in namespace payments.",
        "No warning events found.",
        "All deployments are healthy.",
        DEFAULT_SUMMARY,
    ]
    assert status.message == DEFAULT_SUMMARY
    assert status.state == "completed"
    # No verbose parsed-step dump, and no mechanical "Step completed:" /
    # internal-status noise anywhere in the permanent chat messages.
    joined = "\n".join(messages)
    assert "Required inputs:" not in joined
    assert "Step completed:" not in joined
    assert "current_step_index" not in joined
    assert "status=RUNNING" not in joined


# --- Previous context grows ---------------------------------------------------

def test_previous_results_grow_with_each_step(config, step_executor):
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    run_reply("payments", context, config)

    previous_counts = [len(call.kwargs["previous_results"]) for call in step_executor.await_args_list]
    assert previous_counts == [0, 1, 2, 3]

    # Original execution order is preserved in step 4's previous results.
    step_4_previous = step_executor.await_args_list[3].kwargs["previous_results"]
    assert [item.step.id for item in step_4_previous] == [STEP_1_ID, STEP_2_ID, STEP_3_ID]
    assert [item.result for item in step_4_previous] == [SUCCESS_1, SUCCESS_2, SUCCESS_3]


def test_inputs_are_passed_to_every_step(config, step_executor):
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    run_reply("payments", context, config)

    for call in step_executor.await_args_list:
        _run_id, _step, _declared, collected, _original_request, _config = call.args
        assert collected == {"namespace": "payments"}


# --- Required live ordering: PROGRESS(N) -> MESSAGE(N) -> PROGRESS(N+1) ----------

def test_each_step_result_is_revealed_before_the_next_steps_progress(config, step_executor):
    """For every successful step, the client must observe exactly this
    order: `PROGRESS(step N)`, then the step's own result message, then
    `PROGRESS(step N+1)` -- never two progress events back to back before
    the first step's result is shown, and never everything batched at
    the end.
    """
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2]
    from app.procedure.parser import parse_procedure
    two_step_kb = KB_FOUR_STEPS.replace(
        "### 3. Review recent events\n\nRetrieve recent events from **Namespace**.\n\n"
        "### 4. Inspect deployments\n\nRetrieve the deployments configured in **Namespace**.\n\n",
        "",
    )
    procedure = parse_procedure(two_step_kb)
    context = ProcedureContext(
        confirmed=True,
        run_id="run-1", original_request="", kb_title=procedure.title, kb_content=two_step_kb,
        procedure=procedure, inputs={"namespace": "payments"}, status="READY",
    )

    events, _status = run_reply("payments", context, config)

    # events[0] is the "ready" message; events[7:] is the trailing
    # "Generating procedure summary" progress/clear/message triple. Only
    # the two steps' own events matter for this ordering assertion.
    assert events[1:7] == [
        {"type": "procedure_progress", "message": f"Running step 1: {STEP_1_TITLE}"},
        {"type": "procedure_progress_clear"},
        {"type": "procedure_message", "message": SUCCESS_1.summary},
        {"type": "procedure_progress", "message": f"Running step 2: {STEP_2_TITLE}"},
        {"type": "procedure_progress_clear"},
        {"type": "procedure_message", "message": SUCCESS_2.summary},
    ]
    assert len(events) == 10


# --- STOP on step 1 ------------------------------------------------------------

def test_stop_on_step_1_never_calls_later_steps(config, step_executor, summarizer):
    summarizer.return_value = "The procedure stopped because the namespace does not exist."
    step_executor.return_value = StepResult(status="STOP", summary="Namespace payments does not exist.")
    context = make_ready_context()

    events, status = run_reply("payments", context, config)

    step_executor.assert_awaited_once()
    assert status.state == "stopped"
    messages = message_texts(events)
    # The step's own plain summary is a permanent message...
    assert "Namespace payments does not exist." in messages
    # ...and the final summary -- not the raw step text -- is the response's
    # own top-level `.message` (and the last permanent message).
    assert status.message == "The procedure stopped because the namespace does not exist."
    assert messages[-1] == status.message
    assert status.context.status == "STOPPED"
    assert set(status.context.step_results) == {STEP_1_ID}
    # No step beyond step 1 ever got a "Running step N: ..." progress bubble.
    stages = progress_texts(events)
    assert all(not stage.startswith("Running step 2") for stage in stages)
    assert stages[-1] == "Generating procedure summary"


# --- STOP in the middle ---------------------------------------------------------

def test_stop_in_the_middle_preserves_earlier_results(config, step_executor):
    stop_3 = StepResult(status="STOP", summary="No events endpoint available; stopping.")
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2, stop_3]
    context = make_ready_context()

    events, status = run_reply("payments", context, config)

    assert step_executor.await_count == 3
    assert executed_step_ids(step_executor) == [STEP_1_ID, STEP_2_ID, STEP_3_ID]
    assert status.state == "stopped"
    assert status.context.step_results[STEP_1_ID] == SUCCESS_1
    assert status.context.step_results[STEP_2_ID] == SUCCESS_2
    assert status.context.step_results[STEP_3_ID] == stop_3
    assert status.context.status == "STOPPED"
    # Step 4's progress bubble never appeared.
    stages = progress_texts(events)
    assert not any(stage.startswith("Running step 4") for stage in stages)
    # A final summary was still generated.
    assert stages[-1] == "Generating procedure summary"
    assert message_texts(events)[-1] == DEFAULT_SUMMARY


# --- FAILED in the middle --------------------------------------------------------

def test_failed_in_the_middle_stops_later_steps(config, step_executor):
    failed_2 = StepResult(status="FAILED", summary="Unable to list pods.")
    step_executor.side_effect = [SUCCESS_1, failed_2]
    context = make_ready_context()

    events, status = run_reply("payments", context, config)

    assert step_executor.await_count == 2
    assert executed_step_ids(step_executor) == [STEP_1_ID, STEP_2_ID]
    assert status.state == "failed"
    assert status.context.status == "FAILED"
    assert status.context.step_results[STEP_1_ID] == SUCCESS_1
    assert status.context.step_results[STEP_2_ID] == failed_2
    stages = progress_texts(events)
    assert not any(stage.startswith("Running step 3") for stage in stages)
    assert stages[-1] == "Generating procedure summary"


# --- Final step FAILED -----------------------------------------------------------

def test_final_step_failed_keeps_all_earlier_results(config, step_executor):
    failed_4 = StepResult(status="FAILED", summary="Unable to inspect deployments.")
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2, SUCCESS_3, failed_4]
    context = make_ready_context()

    events, status = run_reply("payments", context, config)

    assert step_executor.await_count == 4
    assert status.state == "failed"
    assert status.context.status == "FAILED"
    assert status.context.step_results == {
        STEP_1_ID: SUCCESS_1, STEP_2_ID: SUCCESS_2, STEP_3_ID: SUCCESS_3, STEP_4_ID: failed_4,
    }
    assert "Unable to inspect deployments." in message_texts(events)
    assert status.message == DEFAULT_SUMMARY


# --- Executor exception, and progress always clears on failure (try/finally-like) -

def test_unexpected_exception_during_step_3_fails_the_procedure(config, step_executor, caplog):
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2, RuntimeError("Authorization: Bearer test-openshift-token")]
    context = make_ready_context()

    events, status = run_reply("payments", context, config)

    assert step_executor.await_count == 3
    assert status.state == "failed"
    assert status.context.status == "FAILED"
    assert status.context.step_results[STEP_1_ID] == SUCCESS_1
    assert status.context.step_results[STEP_2_ID] == SUCCESS_2
    assert status.context.step_results[STEP_3_ID].status == "FAILED"
    assert "test-openshift-token" not in caplog.text
    assert "test-openshift-token" not in status.message
    assert "test-openshift-token" not in "\n".join(message_texts(events))

    # The step-3 progress bubble was shown and then cleared even though the
    # Step Executor raised -- no stale loading bubble remains on failure.
    step_3_progress = {"type": "procedure_progress", "message": f"Running step 3: {STEP_3_TITLE}"}
    position = events.index(step_3_progress)
    assert events[position + 1] == {"type": "procedure_progress_clear"}


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
        confirmed=True,
        run_id="run-1", original_request="", kb_title=procedure.title, kb_content=single_step_kb,
        procedure=procedure, inputs={}, status="READY",
    )
    step_executor.return_value = SUCCESS_1

    events, status = run_reply("", context, config)

    step_executor.assert_awaited_once()
    assert status.state == "completed"
    assert status.context.status == "COMPLETED"
    assert message_texts(events)[-1] == DEFAULT_SUMMARY


# --- UI order ----------------------------------------------------------------------

def test_running_step_progress_stages_for_every_step(config, step_executor):
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    events, _status = run_reply("payments", context, config)

    assert progress_texts(events) == [
        f"Running step 1: {STEP_1_TITLE}", f"Running step 2: {STEP_2_TITLE}",
        f"Running step 3: {STEP_3_TITLE}", f"Running step 4: {STEP_4_TITLE}", "Generating procedure summary",
    ]
    assert message_texts(events) == [
        "Using Namespace: payments.",
        "Namespace payments exists.",
        "Found 8 pods in namespace payments.",
        "No warning events found.",
        "All deployments are healthy.",
        DEFAULT_SUMMARY,
    ]
    # Exactly one progress-show/clear pair precedes every later message:
    # no two progress events are ever adjacent, and no message is ever
    # shown without first clearing whatever progress preceded it.
    assert event_types(events) == [
        "procedure_message",
        "procedure_progress", "procedure_progress_clear", "procedure_message",
        "procedure_progress", "procedure_progress_clear", "procedure_message",
        "procedure_progress", "procedure_progress_clear", "procedure_message",
        "procedure_progress", "procedure_progress_clear", "procedure_message",
        "procedure_progress", "procedure_progress_clear", "procedure_message",
    ]


def test_summary_shown_exactly_once(config, step_executor):
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    events, _status = run_reply("payments", context, config)

    messages = message_texts(events)
    stages = progress_texts(events)
    assert messages.count(DEFAULT_SUMMARY) == 1
    assert messages[-1] == DEFAULT_SUMMARY
    assert stages.count("Generating procedure summary") == 1
    assert stages[-1] == "Generating procedure summary"


# --- Summary model failure: deterministic fallback --------------------------------

def test_summary_generation_failure_falls_back_deterministically(config, step_executor, summarizer):
    summarizer.return_value = None
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    _events, status = run_reply("payments", context, config)

    # The procedure's own outcome is unaffected by the summarizer failing.
    assert status.state == "completed"
    assert status.context.status == "COMPLETED"
    assert status.message == "Procedure completed successfully.\n\n4 of 4 steps completed."


def test_summary_generation_exception_falls_back_deterministically(config, step_executor, summarizer):
    summarizer.side_effect = RuntimeError("Authorization: Bearer test-openshift-token")
    failed_3 = StepResult(status="FAILED", summary="Unable to review events.")
    step_executor.side_effect = [SUCCESS_1, SUCCESS_2, failed_3]
    context = make_ready_context()

    _events, status = run_reply("payments", context, config)

    assert status.state == "failed"
    assert status.context.status == "FAILED"
    assert status.message == f"Procedure failed at: {STEP_3_TITLE}.\n\n2 previous steps completed successfully."
    assert "test-openshift-token" not in status.message


def test_summary_generation_failure_fallback_for_stop(config, step_executor, summarizer):
    summarizer.return_value = None
    stop_2 = StepResult(status="STOP", summary="No pods found; stopping.")
    step_executor.side_effect = [SUCCESS_1, stop_2]
    context = make_ready_context()

    _events, status = run_reply("payments", context, config)

    assert status.state == "stopped"
    assert status.message == f"Procedure stopped at: {STEP_2_TITLE}.\n\n1 previous step completed successfully."


# --- Results preserved in context --------------------------------------------------

def test_all_step_results_are_preserved_in_order(config, step_executor):
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    _events, status = run_reply("payments", context, config)

    assert status.context.step_results == {
        STEP_1_ID: SUCCESS_1, STEP_2_ID: SUCCESS_2, STEP_3_ID: SUCCESS_3, STEP_4_ID: SUCCESS_4,
    }
    assert list(status.context.step_results) == ALL_STEP_IDS  # source order preserved
    assert status.context.current_step_index == 4


# --- No progress indicator while waiting on input -----------------------------------

def test_no_progress_bubble_remains_while_waiting_for_input(config, itsm_mcp, extractor):
    start_procedure(config, itsm_mcp, KB_FOUR_STEPS)
    extractor.return_value = {}

    events, status = run_procedure("/procedure inspect namespace health", config)

    assert status.state == "collecting_inputs"
    assert status.message == "What namespace should I use?"
    # The question itself is the final permanent message, and the very
    # last event overall is a progress_clear (or the message itself) --
    # never a dangling, still-open progress event.
    assert message_texts(events)[-1] == "What namespace should I use?"
    assert event_types(events)[-1] == "procedure_message"


def test_no_extraction_progress_for_the_deterministic_single_field_shortcut(config, step_executor):
    """When exactly one input is missing, a plain-text reply is stored
    directly with no extractor call at all -- so no "Extracting
    parameters" bubble is needed either (see spec: optional to skip it).
    """
    from app.procedure.parser import parse_procedure

    procedure = parse_procedure(KB_FOUR_STEPS)
    context = ProcedureContext(
        confirmed=True,
        run_id="run-1", original_request="inspect namespace health", kb_title=procedure.title,
        kb_content=KB_FOUR_STEPS, procedure=procedure, inputs={}, status="COLLECTING_INPUTS",
    )
    step_executor.side_effect = list(ALL_SUCCESS)

    events, status = run_reply("payments", context, config)

    assert status.state == "completed"
    assert "Extracting parameters" not in progress_texts(events)


# --- Active-procedure cleanup / normal chat regression -----------------------------

def test_only_fixed_steps_execute_for_a_four_step_procedure_started_fresh(
    config, itsm_mcp, extractor, step_executor,
):
    start_procedure(config, itsm_mcp, KB_FOUR_STEPS)
    extractor.return_value = {"namespace": "payments"}
    step_executor.side_effect = list(ALL_SUCCESS)

    _events, status = run_procedure(
        "/procedure inspect namespace health for namespace payments", config,
    )

    assert status.state == "completed"
    assert step_executor.await_count == 4
    assert executed_step_ids(step_executor) == ALL_STEP_IDS


# --- Logging makes the chaining decision visible (backend observability) ----------

def test_logging_reflects_the_full_sequential_run(config, step_executor, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="app")
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    run_reply("payments", context, config)

    assert "Procedure run=run-1 status=RUNNING steps=4" in caplog.text
    for index, step_id in enumerate(ALL_STEP_IDS):
        assert f"Procedure run=run-1 step_index={index} step={step_id} status=starting" in caplog.text
        assert f"Procedure run=run-1 step={step_id} status=SUCCESS" in caplog.text
    assert "Procedure run=run-1 status=COMPLETED" in caplog.text
    assert "Procedure run=run-1 summary_generation=llm" in caplog.text
    # Step summaries are never logged, only run id / status / counts.
    assert "Namespace payments exists." not in caplog.text
    assert "Found 8 pods" not in caplog.text


def test_logging_includes_the_stopping_step_id(config, step_executor, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="app")
    stop_2 = StepResult(status="STOP", summary="No pods found; stopping.")
    step_executor.side_effect = [SUCCESS_1, stop_2]
    context = make_ready_context()

    run_reply("payments", context, config)

    assert f"Procedure run=run-1 status=STOPPED step={STEP_2_ID}" in caplog.text


def test_logging_records_fallback_summary_generation(config, step_executor, summarizer, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="app")
    summarizer.return_value = None
    step_executor.side_effect = list(ALL_SUCCESS)
    context = make_ready_context()

    run_reply("payments", context, config)

    assert "Procedure run=run-1 summary_generation=started" in caplog.text
    assert "Procedure run=run-1 summary_generation=fallback" in caplog.text
