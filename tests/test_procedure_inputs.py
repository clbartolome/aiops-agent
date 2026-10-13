"""Procedure input extraction and missing-input collection.

Mirrors the flow:

    parsed ProcedureDefinition -> apply defaults -> extract from original
    request -> determine missing required inputs -> ask only for what is
    missing -> repeat on each reply -> READY (no step execution yet)

No real LLM call is ever made here: `app.procedure.extract_procedure_inputs`
is always mocked.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult, TextContent

import app.procedure as procedure_module
from app.procedure import handle_cancel, handle_procedure, handle_procedure_input_reply
from app.procedure.models import ProcedureContext, StepResult
from tests.conftest import drain_procedure_events, message_texts

# A single required input, no default.
KB_SINGLE_INPUT = """# Inspect namespace health

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

## Success

The namespace exists.
"""

# Two required inputs; mirrors the "inspect checkout in namespace payments" example.
KB_TWO_INPUTS = """# Validate application state

Validate that an application exists in an OpenShift namespace.

## Procedure

**ID:** validate-application-state
**Version:** 1
**Risk:** low
**Confirmation required:** no

## Required information

- **Namespace** — required — OpenShift namespace containing the application.
- **Application name** — required — Name of the application to inspect.

## Steps

### 1. Retrieve the application

Retrieve **Application name** from **Namespace**.

## Success

The application was found.
"""

# One required input plus one optional input with an explicit default.
KB_WITH_DEFAULT = """# Validate application state

Validate that an application exists and has enough pods.

## Procedure

**ID:** validate-application-state
**Version:** 1
**Risk:** low
**Confirmation required:** no

## Required information

- **Namespace** — required — OpenShift namespace containing the application.
- **Minimum pod count** — optional — Minimum expected number of running pods. Defaults to `1`.

## Steps

### 1. Verify the minimum pod count

Check pod counts against **Minimum pod count**.

## Success

Enough pods are running.
"""


def kb_result(*entries):
    return CallToolResult(content=[
        TextContent(type="text", text=json.dumps({"results": list(entries)}))
    ])


@pytest.fixture
def itsm_mcp(monkeypatch):
    """Patch the MCP server lifecycle so only `rag_search_kb` calls are observed."""
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
    """Mock the structured extractor; tests set `.return_value`/`.side_effect`.

    Never calls a real model/LLM.
    """
    mock = AsyncMock(return_value={})
    monkeypatch.setattr(procedure_module, "extract_procedure_inputs", mock)
    return mock


@pytest.fixture
def step_executor(monkeypatch, summarizer):
    """Mock the Step Executor; tests set `.return_value`/`.side_effect`.

    Defaults to a successful step so tests that reach READY (and therefore
    trigger step execution, and then final-summary generation -- see the
    `summarizer` fixture dependency) do not need to care about either
    unless they are specifically testing step execution. Never calls a
    real model/MCP tool.
    """
    mock = AsyncMock(return_value=StepResult(status="SUCCESS", summary="Step completed."))
    monkeypatch.setattr(procedure_module, "execute_step", mock)
    return mock


def start_procedure(config, itsm_mcp, kb_content, title="Validate application state"):
    itsm_mcp.call_tool.return_value = kb_result(
        {"id": 1, "title": title, "description": kb_content, "score": 0.9}
    )


def make_context(kb_content: str, inputs: dict | None = None, status: str = "COLLECTING_INPUTS",
                  run_id: str = "run-1") -> ProcedureContext:
    """Build a `ProcedureContext` directly from already-parsed KB content, for
    tests that only exercise the reply/continuation path.
    """
    from app.procedure.parser import parse_procedure
    procedure = parse_procedure(kb_content)
    return ProcedureContext(
        run_id=run_id, original_request="", kb_title=procedure.title, kb_content=kb_content,
        procedure=procedure, inputs=inputs or {}, status=status,
    )


def run_procedure(message, config):
    return asyncio.run(drain_procedure_events(handle_procedure(message, config)))


def run_reply(message, context, config):
    return asyncio.run(drain_procedure_events(handle_procedure_input_reply(message, context, config)))


# --- Value already present in the original /procedure request --------------

def test_value_already_in_request_asks_nothing(config, itsm_mcp, extractor, step_executor):
    start_procedure(config, itsm_mcp, KB_SINGLE_INPUT)
    extractor.return_value = {"namespace": "payments"}

    events, status = run_procedure(
        "/procedure inspect namespace health for namespace payments", config,
    )

    # Once READY, the (single) step is executed automatically within the
    # same call (see the step-execution tests below); the compact "ready"
    # message is still shown as an intermediate sequential message.
    assert status.state == "completed"
    assert status.context.inputs == {"namespace": "payments"}
    assert "What namespace" not in status.message
    messages = message_texts(events)
    assert "Using Namespace: payments." in messages
    assert "Step completed." in messages
    assert status.message == "Procedure completed successfully."


# --- Missing input ------------------------------------------------------------

def test_missing_input_is_asked(config, itsm_mcp, extractor):
    start_procedure(config, itsm_mcp, KB_SINGLE_INPUT)
    extractor.return_value = {}

    _events, status = run_procedure("/procedure inspect namespace health", config)

    assert status.state == "collecting_inputs"
    assert status.message == "What namespace should I use?"
    assert status.context.status == "COLLECTING_INPUTS"
    assert "namespace" not in status.context.inputs


# --- Single-value reply: deterministic shortcut, no LLM ---------------------

def test_single_value_reply_uses_no_llm_shortcut(config, extractor, step_executor):
    context = make_context(KB_SINGLE_INPUT, inputs={}, status="COLLECTING_INPUTS")

    _events, status = run_reply("payments", context, config)

    extractor.assert_not_awaited()
    assert status.state == "completed"
    assert status.context.inputs == {"namespace": "payments"}
    assert status.context.status == "COMPLETED"
    assert status.context.run_id == context.run_id


# --- Multiple values already in the original request ------------------------

def test_multiple_values_in_original_request(config, itsm_mcp, extractor, step_executor):
    start_procedure(config, itsm_mcp, KB_TWO_INPUTS)
    extractor.return_value = {"namespace": "payments", "application_name": "checkout"}

    _events, status = run_procedure(
        "/procedure inspect checkout in namespace payments", config,
    )

    assert status.state == "completed"
    assert status.context.inputs == {"namespace": "payments", "application_name": "checkout"}


# --- Partial original request: only the remaining input is asked ------------

def test_partial_original_request_asks_only_remaining(config, itsm_mcp, extractor):
    start_procedure(config, itsm_mcp, KB_TWO_INPUTS)
    extractor.return_value = {"namespace": "payments"}  # application_name not found

    _events, status = run_procedure(
        "/procedure inspect checkout in namespace payments", config,
    )

    assert status.state == "collecting_inputs"
    assert status.context.inputs == {"namespace": "payments"}
    assert status.message == "What application name should I use?"
    assert "Namespace" not in status.message


# --- Multi-field natural reply ------------------------------------------------

def test_multi_field_reply_extracts_both_and_becomes_ready(config, extractor, step_executor):
    context = make_context(KB_TWO_INPUTS, inputs={}, status="COLLECTING_INPUTS")
    extractor.return_value = {"namespace": "payments", "application_name": "checkout"}

    _events, status = run_reply(
        "namespace is payments and the app is checkout", context, config,
    )

    extractor.assert_awaited_once()
    called_args = extractor.await_args.args
    assert called_args[0] == "namespace is payments and the app is checkout"
    assert {item.name for item in called_args[1]} == {"namespace", "application_name"}

    assert status.state == "completed"
    assert status.context.inputs == {"namespace": "payments", "application_name": "checkout"}


# --- Partial reply: only the remaining field is asked afterward -------------

def test_partial_reply_asks_only_remaining_field(config, extractor):
    context = make_context(KB_TWO_INPUTS, inputs={}, status="COLLECTING_INPUTS")
    extractor.return_value = {"namespace": "payments"}

    _events, status = run_reply("namespace is payments", context, config)

    assert status.state == "collecting_inputs"
    assert status.context.inputs == {"namespace": "payments"}
    assert status.message == "What application name should I use?"


# --- No invented input --------------------------------------------------------

def test_extractor_cannot_invent_missing_input(config, extractor):
    # Two fields missing so the LLM path is used (one missing field uses the
    # deterministic shortcut instead, see section 13).
    context = make_context(KB_TWO_INPUTS, inputs={}, status="COLLECTING_INPUTS")
    extractor.return_value = {"namespace": "payments"}  # application_name not provided by the user

    _events, status = run_reply("the namespace is payments", context, config)

    assert status.context.inputs == {"namespace": "payments"}
    assert "application_name" not in status.context.inputs
    assert status.state == "collecting_inputs"


# --- Unknown extractor field rejected ----------------------------------------

def test_unknown_extractor_field_is_rejected(config, extractor):
    # Two missing fields are needed to reach the LLM path (one missing field
    # uses the deterministic shortcut instead, see section 13).
    context = make_context(KB_TWO_INPUTS, inputs={}, status="COLLECTING_INPUTS")
    extractor.return_value = {"cluster": "prod"}  # "cluster" is not declared

    _events, status = run_reply("the cluster is prod", context, config)

    assert status.context.inputs == {}
    assert "cluster" not in status.context.inputs
    assert status.state == "collecting_inputs"


# --- Defaults ------------------------------------------------------------------

def test_declared_defaults_are_applied_before_extraction(config, itsm_mcp, extractor, step_executor):
    start_procedure(config, itsm_mcp, KB_WITH_DEFAULT)
    extractor.return_value = {"namespace": "payments"}

    events, status = run_procedure("/procedure validate app in namespace payments", config)

    # The default was applied deterministically before extraction ran at all.
    assert status.context.inputs["minimum_pod_count"] == 1
    assert status.state == "completed"
    # No question was ever asked about it; it is only shown for visibility,
    # in the intermediate "ready" message (step 1 now runs automatically).
    messages = message_texts(events)
    assert not any("should I use" in message for message in messages)
    assert any("Minimum pod count: 1" in message for message in messages)


def test_optional_input_with_default_is_never_asked(config, itsm_mcp, extractor):
    start_procedure(config, itsm_mcp, KB_WITH_DEFAULT)
    extractor.return_value = {}  # namespace missing; minimum_pod_count already defaulted

    _events, status = run_procedure("/procedure validate app", config)

    assert status.state == "collecting_inputs"
    assert status.message == "What namespace should I use?"
    assert status.context.inputs == {"minimum_pod_count": 1}


# --- Resume same procedure context -------------------------------------------

def test_reply_updates_same_context_not_a_new_one(config, extractor, step_executor):
    context = make_context(KB_TWO_INPUTS, inputs={"namespace": "payments"}, status="COLLECTING_INPUTS",
                            run_id="existing-run")
    extractor.return_value = {}

    _events, status = run_reply("checkout", context, config)

    assert status.context.run_id == "existing-run"
    assert status.context.inputs["namespace"] == "payments"
    # Only "application_name" was missing, so the deterministic single-field
    # shortcut stores the bare reply directly, with no LLM call.
    assert status.context.inputs["application_name"] == "checkout"
    extractor.assert_not_awaited()


# --- Extraction failure recovery ---------------------------------------------

def test_extraction_failure_during_initial_request_asks_directly(config, itsm_mcp, extractor, caplog):
    start_procedure(config, itsm_mcp, KB_SINGLE_INPUT)
    extractor.side_effect = RuntimeError("provider failure Authorization: Bearer test-openshift-token")

    _events, status = run_procedure(
        "/procedure inspect namespace health mentioning something", config,
    )

    assert status.state == "collecting_inputs"
    assert status.message == "I couldn't extract the namespace from that message.\n\nWhat namespace should I use?"
    assert status.context.inputs == {}
    assert "test-openshift-token" not in caplog.text


def test_extraction_failure_during_reply_preserves_context(config, extractor, caplog):
    context = make_context(KB_TWO_INPUTS, inputs={}, status="COLLECTING_INPUTS")
    extractor.side_effect = RuntimeError("boom")

    _events, status = run_reply("something ambiguous", context, config)

    assert status.state == "collecting_inputs"
    assert status.context.inputs == {}
    assert status.context.run_id == context.run_id
    assert "I still need:" in status.message
    assert "Namespace" in status.message and "Application name" in status.message


def test_invalid_extracted_field_type_does_not_crash(config, extractor):
    # An undeclared/unexpected field is simply dropped, never crashes the flow.
    context = make_context(KB_TWO_INPUTS, inputs={}, status="COLLECTING_INPUTS")
    extractor.return_value = {"namespace": "payments", "unexpected_field": "value"}

    _events, status = run_reply("payments namespace, something else too", context, config)

    assert status.context.inputs == {"namespace": "payments"}
    assert "unexpected_field" not in status.context.inputs


# --- Cancellation --------------------------------------------------------------

def test_cancel_clears_context():
    context = make_context(KB_SINGLE_INPUT, inputs={"namespace": "payments"}, status="READY")

    status = handle_cancel(context)

    assert status.state == "cancelled"
    assert status.message == "Procedure cancelled."
    assert status.context is None


# --- No step execution / no operational MCP tool calls -----------------------

def test_no_openshift_or_aap_tool_is_ever_called(config, itsm_mcp, extractor, step_executor):
    """Only the fixed `rag_search_kb` ITSM tool may be called during KB
    retrieval and input collection; the Step Executor (mocked here, see
    `tests/test_procedure_executor.py` for its own MCP-level tests) is a
    separate, dedicated path for any operational MCP tool use.
    """
    start_procedure(config, itsm_mcp, KB_TWO_INPUTS)
    extractor.return_value = {"namespace": "payments", "application_name": "checkout"}

    _events, status = run_procedure("/procedure inspect checkout in namespace payments", config)
    assert status.state == "completed"

    # The only MCP tool call observed anywhere in input collection is the fixed KB search.
    assert itsm_mcp.calls == [("rag_search_kb", {"query": "inspect checkout in namespace payments"})]
    step_executor.assert_awaited_once()
