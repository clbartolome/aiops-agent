import asyncio
from unittest.mock import AsyncMock

import pytest

from app.procedure import (
    EMPTY_PROCEDURE_REQUEST,
    NO_ACTIVE_PROCEDURE,
    NO_KB_ARTICLE_FOUND,
    PROCEDURE_SECTION_MARKER,
    cancel_active_run,
    extract_procedure_query,
    handle_message,
    is_procedure_request,
    resume_procedure_run,
    search_itsm_kb,
    start_procedure_run,
)
from app.procedure.models import StepResult
from app.procedure.runtime import ProcedureRuntime
from app.sessions import Conversation, visible_messages


@pytest.mark.parametrize("message", [
    "/procedure inspect namespace health",
    "  /procedure inspect namespace health",
    "/PROCEDURE inspect namespace health",
    "/procedure",
    "  /procedure  ",
])
def test_procedure_prefix_is_detected(message):
    assert is_procedure_request(message)


@pytest.mark.parametrize("message", [
    "what is the procedure for restarting a pod?",
    "how many pods are running?",
    "procedure inspect namespace health",
    "/procedurelist something",
    "",
    "   ",
])
def test_non_procedure_messages_are_not_routed_to_procedure_mode(message):
    assert not is_procedure_request(message)


def test_extract_procedure_query_strips_prefix_and_whitespace():
    assert extract_procedure_query("/procedure inspect namespace health") == "inspect namespace health"
    assert extract_procedure_query("  /procedure   inspect namespace health  ") == "inspect namespace health"


def test_extract_procedure_query_is_empty_for_bare_command():
    assert extract_procedure_query("/procedure") == ""
    assert extract_procedure_query("/procedure   ") == ""


# --- KB retrieval (unchanged from Step 2) -------------------------------------

def test_search_itsm_kb_uses_existing_itsm_mcp_integration(config, itsm_kb_boundary):
    article = {"title": "Inspect Namespace Health", "content": "# Inspect\n\n## Procedure\n1. Check pods"}
    itsm_kb_boundary.results = [article]

    result = asyncio.run(search_itsm_kb("inspect namespace health", config))

    assert result == article
    itsm_kb_boundary.sessions["itsm"].call_tool.assert_awaited_once_with(
        itsm_kb_boundary.tool_name, {"query": "inspect namespace health"},
    )
    assert set(itsm_kb_boundary.sessions) == {"itsm"}


def test_search_itsm_kb_only_connects_to_the_itsm_server(config, itsm_kb_boundary):
    asyncio.run(search_itsm_kb("inspect namespace health", config))
    assert [server.name for server in itsm_kb_boundary.servers] == ["itsm"]


def test_search_itsm_kb_returns_none_without_match(config, itsm_kb_boundary):
    itsm_kb_boundary.results = []
    assert asyncio.run(search_itsm_kb("inspect namespace health", config)) is None


VALID_PROCEDURE_MARKDOWN = """# Inspect namespace health

Inspect the current state of an OpenShift namespace and report basic workload health.

## Procedure

**ID:** inspect-namespace-health
**Version:** 1
**Risk:** low
**Confirmation required:** no

## Required information

- **Namespace** — required — OpenShift namespace to inspect.

## Steps

### 1. Verify that the namespace exists

Check that **Namespace** exists in the OpenShift cluster.

### 2. List pods in the namespace

Retrieve all pods running in **Namespace**.

### 3. Review recent events

Retrieve recent events from **Namespace**.

## Success

The namespace exists and the requested information has been retrieved successfully.
"""

CONFIRMATION_PROCEDURE_MARKDOWN = """# Restart deployment

Restart a deployment in a namespace.

## Procedure

**ID:** restart-deployment
**Version:** 1
**Risk:** medium
**Confirmation required:** yes

## Required information

## Steps

### 1. Restart the deployment

Restart the deployment.

## Success

The deployment restarted successfully.
"""


@pytest.fixture
def runtime():
    """A fresh `ProcedureRuntime` per test: never the shared production
    singleton, so tests stay independent of each other and of production
    state.
    """
    return ProcedureRuntime()


@pytest.fixture
def fake_step_executor():
    """A `step_executor_factory` double: ignores `config` and always returns
    the same fake async executor, so chat-integration tests never make a
    real MCP/model call. `.calls` records `(step_id, resolved_inputs)`.
    """
    calls = []

    async def executor(step, inputs, previous_results=None, *, approved_calls=frozenset()):
        calls.append((step.id, dict(inputs)))
        return StepResult(outcome="SUCCESS", summary=f"{step.title} done")

    def factory(config):
        return executor

    factory.calls = calls
    return factory


# --- start_procedure_run: validation / retrieval controlled responses --------

def test_start_procedure_run_returns_controlled_validation_for_empty_command(config, itsm_kb_boundary):
    result = asyncio.run(start_procedure_run("/procedure", config))
    assert result == EMPTY_PROCEDURE_REQUEST
    assert itsm_kb_boundary.sessions == {}


def test_start_procedure_run_reports_no_match(config, itsm_kb_boundary):
    itsm_kb_boundary.results = []
    result = asyncio.run(start_procedure_run("/procedure inspect namespace health", config))
    assert result == NO_KB_ARTICLE_FOUND


def test_start_procedure_run_reports_non_executable_knowledge_article(config, itsm_kb_boundary):
    itsm_kb_boundary.results = [{
        "title": "What is a namespace?",
        "content": "# What is a namespace?\n\nA namespace groups resources together.",
    }]
    result = asyncio.run(start_procedure_run("/procedure inspect namespace health", config))
    assert "What is a namespace?" in result
    assert "not an executable procedure" in result.lower()
    assert "knowledge article" in result.lower()


def test_start_procedure_run_reports_a_controlled_error_for_a_malformed_article(config, itsm_kb_boundary):
    itsm_kb_boundary.results = [{
        "title": "Inspect Namespace Health",
        "content": f"# Inspect Namespace Health\n\n{PROCEDURE_SECTION_MARKER}\n1. Check pods\n2. Check events",
    }]
    result = asyncio.run(start_procedure_run("/procedure inspect namespace health", config))
    assert "Inspect Namespace Health" in result
    assert "could not be parsed" in result.lower()


# --- start_procedure_run: creates a LangGraph run, using a real executor factory seam ---

def test_start_procedure_run_interrupts_for_missing_input_and_tracks_the_run(
    config, itsm_kb_boundary, runtime, fake_step_executor,
):
    itsm_kb_boundary.results = [{"title": "Inspect namespace health", "content": VALID_PROCEDURE_MARKDOWN}]
    conversation = Conversation()
    try:
        result = asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation,
            runtime=runtime, step_executor_factory=fake_step_executor,
        ))
        assert "Namespace" in result
        assert fake_step_executor.calls == []
        assert conversation.active_procedure_run_id is not None
    finally:
        conversation.session.close()


def test_start_procedure_run_without_conversation_does_not_crash(
    config, itsm_kb_boundary, runtime, fake_step_executor,
):
    itsm_kb_boundary.results = [{"title": "Inspect namespace health", "content": VALID_PROCEDURE_MARKDOWN}]
    result = asyncio.run(start_procedure_run(
        "/procedure inspect namespace health", config, None,
        runtime=runtime, step_executor_factory=fake_step_executor,
    ))
    assert "Namespace" in result
    assert fake_step_executor.calls == []


# --- resume_procedure_run -------------------------------------------------------

def test_resume_procedure_run_with_structured_reply_completes_the_run(
    config, itsm_kb_boundary, runtime, fake_step_executor,
):
    itsm_kb_boundary.results = [{"title": "Inspect namespace health", "content": VALID_PROCEDURE_MARKDOWN}]
    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation,
            runtime=runtime, step_executor_factory=fake_step_executor,
        ))
        result = asyncio.run(resume_procedure_run("namespace=payments", config, conversation, runtime=runtime))
        assert "Procedure completed successfully." in result
        assert [step_id for step_id, _ in fake_step_executor.calls] == [
            "verify_that_the_namespace_exists", "list_pods_in_the_namespace", "review_recent_events",
        ]
        assert all(inputs == {"namespace": "payments"} for _, inputs in fake_step_executor.calls)
        assert conversation.active_procedure_run_id is None
    finally:
        conversation.session.close()


def test_resume_procedure_run_with_a_bare_single_field_reply(
    config, itsm_kb_boundary, runtime, fake_step_executor,
):
    itsm_kb_boundary.results = [{"title": "Inspect namespace health", "content": VALID_PROCEDURE_MARKDOWN}]
    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation,
            runtime=runtime, step_executor_factory=fake_step_executor,
        ))
        result = asyncio.run(resume_procedure_run("payments", config, conversation, runtime=runtime))
        assert "Procedure completed successfully." in result
    finally:
        conversation.session.close()


def test_resume_procedure_run_with_unrecognized_reply_does_not_consume_the_interrupt(
    config, itsm_kb_boundary, runtime, fake_step_executor,
):
    itsm_kb_boundary.results = [{"title": "Inspect namespace health", "content": VALID_PROCEDURE_MARKDOWN}]
    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation,
            runtime=runtime, step_executor_factory=fake_step_executor,
        ))
        result = asyncio.run(resume_procedure_run("", config, conversation, runtime=runtime))
        assert "did not recognize" in result.lower()
        assert fake_step_executor.calls == []
        assert conversation.active_procedure_run_id is not None
    finally:
        conversation.session.close()


def test_resume_procedure_run_confirmation_accept_and_reject(
    config, itsm_kb_boundary, runtime, fake_step_executor,
):
    itsm_kb_boundary.results = [{"title": "Restart deployment", "content": CONFIRMATION_PROCEDURE_MARKDOWN}]
    conversation = Conversation()
    try:
        first = asyncio.run(start_procedure_run(
            "/procedure restart deployment", config, conversation,
            runtime=runtime, step_executor_factory=fake_step_executor,
        ))
        assert "Proceed?" in first
        assert conversation.active_procedure_run_id is not None

        bad = asyncio.run(resume_procedure_run("maybe", config, conversation, runtime=runtime))
        assert "yes" in bad.lower() and "no" in bad.lower()
        assert conversation.active_procedure_run_id is not None

        result = asyncio.run(resume_procedure_run("no", config, conversation, runtime=runtime))
        assert "cancel" in result.lower()
        assert conversation.active_procedure_run_id is None
        assert fake_step_executor.calls == []
    finally:
        conversation.session.close()


def test_resume_procedure_run_with_no_pending_run_is_controlled(config, runtime):
    conversation = Conversation()
    conversation.active_procedure_run_id = "not-a-real-run"
    try:
        result = asyncio.run(resume_procedure_run("payments", config, conversation, runtime=runtime))
        assert result == NO_ACTIVE_PROCEDURE
        assert conversation.active_procedure_run_id is None
    finally:
        conversation.session.close()


# --- cancellation ----------------------------------------------------------------

def test_cancel_active_run_clears_the_association(config, itsm_kb_boundary, runtime, fake_step_executor):
    itsm_kb_boundary.results = [{"title": "Inspect namespace health", "content": VALID_PROCEDURE_MARKDOWN}]
    conversation = Conversation()
    try:
        asyncio.run(start_procedure_run(
            "/procedure inspect namespace health", config, conversation,
            runtime=runtime, step_executor_factory=fake_step_executor,
        ))
        assert conversation.active_procedure_run_id is not None
        result = asyncio.run(cancel_active_run(conversation, runtime=runtime))
        assert "cancel" in result.lower()
        assert conversation.active_procedure_run_id is None
        assert fake_step_executor.calls == []
    finally:
        conversation.session.close()


def test_cancel_without_an_active_run_is_controlled(config, runtime):
    conversation = Conversation()
    try:
        result = asyncio.run(cancel_active_run(conversation, runtime=runtime))
        assert result == NO_ACTIVE_PROCEDURE
    finally:
        conversation.session.close()


# --- handle_message: the deterministic top-level router -----------------------

def test_handle_message_cancel_without_active_procedure_returns_controlled_response(config, itsm_kb_boundary):
    result = asyncio.run(handle_message("/cancel", config, None))
    assert result == NO_ACTIVE_PROCEDURE
    assert itsm_kb_boundary.sessions == {}


def test_handle_message_falls_through_to_agent_for_normal_chat(config, itsm_kb_boundary):
    result = asyncio.run(handle_message("what is the procedure for inspecting a namespace?", config, None))
    assert result is None
    assert itsm_kb_boundary.sessions == {}


def test_handle_message_starts_a_run_and_resumes_through_the_full_lifecycle(
    config, itsm_kb_boundary, runtime, fake_step_executor,
):
    itsm_kb_boundary.results = [{"title": "Inspect namespace health", "content": VALID_PROCEDURE_MARKDOWN}]
    conversation = Conversation()
    try:
        first = asyncio.run(handle_message(
            "/procedure inspect namespace health", config, conversation,
            runtime=runtime, step_executor_factory=fake_step_executor,
        ))
        assert first is not None and "Namespace" in first
        assert conversation.active_procedure_run_id is not None

        second = asyncio.run(handle_message(
            "namespace=payments", config, conversation, runtime=runtime, step_executor_factory=fake_step_executor,
        ))
        assert "Procedure completed successfully." in second
        assert conversation.active_procedure_run_id is None

        messages = asyncio.run(visible_messages(conversation.session))
        assert [item["role"] for item in messages] == ["user", "assistant", "user", "assistant"]
        assert messages[0]["content"] == "/procedure inspect namespace health"
        assert messages[-1]["content"] == second
    finally:
        conversation.session.close()


def test_handle_message_cancel_while_active_clears_the_run(
    config, itsm_kb_boundary, runtime, fake_step_executor,
):
    itsm_kb_boundary.results = [{"title": "Inspect namespace health", "content": VALID_PROCEDURE_MARKDOWN}]
    conversation = Conversation()
    try:
        asyncio.run(handle_message(
            "/procedure inspect namespace health", config, conversation,
            runtime=runtime, step_executor_factory=fake_step_executor,
        ))
        assert conversation.active_procedure_run_id is not None
        result = asyncio.run(handle_message("/cancel", config, conversation, runtime=runtime))
        assert "cancel" in result.lower()
        assert conversation.active_procedure_run_id is None
        assert fake_step_executor.calls == []
    finally:
        conversation.session.close()


def test_handle_message_without_conversation_starts_a_one_shot_run(
    config, itsm_kb_boundary, runtime, fake_step_executor,
):
    itsm_kb_boundary.results = [{"title": "Inspect namespace health", "content": VALID_PROCEDURE_MARKDOWN}]
    result = asyncio.run(handle_message(
        "/procedure inspect namespace health", config, None,
        runtime=runtime, step_executor_factory=fake_step_executor,
    ))
    assert "Namespace" in result
    assert fake_step_executor.calls == []


def test_handle_message_does_not_duplicate_chat_history_mechanism(
    config, itsm_kb_boundary, runtime, fake_step_executor, monkeypatch,
):
    """The chat session (SDK) is the only history store; LangGraph never writes to it."""
    itsm_kb_boundary.results = [{"title": "Inspect namespace health", "content": VALID_PROCEDURE_MARKDOWN}]
    conversation = Conversation()
    try:
        add_items = AsyncMock(wraps=conversation.session.add_items)
        monkeypatch.setattr(conversation.session, "add_items", add_items)
        asyncio.run(handle_message(
            "/procedure inspect namespace health", config, conversation,
            runtime=runtime, step_executor_factory=fake_step_executor,
        ))
        add_items.assert_awaited_once()
    finally:
        conversation.session.close()
