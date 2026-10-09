import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from agents import MaxTurnsExceeded

from app import web
from app.mcp import MCPConnectionError


def request(method, path, **kwargs):
    async def send():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web.app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())


@pytest.fixture
def agent(monkeypatch, config):
    monkeypatch.setattr(web, "load_config", lambda: config)
    mock = AsyncMock(return_value="The systems are available.")
    monkeypatch.setattr(web, "run_agent", mock)
    return mock


def test_page_does_not_call_agent(agent):
    response = request("GET", "/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Operations Agent" in response.text
    assert "autofocus" in response.text
    agent.assert_not_awaited()


# --- Procedure UI markers ----------------------------------------------------
# This project has no JS test runner, so (as with the assertions above) these
# check the rendered page for the required procedure-mode building blocks.

def test_page_includes_procedure_input_mode(agent):
    page = request("GET", "/").text
    assert 'id="procedure-chip"' in page
    assert "PROCEDURE_PATTERN" in page
    assert "enterProcedureMode" in page
    assert "procedure-input" in page


def test_page_includes_distinct_procedure_message_styling(agent):
    page = request("GET", "/").text
    assert "procedure-user" in page
    assert "appendProcedureMessage" in page


def test_page_includes_transient_procedure_progress(agent):
    page = request("GET", "/").text
    assert "procedure-progress" in page
    assert "procedureProgress" in page
    assert "Searching KB" in page
    # The permanent card/heading from the previous iteration must be gone.
    assert "procedure-card" not in page
    assert "createProcedureCard" not in page


def test_page_includes_animated_pending_indicator(agent):
    page = request("GET", "/").text
    assert "ELLIPSIS_FRAMES" in page
    assert "'...'" in page


def test_page_hides_progress_before_showing_the_final_message(agent):
    page = request("GET", "/").text
    # Both the success and the error path hide the transient bubble before
    # rendering the outcome(s) as normal chat message(s), never both at once.
    # The response is routed by its own "procedure_status" type, not by
    # whether the outgoing message happened to start with "/procedure" (a
    # plain reply while a procedure is waiting on input must take this path
    # too, see test_page_tracks_procedure_awaiting_input_across_replies).
    assert "procedureProgress.hide();\n        if (data.type === 'procedure_status') {" in page
    assert "procedureProgress.hide();\n        appendMessage(`Error: ${error.message}`, 'error');" in page


def test_page_reveals_sequential_procedure_messages_with_progress_between(agent):
    page = request("GET", "/").text
    # A single /procedure response can carry more than one chat message
    # (e.g. "KB found: ..." then the parsed summary); each later one is
    # revealed after showing the reusable progress bubble again, labeled
    # per-stage from the server when provided.
    assert "const messages = data.messages && data.messages.length ? data.messages : [data.message];" in page
    assert "const stages = data.progress_stages || [];" in page
    assert "procedureProgress.show(stages[index - 1] || 'Parsing procedure');" in page


def test_page_tracks_procedure_awaiting_input_across_replies(agent):
    page = request("GET", "/").text
    # A plain reply while a procedure is waiting on input shows the reusable
    # progress bubble too, and the flag is recomputed from the response
    # state (never assumed) after every request.
    assert "let procedureAwaitingInput = false;" in page
    assert "const isProcedureReply = !isProcedure && procedureAwaitingInput;" in page
    assert "else if (isProcedureReply) procedureProgress.show('Extracting inputs');" in page
    assert "procedureAwaitingInput = data.state === 'collecting_inputs';" in page


def test_page_skips_progress_for_empty_procedure_command(agent):
    page = request("GET", "/").text
    assert "if (isProcedure && procedureQuery) procedureProgress.show('Searching KB');" in page


def test_chat_reuses_agent(agent, config):
    response = request("POST", "/api/chat", json={"message": "  Check pods  "})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    agent.assert_awaited_once_with("Check pods", config)
    assert config.model_api_key not in response.text


@pytest.mark.parametrize("body", [{}, {"message": ""}, {"message": "   "}, {"message": None}])
def test_invalid_message_does_not_call_agent(agent, body):
    assert request("POST", "/api/chat", json=body).status_code == 422
    agent.assert_not_awaited()


@pytest.mark.parametrize("error, status, detail", [
    (MCPConnectionError("MCP connection failed: aap"), 502, "aap"),
    (MaxTurnsExceeded("limit"), 422, "turn limit"),
    (RuntimeError("Authorization: Bearer test-secret"), 502, "request failed"),
])
def test_agent_errors_are_safe(agent, error, status, detail, caplog):
    agent.side_effect = error
    response = request("POST", "/api/chat", json={"message": "Check pods"})
    assert response.status_code == status
    assert detail in response.json()["detail"]
    assert "test-secret" not in response.text + caplog.text


def test_configuration_error(agent, monkeypatch):
    monkeypatch.setattr(web, "load_config", Mock(side_effect=ValueError("test-secret")))
    response = request("POST", "/api/chat", json={"message": "Check pods"})
    assert response.status_code == 503
    assert "test-secret" not in response.text
    agent.assert_not_awaited()


def test_provider_error_is_diagnostic_only_on_server(agent, config, caplog):
    agent.side_effect = RuntimeError(
        "Provider does not support tool_choice=required\n"
        f"API key: {config.model_api_key}\n"
        f"Authorization: Bearer {config.mcp_servers[0].token}"
    )
    response = request("POST", "/api/chat", json={"message": "How many pods do we have?"})
    assert response.status_code == 502
    assert "RuntimeError" in caplog.text
    assert "does not support tool_choice=required" in caplog.text
    assert "does not support" not in response.text
    assert config.model_api_key not in caplog.text + response.text
    assert config.mcp_servers[0].token not in caplog.text + response.text
    assert "Authorization" not in caplog.text


def test_error_logging_omits_response_payload(agent, caplog):
    error = RuntimeError("raw response contains sensitive-payload")
    error.body = {"error": {"message": "Unsupported tool choice"}, "request": "sensitive-payload"}
    agent.side_effect = error
    response = request("POST", "/api/chat", json={"message": "Check pods"})
    assert response.status_code == 502
    assert "Unsupported tool choice" in caplog.text
    assert "sensitive-payload" not in caplog.text + response.text


# --- Deterministic /procedure routing ---------------------------------------
# No LLM classifies these messages: only the literal "/procedure" prefix
# diverts from the existing OperationsAgent chat path.

@pytest.fixture
def procedure(monkeypatch):
    from app.procedure.models import ProcedureStatus
    mock = AsyncMock(return_value=ProcedureStatus(state="kb_found", message="KB found: Test"))
    monkeypatch.setattr(web, "handle_procedure", mock)
    return mock


def test_procedure_message_uses_procedure_handler(agent, procedure, config):
    response = request("POST", "/api/chat", json={"message": "/procedure inspect namespace health"})
    assert response.status_code == 200
    assert response.json() == {"type": "procedure_status", "state": "kb_found", "message": "KB found: Test"}
    procedure.assert_awaited_once_with("/procedure inspect namespace health", config)


def test_procedure_payload_includes_messages_when_present(agent, monkeypatch, config):
    from app.procedure.models import ProcedureStatus

    status = ProcedureStatus(
        state="parsed",
        message="Procedure parsed successfully.",
        messages=("KB found: Inspect namespace health", "Procedure parsed successfully."),
    )
    monkeypatch.setattr(web, "handle_procedure", AsyncMock(return_value=status))

    response = request("POST", "/api/chat", json={"message": "/procedure inspect namespace health"})
    assert response.status_code == 200
    assert response.json() == {
        "type": "procedure_status",
        "state": "parsed",
        "message": "Procedure parsed successfully.",
        "messages": ["KB found: Inspect namespace health", "Procedure parsed successfully."],
    }
    agent.assert_not_awaited()


def test_normal_message_uses_existing_agent_path(agent, procedure, config):
    response = request("POST", "/api/chat", json={"message": "inspect namespace health"})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    agent.assert_awaited_once_with("inspect namespace health", config)
    procedure.assert_not_awaited()


def test_natural_language_mention_of_procedure_uses_existing_agent_path(agent, procedure, config):
    message = "what is the procedure for inspecting namespace health?"
    response = request("POST", "/api/chat", json={"message": message})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    agent.assert_awaited_once_with(message, config)
    procedure.assert_not_awaited()


# --- One active procedure per chat session: routing priority ----------------
# A chat session (not the stateless /api/chat endpoint) remembers at most one
# active procedure. These tests drive the real /api/sessions endpoints and
# mock only app.procedure's entry points, so the routing priority itself
# (cancel > input reply > new /procedure > normal chat) is exercised for real.

def make_procedure_context(status="COLLECTING_INPUTS", inputs=None):
    from app.procedure.models import ProcedureContext, ProcedureDefinition

    procedure = ProcedureDefinition(
        id="inspect-namespace-health", version=1, title="Inspect namespace health",
        risk="low", confirmation_required=False,
        inputs=[], steps=[{"id": "step-1", "title": "Step 1", "instruction": "Do it."}],
    )
    return ProcedureContext(
        run_id="run-1", original_request="inspect namespace health", kb_title="Inspect namespace health",
        kb_content="# Inspect namespace health\n", procedure=procedure, inputs=inputs or {}, status=status,
    )


@pytest.fixture
def procedure_lifecycle(monkeypatch):
    """Mock the three app.procedure entry points `web` calls, independent of
    the `procedure`/`agent` fixtures above.
    """
    start = AsyncMock()
    reply = AsyncMock()
    cancel = Mock()
    monkeypatch.setattr(web, "handle_procedure", start)
    monkeypatch.setattr(web, "handle_procedure_input_reply", reply)
    monkeypatch.setattr(web, "handle_cancel", cancel)
    return SimpleNamespace(start=start, reply=reply, cancel=cancel)


def create_test_session():
    return request("POST", "/api/sessions").json()["session_id"]


def test_active_procedure_consumes_next_reply_not_the_agent(agent, procedure_lifecycle, config):
    from app.procedure.models import ProcedureStatus

    session_id = create_test_session()
    waiting_context = make_procedure_context(status="COLLECTING_INPUTS")
    procedure_lifecycle.start.return_value = ProcedureStatus(
        state="collecting_inputs", message="What namespace should I use?", context=waiting_context,
    )

    response = request("POST", f"/api/sessions/{session_id}/messages",
                        json={"message": "/procedure inspect namespace health"})
    assert response.status_code == 200
    assert response.json()["message"] == "What namespace should I use?"
    agent.assert_not_awaited()

    ready_context = make_procedure_context(status="READY", inputs={"namespace": "payments"})
    procedure_lifecycle.reply.return_value = ProcedureStatus(
        state="ready", message="Procedure is ready to execute.", context=ready_context,
    )

    response = request("POST", f"/api/sessions/{session_id}/messages", json={"message": "payments"})
    assert response.status_code == 200
    assert response.json() == {"type": "procedure_status", "state": "ready",
                                "message": "Procedure is ready to execute."}
    procedure_lifecycle.reply.assert_awaited_once_with("payments", waiting_context, config)
    agent.assert_not_awaited()


def test_after_ready_normal_chat_is_not_consumed_by_the_procedure(agent, procedure_lifecycle, config):
    from app.procedure.models import ProcedureStatus

    session_id = create_test_session()
    ready_context = make_procedure_context(status="READY", inputs={"namespace": "payments"})
    procedure_lifecycle.start.return_value = ProcedureStatus(
        state="ready", message="Procedure is ready to execute.", context=ready_context,
    )
    request("POST", f"/api/sessions/{session_id}/messages",
            json={"message": "/procedure inspect namespace health for namespace payments"})

    response = request("POST", f"/api/sessions/{session_id}/messages", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    assert agent.await_args.args[:2] == ("hello", config)
    procedure_lifecycle.reply.assert_not_awaited()


def test_routing_regression_without_an_active_procedure(agent, procedure_lifecycle, config):
    session_id = create_test_session()
    response = request("POST", f"/api/sessions/{session_id}/messages", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    assert agent.await_args.args[:2] == ("hello", config)
    procedure_lifecycle.start.assert_not_awaited()
    procedure_lifecycle.reply.assert_not_awaited()


def test_cancel_clears_the_active_procedure(agent, procedure_lifecycle, config):
    from app.procedure.models import ProcedureStatus

    session_id = create_test_session()
    waiting_context = make_procedure_context(status="COLLECTING_INPUTS")
    procedure_lifecycle.start.return_value = ProcedureStatus(
        state="collecting_inputs", message="What namespace should I use?", context=waiting_context,
    )
    request("POST", f"/api/sessions/{session_id}/messages",
            json={"message": "/procedure inspect namespace health"})

    procedure_lifecycle.cancel.return_value = ProcedureStatus(
        state="cancelled", message="Procedure cancelled.", context=None,
    )
    response = request("POST", f"/api/sessions/{session_id}/messages", json={"message": "/cancel"})
    assert response.status_code == 200
    assert response.json() == {"type": "procedure_status", "state": "cancelled", "message": "Procedure cancelled."}
    procedure_lifecycle.cancel.assert_called_once_with(waiting_context)

    # The context is gone: a normal message now reaches the agent, not the
    # (now inactive) procedure reply handler.
    response = request("POST", f"/api/sessions/{session_id}/messages", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    procedure_lifecycle.reply.assert_not_awaited()


@pytest.mark.parametrize("terminal_status", ["FIRST_STEP_COMPLETED", "STOPPED", "FAILED"])
def test_after_step_execution_normal_chat_is_not_consumed(agent, procedure_lifecycle, config, terminal_status):
    """After the first step's outcome (success, stop, or failure), the
    procedure is no longer actively waiting on input; the next plain
    message must reach the existing OperationsAgent unchanged.
    """
    from app.procedure.models import ProcedureStatus

    session_id = create_test_session()
    terminal_context = make_procedure_context(status=terminal_status, inputs={"namespace": "payments"})
    procedure_lifecycle.start.return_value = ProcedureStatus(
        state=terminal_status.lower(), message="Step completed: Step 1\n\nDone.", context=terminal_context,
    )
    request("POST", f"/api/sessions/{session_id}/messages",
            json={"message": "/procedure inspect namespace health for namespace payments"})

    response = request("POST", f"/api/sessions/{session_id}/messages", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    assert agent.await_args.args[:2] == ("hello", config)
    procedure_lifecycle.reply.assert_not_awaited()


def test_step_execution_payload_includes_the_full_message_sequence(agent, monkeypatch, config):
    """The step-1 outcome rides the same `messages`/`progress_stages`
    contract already used for KB-found/parsed/collected-inputs sequencing;
    no new transport or payload shape is introduced for step execution.
    """
    from app.procedure.models import ProcedureStatus

    status = ProcedureStatus(
        state="first_step_completed",
        message="Step completed: Verify that the namespace exists\n\nNamespace payments exists.",
        messages=(
            "KB found: Inspect namespace health",
            "Procedure parsed successfully.",
            "Inputs collected:\n\n- Namespace: payments\n\nProcedure is ready to execute.",
            "Step completed: Verify that the namespace exists\n\nNamespace payments exists.",
        ),
        progress_stages=("Parsing procedure", "Extracting inputs", "Running: Verify that the namespace exists..."),
    )
    monkeypatch.setattr(web, "handle_procedure", AsyncMock(return_value=status))

    response = request("POST", "/api/chat", json={"message": "/procedure inspect namespace health for payments"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "first_step_completed"
    assert len(payload["messages"]) == 4
    assert payload["progress_stages"][-1] == "Running: Verify that the namespace exists..."
    # Step 2's title never appears anywhere in the payload (it is never run).
    assert "List pods" not in str(payload)
    agent.assert_not_awaited()


def test_procedure_context_never_appears_in_the_response(agent, procedure_lifecycle, config):
    from app.procedure.models import ProcedureStatus

    session_id = create_test_session()
    context = make_procedure_context(status="COLLECTING_INPUTS")
    procedure_lifecycle.start.return_value = ProcedureStatus(
        state="collecting_inputs", message="What namespace should I use?", context=context,
    )
    response = request("POST", f"/api/sessions/{session_id}/messages",
                        json={"message": "/procedure inspect namespace health"})
    assert "context" not in response.json()
    assert context.kb_content not in response.text
