import asyncio
import json
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


def ndjson_lines(response) -> list[dict]:
    """Parse a streamed `application/x-ndjson` response body into one dict
    per line, in order -- the same thing the browser's `readProcedureStream`
    does incrementally, just read back after the (test-only, fully
    buffered) request has completed.
    """
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def make_procedure_mock():
    """A stand-in for `handle_procedure`/`handle_procedure_input_reply`:
    both are async generator functions in the real app (see
    `app.procedure`'s module docstring), so a plain `AsyncMock` (which
    returns a coroutine, not something `async for`-iterable) cannot stand
    in for them directly.

    Calling this mock returns an async generator that yields every event
    in `.events` (empty by default) and then `.return_value` (a
    `ProcedureStatus`, set exactly like any other `Mock`) -- mirroring
    the real functions' "zero or more wire events, then one terminal
    status" shape. Call recording (`.call_args`, `.assert_called_once_with`,
    etc.) works normally since `side_effect` doesn't disable it; only
    `.assert_awaited_*` doesn't apply (this mock is called, not awaited --
    only the generator it returns is ever iterated with `async for`).
    """
    mock = Mock()
    mock.events = ()

    async def generator(*args, **kwargs):
        for event in mock.events:
            yield event
        yield mock.return_value

    mock.side_effect = generator
    return mock


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
    # The permanent card/heading from a previous iteration must be gone.
    assert "procedure-card" not in page
    assert "createProcedureCard" not in page


def test_page_includes_animated_pending_indicator(agent):
    page = request("GET", "/").text
    assert "ELLIPSIS_FRAMES" in page
    assert "'...'" in page


def test_page_reuses_the_single_progress_element_instead_of_accumulating(agent):
    """Exactly one transient progress bubble ever exists: `show()` always
    tears down any previous one first (`hide()`), it is never appended
    again without clearing the old one, so repeated progress events can
    never stack up multiple loading bubbles in the DOM.
    """
    page = request("GET", "/").text
    assert "function show(text) {\n        hide();" in page


def test_page_handles_each_streamed_procedure_event_line_live(agent):
    """The frontend reads the NDJSON response body incrementally (one
    complete line at a time, via a stream reader) and reacts to each line
    immediately -- not after collecting the whole response first.
    """
    page = request("GET", "/").text
    assert "response.body.getReader()" in page
    assert "async function readProcedureStream" in page
    # Each event type is dispatched to the one shared progress bubble or to
    # a permanent chat message -- never both, and progress is always
    # cleared before (or by) a permanent message.
    assert "data.type === 'procedure_progress'" in page
    assert "data.type === 'procedure_progress_clear'" in page
    assert "data.type === 'procedure_message'" in page
    assert "data.type === 'procedure_status'" in page
    assert "procedureAwaitingInput = data.state === 'collecting_inputs';" in page


def test_page_hides_progress_on_every_terminal_path(agent):
    """No stale progress bubble can remain: a permanent message, the
    terminal status line, and the error path all hide it.
    """
    page = request("GET", "/").text
    assert "procedureProgress.hide();\n        appendMessage(data.message, 'assistant');" in page
    assert "procedureProgress.hide();\n        procedureAwaitingInput = data.state === 'collecting_inputs';" in page
    assert "procedureProgress.hide();\n        appendMessage(`Error: ${error.message}`, 'error');" in page


def test_page_does_not_fake_progress_with_a_frontend_timer(agent):
    """Progress must reflect real backend state, not a client-side timer
    cycling through canned labels on a fixed delay."""
    page = request("GET", "/")
    assert "PROCEDURE_STAGE_DELAY_MS" not in page.text


def test_page_tracks_procedure_awaiting_input_across_replies(agent):
    page = request("GET", "/").text
    # A plain reply while a procedure is waiting on input shows the reusable
    # progress bubble too, and the flag is recomputed from the response
    # state (never assumed) after every request.
    assert "let procedureAwaitingInput = false;" in page
    assert "const isProcedureReply = !isProcedure && procedureAwaitingInput;" in page
    assert "else if (isProcedureReply) procedureProgress.show('Extracting parameters');" in page


def test_page_skips_progress_for_empty_procedure_command(agent):
    page = request("GET", "/").text
    assert "if (isProcedure && procedureQuery) procedureProgress.show('Searching KB');" in page


# --- Session message persistence/restoration (frontend) ---------------------
# This project has no JS test runner, so (as with the other page-string
# assertions above) these check the rendered page for the required
# session-switching building blocks.

def test_page_switches_active_session_and_clears_history_before_fetching(agent):
    """Selecting a session sets it active and clears the rendered
    conversation immediately, before its history is fetched -- never left
    showing the *previous* session's messages under the *new* session's
    identity while the fetch is in flight.
    """
    page = request("GET", "/").text
    assert (
        "selectedSession = id;\n      procedureAwaitingInput = false;\n      history.replaceChildren();" in page
    )
    assert "status.textContent = 'Loading conversation...';" in page


def test_page_renders_fetched_session_messages_via_restore_helper(agent):
    page = request("GET", "/").text
    assert "for (const message of session.messages) renderStoredMessage(message);" in page


def test_page_shows_controlled_error_and_clears_stale_messages_on_load_failure(agent):
    """If fetching a session's history fails, no messages are left on
    screen as if they belonged to the newly selected session -- a plain,
    controlled error is shown instead.
    """
    page = request("GET", "/").text
    assert (
        "history.replaceChildren();\n        status.textContent = 'Unable to load this conversation.';" in page
    )


def test_page_restores_procedure_commands_with_their_original_styling(agent):
    """A persisted `/procedure ...` user message restores as the same
    styled "PROCEDURE" line it was rendered as live, using the exact same
    pattern the live submit handler already checks -- no extra wire-format
    metadata is introduced just to remember this.
    """
    page = request("GET", "/").text
    assert "function renderStoredMessage(message) {" in page
    assert "const match = message.content.match(PROCEDURE_PATTERN);" in page
    assert "appendProcedureMessage((match[2] || '').trim());" in page
    assert "appendMessage(message.content, message.role);" in page


def test_page_does_not_reuse_the_procedure_progress_bubble_for_loading(agent):
    page = request("GET", "/").text
    assert "procedureProgress.show('Loading conversation" not in page


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
    mock = make_procedure_mock()
    mock.return_value = ProcedureStatus(state="kb_found", message="KB found: Test")
    monkeypatch.setattr(web, "handle_procedure", mock)
    return mock


def test_procedure_message_uses_procedure_handler(agent, procedure, config):
    response = request("POST", "/api/chat", json={"message": "/procedure inspect namespace health"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert ndjson_lines(response) == [
        {"type": "procedure_status", "state": "kb_found", "message": "KB found: Test"},
    ]
    procedure.assert_called_once_with("/procedure inspect namespace health", config)


def test_procedure_events_are_streamed_before_the_terminal_status_line(agent, monkeypatch, config):
    """Every progress/clear/message event the handler yields rides its own
    NDJSON line, in order, strictly before the one terminal
    `procedure_status` line -- this is the actual wire contract live
    progress depends on, independent of the (now removed)
    `messages`/`progress_stages` batching fields.
    """
    from app.procedure.models import ProcedureStatus

    mock = make_procedure_mock()
    mock.events = (
        {"type": "procedure_progress", "message": "Searching KB"},
        {"type": "procedure_progress_clear"},
        {"type": "procedure_message", "message": "KB found: Inspect namespace health"},
        {"type": "procedure_progress", "message": "Parsing procedure"},
        {"type": "procedure_progress_clear"},
        {"type": "procedure_message", "message": "Procedure loaded: 4 steps."},
    )
    mock.return_value = ProcedureStatus(state="collecting_inputs", message="What namespace should I use?")
    monkeypatch.setattr(web, "handle_procedure", mock)

    response = request("POST", "/api/chat", json={"message": "/procedure inspect namespace health"})
    assert response.status_code == 200
    lines = ndjson_lines(response)
    assert lines[:-1] == list(mock.events)
    assert lines[-1] == {"type": "procedure_status", "state": "collecting_inputs",
                          "message": "What namespace should I use?"}
    agent.assert_not_awaited()


def test_normal_message_uses_existing_agent_path(agent, procedure, config):
    response = request("POST", "/api/chat", json={"message": "inspect namespace health"})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    agent.assert_awaited_once_with("inspect namespace health", config)
    procedure.assert_not_called()


def test_natural_language_mention_of_procedure_uses_existing_agent_path(agent, procedure, config):
    message = "what is the procedure for inspecting namespace health?"
    response = request("POST", "/api/chat", json={"message": message})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    agent.assert_awaited_once_with(message, config)
    procedure.assert_not_called()


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
    the `procedure`/`agent` fixtures above. `start`/`reply` behave like
    `handle_procedure`/`handle_procedure_input_reply` (async generators,
    see `make_procedure_mock`); `cancel` stays a plain `Mock` since
    `handle_cancel` is still a plain, synchronous, non-generator function.
    """
    start = make_procedure_mock()
    reply = make_procedure_mock()
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
    assert ndjson_lines(response)[-1]["message"] == "What namespace should I use?"
    agent.assert_not_awaited()

    ready_context = make_procedure_context(status="READY", inputs={"namespace": "payments"})
    procedure_lifecycle.reply.return_value = ProcedureStatus(
        state="ready", message="Procedure is ready to execute.", context=ready_context,
    )

    response = request("POST", f"/api/sessions/{session_id}/messages", json={"message": "payments"})
    assert response.status_code == 200
    assert ndjson_lines(response) == [
        {"type": "procedure_status", "state": "ready", "message": "Procedure is ready to execute."},
    ]
    procedure_lifecycle.reply.assert_called_once_with("payments", waiting_context, config)
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
    procedure_lifecycle.reply.assert_not_called()


def test_routing_regression_without_an_active_procedure(agent, procedure_lifecycle, config):
    session_id = create_test_session()
    response = request("POST", f"/api/sessions/{session_id}/messages", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    assert agent.await_args.args[:2] == ("hello", config)
    procedure_lifecycle.start.assert_not_called()
    procedure_lifecycle.reply.assert_not_called()


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
    # /cancel is instantaneous: unlike a new /procedure or a reply, it is
    # never streamed, so this stays a single plain JSON response.
    assert response.json() == {"type": "procedure_status", "state": "cancelled", "message": "Procedure cancelled."}
    procedure_lifecycle.cancel.assert_called_once_with(waiting_context)

    # The context is gone: a normal message now reaches the agent, not the
    # (now inactive) procedure reply handler.
    response = request("POST", f"/api/sessions/{session_id}/messages", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    procedure_lifecycle.reply.assert_not_called()


@pytest.mark.parametrize("terminal_status", ["COMPLETED", "STOPPED", "FAILED"])
def test_after_step_execution_normal_chat_is_not_consumed(agent, procedure_lifecycle, config, terminal_status):
    """After the procedure reaches a terminal outcome (success, stop, or
    failure), it is no longer actively waiting on input; the next plain
    message must reach the existing OperationsAgent unchanged.
    """
    from app.procedure.models import ProcedureStatus

    session_id = create_test_session()
    terminal_context = make_procedure_context(status=terminal_status, inputs={"namespace": "payments"})
    procedure_lifecycle.start.return_value = ProcedureStatus(
        state=terminal_status.lower(), message="Procedure completed successfully.", context=terminal_context,
    )
    request("POST", f"/api/sessions/{session_id}/messages",
            json={"message": "/procedure inspect namespace health for namespace payments"})

    response = request("POST", f"/api/sessions/{session_id}/messages", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json() == {"response": agent.return_value}
    assert agent.await_args.args[:2] == ("hello", config)
    procedure_lifecycle.reply.assert_not_called()


def test_step_execution_events_are_streamed_in_order(agent, monkeypatch, config):
    """A step's progress/result events ride the same live NDJSON stream as
    KB-found/parsed/ready sequencing -- no new/second transport is
    introduced for step execution either.
    """
    from app.procedure.models import ProcedureStatus

    mock = make_procedure_mock()
    mock.events = (
        {"type": "procedure_message", "message": "KB found: Inspect namespace health"},
        {"type": "procedure_message", "message": "Procedure loaded: 4 steps."},
        {"type": "procedure_message", "message": "Using Namespace: payments."},
        {"type": "procedure_progress", "message": "Running step 1: Verify that the namespace exists"},
        {"type": "procedure_progress_clear"},
        {"type": "procedure_message", "message": "Namespace payments does not exist."},
        {"type": "procedure_progress", "message": "Generating procedure summary"},
        {"type": "procedure_progress_clear"},
        {"type": "procedure_message", "message": "The procedure stopped because the namespace does not exist."},
    )
    mock.return_value = ProcedureStatus(
        state="stopped", message="The procedure stopped because the namespace does not exist.",
    )
    monkeypatch.setattr(web, "handle_procedure", mock)

    response = request("POST", "/api/chat", json={"message": "/procedure inspect namespace health for payments"})
    assert response.status_code == 200
    lines = ndjson_lines(response)
    assert lines[:-1] == list(mock.events)
    assert lines[-1] == {"type": "procedure_status", "state": "stopped",
                          "message": "The procedure stopped because the namespace does not exist."}
    # No later step's title ever appears in the payload (none are run).
    assert "List pods" not in response.text
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
    assert "context" not in ndjson_lines(response)[-1]
    assert context.kb_content not in response.text


# --- Live delivery: progress must arrive before the slow work completes ------
# This is the key acceptance criterion: a response-buffering regression
# would make this test hang/fail, since it only unblocks the mocked Step
# Executor *after* confirming the first progress line was already read.

def test_first_step_progress_is_observable_before_the_step_executor_completes(config, monkeypatch):
    """Drives `app.web.app` directly at the raw ASGI protocol level (not
    through `httpx.ASGITransport`, which fully drains an app's response
    before returning anything to the caller -- see
    https://github.com/encode/httpx -- and so cannot observe partial
    delivery). Each `send()` call below corresponds to exactly one chunk
    Starlette's `StreamingResponse` wrote to the wire, in real time, as
    `app.web._stream_procedure` produced it; this is the actual
    acceptance criterion: response buffering must not hide progress.
    """
    import app.procedure as procedure_module
    from app.procedure.models import ProcedureContext, StepResult
    from app.procedure.parser import parse_procedure

    received_first_progress = asyncio.Event()
    may_finish = asyncio.Event()

    async def slow_execute_step(*args, **kwargs):
        received_first_progress.set()
        await may_finish.wait()
        return StepResult(status="SUCCESS", summary="Namespace payments exists.")

    monkeypatch.setattr(web, "load_config", lambda: config)
    monkeypatch.setattr(procedure_module, "execute_step", slow_execute_step)
    monkeypatch.setattr(procedure_module, "generate_procedure_summary", AsyncMock(return_value="Done."))

    single_step_kb = """# Inspect namespace health

## Procedure

**ID:** inspect-namespace-health
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
    # Status must be "COLLECTING_INPUTS" for `app.web`'s routing to send the
    # next plain message to `handle_procedure_input_reply` at all; this KB
    # declares no required inputs, so that function settles straight to
    # READY/RUNNING internally regardless of this stored status.
    context = ProcedureContext(
        run_id="run-1", original_request="", kb_title=procedure.title, kb_content=single_step_kb,
        procedure=procedure, inputs={}, status="COLLECTING_INPUTS",
    )

    async def scenario():
        from app.sessions import Conversation

        conversation = Conversation()
        conversation.active_procedure = context
        web.conversations[conversation.session_id] = conversation
        session_id = conversation.session_id

        body = json.dumps({"message": "go"}).encode()
        path = f"/api/sessions/{session_id}/messages"
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
            "path": path, "raw_path": path.encode(), "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            "server": ("testserver", 80), "client": ("testclient", 123), "scheme": "http", "root_path": "",
        }
        request_sent = False

        async def receive():
            nonlocal request_sent
            if request_sent:
                await asyncio.Event().wait()  # never actually awaited again in this test
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        chunks: asyncio.Queue = asyncio.Queue()

        async def send(message):
            if message["type"] == "http.response.body":
                await chunks.put(message.get("body", b""))
                if not message.get("more_body", False):
                    await chunks.put(None)

        app_task = asyncio.create_task(web.app(scope, receive, send))
        try:
            ready_chunk = await asyncio.wait_for(chunks.get(), timeout=1)
            assert json.loads(ready_chunk.decode().strip()) == {
                "type": "procedure_message", "message": "Procedure is ready to execute.",
            }

            progress_chunk = await asyncio.wait_for(chunks.get(), timeout=1)
            progress_line = json.loads(progress_chunk.decode().strip())
            assert progress_line == {"type": "procedure_progress",
                                      "message": "Running step 1: Verify that the namespace exists"}

            # The progress chunk already arrived over the wire; the mocked
            # Step Executor is still deliberately blocked -- this is the
            # anti-batching assertion: nothing was held back until the
            # whole procedure finished.
            await asyncio.wait_for(received_first_progress.wait(), timeout=1)
            assert not may_finish.is_set()
            assert not app_task.done()

            may_finish.set()
            remaining_bytes = []
            while True:
                chunk = await asyncio.wait_for(chunks.get(), timeout=1)
                if chunk is None:
                    break
                remaining_bytes.append(chunk)
            await asyncio.wait_for(app_task, timeout=1)
        finally:
            if not app_task.done():
                app_task.cancel()

        remaining = [json.loads(line) for line in b"".join(remaining_bytes).decode().splitlines() if line.strip()]
        assert remaining[-1]["type"] == "procedure_status"
        assert remaining[-1]["state"] == "completed"

    asyncio.run(scenario())


def test_exact_procedure_reply_is_routed_and_saved(agent, config, monkeypatch):
    import app.procedure as procedure_module
    from tests.test_procedure_inputs import KB_VM_INPUTS, make_context
    from app.sessions import Conversation

    monkeypatch.setattr(web, 'load_config', lambda: config)
    conversation = Conversation()
    conversation.session = SimpleNamespace(add_items=AsyncMock())
    conversation.active_procedure = make_context(KB_VM_INPUTS)
    web.conversations[conversation.session_id] = conversation
    extractor = AsyncMock(return_value={'virtual_machine_name': 'server-01'})
    monkeypatch.setattr(procedure_module, 'extract_procedure_inputs', extractor)
    response = request('POST', f'/api/sessions/{conversation.session_id}/messages',
                       json={'message': 'virtual machine name: server-01'})
    assert response.status_code == 200
    extractor.assert_awaited_once()
    assert extractor.await_args.args[0] == 'virtual machine name: server-01'
    assert conversation.active_procedure.inputs == {'virtual_machine_name': 'server-01'}
    agent.assert_not_awaited()


@pytest.mark.parametrize('terminal', ['COMPLETED', 'STOPPED', 'FAILED', 'CANCELLED'])
def test_terminal_procedure_clears_association_and_returns_to_chat(
    agent, procedure_lifecycle, config, terminal, caplog,
):
    import logging
    from app.procedure.models import ProcedureStatus
    from app.sessions import Conversation
    caplog.set_level(logging.INFO, logger='app.web')
    conversation = Conversation()
    conversation.session = SimpleNamespace(add_items=AsyncMock())
    web.conversations[conversation.session_id] = conversation
    procedure_lifecycle.start.return_value = ProcedureStatus(
        state=terminal.lower(), message='Done.', context=make_procedure_context(status=terminal),
    )
    path = f'/api/sessions/{conversation.session_id}/messages'
    response = request('POST', path, json={'message': '/procedure inspect namespace health'})
    assert response.status_code == 200
    assert conversation.active_procedure is None
    response = request('POST', path, json={'message': 'hello'})
    assert response.json() == {'response': agent.return_value}
    agent.assert_awaited_once_with('hello', config, session=conversation.session)
    procedure_lifecycle.reply.assert_not_called()
    assert f'session={conversation.session_id} route=procedure' in caplog.text
    assert f'session={conversation.session_id} route=operations_agent' in caplog.text


@pytest.mark.parametrize('message', ['hello', 'explain what namespaces are', 'what can you help me with?'])
def test_waiting_procedure_in_another_session_does_not_intercept_chat(
    agent, procedure_lifecycle, config, caplog, message,
):
    import logging
    from app.sessions import Conversation
    caplog.set_level(logging.INFO, logger='app.web')
    first, second = Conversation(), Conversation()
    first.active_procedure = make_procedure_context()
    web.conversations[first.session_id] = first
    web.conversations[second.session_id] = second
    response = request('POST', f'/api/sessions/{second.session_id}/messages', json={'message': message})
    assert response.json() == {'response': agent.return_value}
    agent.assert_awaited_once_with(message, config, session=second.session)
    procedure_lifecycle.start.assert_not_called()
    procedure_lifecycle.reply.assert_not_called()
    assert first.active_procedure.status == 'COLLECTING_INPUTS'
    assert f'session={second.session_id} route=operations_agent' in caplog.text
