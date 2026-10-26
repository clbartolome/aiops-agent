import json
import logging
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Annotated, AsyncIterator

from agents import MaxTurnsExceeded
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints

from app.agent import run_agent
from app.config import load_config
from app.diagnostics import configure_logging, log_failure
from app.mcp import MCPConnectionError
from app.procedure import (
    handle_cancel, handle_procedure, handle_procedure_input_reply, is_cancel_command, is_procedure_command,
)
from app.procedure.models import ProcedureStatus
from app.sessions import Conversation, timestamp, visible_messages

configure_logging()
conversations: dict[str, Conversation] = {}
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    yield
    for conversation in conversations.values():
        conversation.session.close()
    conversations.clear()


app = FastAPI(lifespan=lifespan)
PAGE = Path(__file__).with_name("index.html").read_text(encoding="utf-8")


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return PAGE


def procedure_payload(status: ProcedureStatus) -> dict:
    # The one terminal, machine-readable line for a `/procedure` turn: a
    # small "state" tag plus the final text, for any caller that only
    # wants the end result (e.g. a non-streaming `/api/chat` consumer).
    # The chat UI itself never renders this `message` directly -- every
    # permanent chat message already arrived as its own `procedure_message`
    # event while the stream was running (see `_stream_procedure` below).
    # `status.context` is never serialized: it is server-side-only session
    # state (it may carry the full KB Markdown and collected input values).
    return {"type": "procedure_status", "state": status.state, "message": status.message}


def _encode_event(event: dict) -> bytes:
    return (json.dumps(event) + "\n").encode()


async def _persist_message(conversation: Conversation | None, role: str, content: str) -> None:
    """Store one permanent chat message under this session's own SDK-owned
    history (`app.sessions.visible_messages` reads it back on
    `GET /api/sessions/{id}`) -- the same storage normal chat turns already
    get for free from `Runner.run(..., session=...)` in `app.agent`. Never
    called for transient procedure progress text, and a no-op for the
    stateless `/api/chat` endpoint (`conversation is None`, no session to
    persist into).
    """
    if conversation is not None:
        await conversation.session.add_items([{"role": role, "content": content}])


async def _stream_procedure(
    events: AsyncIterator[dict | ProcedureStatus], conversation: Conversation | None,
) -> AsyncIterator[bytes]:
    """Drain one of `app.procedure`'s event generators, forwarding every
    wire-shaped progress/clear/message event to the client the instant it
    is produced -- never buffered until the whole procedure finishes (see
    `app.procedure`'s module docstring) -- then persist the resulting
    `ProcedureContext` on the conversation and emit the one terminal
    `procedure_status` line the client uses to decide whether it is still
    waiting on this procedure for its next message.

    Every `procedure_message` event (the only kind that becomes a permanent,
    restorable chat message -- never `procedure_progress`/
    `procedure_progress_clear`, which are purely transient presentation
    state) is also stored on the conversation's session as it is produced,
    so it survives switching away from this session and back, or a page
    reload, exactly like a normal chat turn already does.

    Holds the conversation's lock for this entire stream (not just until
    this function is first awaited), exactly as the previous non-streaming
    request/response cycle held it for the whole request: a `StreamingResponse`
    body is drained by Starlette *after* the route handler returns, so the
    lock has to live inside this generator to still serialize two messages
    on the same session.
    """
    lock_cm = conversation.lock if conversation is not None else nullcontext()
    async with lock_cm:
        final_status: ProcedureStatus | None = None
        try:
            async for item in events:
                if isinstance(item, ProcedureStatus):
                    final_status = item
                    continue
                if item.get("type") == "procedure_message":
                    await _persist_message(conversation, "assistant", str(item["message"]))
                yield _encode_event(item)
        except Exception as error:
            # Every expected failure mode (KB/MCP errors, extraction
            # failures, step exceptions, summary-generation failures) is
            # already converted into a normal terminal `ProcedureStatus`
            # inside `app.procedure`; reaching here means something truly
            # unexpected happened mid-stream. Never leave a stale progress
            # bubble or a half-open connection -- clear it and end cleanly.
            log_failure(logging.getLogger(__name__), "Procedure event stream failed", error, ())
            if conversation is not None:
                conversation.active_procedure = None
            yield _encode_event({"type": "procedure_progress_clear"})
            yield _encode_event({"type": "procedure_status", "state": "error",
                                  "message": "The procedure could not be completed."})
            return
        if final_status is not None:
            if conversation is not None:
                conversation.active_procedure = (
                    final_status.context if final_status.context is not None
                    and final_status.context.status in {"COLLECTING_INPUTS", "READY", "RUNNING"}
                    else None
                )
            yield _encode_event(procedure_payload(final_status))


async def route_message(request: ChatRequest, conversation: Conversation | None):
    """Deterministic, LLM-free routing priority (a chat session has at most
    one active procedure):
      1. /cancel while a procedure is active (collecting inputs or ready)
      2. any reply while a procedure is waiting on input
      3. an explicit /procedure prefix starts a new procedure
      4. everything else is the existing, unchanged normal chat path

    Only the two paths that can run slow, multi-step backend work needing
    live progress (a new `/procedure` command, or a reply to one already
    waiting on input) return a streaming response; `/cancel` and normal
    chat are instantaneous/single-shot and keep returning a plain dict,
    exactly as before.
    """
    try:
        config = load_config()
    except ValueError:
        raise HTTPException(503, "Check the server's environment configuration.") from None

    active = conversation.active_procedure if conversation is not None else None
    if active is not None and active.status in {"COMPLETED", "STOPPED", "FAILED", "CANCELLED"}:
        conversation.active_procedure = None
        active = None
    procedure_route = (is_procedure_command(request.message) or (
        active is not None and (active.status == "COLLECTING_INPUTS" or is_cancel_command(request.message))
    ))
    logger.info("session=%s route=%s", conversation.session_id if conversation else "stateless",
                "procedure" if procedure_route else "operations_agent")

    if active is not None and is_cancel_command(request.message):
        lock_cm = conversation.lock if conversation is not None else nullcontext()
        async with lock_cm:
            status = handle_cancel(active)
            if conversation is not None:
                conversation.active_procedure = status.context
            await _persist_message(conversation, "user", request.message)
            await _persist_message(conversation, "assistant", status.message)
        return procedure_payload(status)

    if active is not None and active.status == "COLLECTING_INPUTS":
        await _persist_message(conversation, "user", request.message)
        events = handle_procedure_input_reply(request.message, active, config)
        return StreamingResponse(_stream_procedure(events, conversation), media_type="application/x-ndjson")

    if is_procedure_command(request.message):
        await _persist_message(conversation, "user", request.message)
        events = handle_procedure(request.message, config)
        return StreamingResponse(_stream_procedure(events, conversation), media_type="application/x-ndjson")

    session = conversation.session if conversation is not None else None
    lock_cm = conversation.lock if conversation is not None else nullcontext()
    try:
        async with lock_cm:
            response = (await run_agent(request.message, config) if session is None
                        else await run_agent(request.message, config, session=session))
    except MCPConnectionError as error:
        # This application exception contains only server names and error categories.
        raise HTTPException(502, str(error)) from None
    except MaxTurnsExceeded:
        raise HTTPException(422, "The agent reached its turn limit. Try a more specific request.") from None
    except Exception as error:
        log_failure(logging.getLogger(__name__), "Chat request failed", error,
                    (config.model_api_key, *(server.token for server in config.mcp_servers)))
        raise HTTPException(502, "The agent request failed. Check model and MCP connectivity.") from None
    return {"response": response}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    return await route_message(request, None)


@app.post("/api/sessions", status_code=201)
async def create_session():
    conversation = Conversation()
    conversations[conversation.session_id] = conversation
    return conversation.metadata()


@app.get("/api/sessions")
async def list_sessions():
    return [conversation.metadata() for conversation in conversations.values()]


def get_conversation(session_id: str) -> Conversation:
    conversation = conversations.get(session_id)
    if conversation is None:
        raise HTTPException(404, "Unknown conversation session.")
    return conversation


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    conversation = get_conversation(session_id)
    async with conversation.lock:
        return {**conversation.metadata(),
                "messages": await visible_messages(conversation.session)}


@app.post("/api/sessions/{session_id}/messages")
async def session_message(session_id: str, request: ChatRequest):
    conversation = get_conversation(session_id)
    if conversation.title == "New conversation":
        conversation.title = " ".join(request.message.split())[:60]
    conversation.updated_at = timestamp()
    return await route_message(request, conversation)
