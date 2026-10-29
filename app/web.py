import asyncio
import json
import logging
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Annotated, AsyncIterator, Literal

from agents import MaxTurnsExceeded
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints

from app.incidents import Incident, IncidentRequest, IncidentStore, IncidentSummary
from app.agent import run_agent
from app.config import load_config
from app.diagnostics import configure_logging, log_failure
from app.mcp import MCPConnectionError
from app.procedure import (
    handle_cancel, handle_procedure, handle_procedure_input_reply, is_cancel_command, is_procedure_command,
    prepare_procedure, handle_procedure_confirmation,
)
from app.procedure.models import ProcedureContext, ProcedureStatus
from app.sessions import Conversation, IncidentAgentSession, SessionStore, timestamp, visible_messages

configure_logging()
conversations: dict[str, Conversation] = {}
session_store = SessionStore()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    yield
    for task in session_tasks.values():
        task.cancel()
    await asyncio.gather(*session_tasks.values(), return_exceptions=True)
    session_tasks.clear()
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
    `GET /api/sessions/{id}`). Accepted user input is written here before
    work starts; OperationsAgent writes its permanent output through the
    same SDK session. Never called for transient procedure progress text,
    and a no-op for the stateless `/api/chat` endpoint (`conversation is
    None`, no session to persist into).
    """
    if conversation is not None:
        await conversation.session.add_items([{"role": role, "content": content}])
        if role in {"assistant", "system"}:
            message_changed(conversation)
        else:
            conversation.revision += 1


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
                if conversation is not None:
                    if item.get("type") == "procedure_progress":
                        conversation.progress = str(item["message"])
                    elif item.get("type") in {"procedure_progress_clear", "procedure_message"}:
                        conversation.progress = None
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
            record_incident_outcome(conversation, ProcedureStatus(state="error", message=""))
            await _persist_message(conversation, "assistant", "The procedure could not be completed.")
            yield _encode_event({"type": "procedure_progress_clear"})
            yield _encode_event({"type": "procedure_status", "state": "error",
                                  "message": "The procedure could not be completed."})
            return
        if final_status is not None:
            if conversation is not None:
                conversation.active_procedure = (
                    final_status.context if final_status.context is not None
                    and final_status.context.status in {"COLLECTING_INPUTS", "WAITING_FOR_CONFIRMATION", "READY", "RUNNING"}
                    else None
                )
            record_incident_outcome(conversation, final_status)
            yield _encode_event(procedure_payload(final_status))


TERMINAL_PROCEDURE_STATES = {"COMPLETED", "STOPPED", "FAILED", "CANCELLED"}
SessionMessageRoute = Literal[
    "operations_agent", "procedure_start", "procedure_cancel",
    "procedure_confirmation", "procedure_input", "procedure_busy",
]


def select_session_message_route(
    message: str, active: ProcedureContext | None,
) -> SessionMessageRoute:
    """Choose the one owner of an accepted user message.

    Session kind is deliberately absent: incident and normal sessions share
    this router. A procedure owns input only while it explicitly waits for
    confirmation/input (or is still running); otherwise OperationsAgent does.
    """
    if active is None or active.status in TERMINAL_PROCEDURE_STATES:
        return "procedure_start" if is_procedure_command(message) else "operations_agent"
    if is_cancel_command(message) and active.status in {
        "WAITING_FOR_CONFIRMATION", "COLLECTING_INPUTS", "READY",
    }:
        return "procedure_cancel"
    if active.status == "WAITING_FOR_CONFIRMATION":
        return "procedure_confirmation"
    if active.status == "COLLECTING_INPUTS":
        return "procedure_input"
    if active.status == "RUNNING":
        return "procedure_busy"
    return "operations_agent"


def agent_session(conversation: Conversation):
    if conversation.kind != "incident":
        return conversation.session
    incident = incident_store.for_session(conversation.session_id)
    return IncidentAgentSession(conversation.session, incident) if incident else conversation.session


async def route_message(
    request: ChatRequest,
    conversation: Conversation | None,
    *,
    input_persisted=False,
    lock_held=False,
):
    """Route persisted input through the shared state-based session router.

    Procedure discovery, confirmation, input collection and execution retain
    their existing handlers. Every other message, including incident-session
    free text, uses the single OperationsAgent.
    """
    try:
        config = load_config()
    except ValueError:
        raise HTTPException(503, "Check the server's environment configuration.") from None

    active = conversation.active_procedure if conversation is not None else None
    route = select_session_message_route(request.message, active)
    if active is not None and active.status in TERMINAL_PROCEDURE_STATES:
        conversation.active_procedure = None
        active = None
    logger.info("session=%s route=%s", conversation.session_id if conversation else "stateless", route)

    if route == "procedure_cancel":
        lock_cm = (conversation.lock if conversation is not None and not lock_held else nullcontext())
        async with lock_cm:
            status = handle_cancel(active)
            if conversation is not None:
                conversation.active_procedure = status.context
            if not input_persisted:
                await _persist_message(conversation, "user", request.message)
            await _persist_message(conversation, "assistant", status.message)
            record_incident_outcome(conversation, status)
        return procedure_payload(status)

    if route == "procedure_confirmation":
        if not input_persisted:
            await _persist_message(conversation, "user", request.message)
        events = handle_procedure_confirmation(request.message, active, config)
        return StreamingResponse(_stream_procedure(events, conversation), media_type="application/x-ndjson")

    if route == "procedure_input":
        if not input_persisted:
            await _persist_message(conversation, "user", request.message)
        events = handle_procedure_input_reply(request.message, active, config)
        return StreamingResponse(_stream_procedure(events, conversation), media_type="application/x-ndjson")

    if route == "procedure_start":
        if not input_persisted:
            await _persist_message(conversation, "user", request.message)
        events = handle_procedure(request.message, config)
        return StreamingResponse(_stream_procedure(events, conversation), media_type="application/x-ndjson")

    if route == "procedure_busy":
        message = "The procedure is still running. Please wait for it to finish."
        await _persist_message(conversation, "assistant", message)
        return {"response": message}

    session = agent_session(conversation) if conversation is not None else None
    lock_cm = (conversation.lock if conversation is not None and not lock_held else nullcontext())
    try:
        async with lock_cm:
            response = (await run_agent(request.message, config) if session is None
                        else await run_agent(request.message, config, session=session, input_persisted=input_persisted))
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
    conversation = session_store.create()
    conversations[conversation.session_id] = conversation
    return conversation.metadata()


@app.get("/api/sessions")
async def list_sessions():
    return [{**get_conversation(id).metadata(), **work_state(get_conversation(id))} for id in session_store.list_chat_ids()]


def get_conversation(session_id: str) -> Conversation:
    conversation = conversations.get(session_id)
    if conversation is None:
        conversation = session_store.get(session_id)
        if conversation is None:
            raise HTTPException(404, "Unknown conversation session.")
        conversations[session_id] = conversation
    if conversation.deleting:
        raise HTTPException(409, "This session is being deleted.")
    return conversation


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    conversation = get_conversation(session_id)
    # SDK reads are synchronized internally; never wait for the whole run lock.
    revision = conversation.revision
    messages = await visible_messages(conversation.session)
    for message in messages:
        if message.get("type") == "incident":
            incident = incident_store.get(message["incident_id"])
            message["incident"] = incident.model_dump(mode="json") if incident else None
    return {**conversation.metadata(), **work_state(conversation),
            "kind": conversation.kind, "messages": messages, "revision": revision}



@app.post("/api/sessions/{session_id}/messages")
async def session_message(session_id: str, request: ChatRequest, response: Response,
                          prefer: Annotated[str | None, Header()] = None):
    conversation = get_conversation(session_id)
    if conversation.processing:
        raise HTTPException(409, "This session is busy.")
    # Commit the triggering message before scheduling any long-running work.
    # The lock serializes acceptance and protects against concurrent deletion.
    async with conversation.lock:
        if conversation.processing or conversation.deleting:
            raise HTTPException(409, "This session is busy.")
        await _persist_message(conversation, "user", request.message)
        if conversation.kind == "chat" and conversation.title == "New conversation":
            conversation.title = " ".join(request.message.split())[:60]
        conversation.updated_at = timestamp()
        session_store.save(conversation)
        if prefer == "respond-async" or (conversation.active_procedure is not None and (
                conversation.kind == "incident" or conversation.active_procedure.status == "WAITING_FOR_CONFIRMATION")):
            start_session_work(conversation, request, input_persisted=True)
            response.status_code = 202
            return {"session_id": session_id, **work_state(conversation)}
        if conversation.kind in {"chat", "incident"}:
            # Legacy request-owned calls remain serialized across both the
            # durable input write and the agent run. Async UI calls return
            # above and release this lock after scheduling session work.
            return await route_message(
                request, conversation, input_persisted=True, lock_held=True,
            )
    raise HTTPException(500, "Unsupported session kind.")



incident_store = IncidentStore()


@app.post("/incidents", status_code=201, response_model=Incident)
async def receive_incident(request: IncidentRequest):
    incident = incident_store.create(request)
    await ensure_incident_session(incident)
    logger.info("Incident received id=%s severity=%s", incident.id, incident.severity)
    logger.info("Incident persisted id=%s status=%s", incident.id, incident.status)
    start_incident_preparation(incident)
    return incident_store.get(incident.id)


@app.get("/api/incidents")
async def list_incidents():
    summaries = incident_store.list()
    result = []
    for summary in summaries:
        incident = incident_store.get(summary.id)
        await ensure_incident_session(incident)
        result.append({**summary.model_dump(mode="json"), "session_id": incident.session_id,
                       **work_state(get_conversation(incident.session_id))})
    return result



@app.get("/api/incidents/{incident_id}", response_model=Incident)
async def get_incident(incident_id: str):
    incident = incident_store.get(incident_id)
    if incident is None:
        raise HTTPException(404, "Unknown incident.")
    await ensure_incident_session(incident)
    logger.info("Incident opened id=%s", incident.id)
    return incident


async def ensure_incident_session(incident: Incident):
    """Initialize intake and migrated incidents once, under their session lock."""
    conversation = conversations.get(incident.session_id)
    if conversation is None:
        conversation = session_store.get(incident.session_id)
        if conversation is None:
            conversation = session_store.create(session_id=incident.session_id,
                                                title=incident.title, kind="incident")
        conversations[incident.session_id] = conversation
    if conversation.deleting:
        raise HTTPException(409, "This session is being deleted.")
    if conversation.processing:
        return
    async with conversation.lock:
        if not await conversation.session.get_items(limit=1):
            # Persistent SDK message reference; incident record owns its body.
            await conversation.session.add_items([{"type": "incident", "incident_id": incident.id}])
            message_changed(conversation)


# Application-owned tasks survive HTTP disconnects/navigation, but not restart.
# One run per session; separate sessions can run concurrently.
session_tasks: dict[str, asyncio.Task] = {}


def work_state(conversation):
    active = conversation.active_procedure
    return {"processing": conversation.processing, "progress": conversation.progress,
            "unread": bool(conversation.unread), "revision": conversation.revision,
            "awaiting_input": active is not None and active.status == "COLLECTING_INPUTS",
            "confirmation": {"run_id": active.run_id} if active is not None
            and active.status == "WAITING_FOR_CONFIRMATION" else None}


def message_changed(conversation):
    conversation.revision += 1
    conversation.unread = True
    conversation.updated_at = timestamp()
    session_store.save(conversation)


def start_session_work(conversation, request=None, *, work=None, input_persisted=False):
    """Capture the owner, never a browser selection. Optional work is the
    integration seam used by incident preparation. Incident execution MUST
    require human confirmation before its first operational step.
    """
    if conversation.processing or conversation.deleting:
        raise HTTPException(409, "This session is busy.")
    conversation.processing = True
    conversation.progress = "Working"
    conversation.revision += 1
    task = asyncio.create_task(run_session_work(conversation, request, work=work, input_persisted=input_persisted))
    session_tasks[conversation.session_id] = task
    def forget(finished):
        if session_tasks.get(conversation.session_id) is finished:
            session_tasks.pop(conversation.session_id, None)
    task.add_done_callback(forget)
    return task


async def run_session_work(conversation, request, *, work=None, input_persisted=False):
    procedure_owned = work is not None
    try:
        if work is not None:
            await work(conversation)
        else:
            procedure_owned = select_session_message_route(
                request.message, conversation.active_procedure,
            ).startswith("procedure_")
            if conversation.kind == "chat" and conversation.title == "New conversation":
                conversation.title = " ".join(request.message.split())[:60]
            session_store.save(conversation)
            result = await route_message(request, conversation, input_persisted=input_persisted)
            if isinstance(result, StreamingResponse):
                # Drain server-side; browser lifetime never owns this iterator.
                async for _ in result.body_iterator:
                    pass
            elif "response" in result:
                # OperationsAgent already persists via the SDK session.
                message_changed(conversation)
    except asyncio.CancelledError:
        if procedure_owned:
            record_incident_outcome(conversation, ProcedureStatus(state="error", message=""))
        raise
    except Exception:
        # No provider/credential details in persisted error text.
        logger.warning("Session work failed session=%s", conversation.session_id)
        if procedure_owned:
            record_incident_outcome(conversation, ProcedureStatus(state="error", message=""))
        await _persist_message(conversation, "assistant", "The request could not be completed.")
    finally:
        conversation.processing = False
        conversation.progress = None
        conversation.revision += 1
        conversation.updated_at = timestamp()
        session_store.save(conversation)


class ReadRequest(BaseModel):
    revision: int


@app.post("/api/sessions/{session_id}/read")
async def mark_session_read(session_id: str, request: ReadRequest):
    conversation = get_conversation(session_id)
    # Acknowledge only the snapshot actually rendered, not newer unseen output.
    if request.revision == conversation.revision:
        conversation.unread = False
        session_store.save(conversation)
    return work_state(conversation)


@app.get("/api/sessions/{session_id}/state")
async def session_state(session_id: str):
    return work_state(get_conversation(session_id))


async def delete_session_data(conversation):
    # Refuse both background work and legacy request-owned runs. Mark deletion
    # before awaiting SDK cleanup so no new work can start during that await.
    if conversation.processing or conversation.deleting or conversation.lock.locked():
        raise HTTPException(409, "Cannot delete a session while it is busy.")
    conversation.deleting = True
    try:
        await conversation.session.clear_session()
        session_store.delete(conversation.session_id)
        conversation.session.close()
        conversations.pop(conversation.session_id, None)
    finally:
        conversation.deleting = False


@app.delete("/api/sessions/{session_id}", status_code=204)
async def delete_conversation(session_id: str):
    conversation = get_conversation(session_id)
    if conversation.kind != "chat":
        raise HTTPException(409, "Delete incident-owned sessions through the incident endpoint.")
    await delete_session_data(conversation)
    return Response(status_code=204)


@app.delete("/api/incidents/{incident_id}", status_code=204)
async def delete_incident(incident_id: str):
    incident = incident_store.get(incident_id)
    if incident is None:
        raise HTTPException(404, "Unknown incident.")
    await ensure_incident_session(incident)
    await delete_session_data(get_conversation(incident.session_id))
    incident_store.delete(incident_id)
    return Response(status_code=204)


def record_incident_outcome(conversation, status):
    if conversation is None or conversation.kind != "incident":
        return
    incident = incident_store.for_session(conversation.session_id)
    if incident is None:
        return
    state = ("DONE" if status.state == "completed" else "IN_PROGRESS"
             if status.state in {"collecting_inputs", "waiting_for_confirmation"} else "NEW")
    incident_store.update_workflow(incident.id, state,
                                   status.context.run_id if status.context else None)


def start_incident_preparation(incident):
    incident_store.update_workflow(incident.id, "IN_PROGRESS")
    async def prepare(owner):
        context = f"{incident.title}\n\n{incident.message}"
        events = prepare_procedure(context, load_config(), source="incident", source_id=incident.id)
        async for _ in _stream_procedure(events, owner):
            pass
    start_session_work(get_conversation(incident.session_id), work=prepare)


class ConfirmationRequest(BaseModel):
    run_id: str
    action: Literal["approve", "cancel"]


@app.post("/api/sessions/{session_id}/procedure-confirmation", status_code=202)
async def confirm_procedure(session_id: str, request: ConfirmationRequest):
    conversation = get_conversation(session_id)
    if conversation.processing:
        raise HTTPException(409, "This session is busy.")
    async with conversation.lock:
        active = conversation.active_procedure
        if (conversation.processing or conversation.deleting or active is None or active.run_id != request.run_id
                or active.status != "WAITING_FOR_CONFIRMATION" or active.confirmed):
            raise HTTPException(409, "This procedure is no longer waiting for confirmation.")
        message = ChatRequest(message="yes" if request.action == "approve" else "cancel")
        await _persist_message(conversation, "user", message.message)
        start_session_work(conversation, message, input_persisted=True)
        return {"session_id": session_id, **work_state(conversation)}
