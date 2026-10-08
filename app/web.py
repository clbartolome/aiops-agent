from contextlib import asynccontextmanager
from pathlib import Path
import logging
from typing import Annotated

from agents import MaxTurnsExceeded
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    return await send_message(request)


def procedure_payload(status: ProcedureStatus) -> dict:
    # The minimal contract: the final text is always a normal chat message;
    # "state" is kept only in case a later stage needs to branch on it.
    # "messages"/"progress_stages" are only present when a single request
    # produces more than one sequential chat message (e.g. "KB found: ..."
    # then the parsed summary, then the collected-inputs/ready outcome).
    # `status.context` is never serialized: it is server-side-only session
    # state (it may carry the full KB Markdown and collected input values).
    payload = {"type": "procedure_status", "state": status.state, "message": status.message}
    if status.messages:
        payload["messages"] = list(status.messages)
    if status.progress_stages:
        payload["progress_stages"] = list(status.progress_stages)
    return payload


async def send_message(request: ChatRequest, conversation: Conversation | None = None) -> dict:
    try:
        config = load_config()
    except ValueError:
        raise HTTPException(503, "Check the server's environment configuration.") from None

    # Deterministic, LLM-free routing priority (a chat session has at most
    # one active procedure):
    #   1. /cancel while a procedure is active (collecting inputs or ready)
    #   2. any reply while a procedure is waiting on input
    #   3. an explicit /procedure prefix starts a new procedure
    #   4. everything else is the existing, unchanged normal chat path
    active = conversation.active_procedure if conversation is not None else None

    if active is not None and is_cancel_command(request.message):
        status = handle_cancel(active)
        if conversation is not None:
            conversation.active_procedure = status.context
        return procedure_payload(status)

    if active is not None and active.status == "COLLECTING_INPUTS":
        status = await handle_procedure_input_reply(request.message, active, config)
        if conversation is not None:
            conversation.active_procedure = status.context
        return procedure_payload(status)

    if is_procedure_command(request.message):
        status = await handle_procedure(request.message, config)
        if conversation is not None:
            conversation.active_procedure = status.context
        return procedure_payload(status)

    session = conversation.session if conversation is not None else None
    try:
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
    async with conversation.lock:
        result = await send_message(request, conversation)
        if conversation.title == "New conversation":
            conversation.title = " ".join(request.message.split())[:60]
        conversation.updated_at = timestamp()
        return result
