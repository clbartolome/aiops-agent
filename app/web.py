from contextlib import asynccontextmanager
from pathlib import Path
import logging
from typing import Annotated

from agents import MaxTurnsExceeded
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, StrictBool, StringConstraints

from app.agent import run_agent
from app.config import load_config
from app.diagnostics import configure_logging, log_failure
from app.mcp import MCPConnectionError, MCPExecutionError, PreparedMCPExecutor
from app.procedure_runtime import (ProcedureRuntime, ProcedureRuntimeValidationError,
                                   ProcedureInputError, WAITING, runtime_message, procedure_result)
from app.procedures import ProcedureRetrievalError, procedure_query, run_procedure
from app.sessions import Conversation, timestamp, visible_messages


configure_logging()
conversations: dict[str, Conversation] = {}
procedure_runtime = ProcedureRuntime()


@asynccontextmanager
async def lifespan(app):
    yield
    for conversation in conversations.values():
        conversation.session.close()
    conversations.clear()
    for run_id in list(procedure_runtime.executors):
        await procedure_runtime.checkpointer.adelete_thread(run_id)
    procedure_runtime.executors.clear()
    procedure_runtime.locks.clear()
    procedure_runtime.identities.clear()
    procedure_runtime._cancel_requested.clear()


app = FastAPI(lifespan=lifespan)
PAGE = Path(__file__).with_name("index.html").read_text(encoding="utf-8")


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    debug_mode: StrictBool = False


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return PAGE


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, str]:
    return await send_message(request)


async def send_message(request: ChatRequest, session=None, conversation=None) -> dict:
    try:
        query = None if conversation is not None and conversation.active_procedure_run_id else procedure_query(request.message)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    try:
        config = load_config()
    except ValueError:
        raise HTTPException(503, "Check the server's environment configuration.") from None
    procedure_state = None
    failure_category = None

    def failure(category):
        nonlocal failure_category
        failure_category = category
    if request.message.strip() == '/cancel' and not (conversation and conversation.active_procedure_run_id):
        response = 'There is no active procedure to cancel.'
        if session is not None:
            await session.add_items([{'role': 'user', 'content': request.message}, {'role': 'assistant', 'content': response}])
        return {'response': response}
    if conversation is not None and conversation.active_procedure_run_id:
        run_id = conversation.active_procedure_run_id
        try:
            if request.message.split()[0] == '/procedure':
                procedure_state = await procedure_runtime.snapshot(run_id)
                response = 'A procedure is already active. Provide the requested information or use /cancel before starting another procedure.'
                await session.add_items([{'role': 'user', 'content': request.message}, {'role': 'assistant', 'content': response}])
                return {'response': response, 'procedure': procedure_result(procedure_state)}
            if request.message.strip() == '/cancel':
                procedure_state = await procedure_runtime.cancel(run_id)
            else:
                payload = await procedure_runtime.parse_reply(run_id, request.message, config)
                procedure_state = await procedure_runtime.resume(run_id, payload)
            response = runtime_message(procedure_state)
        except ProcedureInputError as error:
            procedure_state = await procedure_runtime.snapshot(run_id)
            procedure_runtime._log(run_id, f'category=INPUT_ERROR requested_fields={procedure_state["requested_fields"]} reason={error}')
            response = str(error) + "\n\n" + runtime_message(procedure_state)
        if procedure_state['status'] in {'COMPLETED', 'FAILED', 'CANCELLED'}:
            conversation.active_procedure_run_id = None
        await session.add_items([{'role': 'user', 'content': request.message}, {'role': 'assistant', 'content': response}])
        return {'response': response, 'procedure': procedure_result(procedure_state)}

    async def launch(bound, functions, servers):
        nonlocal procedure_state
        def started(run_id):
            conversation.active_procedure_run_id = run_id
            procedure_runtime._log(run_id, 'retrieval=success compilation=success binding=success runtime_validation=success')
        try:
            executor = PreparedMCPExecutor(bound, functions, servers, config)
            procedure_state = await procedure_runtime.start(bound, executor,
                on_started=started)
        except (ProcedureRuntimeValidationError, MCPExecutionError, ValueError) as error:
            failure(getattr(error, 'category', 'VALIDATION_ERROR'))
            log_failure(logging.getLogger(__name__), "Procedure runtime validation failed", error,
                        (config.model_api_key, *(server.token for server in config.mcp_servers)))
            return ("The procedure was compiled and bound, but is not supported by the execution runtime. "
                    "No steps have been executed.")
        if procedure_state['status'] in WAITING:
            conversation.active_procedure_run_id = procedure_state['run_id']
        elif procedure_state['status'] in {'COMPLETED', 'FAILED', 'CANCELLED'}:
            conversation.active_procedure_run_id = None
        return runtime_message(procedure_state)

    try:
        runner = run_agent if query is None else run_procedure
        message = request.message if query is None else query
        if query is not None and conversation is not None:
            response = await run_procedure(message, config, session=session, on_bound=launch, on_failure=failure)
        elif query is not None and request.debug_mode:
            response = await run_procedure(message, config, session=session, debug=True)
        else:
            response = (await runner(message, config) if session is None
                        else await runner(message, config, session=session))
    except (MCPConnectionError, ProcedureRetrievalError) as error:
        # This application exception contains only server names and error categories.
        raise HTTPException(502, str(error)) from None
    except MaxTurnsExceeded:
        raise HTTPException(422, "The agent reached its turn limit. Try a more specific request.") from None
    except Exception as error:
        log_failure(logging.getLogger(__name__), "Chat request failed", error,
                    (config.model_api_key, *(server.token for server in config.mcp_servers)))
        raise HTTPException(502, "The agent request failed. Check model and MCP connectivity.") from None
    return {"response": response, **({"procedure": procedure_result(procedure_state)} if procedure_state else {}),
            **({'failure_category': failure_category} if failure_category else {})}


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
    active_at_arrival = conversation.active_procedure_run_id
    reject_start = active_at_arrival and request.message.split()[0] == '/procedure'
    cancelled = None
    if active_at_arrival and request.message.strip() == '/cancel':
        # Cancellation signals the runtime even while the session turn is executing.
        # The in-flight MCP call finishes; subsequent steps are skipped.
        cancelled = await procedure_runtime.cancel(active_at_arrival)
    async with conversation.lock:
        if reject_start or cancelled is not None:
            response = (runtime_message(cancelled) if cancelled is not None else
                        'A procedure is already active. Finish it or use /cancel before starting another procedure.')
            if cancelled is not None and conversation.active_procedure_run_id == active_at_arrival:
                conversation.active_procedure_run_id = None
            await conversation.session.add_items([{'role': 'user', 'content': request.message}, {'role': 'assistant', 'content': response}])
            result = {'response': response, **({'procedure': procedure_result(cancelled)} if cancelled is not None else {})}
        else:
            result = await send_message(request, conversation.session, conversation)
        if conversation.title == "New conversation":
            conversation.title = " ".join(request.message.split())[:60]
        conversation.updated_at = timestamp()
        return result
