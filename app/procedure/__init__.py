"""Procedure-mode routing: deterministic detection, KB retrieval via the
existing ITSM MCP/RAG integration, executable-procedure verification,
deterministic Markdown parsing into a fixed `ProcedureDefinition`, the
LangGraph runtime that executes it, and the real Agents SDK/MCP Step
Executor it calls for each step.

LangGraph owns orchestration (current step, pause/resume, confirmation,
completion, failure, cancellation); the Step Executor it calls
(`app.procedure.step_executor.make_step_executor`) is only asked "can you
complete this exact step?" and never controls sequencing. Routing itself is
always deterministic: the LLM never invents the procedure plan or decides
routing.
"""
import json
import logging
import re
from collections.abc import Awaitable, Callable

from agents.mcp import MCPServerManager

from app.config import Config
from app.diagnostics import log_failure
from app.mcp import MCPConnectionError, create_mcp_servers
from app.procedure.extraction import extract_field_values
from app.procedure.models import ProcedureDefinition
from app.procedure.parser import ProcedureParseError, normalize_identifier, parse_procedure_markdown
from app.procedure.runtime import (
    ProcedureOutcome,
    ProcedureRunNotFound,
    ProcedureRuntime,
    StepExecutor,
    WAITING_FOR_APPROVAL,
    WAITING_FOR_CONFIRMATION,
    WAITING_FOR_INPUT,
    default_runtime,
)
from app.procedure.step_executor import make_step_executor

logger = logging.getLogger(__name__)

PROCEDURE_PREFIX = "/procedure"
CANCEL_COMMAND = "/cancel"
PROCEDURE_SECTION_MARKER = "## Procedure"

EMPTY_PROCEDURE_REQUEST = (
    "A procedure request must include a description after /procedure. "
    "Example: /procedure inspect namespace health"
)
ITSM_NOT_CONFIGURED = (
    "The ITSM knowledge base is not configured; procedure mode is unavailable."
)
NO_KB_ARTICLE_FOUND = "No matching knowledge base article was found for this request."
NO_ACTIVE_PROCEDURE = "There is no active procedure to cancel."

# Deterministic name hints used to find the existing ITSM KB/RAG search tool
# among the tools the ITSM MCP server exposes. No LLM is used to pick a tool.
_KB_SEARCH_NAME_HINTS = ("kb", "knowledge")


def is_procedure_request(message: str) -> bool:
    """Deterministic routing check. Never uses a model to classify the request.

    Only an exact "/procedure" prefix (optionally followed by whitespace and a
    query) enters procedure mode; a word like "/procedurelist" does not.
    """
    stripped = message.strip()
    lowered = stripped.lower()
    if not lowered.startswith(PROCEDURE_PREFIX):
        return False
    rest = stripped[len(PROCEDURE_PREFIX):]
    return rest == "" or rest[0].isspace()


def extract_procedure_query(message: str) -> str:
    """Strip the leading "/procedure" marker and surrounding whitespace."""
    return message.strip()[len(PROCEDURE_PREFIX):].strip()


def _is_kb_search_tool(name: str) -> bool:
    lowered = name.lower()
    return "search" in lowered and any(hint in lowered for hint in _KB_SEARCH_NAME_HINTS)


def _best_match(result) -> dict | None:
    """Parse a CallToolResult's text content into the best-ranked KB article.

    The existing ITSM KB/RAG tool is assumed to already rank its results; this
    takes the top one without re-scoring or re-ranking anything itself.
    """
    for item in result.content:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("articles"), list):
            data = data["articles"]
        if isinstance(data, list):
            return data[0] if data else None
        if isinstance(data, dict):
            return data
    return None


def _article_text(article: dict, *keys: str) -> str:
    for key in keys:
        value = article.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


async def search_itsm_kb(query: str, config: Config) -> dict | None:
    """Search the existing ITSM MCP/RAG integration for the best-matching KB
    article and return it, or None if ITSM is not configured, it exposes no
    KB/RAG search tool, or there is no match.

    This does not implement another RAG system and never talks to the ITSM
    backend directly: it only goes through the already-configured ITSM MCP
    server, the same way the direct-agent path does for its other tools.
    """
    itsm_config = next((server for server in config.mcp_servers if server.name == "itsm"), None)
    if itsm_config is None:
        return None
    secrets = (itsm_config.token,)
    manager = MCPServerManager(create_mcp_servers((itsm_config,)))
    async with manager:
        if manager.errors:
            for server, error in manager.errors.items():
                log_failure(logger, f"MCP connection failed server={server.name}", error, secrets)
            failures = ", ".join(
                f"{server.name} ({type(error).__name__})" for server, error in manager.errors.items()
            )
            raise MCPConnectionError(f"MCP connection/initialization failed: {failures}")
        if not manager.active_servers:
            return None
        server = manager.active_servers[0]
        logger.info("ITSM KB search tool discovery starting")
        tools = await server.list_tools()
        tool = next((tool for tool in tools if _is_kb_search_tool(tool.name)), None)
        if tool is None:
            logger.info("ITSM KB search tool not found among %d tools", len(tools))
            return None
        required = (tool.input_schema or {}).get("required") or []
        argument_name = required[0] if required else "query"
        logger.info("ITSM KB search tool=%s", tool.name)
        result = await server.call_tool(tool.name, {argument_name: query})
    return _best_match(result)


async def _retrieve_definition(query: str, config: Config) -> tuple[ProcedureDefinition | None, str | None]:
    """Search, verify the executable-procedure marker, and parse.

    Returns `(definition, None)` on success, or `(None, message)` with a
    controlled, user-safe response when retrieval or parsing does not
    produce an executable procedure.
    """
    article = await search_itsm_kb(query, config)
    if article is None:
        return None, NO_KB_ARTICLE_FOUND
    title = _article_text(article, "title", "name") or "Untitled article"
    content = _article_text(article, "content", "body", "markdown", "text")
    if PROCEDURE_SECTION_MARKER not in content:
        return None, (
            f'Found knowledge base article "{title}", but it is a knowledge article, '
            f"not an executable procedure (missing a '{PROCEDURE_SECTION_MARKER}' section)."
        )
    try:
        definition = parse_procedure_markdown(content)
    except ProcedureParseError as error:
        logger.info("Procedure parse failed title=%r error=%s", title, error)
        return None, f'Found a candidate procedure article "{title}", but it could not be parsed: {error}'
    logger.info(
        "Procedure parsed procedure=%s version=%d inputs=%s steps=%s",
        definition.id, definition.version,
        [item.name for item in definition.inputs], [step.id for step in definition.steps],
    )
    return definition, None


async def start_procedure_run(message: str, config: Config, conversation=None, *,
                               runtime: ProcedureRuntime = default_runtime,
                               step_executor_factory: Callable[[Config], StepExecutor] = make_step_executor,
                               ) -> str:
    """Handle a new "/procedure ..." request when no run is already active.

    Retrieves and parses the KB article, then hands the fixed
    `ProcedureDefinition` to the LangGraph runtime, which decides whether the
    run pauses (missing input / confirmation) or finishes immediately. The
    real Step Executor (bound to this request's model/MCP `config`) is
    created once per run and reused for every step and every resume of that
    run; tests inject a fake `step_executor_factory` to avoid any real
    MCP/model calls.
    """
    query = extract_procedure_query(message)
    if not query:
        return EMPTY_PROCEDURE_REQUEST
    definition, error_message = await _retrieve_definition(query, config)
    if definition is None:
        return error_message

    outcome = await runtime.start(definition, step_executor=step_executor_factory(config))
    if conversation is not None and outcome.status in (
        WAITING_FOR_INPUT, WAITING_FOR_CONFIRMATION, WAITING_FOR_APPROVAL,
    ):
        conversation.active_procedure_run_id = outcome.run_id
    return outcome.message


# Deterministic, structured resume convention (e.g. "namespace=payments" or
# "Namespace: payments"). This is intentionally not natural-language
# extraction; that can be added later without changing LangGraph's contract.
_FIELD_VALUE_RE = re.compile(r"([A-Za-z][A-Za-z0-9 _-]*?)\s*[:=]\s*([^,\n]+)")
_CONFIRM_YES = {"yes", "y", "confirm", "approve", "approved", "continue", "proceed"}
_CONFIRM_NO = {"no", "n", "reject", "rejected", "decline", "declined", "stop"}


def _parse_confirmation_reply(message: str) -> bool | None:
    normalized = message.strip().lower()
    if normalized in _CONFIRM_YES:
        return True
    if normalized in _CONFIRM_NO:
        return False
    return None


def _parse_field_values(message: str, field_names: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in _FIELD_VALUE_RE.finditer(message):
        key = normalize_identifier(match.group(1))
        if key in field_names:
            values[key] = match.group(2).strip()
    if not values and len(field_names) == 1:
        # A single requested field: accept a bare reply as its value.
        bare = message.strip()
        if bare:
            values[field_names[0]] = bare
    return values


async def resume_procedure_run(message: str, config: Config, conversation, *,
                                runtime: ProcedureRuntime = default_runtime,
                                field_extractor: Callable[[str, list[str], Config], Awaitable[dict[str, str]]]
                                = extract_field_values,
                                ) -> str:
    """Handle a message while `conversation.active_procedure_run_id` is set
    and the message is not "/cancel". Interprets the raw reply
    deterministically against whatever the run is currently waiting for.
    """
    run_id = conversation.active_procedure_run_id
    pending = runtime.peek(run_id)
    if pending is None:
        # Defensive: the run is gone (e.g. process restarted). Clear the
        # stale association instead of resuming a thread that no longer exists.
        conversation.active_procedure_run_id = None
        return NO_ACTIVE_PROCEDURE

    if pending.status in (WAITING_FOR_CONFIRMATION, WAITING_FOR_APPROVAL):
        approved = _parse_confirmation_reply(message)
        if approved is None:
            return "Please reply 'yes' to proceed or 'no' to cancel."
        outcome = await runtime.resume(run_id, approved)
    elif pending.status == WAITING_FOR_INPUT:
        field_names = [field["name"] for field in pending.waiting_fields]
        values = _parse_field_values(message, field_names)
        if not values and len(field_names) > 1:
            # Only reached for a natural-language reply covering more than
            # one field; a single requested field is always resolved
            # deterministically above (a bare reply is accepted as-is).
            values = await field_extractor(message, field_names, config)
        if not values:
            return "I did not recognize any of the requested values.\n\n" + pending.message
        outcome = await runtime.resume(run_id, values)
    else:  # pragma: no cover - defensive: only waiting statuses are ever "pending"
        conversation.active_procedure_run_id = None
        return NO_ACTIVE_PROCEDURE

    if outcome.status in (WAITING_FOR_INPUT, WAITING_FOR_CONFIRMATION, WAITING_FOR_APPROVAL):
        conversation.active_procedure_run_id = run_id
    else:
        conversation.active_procedure_run_id = None
    return outcome.message


def _is_cancel_command(message: str) -> bool:
    return message.strip().lower() == CANCEL_COMMAND


async def cancel_active_run(conversation, *, runtime: ProcedureRuntime = default_runtime) -> str:
    """Cancel the session's active run, if any. No automatic rollback."""
    run_id = conversation.active_procedure_run_id
    if not run_id:
        return NO_ACTIVE_PROCEDURE
    try:
        outcome: ProcedureOutcome | None = await runtime.cancel(run_id)
    except ProcedureRunNotFound:
        outcome = None
    conversation.active_procedure_run_id = None
    return outcome.message if outcome is not None else NO_ACTIVE_PROCEDURE


async def handle_message(message: str, config: Config, conversation=None, *,
                          runtime: ProcedureRuntime = default_runtime,
                          step_executor_factory: Callable[[Config], StepExecutor] = make_step_executor,
                          field_extractor: Callable[[str, list[str], Config], Awaitable[dict[str, str]]]
                          = extract_field_values,
                          ) -> str | None:
    """The deterministic top-level procedure router.

    Routing priority (never decided by an LLM):
        1. "/cancel" — cancel the active run, or a controlled "no active
           procedure" response if there is none.
        2. an active run waiting for this session — resume it.
        3. "/procedure ..." — start a new run.
        4. anything else — return None so the caller falls through to the
           existing OperationsAgent, unchanged.
    """
    if _is_cancel_command(message):
        if conversation is not None and conversation.active_procedure_run_id:
            response = await cancel_active_run(conversation, runtime=runtime)
        else:
            response = NO_ACTIVE_PROCEDURE
    elif conversation is not None and conversation.active_procedure_run_id:
        response = await resume_procedure_run(
            message, config, conversation, runtime=runtime, field_extractor=field_extractor,
        )
    elif is_procedure_request(message):
        response = await start_procedure_run(
            message, config, conversation, runtime=runtime, step_executor_factory=step_executor_factory,
        )
    else:
        return None

    if conversation is not None:
        # Reuse the existing session mechanism: no second chat-history store.
        await conversation.session.add_items([
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ])
    return response
