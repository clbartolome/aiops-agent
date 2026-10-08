"""Deterministic `/procedure` routing, knowledge-base retrieval, and parsing.

Routing is a pure prefix check: only messages that start with the explicit
`/procedure` command enter this module. No LLM is used to classify or route
messages, and natural-language mentions of "procedure" never reach here.

The flow for this iteration is:

    /procedure request -> rag_search_kb -> KB found -> parse that exact KB

The KB is retrieved exactly once via the fixed `rag_search_kb` MCP tool; the
retrieved title/content is then parsed deterministically with plain Python
(see `app.procedure.parser`) — no second MCP call, and no LLM involved in
either retrieval or parsing. User parameter extraction and step execution
are not implemented yet.
"""
import json
import logging
import re

from agents.mcp import MCPServerManager

from app.config import Config
from app.diagnostics import log_failure
from app.mcp import create_mcp_servers
from app.procedure.models import KBResult, ProcedureContext, ProcedureDefinition, ProcedureRequest, ProcedureStatus
from app.procedure.parser import ProcedureParseError, parse_procedure

logger = logging.getLogger(__name__)

PROCEDURE_PREFIX = "/procedure"
PROCEDURE_MARKER = "## Procedure"
RAG_SEARCH_TOOL = "rag_search_kb"
ITSM_SERVER_NAME = "itsm"

EMPTY_QUERY_MESSAGE = "Please provide a procedure request after /procedure."
NO_RESULT_MESSAGE = "No matching procedure was found."
NOT_EXECUTABLE_MESSAGE = "KB found, but it is not an executable procedure."
MCP_FAILURE_MESSAGE = "Unable to retrieve a matching procedure from the knowledge base."

# Matches "/procedure" alone, or "/procedure" followed by whitespace and the
# rest of the request. Anything else (e.g. "/proceduresomething") is rejected.
_COMMAND_PATTERN = re.compile(rf"^{re.escape(PROCEDURE_PREFIX)}(?:\s+(.*))?$", re.DOTALL)


class ProcedureKBError(RuntimeError):
    """A credential-free failure retrieving a KB procedure via rag_search_kb."""


def is_procedure_command(message: str) -> bool:
    """True only for messages explicitly starting with the `/procedure` command."""
    return _COMMAND_PATTERN.match(message.strip()) is not None


def extract_procedure_query(message: str) -> str:
    """Return the trimmed text after `/procedure` (empty if none was supplied)."""
    match = _COMMAND_PATTERN.match(message.strip())
    if not match:
        return ""
    return (match.group(1) or "").strip()


def is_executable_procedure(content: str) -> bool:
    """A KB article is an executable procedure only if it has a `## Procedure` section."""
    return PROCEDURE_MARKER in content


def _find_itsm_config(config: Config):
    return next((server for server in config.mcp_servers if server.name == ITSM_SERVER_NAME), None)


async def _call_rag_search_kb(query: str, config: Config) -> KBResult | None:
    """Call the fixed `rag_search_kb` MCP tool directly, bypassing the LLM.

    No other KB search tool is discovered or considered.
    """
    itsm_config = _find_itsm_config(config)
    if itsm_config is None:
        raise ProcedureKBError("The ITSM MCP server is not configured.")

    servers = create_mcp_servers((itsm_config,))
    async with MCPServerManager(servers, strict=True) as manager:
        server = manager.active_servers[0]
        result = await server.call_tool(RAG_SEARCH_TOOL, {"query": query})

    if result.is_error:
        raise ProcedureKBError("rag_search_kb returned an error result.")

    text = next((part.text for part in result.content if getattr(part, "text", None)), None)
    if text is None:
        return None

    payload = json.loads(text)
    results = payload.get("results") or []
    if not results:
        return None

    best = max(results, key=lambda item: item.get("score", 0))
    title = str(best.get("title") or "").strip()
    if not title:
        return None
    return KBResult(title=title, content=str(best.get("description") or ""))


def _format_parsed_summary(procedure: ProcedureDefinition) -> str:
    """Build the user-visible, human-readable parsed-procedure summary.

    Never renders raw Markdown or raw Pydantic/JSON output.
    """
    required = [item for item in procedure.inputs if item.required]
    optional = [item for item in procedure.inputs if not item.required]

    lines = ["Procedure parsed successfully.", ""]
    if required:
        lines.append("Required inputs:")
        lines.extend(f"- {item.label}" for item in required)
    else:
        lines.append("No required inputs.")
    if optional:
        lines.append("")
        lines.append("Optional inputs:")
        for item in optional:
            suffix = f" (default: {item.default})" if item.default is not None else ""
            lines.append(f"- {item.label}{suffix}")
    lines.append("")
    lines.append("Steps:")
    lines.extend(f"{index}. {step.title}" for index, step in enumerate(procedure.steps, start=1))
    return "\n".join(lines)


def _parse_kb_procedure(message: str, kb: KBResult) -> ProcedureStatus:
    """Parse the already-retrieved KB (no second `rag_search_kb` call) and
    build the two sequential user-visible messages: "KB found: ..." followed
    by either the parsed summary or a controlled parse error.
    """
    kb_found_message = f"KB found: {kb.title}"
    logger.info("Procedure parsing started title=%s", kb.title)
    try:
        procedure = parse_procedure(kb.content)
    except ProcedureParseError as error:
        logger.error("Procedure parsing failed reason=%s", error)
        error_message = f"Unable to parse the procedure.\n\n{error}"
        return ProcedureStatus(
            state="parse_error", message=error_message, title=kb.title,
            messages=(kb_found_message, error_message),
        )

    # Retained for this request only; no execution state is added yet.
    ProcedureContext(
        original_request=message, kb_title=kb.title, kb_content=kb.content, procedure=procedure,
    )
    logger.info(
        "Procedure parsed procedure=%s version=%d inputs=%s steps=%s",
        procedure.id, procedure.version,
        [item.name for item in procedure.inputs], [step.id for step in procedure.steps],
    )

    summary_message = _format_parsed_summary(procedure)
    return ProcedureStatus(
        state="parsed", message=summary_message, title=kb.title,
        messages=(kb_found_message, summary_message),
    )


async def handle_procedure(message: str, config: Config) -> ProcedureStatus:
    """Run the deterministic `/procedure` path and return a UI-facing status.

    Flow: extract the query -> call `rag_search_kb` once -> if the KB is an
    executable procedure, parse it deterministically (no LLM, no second MCP
    call) and report the parsed structure. Does not extract user parameters,
    ask for missing inputs, or execute any step yet.
    """
    query = extract_procedure_query(message)
    if not query:
        return ProcedureStatus(state="empty_query", message=EMPTY_QUERY_MESSAGE)

    secrets = (config.model_api_key, *(server.token for server in config.mcp_servers))
    try:
        kb = await _call_rag_search_kb(query, config)
    except Exception as error:
        log_failure(logger, "Procedure KB retrieval failed", error, secrets)
        return ProcedureStatus(state="mcp_error", message=MCP_FAILURE_MESSAGE)

    request = ProcedureRequest(original_message=message, query=query, kb=kb)

    if request.kb is None:
        return ProcedureStatus(state="no_result", message=NO_RESULT_MESSAGE)

    if not is_executable_procedure(request.kb.content):
        return ProcedureStatus(
            state="not_executable",
            message=NOT_EXECUTABLE_MESSAGE,
            title=request.kb.title,
        )

    return _parse_kb_procedure(message, request.kb)
