"""Deterministic `/procedure` routing and knowledge-base retrieval.

Routing is a pure prefix check: only messages that start with the explicit
`/procedure` command enter this module. No LLM is used to classify or route
messages, and natural-language mentions of "procedure" never reach here.

This iteration only retrieves and displays the best-matching knowledge-base
article via the existing `rag_search_kb` MCP tool on the ITSM server. It does
not parse, validate, or execute the procedure.
"""
import json
import logging
import re

from agents.mcp import MCPServerManager

from app.config import Config
from app.diagnostics import log_failure
from app.mcp import create_mcp_servers
from app.procedure.models import KBResult, ProcedureRequest, ProcedureStatus

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


async def handle_procedure(message: str, config: Config) -> ProcedureStatus:
    """Run the deterministic `/procedure` path and return a UI-facing status.

    Does not parse, validate, or execute the retrieved procedure; this ends
    once the best-matching KB has been retrieved (or an error state is set).
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

    return ProcedureStatus(
        state="kb_found",
        message=f"KB found: {request.kb.title}",
        title=request.kb.title,
    )
