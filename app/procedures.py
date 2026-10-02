"""Explicit procedure lookup through the existing ITSM MCP knowledge base."""
import json
import logging
import re

from agents import Session

from app.config import Config
from app.diagnostics import log_failure
from app.mcp import create_mcp_servers

logger = logging.getLogger(__name__)
EMPTY_PROCEDURE = "Provide a procedure request after /procedure."


class ProcedureRetrievalError(RuntimeError):
    """A credential-free KB retrieval failure."""


def procedure_query(message: str) -> str | None:
    parts = message.strip().split(maxsplit=1)
    if not parts or parts[0] != "/procedure":
        return None
    if len(parts) == 1:
        raise ValueError(EMPTY_PROCEDURE)
    return parts[1].strip()


def has_procedure_section(markdown: str) -> bool:
    # Check only the marker, excluding examples inside fenced/indented code.
    fence = None
    for line in markdown.splitlines():
        text = line.lstrip(" ")
        if len(line) - len(text) > 3:
            continue
        if fence:
            if re.fullmatch(re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*", text):
                fence = None
            continue
        opening = re.match(r"(`{3,}|~{3,})", text)
        if opening:
            fence = opening[0]
        elif re.fullmatch(r"##[ \t]+Procedure(?:[ \t]+#+)?[ \t]*", text):
            return True
    return False


def kb_payload(result) -> dict:
    wire = result.model_dump(by_alias=True)
    if wire.get("isError"):
        raise ProcedureRetrievalError("The ITSM knowledge-base tool returned an error.")
    payload = wire.get("structuredContent")
    if payload is None:
        text = next((part.text for part in result.content if part.type == "text"), "")
        payload = json.loads(text)
    if isinstance(payload, dict) and "result" in payload:
        payload = payload["result"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ProcedureRetrievalError("The ITSM knowledge-base response was invalid.")
    return payload


async def run_procedure(query: str, config: Config, session: Session | None = None) -> str:
    query = query.strip()
    if not query:
        raise ValueError(EMPTY_PROCEDURE)
    itsm = tuple(server for server in config.mcp_servers if server.name == "itsm")
    if not itsm:
        raise ProcedureRetrievalError("The ITSM knowledge-base MCP server is not configured.")
    server = create_mcp_servers(itsm)[0]
    try:
        async with server:
            tools = {tool.name for tool in await server.list_tools()}
            if not {"rag_search_kb", "get_kb_article"} <= tools:
                raise ProcedureRetrievalError("The required ITSM knowledge-base tools are unavailable.")
            logger.info("Procedure KB search tool=rag_search_kb")
            search = kb_payload(await server.call_tool("rag_search_kb", {"query": query, "top_k": 1}))
            matches = search["results"]
            if not isinstance(matches, list):
                raise ProcedureRetrievalError("The ITSM knowledge-base search response was invalid.")
            if not matches:
                response = "No matching knowledge-base article was found for this procedure request."
            else:
                article_id = matches[0]["id"]
                if type(article_id) is not int:
                    raise ProcedureRetrievalError("The ITSM knowledge-base article ID was invalid.")
                logger.info("Procedure KB retrieval tool=get_kb_article")
                article = kb_payload(await server.call_tool("get_kb_article", {"article_id": article_id}))
                markdown, title = article["description"], article["title"]
                if not isinstance(markdown, str) or not isinstance(title, str):
                    raise ProcedureRetrievalError("The ITSM knowledge-base article was invalid.")
                if has_procedure_section(markdown):
                    status = "Procedure candidate found. No steps have been executed."
                else:
                    status = ("A relevant knowledge article was found, but it is not an executable "
                              "procedure (missing `## Procedure`).")
                response = f"{status}\n\n# {title}\n\n{markdown}"
    except Exception as error:
        log_failure(logger, "Procedure KB retrieval failed", error,
                    (config.model_api_key, *(server.token for server in config.mcp_servers)))
        raise ProcedureRetrievalError(
            "Procedure knowledge-base retrieval failed. Check ITSM MCP connectivity and KB tools."
        ) from None
    if session is not None:
        await session.add_items([
            {"role": "user", "content": f"/procedure {query}"},
            {"role": "assistant", "content": response},
        ])
    return response
