"""Explicit procedure lookup through the existing ITSM MCP knowledge base."""
import json
import logging
import re

from agents import Session
from agents.mcp import MCPServerManager

from app.config import Config
from app.diagnostics import log_failure
from app.mcp import create_mcp_servers, discover_available_tools, discover_function_tools
from app.procedure_compiler import ProcedureCompilationError, compile_procedure
from app.procedure_binding import ProcedureBindingError, bind_procedure, binding_summary

logger = logging.getLogger(__name__)
EMPTY_PROCEDURE = "Provide a procedure request after /procedure."


class ProcedureRetrievalError(RuntimeError):
    """A credential-free KB retrieval failure."""
    category = 'RETRIEVAL_ERROR'


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


async def bind_current_tools(procedure, config: Config, itsm_server, *, on_bound=None):
    """Reuse the retrieval connection and discover only additional target systems."""
    secrets = (config.model_api_key, *(item.token for item in config.mcp_servers))
    configured = {item.name for item in config.mcp_servers}
    for step in procedure.steps:
        if step.system is None or step.system not in configured:
            reason = "Target system is unknown." if step.system is None else "No matching MCP server."
            log_failure(logger, "Procedure binding failed", RuntimeError(
                f"procedure={procedure.id} step={step.id} error={reason}"), secrets)
            raise ProcedureBindingError(reason)
    target_systems = {step.system for step in procedure.steps} - {itsm_server.name}
    try:
        servers = create_mcp_servers(tuple(item for item in config.mcp_servers if item.name in target_systems))
        async with MCPServerManager(servers) as manager:
            if manager.errors:
                for server, error in manager.errors.items():
                    log_failure(logger, "Procedure binding MCP discovery failed", error, secrets,
                                context=f"procedure={procedure.id} step={next(step.id for step in procedure.steps if step.system == server.name)} server={server.name} error=")
                raise ProcedureBindingError("Required MCP server could not be discovered.")
            active = [itsm_server, *manager.active_servers]
            if on_bound is None:
                tools = await discover_available_tools(active)
            else:
                functions = await discover_function_tools(active)
                tools = await discover_available_tools(active, function_tools=functions)
            bound = await bind_procedure(procedure, tools, config)
            if on_bound is not None:
                bound = await on_bound(bound, functions, active)
        if manager.errors:
            raise ProcedureBindingError("MCP discovery cleanup failed.")
        return bound
    except ProcedureBindingError:
        raise
    except Exception as error:
        log_failure(logger, "Procedure binding MCP discovery failed", error, secrets,
                    context=f"procedure={procedure.id} error=")
        raise ProcedureBindingError("MCP tool discovery failed.") from None


async def run_procedure(query: str, config: Config, session: Session | None = None, *, on_bound=None, on_failure=None, debug=False) -> str:
    query = query.strip()
    if not query:
        raise ValueError(EMPTY_PROCEDURE)
    itsm = tuple(server for server in config.mcp_servers if server.name == "itsm")
    if not itsm:
        raise ProcedureRetrievalError("The ITSM knowledge-base MCP server is not configured.")
    server = create_mcp_servers(itsm)[0]
    source = None
    ready = None

    async def capture(bound, functions, servers):
        nonlocal ready
        ready = (bound, functions, servers)
        return bound
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
                logger.info("KB retrieved stage=kb_retrieved article=%s", article_id)
                if has_procedure_section(markdown):
                    logger.info("Procedure marker found stage=procedure_marker article=%s", article_id)
                    source = markdown if re.search(r"^# ", markdown, re.M) else f"# {title}\n\n{markdown}"
                    status = "Procedure candidate found."
                else:
                    status = ("A relevant knowledge article was found, but it is not an executable "
                              "procedure (missing `## Procedure`).")
                response = f"{status}\n\n# {title}\n\n{markdown}"
            if source is not None:
                try:
                    definition = await compile_procedure(source, config)
                except ProcedureCompilationError as error:
                    if on_failure is not None:
                        on_failure(error.category)
                    response = ("A procedure knowledge article was found but cannot currently be executed safely: "
                                "compilation or validation failed. No steps have been executed.")
                else:
                    try:
                        result = (await bind_current_tools(definition, config, server) if on_bound is None else
                                  await bind_current_tools(definition, config, server, on_bound=capture))
                        response = binding_summary(result, debug=debug) if not isinstance(result, str) else result
                    except ProcedureBindingError:
                        ready = None
                        if on_failure is not None:
                            on_failure('BINDING_ERROR')
                        response = ("The procedure was found and compiled, but one or more steps cannot be mapped "
                                    "safely to the available system tools. No steps have been executed.")
    except Exception as error:
        log_failure(logger, "Procedure KB retrieval failed category=RETRIEVAL_ERROR", error,
                    (config.model_api_key, *(server.token for server in config.mcp_servers)))
        raise ProcedureRetrievalError(
            "Procedure knowledge-base retrieval failed. Check ITSM MCP connectivity and KB tools."
        ) from None
    if ready is not None:
        # All retrieval/binding lifecycle checks, including cleanup, have passed.
        # The runtime reuses these SDK servers/handles; it creates no new clients.
        response = await on_bound(*ready)
    if session is not None:
        await session.add_items([
            {"role": "user", "content": f"/procedure {query}"},
            {"role": "assistant", "content": response},
        ])
    return response
