from functools import partial
from importlib.metadata import version
import logging
from urllib.parse import urlsplit

from agents import Agent, RunContextWrapper, Tool
from agents.mcp import MCPServerStreamableHttp, MCPToolCustomDataContext

from app.config import MCPConfig
from app.diagnostics import log_failure

# The SDK's HTTP client factory must match the installed MCP transport stack.
if int(version("mcp").split(".")[0]) >= 2:
    import httpx2 as mcp_http
else:
    import httpx as mcp_http


class MCPConnectionError(RuntimeError):
    """A credential-free MCP lifecycle failure for the CLI."""


def record_mcp_result(context: MCPToolCustomDataContext) -> dict[str, bool]:
    # SDK-only metadata: no tool content, arguments, or credentials are recorded.
    logging.getLogger(__name__).info(
        "MCP tool=%s completed=true status=%s",
        context.tool_display_name, "error" if context.is_error else "success",
    )
    return {"mcp_executed": True, "mcp_success": not context.is_error}


def mcp_tool_error(context, error: Exception, *, secrets=()) -> str:
    log_failure(logging.getLogger(__name__), f"MCP tool={context.tool_name} status=failed", error, secrets)
    return "The information could not be retrieved: the MCP tool call failed."


def create_http_client(headers=None, timeout=None, auth=None, *, verify: bool | None):
    options = {"follow_redirects": False}
    if verify is not None:
        options["verify"] = verify
    if headers is not None:
        options["headers"] = headers
    if timeout is not None:
        options["timeout"] = timeout
    if auth is not None:
        options["auth"] = auth
    return mcp_http.AsyncClient(**options)


def create_mcp_servers(configs: tuple[MCPConfig, ...]) -> list[MCPServerStreamableHttp]:
    servers = []
    for config in configs:
        params = {
            "url": config.url,
            "httpx_client_factory": partial(
                create_http_client,
                verify=config.tls_verify if urlsplit(config.url).scheme == "https" else None,
            ),
        }
        token = (config.token or "").strip()
        if token:
            params["headers"] = {"Authorization": f"Bearer {token}"}
        servers.append(MCPServerStreamableHttp(
            name=config.name,
            params=params,
            cache_tools_list=True,
            custom_data_extractor=record_mcp_result,
            failure_error_function=partial(mcp_tool_error, secrets=tuple(c.token for c in configs)),
        ))
    return servers


def filter_mcp_tools(discovered_tools: list[Tool], allowed_names: set[str] | frozenset[str]) -> list[Tool]:
    """Keep only exact discovered names, retaining SDK invocation and metadata."""
    return [tool for tool in discovered_tools if tool.name in allowed_names]


async def expose_allowed_mcp_tools(agent: Agent, configs: tuple[MCPConfig, ...]) -> None:
    """Discover the complete SDK MCP surface, then freeze its allowed subset.

    These are the SDK's original FunctionTools bound to the live servers;
    lifecycle management remains with the caller's MCPServerManager. Clearing
    the agent's discovery list prevents Runner from adding unfiltered tools.
    Local tools are preserved separately and never subject to MCP allowlists.
    """
    discovered = await agent.get_mcp_tools(RunContextWrapper(context=None))
    exposed: list[Tool] = []
    logger = logging.getLogger(__name__)
    for config in configs:
        prefix = f"mcp_{config.name}__"
        server_tools = [tool for tool in discovered if tool.name.startswith(prefix)]
        configured = config.effective_allowed_tools
        allowed = {name for name in configured if name.startswith(prefix)}
        for name in sorted(configured - allowed):
            logger.warning("MCP server=%s invalid_tool_prefix=%s", config.name, name)
        found = {tool.name for tool in server_tools}
        for name in sorted(configured - found):
            logger.warning("MCP server=%s configured_tool_not_found=%s", config.name, name)
        filtered = filter_mcp_tools(server_tools, allowed)
        logger.info("MCP server=%s discovered=%d allowed=%d exposed=%d",
                    config.name, len(server_tools), len(configured), len(filtered))
        exposed.extend(filtered)
    logger.info("MCP tools exposed count=%d names=%s", len(exposed), [tool.name for tool in exposed])
    agent.tools = [*agent.tools, *exposed]
    agent.mcp_servers = []
