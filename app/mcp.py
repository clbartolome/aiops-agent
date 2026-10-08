from collections.abc import Callable
from functools import partial
from importlib.metadata import version
import logging
from urllib.parse import urlsplit

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


def create_mcp_servers(
    configs: tuple[MCPConfig, ...],
    *,
    failure_error_function: Callable[..., object] | None = None,
) -> list[MCPServerStreamableHttp]:
    """Build the MCP servers shared by every caller (direct agent, procedure
    KB search, procedure Step Executor): identical client/auth/TLS/transport
    construction regardless of caller.

    `failure_error_function` is an optional override of the default
    credential-safe formatter below. It exists so the procedure Step
    Executor can additionally recognize its own tool-call-boundary safety
    signals (approval required, out-of-scope, duplicate side effect)
    without this module knowing anything about procedures; every other
    caller is unaffected and keeps the exact default formatter.
    """
    servers = []
    effective_failure_error_function = failure_error_function or partial(
        mcp_tool_error, secrets=tuple(c.token for c in configs),
    )
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
            failure_error_function=effective_failure_error_function,
        ))
    return servers
