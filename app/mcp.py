from functools import partial
from contextlib import AsyncExitStack, asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
import json
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
    if isinstance(context.run_context.context, ProcedureToolCapture):
        context.run_context.context.result = {
            "isError": bool(context.is_error),
            "structuredContent": (
                deepcopy(dict(context.structured_content))
                if context.structured_content is not None else None
            ),
            "content": deepcopy(context.tool_output),
        }
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


async def discover_function_tools(servers):
    """Existing SDK discovery, retaining invocation handles for an already-bound run."""
    from agents import Agent, RunContextWrapper
    discovery = Agent(name="ProcedureToolDiscovery", mcp_servers=list(servers),
                      mcp_config={"include_server_in_tool_names": True})
    return await discovery.get_all_tools(RunContextWrapper(context=None))


async def discover_available_tools(servers, *, function_tools=None):
    """Normalize SDK-discovered tools on already connected servers; never invoke them."""
    from agents import FunctionTool
    from agents.tool import ToolOriginType, get_function_tool_origin
    from app.procedure_binding_models import AvailableTool

    tools = await discover_function_tools(servers) if function_tools is None else function_tools
    # SDK preserves server/tool order. Join cached raw metadata to those exact
    # handles; no second discovery or manually maintained tool names are needed.
    sources = {server.name: iter(server.cached_tools or []) for server in servers}
    metadata = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            origin = get_function_tool_origin(tool)
            if origin is not None and origin.type == ToolOriginType.MCP:
                source = next(sources.get(origin.mcp_server_name, iter(())), None)
                if source is None or source.model_dump(by_alias=True)['inputSchema'] != tool.params_json_schema:
                    raise ValueError('SDK tool metadata cannot be matched safely.')
                metadata.append(AvailableTool(
                    server=origin.mcp_server_name, name=tool.name,
                    description=tool.description, input_schema=tool.params_json_schema,
                    output_schema=source.model_dump(by_alias=True).get('outputSchema'),
                    annotations=source.annotations.model_dump(by_alias=True, exclude_unset=True) if source.annotations else {},
                    operation_metadata={'http_method': (source.model_dump(by_alias=True).get('_meta') or {}).get('http_method')},
                    raw_name=source.name,
                ))
    return metadata


@dataclass
class ProcedureToolCapture:
    # SDK result capture is opt-in; direct agent custom data/history is unchanged.
    result: dict | None = None


class MCPExecutionError(RuntimeError):
    def __init__(self, category, result=None, *, reason=None):
        super().__init__("The bound MCP tool failed.")
        self.category = category
        self.result = result
        self.reason = reason or category


class PreparedMCPExecutor:
    """Exact SDK handles captured during binding, with lazy existing-server connections."""
    def __init__(self, bound, function_tools, servers, config):
        from agents import FunctionTool
        from agents.tool import ToolOriginType, get_function_tool_origin, set_function_tool_failure_error_function
        self.secrets = (config.model_api_key, *(server.token for server in config.mcp_servers))
        self.servers = {server.name: server for server in servers}
        offered = {}
        for tool in function_tools:
            if isinstance(tool, FunctionTool):
                origin = get_function_tool_origin(tool)
                if origin and origin.type == ToolOriginType.MCP and origin.mcp_server_name:
                    key = (origin.mcp_server_name, tool.name)
                    if key in offered:
                        raise ValueError("Duplicate captured MCP tool identity.")
                    offered[key] = tool
        self.tools, self.schemas = {}, {}
        self.bound = bound.model_copy(deep=True)
        # Metadata from the same cached discovery used by the binder.
        raw = {server.name: iter(server.cached_tools or []) for server in servers}
        self.sources = {}
        bound_keys = {(step.mcp_server, step.tool_name) for step in bound.steps}
        for key in offered:
            source = next(raw[key[0]], None)
            if source is None:
                raise ValueError('Captured MCP tool metadata is unavailable.')
            if key in bound_keys:
                self.sources[key] = source.name
        for step in bound.steps:
            key = (step.mcp_server, step.tool_name)
            if key not in offered or step.mcp_server not in self.servers:
                raise MCPExecutionError('TOOL_CATALOG_CHANGED')
            copied = replace(offered[key], params_json_schema=deepcopy(offered[key].params_json_schema))
            self.tools[key] = set_function_tool_failure_error_function(copied, None)
            self.schemas[key] = deepcopy(copied.params_json_schema)

    @asynccontextmanager
    async def open(self):
        from agents import RunConfig
        from agents.mcp import MCPServerManager
        from agents.tool_context import ToolContext
        connected = set()
        managers = []
        async with AsyncExitStack() as stack:
            async def connect(server):
                if server.name not in connected:
                    if server.session is None:
                        manager = await stack.enter_async_context(MCPServerManager([server], strict=True))
                        managers.append(manager)
                    connected.add(server.name)

            async def verify_catalog():
                # Refresh metadata only to check the frozen identities. Never
                # discover FunctionTools, rebind, or select a substitute here.
                from app.procedure_binding_models import AvailableTool, execution_risk, catalog_fingerprint
                from agents.tool import ToolOriginType, get_function_tool_origin
                for step in self.bound.steps:
                    handle = self.tools.get((step.mcp_server, step.tool_name))
                    origin = get_function_tool_origin(handle) if handle is not None else None
                    if (handle is None or handle.name != step.tool_name or handle.params_json_schema != step.input_schema
                            or origin is None or origin.type != ToolOriginType.MCP or origin.mcp_server_name != step.mcp_server):
                        raise MCPExecutionError('TOOL_CATALOG_CHANGED', reason=f'Missing or changed captured handle server={step.mcp_server} tool={step.tool_name}')
                observed = []
                for server_name in sorted({step.mcp_server for step in self.bound.steps}):
                    server = self.servers[server_name]
                    await connect(server)
                    server.invalidate_tools_cache()
                    current = await server.list_tools()
                    by_name = {tool.name: tool for tool in current}
                    if len(by_name) != len(current):
                        raise MCPExecutionError('TOOL_CATALOG_CHANGED')
                    for step in self.bound.steps:
                        if step.mcp_server != server_name:
                            continue
                        source = by_name.get(self.sources[(server_name, step.tool_name)])
                        if source is None:
                            raise MCPExecutionError('TOOL_CATALOG_CHANGED', reason=f'Bound tool disappeared server={server_name} tool={step.tool_name}')
                        metadata = AvailableTool(server=server_name, name=step.tool_name, raw_name=source.name,
                            input_schema=source.model_dump(by_alias=True)['inputSchema'], output_schema=source.model_dump(by_alias=True).get('outputSchema'),
                            annotations=source.annotations.model_dump(by_alias=True, exclude_unset=True) if source.annotations else {},
                            operation_metadata={'http_method': (source.model_dump(by_alias=True).get('_meta') or {}).get('http_method')})
                        current_step = step.model_copy(update={'input_schema': metadata.input_schema,
                            'output_schema': metadata.output_schema, 'execution_risk': execution_risk(metadata)})
                        if catalog_fingerprint([current_step]) != catalog_fingerprint([step]):
                            raise MCPExecutionError('TOOL_CATALOG_CHANGED', reason=f'Bound schema or risk changed server={server_name} tool={step.tool_name}')
                        observed.append(current_step)
                if catalog_fingerprint(observed) != self.bound.catalog_fingerprint:
                    raise MCPExecutionError('TOOL_CATALOG_CHANGED')

            async def invoke(step, arguments):
                key = (step.mcp_server, step.tool_name)
                if key not in self.tools:
                    raise MCPExecutionError("BoundToolUnavailable")
                server = self.servers[step.mcp_server]
                await connect(server)
                capture = ProcedureToolCapture()
                context = ToolContext(context=capture, tool_name=step.tool_name,
                                      tool_call_id="procedure-" + step.step_id,
                                      tool_arguments=json.dumps(arguments),
                                      run_config=RunConfig(tracing_disabled=True, trace_include_sensitive_data=False))
                try:
                    await self.tools[key].on_invoke_tool(context, json.dumps(arguments))
                except Exception as error:
                    # Raw transport/provider exceptions may include request values.
                    raise MCPExecutionError(type(error).__name__) from None
                if capture.result is None:
                    raise MCPExecutionError("MissingMCPResult")
                if capture.result["isError"]:
                    raise MCPExecutionError("MCPToolError", capture.result)
                return capture.result
            invoke.verify_catalog = verify_catalog
            yield invoke
        if any(manager.errors for manager in managers):
            raise MCPExecutionError("MCPCleanupError")
