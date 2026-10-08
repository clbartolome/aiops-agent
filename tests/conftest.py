import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from app.config import Config, MCPConfig


@pytest.fixture
def config():
    return Config("test-model", "http://localhost:11434/v1", "test-model-key", (
        MCPConfig("openshift", "https://openshift.test/mcp", "test-openshift-token"),
        MCPConfig("aap", "https://aap.test/mcp"),
        MCPConfig("itsm", "https://itsm.test/mcp"),
    ))


@pytest.fixture
def mcp_boundary(monkeypatch):
    """Fake MCP sessions; use real SDK discovery, execution and cleanup."""
    boundary = SimpleNamespace(sessions={}, servers=[], closed=[], failures={}, tool_results={})
    definitions = {
        "openshift": ("get_pod_count", "namespace"),
        "aap": ("get_workflow_status", "workflow_id"),
        "itsm": ("get_ticket", "ticket_id"),
    }

    boundary.definitions = definitions

    async def connect(server):
        boundary.servers.append(server)
        server.exit_stack.callback(boundary.closed.append, server.name)
        if server.name in boundary.failures:
            raise boundary.failures[server.name]
        name, argument = definitions[server.name]
        tool = Tool(name=name, description=f"Read {argument}", inputSchema={
            "type": "object", "properties": {argument: {"type": "string"}},
            "required": [argument], "additionalProperties": False,
        })

        async def call_tool(tool_name, arguments):
            assert tool_name == name
            if server.name in boundary.tool_results:
                result = boundary.tool_results[server.name]
                if isinstance(result, Exception):
                    raise result
                return result
            data = dict(arguments)
            if name == "get_pod_count":
                data.update(pod_count=3, fake=True)
            elif name == "get_workflow_status":
                data.update(status="succeeded", fake=True)
            elif name == "get_status":
                data.update(server=server.name, status="ready")
            else:
                data.update(status="open", fake=True)
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(data))])

        session = SimpleNamespace(
            list_tools=AsyncMock(return_value=ListToolsResult(tools=[
                tool, Tool(name=f"delete_{server.name.lower()}", description="Delete resources",
                           inputSchema=tool.model_dump(by_alias=True)["inputSchema"]),
            ])),
            call_tool=AsyncMock(side_effect=call_tool),
        )
        boundary.sessions[server.name] = session
        server.session = session

    monkeypatch.setattr(MCPServerStreamableHttp, "connect", connect)
    yield boundary
    assert all(server.session is None for server in boundary.servers)
    assert sorted(boundary.closed) == sorted(server.name for server in boundary.servers)


@pytest.fixture
def risk_boundary(monkeypatch):
    """Fake MCP sessions exposing one tool at each risk level the Step
    Executor's tool-call-boundary gate must classify: `get_status` (READ),
    `launch_job` (WRITE), `delete_<server>` (DESTRUCTIVE), and `handle_event`
    (unclassifiable by name/description, so `UNKNOWN`).

    Kept separate from `mcp_boundary` on purpose: several existing tests
    assert an exact discovered-tool count against that fixture, and this
    one is only used by the procedure risk/approval tests.
    """
    boundary = SimpleNamespace(sessions={}, servers=[], closed=[], calls={}, tool_results={})

    async def connect(server):
        boundary.servers.append(server)
        server.exit_stack.callback(boundary.closed.append, server.name)
        tools = [
            Tool(name="get_status", description="Get the current status", inputSchema={
                "type": "object", "properties": {}, "additionalProperties": True,
            }),
            Tool(name="launch_job", description="Launch a job", inputSchema={
                "type": "object", "properties": {"name": {"type": "string"}},
                "required": ["name"], "additionalProperties": False,
            }),
            Tool(name=f"delete_{server.name.lower()}", description="Delete resources", inputSchema={
                "type": "object", "properties": {}, "additionalProperties": True,
            }),
            Tool(name="handle_event", description="Handle an incoming event", inputSchema={
                "type": "object", "properties": {}, "additionalProperties": True,
            }),
        ]

        async def call_tool(tool_name, arguments):
            boundary.calls.setdefault(server.name, []).append((tool_name, dict(arguments or {})))
            if server.name in boundary.tool_results:
                raise boundary.tool_results[server.name]
            return CallToolResult(content=[
                TextContent(type="text", text=json.dumps({"ok": True, "tool": tool_name})),
            ])

        session = SimpleNamespace(
            list_tools=AsyncMock(return_value=ListToolsResult(tools=tools)),
            call_tool=AsyncMock(side_effect=call_tool),
        )
        boundary.sessions[server.name] = session
        server.session = session

    monkeypatch.setattr(MCPServerStreamableHttp, "connect", connect)
    yield boundary
    assert all(server.session is None for server in boundary.servers)
    assert sorted(boundary.closed) == sorted(server.name for server in boundary.servers)


@pytest.fixture
def e2e_mcp_boundary(monkeypatch):
    """Fake MCP sessions across all three configured servers, tailored to
    the realistic KB fixtures under `fixtures/procedures/`. Covers both
    procedure startup (ITSM KB search) and step execution (openshift/aap)
    in one boundary, since both share the same `MCPServerStreamableHttp`
    transport and only one fixture can patch it per test.

    itsm: `search_knowledge_base`, returning `kb_results` (same shape/role
    as the standalone `itsm_kb_boundary` fixture).
    openshift: `get_namespace_status` (READ), `list_pods` (READ),
    `get_recent_events` (READ), `get_application_logs` (READ).
    aap: `find_job_template` (READ), `get_job_template_details` (READ),
    `launch_job_template` (WRITE), `get_job_status` (READ).

    `tool_results[server][tool_name]` can be set to an `Exception` to force
    a tool failure, or to a dict to override the default success payload.
    `calls[server]` records every `(tool_name, arguments)` pair actually
    sent to that server, for duplicate/side-effect assertions.
    """
    boundary = SimpleNamespace(sessions={}, servers=[], closed=[], calls={}, tool_results={}, kb_results=[])

    tool_specs = {
        "itsm": [
            ("search_knowledge_base", "Search the ITSM knowledge base", {"query": {"type": "string"}}),
        ],
        "openshift": [
            ("get_namespace_status", "Get the status of a namespace", {"namespace": {"type": "string"}}),
            ("list_pods", "List pods in a namespace", {"namespace": {"type": "string"}}),
            ("get_recent_events", "Get recent events in a namespace", {"namespace": {"type": "string"}}),
            ("get_application_logs", "Get recent application logs", {
                "namespace": {"type": "string"}, "application_name": {"type": "string"},
            }),
        ],
        "aap": [
            ("find_job_template", "Find a job template by name", {"name": {"type": "string"}}),
            ("get_job_template_details", "Get job template details", {"name": {"type": "string"}}),
            ("launch_job_template", "Launch a job template", {"name": {"type": "string"}}),
            ("get_job_status", "Get the status of a launched job", {"name": {"type": "string"}}),
        ],
    }

    def default_payload(tool_name: str, arguments: dict) -> dict:
        if tool_name == "search_knowledge_base":
            return boundary.kb_results
        if tool_name == "get_namespace_status":
            return {"namespace": arguments.get("namespace"), "exists": True}
        if tool_name == "list_pods":
            return {"pods": [{"name": "pod-1", "status": "Running"}]}
        if tool_name == "get_recent_events":
            return {"events": []}
        if tool_name == "get_application_logs":
            return {"logs": "no errors"}
        if tool_name == "find_job_template":
            return {"found": True, "name": arguments.get("name")}
        if tool_name == "get_job_template_details":
            return {"inventory": "prod", "playbook": "deploy.yml"}
        if tool_name == "launch_job_template":
            return {"job_id": "job-123", "status": "launched"}
        if tool_name == "get_job_status":
            return {"status": "successful"}
        return {"ok": True}  # pragma: no cover - defensive

    async def connect(server):
        boundary.servers.append(server)
        server.exit_stack.callback(boundary.closed.append, server.name)
        specs = tool_specs.get(server.name, [])
        tools = [
            Tool(name=name, description=description, inputSchema={
                "type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False,
            })
            for name, description, properties in specs
        ]

        async def call_tool(tool_name, arguments):
            boundary.calls.setdefault(server.name, []).append((tool_name, dict(arguments or {})))
            override = boundary.tool_results.get(server.name, {}).get(tool_name)
            if isinstance(override, Exception):
                raise override
            payload = override if override is not None else default_payload(tool_name, arguments or {})
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))])

        session = SimpleNamespace(
            list_tools=AsyncMock(return_value=ListToolsResult(tools=tools)),
            call_tool=AsyncMock(side_effect=call_tool),
        )
        boundary.sessions[server.name] = session
        server.session = session

    monkeypatch.setattr(MCPServerStreamableHttp, "connect", connect)
    yield boundary
    assert all(server.session is None for server in boundary.servers)
    assert sorted(boundary.closed) == sorted(server.name for server in boundary.servers)


@pytest.fixture
def itsm_kb_boundary(monkeypatch):
    """Fake only the ITSM MCP server's existing KB/RAG search tool.

    This is isolated from `mcp_boundary` on purpose: the procedure path only
    ever connects to the ITSM server, and this fixture lets tests assert that
    without perturbing the direct-agent tests' tool counts/behavior.
    """
    boundary = SimpleNamespace(sessions={}, servers=[], closed=[], results=[])
    boundary.tool_name = "search_knowledge_base"

    async def connect(server):
        assert server.name == "itsm", "The procedure path must only connect to the ITSM server"
        boundary.servers.append(server)
        server.exit_stack.callback(boundary.closed.append, server.name)
        tool = Tool(name=boundary.tool_name, description="Search the ITSM knowledge base", inputSchema={
            "type": "object", "properties": {"query": {"type": "string"}},
            "required": ["query"], "additionalProperties": False,
        })

        async def call_tool(tool_name, arguments):
            assert tool_name == boundary.tool_name
            return CallToolResult(content=[TextContent(
                type="text", text=json.dumps(boundary.results),
            )])

        session = SimpleNamespace(
            list_tools=AsyncMock(return_value=ListToolsResult(tools=[tool])),
            call_tool=AsyncMock(side_effect=call_tool),
        )
        boundary.sessions[server.name] = session
        server.session = session

    monkeypatch.setattr(MCPServerStreamableHttp, "connect", connect)
    yield boundary
    assert all(server.session is None for server in boundary.servers)
    assert sorted(boundary.closed) == sorted(server.name for server in boundary.servers)
