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
