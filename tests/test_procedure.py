import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult, TextContent

from app.procedure import (
    EMPTY_QUERY_MESSAGE,
    MCP_FAILURE_MESSAGE,
    NO_RESULT_MESSAGE,
    RAG_SEARCH_TOOL,
    extract_procedure_query,
    handle_procedure,
    is_executable_procedure,
    is_procedure_command,
)

EXECUTABLE_KB = (
    "# Inspect namespace health\n\n"
    "## Procedure\n\n"
    "**ID:** inspect-namespace-health\n"
)
NON_EXECUTABLE_KB = "# Inspect namespace health\n\nJust a description with no procedure markup."


def kb_result(*entries):
    return CallToolResult(content=[
        TextContent(type="text", text=json.dumps({"results": list(entries)}))
    ])


@pytest.fixture
def itsm_mcp(monkeypatch):
    """Patch the MCP server lifecycle so only `rag_search_kb` calls are observed."""
    calls = []
    call_tool = AsyncMock()

    async def connect(self):
        pass

    async def cleanup(self):
        pass

    async def traced_call_tool(self, tool_name, arguments, meta=None):
        calls.append((tool_name, arguments))
        return await call_tool(tool_name, arguments)

    monkeypatch.setattr(MCPServerStreamableHttp, "connect", connect)
    monkeypatch.setattr(MCPServerStreamableHttp, "cleanup", cleanup)
    monkeypatch.setattr(MCPServerStreamableHttp, "call_tool", traced_call_tool)
    return SimpleNamespace(calls=calls, call_tool=call_tool)


# --- Routing (pure string checks, no LLM) -----------------------------------

@pytest.mark.parametrize("message, expected", [
    ("/procedure inspect namespace health", True),
    ("/procedure", True),
    ("/procedure    ", True),
    ("  /procedure inspect namespace health  ", True),
    ("inspect namespace health", False),
    ("what is the procedure for inspecting namespace health?", False),
    ("/proceduresomething", False),
])
def test_is_procedure_command(message, expected):
    assert is_procedure_command(message) is expected


@pytest.mark.parametrize("message, expected", [
    ("/procedure inspect namespace health", "inspect namespace health"),
    ("/procedure   inspect namespace health   ", "inspect namespace health"),
    ("/procedure", ""),
    ("/procedure    ", ""),
])
def test_extract_procedure_query(message, expected):
    assert extract_procedure_query(message) == expected


@pytest.mark.parametrize("content, expected", [
    (EXECUTABLE_KB, True),
    (NON_EXECUTABLE_KB, False),
])
def test_is_executable_procedure(content, expected):
    assert is_executable_procedure(content) is expected


# --- Empty command: never calls the MCP -------------------------------------

@pytest.mark.parametrize("message", ["/procedure", "/procedure    "])
def test_empty_command_does_not_call_mcp(message, config, itsm_mcp):
    status = asyncio.run(handle_procedure(message, config))
    assert status.state == "empty_query"
    assert status.message == EMPTY_QUERY_MESSAGE
    itsm_mcp.call_tool.assert_not_awaited()


# --- Fixed tool and search query --------------------------------------------

def test_calls_exactly_rag_search_kb(config, itsm_mcp):
    itsm_mcp.call_tool.return_value = kb_result(
        {"id": 1, "title": "Inspect namespace health", "description": EXECUTABLE_KB, "score": 0.9}
    )
    asyncio.run(handle_procedure("/procedure inspect namespace health", config))
    assert len(itsm_mcp.calls) == 1
    tool_name, _arguments = itsm_mcp.calls[0]
    assert tool_name == RAG_SEARCH_TOOL


def test_search_query_extracted_from_command(config, itsm_mcp):
    itsm_mcp.call_tool.return_value = kb_result(
        {"id": 1, "title": "Inspect namespace health", "description": EXECUTABLE_KB, "score": 0.9}
    )
    asyncio.run(handle_procedure("/procedure inspect namespace health", config))
    _tool_name, arguments = itsm_mcp.calls[0]
    assert arguments["query"] == "inspect namespace health"


# --- Successful KB retrieval --------------------------------------------------

def test_successful_kb_retrieval(config, itsm_mcp):
    itsm_mcp.call_tool.return_value = kb_result(
        {"id": 1, "title": "Inspect namespace health", "description": EXECUTABLE_KB, "score": 0.9},
        {"id": 2, "title": "Other article", "description": EXECUTABLE_KB, "score": 0.1},
    )
    status = asyncio.run(handle_procedure("/procedure inspect namespace health", config))
    assert status.state == "kb_found"
    assert status.message == "KB found: Inspect namespace health"
    assert status.title == "Inspect namespace health"


# --- KB not found -------------------------------------------------------------

def test_no_matching_kb(config, itsm_mcp):
    itsm_mcp.call_tool.return_value = kb_result()
    status = asyncio.run(handle_procedure("/procedure inspect namespace health", config))
    assert status.state == "no_result"
    assert status.message == NO_RESULT_MESSAGE


# --- Non-executable KB --------------------------------------------------------

def test_non_executable_kb_stops_the_procedure_path(config, itsm_mcp):
    itsm_mcp.call_tool.return_value = kb_result(
        {"id": 1, "title": "Inspect namespace health", "description": NON_EXECUTABLE_KB, "score": 0.9}
    )
    status = asyncio.run(handle_procedure("/procedure inspect namespace health", config))
    assert status.state == "not_executable"
    assert "KB found: Inspect namespace health" in status.message
    assert "This KB is not an executable procedure." in status.message


# --- MCP failure ---------------------------------------------------------------

def test_mcp_failure_returns_controlled_error_state(config, itsm_mcp):
    itsm_mcp.call_tool.side_effect = RuntimeError("boom")
    status = asyncio.run(handle_procedure("/procedure inspect namespace health", config))
    assert status.state == "mcp_error"
    assert status.message == MCP_FAILURE_MESSAGE


def test_mcp_error_result_returns_controlled_error_state(config, itsm_mcp):
    itsm_mcp.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="boom")], is_error=True,
    )
    status = asyncio.run(handle_procedure("/procedure inspect namespace health", config))
    assert status.state == "mcp_error"
    assert status.message == MCP_FAILURE_MESSAGE
