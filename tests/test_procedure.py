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
    NOT_EXECUTABLE_MESSAGE,
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

# A fully valid KB article matching the fixed Markdown format exactly, so the
# deterministic parser (app.procedure.parser) succeeds end to end.
VALID_PROCEDURE_KB = """# Inspect namespace health

Inspect the current state of an OpenShift namespace and report basic workload health.

## Procedure

**ID:** inspect-namespace-health
**Version:** 1
**Risk:** low
**Confirmation required:** no

## Required information

- **Namespace** — required — OpenShift namespace to inspect.

## Steps

### 1. Verify that the namespace exists

Check that **Namespace** exists in the OpenShift cluster.

If the namespace does not exist, stop the procedure and inform the user.

### 2. List pods in the namespace

Retrieve all pods running in **Namespace**.

### 3. Review recent events

Retrieve recent events from **Namespace**, including warnings and errors when available.

### 4. Inspect deployments

Retrieve the deployments configured in **Namespace** and their current state.

## Success

The procedure is successful when the namespace exists and the pod, event, and deployment information has been retrieved successfully.
"""


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


# --- Successful KB retrieval + deterministic parsing --------------------------

def test_successful_kb_retrieval_and_parse(config, itsm_mcp):
    """End-to-end: /procedure request -> rag_search_kb (once) -> KB found ->
    parse that exact KB -> ProcedureDefinition -> two sequential chat messages.
    """
    itsm_mcp.call_tool.return_value = kb_result(
        {"id": 1, "title": "Inspect namespace health", "description": VALID_PROCEDURE_KB, "score": 0.9},
        {"id": 2, "title": "Other article", "description": VALID_PROCEDURE_KB, "score": 0.1},
    )
    status = asyncio.run(handle_procedure("/procedure inspect namespace health", config))

    # rag_search_kb was called exactly once; parsing never re-queries the KB.
    assert len(itsm_mcp.calls) == 1

    assert status.state == "parsed"
    assert status.title == "Inspect namespace health"
    assert status.messages == (
        "KB found: Inspect namespace health",
        status.message,
    )
    assert status.message == (
        "Procedure parsed successfully.\n"
        "\n"
        "Required inputs:\n"
        "- Namespace\n"
        "\n"
        "Steps:\n"
        "1. Verify that the namespace exists\n"
        "2. List pods in the namespace\n"
        "3. Review recent events\n"
        "4. Inspect deployments"
    )


def test_parse_error_kb_stops_without_falling_back_to_agent(config, itsm_mcp):
    """A KB with the `## Procedure` marker but invalid/missing metadata must
    produce a controlled parse error, never fall back to the OperationsAgent.
    """
    broken_kb = (
        "# Inspect namespace health\n\n"
        "## Procedure\n\n"
        "**ID:** inspect-namespace-health\n"
        "**Risk:** low\n"
        "**Confirmation required:** no\n"
    )
    itsm_mcp.call_tool.return_value = kb_result(
        {"id": 1, "title": "Inspect namespace health", "description": broken_kb, "score": 0.9}
    )
    status = asyncio.run(handle_procedure("/procedure inspect namespace health", config))
    assert status.state == "parse_error"
    assert status.messages == (
        "KB found: Inspect namespace health",
        "Unable to parse the procedure.\n\nMissing required metadata: Version.",
    )
    assert status.message == status.messages[1]


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
    assert status.message == NOT_EXECUTABLE_MESSAGE


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
