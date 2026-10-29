import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

import app.procedure as procedure_module
from app.config import Config, MCPConfig
from app.procedure.models import ProcedureStatus

# The default canned summary matches the old, deterministic
# "Procedure completed successfully." terminal message exactly, so tests
# that only care about the happy path need no changes to their final
# `message`/`messages[-1]` assertions; tests that specifically exercise
# summary generation override `.return_value`/`.side_effect` directly.
DEFAULT_SUMMARY = "Procedure completed successfully."


async def drain_procedure_events(generator) -> tuple[list[dict], ProcedureStatus]:
    """Run one of `app.procedure`'s event-generator functions
    (`handle_procedure`/`handle_procedure_input_reply`, or any of their
    internal helpers) to completion, splitting its output into
    `(events, final_status)`:

    - `events` is every wire-shaped progress/clear/message dict it
      yielded, in the exact order they were produced.
    - `final_status` is the terminal `ProcedureStatus` it ends with
      (always the last item yielded; never `None` for these generators).

    This is test-only plumbing mirroring `app.web._stream_procedure` (the
    real client just forwards `events` to the wire as it drains the same
    generator; it never collects them into a list first).
    """
    events = []
    final_status = None
    async for item in generator:
        if isinstance(item, ProcedureStatus):
            final_status = item
        else:
            events.append(item)
    return events, final_status


def progress_texts(events: list[dict]) -> list[str]:
    """Every `procedure_progress` label, in order, dropping clear/message events."""
    return [event["message"] for event in events if event["type"] == "procedure_progress"]


def message_texts(events: list[dict]) -> list[str]:
    """Every permanent `procedure_message` text, in order."""
    return [event["message"] for event in events if event["type"] == "procedure_message"]


def event_types(events: list[dict]) -> list[str]:
    """Just the `type` of each event, in order -- for asserting exact
    interleaving (e.g. progress -> clear -> message -> progress -> ...).
    """
    return [event["type"] for event in events]


@pytest.fixture
def config():
    return Config("test-model", "http://localhost:11434/v1", "test-model-key", (
        MCPConfig("openshift", "https://openshift.test/mcp", "test-openshift-token", allowed_tools=frozenset({
            "mcp_openshift__get_pod_count", "mcp_openshift__get_status", "mcp_openshift__delete_openshift",
            "mcp_openshift__pods_list_in_namespace",
        })),
        MCPConfig("aap", "https://aap.test/mcp", allowed_tools=frozenset({
            "mcp_aap__get_workflow_status", "mcp_aap__get_status", "mcp_aap__delete_aap",
        })),
        MCPConfig("itsm", "https://itsm.test/mcp", allowed_tools=frozenset({
            "mcp_itsm__get_ticket", "mcp_itsm__get_status", "mcp_itsm__delete_itsm", "mcp_itsm__rag_search_kb",
        })),
    ))


@pytest.fixture
def summarizer(monkeypatch):
    """Mock the final-summary LLM call (`app.procedure.summarizer`); never
    makes a real model call. Tests set `.return_value`/`.side_effect`
    (e.g. return `None` to exercise the deterministic fallback).
    """
    mock = AsyncMock(return_value=DEFAULT_SUMMARY)
    monkeypatch.setattr(procedure_module, "generate_procedure_summary", mock)
    return mock


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


@pytest.fixture(autouse=True)
def durable_sessions(tmp_path, monkeypatch):
    from app import web
    from app.sessions import SessionStore
    monkeypatch.setattr(web, 'session_store', SessionStore(tmp_path / 'sessions.sqlite3'))
    monkeypatch.setattr(web, 'conversations', {})
    yield
    for conversation in web.conversations.values():
        if hasattr(conversation.session, 'close'):
            conversation.session.close()


async def prepare_and_approve(message, config):
    """Input/execution regressions explicitly approve before testing that phase."""
    from app.procedure import handle_procedure, handle_procedure_confirmation
    events, status = await drain_procedure_events(handle_procedure(message, config))
    if status.state == 'waiting_for_confirmation':
        continued, status = await drain_procedure_events(handle_procedure_confirmation('yes', status.context, config))
        events += continued
    return events, status
