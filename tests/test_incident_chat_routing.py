"""Incident sessions use the shared state router and OperationsAgent."""
import asyncio

import httpx
import pytest

from app import web
from app.incidents import IncidentStore
from app.procedure.models import StepResult
from test_incident_procedure import (
    PAYLOAD,
    client,
    integration,
    itsm_mcp,
    messages,
    post_incident,
    reply,
)
from test_procedure import kb_result


@pytest.mark.parametrize(
    ("state", "message", "expected"),
    [
        (None, "hello", "operations_agent"),
        ("WAITING_FOR_CONFIRMATION", "yes", "procedure_confirmation"),
        ("WAITING_FOR_CONFIRMATION", "cancel", "procedure_confirmation"),
        ("WAITING_FOR_CONFIRMATION", "/cancel", "procedure_cancel"),
        ("COLLECTING_INPUTS", "namespace is aap", "procedure_input"),
        ("RUNNING", "hello", "procedure_busy"),
        ("COMPLETED", "hello", "operations_agent"),
        ("FAILED", "hello", "operations_agent"),
        ("STOPPED", "hello", "operations_agent"),
        ("CANCELLED", "hello", "operations_agent"),
    ],
)
def test_shared_router_depends_on_procedure_state_not_session_kind(state, message, expected):
    active = None if state is None else type("Context", (), {"status": state})()
    assert web.select_session_message_route(message, active) == expected


def test_incident_without_active_procedure_uses_own_agent_session(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "incident_store", IncidentStore(tmp_path / "incidents.sqlite3"))
    monkeypatch.setattr(web, "start_incident_preparation", lambda incident: None)
    monkeypatch.setattr(web, "load_config", lambda: object())

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        observed = {}

        async def agent(message, config, session, *, input_persisted=False):
            observed.update(
                message=message,
                session_id=session.session_id,
                input_persisted=input_persisted,
                history=await session.get_items(),
            )
            started.set()
            await release.wait()
            await session.add_items([{"role": "assistant", "content": "The server is healthy."}])
            return "The server is healthy."

        monkeypatch.setattr(web, "run_agent", agent)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web.app), base_url="http://test",
        ) as http:
            incident = (await http.post("/incidents", json=PAYLOAD)).json()
            incident_session = incident["session_id"]
            normal_session = (await http.post("/api/sessions")).json()["session_id"]

            response = await http.post(
                f"/api/sessions/{incident_session}/messages",
                json={"message": "what is the current status of this server?"},
                headers={"Prefer": "respond-async"},
            )
            assert response.status_code == 202
            task = web.session_tasks[incident_session]
            await started.wait()

            # Acceptance is durable before the blocked agent completes, and
            # work remains owned by the incident while another chat is viewed.
            midway = (await http.get(f"/api/sessions/{incident_session}")).json()
            assert midway["processing"]
            assert midway["messages"][-1] == {
                "role": "user", "content": "what is the current status of this server?",
            }
            assert (await http.get(f"/api/sessions/{normal_session}")).json()["messages"] == []
            assert observed["session_id"] == incident_session
            assert observed["input_persisted"]
            assert "High CPU on web-server-prod" in observed["history"][0]["content"]

            release.set()
            await task
            final = (await http.get(f"/api/sessions/{incident_session}")).json()
            assert not final["processing"] and final["unread"]
            assert final["messages"][-2:] == [
                {"role": "user", "content": "what is the current status of this server?"},
                {"role": "assistant", "content": "The server is healthy."},
            ]
            assert (await http.get(f"/api/sessions/{normal_session}")).json()["messages"] == []

    asyncio.run(scenario())


def test_no_kb_releases_incident_chat_to_operations_agent(integration):
    rag, _extractor, executor, agent = integration
    rag.call_tool.return_value = kb_result()

    async def scenario():
        async with client() as http:
            incident = await post_incident(http)
            session_id = incident["session_id"]
            assert web.get_conversation(session_id).active_procedure is None
            assert "No matching procedure" in (await messages(http, session_id))[-1]["content"]

            await reply(http, session_id, "can you inspect the affected node manually?")
            agent.assert_awaited_once()
            assert agent.await_args.args[0] == "can you inspect the affected node manually?"
            assert agent.await_args.kwargs["session"].session_id == session_id
            executor.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal", ["cancelled", "completed", "failed"])
def test_terminal_procedure_releases_incident_chat(integration, terminal):
    _rag, _extractor, executor, agent = integration
    if terminal == "failed":
        executor.return_value = StepResult(status="FAILED", summary="Stopped safely.")

    async def scenario():
        async with client() as http:
            incident = await post_incident(http)
            session_id = incident["session_id"]
            await reply(http, session_id, "cancel" if terminal == "cancelled" else "yes")
            assert web.get_conversation(session_id).active_procedure is None
            agent.assert_not_awaited()

            await reply(http, session_id, "check the current cluster state")
            agent.assert_awaited_once()
            assert agent.await_args.args[0] == "check the current cluster state"
            assert agent.await_args.kwargs["session"].session_id == session_id

    asyncio.run(scenario())
