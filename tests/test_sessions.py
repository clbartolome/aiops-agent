import asyncio
from unittest.mock import Mock
from uuid import UUID

import httpx
import pytest

from app import web
from app.agent import NO_LIVE_DATA
from test_agent import model, answer, call, exposed_name
from test_web import make_procedure_mock


@pytest.fixture(autouse=True)
def sessions(monkeypatch, config):
    monkeypatch.setattr(web, "conversations", {})
    monkeypatch.setattr(web, "load_config", lambda: config)
    yield
    for conversation in web.conversations.values():
        conversation.session.close()


def test_creation_listing_and_unknown_session():
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            first = await client.post('/api/sessions')
            second = await client.post('/api/sessions')
            assert first.status_code == second.status_code == 201
            a, b = first.json(), second.json()
            assert UUID(a['session_id']) != UUID(b['session_id'])
            assert set(a) == {'session_id', 'title', 'created_at', 'updated_at'}
            listed = (await client.get('/api/sessions')).json()
            assert [{key: item[key] for key in a} for item in listed] == [a, b]
            assert all(not item['processing'] and not item['unread'] for item in listed)
            assert (await client.get('/api/sessions/' + a['session_id'])).json()['messages'] == []
            assert (await client.post('/api/sessions/missing/messages', json={'message': 'hello'})).status_code == 404
            assert (await client.get('/api/sessions/missing')).status_code == 404
            for message in ('', '   ', None):
                assert (await client.post('/api/sessions/' + a['session_id'] + '/messages', json={'message': message})).status_code == 422
    asyncio.run(scenario())


def test_multiturn_mcp_and_switching(model, mcp_boundary, monkeypatch):
    original = web.run_agent
    used_sessions = []

    async def tracked(message, config, session):
        used_sessions.append(session)
        return await original(message, config, session=session)

    monkeypatch.setattr(web, 'run_agent', tracked)

    def respond(**kwargs):
        inputs = kwargs['input']
        if model.await_count == 1:
            return call('request_user_input', {'question': 'The namespace name is needed.'})
        if model.await_count == 2:
            assert 'How many pods are in the namespace?' in repr(inputs)
            assert 'The namespace name is needed.' in repr(inputs)
            assert 'openshift-ingress' in repr(inputs)
            return call(exposed_name(kwargs['tools'], 'openshift', 'get_pod_count'), {'namespace': 'openshift-ingress'})
        if model.await_count == 3:
            assert inputs[-1]['type'] == 'function_call_output'
            return answer('There are 3 pods in openshift-ingress.')
        assert 'openshift-ingress' not in repr(inputs)
        assert 'How many pods' not in repr(inputs)
        return answer('Hello.')

    model.side_effect = respond

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            a = (await client.post('/api/sessions')).json()['session_id']
            b = (await client.post('/api/sessions')).json()['session_id']
            path_a, path_b = '/api/sessions/' + a, '/api/sessions/' + b
            first = await client.post(path_a + '/messages', json={'message': 'How many pods are in the namespace?'})
            assert first.status_code == 200 and first.json()['response'] != NO_LIVE_DATA
            for mcp_session in mcp_boundary.sessions.values():
                mcp_session.call_tool.assert_not_awaited()
            assert (await client.get(path_a)).json()['messages'][-1]['content'] == first.json()['response']
            second = await client.post(path_a + '/messages', json={'message': 'openshift-ingress'})
            assert second.status_code == 200 and second.json()['response'] != NO_LIVE_DATA
            mcp_boundary.sessions['openshift'].call_tool.assert_awaited_once_with('get_pod_count', {'namespace': 'openshift-ingress'})
            assert (await client.get(path_b)).json()['messages'] == []
            await client.post(path_b + '/messages', json={'message': 'hello'})
            a_history = (await client.get(path_a)).json()['messages']
            b_history = (await client.get(path_b)).json()['messages']
            assert [item['role'] for item in a_history] == ['user', 'assistant', 'user', 'assistant']
            assert [item['content'] for item in b_history] == ['hello', 'Hello.']
            assert (await client.get(path_a)).json()['messages'] == a_history
            assert used_sessions[0] is used_sessions[1]
            assert used_sessions[0] is not used_sessions[2]
            assert used_sessions[0].session_id == a
    asyncio.run(scenario())


def test_rejected_answer_does_not_leak_into_history(model):
    model.return_value = answer('There are 999 pods in secret-namespace.')

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id
            result = await client.post(path + '/messages', json={'message': 'Count pods in production'})
            assert result.json()['response'] == NO_LIVE_DATA
            history = await client.get(path)
            assert '999' not in history.text and 'secret-namespace' not in history.text
            assert history.json()['messages'][-1]['content'] == NO_LIVE_DATA
    asyncio.run(scenario())


def test_same_session_requests_are_serialized(monkeypatch):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        active = 0
        calls = 0

        async def run(message, config, session):
            nonlocal active, calls
            active += 1
            calls += 1
            assert active == 1
            started.set()
            await release.wait()
            active -= 1
            return 'Done'

        monkeypatch.setattr(web, 'run_agent', run)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id + '/messages'
            first = asyncio.create_task(client.post(path, json={'message': 'one'}))
            await started.wait()
            second = asyncio.create_task(client.post(path, json={'message': 'two'}))
            await asyncio.sleep(0)
            assert calls == 1
            release.set()
            assert all(response.status_code == 200 for response in await asyncio.gather(first, second))
            assert calls == 2
    asyncio.run(scenario())


def test_session_failure_is_safe(monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError('Authorization: Bearer test-secret')
    monkeypatch.setattr(web, 'run_agent', fail)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            response = await client.post('/api/sessions/' + session_id + '/messages', json={'message': 'hello'})
            assert response.status_code == 502 and 'test-secret' not in response.text
    asyncio.run(scenario())


# --- Procedure message persistence/restoration ------------------------------
# `app.procedure`'s event generators are mocked (as in test_web.py); these
# tests exercise the real `/api/sessions` storage/restoration path to prove
# procedure output survives exactly like a normal chat turn already does.

def test_procedure_messages_are_persisted_and_restored(monkeypatch):
    from app.procedure.models import ProcedureStatus

    mock = make_procedure_mock()
    mock.events = (
        {"type": "procedure_progress", "message": "Searching KB"},
        {"type": "procedure_progress_clear"},
        {"type": "procedure_message", "message": "KB found: Inspect namespace health"},
        {"type": "procedure_progress", "message": "Parsing procedure"},
        {"type": "procedure_progress_clear"},
        {"type": "procedure_message", "message": "Procedure loaded: 1 step."},
        {"type": "procedure_message", "message": "What namespace should I use?"},
    )
    mock.return_value = ProcedureStatus(state="collecting_inputs", message="What namespace should I use?")
    monkeypatch.setattr(web, "handle_procedure", mock)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            session_id = (await client.post("/api/sessions")).json()["session_id"]
            path = "/api/sessions/" + session_id
            await client.post(path + "/messages", json={"message": "/procedure inspect namespace health"})

            # Fetched twice -- simulating a page reload re-fetching the same
            # session -- returns the exact same, already-ordered history.
            first_fetch = (await client.get(path)).json()["messages"]
            second_fetch = (await client.get(path)).json()["messages"]
            assert first_fetch == second_fetch
            assert first_fetch == [
                {"role": "user", "content": "/procedure inspect namespace health"},
                {"role": "assistant", "content": "KB found: Inspect namespace health"},
                {"role": "assistant", "content": "Procedure loaded: 1 step."},
                {"role": "assistant", "content": "What namespace should I use?"},
            ]
    asyncio.run(scenario())


def test_transient_procedure_progress_is_never_persisted(monkeypatch):
    from app.procedure.models import ProcedureStatus

    mock = make_procedure_mock()
    mock.events = (
        {"type": "procedure_progress", "message": "Running step 1: Verify that the namespace exists"},
        {"type": "procedure_progress_clear"},
        {"type": "procedure_message", "message": "Namespace payments exists."},
        {"type": "procedure_progress", "message": "Generating procedure summary"},
        {"type": "procedure_progress_clear"},
        {"type": "procedure_message", "message": "Procedure completed successfully."},
    )
    mock.return_value = ProcedureStatus(state="completed", message="Procedure completed successfully.")
    monkeypatch.setattr(web, "handle_procedure", mock)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            session_id = (await client.post("/api/sessions")).json()["session_id"]
            path = "/api/sessions/" + session_id
            await client.post(path + "/messages", json={"message": "/procedure inspect namespace health"})

            history = (await client.get(path)).json()["messages"]
            contents = [item["content"] for item in history]
            assert "Running step 1: Verify that the namespace exists" not in contents
            assert "Generating procedure summary" not in contents
            assert {"role": "assistant", "content": "Namespace payments exists."} in history
            assert {"role": "assistant", "content": "Procedure completed successfully."} in history
    asyncio.run(scenario())


def test_procedure_messages_are_isolated_between_sessions(monkeypatch):
    from app.procedure.models import ProcedureStatus

    mock = make_procedure_mock()
    monkeypatch.setattr(web, "handle_procedure", mock)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            a = (await client.post("/api/sessions")).json()["session_id"]
            b = (await client.post("/api/sessions")).json()["session_id"]
            path_a, path_b = "/api/sessions/" + a, "/api/sessions/" + b

            mock.events = ({"type": "procedure_message", "message": "KB found: Inspect namespace health"},)
            mock.return_value = ProcedureStatus(state="collecting_inputs", message="What namespace?")
            await client.post(path_a + "/messages", json={"message": "/procedure inspect namespace health"})

            mock.events = ({"type": "procedure_message", "message": "KB found: Restart workflow"},)
            mock.return_value = ProcedureStatus(state="collecting_inputs", message="What workflow?")
            await client.post(path_b + "/messages", json={"message": "/procedure restart workflow"})

            history_a = [item["content"] for item in (await client.get(path_a)).json()["messages"]]
            history_b = [item["content"] for item in (await client.get(path_b)).json()["messages"]]
            assert "KB found: Inspect namespace health" in history_a
            assert "KB found: Restart workflow" not in history_a
            assert "KB found: Restart workflow" in history_b
            assert "KB found: Inspect namespace health" not in history_b
    asyncio.run(scenario())


def test_cancel_confirmation_is_persisted(monkeypatch):
    from app.procedure.models import ProcedureContext, ProcedureDefinition, ProcedureStatus

    start = make_procedure_mock()
    procedure = ProcedureDefinition(
        id="inspect-namespace-health", version=1, title="Inspect namespace health",
        risk="low", confirmation_required=False,
        inputs=[], steps=[{"id": "step-1", "title": "Step 1", "instruction": "Do it."}],
    )
    waiting_context = ProcedureContext(
        run_id="run-1", original_request="inspect namespace health", kb_title="Inspect namespace health",
        kb_content="# Inspect namespace health\n", procedure=procedure, inputs={}, status="COLLECTING_INPUTS",
    )
    start.return_value = ProcedureStatus(
        state="collecting_inputs", message="What namespace should I use?", context=waiting_context,
    )
    monkeypatch.setattr(web, "handle_procedure", start)
    cancel = Mock(return_value=ProcedureStatus(state="cancelled", message="Procedure cancelled.", context=None))
    monkeypatch.setattr(web, "handle_cancel", cancel)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            session_id = (await client.post("/api/sessions")).json()["session_id"]
            path = "/api/sessions/" + session_id
            await client.post(path + "/messages", json={"message": "/procedure inspect namespace health"})
            await client.post(path + "/messages", json={"message": "/cancel"})

            history = (await client.get(path)).json()["messages"]
            assert {"role": "user", "content": "/cancel"} in history
            assert {"role": "assistant", "content": "Procedure cancelled."} in history
    asyncio.run(scenario())


@pytest.mark.parametrize('outcome', ['success', 'failure', 'skipped', 'protocol_error'])
def test_live_tool_retry_is_internal_and_preserves_history(model, config, mcp_boundary, outcome):
    from app.agent import LIVE_TOOL_RETRY_INSTRUCTIONS, INVALID_MODEL_OUTPUT, run_agent
    from app.diagnostics import ProtocolTextError
    from app.sessions import visible_messages
    from agents import SQLiteSession

    initial_answer = 'UNVERIFIED initial answer'
    final_answer = 'There are 3 pods.'

    def respond(**kwargs):
        if model.await_count == 1:
            return answer(initial_answer)
        assert LIVE_TOOL_RETRY_INSTRUCTIONS in kwargs['system_instructions']
        assert initial_answer not in repr(kwargs['input'])
        assert 'Earlier general discussion' in repr(kwargs['input'])
        if outcome == 'protocol_error':
            raise ProtocolTextError('invalid protocol')
        if model.await_count == 2 and outcome != 'skipped':
            return call(exposed_name(kwargs['tools'], 'openshift', 'get_pod_count'),
                        {'namespace': 'openshift-mcp'})
        return answer(final_answer)

    model.side_effect = respond
    if outcome == 'failure':
        mcp_boundary.tool_results['openshift'] = RuntimeError('Query failed')

    async def scenario():
        session = SQLiteSession('retry-history')
        try:
            await session.add_items([{'role': 'user', 'content': 'Earlier general discussion'},
                                     {'role': 'assistant', 'content': 'Earlier answer'}])
            message = 'how many pods are in openshift-mcp namespace?'
            result = await run_agent(message, config, session=session)
            expected = (final_answer if outcome == 'success' else
                        INVALID_MODEL_OUTPUT if outcome == 'protocol_error' else NO_LIVE_DATA)
            assert result == expected
            history = await visible_messages(session)
            assert history == [
                {'role': 'user', 'content': 'Earlier general discussion'},
                {'role': 'assistant', 'content': 'Earlier answer'},
                {'role': 'user', 'content': message},
                {'role': 'assistant', 'content': expected},
            ]
            assert LIVE_TOOL_RETRY_INSTRUCTIONS not in repr(await session.get_items())
            assert initial_answer not in repr(await session.get_items())
        finally:
            session.close()

    asyncio.run(scenario())
