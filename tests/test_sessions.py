import asyncio
from uuid import UUID

import httpx
import pytest

from app import web
from app.agent import NO_LIVE_DATA
from test_agent import model, answer, call, exposed_name


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
            assert (await client.get('/api/sessions')).json() == [a, b]
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
            # UUIDs/timestamps can contain 999; check stored messages for leaked content.
            messages = str(history.json()['messages'])
            assert '999' not in messages and 'secret-namespace' not in messages
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
