import asyncio
import sqlite3

import httpx
import pytest

from app import web
from app.incidents import IncidentStore
from test_web import request


@pytest.fixture(autouse=True)
def incidents(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'start_incident_preparation', lambda incident: None)
    monkeypatch.setattr(web, 'incident_store', IncidentStore(tmp_path / 'incidents.sqlite3'))


def assert_no_session_rows(id):
    assert id not in web.conversations
    assert web.session_store.get(id) is None
    with sqlite3.connect(web.session_store.path) as db:
        for table in ('agent_messages', 'agent_sessions', 'conversation_metadata'):
            assert db.execute(f'SELECT count(*) FROM {table} WHERE session_id=?', (id,)).fetchone()[0] == 0


def test_delete_conversation_and_messages():
    id = request('POST', '/api/sessions').json()['session_id']
    owner = web.get_conversation(id)
    asyncio.run(owner.session.add_items([{'role':'user','content':'Hello'}, {'role':'assistant','content':'Hi'}]))
    owner.unread = True
    web.session_store.save(owner)
    assert request('DELETE', '/api/sessions/' + id).status_code == 204
    assert_no_session_rows(id)
    assert request('GET', '/api/sessions/' + id).status_code == 404
    assert request('GET', '/api/sessions').json() == []
    assert request('DELETE', '/api/sessions/' + id).status_code == 404


def test_delete_incident_owns_session_cleanup():
    incident = request('POST', '/incidents', json={'title':'CPU','message':'High','severity':'critical'}).json()
    id = incident['session_id']
    assert request('DELETE', '/api/sessions/' + id).status_code == 409
    assert web.incident_store.get(incident['id']) is not None
    asyncio.run(web.get_conversation(id).session.add_items([{'role':'user','content':'Follow up'}]))
    assert request('DELETE', '/api/incidents/' + incident['id']).status_code == 204
    assert web.incident_store.get(incident['id']) is None
    assert_no_session_rows(id)
    assert request('GET', '/api/incidents').json() == []
    assert request('GET', '/api/sessions/' + id).status_code == 404
    assert request('DELETE', '/api/incidents/' + incident['id']).status_code == 404


@pytest.mark.parametrize('kind', ['chat', 'incident'])
def test_reject_delete_during_work(kind):
    async def scenario():
        gate = asyncio.Event()
        async def work(owner):
            await gate.wait()
            await web._persist_message(owner, 'assistant', 'Completed')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            if kind == 'incident':
                incident = (await client.post('/incidents', json={'title':'CPU','message':'High','severity':'info'})).json()
                id = incident['session_id']
                path = '/api/incidents/' + incident['id']
            else:
                id = (await client.post('/api/sessions')).json()['session_id']
                path = '/api/sessions/' + id
            task = web.start_session_work(web.get_conversation(id), work=work)
            assert (await asyncio.wait_for(client.delete(path), 1)).status_code == 409
            gate.set()
            await task
            assert (await client.get('/api/sessions/' + id)).json()['messages'][-1]['content'] == 'Completed'
            assert (await client.delete(path)).status_code == 204
            assert_no_session_rows(id)
    asyncio.run(scenario())


def test_sidebar_styles():
    page = request('GET', '/').text
    assert 'minmax(280px, 300px) minmax(0, 1fr) minmax(300px, 320px)' in page
    assert '.context-select' in page and 'font-size: 13px' in page
    assert '-webkit-line-clamp: 2' in page
    assert '.unread-indicator { color: #facc15' in page
    assert 'animation: sidebar-spin .85s linear infinite' in page
    assert '@media (max-width: 1200px)' in page
    assert '@media (max-width: 700px)' in page
    assert 'window.confirm(`Delete this ${kind}?`)' in page


def test_delete_empty_session():
    id = request('POST', '/api/sessions').json()['session_id']
    assert request('DELETE', '/api/sessions/' + id).status_code == 204
    assert_no_session_rows(id)


def test_new_work_rejected_during_cleanup(monkeypatch):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            id = (await client.post('/api/sessions')).json()['session_id']
            owner = web.get_conversation(id)
            original = owner.session.clear_session
            async def paused_clear():
                started.set()
                await release.wait()
                await original()
            monkeypatch.setattr(owner.session, 'clear_session', paused_clear)
            deleting = asyncio.create_task(client.delete('/api/sessions/' + id))
            await started.wait()
            response = await client.post('/api/sessions/' + id + '/messages',
                                         headers={'Prefer':'respond-async'}, json={'message':'hello'})
            assert response.status_code == 409
            assert not owner.processing
            release.set()
            assert (await deleting).status_code == 204
            assert_no_session_rows(id)
    asyncio.run(scenario())
