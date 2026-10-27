from unittest.mock import AsyncMock, Mock

import pytest

from app import web
from app.incidents import IncidentStore
from test_web import request


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    # Storage/view tests isolate intake scheduling; integration tests exercise it.
    monkeypatch.setattr(web, 'start_incident_preparation', lambda incident: None)
    storage = IncidentStore(tmp_path / 'incidents.sqlite3')
    monkeypatch.setattr(web, 'incident_store', storage)
    return storage


def create(**overrides):
    return request('POST', '/incidents', json={
        'title': 'High CPU', 'message': 'CPU above threshold',
        'severity': 'critical', **overrides})


def test_create_and_persistence(store):
    response = create()
    assert response.status_code == 201
    data = response.json()
    assert data['status'] == 'NEW'
    assert IncidentStore(store.path).get(data['id']).message == 'CPU above threshold'


@pytest.mark.parametrize('severity', ['critical', 'warning', 'info', 'CRITICAL'])
def test_valid_severity(severity):
    assert create(severity=severity).json()['severity'] == severity.lower()


@pytest.mark.parametrize('fields', [
    {'severity': 'high'}, {'severity': 'low'}, {'severity': 'error'},
    {'title': ''}, {'title': '  '}, {'message': ''}, {'message': '  '},
    {'title': None}, {'message': None}, {'severity': None},
])
def test_invalid(fields):
    assert create(**fields).status_code == 422


@pytest.mark.parametrize('field', ['title', 'message', 'severity'])
def test_required(field):
    payload = {'title': 'CPU', 'message': 'Hot', 'severity': 'info'}
    del payload[field]
    assert request('POST', '/incidents', json=payload).status_code == 422


def test_list_detail_and_no_automation(monkeypatch):
    agent = AsyncMock()
    procedure = Mock()
    monkeypatch.setattr(web, 'run_agent', agent)
    monkeypatch.setattr(web, 'handle_procedure', procedure)
    first = create().json()
    second = create(title='Second').json()
    listing = request('GET', '/api/incidents').json()
    assert [item['id'] for item in listing] == [second['id'], first['id']]
    assert 'message' not in listing[0]
    assert request('GET', '/api/incidents/' + first['id']).json() == first
    assert request('GET', '/api/incidents/unknown').status_code == 404
    agent.assert_not_called()
    procedure.assert_not_called()
    assert web.incident_store.get(first['id']).status == 'NEW'


def test_inbox_ui():
    page = request('GET', '/').text
    assert 'aria-label="Incidents"' in page
    for severity in ('critical', 'warning', 'info'):
        assert f'.severity-{severity}' in page
    assert "setTimeout(pollContexts, 1000)" in page
    assert "entry.addEventListener('click', () => openIncident(incident.id))" in page
    opening = page.split('async function openIncident(id)')[1].split('async function refreshIncidents')[0]
    assert 'await selectContext' in opening
    assert '/messages' not in opening
    assert 'procedure' not in opening
    assert 'restoreIncidents' not in page
    assert 'localStorage' not in page


def test_inbox_javascript_behavior():
    import shutil
    import subprocess
    if not shutil.which('node'):
        pytest.skip('Node is required for the lightweight inbox DOM test')
    subprocess.run(['node', 'tests/incident_ui.cjs'], check=True)


def test_owned_session_history_and_restart(store, monkeypatch):
    import asyncio
    from app.sessions import SessionStore
    async def answer(message, config, session, *, input_persisted=False):
        assert input_persisted
        await session.add_items([{'role': 'assistant', 'content': 'Reply to ' + message}])
        return 'Reply to ' + message

    agent = AsyncMock(side_effect=answer)
    procedure = Mock()
    monkeypatch.setattr(web, 'run_agent', agent)
    monkeypatch.setattr(web, 'handle_procedure', procedure)
    monkeypatch.setattr(web, 'load_config', lambda: object())
    a = request('POST', '/api/sessions').json()['session_id']
    asyncio.run(web.get_conversation(a).session.add_items([{'role': 'user', 'content': 'Normal history'}]))
    incidents = [create(title=title).json() for title in ('X', 'Y')]
    assert len({a, *(i['session_id'] for i in incidents)}) == 3
    assert [s['session_id'] for s in request('GET', '/api/sessions').json()] == [a]
    for incident in incidents:
        path = '/api/sessions/' + incident['session_id']
        initial = request('GET', path).json()
        assert initial['kind'] == 'incident'
        assert initial['messages'] == [{'type': 'incident', 'incident_id': incident['id'], 'incident': incident}]
        for text in ('hello from ' + incident['title'], 'follow up from ' + incident['title']):
            assert request('POST', path + '/messages', json={'message': text}).json() == {
                'response': 'Reply to ' + text,
            }
        assert web.get_conversation(incident['session_id']).active_procedure is None
    # Recreate the persistence layer/cache as on process restart.
    for conversation in web.conversations.values():
        conversation.session.close()
    web.conversations.clear()
    monkeypatch.setattr(web, 'session_store', SessionStore(web.session_store.path))
    monkeypatch.setattr(web, 'incident_store', IncidentStore(store.path))
    assert request('GET', '/api/sessions').json()[0]['session_id'] == a
    assert request('GET', '/api/sessions/' + a).json()['messages'] == [{'role': 'user', 'content': 'Normal history'}]
    for incident in incidents:
        reopened = request('GET', '/api/sessions/' + incident['session_id']).json()['messages']
        assert len(reopened) == 5
        assert reopened[0]['incident'] == incident
        assert reopened[1]['content'] == 'hello from ' + incident['title']
        assert reopened[2]['content'] == 'Reply to hello from ' + incident['title']
        assert reopened[3]['content'] == 'follow up from ' + incident['title']
        assert reopened[4]['content'] == 'Reply to follow up from ' + incident['title']
        assert not any('progress' in m.get('type', '') for m in reopened)
        assert web.incident_store.get(incident['id']).status == 'NEW'
    assert agent.await_count == 4
    procedure.assert_not_called()


def test_legacy_incident_migration(tmp_path, monkeypatch):
    import sqlite3
    path = tmp_path / 'legacy.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE incidents (id TEXT PRIMARY KEY,title TEXT,message TEXT,severity TEXT,status TEXT,created_at TEXT)')
        db.execute("INSERT INTO incidents VALUES ('old','Old alert','Body','info','NEW','2026-01-01T00:00:00Z')")
    monkeypatch.setattr(web, 'incident_store', IncidentStore(path))
    incident = request('GET', '/api/incidents/old').json()
    assert incident['session_id'] == 'incident-old'
    history = request('GET', '/api/sessions/incident-old').json()['messages']
    assert len(history) == 1
    request('GET', '/api/incidents/old')
    assert request('GET', '/api/sessions/incident-old').json()['messages'] == history
