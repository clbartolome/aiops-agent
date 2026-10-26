"""Task ownership and HTTP acceptance, with controlled asynchronous boundaries."""
import asyncio

import httpx

from app import web
from app.sessions import SessionStore


def test_background_ownership_concurrency_and_read(monkeypatch):
    async def scenario():
        gates = {text: asyncio.Event() for text in ('A', 'B')}
        started = {text: asyncio.Event() for text in gates}

        async def agent(message, config, session):
            await session.add_items([{'role': 'user', 'content': message}])
            started[message].set()
            await gates[message].wait()
            await session.add_items([{'role': 'assistant', 'content': 'Answer ' + message}])
            return 'Answer ' + message

        monkeypatch.setattr(web, 'run_agent', agent)
        monkeypatch.setattr(web, 'load_config', lambda: object())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            a, b = [(await client.post('/api/sessions')).json()['session_id'] for _ in range(2)]
            async def send(id, text):
                return await client.post(f'/api/sessions/{id}/messages', json={'message': text}, headers={'Prefer':'respond-async'})
            assert (await send(a, 'A')).status_code == 202
            task_a = web.session_tasks[a]
            await started['A'].wait()
            # Viewing A or B must not wait for A's run lock.
            for id in (a, b):
                assert (await asyncio.wait_for(client.get(f'/api/sessions/{id}'), 1)).status_code == 200
            assert (await send(a, 'A')).status_code == 409
            assert (await send(b, 'B')).status_code == 202
            task_b = web.session_tasks[b]
            await started['B'].wait()
            listed = (await client.get('/api/sessions')).json()
            assert all(row['processing'] for row in listed)
            gates['A'].set()
            await task_a
            assert not task_b.done()
            state = (await client.get(f'/api/sessions/{a}/state')).json()
            assert not state['processing'] and state['unread'] and state['progress'] is None
            history_a = (await client.get(f'/api/sessions/{a}')).json()['messages']
            history_b = (await client.get(f'/api/sessions/{b}')).json()['messages']
            assert [m['content'] for m in history_a] == ['A', 'Answer A']
            assert [m['content'] for m in history_b] == ['B']
            # A stale view cannot clear a newer notification.
            assert (await client.post(f'/api/sessions/{a}/read', json={'revision': 0})).json()['unread']
            assert not (await client.post(f'/api/sessions/{a}/read', json={'revision': state['revision']})).json()['unread']
            gates['B'].set()
            await task_b
            assert [m['content'] for m in (await client.get(f'/api/sessions/{b}')).json()['messages']] == ['B', 'Answer B']
    asyncio.run(scenario())


def test_incident_work_progress_error_and_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(web, 'start_incident_preparation', lambda incident: None)
    from app.incidents import IncidentStore
    monkeypatch.setattr(web, 'incident_store', IncidentStore(tmp_path / 'incidents.sqlite3'))

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        async def work(owner):
            owner.progress = 'Running step 2'
            started.set()
            await release.wait()
            raise RuntimeError('private provider detail')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            incident = (await client.post('/incidents', json={'title':'CPU', 'message':'Hot', 'severity':'critical'})).json()
            id = incident['session_id']
            normal = (await client.post('/api/sessions')).json()['session_id']
            task = web.start_session_work(web.get_conversation(id), work=work)
            await started.wait()
            assert (await client.get(f'/api/sessions/{id}')).json()['progress'] == 'Running step 2'
            assert (await client.get(f'/api/sessions/{normal}')).json()['progress'] is None
            listing = (await asyncio.wait_for(client.get('/api/incidents'), 1)).json()[0]
            assert listing['processing'] and listing['session_id'] == id
            assert (await client.post(f'/api/sessions/{normal}/messages', json={'message':' '}, headers={'Prefer':'respond-async'})).status_code == 422
            release.set()
            await task
            listing = (await client.get('/api/incidents')).json()[0]
            assert listing['unread'] and not listing['processing']
            history = (await client.get(f'/api/sessions/{id}')).json()['messages']
            assert len(history) == 2
            assert history[-1]['content'] == 'The request could not be completed.'
            assert (await client.get(f'/api/sessions/{normal}')).json()['messages'] == []
            # Durable unread/messages, but never stale runtime processing/progress.
            for conversation in web.conversations.values():
                conversation.session.close()
            web.conversations.clear()
            monkeypatch.setattr(web, 'session_store', SessionStore(web.session_store.path))
            restored = (await client.get(f'/api/sessions/{id}')).json()
            assert restored['unread'] and not restored['processing'] and restored['progress'] is None
            assert restored['messages'] == history
    asyncio.run(scenario())


def test_background_procedure_persists_each_message(monkeypatch):
    from app.procedure.models import ProcedureStatus
    async def scenario():
        produced, release = asyncio.Event(), asyncio.Event()
        async def procedure(*args):
            yield {'type':'procedure_message', 'message':'KB found: test'}
            yield {'type':'procedure_progress', 'message':'Running step 2'}
            produced.set()
            await release.wait()
            yield {'type':'procedure_message', 'message':'Finished'}
            yield ProcedureStatus(state='completed', message='Finished')
        monkeypatch.setattr(web, 'load_config', lambda: object())
        monkeypatch.setattr(web, 'handle_procedure', procedure)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            id = (await client.post('/api/sessions')).json()['session_id']
            assert (await client.post(f'/api/sessions/{id}/messages', json={'message':'/procedure test'}, headers={'Prefer':'respond-async'})).status_code == 202
            task = web.session_tasks[id]
            await produced.wait()
            midway = (await asyncio.wait_for(client.get(f'/api/sessions/{id}'), 1)).json()
            assert midway['messages'][-1]['content'] == 'KB found: test'
            assert midway['progress'] == 'Running step 2'
            assert midway['unread']
            release.set()
            await task
            final = (await client.get(f'/api/sessions/{id}')).json()
            assert [m['content'] for m in final['messages']] == ['/procedure test','KB found: test','Finished']
            assert not final['processing'] and final['progress'] is None
    asyncio.run(scenario())
