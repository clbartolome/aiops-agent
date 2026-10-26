"""Task ownership and HTTP acceptance, with controlled asynchronous boundaries."""
import asyncio

import httpx

from app import web
from app.sessions import SessionStore


def test_background_ownership_concurrency_and_read(monkeypatch):
    async def scenario():
        gates = {text: asyncio.Event() for text in ('A', 'B')}
        started = {text: asyncio.Event() for text in gates}

        async def agent(message, config, session, *, input_persisted=False):
            assert input_persisted
            # The endpoint has committed input before entering the agent.
            assert (await session.get_items())[-1] == {"role": "user", "content": message}
            # No second SDK input write is needed.
            if not input_persisted:
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


def test_user_message_survives_background_failure(monkeypatch):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def failing_agent(message, config, session, *, input_persisted=False):
            assert input_persisted
            started.set()
            await release.wait()
            raise RuntimeError("private provider failure")

        monkeypatch.setattr(web, "run_agent", failing_agent)
        monkeypatch.setattr(web, "load_config", lambda: object())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web.app), base_url="http://test",
        ) as client:
            session_id = (await client.post("/api/sessions")).json()["session_id"]
            response = await client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "what about the deployments?"},
                headers={"Prefer": "respond-async"},
            )
            assert response.status_code == 202
            task = web.session_tasks[session_id]
            await started.wait()

            # The accepted user row is durable and visible before agent work
            # finishes, including after repeated history refreshes.
            for _ in range(2):
                state = (await client.get(f"/api/sessions/{session_id}")).json()
                assert state["processing"]
                assert state["messages"] == [
                    {"role": "user", "content": "what about the deployments?"},
                ]

            release.set()
            await task
            state = (await client.get(f"/api/sessions/{session_id}")).json()
            assert not state["processing"]
            assert state["messages"] == [
                {"role": "user", "content": "what about the deployments?"},
                {"role": "assistant", "content": "The request could not be completed."},
            ]

    asyncio.run(scenario())


def test_procedure_reply_is_persisted_before_input_processing(monkeypatch):
    import app.procedure as pipeline
    from test_procedure_inputs import KB_VM_INPUTS, make_context

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked_extractor(message, inputs, config):
            started.set()
            await release.wait()
            return {"virtual_machine_name": "server-01"}

        monkeypatch.setattr(pipeline, "extract_procedure_inputs", blocked_extractor)
        monkeypatch.setattr(web, "load_config", lambda: object())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web.app), base_url="http://test",
        ) as client:
            session_id = (await client.post("/api/sessions")).json()["session_id"]
            owner = web.get_conversation(session_id)
            owner.active_procedure = make_context(KB_VM_INPUTS)
            response = await client.post(
                f"/api/sessions/{session_id}/messages",
                json={"message": "virtual machine is server-01"},
                headers={"Prefer": "respond-async"},
            )
            assert response.status_code == 202
            task = web.session_tasks[session_id]
            await started.wait()
            midway = (await client.get(f"/api/sessions/{session_id}")).json()
            assert midway["processing"]
            assert midway["messages"][-1] == {
                "role": "user", "content": "virtual machine is server-01",
            }
            release.set()
            await task
            final = (await client.get(f"/api/sessions/{session_id}")).json()
            assert [item["content"] for item in final["messages"]].count(
                "virtual machine is server-01",
            ) == 1

    asyncio.run(scenario())


def test_incident_approval_is_persisted_before_procedure_resumes(monkeypatch):
    import app.procedure as pipeline
    from test_procedure_inputs import KB_VM_INPUTS, make_context

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked_extractor(message, inputs, config):
            started.set()
            await release.wait()
            return {}

        monkeypatch.setattr(pipeline, "extract_procedure_inputs", blocked_extractor)
        monkeypatch.setattr(web, "load_config", lambda: object())
        owner = web.session_store.create(kind="incident", title="CPU incident")
        web.conversations[owner.session_id] = owner
        owner.active_procedure = make_context(
            KB_VM_INPUTS, status="WAITING_FOR_CONFIRMATION",
        ).model_copy(update={
            "confirmed": False,
            "source": "incident",
            "source_id": "incident-1",
            "original_request": "High CPU on server-01",
        })

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web.app), base_url="http://test",
        ) as client:
            response = await client.post(
                f"/api/sessions/{owner.session_id}/procedure-confirmation",
                json={"run_id": owner.active_procedure.run_id, "action": "approve"},
            )
            assert response.status_code == 202
            task = web.session_tasks[owner.session_id]
            await started.wait()
            midway = (await client.get(f"/api/sessions/{owner.session_id}")).json()
            assert midway["processing"]
            assert midway["messages"][-1] == {"role": "user", "content": "yes"}
            release.set()
            await task
            final = (await client.get(f"/api/sessions/{owner.session_id}")).json()
            assert [item["content"] for item in final["messages"]].count("yes") == 1

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
