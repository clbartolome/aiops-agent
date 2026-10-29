"""Real shared retrieval/parser/orchestration; mock only external boundaries."""
import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from app import web
import app.procedure as pipeline
from app.incidents import IncidentStore
from app.procedure.models import StepResult
from test_procedure import itsm_mcp, kb_result
from test_procedure_inputs import KB_SINGLE_INPUT

KB = KB_SINGLE_INPUT.replace(
    '- **Namespace** — required — OpenShift namespace to inspect.',
    '- **Host** — required — Host to inspect.\n'
    '- **Node** — required — Worker node.\n'
    '- **Threshold** — required — CPU percentage.\n'
    '- **Namespace** — required — OpenShift namespace to inspect.')
PAYLOAD = {'title':'High CPU on web-server-prod',
           'message':'CPU utilization above 95% for 10 minutes on node worker-03.',
           'severity':'critical'}
VALUES = {'host':'web-server-prod', 'node':'worker-03', 'threshold':95, 'namespace':'production'}


@pytest.fixture
def integration(monkeypatch, tmp_path, config, itsm_mcp, summarizer):
    monkeypatch.setattr(web, 'incident_store', IncidentStore(tmp_path / 'incidents.sqlite3'))
    monkeypatch.setattr(web, 'load_config', lambda: config)
    agent = AsyncMock()
    monkeypatch.setattr(web, 'run_agent', agent)
    extraction = AsyncMock(return_value={**VALUES, 'undeclared_secret':'discard'})
    monkeypatch.setattr(pipeline, 'extract_procedure_inputs', extraction)
    executor = AsyncMock(return_value=StepResult(status='SUCCESS', summary='Namespace exists.'))
    monkeypatch.setattr(pipeline, 'execute_step', executor)
    itsm_mcp.call_tool.return_value = kb_result({'title':'Investigate high CPU', 'description':KB, 'score':1})
    return itsm_mcp, extraction, executor, agent


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test')


async def settle(id):
    task = web.session_tasks.get(id)
    if task:
        await asyncio.wait_for(task, 2)


async def post_incident(c, payload=None):
    response = await asyncio.wait_for(c.post('/incidents', json=payload or PAYLOAD), 1)
    assert response.status_code == 201
    incident = response.json()
    assert incident['status'] == 'IN_PROGRESS'
    await settle(incident['session_id'])
    return incident


async def messages(c, id):
    return (await c.get('/api/sessions/' + id)).json()['messages']


async def reply(c, id, text):
    response = await c.post(f'/api/sessions/{id}/messages', json={'message':text}, headers={'Prefer':'respond-async'})
    assert response.status_code == 202
    await settle(id)


def test_automatic_shared_preparation_and_mandatory_confirmation(integration):
    rag, extractor, executor, agent = integration
    async def scenario():
        async with client() as c:
            normal = (await c.post('/api/sessions')).json()['session_id']
            incident = await post_incident(c)
            id = incident['session_id']
            # No GET/open was needed to start or complete preparation.
            assert rag.calls == [('rag_search_kb', {'query':PAYLOAD['title']+'\n\n'+PAYLOAD['message']})]
            extractor.assert_not_awaited()
            context = web.get_conversation(id).active_procedure
            assert context.source == 'incident' and context.source_id == incident['id']
            assert context.inputs == {}
            assert context.status == 'WAITING_FOR_CONFIRMATION' and not context.confirmed
            assert not context.procedure.confirmation_required
            assert web.incident_store.get(incident['id']).procedure_run_id == context.run_id
            stored = await messages(c, id)
            texts = '\n'.join(m.get('content','') for m in stored)
            assert 'KB found: Investigate high CPU' in texts
            assert 'Incident parameters identified:' not in texts
            assert 'undeclared_secret' not in texts
            assert 'Do you want to run this procedure for this incident?' in texts
            assert not any(m.get('type') == 'procedure_progress' for m in stored)
            assert await messages(c, normal) == []
            state = (await c.get(f'/api/sessions/{id}/state')).json()
            assert state['unread'] and not state['processing'] and state['progress'] is None
            await c.post(f'/api/sessions/{id}/read', json={'revision':state['revision']})
            assert not (await c.get(f'/api/sessions/{id}/state')).json()['unread']
            executor.assert_not_awaited()
            agent.assert_not_awaited()
    asyncio.run(scenario())


@pytest.mark.parametrize('confirmation', ['yes', 'run it', 'proceed'])
def test_positive_confirmation_reuses_executor(integration, confirmation):
    rag, extractor, executor, agent = integration
    async def scenario():
        async with client() as c:
            incident = await post_incident(c)
            id = incident['session_id']
            executor.assert_not_awaited()
            await reply(c, id, confirmation)
            executor.assert_awaited_once()
            assert executor.await_args.args[3] == VALUES
            assert web.incident_store.get(incident['id']).status == 'DONE'
            assert web.get_conversation(id).active_procedure is None
            assert (await messages(c,id))[-1]['content'] == 'Procedure completed successfully.'
            assert len(rag.calls) == 1  # No second KB lookup on confirmation.
            agent.assert_not_awaited()
    asyncio.run(scenario())


@pytest.mark.parametrize('rejection', ['no', 'do not run it', 'cancel', '/cancel'])
def test_rejection_does_not_execute(integration, rejection):
    executor = integration[2]
    async def scenario():
        async with client() as c:
            incident = await post_incident(c)
            # API callers without Prefer still use the session-owned task.
            response = await c.post('/api/sessions/' + incident['session_id'] + '/messages', json={'message':rejection})
            assert response.status_code == 202
            await settle(incident['session_id'])
            assert (await messages(c,incident['session_id']))[-1]['content'] == 'Procedure cancelled.'
            assert web.get_conversation(incident['session_id']).active_procedure is None
            assert web.incident_store.get(incident['id']).status == 'NEW'
            executor.assert_not_awaited()
    asyncio.run(scenario())


def test_approval_then_missing_input_reply_executes(integration):
    rag, extractor, executor, agent = integration
    extractor.return_value = {key:value for key,value in VALUES.items() if key != 'namespace'}
    async def scenario():
        async with client() as c:
            incident = await post_incident(c)
            id = incident['session_id']
            assert web.get_conversation(id).active_procedure.status == 'WAITING_FOR_CONFIRMATION'
            extractor.assert_not_awaited()
            await reply(c,id,'maybe later')
            extractor.assert_not_awaited()
            await reply(c,id,'yes')
            assert web.get_conversation(id).active_procedure.status == 'COLLECTING_INPUTS'
            assert 'namespace' in (await messages(c,id))[-1]['content']
            executor.assert_not_awaited()
            await reply(c,id,'namespace is production')
            assert web.get_conversation(id).active_procedure is None
            assert executor.await_args.args[3] == VALUES
            assert len(rag.calls) == 1
            agent.assert_not_awaited()
    asyncio.run(scenario())


@pytest.mark.parametrize('result', ['empty','not_executable','parse_error','mcp_error'])
def test_preparation_failures_never_execute(integration, result):
    rag, extractor, executor, agent = integration
    if result == 'empty': rag.call_tool.return_value = kb_result()
    elif result == 'not_executable': rag.call_tool.return_value = kb_result({'title':'Plain KB','description':'Advice only','score':1})
    elif result == 'parse_error': rag.call_tool.return_value = kb_result({'title':'Broken','description':'## Procedure\nBroken','score':1})
    else: rag.call_tool.side_effect = RuntimeError('private detail')
    async def scenario():
        async with client() as c:
            incident = await post_incident(c)
            owner = web.get_conversation(incident['session_id'])
            assert not owner.processing and owner.active_procedure is None
            assert web.incident_store.get(incident['id']).status == 'NEW'
            texts = '\n'.join(m.get('content','') for m in await messages(c,incident['session_id']))
            if result in {'empty','not_executable'}: assert 'No matching procedure was found for this incident.' in texts
            assert 'private detail' not in texts
            executor.assert_not_awaited()
            agent.assert_not_awaited()
    asyncio.run(scenario())


@pytest.mark.parametrize('status', ['STOP','FAILED'])
def test_execution_failure_is_not_done(integration, status):
    integration[2].return_value = StepResult(status=status,summary='Stopped safely.')
    async def scenario():
        async with client() as c:
            incident = await post_incident(c)
            await reply(c, incident['session_id'], 'yes')
            assert web.incident_store.get(incident['id']).status == 'NEW'
            integration[2].assert_awaited_once()
    asyncio.run(scenario())


def test_multiple_incidents_prepare_without_opening(integration):
    rag, extractor, executor, agent = integration
    async def scenario():
        started = {name:asyncio.Event() for name in ('X','Y')}
        gates = {name:asyncio.Event() for name in started}
        async def retrieve(tool, arguments):
            name = arguments['query'].split('\n')[0]
            started[name].set()
            await gates[name].wait()
            return kb_result({'title':'Procedure '+name,'description':KB,'score':1})
        rag.call_tool.side_effect = retrieve
        async with client() as c:
            normal = (await c.post('/api/sessions')).json()['session_id']
            incidents = {}
            for name in started:
                response = await asyncio.wait_for(c.post('/incidents',json={**PAYLOAD,'title':name}),1)
                incidents[name] = response.json()
                await asyncio.wait_for(started[name].wait(),1)
            assert (await c.get('/api/sessions/'+normal)).json()['messages'] == []
            assert all(row['processing'] for row in (await c.get('/api/incidents')).json())
            gates['X'].set()
            await settle(incidents['X']['session_id'])
            assert web.get_conversation(incidents['Y']['session_id']).processing
            gates['Y'].set()
            await settle(incidents['Y']['session_id'])
            for name,incident in incidents.items():
                texts = '\n'.join(m.get('content','') for m in await messages(c,incident['session_id']))
                assert 'KB found: Procedure '+name in texts
                assert 'KB found: Procedure '+('Y' if name == 'X' else 'X') not in texts
                assert web.get_conversation(incident['session_id']).unread
            executor.assert_not_awaited()
    asyncio.run(scenario())



def test_execution_gate_cannot_be_bypassed(integration, config):
    executor = integration[2]
    async def scenario():
        async with client() as c:
            incident = await post_incident(c)
            owner = web.get_conversation(incident['session_id'])
            unconfirmed = owner.active_procedure.model_copy(update={"status":"READY"})
            with pytest.raises(ValueError, match="explicit human confirmation"):
                async for _ in pipeline._execute_steps(unconfirmed, config):
                    pass
            executor.assert_not_awaited()
    asyncio.run(scenario())


def test_confirmed_run_preserves_sequential_execution(integration):
    rag, extractor, executor, agent = integration
    two_steps = KB.replace('## Success', '### 2. Check CPU\n\nInspect CPU for **Host**.\n\n## Success')
    rag.call_tool.return_value = kb_result({'title':'CPU check','description':two_steps,'score':1})
    async def scenario():
        async with client() as c:
            incident = await post_incident(c)
            executor.assert_not_awaited()
            await reply(c,incident['session_id'],'yes')
            assert executor.await_count == 2
            first, second = executor.await_args_list
            assert first.kwargs['previous_results'] == []
            assert len(second.kwargs['previous_results']) == 1
            assert second.kwargs['previous_results'][0].step.id == first.args[1].id
            assert web.incident_store.get(incident['id']).status == 'DONE'
    asyncio.run(scenario())
