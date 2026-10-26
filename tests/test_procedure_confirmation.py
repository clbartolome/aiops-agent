"""Approval happens before extraction for both sources, including UI actions."""
import asyncio
from unittest.mock import Mock

import pytest

from app import web
import app.procedure as pipeline
from test_incident_procedure import integration, itsm_mcp, client, settle, post_incident, messages, PAYLOAD


async def prepare(c, source):
    if source == 'incident':
        incident = await post_incident(c)
        return incident['session_id'], incident['id']
    id = (await c.post('/api/sessions')).json()['session_id']
    response = await c.post(f'/api/sessions/{id}/messages', headers={'Prefer':'respond-async'},
                            json={'message':'/procedure '+PAYLOAD['title']+'\n\n'+PAYLOAD['message']})
    assert response.status_code == 202
    await settle(id)
    return id, None


@pytest.mark.parametrize('source', ['user', 'incident'])
@pytest.mark.parametrize('action', ['approve', 'cancel'])
def test_proposal_and_button_action(integration, monkeypatch, source, action):
    rag, extractor, executor, agent = integration
    parser = Mock(wraps=pipeline.parse_procedure)
    monkeypatch.setattr(pipeline, 'parse_procedure', parser)
    async def scenario():
        async with client() as c:
            id, incident_id = await prepare(c, source)
            owner = web.get_conversation(id)
            context = owner.active_procedure
            assert context.status == 'WAITING_FOR_CONFIRMATION'
            assert not context.confirmed and context.inputs == {}
            assert not owner.processing and owner.progress is None
            extractor.assert_not_awaited()
            executor.assert_not_awaited()
            agent.assert_not_awaited()
            assert [name for name, _ in rag.calls] == ['rag_search_kb']
            first = await messages(c,id)
            assert first[-1]['content'].startswith('Procedure found: ')
            assert 'Steps: 1\nRisk: low' in first[-1]['content']
            assert 'Incident parameters identified' not in first[-1]['content']
            for _ in range(3):
                snapshot = (await c.get(f'/api/sessions/{id}')).json()
                assert snapshot['confirmation'] == {'run_id':context.run_id}
                assert snapshot['messages'] == first
            assert sum(m.get('content','').startswith('Procedure found: ') for m in first) == 1
            if incident_id:
                assert owner.unread
            endpoint = f'/api/sessions/{id}/procedure-confirmation'
            payload = {'run_id':context.run_id, 'action':action}
            assert (await c.post(endpoint,json={**payload,'run_id':'stale'})).status_code == 409
            assert (await c.post(endpoint,json=payload)).status_code == 202
            assert (await c.post(endpoint,json=payload)).status_code == 409
            await settle(id)
            assert len(rag.calls) == parser.call_count == 1
            assert owner.active_procedure is None
            if action == 'approve':
                extractor.assert_awaited_once()
                assert extractor.await_args.args[0] == context.original_request
                executor.assert_awaited_once()
            else:
                extractor.assert_not_awaited()
                executor.assert_not_awaited()
                assert (await messages(c,id))[-1]['content'] == 'Procedure cancelled.'
                if incident_id:
                    assert web.incident_store.get(incident_id).status == 'NEW'
                    assert (await c.get('/api/incidents')).json()[0]['id'] == incident_id
            assert (await c.post(endpoint,json=payload)).status_code == 409
    asyncio.run(scenario())


def test_cancel_runtime_status_and_no_extraction(integration, config):
    from tests.conftest import drain_procedure_events
    async def scenario():
        async with client() as c:
            id, _ = await prepare(c,'user')
            _, status = await drain_procedure_events(pipeline.handle_procedure_confirmation(
                'cancel', web.get_conversation(id).active_procedure, config))
            assert status.context.status == 'CANCELLED'
            integration[1].assert_not_awaited()
            integration[2].assert_not_awaited()
    asyncio.run(scenario())
