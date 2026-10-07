import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from agents import OpenAIChatCompletionsModel
from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool, ToolAnnotations

from app import web, procedures
from app.procedure_runtime import ProcedureRuntime
from test_procedure_runtime import bound
from test_procedures import kb, real_binding_discovery
from test_agent import answer


@pytest.fixture
def execution(kb, bound, monkeypatch):
    state = SimpleNamespace(calls=[], sessions=[], failure=None, transport_failure=False, tool_started=None, tool_release=None,
                            extra_tools=[], catalog_change=None)
    monkeypatch.setattr(web, 'procedure_runtime', ProcedureRuntime())
    kb.compile.return_value = bound.procedure
    monkeypatch.setattr(procedures, 'bind_current_tools', real_binding_discovery)
    state.binder = AsyncMock(wraps=procedures.bind_procedure)
    monkeypatch.setattr(procedures, 'bind_procedure', state.binder)
    connect = MCPServerStreamableHttp.connect

    async def connected(server):
        await connect(server)
        if server.name != 'openshift':
            return
        state.sessions.append(server.session)
        server.session.list_tools.return_value = ListToolsResult(tools=[
            Tool(name=name, description=description, annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
                 outputSchema={'type': 'object', 'properties': {'operation': {'type': 'string'}, 'namespace': {'type': 'string'}, 'failed': {'type': 'boolean'}}}, inputSchema={
                'type': 'object', 'properties': {arg: {'type': 'string', 'minLength': 1} for arg in args},
                'required': args, 'additionalProperties': False,
            }) for name, description, args in [
                ('namespace_get', 'Get a namespace', ['namespace']),
                ('application_get', 'Get an application', ['namespace', 'name']),
                ('events_list', 'List namespace events', ['namespace']),
            ]
        ] + state.extra_tools)
        if state.catalog_change == 'schema':
            data = server.session.list_tools.return_value.tools[0].model_dump(by_alias=True)
            data['inputSchema']['properties']['namespace']['type'] = 'integer'
            server.session.list_tools.return_value.tools[0] = Tool.model_validate(data)
        elif state.catalog_change == 'missing':
            server.session.list_tools.return_value.tools.pop(0)

        async def call(name, arguments):
            state.calls.append((name, deepcopy(arguments)))
            if name == 'namespace_get' and state.tool_started is not None:
                state.tool_started.set()
                await state.tool_release.wait()
            failed = name == state.failure
            if failed and state.transport_failure:
                raise RuntimeError('private-operational-result transport details')
            return CallToolResult(isError=failed,
                structuredContent={'operation': name, 'namespace': arguments['namespace'], 'failed': failed},
                content=[TextContent(type='text', text='private-operational-result')])
        server.session.call_tool = AsyncMock(side_effect=call)

    monkeypatch.setattr(MCPServerStreamableHttp, 'connect', connected)
    state.model = AsyncMock(side_effect=AssertionError('No model during execution or clear binding'))
    monkeypatch.setattr(OpenAIChatCompletionsModel, 'get_response', state.model)
    state.agent = kb.direct
    return state


def test_session_starts_pauses_resumes_and_keeps_chat_separate(execution, kb):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id
            response = await client.post(path + '/messages', json={'message': '/procedure inspect namespace'})
            assert response.status_code == 200
            data = response.json()
            run_id = data['procedure']['run_id']
            assert data['procedure']['status'] == 'WAITING_FOR_INPUT'
            assert data['procedure']['required_information'] == ['Namespace', 'Application name']
            assert 'interrupt' not in data['procedure']
            assert execution.calls == []
            assert kb.call_tool.await_count == 2
            assert web.conversations[session_id].active_procedure_run_id == run_id
            assert execution.sessions[0].list_tools.await_count == 1

            invalid = await client.post(path + '/messages', json={'message': '{"unknown":"not-declared"}'})
            assert invalid.json()['procedure']['run_id'] == run_id
            assert invalid.json()['procedure']['status'] == 'WAITING_FOR_INPUT'
            assert execution.calls == []
            # Chat history is not a procedure input source; only Command(resume=...) updates the checkpoint.
            await web.conversations[session_id].session.add_items([{'role': 'assistant', 'content': 'namespace=fabricated-from-history'}])
            done = await client.post(path + '/messages', json={'message': 'namespace=openshift-ingress\napplication_name=router'})
            assert done.status_code == 200
            result = done.json()['procedure']
            assert result['run_id'] == run_id
            assert result['status'] == 'COMPLETED'
            assert result['steps_executed'] == 3
            assert execution.calls == [
                ('namespace_get', {'namespace': 'openshift-ingress'}),
                ('application_get', {'namespace': 'openshift-ingress', 'name': 'router'}),
                ('events_list', {'namespace': 'openshift-ingress'}),
            ]
            assert web.conversations[session_id].active_procedure_run_id is None
            assert len(execution.sessions) == 2  # Binding connection + lazy execution reconnect of the same server instance.
            assert execution.sessions[1].list_tools.await_count == 1  # Fingerprint verification only; no tool selection.
            snapshot = await web.procedure_runtime.snapshot(run_id)
            assert snapshot['step_results']['application-check']['structuredContent']['namespace'] == 'openshift-ingress'
            assert 'private-operational-result' not in done.text
            history = (await client.get(path)).json()['messages']
            assert history[-1]['content'] == done.json()['response']
            assert 'private-operational-result' not in str(history)
            followup = await client.post(path + '/messages', json={'message': 'hello'})
            assert followup.json()['response'] == 'Direct answer'
            execution.agent.assert_awaited_once()
            execution.model.assert_not_awaited()
    asyncio.run(scenario())


@pytest.mark.parametrize('reply, expected, calls', [('yes', 'COMPLETED', 3), ('no', 'CANCELLED', 0)])
def test_session_confirmation_before_first_tool(execution, kb, reply, expected, calls):
    kb.compile.return_value.confirmation_required = True
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id + '/messages'
            start = (await client.post(path, json={'message': '/procedure inspect namespace'})).json()
            supplied = (await client.post(path, json={'message': '{"namespace":"production","application_name":"router"}'})).json()
            assert supplied['procedure']['run_id'] == start['procedure']['run_id']
            assert supplied['procedure']['status'] == 'WAITING_FOR_CONFIRMATION'
            assert execution.calls == []
            bad = (await client.post(path, json={'message': 'unclear confirmation'})).json()
            assert bad['procedure']['status'] == 'WAITING_FOR_CONFIRMATION'
            assert execution.calls == []
            done = (await client.post(path, json={'message': reply})).json()
            assert done['procedure']['status'] == expected
            assert len(execution.calls) == calls
            execution.agent.assert_not_awaited()
            execution.model.assert_not_awaited()
    asyncio.run(scenario())


@pytest.mark.parametrize('transport_failure', [False, True])
def test_session_tool_error_stops_and_does_not_leak_results(execution, caplog, transport_failure):
    execution.failure = 'application_get'
    execution.transport_failure = transport_failure
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id + '/messages'
            start = (await client.post(path, json={'message': '/procedure inspect namespace'})).json()
            done = (await client.post(path, json={'message': 'namespace=production\napplication_name=router'})).json()
            assert done['procedure']['run_id'] == start['procedure']['run_id']
            assert done['procedure']['status'] == 'FAILED'
            assert [item['status'] for item in done['procedure']['steps']] == ['SUCCESS', 'FAILED', 'NOT_EXECUTED']
            assert [call[0] for call in execution.calls] == ['namespace_get', 'application_get']
            assert 'private-operational-result' not in str(done) + caplog.text
            snapshot = await web.procedure_runtime.snapshot(done['procedure']['run_id'])
            # The SDK wraps transport exceptions before returning to the runtime.
            assert snapshot['step_errors']['application-check']['category'] == 'TOOL_ERROR'
            assert snapshot['step_errors']['application-check']['reason'] == ('AgentsException' if transport_failure else 'MCPToolError')
            if not transport_failure:
                assert snapshot['step_results']['application-check']['isError'] is True
    asyncio.run(scenario())


def test_sessions_with_same_procedure_have_independent_execution(execution):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            a = (await client.post('/api/sessions')).json()['session_id']
            b = (await client.post('/api/sessions')).json()['session_id']
            path_a, path_b = '/api/sessions/' + a + '/messages', '/api/sessions/' + b + '/messages'
            one = (await client.post(path_a, json={'message': '/procedure inspect namespace'})).json()['procedure']
            two = (await client.post(path_b, json={'message': '/procedure inspect namespace'})).json()['procedure']
            assert one['run_id'] != two['run_id']
            await client.post(path_a, json={'message': 'namespace=first\napplication_name=router'})
            second_state = await web.procedure_runtime.snapshot(two['run_id'])
            assert second_state['inputs'] == {}
            assert second_state['step_results'] == {}
            assert second_state['status'] == 'WAITING_FOR_INPUT'
            await client.post(path_b, json={'message': 'namespace=second\napplication_name=router'})
            assert [call[1]['namespace'] for call in execution.calls] == ['first'] * 3 + ['second'] * 3
            assert web.conversations[a].active_procedure_run_id is None
            assert web.conversations[b].active_procedure_run_id is None
            execution.model.assert_not_awaited()
    asyncio.run(scenario())


def test_unsupported_condition_is_rejected_before_creating_run(execution, kb):
    from app.procedure_models import ProcedureCondition
    kb.compile.return_value.steps[1].condition = ProcedureCondition(operand={'step_id': 'namespace-check', 'field': 'exists'}, operator='is_true', source_text='If namespace exists.')
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            response = await client.post('/api/sessions/' + session_id + '/messages', json={'message': '/procedure inspect namespace'})
            assert response.status_code == 200
            assert 'not supported by the execution runtime' in response.json()['response']
            assert 'No steps have been executed' in response.json()['response']
            assert 'procedure' not in response.json()
            assert execution.calls == []
            assert web.procedure_runtime.executors == {}
            assert web.conversations[session_id].active_procedure_run_id is None
            assert kb.call_tool.await_count == 2
    asyncio.run(scenario())


def test_complete_markdown_flow_and_natural_language_inputs(execution, kb, bound, monkeypatch):
    from app.procedure_compiler import compile_procedure, SemanticEnrichment
    from test_procedure_compiler import semantic_payload
    from app.procedure_runtime import ProcedureInputValues
    from app.procedure_models import ProcedureDefinition
    monkeypatch.setattr(procedures, 'compile_procedure', compile_procedure)
    markdown = (f'# {bound.procedure.title}\n\nInspect OpenShift.\n\n## Procedure\n\n'
        f'**ID:** {bound.procedure.id}\n**Version:** 1\n**Risk:** low\n**Confirmation required:** no\n\n'
        '## Required information\n\n- **Namespace** — required — Target namespace.\n'
        '- **Application name** — required — Application to inspect.\n\n## Steps\n\n')
    labels = {item.name: item.label for item in bound.procedure.inputs}
    for index, step in enumerate(bound.procedure.steps, 1):
        markdown += f'### {index}. {step.title}\n\n**ID:** {step.id}\n\n{step.description}\n\n'
        markdown += '\n'.join(f'- `{arg.name}` from **{labels[arg.source_input]}**' for arg in step.arguments) + '\n\n'
    markdown += '## Success\n\n' + bound.procedure.success_criteria
    kb.article['description'] = markdown
    execution.model.side_effect = [answer(json.dumps(semantic_payload(bound.procedure))),
        answer(json.dumps({'values': {'namespace': 'openshift-ingress', 'application_name': 'router-default'}}))]
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id + '/messages'
            start = (await client.post(path, json={'message': '/procedure inspect namespace'})).json()
            assert start['procedure']['status'] == 'WAITING_FOR_INPUT'
            assert execution.calls == []
            done = (await client.post(path, json={'message': 'namespace is openshift-ingress and application is router-default'})).json()
            assert done['procedure']['run_id'] == start['procedure']['run_id']
            assert done['procedure']['status'] == 'COMPLETED'
            assert 'Procedure completed successfully.' in done['response']
            assert '✓ Get namespace' in done['response']
            assert all(value not in done['response'] for value in ('mcp_', 'Run:', start['procedure']['run_id'], 'thread_id'))
            assert [call[0] for call in execution.calls] == ['namespace_get', 'application_get', 'events_list']
            assert execution.calls[1][1]['name'] == 'router-default'
            assert kb.call_tool.await_count == 2  # No RAG after startup.
            execution.binder.assert_awaited_once()  # No binding during execution.
            assert execution.model.await_count == 2  # Compile + input extraction only.
            compile_call, extract_call = execution.model.await_args_list
            assert issubclass(compile_call.kwargs['output_schema'].output_type, SemanticEnrichment)
            assert extract_call.kwargs['output_schema'].output_type is ProcedureInputValues
            assert extract_call.kwargs['tools'] == [] and extract_call.kwargs['handoffs'] == []
            payload = json.loads(extract_call.kwargs['input'])
            assert [field['name'] for field in payload['requested_fields']] == ['namespace', 'application_name']
            assert set(payload) == {'requested_fields', 'message'}
            assert web.conversations[session_id].active_procedure_run_id is None
    asyncio.run(scenario())


def test_single_value_reply_needs_no_model(execution, kb):
    kb.compile.return_value.inputs[1].default = 'router-default'
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id + '/messages'
            start = (await client.post(path, json={'message': '/procedure inspect namespace'})).json()
            assert start['procedure']['required_information'] == ['Namespace']
            done = (await client.post(path, json={'message': 'openshift-ingress'})).json()
            assert done['procedure']['run_id'] == start['procedure']['run_id']
            assert done['procedure']['status'] == 'COMPLETED'
            assert execution.calls[0][1] == {'namespace': 'openshift-ingress'}
            execution.model.assert_not_awaited()
    asyncio.run(scenario())


@pytest.mark.parametrize('values', [{'namespace': 'openshift-ingress'}, {}])
def test_partial_natural_reply_requests_only_missing_fields(execution, values):
    execution.model.side_effect = None
    execution.model.return_value = answer(json.dumps({'values': values}))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id + '/messages'
            start = (await client.post(path, json={'message': '/procedure inspect namespace'})).json()
            partial = (await client.post(path, json={'message': 'namespace is openshift-ingress'})).json()
            assert partial['procedure']['run_id'] == start['procedure']['run_id']
            assert partial['procedure']['status'] == 'WAITING_FOR_INPUT'
            assert partial['procedure']['required_information'] == (['Application name'] if values else ['Namespace', 'Application name'])
            snapshot = await web.procedure_runtime.snapshot(start['procedure']['run_id'])
            assert snapshot['inputs'] == values
            assert execution.calls == []
            if values:
                done = (await client.post(path, json={'message': 'router-default'})).json()
                assert done['procedure']['status'] == 'COMPLETED'
                assert execution.model.await_count == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('output', [
    {'values': {'undeclared': 'secret-private-value'}},
    {'values': {'namespace': ['secret-private-value']}},
    {'values': {'namespace': 1.5}},
    {'values': {'namespace': 'secret-private-value'}, 'steps': []},
])
def test_invalid_extraction_does_not_mutate_state_or_execute(execution, output, caplog):
    execution.model.side_effect = None
    execution.model.return_value = answer(json.dumps(output))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id + '/messages'
            start = (await client.post(path, json={'message': '/procedure inspect namespace'})).json()
            before = await web.procedure_runtime.snapshot(start['procedure']['run_id'])
            reply = (await client.post(path, json={'message': 'some natural language reply'})).json()
            assert reply['procedure']['status'] == 'WAITING_FOR_INPUT'
            assert await web.procedure_runtime.snapshot(start['procedure']['run_id']) == before
            assert 'secret-private-value' not in reply['response'] + caplog.text
            assert execution.calls == []
    asyncio.run(scenario())


@pytest.mark.parametrize('confirmation', [False, True])
def test_cancel_and_active_procedure_ownership(execution, kb, confirmation):
    kb.compile.return_value.confirmation_required = confirmation
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id + '/messages'
            start = (await client.post(path, json={'message': '/procedure inspect namespace'})).json()
            if confirmation:
                ready = (await client.post(path, json={'message': 'namespace=production\napplication_name=router'})).json()
                assert 'Risk: low' in ready['response'] and 'Proceed? [yes/no]' in ready['response']
                assert 'mcp_' not in ready['response'] and start['procedure']['run_id'] not in ready['response']
            rejected = (await client.post(path, json={'message': '/procedure something else'})).json()
            assert 'already active' in rejected['response']
            assert web.conversations[session_id].active_procedure_run_id == start['procedure']['run_id']
            assert kb.compile.await_count == 1 and kb.call_tool.await_count == 2
            cancelled = (await client.post(path, json={'message': '/cancel'})).json()
            assert cancelled['procedure']['status'] == 'CANCELLED'
            assert web.conversations[session_id].active_procedure_run_id is None
            state = await web.procedure_runtime.snapshot(start['procedure']['run_id'])
            assert state['status'] == 'CANCELLED' and state['interrupt'] is None
            assert execution.calls == []
            assert 'no active procedure' in (await client.post(path, json={'message': '/cancel'})).json()['response']
            assert (await client.post(path, json={'message': 'how many namespaces are there?'})).json()['response'] == 'Direct answer'
            execution.agent.assert_awaited_once()
    asyncio.run(scenario())


def test_running_cancellation_finishes_inflight_call_and_skips_later_steps(execution):
    async def scenario():
        execution.tool_started, execution.tool_release = asyncio.Event(), asyncio.Event()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id + '/messages'
            start = (await client.post(path, json={'message': '/procedure inspect namespace'})).json()
            executing = asyncio.create_task(client.post(path, json={'message': 'namespace=production\napplication_name=router'}))
            await asyncio.wait_for(execution.tool_started.wait(), 5)
            assert web.conversations[session_id].active_procedure_run_id == start['procedure']['run_id']
            other = asyncio.create_task(client.post(path, json={'message': '/procedure another procedure'}))
            cancel = asyncio.create_task(client.post(path, json={'message': '/cancel'}))
            # Wait until the cancellation request is recorded, before releasing the tool.
            for _ in range(100):
                if start['procedure']['run_id'] in web.procedure_runtime._cancel_requested:
                    break
                await asyncio.sleep(0)
            assert start['procedure']['run_id'] in web.procedure_runtime._cancel_requested
            execution.tool_release.set()
            done, rejected, cancelled = await asyncio.wait_for(asyncio.gather(executing, other, cancel), 5)
            assert done.json()['procedure']['status'] == cancelled.json()['procedure']['status'] == 'CANCELLED'
            assert 'already active' in rejected.json()['response']
            assert len(execution.calls) == 1
            assert web.conversations[session_id].active_procedure_run_id is None
            assert len(web.procedure_runtime.executors) == 1
            execution.model.assert_not_awaited()
    asyncio.run(scenario())


def test_extraction_cannot_change_previously_collected_input(execution):
    execution.model.side_effect = [
        answer(json.dumps({'values': {'namespace': 'openshift-ingress'}})),
        answer(json.dumps({'values': {'namespace': 'different', 'application_name': 'router-default'}})),
        answer(json.dumps({'values': {'application_name': 'router-default'}})),
    ]
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id + '/messages'
            start = (await client.post(path, json={'message': '/procedure inspect namespace'})).json()
            await client.post(path, json={'message': 'namespace is openshift-ingress'})
            before = await web.procedure_runtime.snapshot(start['procedure']['run_id'])
            rejected = (await client.post(path, json={'message': 'application is router-default'})).json()
            assert rejected['procedure']['status'] == 'WAITING_FOR_INPUT'
            assert await web.procedure_runtime.snapshot(start['procedure']['run_id']) == before
            assert execution.calls == []
            assert [field['name'] for field in json.loads(execution.model.await_args.kwargs['input'])['requested_fields']] == ['application_name']
            explicit = (await client.post(path, json={'message': '{"namespace":"different","application_name":"router-default"}'})).json()
            assert explicit['procedure']['status'] == 'WAITING_FOR_INPUT'
            assert await web.procedure_runtime.snapshot(start['procedure']['run_id']) == before
            done = (await client.post(path, json={'message': 'application is router-default'})).json()
            assert done['procedure']['status'] == 'COMPLETED'
            assert execution.calls[1][1] == {'namespace': 'openshift-ingress', 'name': 'router-default'}
            assert execution.model.await_count == 3
            execution.binder.assert_awaited_once()
    asyncio.run(scenario())


def test_extraction_provider_failure_keeps_run_waiting(execution, caplog):
    execution.model.side_effect = RuntimeError('private-provider-payload token=credential')
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id + '/messages'
            start = (await client.post(path, json={'message': '/procedure inspect namespace'})).json()
            reply = (await client.post(path, json={'message': 'namespace is openshift-ingress'})).json()
            assert reply['procedure']['run_id'] == start['procedure']['run_id']
            assert reply['procedure']['status'] == 'WAITING_FOR_INPUT'
            assert 'could not identify' in reply['response']
            assert 'private-provider-payload' not in caplog.text + reply['response']
            assert execution.calls == []
    asyncio.run(scenario())


@pytest.mark.parametrize('stage', ['compilation', 'binding'])
def test_startup_failure_is_atomic(execution, kb, stage):
    from app.procedure_compiler import ProcedureCompilationError
    from app.procedure_binding import ProcedureBindingError
    if stage == 'compilation':
        kb.compile.side_effect = ProcedureCompilationError('test failure')
    else:
        execution.binder.side_effect = ProcedureBindingError('test failure')
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            reply = (await client.post('/api/sessions/' + session_id + '/messages', json={'message': '/procedure inspect namespace'})).json()
            assert 'No steps have been executed.' in reply['response']
            assert execution.calls == []
            assert web.procedure_runtime.executors == {}
            assert web.conversations[session_id].active_procedure_run_id is None
    asyncio.run(scenario())


def test_running_checkpoint_keeps_session_ownership(execution):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id = (await client.post('/api/sessions')).json()['session_id']
            path = '/api/sessions/' + session_id + '/messages'
            start = (await client.post(path, json={'message': '/procedure inspect namespace'})).json()
            run_id = start['procedure']['run_id']
            # A request can end while a run remains RUNNING; chat must not resume
            # or abandon that run merely because its status is not WAITING.
            await web.procedure_runtime.graph.aupdate_state(web.procedure_runtime.config(run_id),
                {'status': 'RUNNING', 'requested_fields': []}, as_node='confirm_if_required')
            reply = (await client.post(path, json={'message': 'how many namespaces are there?'})).json()
            assert reply['procedure']['status'] == 'RUNNING'
            assert reply['procedure']['progress'] == {'step': 1, 'total': 3, 'title': 'Get namespace'}
            assert 'Step 1/3 · Get namespace' in reply['response']
            assert web.conversations[session_id].active_procedure_run_id == run_id
            execution.agent.assert_not_awaited()
            execution.model.assert_not_awaited()
            assert execution.calls == []
            cancelled = (await client.post(path, json={'message': '/cancel'})).json()
            assert cancelled['procedure']['status'] == 'CANCELLED'
            assert web.conversations[session_id].active_procedure_run_id is None
    asyncio.run(scenario())
