"""Offline prototype evaluations: real compiler validation, SDK binding and LangGraph."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations
from pydantic import ValidationError

from app import web, procedures, procedure_binding
from app.procedure_binding import ProcedureBindingError, bind_procedure
from app.procedure_binding_models import AvailableTool, BoundProcedureDefinition, ToolSelection, execution_risk, catalog_fingerprint
from app.procedure_compiler import compile_procedure
from test_procedure_compiler import semantic_payload
from app.procedure_models import ProcedureCondition, ProcedureDefinition, ProcedureInput
from app.procedure_runtime import (
    ConditionError, ProcedureInputError, ProcedureRuntimeValidationError,
    evaluate_condition, final_arguments, procedure_result, runtime_message,
)
from test_procedure_runtime import bound, runtime, FakeExecutor
from test_procedure_runtime_web import execution, kb
from test_agent import answer, call, exposed_name, model
from app.agent import run_agent

FIXTURES = Path(__file__).parent / 'fixtures' / 'procedures'


def condition(op, field='value', *, step='namespace-check', **kwargs):
    return ProcedureCondition(operand={'step_id': step, 'field': field}, operator=op,
                              source_text='A source-backed test rule.', **kwargs)


@pytest.mark.parametrize('op,left,right,expected', [
    ('exists', {}, None, True), ('not_exists', None, None, True),
    ('equals', 'Running', 'Running', True), ('not_equals', 'Pending', 'Running', True),
    ('greater_than', 3, 2, True), ('greater_than_or_equal', 2, 2, True),
    ('less_than', 1, 2, True), ('less_than_or_equal', 2, 2, True),
    ('is_true', True, None, True), ('is_false', False, None, True),
    ('is_empty', [], None, True), ('is_not_empty', [1], None, True),
    ('equals', 1, 1.0, True), ('exists', None, None, False), ('is_true', False, None, False),
    ('equals', None, 'Running', False), ('not_equals', None, False, True),
])
def test_closed_condition_operators(op, left, right, expected):
    unary = op in {'exists', 'not_exists', 'is_true', 'is_false', 'is_empty', 'is_not_empty'}
    rule = condition(op, **({} if unary else {'value': right}))
    assert evaluate_condition(rule, {'namespace-check': {'structuredContent': {'value': left}}}, {}) is expected


@pytest.mark.parametrize('op,left,right', [
    ('is_true', 'true', None), ('is_false', 0, None), ('greater_than', '3', 2),
    ('equals', True, 1), ('is_empty', False, None), ('greater_than', 2, True),
    ('not_equals', {}, 'Running'), ('equals', float('inf'), 1.0),
])
def test_condition_type_errors_never_coerce(op, left, right):
    if op == 'greater_than' and right is True:
        with pytest.raises(ValidationError):
            condition(op, value=right)
        return
    unary = op in {'is_true', 'is_false', 'is_empty'}
    with pytest.raises(ConditionError):
        evaluate_condition(condition(op, **({} if unary else {'value': right})),
                           {'namespace-check': {'structuredContent': {'value': left}}}, {})


@pytest.mark.parametrize('data', [
    {'source_step': 'namespace-check', 'expression': 'anything'},
    {'operator': 'matches'}, {'operand': {'step_id': 'namespace-check', 'field': 'pods[0]'}},
    {'operand': {'step_id': 'namespace-check', 'field': '__class__'}},
    {'operator': 'is_true', 'value': True}, {'operator': 'exists', 'source_input': 'namespace'},
    {'operator': 'greater_than', 'value': '2'}, {'operator': 'less_than', 'value': float('inf')},
    {'operator': 'equals'}, {'operator': 'equals', 'source_input': 'namespace', 'value': 'other'},
])
def test_unsupported_condition_models_are_rejected(data):
    payload = {'operand': {'step_id': 'namespace-check', 'field': 'value'}, 'operator': 'is_true',
               'source_text': 'A source-backed test rule.'}
    payload.update(data)
    with pytest.raises(ValidationError):
        ProcedureCondition.model_validate(payload)


def test_nested_result_path_and_declared_comparison_input():
    rule = condition('greater_than_or_equal', 'pods.count', source_input='minimum_pod_count')
    results = {'namespace-check': {'structuredContent': {'pods': {'count': 3}}}}
    assert evaluate_condition(rule, results, {'minimum_pod_count': 2}) is True
    with pytest.raises(ConditionError):
        evaluate_condition(rule, results, {})
    with pytest.raises(ConditionError):
        evaluate_condition(condition('not_exists', 'missing'), results, {})
    with pytest.raises(ConditionError):
        evaluate_condition(rule, {'namespace-check': {'content': 'there are 3 pods'}}, {'minimum_pod_count': 2})


def rebuild(bound):
    return BoundProcedureDefinition(procedure=bound.procedure, steps=bound.steps)


class ResultExecutor(FakeExecutor):
    def __init__(self, bound, outputs):
        super().__init__(bound)
        self.outputs = outputs

    def open(self):
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def opened():
            async with super(ResultExecutor, self).open() as invoke:
                async def wrapped(step, arguments):
                    result = await invoke(step, arguments)
                    if step.step_id in self.outputs:
                        result['structuredContent'] = deepcopy(self.outputs[step.step_id])
                    return result
                yield wrapped
        return opened()


def test_valid_precondition_and_numeric_input_run_without_llm(bound, runtime, monkeypatch):
    from agents import OpenAIChatCompletionsModel
    model_call = AsyncMock(side_effect=AssertionError('No runtime condition LLM'))
    monkeypatch.setattr(OpenAIChatCompletionsModel, 'get_response', model_call)
    bound.procedure.inputs.append(ProcedureInput(label='Minimum pod count', required=False, default=2))
    bound.procedure.steps[1].condition = condition('greater_than_or_equal', 'count', source_input='minimum_pod_count')
    executor = ResultExecutor(bound, {'namespace-check': {'count': 3}})
    state = asyncio.run(runtime.start(bound, executor, {'namespace': 'demo', 'application_name': 'router'}))
    assert state['status'] == 'COMPLETED' and len(executor.calls) == 3
    model_call.assert_not_awaited()
    audit = list(state['step_audit'].values())
    assert [row['started_order'] for row in audit] == [1, 3, 5]
    assert [row['completed_order'] for row in audit] == [2, 4, 6]
    assert all(row['status'] == 'SUCCESS' and row['tool_name'].startswith('mcp_') for row in audit)
    assert 'tool_name' not in json.dumps(procedure_result(state))


@pytest.mark.parametrize('missing', [False, True])
def test_unsatisfied_or_unresolvable_precondition_stops_before_later_calls(bound, runtime, missing, caplog):
    bound.procedure.steps[1].condition = condition('is_true', 'ready')
    executor = ResultExecutor(bound, {'namespace-check': {} if missing else {'ready': False}})
    state = asyncio.run(runtime.start(bound, executor, {'namespace': 'demo', 'application_name': 'router'}))
    assert state['status'] == 'FAILED' and state['failure_category'] == 'CONDITION_ERROR'
    assert [item[0] for item in executor.calls] == ['namespace-check']
    assert procedure_result(state)['steps_executed'] == 1
    assert state['step_audit']['application-check']['operation_attempted'] is False
    assert 'reference=namespace-check field=ready operator=is_true' in caplog.text


def test_current_step_stop_condition_stores_result_and_stops(bound, runtime):
    bound.procedure.steps[0].stop_condition = condition('is_not_empty', 'items')
    executor = FakeExecutor(bound)
    state = asyncio.run(runtime.start(bound, executor, {'namespace': 'demo', 'application_name': 'router'}))
    assert state['status'] == 'FAILED' and state['failure_category'] == 'CONDITION_ERROR'
    assert state['step_results']['namespace-check']['structuredContent']['items'] == [1, 2]
    assert len(executor.calls) == 1
    assert state['step_errors']['namespace-check']['reason'] == 'Stop condition was met.'


def test_previous_result_stop_is_checked_before_current_write(bound, runtime):
    bound.procedure.steps[1].stop_condition = condition('is_not_empty', 'items')
    bound.steps[1] = bound.steps[1].model_copy(update={'execution_risk': 'WRITE'})
    bound = rebuild(bound)
    executor = FakeExecutor(bound)
    state = asyncio.run(runtime.start(bound, executor, {'namespace': 'demo', 'application_name': 'router'}))
    assert state['status'] == 'FAILED' and state['failure_category'] == 'CONDITION_ERROR'
    assert [item[0] for item in executor.calls] == ['namespace-check']
    assert state['step_audit']['application-check']['operation_attempted'] is False


@pytest.mark.parametrize('field,operator', [('unknown', 'exists'), ('ready', 'greater_than')])
def test_known_incompatible_result_schema_prevents_any_execution(bound, runtime, field, operator):
    bound.steps[0] = bound.steps[0].model_copy(update={'output_schema': {'type': 'object', 'properties': {'ready': {'type': 'boolean'}}}})
    bound = rebuild(bound)
    bound.procedure.steps[1].condition = condition(operator, field, **({'value': 2} if operator == 'greater_than' else {}))
    executor = FakeExecutor(bound)
    with pytest.raises(ProcedureRuntimeValidationError):
        asyncio.run(runtime.start(bound, executor))
    assert executor.calls == [] and runtime.executors == {}


@pytest.mark.parametrize('risk', ['WRITE', 'DESTRUCTIVE', 'UNKNOWN'])
@pytest.mark.parametrize('approved', [True, False])
def test_risky_steps_require_approval_before_first_change(bound, runtime, risk, approved):
    bound.steps[1] = bound.steps[1].model_copy(update={'execution_risk': risk})
    bound = rebuild(bound)
    executor = FakeExecutor(bound)
    async def scenario():
        state = await runtime.start(bound, executor, {'namespace': 'private-input', 'application_name': 'router'})
        assert state['status'] == 'WAITING_FOR_CONFIRMATION' and state['interrupt']['type'] == 'approval_required'
        assert [item[0] for item in executor.calls] == ['namespace-check']
        summary = runtime_message(state)
        assert 'perform changes' in summary and 'Get application' in summary
        assert 'private-input' not in summary and 'mcp_' not in summary and state['run_id'] not in summary
        state = await runtime.resume(state['run_id'], approved)
        assert state['status'] == ('COMPLETED' if approved else 'CANCELLED')
        assert len(executor.calls) == (3 if approved else 1)
        if not approved:
            assert state['failure_category'] == 'APPROVAL_REJECTED'
    asyncio.run(scenario())


def test_procedure_confirmation_explicitly_covers_all_changes(bound, runtime):
    bound.procedure.confirmation_required = True
    bound.steps[0] = bound.steps[0].model_copy(update={'execution_risk': 'WRITE'})
    bound = rebuild(bound)
    executor = FakeExecutor(bound)
    async def scenario():
        state = await runtime.start(bound, executor, {'namespace': 'demo', 'application_name': 'router'})
        assert executor.calls == [] and 'perform changes' in runtime_message(state)
        state = await runtime.resume(state['run_id'], True)
        assert state['status'] == 'COMPLETED' and state['risky_steps_approved'] is True
        assert len(executor.calls) == 3  # No second redundant approval.
    asyncio.run(scenario())


@pytest.mark.parametrize('annotations,metadata,expected', [
    ({}, {}, 'UNKNOWN'), ({'readOnlyHint': True}, {}, 'READ'),
    ({'readOnlyHint': False, 'destructiveHint': False}, {}, 'WRITE'),
    ({'destructiveHint': True}, {}, 'DESTRUCTIVE'),
    ({'readOnlyHint': True, 'destructiveHint': True}, {}, 'UNKNOWN'),
    ({}, {'http_method': 'GET'}, 'READ'), ({}, {'http_method': 'PATCH'}, 'WRITE'),
    ({}, {'http_method': 'DELETE'}, 'DESTRUCTIVE'),
    ({'readOnlyHint': True}, {'http_method': 'POST'}, 'UNKNOWN'),
])
def test_risk_classification_requires_explicit_metadata(annotations, metadata, expected):
    assert execution_risk(AvailableTool(server='openshift', name='mcp_openshift__get_pods', input_schema={},
                                      annotations=annotations, operation_metadata=metadata)) == expected


def test_contradictory_mutating_tool_name_is_not_read():
    assert execution_risk(AvailableTool(server='aap', name='mcp_aap__users_delete', input_schema={},
                                      annotations={'readOnlyHint': True})) == 'UNKNOWN'


def test_argument_validation_and_plan_immutability(bound, runtime):
    with pytest.raises(ValidationError, match='frozen'):
        bound.steps[0].tool_name = 'mcp_openshift__other'
    with pytest.raises(ProcedureInputError):
        final_arguments(bound, 0, {})
    with pytest.raises(ProcedureInputError):
        final_arguments(bound, 0, {'namespace': {'invented': 'value'}})
    executor = FakeExecutor(bound)
    async def scenario():
        state = await runtime.start(bound, executor)
        changed = deepcopy(state['bound_procedure'])
        changed['procedure']['steps'][0]['action'] = 'delete'
        await runtime.graph.aupdate_state(runtime.config(state['run_id']), {'bound_procedure': changed}, as_node='collect_inputs')
        await runtime.graph.ainvoke(None, runtime.config(state['run_id']))
        state = await runtime.resume(state['run_id'], {'namespace': 'demo', 'application_name': 'router'})
        assert state['status'] == 'FAILED' and state['failure_category'] == 'VALIDATION_ERROR'
        assert executor.calls == []
    asyncio.run(scenario())


def definition_for(name, base):
    if name == 'namespace-inspection':
        result = base.procedure.model_copy(deep=True)
        result.steps[0].description = 'Get the namespace identified by **Namespace** in OpenShift.'
        result.steps[1].description = 'Get **Application name** from **Namespace**.'
        result.steps[2].description = 'List recent events from **Namespace**.'
        result.success_criteria = 'The requested namespace information has been retrieved.'
        result.steps[0].stop_condition = ProcedureCondition(operand={'step_id': 'namespace-check', 'field': 'failed'}, operator='is_true',
            source_text='If the namespace check reports that it failed, stop the procedure.')
        result.steps[1].condition = ProcedureCondition(operand={'step_id': 'namespace-check', 'field': 'failed'}, operator='is_false',
            source_text='Continue only if the namespace check reports that it did not fail.')
        return result
    write = name == 'create-application'
    return ProcedureDefinition.model_validate(dict(
        id='create-namespace-application' if write else 'inspect-one-namespace', version=1,
        title='Create a namespace application' if write else 'Inspect a namespace', risk='medium' if write else 'low',
        confirmation_required=False, inputs=[{'label': 'Namespace'}] + ([{'label': 'Application name'}] if write else []),
        steps=[dict(title='Get namespace', description='Get the namespace identified by **Namespace** in OpenShift.',
                    system='openshift', action='get', resource='namespace', arguments=[{'name': 'namespace', 'source_input': 'namespace'}])] +
              ([dict(title='Create application', description='Create **Application name** in **Namespace**.', system='openshift',
                     action='create', resource='application', arguments=[{'name': 'namespace', 'source_input': 'namespace'},
                                                                   {'name': 'application_name', 'source_input': 'application_name'}])] if write else []),
        success_criteria='The application has been created.' if write else 'The namespace has been retrieved.'))


@pytest.fixture
def fixture_flow(execution, kb, bound, monkeypatch):
    monkeypatch.setattr(procedures, 'compile_procedure', compile_procedure)
    def prepare(name):
        expected = definition_for(name, bound)
        kb.article['description'] = (FIXTURES / (name + '.md')).read_text()
        execution.responses = [answer(json.dumps(semantic_payload(expected)))]
        def respond(**kwargs):
            assert execution.responses, 'Unexpected LLM call after compilation/input extraction'
            return execution.responses.pop(0)
        execution.model.side_effect = respond
        if name == 'create-application':
            execution.extra_tools = [Tool(name='applications_create', description='Create an application in a namespace',
                annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False), inputSchema={
                    'type': 'object', 'properties': {'namespace': {'type': 'string'}, 'name': {'type': 'string'}},
                    'required': ['namespace', 'name'], 'additionalProperties': False})]
        return expected
    return prepare


async def start(client):
    session_id = (await client.post('/api/sessions')).json()['session_id']
    path = '/api/sessions/' + session_id + '/messages'
    data = (await client.post(path, json={'message': '/procedure evaluation fixture'})).json()
    return session_id, path, data


@pytest.mark.parametrize('name', ['namespace-inspection', 'missing-input'])
def test_fixture_read_only_happy_path_and_missing_inputs(fixture_flow, execution, name):
    fixture_flow(name)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id, path, initial = await start(client)
            assert initial['procedure']['status'] == 'WAITING_FOR_INPUT' and execution.calls == []
            reply = '{"namespace":"demo","application_name":"router"}' if name == 'namespace-inspection' else 'demo'
            final = (await client.post(path, json={'message': reply})).json()
            assert final['procedure']['run_id'] == initial['procedure']['run_id']
            assert final['procedure']['status'] == 'COMPLETED'  # No extra READ approval.
            assert web.conversations[session_id].active_procedure_run_id is None
            assert execution.model.await_count == 1 and execution.binder.await_count == 1
            assert [item[0] for item in execution.calls] == (['namespace_get', 'application_get', 'events_list'] if name == 'namespace-inspection' else ['namespace_get'])
    asyncio.run(scenario())


@pytest.mark.parametrize('approve', [True, False])
def test_fixture_write_approval_and_rejection(fixture_flow, execution, approve):
    fixture_flow('create-application')
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id, path, initial = await start(client)
            ready = (await client.post(path, json={'message': 'namespace=demo\napplication_name=private-application-name'})).json()
            assert ready['procedure']['status'] == 'WAITING_FOR_CONFIRMATION'
            assert [item[0] for item in execution.calls] == ['namespace_get']
            assert 'Create application' in ready['response'] and 'perform changes' in ready['response']
            assert 'private-application-name' not in ready['response'] and 'mcp_' not in ready['response']
            final = (await client.post(path, json={'message': 'yes' if approve else 'no'})).json()
            assert final['procedure']['run_id'] == initial['procedure']['run_id']
            assert final['procedure']['status'] == ('COMPLETED' if approve else 'CANCELLED')
            assert [item[0] for item in execution.calls] == (['namespace_get', 'applications_create'] if approve else ['namespace_get'])
            assert web.conversations[session_id].active_procedure_run_id is None
            execution.binder.assert_awaited_once()
            assert execution.model.await_count == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('change', ['schema', 'missing', 'handle'])
def test_live_catalog_verification_prevents_operational_calls(fixture_flow, execution, change):
    fixture_flow('namespace-inspection')
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            session_id, path, initial = await start(client)
            if change == 'handle':
                executor = web.procedure_runtime.executors[initial['procedure']['run_id']]
                executor.tools.pop(('openshift', 'mcp_openshift__application_get'))
            else:
                execution.catalog_change = change
            final = (await client.post(path, json={'message': 'namespace=demo\napplication_name=router'})).json()
            assert final['procedure']['status'] == 'FAILED'
            assert final['procedure']['failure_category'] == 'TOOL_CATALOG_CHANGED'
            assert final['procedure']['steps_executed'] == 0 and execution.calls == []
            assert web.conversations[session_id].active_procedure_run_id is None
            assert len(web.procedure_runtime.executors) == 1
            execution.binder.assert_awaited_once()
            assert execution.model.await_count == 1
    asyncio.run(scenario())


def test_schema_change_while_waiting_for_write_approval_prevents_write(fixture_flow, execution):
    fixture_flow('create-application')
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            _, path, _ = await start(client)
            await client.post(path, json={'message': 'namespace=demo\napplication_name=router'})
            execution.catalog_change = 'schema'
            final = (await client.post(path, json={'message': 'yes'})).json()
            assert final['procedure']['failure_category'] == 'TOOL_CATALOG_CHANGED'
            assert [item[0] for item in execution.calls] == ['namespace_get']
    asyncio.run(scenario())


@pytest.mark.parametrize('model_ambiguity', [False, True])
def test_equivalent_binder_candidates_fail_even_if_model_picks_one(fixture_flow, execution, monkeypatch, caplog, model_ambiguity):
    fixture_flow('create-application')
    duplicate = execution.extra_tools[0].model_copy(update={'name': 'applications_create_alternate'})
    if model_ambiguity:
        duplicate.description = 'Create an application using a different provisioning workflow'
    execution.extra_tools.append(duplicate)
    selection = ToolSelection(tool_name='mcp_openshift__applications_create', ambiguous=model_ambiguity, argument_bindings=[
        {'procedure_argument': 'namespace', 'tool_argument': 'namespace'}, {'procedure_argument': 'application_name', 'tool_argument': 'name'}])
    monkeypatch.setattr(procedure_binding, 'select_tool', AsyncMock(return_value=selection))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            _, _, reply = await start(client)
            assert reply['failure_category'] == 'BINDING_ERROR'
            assert execution.calls == [] and web.procedure_runtime.executors == {}
            execution.agent.assert_not_awaited()
            assert 'semantic_candidates=2' in caplog.text and 'result=ambiguous' in caplog.text
    asyncio.run(scenario())


def test_hallucinated_tool_selection_never_falls_back(fixture_flow, execution, monkeypatch):
    fixture_flow('create-application')
    execution.extra_tools.append(execution.extra_tools[0].model_copy(update={
        'name': 'applications_create_alternate', 'description': 'Create an application using an alternative strategy'}))
    monkeypatch.setattr(procedure_binding, 'select_tool', AsyncMock(return_value=ToolSelection(
        tool_name='mcp_openshift__invented', argument_bindings=[])))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            _, _, reply = await start(client)
            assert reply['failure_category'] == 'BINDING_ERROR'
            assert execution.calls == [] and web.procedure_runtime.executors == {}
            execution.agent.assert_not_awaited()
    asyncio.run(scenario())


def test_unrelated_catalog_changes_do_not_rebind_the_plan(fixture_flow, execution):
    fixture_flow('namespace-inspection')
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            _, path, initial = await start(client)
            snapshot = await web.procedure_runtime.snapshot(initial['procedure']['run_id'])
            before = snapshot['bound_procedure']['catalog_fingerprint']
            execution.extra_tools.append(Tool(name='unrelated_delete', inputSchema={'type': 'object', 'properties': {}}))
            final = (await client.post(path, json={'message': 'namespace=demo\napplication_name=router'})).json()
            assert final['procedure']['status'] == 'COMPLETED'
            snapshot = await web.procedure_runtime.snapshot(initial['procedure']['run_id'])
            assert snapshot['bound_procedure']['catalog_fingerprint'] == before
            execution.binder.assert_awaited_once()
            assert execution.model.await_count == 1
            assert [item[0] for item in execution.calls] == ['namespace_get', 'application_get', 'events_list']
    asyncio.run(scenario())


def test_catalog_fingerprint_is_stable_and_duplicate_metadata_cannot_disagree(bound):
    before = bound.catalog_fingerprint
    changed = [step.model_copy(deep=True) for step in reversed(bound.steps)]
    for step in changed:
        step.input_schema['required'].reverse()
        step.input_schema['properties'] = dict(reversed(list(step.input_schema['properties'].items())))
    assert catalog_fingerprint(changed) == before
    duplicate = bound.steps[0].model_copy(update={'execution_risk': 'WRITE'})
    with pytest.raises(ValueError, match='conflicting metadata'):
        catalog_fingerprint([bound.steps[0], duplicate])


def test_unknown_mapped_argument_is_rejected_locally(bound, runtime):
    from app.procedure_binding_models import ToolArgumentBinding
    bound.steps[0] = bound.steps[0].model_copy(update={'argument_bindings': [
        ToolArgumentBinding(procedure_argument='namespace', tool_argument='fabricated')]})
    bound = rebuild(bound)
    executor = FakeExecutor(bound)
    # Use the genuine tool schema, rather than a fake schema generated from the bad mapping.
    executor.schemas[(bound.steps[0].mcp_server, bound.steps[0].tool_name)] = bound.steps[0].input_schema
    with pytest.raises(ProcedureInputError, match='Unknown tool argument'):
        final_arguments(bound, 0, {'namespace': 'demo'})
    with pytest.raises(ProcedureRuntimeValidationError, match='invalid argument binding'):
        asyncio.run(runtime.start(bound, executor, {'namespace': 'demo', 'application_name': 'router'}))
    assert executor.calls == [] and runtime.executors == {}


def test_binding_preview_tool_names_require_explicit_debug(kb):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            safe = (await client.post('/api/chat', json={'message': '/procedure create user'})).json()
            debug = (await client.post('/api/chat', json={'message': '/procedure create user', 'debug_mode': True})).json()
            assert 'mcp_' not in safe['response'] and 'mcp_' in debug['response']
            assert 'No steps have been executed' in safe['response'] + debug['response']
            kb.direct.assert_not_awaited()
    asyncio.run(scenario())


def test_hallucinated_compiler_step_fails_before_binding(fixture_flow, execution):
    expected = fixture_flow('missing-input')
    extra = expected.steps[0].model_copy(update={'id': 'invented-step', 'title': 'Invented step'})
    expected.steps.append(extra)
    execution.responses[0] = answer(json.dumps(semantic_payload(expected)))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            _, _, reply = await start(client)
            assert reply['failure_category'] == 'VALIDATION_ERROR' and 'No steps have been executed' in reply['response']
            assert execution.calls == [] and web.procedure_runtime.executors == {}
            execution.binder.assert_not_awaited()
            execution.agent.assert_not_awaited()
    asyncio.run(scenario())


def test_compiler_unknown_stop_reference_fails_before_binding(fixture_flow, execution):
    expected = fixture_flow('namespace-inspection')
    expected.steps[0].stop_condition.operand.step_id = 'nonexistent_step'
    execution.responses[0] = answer(json.dumps(semantic_payload(expected)))
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url='http://test') as client:
            _, _, reply = await start(client)
            assert reply['failure_category'] == 'VALIDATION_ERROR'
            assert execution.calls == [] and web.procedure_runtime.executors == {}
            execution.binder.assert_not_awaited()
    asyncio.run(scenario())


def test_direct_agent_recovers_from_initial_tool_error(model, config, mcp_boundary, caplog):
    mcp_boundary.tool_results['openshift'] = CallToolResult(isError=True, content=[TextContent(type='text', text='Unknown namespace')])
    def respond(**kwargs):
        tool_name = exposed_name(kwargs['tools'], 'openshift', 'get_pod_count')
        if model.await_count == 1:
            response = call(tool_name, {'namespace': 'mistyped'})
            response.output[0].call_id = 'attempt-one'
            return response
        if model.await_count == 2:
            mcp_boundary.tool_results.pop('openshift')
            response = call(tool_name, {'namespace': 'payments'})
            response.output[0].call_id = 'attempt-two'
            return response
        return answer('Payments: 3 pods.')
    model.side_effect = respond
    assert asyncio.run(run_agent('Check payments pods', config)) == 'Payments: 3 pods.'
    assert mcp_boundary.sessions['openshift'].call_tool.await_count == 2
    assert 'succeeded=1 failed=1' in caplog.text
