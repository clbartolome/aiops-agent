import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from uuid import UUID

import pytest

from app.mcp import MCPExecutionError
from app.procedure_binding_models import BoundProcedureDefinition
from app.procedure_runtime import (
    ProcedureRuntime, ProcedureRuntimeValidationError, ProcedureInputError,
    procedure_result, runtime_message,
)


@pytest.fixture
def bound():
    return BoundProcedureDefinition.model_validate({
        'procedure': dict(id='inspect-namespace', version=1, title='Inspect namespace', risk='low',
            confirmation_required=False, inputs=[dict(label='Namespace'), dict(label='Application name')],
            steps=[dict(id=step_id, title=title, description=title, system='openshift', action=action, resource=resource,
                        arguments=[dict(name=name, source_input=source) for name, source in arguments])
                   for step_id, title, action, resource, arguments in [
                       ('namespace-check', 'Get namespace', 'get', 'namespace', [('namespace', 'namespace')]),
                       ('application-check', 'Get application', 'get', 'application', [('namespace', 'namespace'), ('application_name', 'application_name')]),
                       ('events-list', 'List events', 'list', 'events', [('namespace', 'namespace')]),
                   ]], success_criteria='The requested system information was retrieved.'),
        'steps': [dict(step_id=step_id, mcp_server='openshift', tool_name='mcp_openshift__' + tool,
                       argument_bindings=[dict(procedure_argument=name, tool_argument=target) for name, target in mapping], execution_risk='READ',
                       input_schema=dict(type='object', properties={target: {'type': 'string', 'minLength': 1} for _, target in mapping},
                                         required=[target for _, target in mapping], additionalProperties=False))
                  for step_id, tool, mapping in [
                      ('namespace-check', 'namespace_get', [('namespace', 'namespace')]),
                      ('application-check', 'application_get', [('namespace', 'namespace'), ('application_name', 'name')]),
                      ('events-list', 'events_list', [('namespace', 'namespace')]),
                  ]],
    })


class FakeExecutor:
    def __init__(self, bound, failure=None):
        self.secrets = ('private-input-value',)
        self.calls = []
        self.failure = failure
        self.schemas = {(step.mcp_server, step.tool_name): dict(type='object',
            properties={arg.tool_argument: {'type': 'string', 'minLength': 1} for arg in step.argument_bindings},
            required=[arg.tool_argument for arg in step.argument_bindings], additionalProperties=False) for step in bound.steps}

    @asynccontextmanager
    async def open(self):
        async def invoke(step, arguments):
            self.calls.append((step.step_id, step.tool_name, deepcopy(arguments)))
            if step.step_id == self.failure:
                raise MCPExecutionError('MCPToolError', {'isError': True, 'structuredContent': {'failure': True}})
            return {'isError': False, 'structuredContent': {'step': step.step_id, 'items': [1, 2]}, 'content': {'type': 'text', 'text': 'raw-result'}}
        yield invoke


@pytest.fixture
def executor(bound):
    return FakeExecutor(bound)


@pytest.fixture
def runtime():
    return ProcedureRuntime()


def test_new_unique_runs_are_real_langgraph_threads(bound, executor, runtime):
    async def scenario():
        a = await runtime.start(bound, executor)
        b = await runtime.start(bound, executor)
        assert UUID(a['run_id']) != UUID(b['run_id'])
        assert a['procedure_id'] == b['procedure_id'] == 'inspect-namespace'
        snapshot = await runtime.graph.aget_state(runtime.config(a['run_id']))
        assert snapshot.config['configurable']['thread_id'] == a['run_id']
        assert snapshot.values['status'] == 'WAITING_FOR_INPUT'
        assert executor.calls == []
    asyncio.run(scenario())


def test_missing_fields_interrupt_before_any_tool(bound, executor, runtime):
    async def scenario():
        state = await runtime.start(bound, executor, {'application_name': 'router'})
        assert state['status'] == 'WAITING_FOR_INPUT'
        assert state['interrupt']['type'] == 'input_required'
        assert state['interrupt']['run_id'] == state['run_id']
        assert state['interrupt']['fields'] == [{'name': 'namespace', 'label': 'Namespace', 'description': None}]
        assert executor.calls == []
        assert 'Namespace' in runtime_message(state)
    asyncio.run(scenario())


def test_resume_same_thread_executes_exact_bound_tools_in_order(bound, executor, runtime):
    async def scenario():
        state = await runtime.start(bound, executor, {'application_name': 'router'})
        run_id = state['run_id']
        state = await runtime.resume(run_id, {'namespace': 'openshift-ingress'})
        assert state['run_id'] == run_id
        assert state['inputs'] == {'application_name': 'router', 'namespace': 'openshift-ingress'}
        assert state['status'] == 'COMPLETED'
        assert state['current_step_index'] == 3
        assert [call[:2] for call in executor.calls] == [(step.step_id, step.tool_name) for step in bound.steps]
        assert executor.calls[1][2] == {'namespace': 'openshift-ingress', 'name': 'router'}
        assert set(state['step_results']) == {'namespace-check', 'application-check', 'events-list'}
        assert state['step_results']['application-check']['structuredContent']['items'] == [1, 2]
        assert state['step_errors'] == {}
        assert state['interrupt'] is None
        assert procedure_result(state)['steps_executed'] == 3
        assert all(item['status'] == 'SUCCESS' for item in procedure_result(state)['steps'])
        persisted = await runtime.graph.aget_state(runtime.config(run_id))
        assert persisted.values['inputs']['namespace'] == 'openshift-ingress'
        with pytest.raises(ProcedureInputError, match='not waiting'):
            await runtime.resume(run_id, {'namespace': 'different'})
        assert len(executor.calls) == 3
    asyncio.run(scenario())


def test_multiple_missing_fields_and_partial_resumes(bound, executor, runtime):
    async def scenario():
        state = await runtime.start(bound, executor)
        run_id = state['run_id']
        assert state['requested_fields'] == ['namespace', 'application_name']
        state = await runtime.resume(run_id, {'namespace': 'openshift-ingress'})
        assert state['requested_fields'] == ['application_name']
        assert state['status'] == 'WAITING_FOR_INPUT'
        assert executor.calls == []
        state = await runtime.resume(run_id, {'application_name': 'router'})
        assert state['run_id'] == run_id and state['status'] == 'COMPLETED'
        assert len(executor.calls) == 3
    asyncio.run(scenario())


@pytest.mark.parametrize('confirmed, expected, count', [(True, 'COMPLETED', 3), (False, 'CANCELLED', 0)])
def test_confirmation_happens_before_any_operational_tool(bound, executor, runtime, confirmed, expected, count):
    bound.procedure.confirmation_required = True
    async def scenario():
        state = await runtime.start(bound, executor, {'namespace': 'production', 'application_name': 'router'})
        assert state['status'] == 'WAITING_FOR_CONFIRMATION'
        assert state['interrupt']['type'] == 'confirmation_required'
        assert executor.calls == []
        state = await runtime.resume(state['run_id'], {'confirmed': confirmed})
        assert state['status'] == expected
        assert len(executor.calls) == count
    asyncio.run(scenario())


def test_input_then_confirmation_does_not_repeat_side_effects(bound, executor, runtime):
    bound.procedure.confirmation_required = True
    async def scenario():
        state = await runtime.start(bound, executor)
        state = await runtime.resume(state['run_id'], {'namespace': 'production', 'application_name': 'router'})
        assert state['status'] == 'WAITING_FOR_CONFIRMATION'
        assert executor.calls == []
        with pytest.raises(ProcedureInputError, match='yes or no'):
            await runtime.resume(state['run_id'], 'yes')
        assert executor.calls == []
        state = await runtime.resume(state['run_id'], True)
        assert state['status'] == 'COMPLETED'
        assert len(executor.calls) == 3
    asyncio.run(scenario())


def test_tool_failure_stops_remaining_steps_and_preserves_results(bound, runtime):
    executor = FakeExecutor(bound, failure='application-check')
    async def scenario():
        state = await runtime.start(bound, executor, {'namespace': 'production', 'application_name': 'router'})
        assert state['status'] == 'FAILED'
        assert [call[0] for call in executor.calls] == ['namespace-check', 'application-check']
        assert state['step_results']['namespace-check']['structuredContent']['step'] == 'namespace-check'
        assert state['step_results']['application-check']['isError'] is True
        assert state['step_errors']['application-check'] == {'category': 'TOOL_ERROR', 'reason': 'MCPToolError', 'message': 'The bound system tool failed.'}
        assert [item['status'] for item in procedure_result(state)['steps']] == ['SUCCESS', 'FAILED', 'NOT_EXECUTED']
        assert procedure_result(state)['steps_executed'] == 2
        with pytest.raises(ProcedureInputError):
            await runtime.resume(state['run_id'], {})
        assert len(executor.calls) == 2
    asyncio.run(scenario())


@pytest.mark.parametrize('field', ['condition', 'stop_condition'])
def test_free_form_conditions_fail_before_graph_or_mcp(bound, executor, runtime, field):
    from app.procedure_models import ProcedureCondition
    setattr(bound.procedure.steps[1], field, ProcedureCondition.model_construct(source_step='namespace-check', expression='If namespace exists.'))
    with pytest.raises(ProcedureRuntimeValidationError, match='invalid'):
        asyncio.run(runtime.start(bound, executor))
    assert executor.calls == []
    assert runtime.executors == {}
    assert list(runtime.checkpointer.list(None)) == []


def test_invalid_bound_plan_fails_before_graph(bound, executor, runtime):
    bound.steps.pop()
    with pytest.raises(ProcedureRuntimeValidationError, match='invalid'):
        asyncio.run(runtime.start(bound, executor))
    assert executor.calls == [] and runtime.executors == {}


@pytest.mark.parametrize('payload', [
    {'unknown': 'value'}, {'namespace': ['not', 'scalar']}, {'namespace': 2},
    {'namespace': float('nan')}, {'bound_procedure': {}},
])
def test_resume_rejects_invalid_inputs_without_mutating_checkpoint(bound, executor, runtime, payload):
    async def scenario():
        state = await runtime.start(bound, executor)
        before = await runtime.snapshot(state['run_id'])
        with pytest.raises(ProcedureInputError):
            await runtime.resume(state['run_id'], payload)
        assert await runtime.snapshot(state['run_id']) == before
        assert executor.calls == []
    asyncio.run(scenario())


def test_plan_is_immutable_and_inputs_are_not_logged(bound, executor, runtime, caplog):
    async def scenario():
        state = await runtime.start(bound, executor)
        bound.steps[0] = bound.steps[0].model_copy(update={'tool_name': 'mcp_openshift__unselected'})
        bound.procedure.steps.reverse()
        state = await runtime.resume(state['run_id'], {'namespace': 'private-input-value', 'application_name': 'router'})
        assert state['status'] == 'COMPLETED'
        assert executor.calls[0][1] == 'mcp_openshift__namespace_get'
        assert 'private-input-value' not in caplog.text
    asyncio.run(scenario())


def test_defaults_and_optional_missing_arguments(bound, executor, runtime):
    bound.procedure.inputs[0].default = 'default-namespace'
    bound.procedure.inputs[1].required = False
    executor.schemas[(bound.steps[1].mcp_server, bound.steps[1].tool_name)]['required'] = ['namespace']
    bound.steps[1].input_schema['required'] = ['namespace']
    bound = BoundProcedureDefinition(procedure=bound.procedure, steps=bound.steps)
    state = asyncio.run(runtime.start(bound, executor))
    assert state['status'] == 'COMPLETED'
    assert state['inputs'] == {'namespace': 'default-namespace'}
    assert executor.calls[1][2] == {'namespace': 'default-namespace'}


def test_chat_value_parsing_is_deterministic(bound, executor, runtime):
    async def scenario():
        state = await runtime.start(bound, executor)
        assert await runtime.parse_message(state['run_id'], 'Namespace=123\nApplication name=router') == {
            'namespace': '123', 'application_name': 'router'}
        assert await runtime.parse_message(state['run_id'], '{"namespace":"openshift-ingress","application_name":"router"}') == {
            'namespace': 'openshift-ingress', 'application_name': 'router'}
        with pytest.raises(ProcedureInputError):
            await runtime.parse_message(state['run_id'], 'unstructured response for two fields')
    asyncio.run(scenario())
