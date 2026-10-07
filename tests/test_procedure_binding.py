import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from agents import OpenAIChatCompletionsModel
from pydantic import ValidationError

from app import procedure_binding as binding
from app.procedure_binding import ProcedureBindingError, bind_procedure, validate_selection
from app.procedure_binding_models import (
    AvailableTool, BoundProcedureDefinition, BoundProcedureStep, ToolArgumentBinding, ToolSelection,
)
from app.procedure_models import ProcedureDefinition
from test_agent import answer


def tool(name='pods_list_in_namespace', *, server='openshift', parameters=None, required=None, description=None):
    parameters = {'namespace': {'type': 'string'}} if parameters is None else parameters
    return AvailableTool(server=server, name=f'mcp_{server}__{name}', description=description if description is not None else name.replace('_', ' '),
                         input_schema={'type': 'object', 'properties': parameters,
                                       'required': list(parameters) if required is None else required})


@pytest.fixture
def procedure():
    return ProcedureDefinition.model_validate(dict(
        id='inspect-namespace', version=1, title='Inspect namespace', risk='low', confirmation_required=False,
        inputs=[dict(label='Namespace'), dict(label='Application name')],
        steps=[dict(title='List pods', description='Retrieve pods in **Namespace**.',
                    system='openshift', action='list', resource='pods',
                    arguments=[dict(name='namespace', source_input='namespace')])],
        success_criteria='The namespace pods were retrieved.',
    ))


def selection(candidate, mapping=None):
    mapping = {'namespace': 'namespace'} if mapping is None else mapping
    return ToolSelection(tool_name=candidate.name if isinstance(candidate, AvailableTool) else candidate,
                         argument_bindings=[ToolArgumentBinding(procedure_argument=name, tool_argument=target)
                                            for name, target in mapping.items()])


@pytest.fixture
def selector(monkeypatch):
    mock = AsyncMock(side_effect=AssertionError('No model selection should be needed'))
    monkeypatch.setattr(binding, 'select_tool', mock)
    return mock


def test_single_clear_candidate_uses_no_model_or_execution(procedure, config, selector):
    before = procedure.model_dump_json()
    candidates = [tool(), tool('users_create', server='aap'), tool('tickets_list', server='itsm'),
                  tool('pods_delete'), tool('pods_list_all', parameters={})]
    result = asyncio.run(bind_procedure(procedure, candidates, config))
    assert isinstance(result, BoundProcedureDefinition)
    assert result.steps[0].mcp_server == 'openshift'
    assert result.steps[0].tool_name == candidates[0].name
    assert result.steps[0].argument_bindings == [ToolArgumentBinding(procedure_argument='namespace', tool_argument='namespace')]
    assert procedure.model_dump_json() == before
    assert 'mcp_' not in result.procedure.model_dump_json()
    selector.assert_not_awaited()


def test_filtering_excludes_wrong_servers_and_incompatible_schemas(procedure, config, selector, caplog):
    import logging
    caplog.set_level(logging.INFO)
    a, b = tool(), tool('pods_list')
    catalog = [a, b, tool('pods_get'), tool('users_list', server='aap'), tool('tickets_list', server='itsm'),
               tool('pods_list_cluster', parameters={}), tool('pods_list_selector', parameters={'selector': {'type': 'string'}})]
    selector.side_effect = None
    selector.return_value = selection(b)
    result = asyncio.run(bind_procedure(procedure, catalog, config))
    assert result.steps[0].tool_name == b.name
    filtered = selector.await_args.args[1]
    assert filtered == [a, b]
    assert 'all_tools=7' in caplog.text
    assert 'server_candidates=5' in caplog.text
    assert 'schema_candidates=3' in caplog.text
    assert 'strong_purpose_matches=2' in caplog.text
    assert 'semantic_selection=1' in caplog.text


def test_real_structured_selection_receives_only_filtered_candidates(procedure, config, monkeypatch):
    a, b = tool(), tool('pods_list')
    model = AsyncMock(return_value=answer(selection(b).model_dump_json()))
    monkeypatch.setattr(OpenAIChatCompletionsModel, 'get_response', model)
    result = asyncio.run(bind_procedure(procedure, [a, b, tool('pods_list', server='aap'), tool('pods_list_bad', parameters={})], config))
    assert result.steps[0].tool_name == b.name
    call = model.await_args.kwargs
    payload = json.loads(call['input'])
    assert [item['tool_name'] for item in payload['candidates']] == [a.name, b.name]
    assert payload['step'] == procedure.steps[0].model_dump()
    assert all('input_schema' in item and 'description' in item and item['server'] == 'openshift' for item in payload['candidates'])
    assert call['tools'] == [] and call['handoffs'] == []
    assert call['output_schema'].output_type is ToolSelection
    assert call['system_instructions'] == binding.BINDER_PROMPT


def test_verify_namespace_can_select_get_with_different_description_verb(procedure, config, selector):
    step = procedure.steps[0]
    step.action, step.resource = 'verify', 'namespace'
    candidate = tool('namespace_get', description='Retrieve a namespace by name')
    selector.side_effect = None
    selector.return_value = selection(candidate)
    result = asyncio.run(bind_procedure(procedure, [candidate], config))
    assert selector.await_args.args[1] == [candidate]
    assert result.steps[0].tool_name == candidate.name


def test_zero_strong_matches_retains_all_schema_candidates(procedure, config, selector, caplog):
    import logging
    caplog.set_level(logging.DEBUG, logger='app.procedure_binding')
    procedure.steps[0].action, procedure.steps[0].resource = 'verify', 'namespace'
    candidates = [tool(name, description=description) for name, description in [
        ('namespaces_list', 'List namespaces'), ('namespace_lookup', 'Retrieve namespace information'),
        ('resource_inspect', 'Inspect a resource')]]
    selector.side_effect = None
    selector.return_value = selection(candidates[1])
    catalog = candidates + [tool('namespace_lookup', server='aap'), tool('namespace_bad_schema', parameters={})]
    result = asyncio.run(bind_procedure(procedure, catalog, config))
    selector.assert_awaited_once()
    assert selector.await_args.args[1] == candidates
    assert result.steps[0].tool_name == candidates[1].name
    assert 'server_candidates=4 resource_candidates=4 schema_candidates=3 strong_purpose_matches=0 semantic_candidates=3 semantic_selection=1' in caplog.text
    assert 'Procedure binding candidate names' in caplog.text
    assert all(tool.name in caplog.text for tool in candidates)


@pytest.mark.parametrize('bad,reason', [
    (selection('mcp_openshift__invented'), 'does not exist'),
    (selection('mcp_aap__namespace_lookup'), 'wrong MCP server'),
    (selection('mcp_openshift__namespace_bad_schema'), 'filtered candidate set'),
    (selection('mcp_openshift__namespace_lookup', {'namespace': 'fabricated'}), 'tool argument does not exist'),
])
def test_fallback_candidates_keep_post_selection_validation(procedure, config, selector, bad, reason):
    procedure.steps[0].action, procedure.steps[0].resource = 'verify', 'namespace'
    candidates = [tool('namespace_lookup', description='Retrieve namespace information'), tool('namespaces_list')]
    selector.side_effect = None
    selector.return_value = bad
    catalog = candidates + [tool('namespace_lookup', server='aap'), tool('namespace_bad_schema', parameters={})]
    with pytest.raises(ProcedureBindingError, match=reason):
        asyncio.run(bind_procedure(procedure, catalog, config))
    assert selector.await_args.args[1] == candidates


def test_questionable_single_candidate_requires_semantic_selection(procedure, config, selector):
    procedure.steps[0].action, procedure.steps[0].resource = 'verify', 'namespace'
    candidate = tool('namespace_delete')
    selector.side_effect = None
    selector.return_value = selection(None, {})
    with pytest.raises(ProcedureBindingError, match='Ambiguous candidates'):
        asyncio.run(bind_procedure(procedure, [candidate], config))
    selector.assert_awaited_once()


def test_health_fixture_first_step_reaches_semantic_selection(config, selector):
    from pathlib import Path
    from app.procedure_compiler import parse_source, merge_semantics, SemanticEnrichment
    from test_procedure_compiler import health_semantics
    source = parse_source((Path(__file__).parent / 'fixtures/procedures/inspect-namespace-health.md').read_text())
    procedure = merge_semantics(source, SemanticEnrichment.model_validate(health_semantics()))
    namespace_candidate = tool('namespace_lookup', description='Retrieve a namespace by name')
    candidates = [namespace_candidate, tool('namespaces_list'), tool('resource_inspect', description='Inspect a resource')]
    selector.side_effect = None
    selector.return_value = selection(None, {})
    with pytest.raises(ProcedureBindingError, match='Ambiguous candidates'):
        asyncio.run(bind_procedure(procedure, candidates, config))
    assert selector.await_args.args[0].id == 'verify_that_the_namespace_exists'
    assert selector.await_args.args[1] == candidates


@pytest.mark.parametrize('name,expected', [
    ('pods_list_in_namespace', 'pod'), ('pods_get', 'pod'), ('get_pods', 'pod'),
    ('events_list', 'event'), ('deployments_list', 'deployment'), ('namespaces_get', 'namespace'),
    ('pods_top', 'pod'), ('resource_get', None),
    ('pods_log', 'pod'), ('nodes_stats_summary', 'node'), ('configuration_view', 'configuration'),
])
def test_primary_resource_comes_from_operation_not_scope(name, expected):
    candidate = tool(name, description='Operate on resources in a namespace')
    assert binding.primary_resource(candidate) == expected


def test_unknown_name_uses_direct_object_description_not_input_schema():
    assert binding.primary_resource(tool('opaque', description='List pods in a namespace')) == 'pod'
    assert binding.primary_resource(tool('opaque', description='Opaque operation')) is None
    assert binding.primary_resource(tool('pods_list_in_namespace', description='Get a namespace')) == 'pod'


@pytest.mark.parametrize('resource,candidate_name', [('namespace', 'pods_list_in_namespace'),
    ('event', 'pods_list_in_namespace'), ('deployment', 'events_list')])
def test_known_resource_mismatch_fails_before_model(procedure, config, selector, resource, candidate_name):
    procedure.steps[0].resource = resource
    with pytest.raises(ProcedureBindingError, match='no resource-compatible tool'):
        asyncio.run(bind_procedure(procedure, [tool(candidate_name)], config))
    selector.assert_not_awaited()


@pytest.mark.parametrize('resource,name', [('pod', 'pods_list_in_namespace'), ('event', 'events_list'),
                                        ('deployment', 'deployments_list')])
def test_singular_resources_bind_only_the_correct_tools(procedure, config, selector, resource, name):
    procedure.steps[0].resource = resource
    catalog = [tool('pods_list_in_namespace'), tool('events_list'), tool('deployments_list')]
    result = asyncio.run(bind_procedure(procedure, catalog, config))
    assert result.steps[0].tool_name == 'mcp_openshift__' + name
    selector.assert_not_awaited()


def test_post_selection_resource_validation_does_not_trust_model(procedure):
    procedure.steps[0].resource, procedure.steps[0].action = 'namespace', 'verify'
    wrong = tool('pods_list_in_namespace')
    # Even an accidentally overbroad offered set must not bypass resource checks.
    with pytest.raises(ProcedureBindingError, match='semantic_resource=namespace tool_resource=pod'):
        validate_selection(procedure, procedure.steps[0], selection(wrong), [wrong], [wrong])


def test_unknown_resource_requires_constrained_selection(procedure, config, selector):
    candidate = tool('resource_get', description='Retrieve a generic resource')
    procedure.steps[0].action, procedure.steps[0].resource = 'verify', 'namespace'
    selector.side_effect = None
    selector.return_value = selection(candidate)
    result = asyncio.run(bind_procedure(procedure, [candidate], config))
    assert binding.primary_resource(candidate) is None
    assert result.steps[0].tool_name == candidate.name
    selector.assert_awaited_once()


def test_generic_resource_match_still_requires_model(procedure, config, selector):
    procedure.steps[0].resource, procedure.steps[0].action = 'resource', 'get'
    candidate = tool('resources_get', description='Get a resource')
    selector.side_effect = None
    selector.return_value = selection(candidate)
    asyncio.run(bind_procedure(procedure, [candidate], config))
    selector.assert_awaited_once()


@pytest.mark.parametrize('bad, reason', [
    (selection('mcp_openshift__invented'), 'does not exist'),
    (selection('mcp_aap__pods_list'), 'wrong MCP server'),
    (selection('mcp_openshift__pods_delete'), 'filtered candidate set'),
    (selection('mcp_openshift__pods_list', {'namespace': 'fabricated'}), 'tool argument does not exist'),
    (selection('mcp_openshift__pods_list', {}), 'missing mapping'),
    (selection('mcp_openshift__pods_list', {'namespace': 'namespace', 'invented': 'foo'}), 'unknown procedure argument'),
    (selection(None, {}), 'Ambiguous candidates'),
])
def test_model_output_is_validated_and_failures_are_logged(procedure, config, selector, caplog, bad, reason):
    a, b = tool(), tool('pods_list')
    selector.side_effect = None
    selector.return_value = bad
    with pytest.raises(ProcedureBindingError, match=reason):
        asyncio.run(bind_procedure(procedure, [a, b, tool('pods_list', server='aap'), tool('pods_delete')], config))
    assert 'step=list_pods' in caplog.text
    assert 'procedure=inspect-namespace' in caplog.text
    assert reason in caplog.text


def test_renamed_tool_parameter_does_not_rename_semantic_input(procedure, config, selector):
    procedure.steps[0].action = 'get'
    procedure.steps[0].resource = 'application'
    from app.procedure_models import StepArgument
    procedure.steps[0].arguments.append(StepArgument(name='application_name', source_input='application_name'))
    candidate = tool('application_get', parameters={'namespace': {'type': 'string'}, 'name': {'type': 'string'}})
    before = procedure.model_dump_json()
    result = asyncio.run(bind_procedure(procedure, [candidate], config))
    assert [(arg.procedure_argument, arg.tool_argument) for arg in result.steps[0].argument_bindings] == [
        ('namespace', 'namespace'), ('application_name', 'name')]
    assert procedure.model_dump_json() == before
    assert procedure.inputs[1].name == 'application_name'
    selector.assert_not_awaited()


@pytest.mark.parametrize('system, reason', [(None, 'unknown'), ('aap', 'No matching MCP server')])
def test_unknown_or_missing_server_is_not_guessed(procedure, config, selector, system, reason):
    procedure.steps[0].system = system
    with pytest.raises(ProcedureBindingError, match=reason):
        asyncio.run(bind_procedure(procedure, [tool()], config))
    selector.assert_not_awaited()


@pytest.mark.parametrize('schema', [
    {'type': 'object', 'properties': {}},
    {'type': 'object', 'properties': {'namespace': {'type': 'array', 'items': {'type': 'string'}}}},
    {'type': 'object', 'properties': {'namespace': {'type': 'string'}, 'cluster': {'type': 'string'}}, 'required': ['cluster']},
    {'type': 'object', 'properties': {'namespace': {'type': 'string'}}, 'required': ['missing']},
    {'type': 'object', 'properties': {'namespace': {'type': 'nonsense'}}},
    {'type': 'object', 'properties': {'namespace': {'type': 'string'}}, 'oneOf': [{}, {}]},
    {'type': 'object', 'properties': {'namespace': {'type': 'string'}}, 'minProperties': 2},
    {'type': 'object', 'properties': {'namespace': {'type': 'string'}}, 'propertyNames': {'pattern': '^other$'}},
    {'type': 'object', 'properties': {'namespace': {'type': 'string'}}, 'const': {'namespace': 'fixed-runtime-value'}},
    {'type': 'object', 'properties': {'namespace': {'type': 'null'}}},
    {'type': 'object', 'properties': {'<|namespace|>': {'type': 'string'}}},
])
def test_unsafe_schema_candidates_are_rejected(procedure, config, selector, schema):
    candidate = tool()
    candidate.input_schema = schema
    with pytest.raises(ProcedureBindingError, match='No compatible tools'):
        asyncio.run(bind_procedure(procedure, [candidate], config))
    selector.assert_not_awaited()


def test_default_types_and_constraints_are_checked(procedure, config, selector):
    procedure.inputs[0].default = 1
    with pytest.raises(ProcedureBindingError, match='No compatible tools'):
        asyncio.run(bind_procedure(procedure, [tool()], config))
    procedure.inputs[0].default = 'production'
    candidate = tool(parameters={'namespace': {'type': 'string', 'enum': ['staging']}})
    with pytest.raises(ProcedureBindingError, match='No compatible tools'):
        asyncio.run(bind_procedure(procedure, [candidate], config))


def test_no_procedure_steps_are_bound_to_conditions(procedure, config, selector):
    from app.procedure_models import ProcedureCondition
    procedure.steps[0].stop_condition = ProcedureCondition(operand={'step_id': 'list_pods', 'field': 'pods'}, operator='is_empty', source_text='If there are no pods, stop.')
    result = asyncio.run(bind_procedure(procedure, [tool()], config))
    assert len(result.steps) == 1
    assert result.procedure.steps[0].stop_condition == procedure.steps[0].stop_condition
    selector.assert_not_awaited()


def test_partial_binding_is_never_returned(procedure, config, selector, caplog):
    procedure.steps.extend([deepcopy(procedure.steps[0]), deepcopy(procedure.steps[0])])
    procedure.steps[1].id = 'list-pods-again'
    procedure.steps[2].id = 'get-application'
    procedure.steps[2].resource = 'application'
    procedure.steps[2].action = 'get'
    with pytest.raises(ProcedureBindingError, match='no resource-compatible tool'):
        asyncio.run(bind_procedure(procedure, [tool()], config))
    assert 'step=get-application' in caplog.text
    selector.assert_not_awaited()


def test_bound_model_rejects_incomplete_or_duplicate_step_bindings(procedure):
    with pytest.raises(ValidationError, match='Every semantic operation'):
        BoundProcedureDefinition(procedure=procedure, steps=[])
    step = BoundProcedureStep(step_id='list_pods', mcp_server='openshift', tool_name=tool().name,
                              argument_bindings=[ToolArgumentBinding(procedure_argument='namespace', tool_argument='namespace')])
    with pytest.raises(ValidationError, match='Every semantic operation'):
        BoundProcedureDefinition(procedure=procedure, steps=[step, step])


def test_ambiguous_large_catalog_is_not_sent_to_model(procedure, config, selector):
    candidates = [tool(f'pods_list_{index}') for index in range(binding.MAX_SELECTION_CANDIDATES + 1)]
    with pytest.raises(ProcedureBindingError, match='too broad'):
        asyncio.run(bind_procedure(procedure, candidates, config))
    selector.assert_not_awaited()


@pytest.mark.parametrize('text', ['sensitive-provider-payload', '{"tool_name":"invented"}', '<|channel|>analysis'])
def test_invalid_structured_output_is_safe(procedure, config, monkeypatch, caplog, text):
    model = AsyncMock(return_value=answer(text))
    monkeypatch.setattr(OpenAIChatCompletionsModel, 'get_response', model)
    with pytest.raises(ProcedureBindingError):
        asyncio.run(bind_procedure(procedure, [tool(), tool('pods_list')], config))
    assert text not in caplog.text


def test_unclear_single_candidate_still_needs_semantic_confirmation(procedure, config, selector):
    candidate = tool('inspect', description='Inspect some resource')
    selector.side_effect = None
    selector.return_value = selection(None, {})
    with pytest.raises(ProcedureBindingError, match='Ambiguous candidates'):
        asyncio.run(bind_procedure(procedure, [candidate], config))
    selector.assert_awaited_once()


@pytest.mark.parametrize('mappings, reason', [
    ([('namespace', 'namespace'), ('namespace', 'namespace_name')], 'duplicate argument'),
    ([('namespace', 'namespace_name')], 'Required tool argument cannot be mapped'),
])
def test_existing_parameter_names_are_not_enough_to_validate_mapping(procedure, mappings, reason):
    candidate = tool(parameters={'namespace': {'type': 'string'}, 'namespace_name': {'type': 'string'}}, required=['namespace'])
    choice = ToolSelection(tool_name=candidate.name, argument_bindings=[
        ToolArgumentBinding(procedure_argument=name, tool_argument=target) for name, target in mappings])
    with pytest.raises(ProcedureBindingError, match=reason):
        validate_selection(procedure, procedure.steps[0], choice, [candidate], [candidate])


def test_catalog_duplicate_names_fail_closed(procedure, config, selector):
    with pytest.raises(ProcedureBindingError, match='duplicate concrete tool names'):
        asyncio.run(bind_procedure(procedure, [tool(), tool()], config))
    selector.assert_not_awaited()


def test_required_tool_input_cannot_depend_on_an_absent_optional_value(procedure, config, selector):
    procedure.inputs[0].required = False
    with pytest.raises(ProcedureBindingError, match='No compatible tools'):
        asyncio.run(bind_procedure(procedure, [tool()], config))
    procedure.inputs[0].default = 'default-namespace'
    bound = asyncio.run(bind_procedure(procedure, [tool()], config))
    assert bound.steps[0].argument_bindings[0].tool_argument == 'namespace'
    assert 'default-namespace' not in bound.steps[0].model_dump_json()


def test_boolean_property_schemas_fail_closed(procedure, config, selector):
    candidate = tool(parameters={'namespace': True})
    with pytest.raises(ProcedureBindingError, match='No compatible tools'):
        asyncio.run(bind_procedure(procedure, [candidate], config))


def test_generic_parameter_name_maps_step_argument_not_input_identifier(procedure, config, selector):
    from app.procedure_models import StepArgument
    procedure.steps[0].arguments = [StepArgument(name='namespace_name', source_input='namespace')]
    result = asyncio.run(bind_procedure(procedure, [tool()], config))
    assert result.steps[0].argument_bindings[0].procedure_argument == 'namespace_name'
    assert result.steps[0].argument_bindings[0].tool_argument == 'namespace'
    assert result.procedure.steps[0].arguments[0].source_input == 'namespace'


def test_large_catalog_is_reduced_without_sending_catalog_to_model(procedure, config, selector, caplog):
    import logging
    caplog.set_level(logging.INFO)
    targets = [tool(), tool('pods_list')]
    catalog = targets + [tool(f'workflow_get_{index}', server='aap') for index in range(100)] + [
        tool(f'events_list_{index}', parameters={'selector': {'type': 'string'}}) for index in range(98)]
    selector.side_effect = None
    selector.return_value = selection(targets[0])
    asyncio.run(bind_procedure(procedure, catalog, config))
    assert selector.await_args.args[1] == targets
    assert 'all_tools=200' in caplog.text
    assert 'server_candidates=100' in caplog.text
    assert 'schema_candidates=2' in caplog.text


def test_sdk_discovery_reuses_connected_sessions_and_namespacing(config, mcp_boundary):
    from agents.mcp import MCPServerManager
    from app.mcp import create_mcp_servers, discover_available_tools

    async def scenario():
        servers = create_mcp_servers(config.mcp_servers)
        async with MCPServerManager(servers):
            for server in servers:
                await server.list_tools()
            metadata = await discover_available_tools(servers)
            assert {item.server for item in metadata} == {'openshift', 'aap', 'itsm'}
            assert {item.name for item in metadata} == {
                'mcp_openshift__get_pod_count', 'mcp_openshift__delete_openshift',
                'mcp_aap__get_workflow_status', 'mcp_aap__delete_aap',
                'mcp_itsm__get_ticket', 'mcp_itsm__delete_itsm',
            }
            assert metadata[0].input_schema['properties']['namespace']['type'] == 'string'
            assert metadata[0].description == 'Read namespace'
            assert len(mcp_boundary.servers) == 3
            for session in mcp_boundary.sessions.values():
                session.list_tools.assert_awaited_once()
                session.call_tool.assert_not_awaited()
    asyncio.run(scenario())
