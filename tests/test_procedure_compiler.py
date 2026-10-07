import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from agents import OpenAIChatCompletionsModel
from pydantic import ValidationError

from app.procedure_compiler import (
    ProcedureCompilationError, ProcedureValidationError, compile_procedure, parse_source, merge_semantics, SemanticEnrichment, StepSemantics,
    source_steps, semantic_input, canonicalize_unary_conditions,
)
from app.procedure_models import ProcedureCondition, ProcedureDefinition
from test_agent import answer


@pytest.mark.parametrize('operator', ['exists', 'not_exists', 'is_true', 'is_false', 'is_empty', 'is_not_empty'])
@pytest.mark.parametrize('explicit_nulls', [False, True])
def test_unary_condition_without_rhs(operator, explicit_nulls):
    data = dict(operand=dict(step_id='verify_that_the_namespace_exists'), operator=operator,
                source_text='If the namespace does not exist, stop the procedure.')
    if explicit_nulls:
        data.update(value=None, source_input=None)
    result = ProcedureCondition.model_validate(data)
    assert result.value is None and result.source_input is None
    assert ProcedureCondition.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize('operator,rhs', [
    ('equals', dict(value='Running')), ('equals', dict(source_input='expected_status')),
    ('greater_than_or_equal', dict(source_input='minimum_pod_count')),
    ('equals', dict(value=False)), ('greater_than_or_equal', dict(value=0)),
])
def test_comparison_condition_has_one_rhs(operator, rhs):
    assert ProcedureCondition(operand=dict(step_id='check'), operator=operator, source_text='Source rule.', **rhs)


@pytest.mark.parametrize('operator', ['equals', 'not_equals', 'greater_than', 'greater_than_or_equal', 'less_than', 'less_than_or_equal'])
@pytest.mark.parametrize('rhs', [{}, dict(value=None, source_input=None), dict(value='Running', source_input='expected_status')])
def test_comparison_condition_invalid_rhs(operator, rhs):
    with pytest.raises(ValidationError, match='exactly one'):
        ProcedureCondition(operand=dict(step_id='check'), operator=operator, source_text='Source rule.', **rhs)


@pytest.mark.parametrize('operator', ['exists', 'not_exists', 'is_true', 'is_false', 'is_empty', 'is_not_empty'])
@pytest.mark.parametrize('rhs', [dict(value='Running'), dict(value=False), dict(value=0), dict(source_input='expected_status'), dict(source_input='')])
def test_unary_condition_rejects_rhs(operator, rhs):
    with pytest.raises(ValidationError, match='Unary condition'):
        ProcedureCondition(operand=dict(step_id='check'), operator=operator, source_text='Source rule.', **rhs)

MARKDOWN = '''# Create operations user

Creates a standard operations user.

## Procedure

**ID:** create-operations-user
**Version:** 1
**Risk:** medium
**Confirmation required:** yes

## Required information

- **Username** — required — Username of the new user.
- **Full name** — required — Full name of the user.
- **Email** — required — Corporate email address.
- **Team** — optional — Defaults to `operations`.

## Steps

### 1. Check whether the user already exists

Check whether the requested user already exists in AAP.

Use:
- `username` from **Username**

If the user already exists, stop the procedure and inform the user.

### 2. Create the user

Create the user in AAP.

Use:
- `username` from **Username**
- `full_name` from **Full name**
- `email` from **Email**

Run this step only if the previous step confirms that the user does not exist.

### 3. Add the user to the operations team

Add the new user to the requested team.

Use:
- `username` from **Username**
- `team` from **Team**

### 4. Verify the user

Verify that the user exists and belongs to the expected team.

Use:
- `username` from **Username**

## Success

The procedure is successful when the user exists and belongs to the expected team.
'''


def extraction():
    return dict(
        id='create-operations-user', version=1, title='Create operations user',
        description='Creates a standard operations user.', risk='medium', confirmation_required=True,
        inputs=[dict(label=label, required=required, default=default) for label, required, default in [
            ('Username', True, None), ('Full name', True, None), ('Email', True, None), ('Team', False, 'operations'),
        ]],
        steps=[dict(title=title, description=description, system="aap", action=action, resource="user",
                    arguments=[dict(name=name, source_input=source) for name, source in arguments])
               for title, description, action, arguments in [
                   ('Check whether the user already exists', 'Check whether the requested user already exists in AAP.', 'get', [('username', 'username')]),
                   ('Create the user', 'Create the user in AAP.', 'create', [('username', 'username'), ('full_name', 'full_name'), ('email', 'email')]),
                   ('Add the user to the operations team', 'Add the new user to the requested team.', 'update', [('username', 'username'), ('team', 'team')]),
                   ('Verify the user', 'Verify that the user exists and belongs to the expected team.', 'verify', [('username', 'username')]),
               ]],
        success_criteria='The procedure is successful when the user exists and belongs to the expected team.',
    )


@pytest.fixture
def data():
    data = extraction()
    data['steps'][1]['condition'] = dict(
        operand=dict(step_id='check_whether_the_user_already_exists', field='exists'), operator='is_false',
        source_text='Run this step only if the previous step confirms that the user does not exist.',
    )
    data['steps'][0]['stop_condition'] = dict(
        operand=dict(step_id='check_whether_the_user_already_exists', field='exists'), operator='is_true',
        source_text='If the user already exists, stop the procedure and inform the user.',
    )
    return data


def semantic_payload(definition):
    """Adapt fixture intent to the semantic-only wire contract, not production code."""
    from app.procedure_models import identifier
    data = definition.model_dump() if hasattr(definition, 'model_dump') else deepcopy(definition)
    aliases = {step.get('id') or identifier(step['title']): identifier(step['title']) for step in data['steps']}
    steps = []
    for step in data['steps']:
        entry = {key: deepcopy(step[key]) for key in ('system', 'action', 'resource', 'arguments', 'condition', 'stop_condition') if key in step}
        entry['step_id'] = identifier(step['title'])
        for field in ('condition', 'stop_condition'):
            rule = entry.get(field)
            if rule:
                reference = rule['operand']['step_id']
                rule['operand']['step_id'] = aliases.get(reference, reference)
        steps.append(entry)
    return dict(steps=steps)


def mock_enrichment(monkeypatch, payload):
    import json
    model = AsyncMock(return_value=answer(json.dumps(payload)))
    monkeypatch.setattr(OpenAIChatCompletionsModel, 'get_response', model)
    return model


def test_valid_structured_compilation(config, monkeypatch, data, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    payload = semantic_payload(data)
    model = mock_enrichment(monkeypatch, payload)
    result = asyncio.run(compile_procedure(MARKDOWN, config))
    source = parse_source(MARKDOWN)
    assert [item.label for item in result.inputs] == [item.label for item in source.inputs]
    assert [item.name for item in result.inputs] == ['username', 'full_name', 'email', 'team']
    assert [step.id for step in result.steps] == [step.id for step in source.steps]
    assert [step.description for step in result.steps] == [step.source_text for step in source.steps]
    assert result.steps[0].stop_condition.operand.step_id == result.steps[0].id
    assert result.success_criteria == source.success_text
    kwargs = model.await_args.kwargs
    assert kwargs['tools'] == [] and kwargs['handoffs'] == []
    assert issubclass(kwargs['output_schema'].output_type, SemanticEnrichment)
    output_schema = kwargs['output_schema'].output_type.model_json_schema()
    assert output_schema['properties']['steps']['minItems'] == 4
    assert output_schema['properties']['steps']['maxItems'] == 4
    assert json.loads(kwargs['input']) == semantic_input(source)
    assert 'Source procedure parsed' in caplog.text
    assert 'Semantic enrichment completed' in caplog.text
    assert 'Deterministic validation completed' in caplog.text


@pytest.mark.parametrize('change', [
    lambda d: d.update(version=0), lambda d: d.update(risk='critical'),
    lambda d: d.update(id=' '), lambda d: d.update(steps=[]),
    lambda d: d['steps'][0].update(resource=' '),
    lambda d: d['steps'][0].update(action=' '),
    lambda d: d['steps'][0].update(description=' '),
    lambda d: d['steps'][0]['arguments'][0].update(source_input='unknown'),
    lambda d: d['steps'][1]['condition'].get('operand').update(step_id='unknown'),
    lambda d: d['steps'][1]['condition'].get('operand').update(step_id='verify_the_user'),
    lambda d: d['steps'][1]['condition'].get('operand').update(step_id='create_the_user'),
    lambda d: d['steps'][0]['stop_condition'].update(source_text=' '),
    lambda d: d['inputs'].append(dict(label='Full-name')),
    lambda d: d['inputs'].append(deepcopy(d['inputs'][0])),
    lambda d: d['steps'][3].update(id='create_the_user'),
    lambda d: d['inputs'][3].update(default=1.5),
    lambda d: d.update(code='print(1)'),
])
def test_closed_model_rejects_invalid_definitions(data, change):
    change(data)
    with pytest.raises(ValidationError):
        ProcedureDefinition.model_validate(data)



def test_parser_extracts_only_step_headings_and_preserves_boundaries():
    source = MARKDOWN.replace('Create the user in AAP.', 'Create the user in AAP.\n\n1. An internal list item\n2. Another item')
    parsed = parse_source(source)
    assert len(parsed.steps) == 4
    assert [step.index for step in parsed.steps] == [0, 1, 2, 3]
    assert [step.title for step in parsed.steps] == [title for title, _ in source_steps(source)]
    assert '2. Another item' in parsed.steps[1].source_text
    assert '## Success' not in parsed.steps[-1].source_text
    assert all(step.title not in ('Procedure', 'Required information', 'Steps', 'Success') for step in parsed.steps)


def test_parser_owns_labels_identifiers_defaults_and_order():
    source = MARKDOWN.replace('Username', 'Namespace').replace('Full name', 'Application name')
    parsed = parse_source(source)
    assert [(item.label, item.name, item.required, item.default) for item in parsed.inputs] == [
        ('Namespace', 'namespace', True, None), ('Application name', 'application_name', True, None),
        ('Email', 'email', True, None), ('Team', 'team', False, 'operations')]
    assert parsed.steps[0].id == 'check_whether_the_user_already_exists'


@pytest.mark.parametrize('literal,expected', [('`1`', 1), ('true', True), ('false', False), ('-2', -2), ('"operations"', 'operations')])
def test_parser_explicit_defaults(literal, expected):
    source = MARKDOWN.replace('`operations`', literal)
    item = parse_source(source).inputs[-1]
    assert type(item.default) is type(expected) and item.default == expected


@pytest.mark.parametrize('default', ['some value', '1.5', '1 or 2', '`one` or `two`'])
def test_unsupported_defaults_fail_before_llm(default, config, monkeypatch):
    model = mock_enrichment(monkeypatch, {})
    with pytest.raises(ProcedureValidationError, match='default'):
        asyncio.run(compile_procedure(MARKDOWN.replace('`operations`', default), config))
    model.assert_not_awaited()


@pytest.mark.parametrize('change', [
    lambda source: source.replace('**Version:** 1', '**Version:** 0'),
    lambda source: source.replace('**Risk:** medium', '**Risk:** unknown'),
    lambda source: source.replace('**Confirmation required:** yes', '**Confirmation required:** maybe'),
    lambda source: source.replace('### 2.', '### 5.'),
    lambda source: source.replace('### 4. Verify the user', '### fourth. Verify the user'),
    lambda source: source.replace('### 4. Verify the user', '### 4. CREATE-the-user!'),
    lambda source: source.replace('- **Team**', '- **FULL-NAME**'),
    lambda source: source.replace('## Success', '## Steps'),
    lambda source: source.replace('## Steps', '## Extra'),
])
def test_parser_rejects_invalid_structure(change):
    with pytest.raises((ProcedureValidationError, ValidationError)):
        parse_source(change(MARKDOWN))


@pytest.mark.parametrize('change', ['missing', 'extra', 'unknown', 'duplicate'])
def test_enrichment_requires_exact_source_coverage(data, change):
    payload = semantic_payload(data)
    if change == 'missing':
        payload['steps'].pop()
    elif change == 'extra':
        extra = deepcopy(payload['steps'][0]); extra['step_id'] = 'invented_step'; payload['steps'].append(extra)
    elif change == 'unknown':
        payload['steps'][0]['step_id'] = 'unknown_step'
    else:
        payload['steps'][1]['step_id'] = payload['steps'][0]['step_id']
    with pytest.raises(ProcedureValidationError, match='coverage|Duplicate'):
        merge_semantics(parse_source(MARKDOWN), SemanticEnrichment.model_validate(payload))


def test_reordered_semantics_merge_in_source_order(data):
    payload = semantic_payload(data)
    payload['steps'].reverse()
    source = parse_source(MARKDOWN)
    result = merge_semantics(source, SemanticEnrichment.model_validate(payload))
    assert [step.id for step in result.steps] == [step.id for step in source.steps]
    assert result.steps[1].action == 'create'


@pytest.mark.parametrize('field,value', [('title', 'Invented title'), ('description', 'Invented prose'), ('id', 'invented_id')])
def test_llm_cannot_return_source_owned_fields(data, field, value):
    payload = semantic_payload(data); payload['steps'][0][field] = value
    with pytest.raises(ValidationError, match='Extra inputs'):
        SemanticEnrichment.model_validate(payload)


@pytest.mark.parametrize('kind', ['operator', 'future', 'unknown', 'input', 'missing_action', 'implementation'])
def test_invalid_semantics_fail_closed(config, monkeypatch, data, kind, caplog):
    payload = semantic_payload(data)
    if kind == 'operator':
        payload['steps'][0]['stop_condition']['operator'] = 'eval'
    elif kind in ('future', 'unknown'):
        payload['steps'][0]['stop_condition']['operand']['step_id'] = payload['steps'][-1]['step_id'] if kind == 'future' else 'nonexistent_step'
    elif kind == 'input':
        payload['steps'][1]['arguments'][0]['source_input'] = 'invented_input'
    elif kind == 'missing_action':
        payload['steps'][2].pop('action')
    else:
        payload['steps'][0]['resource'] = 'mcp_openshift__pods_list'
    mock_enrichment(monkeypatch, payload)
    with pytest.raises(ProcedureCompilationError):
        asyncio.run(compile_procedure(MARKDOWN, config))
    assert 'Procedure validation failed' in caplog.text
    assert 'field=' in caplog.text


@pytest.mark.parametrize('text', ['Not JSON', '{"steps":', '<|channel|>analysis'])
def test_invalid_structured_output_is_safe(config, monkeypatch, text, caplog):
    monkeypatch.setattr(OpenAIChatCompletionsModel, 'get_response', AsyncMock(return_value=answer(text)))
    with pytest.raises(ProcedureCompilationError):
        asyncio.run(compile_procedure(MARKDOWN, config))
    assert text not in caplog.text


def test_structure_is_parsed_before_model_call(config, monkeypatch):
    model = mock_enrichment(monkeypatch, {})
    with pytest.raises(ProcedureCompilationError):
        asyncio.run(compile_procedure('Not a procedure', config))
    model.assert_not_awaited()


def test_no_condition_prose_reinterpretation(data):
    payload = semantic_payload(data)
    payload['steps'][0]['stop_condition']['source_text'] = 'Paraphrased provenance.'
    result = merge_semantics(parse_source(MARKDOWN), SemanticEnrichment.model_validate(payload))
    assert result.steps[0].stop_condition.operator == 'is_true'


@pytest.fixture
def health_source():
    from pathlib import Path
    return (Path(__file__).parent / 'fixtures/procedures/inspect-namespace-health.md').read_text()


def health_semantics():
    ids = ['verify_that_the_namespace_exists', 'list_pods_in_the_namespace', 'review_recent_events', 'inspect_deployments']
    payload = {'steps': [dict(step_id=step_id, system='openshift', action=action, resource=resource,
                             arguments=[dict(name='namespace', source_input='namespace')])
                         for step_id, action, resource in zip(ids, ['verify', 'list', 'list', 'list'],
                                                             ['namespace', 'pods', 'events', 'deployments'])]}
    payload['steps'][0]['stop_condition'] = dict(operand=dict(step_id=ids[0]), operator='not_exists',
        value=None, source_input=None, source_text='Stop if the namespace was not found.')
    return payload


def test_real_health_article_source_parser(health_source):
    source = parse_source(health_source)
    assert source.id == 'inspect-namespace-health'
    assert [(item.label, item.name, item.required) for item in source.inputs] == [('Namespace', 'namespace', True)]
    assert [step.title for step in source.steps] == ['Verify that the namespace exists', 'List pods in the namespace',
                                                   'Review recent events', 'Inspect deployments']
    assert [step.id for step in source.steps] == ['verify_that_the_namespace_exists', 'list_pods_in_the_namespace',
                                                'review_recent_events', 'inspect_deployments']
    assert 'If the namespace does not exist' in source.steps[0].source_text
    assert '## Success' not in source.steps[-1].source_text


def test_three_step_source_has_exactly_three_steps(health_source):
    import re
    markdown = re.sub(r'### 4\. Inspect deployments.*?(?=## Success)', '', health_source, flags=re.S)
    source = parse_source(markdown)
    assert [step.id for step in source.steps] == ['verify_that_the_namespace_exists', 'list_pods_in_the_namespace', 'review_recent_events']


@pytest.mark.parametrize('missing_or_extra', ['missing', 'extra'])
def test_real_four_step_source_requires_four_semantic_results(health_source, missing_or_extra):
    payload = health_semantics()
    if missing_or_extra == 'missing':
        payload['steps'].pop()
    else:
        extra = deepcopy(payload['steps'][0])
        extra['step_id'] = 'not_in_the_source'
        payload['steps'].append(extra)
    with pytest.raises(ProcedureValidationError, match='coverage differs from source'):
        merge_semantics(parse_source(health_source), SemanticEnrichment.model_validate(payload))


def test_real_health_article_semantic_enrichment(config, monkeypatch, health_source, caplog):
    import logging
    caplog.set_level(logging.INFO)
    payload = health_semantics()
    payload['steps'].reverse()
    model = mock_enrichment(monkeypatch, payload)
    result = asyncio.run(compile_procedure(health_source, config))
    assert [step.id for step in result.steps] == [step.id for step in parse_source(health_source).steps]
    assert result.steps[0].condition is None
    assert result.steps[0].stop_condition.operator == 'not_exists'
    assert all(step.system == 'openshift' for step in result.steps)
    assert all([arg.source_input for arg in step.arguments] == ['namespace'] for step in result.steps)
    supplied = json.loads(model.await_args.kwargs['input'])
    assert supplied['procedure_context']['description'].startswith('Inspect the current state of an OpenShift namespace')
    assert supplied['procedure_context']['inputs'][0]['description'] == 'OpenShift namespace to inspect.'
    assert supplied['steps'][0]['source_text'] == parse_source(health_source).steps[0].source_text
    assert 'If the namespace does not exist, stop the procedure and inform the user.' in supplied['steps'][0]['source_text']
    assert 'Source procedure parsed' in caplog.text and 'Semantic enrichment completed' in caplog.text
    assert '"arguments": [{"name": "namespace", "source_input": "namespace"}]' in caplog.text


def test_global_system_context_reaches_enrichment(config, monkeypatch, health_source):
    # A later step need not repeat the globally stated system.
    model = mock_enrichment(monkeypatch, health_semantics())
    result = asyncio.run(compile_procedure(health_source, config))
    request = json.loads(model.await_args.kwargs['input'])
    assert 'OpenShift' in request['procedure_context']['description']
    assert 'OpenShift' not in request['steps'][1]['source_text']
    assert request['steps'][1]['step_title'] == 'List pods in the namespace'
    assert result.steps[1].system == 'openshift'


def test_unidentified_system_is_not_invented(config, monkeypatch, health_source):
    source = health_source.replace('OpenShift', 'target')
    payload = health_semantics()
    for step in payload['steps']:
        step['system'] = None
    mock_enrichment(monkeypatch, payload)
    result = asyncio.run(compile_procedure(source, config))
    assert all(step.system is None for step in result.steps)


def test_semantic_source_debug_logging_contains_complete_body(config, monkeypatch, health_source, caplog):
    import logging
    caplog.set_level(logging.DEBUG, logger='app.procedure_compiler')
    mock_enrichment(monkeypatch, health_semantics())
    asyncio.run(compile_procedure(health_source, config))
    record = next(record for record in caplog.records if 'Semantic source' in record.message
                  and 'step=verify_that_the_namespace_exists' in record.message)
    assert 'Check that **Namespace** exists in the OpenShift cluster.' in record.message
    assert 'If the namespace does not exist, stop the procedure and inform the user.' in record.message


def test_source_prose_logging_is_opt_in(config, monkeypatch, health_source, caplog):
    import logging
    caplog.set_level(logging.INFO, logger='app.procedure_compiler')
    mock_enrichment(monkeypatch, health_semantics())
    asyncio.run(compile_procedure(health_source, config))
    assert 'Semantic source' not in caplog.text
    assert 'Check that **Namespace**' not in caplog.text


def test_health_semantics_reach_unchanged_binder(config, monkeypatch, health_source):
    import app.procedure_binding as binding
    from app.procedure_binding import bind_procedure
    from test_procedure_binding import tool, selection
    mock_enrichment(monkeypatch, health_semantics())
    procedure = asyncio.run(compile_procedure(health_source, config))
    catalog = [tool(name=name, description=description) for name, description in [
        ('namespaces_get', 'Get a namespace'), ('pods_list', 'List pods in a namespace'),
        ('events_list', 'List events in a namespace'), ('deployments_list', 'List deployments in a namespace')]]
    async def select(step, candidates, allowed_mappings, config):
        expected = next(item for item in candidates if item.name.split('__')[-1].startswith(step.resource + '_'))
        return selection(expected)
    monkeypatch.setattr(binding, 'select_tool', AsyncMock(side_effect=select))
    bound = asyncio.run(bind_procedure(procedure, catalog, config))
    assert len(bound.steps) == 4
    assert all(step.mcp_server == 'openshift' for step in bound.steps)


def test_explicit_declared_input_reference_cannot_be_omitted(health_source):
    payload = health_semantics()
    payload['steps'][0]['arguments'] = []
    with pytest.raises(ProcedureValidationError, match='Explicit source input references were omitted'):
        merge_semantics(parse_source(health_source), SemanticEnrichment.model_validate(payload))


def test_full_procedure_output_is_rejected(config, monkeypatch, data):
    mock_enrichment(monkeypatch, data)
    with pytest.raises(ProcedureCompilationError):
        asyncio.run(compile_procedure(MARKDOWN, config))


def test_provider_failure_hides_payload(config, monkeypatch, caplog):
    monkeypatch.setattr(OpenAIChatCompletionsModel, 'get_response', AsyncMock(side_effect=RuntimeError('private-model-payload')))
    with pytest.raises(ProcedureCompilationError):
        asyncio.run(compile_procedure(MARKDOWN, config))
    assert 'private-model-payload' not in caplog.text


def test_compiler_logs_no_input_defaults_or_condition_values(config, monkeypatch, data, caplog):
    import logging
    caplog.set_level(logging.INFO)
    source = MARKDOWN.replace('`operations`', '`private-default-value`')
    payload = semantic_payload(data)
    payload['steps'][0]['stop_condition'].update(operator='equals', value='private-condition-value')
    mock_enrichment(monkeypatch, payload)
    asyncio.run(compile_procedure(source, config))
    assert 'private-default-value' not in caplog.text
    assert 'private-condition-value' not in caplog.text


@pytest.mark.parametrize('operator', ['exists', 'not_exists', 'is_true', 'is_false', 'is_empty', 'is_not_empty'])
@pytest.mark.parametrize('rhs', [dict(value='namespace'), dict(source_input='namespace'),
                                 dict(value=False, source_input='namespace')])
def test_semantic_compiler_canonicalizes_unary_rhs(config, monkeypatch, health_source, operator, rhs, caplog):
    import logging
    caplog.set_level(logging.DEBUG, logger='app.procedure_compiler')
    payload = health_semantics()
    rule = payload['steps'][0]['stop_condition']
    rule.update(operator=operator, **rhs)
    mock_enrichment(monkeypatch, payload)
    result = asyncio.run(compile_procedure(health_source, config))
    canonical = result.steps[0].stop_condition
    assert canonical.operator == operator
    assert canonical.value is None and canonical.source_input is None
    assert 'Raw semantic condition stage=before_pydantic' in caplog.text
    assert 'Canonical semantic condition stage=before_pydantic' in caplog.text
    assert caplog.text.index('Raw semantic condition') < caplog.text.index('Pydantic validation completed')


def test_comparison_enrichment_is_unchanged(data):
    payload = semantic_payload(data)
    payload['steps'][0]['stop_condition'].update(operator='greater_than_or_equal', value=None,
                                                source_input='minimum_pod_count')
    original = deepcopy(payload)
    canonical = canonicalize_unary_conditions(payload)
    assert canonical['steps'][0]['stop_condition'] == payload['steps'][0]['stop_condition']
    assert payload == original
    assert ProcedureCondition.model_validate(canonical['steps'][0]['stop_condition']).source_input == 'minimum_pod_count'


@pytest.mark.parametrize('operator,rhs', [
    ('greater_than', dict(value=None, source_input=None)),
    ('equals', dict(value='Running', source_input='namespace')),
    ('unknown', dict(value='namespace', source_input='namespace')),
])
def test_semantic_canonicalization_does_not_repair_other_conditions(config, monkeypatch, health_source, operator, rhs):
    payload = health_semantics()
    payload['steps'][0]['stop_condition'].update(operator=operator, **rhs)
    assert canonicalize_unary_conditions(payload) == payload
    mock_enrichment(monkeypatch, payload)
    with pytest.raises(ProcedureCompilationError):
        asyncio.run(compile_procedure(health_source, config))


def test_unary_canonicalization_does_not_repair_unknown_step(config, monkeypatch, health_source):
    payload = health_semantics()
    payload['steps'][0]['stop_condition']['operand']['step_id'] = 'invented_step'
    payload['steps'][0]['stop_condition']['value'] = 'namespace'
    mock_enrichment(monkeypatch, payload)
    with pytest.raises(ProcedureCompilationError):
        asyncio.run(compile_procedure(health_source, config))


def test_raw_condition_diagnostics_redact_arbitrary_literals(config, monkeypatch, health_source, caplog):
    import logging
    caplog.set_level(logging.DEBUG, logger='app.procedure_compiler')
    payload = health_semantics()
    payload['steps'][0]['stop_condition']['value'] = 'private-condition-payload'
    mock_enrichment(monkeypatch, payload)
    asyncio.run(compile_procedure(health_source, config))
    assert 'private-condition-payload' not in caplog.text
    assert 'redacted str literal' in caplog.text


def test_condition_schema_explains_rhs_constraints():
    properties = ProcedureCondition.model_json_schema()['properties']
    for field in ('value', 'source_input'):
        assert 'Comparison operators only' in properties[field]['description']
        assert 'Must be null for all unary operators' in properties[field]['description']
