import asyncio
from unittest.mock import AsyncMock

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from app import agent as agent_module
from app.agent import run_agent
from scripts.diagnose_model_tools import log_request_tools, select_tools


def test_sdk_request_does_not_enable_unsupported_strict_tool_mode(config, mcp_boundary, monkeypatch):
    """Use the real SDK serializer, reproducing the provider's strict-mode rejection."""
    question = "Provide a namespace to continue."

    def completion(**kwargs):
        tools = kwargs['tools']
        assert len(tools) == 7
        if any(tool['function'].get('strict') for tool in tools):
            raise RuntimeError('Internal server error: structure_info not used with HarmonyParser')
        local = next(tool['function'] for tool in tools if tool['function']['name'] == 'request_user_input')
        assert local['parameters']['required'] == ['question']
        assert local['parameters']['properties']['question']['type'] == 'string'
        return ChatCompletion.model_validate({
            'id': 'test', 'created': 0, 'model': 'gpt-oss-20b', 'object': 'chat.completion',
            'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {
                'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'question_1', 'type': 'function', 'function': {
                    'name': 'request_user_input', 'arguments': '{"question":"' + question + '"}',
                }}],
            }}],
        })

    completions = AsyncMock(side_effect=completion)

    def client(**kwargs):
        instance = AsyncOpenAI(**kwargs)
        monkeypatch.setattr(instance.chat.completions, 'create', completions)
        return instance

    monkeypatch.setattr(agent_module, 'AsyncOpenAI', client)
    monkeypatch.setattr(AsyncOpenAI, 'request', AsyncMock(side_effect=AssertionError('No network')))
    assert asyncio.run(run_agent('How many pods are in the namespace?', config)) == question
    assert completions.await_count == 1
    for session in mcp_boundary.sessions.values():
        session.call_tool.assert_not_awaited()


def test_diagnostic_selection_is_temporary_and_preserves_tools():
    from copy import deepcopy
    from dataclasses import replace

    local = agent_module.request_user_input
    mcp = replace(local, name='mcp_openshift__pods_list', strict_json_schema=False)
    original_schema = deepcopy(mcp.params_json_schema)
    tools = [mcp, local]
    assert select_tools(tools, 'mcp-only', 'openshift') == [mcp]
    strict_mcp = select_tools(tools, 'single-strict-mcp', 'openshift')[0]
    assert strict_mcp.strict_json_schema is True
    assert mcp.strict_json_schema is False
    assert mcp.params_json_schema == original_schema
    assert select_tools(tools, 'all', 'openshift') == tools
    assert local.strict_json_schema is False


def test_tool_diagnostic_logging_omits_payload_values(caplog):
    import logging

    caplog.set_level(logging.INFO, logger='app.tool_diagnosis')
    log_request_tools([{'function': {
        'name': 'request_user_input', 'strict': False,
        'description': 'sensitive-description',
        'parameters': {'type': 'object', 'required': ['question'], 'properties': {
            'question': {'type': 'string', 'default': 'sensitive-default'},
        }},
    }}])
    assert 'count=1' in caplog.text and 'origin=local' in caplog.text
    assert 'strict=False' in caplog.text and 'properties=1 required=1' in caplog.text
    assert 'sensitive-' not in caplog.text
