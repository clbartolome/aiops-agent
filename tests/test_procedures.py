import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from app import web
from app.procedures import has_procedure_section, procedure_query
from test_web import request


@pytest.fixture
def kb(monkeypatch, config):
    state = SimpleNamespace(
        search={"results": [{"id": 7, "title": "Create operations user", "score": .9}]},
        article={"id": 7, "title": "Create operations user", "description": "## Procedure\n\n**ID:** create-operations-user\n\n## Steps\nCreate the user."},
        error=None, structured=False, servers=[],
    )

    async def call_tool(name, arguments):
        if state.error:
            if isinstance(state.error, Exception):
                raise state.error
            return state.error
        if name == 'rag_search_kb':
            payload = state.search
        else:
            assert name == 'get_kb_article' and arguments == {'article_id': 7}
            payload = state.article
        result = {'content': [{'type': 'text', 'text': json.dumps(payload)}]}
        if state.structured:
            result['structuredContent'] = {'result': json.dumps(payload)}
        return CallToolResult.model_validate(result)

    state.call_tool = AsyncMock(side_effect=call_tool)

    async def connect(server):
        state.servers.append(server)
        server.session = SimpleNamespace(
            list_tools=AsyncMock(return_value=ListToolsResult(tools=[
                Tool(name='rag_search_kb', inputSchema={'type': 'object', 'properties': {
                    'query': {'type': 'string'}, 'top_k': {'type': 'integer'},
                }, 'required': ['query']}),
                Tool(name='get_kb_article', inputSchema={'type': 'object', 'properties': {
                    'article_id': {'type': 'integer'},
                }, 'required': ['article_id']}),
            ])), call_tool=state.call_tool,
        )

    monkeypatch.setattr(MCPServerStreamableHttp, 'connect', connect)
    monkeypatch.setattr(web, 'load_config', lambda: config)
    state.direct = AsyncMock(return_value='Direct answer')
    monkeypatch.setattr(web, 'run_agent', state.direct)
    monkeypatch.setattr(web, 'conversations', {})
    yield state
    assert all(server.session is None for server in state.servers)
    for conversation in web.conversations.values():
        conversation.session.close()


@pytest.mark.parametrize('message', [
    'how many pods are running?', 'what is the procedure for creating a user?',
    '/procedures create user', '/Procedure create user',
])
def test_direct_routing_is_unchanged(kb, config, message):
    response = request('POST', '/api/chat', json={'message': message})
    assert response.status_code == 200 and response.json()['response'] == 'Direct answer'
    kb.direct.assert_awaited_once_with(message, config)
    kb.call_tool.assert_not_awaited()


@pytest.mark.parametrize('structured', [False, True])
def test_procedure_retrieves_best_article_using_itsm_mcp(kb, structured):
    kb.structured = structured
    response = request('POST', '/api/chat', json={'message': '  /procedure create operations user  '})
    assert response.status_code == 200
    assert 'Procedure candidate found' in response.json()['response']
    assert kb.article['description'] in response.json()['response']
    assert 'No steps have been executed' in response.json()['response']
    kb.direct.assert_not_awaited()
    assert [server.name for server in kb.servers] == ['itsm']
    assert [call.args for call in kb.call_tool.await_args_list] == [
        ('rag_search_kb', {'query': 'create operations user', 'top_k': 1}),
        ('get_kb_article', {'article_id': 7}),
    ]


def test_non_procedure_article_is_identified(kb):
    kb.article['description'] = '# Overview\nHow operations users work.'
    response = request('POST', '/api/chat', json={'message': '/procedure create operations user'})
    assert response.status_code == 200
    assert 'not an executable procedure' in response.json()['response']
    assert kb.article['description'] in response.json()['response']


@pytest.mark.parametrize('message', ['/procedure', ' /procedure  ', '/procedure\n\t'])
def test_empty_command_is_a_validation_error(kb, message):
    response = request('POST', '/api/chat', json={'message': message})
    assert response.status_code == 422
    assert 'after /procedure' in response.json()['detail']
    kb.direct.assert_not_awaited()
    kb.call_tool.assert_not_awaited()


def test_no_matching_article_does_not_fetch_or_execute(kb):
    kb.search = {'results': []}
    response = request('POST', '/api/chat', json={'message': '/procedure missing article'})
    assert response.status_code == 200
    assert 'No matching' in response.json()['response']
    assert kb.call_tool.await_count == 1
    kb.direct.assert_not_awaited()


@pytest.mark.parametrize('failure', [
    RuntimeError('Authorization: Bearer test-secret'),
    CallToolResult(isError=True, content=[TextContent(type='text', text='test-secret')]),
])
def test_retrieval_failures_are_safe(kb, failure, caplog):
    kb.error = failure
    response = request('POST', '/api/chat', json={'message': '/procedure create operations user'})
    assert response.status_code == 502
    assert 'knowledge-base retrieval failed' in response.json()['detail']
    assert 'test-secret' not in response.text + caplog.text
    kb.direct.assert_not_awaited()


def test_procedure_history_uses_same_sdk_session_as_direct_followup(kb, config):
    first = request('POST', '/api/sessions').json()['session_id']
    second = request('POST', '/api/sessions').json()['session_id']
    path = f'/api/sessions/{first}'
    result = request('POST', path + '/messages', json={'message': '/procedure create operations user'})
    assert result.status_code == 200
    messages = request('GET', path).json()['messages']
    assert messages == [
        {'role': 'user', 'content': '/procedure create operations user'},
        {'role': 'assistant', 'content': result.json()['response']},
    ]
    assert request('GET', f'/api/sessions/{second}').json()['messages'] == []

    async def direct(message, config, session):
        assert session is web.conversations[first].session
        items = await session.get_items()
        assert items[0]['content'] == '/procedure create operations user'
        assert items[1]['content'] == result.json()['response']
        return 'Direct follow-up answer'

    kb.direct.side_effect = direct
    followup = request('POST', path + '/messages', json={'message': 'Explain the article'})
    assert followup.status_code == 200
    kb.direct.assert_awaited_once_with('Explain the article', config, session=web.conversations[first].session)
    assert kb.call_tool.await_count == 2


@pytest.mark.parametrize('markdown, expected', [
    ('## Procedure\nInstructions', True), ('# Title\n\n## Procedure   \n', True),
    ('## Procedure ##', True), ('### Procedure', False), ('## Procedure notes', False),
    ('A sentence mentioning ## Procedure', False), ('    ## Procedure', False),
    ('```markdown\n## Procedure\n```', False), ('~~~\n## Procedure\n~~~', False),
    ('```\n## Procedure\n```\n## Procedure', True),
])
def test_only_actual_procedure_heading_is_recognized(markdown, expected):
    assert has_procedure_section(markdown) is expected


def test_query_removes_only_command_and_surrounding_whitespace():
    assert procedure_query('  /procedure\tcreate  operations user  ') == 'create  operations user'
    assert procedure_query('what is the procedure?') is None
