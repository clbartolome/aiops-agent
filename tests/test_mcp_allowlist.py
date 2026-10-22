import asyncio
from dataclasses import replace
import logging
from types import SimpleNamespace

from agents.mcp import MCPServerStreamableHttp
from mcp.types import Tool
import pytest

from app.agent import run_agent
from app.config import DEFAULT_MCP_ALLOWED_TOOLS, MCPConfig, load_config, parse_allowed_tools
from app.mcp import filter_mcp_tools
from app.procedure.executor import execute_step, CLASSIFIER_INSTRUCTIONS
from app.procedure.models import ProcedureStep
from tests.test_agent import model, answer
from tests.test_config import environment


def test_parse_comma_separated_values():
    assert parse_allowed_tools('a,b, c ,,d,a') == {'a', 'b', 'c', 'd'}


def test_filter_is_exact_and_retains_objects():
    tools = [SimpleNamespace(name=name) for name in ['a', 'b', 'c', 'aa']]
    assert filter_mcp_tools(tools, {'a', 'c', 'missing_tool'}) == [tools[0], tools[2]]
    assert filter_mcp_tools(tools, set()) == []


def test_unset_allowlists_are_bounded_demo_defaults(environment, monkeypatch):
    for server in ['OPENSHIFT', 'AAP', 'ITSM']:
        monkeypatch.delenv(f'{server}_MCP_ALLOWED_TOOLS', raising=False)
    config = load_config()
    assert [len(item.effective_allowed_tools) for item in config.mcp_servers] == [8, 11, 10]
    assert [item.effective_allowed_tools for item in config.mcp_servers] == [
        DEFAULT_MCP_ALLOWED_TOOLS[name] for name in ['openshift', 'aap', 'itsm']
    ]
    assert MCPConfig('unknown', 'https://unknown.test/mcp').effective_allowed_tools == set()


def test_env_empty_is_not_default(environment, monkeypatch):
    monkeypatch.setenv('OPENSHIFT_MCP_ALLOWED_TOOLS', '')
    monkeypatch.setenv('AAP_MCP_ALLOWED_TOOLS', ' mcp_aap__jobs_list,,mcp_aap__jobs_list ')
    config = load_config()
    assert config.mcp_servers[0].effective_allowed_tools == set()
    assert config.mcp_servers[1].effective_allowed_tools == {'mcp_aap__jobs_list'}


@pytest.mark.parametrize('path', ['operations', 'step'])
@pytest.mark.parametrize('policy', ['defaults', 'empty', 'custom'])
def test_both_agents_expose_only_allowed_discovered_tools(
    model, config, mcp_boundary, monkeypatch, caplog, path, policy,
):
    caplog.set_level(logging.INFO, logger='app.mcp')
    original_connect = MCPServerStreamableHttp.connect

    async def connect(server):
        await original_connect(server)
        # Advertise all 29 demo tools plus unrelated administration tools.
        names = sorted(DEFAULT_MCP_ALLOWED_TOOLS[server.name]) + [f'mcp_{server.name}__users_destroy']
        server.session.list_tools.return_value.tools = [
            Tool(name=name.removeprefix(f'mcp_{server.name}__'), description='Test tool',
                 inputSchema={'type': 'object', 'properties': {}})
            for name in names
        ]

    monkeypatch.setattr(MCPServerStreamableHttp, 'connect', connect)
    configs = []
    expected = set()
    for server in config.mcp_servers:
        allowed = None if policy == 'defaults' else frozenset()
        if policy == 'custom':
            allowed = frozenset({next(iter(sorted(DEFAULT_MCP_ALLOWED_TOOLS[server.name]))),
                                 f'mcp_{server.name}__missing_tool',
                                 'mcp_wrong_server__pods_list'})
        configs.append(replace(server, allowed_tools=allowed))
        expected.update(DEFAULT_MCP_ALLOWED_TOOLS[server.name] if allowed is None else
                        allowed & DEFAULT_MCP_ALLOWED_TOOLS[server.name])
    config = replace(config, mcp_servers=tuple(configs))
    surfaces = []

    def respond(**kwargs):
        names = {tool.name for tool in kwargs['tools']}
        if kwargs['system_instructions'] == CLASSIFIER_INSTRUCTIONS:
            assert names == set()
            return answer('{"status":"FAILED","summary":"No evidence."}')
        surfaces.append(names)
        return answer('A general answer.')

    model.side_effect = respond
    if path == 'operations':
        asyncio.run(run_agent('hello', config))
        assert surfaces == [expected | {'request_user_input'}]
    else:
        asyncio.run(execute_step('filtered-run', ProcedureStep(
            id='check', title='Check', instruction='Check the current system.'), [], {}, '', config))
        assert surfaces == [expected]
    for server in mcp_boundary.sessions.values():
        server.list_tools.assert_awaited_once()
        server.call_tool.assert_not_awaited()
    assert all('users_destroy' not in name for names in surfaces for name in names)
    assert f'MCP tools exposed count={len(expected)}' in caplog.text
    if policy == 'defaults':
        assert len(expected) == 29
        assert 'mcp_itsm__rag_search_kb' in expected
        assert 'MCP server=openshift discovered=9 allowed=8 exposed=8' in caplog.text
        assert 'MCP server=aap discovered=12 allowed=11 exposed=11' in caplog.text
        assert 'MCP server=itsm discovered=11 allowed=10 exposed=10' in caplog.text
    if policy == 'custom':
        assert 'configured_tool_not_found=mcp_aap__missing_tool' in caplog.text
        assert 'invalid_tool_prefix=mcp_wrong_server__pods_list' in caplog.text
