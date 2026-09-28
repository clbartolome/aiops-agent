import asyncio
import json
import ssl
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents import OpenAIChatCompletionsModel
from agents.mcp import MCPServerManager

from app.agent import run_agent
import app.mcp as mcp_module
from app.config import Config, MCPConfig
from app.mcp import MCPConnectionError, create_mcp_servers, mcp_http


@pytest.mark.parametrize("token", [None, "", "   ", " test-token "])
@pytest.mark.parametrize("url", ["http://localhost:8000/mcp", "https://openshift.test/mcp"])
@pytest.mark.parametrize("verify", [True, False])
@pytest.mark.parametrize("unavailable", [False, True])
def test_http_transport_auth_tls_and_cleanup(monkeypatch, token, verify, unavailable, caplog, url):
    """Run the real MCP handshake and discovery against an in-memory HTTP peer."""
    requests, clients, options = [], [], []
    original_client = mcp_http.AsyncClient
    original_ssl_factory = ssl.create_default_context
    original_https_factory = ssl._create_default_https_context

    def respond(request):
        requests.append(request)
        if unavailable:
            raise mcp_http.ConnectError("Bearer test-secret", request=request)
        if request.method != "POST":
            return mcp_http.Response(405)
        message = json.loads(request.content)
        if "id" not in message:
            return mcp_http.Response(202)
        if message["method"] == "server/discover":
            # MCP v2 probes this before falling back to the v1 handshake.
            return mcp_http.Response(200, json={
                "jsonrpc": "2.0", "id": message["id"],
                "error": {"code": -32601, "message": "Method not found"},
            })
        if message["method"] == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "offline-test", "version": "1"},
            }
        else:
            assert message["method"] == "tools/list"
            result = {"tools": []}
        return mcp_http.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    def client(**kwargs):
        options.append(kwargs.copy())
        instance = original_client(**kwargs, transport=mcp_http.MockTransport(respond))
        clients.append(instance)
        return instance

    monkeypatch.setattr(mcp_module, "mcp_http", SimpleNamespace(AsyncClient=client))
    configs = (MCPConfig("openshift", url, token, verify),)
    servers = create_mcp_servers(configs)

    async def exercise():
        async with MCPServerManager(servers, strict=True):
            assert await servers[0].list_tools() == []

    if unavailable:
        with pytest.raises(MCPConnectionError, match="openshift") as error:
            asyncio.run(run_agent("Check pods", Config("test", "https://model.test/v1", "test", configs)))
        assert "test-secret" not in str(error.value)
        assert "test-secret" not in caplog.text
    else:
        asyncio.run(exercise())
    assert requests and clients
    if url.startswith("https://"):
        assert all(option["verify"] is verify for option in options)
    else:
        assert all("verify" not in option for option in options)
    assert all(option["follow_redirects"] is False for option in options)
    assert all(client.is_closed for client in clients)
    for request in requests:
        if token and token.strip():
            assert request.headers["Authorization"] == f"Bearer {token.strip()}"
        else:
            assert "Authorization" not in request.headers
    assert "test-token" not in caplog.text
    assert ssl.create_default_context is original_ssl_factory
    assert ssl._create_default_https_context is original_https_factory
    assert ssl.create_default_context().verify_mode == ssl.CERT_REQUIRED


def test_independent_servers(config):
    configs = (replace(config.mcp_servers[0], tls_verify=False), *config.mcp_servers[1:])
    servers = create_mcp_servers(configs)
    assert [server.name for server in servers] == ["openshift", "aap", "itsm"]
    assert [server.params["url"] for server in servers] == [item.url for item in configs]
    assert [server.params["httpx_client_factory"].keywords["verify"] for server in servers] == [False, True, True]
    assert "Authorization" in servers[0].params["headers"]
    assert "headers" not in servers[1].params
    assert "headers" not in servers[2].params
    assert all(server.cache_tools_list for server in servers)


def test_all_discovered_tools_are_available(config, mcp_boundary):
    servers = create_mcp_servers(config.mcp_servers)

    async def exercise():
        async with MCPServerManager(servers):
            for server in servers:
                discovered = mcp_boundary.sessions[server.name].list_tools.return_value.tools
                assert await server.list_tools() == discovered

    asyncio.run(exercise())


def test_connection_failure_is_safe_and_closes_all_resources(config, mcp_boundary, monkeypatch, caplog):
    secret = config.mcp_servers[0].token
    mcp_boundary.failures["aap"] = ConnectionError(f"Authorization: Bearer {secret}")
    model = AsyncMock(side_effect=AssertionError("Model must not run after connection failure"))
    monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", model)
    with pytest.raises(MCPConnectionError, match=r"aap \(ConnectionError\)") as error:
        asyncio.run(run_agent("Check pods", config))
    assert secret not in str(error.value)
    assert secret not in caplog.text
    assert "Authorization" not in caplog.text
    assert set(mcp_boundary.closed) == {"openshift", "aap", "itsm"}
    model.assert_not_awaited()


def test_model_failure_closes_mcp_sessions(config, mcp_boundary, monkeypatch):
    monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", AsyncMock(
        side_effect=RuntimeError("Model unavailable")
    ))
    with pytest.raises(RuntimeError, match="Model unavailable"):
        asyncio.run(run_agent("Check pods", config))
    assert set(mcp_boundary.closed) == {"openshift", "aap", "itsm"}
