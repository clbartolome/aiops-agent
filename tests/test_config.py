import pytest

from app.config import load_config
from app.mcp import create_mcp_servers


@pytest.fixture
def environment(monkeypatch):
    values = {
        "MODEL_NAME": "local-model",
        "MODEL_BASE_URL": "http://localhost:11434/v1",
        "MODEL_API_KEY": "test-secret",
        "OPENSHIFT_MCP_URL": "https://openshift.test/mcp",
        "AAP_MCP_URL": "https://aap.test/mcp",
        "ITSM_MCP_URL": "https://itsm.test/mcp",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


def test_load_config(environment):
    config = load_config()
    assert config.model_name == environment["MODEL_NAME"]
    assert config.model_base_url == environment["MODEL_BASE_URL"]
    assert config.model_api_key == environment["MODEL_API_KEY"]
    assert config.model_api_key not in repr(config)


@pytest.mark.parametrize("name", ["MODEL_NAME", "MODEL_BASE_URL", "MODEL_API_KEY"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_required_config(environment, monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        load_config()


@pytest.mark.parametrize("prefix", ["OPENSHIFT", "AAP", "ITSM"])
@pytest.mark.parametrize("verify", ["true", "false"])
def test_mcp_config(environment, monkeypatch, prefix, verify):
    monkeypatch.setenv(f"{prefix}_MCP_TLS_VERIFY", verify)
    monkeypatch.setenv(f"{prefix}_MCP_TOKEN", "test-token")
    servers = load_config().mcp_servers
    assert [server.name for server in servers] == ["openshift", "aap", "itsm"]
    server = servers[["OPENSHIFT", "AAP", "ITSM"].index(prefix)]
    assert server.url == environment[f"{prefix}_MCP_URL"]
    assert server.tls_verify is (verify == "true")
    assert server.token == "test-token"
    assert "test-token" not in repr(servers)
    transport = create_mcp_servers((server,))[0]
    assert transport.params["headers"] == {"Authorization": "Bearer test-token"}


@pytest.mark.parametrize("token", [None, "", "   "])
def test_optional_tokens(environment, monkeypatch, token):
    for prefix in ("OPENSHIFT", "AAP", "ITSM"):
        monkeypatch.delenv(f"{prefix}_MCP_TOKEN", raising=False)
        monkeypatch.delenv(f"{prefix}_MCP_TLS_VERIFY", raising=False)
        if token is not None:
            monkeypatch.setenv(f"{prefix}_MCP_TOKEN", token)
    for server in load_config().mcp_servers:
        assert server.token == ""
        assert server.tls_verify is True
        assert "headers" not in create_mcp_servers((server,))[0].params


@pytest.mark.parametrize("url", ["", "file:///tmp/mcp", "https://user:secret@host/mcp",
                                 "https://host/mcp?token=secret", "https://host/#secret"])
def test_invalid_mcp_url(environment, monkeypatch, url):
    monkeypatch.setenv("AAP_MCP_URL", url)
    with pytest.raises(ValueError, match="AAP_MCP_URL") as error:
        load_config()
    assert "secret" not in str(error.value)


def test_invalid_tls_setting(environment, monkeypatch):
    monkeypatch.setenv("ITSM_MCP_TLS_VERIFY", "maybe")
    with pytest.raises(ValueError, match="ITSM_MCP_TLS_VERIFY"):
        load_config()


@pytest.mark.parametrize("url", ["http://localhost:8000/mcp", "https://aap.internal.example/mcp"])
def test_http_and_https_urls(environment, monkeypatch, url):
    monkeypatch.setenv("OPENSHIFT_MCP_URL", url)
    config = load_config().mcp_servers[0]
    assert config.url == url
    assert create_mcp_servers((config,))[0].params["url"] == url
