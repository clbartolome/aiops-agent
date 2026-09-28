import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


MAX_TURNS = 5


@dataclass(frozen=True)
class MCPConfig:
    name: str
    url: str
    token: str | None = field(default="", repr=False)
    tls_verify: bool = True


@dataclass(frozen=True)
class Config:
    model_name: str
    model_base_url: str
    model_api_key: str = field(repr=False)
    mcp_servers: tuple[MCPConfig, ...] = ()


def load_mcp_config(prefix: str, name: str) -> MCPConfig:
    url = os.environ.get(f"{prefix}_MCP_URL", "").strip()
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme in ("http", "https") and parsed.hostname
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{prefix}_MCP_URL must be an HTTP(S) URL without credentials, query, or fragment")
    verify = os.environ.get(f"{prefix}_MCP_TLS_VERIFY", "true").strip().lower()
    if verify not in ("true", "false"):
        raise ValueError(f"{prefix}_MCP_TLS_VERIFY must be true or false")
    return MCPConfig(
        name=name,
        url=url,
        token=os.environ.get(f"{prefix}_MCP_TOKEN", "").strip(),
        tls_verify=verify == "true",
    )


def load_config() -> Config:
    values = {}
    for name in ("MODEL_NAME", "MODEL_BASE_URL", "MODEL_API_KEY"):
        value = os.environ.get(name, "").strip()
        if not value:
            raise ValueError(f"{name} must be set")
        values[name] = value
    return Config(
        values["MODEL_NAME"], values["MODEL_BASE_URL"], values["MODEL_API_KEY"],
        mcp_servers=(
            load_mcp_config("OPENSHIFT", "openshift"),
            load_mcp_config("AAP", "aap"),
            load_mcp_config("ITSM", "itsm"),
        ),
    )
