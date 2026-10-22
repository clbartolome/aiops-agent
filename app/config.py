import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


MAX_TURNS = 5


DEFAULT_MCP_ALLOWED_TOOLS = {
    "openshift": frozenset(f"mcp_openshift__{name}" for name in (
        "namespaces_list", "projects_list", "pods_get", "pods_list", "pods_list_in_namespace",
        "pods_log", "resources_get", "resources_list",
    )),
    "aap": frozenset(f"mcp_aap__{name}" for name in (
        "job_templates_list", "job_templates_retrieve", "job_templates_launch_create",
        "jobs_list", "jobs_retrieve", "jobs_stdout_retrieve", "workflow_job_templates_list",
        "workflow_job_templates_retrieve", "workflow_job_templates_launch_create",
        "workflow_jobs_list", "workflow_jobs_retrieve",
    )),
    "itsm": frozenset(f"mcp_itsm__{name}" for name in (
        "rag_search_kb", "search_kb", "get_kb_article", "list_incidents", "get_incident",
        "list_request_templates", "get_request_template", "list_requests", "get_request", "create_request",
    )),
}


def parse_allowed_tools(value: str) -> frozenset[str]:
    return frozenset(entry.strip() for entry in value.split(",") if entry.strip())


@dataclass(frozen=True)
class MCPConfig:
    name: str
    url: str
    token: str | None = field(default="", repr=False)
    tls_verify: bool = True
    # None means demo defaults; an explicit empty set means expose nothing.
    allowed_tools: frozenset[str] | None = None

    @property
    def effective_allowed_tools(self) -> frozenset[str]:
        if self.allowed_tools is not None:
            return self.allowed_tools
        return DEFAULT_MCP_ALLOWED_TOOLS.get(self.name, frozenset())


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
        allowed_tools=(parse_allowed_tools(os.environ[f"{prefix}_MCP_ALLOWED_TOOLS"])
                       if f"{prefix}_MCP_ALLOWED_TOOLS" in os.environ else None),
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
