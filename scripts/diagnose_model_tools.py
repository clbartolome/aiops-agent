"""Temporary tool-set comparisons; never changes the web application's tools.

Run as: python -m scripts.diagnose_model_tools --case mcp-only
Load the same environment as make run-app before running this command.
"""
import argparse
import asyncio
from copy import deepcopy
from dataclasses import replace
import logging
from unittest.mock import patch

from agents import Agent
from agents.strict_schema import ensure_strict_json_schema
from openai import AsyncOpenAI

from app import agent as agent_module
from app.config import load_config
from app.diagnostics import configure_logging

logger = logging.getLogger("app.tool_diagnosis")
CASES = ("all", "mcp-only", "single-server", "single-mcp", "local-only",
         "single-mixed", "non-strict-local", "all-non-strict-local", "single-strict-mcp",
         "strict-local", "all-strict-local")


def select_tools(tools, case, server_name):
    local = [tool for tool in tools if tool.name == "request_user_input"]
    mcp = [tool for tool in tools if tool.name != "request_user_input"]
    server = [tool for tool in mcp if tool.name.startswith(f"mcp_{server_name}__")]
    # Prefer a pod listing tool for the read-only diagnostic request.
    single = next((tool for tool in server if "pods_list" in tool.name),
                  server[0] if server else None)
    one = [single] if single is not None else []
    non_strict = [replace(tool, strict_json_schema=False) for tool in local]
    if case in ("single-strict-mcp", "strict-local", "all-strict-local"):
        selected = one if case == "single-strict-mcp" else local
        strict = [replace(tool, strict_json_schema=True,
                          params_json_schema=ensure_strict_json_schema(deepcopy(tool.params_json_schema)))
                  for tool in selected]
        return mcp + strict if case == "all-strict-local" else strict
    return {
        "all": tools, "mcp-only": mcp, "single-server": server,
        "single-mcp": one, "local-only": local, "single-mixed": one + local,
        "non-strict-local": non_strict, "all-non-strict-local": mcp + non_strict,
    }[case]


def log_request_tools(definitions):
    logger.info("Model request tools count=%d", len(definitions))
    for definition in definitions:
        function = definition["function"]
        schema = function.get("parameters", {})
        logger.info(
            "Tool name=%s origin=%s strict=%s schema_type=%s properties=%d required=%d",
            function["name"],
            "local" if function["name"] == "request_user_input" else "MCP",
            function.get("strict"), schema.get("type"),
            len(schema.get("properties", {})), len(schema.get("required", [])),
        )


async def diagnose(case, message):
    config = load_config()
    get_tools = Agent.get_all_tools
    request_count = 0
    server_name = config.mcp_servers[0].name

    async def filtered_tools(self, context):
        tools = await get_tools(self, context)
        return select_tools(tools, case, server_name)

    def client(**kwargs):
        instance = AsyncOpenAI(**kwargs, max_retries=0, timeout=60)
        create = instance.chat.completions.create

        async def logged_create(*args, **options):
            nonlocal request_count
            request_count += 1
            log_request_tools(options.get("tools", []))
            return await create(*args, **options)

        instance.chat.completions.create = logged_create
        return instance

    logger.info("Diagnostic case=%s model=%s", case, config.model_name)
    with patch.object(Agent, "get_all_tools", filtered_tools), patch.object(agent_module, "AsyncOpenAI", client):
        try:
            result = await agent_module.run_agent(message, config)
        except Exception as error:
            # run_agent already logs credential-safe error details.
            logger.info("Diagnostic case=%s outcome=request_failed error=%s model_requests=%d",
                        case, type(error).__name__, request_count)
            return False
        logger.info("Diagnostic case=%s outcome=%s model_requests=%d", case,
                    "controlled_response" if result in (agent_module.NO_LIVE_DATA, agent_module.INVALID_MODEL_OUTPUT)
                    else "answer_or_question", request_count)
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--message", default="How many pods are in namespace openshift-ingress?")
    args = parser.parse_args()
    configure_logging()
    asyncio.run(diagnose(args.case, args.message))
