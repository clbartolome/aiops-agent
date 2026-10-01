import logging
from urllib.parse import urlsplit

from agents import (
    Agent, ModelSettings, OpenAIChatCompletionsModel, RunConfig,
    RunContextWrapper, RunHooks, Runner, Session, ToolCallItem, ToolCallOutputItem,
    function_tool,
)
from agents.mcp import MCPServerManager
from openai import AsyncOpenAI

from app.config import MAX_TURNS, Config
from app.diagnostics import ProtocolTextError, log_failure, log_model_response
from app.mcp import MCPConnectionError, create_mcp_servers
from app.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ToolDiagnostics(RunHooks):
    async def on_llm_end(self, context, agent, response) -> None:
        log_model_response(logger, response)

    async def on_tool_start(self, context, agent, tool) -> None:
        logger.info("Tool=%s status=starting", tool.name)


NO_LIVE_DATA = (
    "The requested live system information could not be retrieved and verified. "
    "I cannot provide a factual system-state answer. "
    "Please provide the required system/resource details or try again."
)

INVALID_MODEL_OUTPUT = (
    "The model returned invalid tool-protocol text instead of a usable answer. "
    "Please check the model/provider's tool-calling compatibility."
)


def requires_live_data(message: str) -> bool:
    # Only unambiguous general questions bypass the requirement. Mixed or unknown
    # requests default to requiring tools; no extra model/router is involved.
    general_questions = {
        "hello", "hi", "hey", "thanks", "thank you",
        "what can you do", "what can you check",
        "what is a kubernetes pod", "explain what a kubernetes pod is",
    }
    return " ".join(message.lower().split()).rstrip(".!?") not in general_questions


# This provider's Harmony parser rejects strict tool-schema enforcement.
# The SDK still validates the required question argument locally.
@function_tool(strict_mode=False, failure_error_function=None)
async def request_user_input(question: str) -> str:
    """Ask the user for missing information needed to continue the request."""
    if not question.strip():
        raise ValueError("A clarification question must not be empty.")
    return question


async def run_agent(message: str, config: Config, session: Session | None = None) -> str:
    operational = requires_live_data(message)
    secrets = (config.model_api_key, *(server.token for server in config.mcp_servers))
    logger.info("Model=%s provider_host=%s", config.model_name, urlsplit(config.model_base_url).hostname)
    manager = MCPServerManager(create_mcp_servers(config.mcp_servers))
    logger.info("MCP connecting servers=%s", [server.name for server in config.mcp_servers])
    async with manager, AsyncOpenAI(
        base_url=config.model_base_url, api_key=config.model_api_key
    ) as client:
        if manager.errors:
            for server, error in manager.errors.items():
                log_failure(logger, f"MCP connection failed server={server.name}", error, secrets)
            # Never interpolate exception messages: transports may include credentials.
            failures = ", ".join(
                f"{server.name} ({type(error).__name__})"
                for server, error in manager.errors.items()
            )
            raise MCPConnectionError(f"MCP connection/initialization failed: {failures}")
        logger.info("MCP connected servers=%s", [server.name for server in manager.active_servers])
        agent = Agent(
            name="OperationsAgent",
            instructions=SYSTEM_PROMPT,
            model=OpenAIChatCompletionsModel(
                model=config.model_name, openai_client=client,
                should_replay_reasoning_content=lambda context: False,
            ),
            tools=[request_user_input],
            tool_use_behavior={"stop_at_tool_names": [request_user_input.name]},
            mcp_servers=manager.active_servers,
            mcp_config={"include_server_in_tool_names": True},
            model_settings=ModelSettings(tool_choice="auto"),
        )
        logger.info("MCP tool discovery starting")
        try:
            tools = await agent.get_all_tools(RunContextWrapper(context=None))
        except Exception as error:
            log_failure(logger, "MCP tool discovery failed", error, secrets)
            raise
        logger.info("MCP tools exposed count=%d names=%s", len(tools), [tool.name for tool in tools])
        try:
            result = await Runner.run(
                agent, message, max_turns=MAX_TURNS, hooks=ToolDiagnostics(),
                run_config=RunConfig(tracing_disabled=True), session=session,
            )
        except ProtocolTextError as error:
            log_failure(logger, "Invalid model response", error, secrets)
            final_output = INVALID_MODEL_OUTPUT
        except Exception as error:
            log_failure(logger, "Model/agent request failed", error, secrets)
            raise
        else:
            question_calls = {
                item.raw_item.call_id for item in result.new_items
                if isinstance(item, ToolCallItem)
                and getattr(item.raw_item, "name", None) == request_user_input.name
            }
            outputs = [item for item in result.new_items if isinstance(item, ToolCallOutputItem)]
            questions = [item for item in outputs if item.call_id in question_calls]
            mcp_outputs = [item for item in outputs if item.call_id not in question_calls]
            executed = any((item.custom_data or {}).get("mcp_executed") for item in mcp_outputs)
            attempted = len(mcp_outputs)
            succeeded = sum(
                bool((item.custom_data or {}).get("mcp_executed")
                     and (item.custom_data or {}).get("mcp_success")) for item in mcp_outputs
            )
            failed = attempted - succeeded
            logger.info(
                "MCP calls attempted=%d succeeded=%d failed=%d tool_executed=%s user_input_requested=%s",
                attempted, succeeded, failed, executed, bool(questions),
            )
            if questions:
                final_output = questions[0].output
            elif operational and not succeeded:
                final_output = NO_LIVE_DATA
            else:
                final_output = result.final_output
            if session is not None:
                if questions:
                    # StopAtTools stores the tool exchange in the SDK session.
                    # Add its visible answer there too, for the unchanged UI.
                    await session.add_items([{"role": "assistant", "content": final_output}])
                elif final_output != result.final_output:
                    # Replace the rejected final message in SDK storage so both
                    # future turns and the browser see the same safe answer.
                    await session.pop_item()
                    await session.add_items([{"role": "assistant", "content": final_output}])
    if manager.errors:
        names = ", ".join(server.name for server in manager.errors)
        raise MCPConnectionError(f"MCP cleanup failed: {names}")
    return final_output
