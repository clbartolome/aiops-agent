"""Dedicated Step Executor: executes exactly one procedure step at a time.

The orchestration layer (`app.procedure`) decides, in plain Python, which
step runs and when (e.g. `procedure.steps[0]`, then -- only on SUCCESS --
`procedure.steps[1]`), and may pass already-completed steps' results as
compact context via `previous_results`. This module never decides which
step to execute next; it only executes the one step it is given.

Reuses the same OpenAI Agents SDK + MCP integration already used by the
main `OperationsAgent` (`app.agent`): the same `create_mcp_servers`/
`MCPServerManager` connection, discovery, and cleanup; the same
`ToolDiagnostics` hook; the same `ProtocolTextError` provider-protocol
safety net. No new MCP client, auth, transport, or discovery logic is
introduced here, and this is an internal sub-agent, not the main chat
agent.

Execution happens in two internal phases. This split exists because this
provider does not reliably combine live tool calling with a structured
final-answer schema in a single run: empirically, giving the agent both
MCP tools and a Pydantic `output_type` causes it to skip tool calls and
answer straight from the model's own (unverified) guess, which is exactly
the hallucinated-success failure mode this module must prevent.

    1. "operational" phase: a normal, free-text agentic run with the
       step's own MCP tools available and `tool_choice="auto"` (the same
       shape as `app.agent.run_agent`), restricted to only this step's
       title/instruction/resolved inputs. The model may call MCP tools
       agentically -- it is never told which tool to use -- and may
       recover from a failed call with a corrected one. It produces a
       concise free-text conclusion, never a copy of raw MCP payloads.
    2. "classification" phase: a small, tool-less, structured-output
       model call (the same shape as `app.procedure.extractor`) that
       reads only the step instruction and the phase-1 free-text
       conclusion -- never raw MCP payloads -- and returns the small
       `StepResult(status, summary)`.

Tool-call evidence (attempted/succeeded/failed counts, never raw
payloads) is tracked in plain Python from phase 1's own tool-call output
items, reusing the SDK's existing `custom_data` instrumentation (see
`app.mcp.record_mcp_result`). A self-reported SUCCESS or STOP is rejected
and converted to FAILED if there is no successful live MCP evidence to
support it -- this decision is never left to the model.
"""
import logging

from agents import (
    Agent, AgentOutputSchema, ModelSettings, OpenAIChatCompletionsModel, RunConfig,
    Runner, ToolCallOutputItem,
)
from agents.mcp import MCPServerManager
from openai import AsyncOpenAI

from app.agent import ToolDiagnostics
from app.config import MAX_TURNS, Config
from app.diagnostics import ProtocolTextError, log_failure
from app.mcp import MCPConnectionError, create_mcp_servers
from app.procedure.models import (
    PrimitiveValue, ProcedureInput, ProcedureStep, StepExecutionContext, StepResult,
)

logger = logging.getLogger(__name__)

# Verbatim per spec: the Step Executor must not be told which tool to use,
# must not execute later steps, and must not perform extra remediation.
STEP_EXECUTOR_INSTRUCTIONS = """You execute exactly one step of a fixed IT operations procedure.

Complete only the supplied step.

Use the available MCP tools as needed.

You may make multiple tool calls if required to complete this step.

Do not execute later procedure steps.

Do not perform remediation or additional operations unless explicitly required by the current step.

Use live MCP results for operational facts.

Return a concise result for this step."""

CLASSIFIER_INSTRUCTIONS = """Classify the outcome of one already-executed IT operations procedure step.

You have no tools. Never call a tool and never invent information that is
not present below.

Base your classification only on the step instruction and the result text
given to you.

Return status "SUCCESS" if the step was completed as instructed.
Return status "STOP" only if the step instruction explicitly says to stop
the procedure under some condition, and the result text shows that
condition was observed.
Return status "FAILED" if the step could not be completed or verified.

Return a short, concise summary of the outcome (one or two sentences).
Never copy raw tool output; describe the outcome in your own words."""

FAILED_NO_EVIDENCE_SUMMARY = "Unable to verify this step: no successful tool result was available."
FAILED_INVALID_RESPONSE_SUMMARY = "Unable to complete this step: the model did not return a usable result."
FAILED_PROTOCOL_ERROR_SUMMARY = "Unable to complete this step due to an invalid model response."

# Operational conclusions (SUCCESS/STOP) that must be backed by live MCP evidence.
_EVIDENCE_REQUIRED_STATUSES = ("SUCCESS", "STOP")


class _StepEvidence:
    """Live MCP tool-call evidence for one step run (counts only, no payloads)."""

    __slots__ = ("attempted", "succeeded", "failed")

    def __init__(self, attempted: int, succeeded: int, failed: int):
        self.attempted = attempted
        self.succeeded = succeeded
        self.failed = failed


def _resolved_inputs_block(declared: list[ProcedureInput], collected: dict[str, PrimitiveValue]) -> str:
    lines = [f"{item.label} = {collected[item.name]}" for item in declared if item.name in collected]
    return "\n".join(lines) if lines else "(none)"


def format_previous_step_results(previous_results: list[StepExecutionContext]) -> str:
    """Deterministically format already-completed steps as compact context
    for the next step's prompt (title, status, summary only -- never raw
    MCP payloads). No LLM is involved in producing this text.
    """
    lines = ["Previous completed steps:", ""]
    for item in previous_results:
        lines.append(f"- {item.step.title}")
        lines.append(f"  Status: {item.result.status}")
        lines.append(f"  Summary: {item.result.summary}")
    return "\n".join(lines)


def _operational_prompt(
    step: ProcedureStep, declared: list[ProcedureInput], collected: dict[str, PrimitiveValue],
    original_request: str, previous_results: list[StepExecutionContext],
) -> str:
    lines = [
        f"Step title: {step.title}", "",
        f"Instruction:\n{step.instruction}", "",
        f"Resolved inputs:\n{_resolved_inputs_block(declared, collected)}",
    ]
    if previous_results:
        lines += ["", format_previous_step_results(previous_results)]
    if original_request:
        lines += ["", f"Original procedure request:\n{original_request}"]
    return "\n".join(lines)


async def _run_operational_phase(
    run_id: str, step: ProcedureStep, declared: list[ProcedureInput], collected: dict[str, PrimitiveValue],
    original_request: str, config: Config, previous_results: list[StepExecutionContext],
) -> tuple[str, _StepEvidence]:
    """Run the real, tool-using step agent. Returns (free-text conclusion, evidence).

    Reuses exactly the same MCP connection/discovery/cleanup, hooks, and
    provider-protocol protection as `app.agent.run_agent`.
    """
    secrets = (config.model_api_key, *(server.token for server in config.mcp_servers))
    manager = MCPServerManager(create_mcp_servers(config.mcp_servers))
    async with manager, AsyncOpenAI(base_url=config.model_base_url, api_key=config.model_api_key) as client:
        if manager.errors:
            for server, error in manager.errors.items():
                log_failure(logger, f"Procedure run={run_id} step={step.id} MCP connection failed "
                                     f"server={server.name}", error, secrets)
            failures = ", ".join(
                f"{server.name} ({type(error).__name__})" for server, error in manager.errors.items()
            )
            raise MCPConnectionError(f"MCP connection/initialization failed: {failures}")

        agent = Agent(
            name="ProcedureStepExecutor",
            instructions=STEP_EXECUTOR_INSTRUCTIONS,
            model=OpenAIChatCompletionsModel(
                model=config.model_name, openai_client=client,
                should_replay_reasoning_content=lambda context: False,
            ),
            mcp_servers=manager.active_servers,
            mcp_config={"include_server_in_tool_names": True},
            model_settings=ModelSettings(tool_choice="auto"),
        )
        prompt = _operational_prompt(step, declared, collected, original_request, previous_results)
        try:
            result = await Runner.run(
                agent, prompt, max_turns=MAX_TURNS, hooks=ToolDiagnostics(),
                run_config=RunConfig(tracing_disabled=True),
            )
        except ProtocolTextError as error:
            log_failure(logger, f"Procedure run={run_id} step={step.id} invalid model response", error, secrets)
            return "", _StepEvidence(attempted=0, succeeded=0, failed=0)
        except Exception as error:
            log_failure(logger, f"Procedure run={run_id} step={step.id} execution request failed", error, secrets)
            raise

        outputs = [item for item in result.new_items if isinstance(item, ToolCallOutputItem)]
        attempted = len(outputs)
        succeeded = sum(
            bool((item.custom_data or {}).get("mcp_executed") and (item.custom_data or {}).get("mcp_success"))
            for item in outputs
        )
        evidence = _StepEvidence(attempted=attempted, succeeded=succeeded, failed=attempted - succeeded)
        conclusion = result.final_output if isinstance(result.final_output, str) else ""

    if manager.errors:
        names = ", ".join(server.name for server in manager.errors)
        raise MCPConnectionError(f"MCP cleanup failed: {names}")
    return conclusion, evidence


async def _classify_result(step: ProcedureStep, conclusion: str, config: Config) -> StepResult:
    """Classify the operational phase's free-text conclusion into a `StepResult`.

    A small, tool-less, structured-output model call (the same shape as
    `app.procedure.extractor`); it only ever sees the step instruction and
    the already-concise free-text conclusion -- never raw MCP payloads.
    """
    prompt = f"Step instruction:\n{step.instruction}\n\nResult text:\n{conclusion}"
    async with AsyncOpenAI(base_url=config.model_base_url, api_key=config.model_api_key) as client:
        agent = Agent(
            name="ProcedureStepClassifier",
            instructions=CLASSIFIER_INSTRUCTIONS,
            model=OpenAIChatCompletionsModel(
                model=config.model_name, openai_client=client,
                should_replay_reasoning_content=lambda context: False,
            ),
            tools=[],
            mcp_servers=[],
            output_type=AgentOutputSchema(StepResult, strict_json_schema=False),
            model_settings=ModelSettings(tool_choice="none"),
        )
        result = await Runner.run(agent, prompt, max_turns=1, run_config=RunConfig(tracing_disabled=True))

    output = result.final_output
    if isinstance(output, StepResult):
        return output
    return StepResult(status="FAILED", summary=FAILED_INVALID_RESPONSE_SUMMARY)


async def execute_step(
    run_id: str, step: ProcedureStep, declared: list[ProcedureInput], collected: dict[str, PrimitiveValue],
    original_request: str, config: Config, previous_results: list[StepExecutionContext] | None = None,
) -> StepResult:
    """Execute exactly one procedure step and return a validated `StepResult`.

    The caller always supplies `step` (e.g. `procedure.steps[0]`, or
    `procedure.steps[1]` once step 1 succeeds); this function never
    selects which step to run and never executes more than one. Tool
    selection within the step is fully agentic -- the model is never told
    which MCP tool to call.

    `previous_results`, when given, carries already-completed steps as
    compact context (title/status/summary only); it never changes which
    step is executed, only what context this one step sees.

    A self-reported SUCCESS/STOP without any successful live MCP evidence
    is rejected and converted to FAILED; this is decided here in plain
    Python, never by the model.
    """
    previous_results = previous_results or []
    logger.info(
        "Procedure run=%s step=%s previous_results=%d status=starting", run_id, step.id, len(previous_results),
    )
    conclusion, evidence = await _run_operational_phase(
        run_id, step, declared, collected, original_request, config, previous_results,
    )

    if not conclusion:
        result = StepResult(status="FAILED", summary=FAILED_PROTOCOL_ERROR_SUMMARY)
    else:
        result = await _classify_result(step, conclusion, config)
        if result.status in _EVIDENCE_REQUIRED_STATUSES and evidence.succeeded == 0:
            logger.info("Procedure run=%s step=%s unsupported_status=%s evidence_rejected=true",
                        run_id, step.id, result.status)
            result = StepResult(status="FAILED", summary=FAILED_NO_EVIDENCE_SUMMARY)

    logger.info(
        "Procedure run=%s step=%s status=%s mcp_attempted=%d mcp_succeeded=%d mcp_failed=%d",
        run_id, step.id, result.status, evidence.attempted, evidence.succeeded, evidence.failed,
    )
    return result
