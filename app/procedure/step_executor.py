"""The real procedure Step Executor.

A single bounded Agents SDK run — using the existing OpenAI Agents SDK and
MCP server infrastructure, unchanged — that completes exactly one
`ProcedureStep` and returns a structured `StepResult`.

LangGraph (`app.procedure.runtime`) owns the procedure workflow: which step
runs, in what order, pause/resume, confirmation, and what happens after this
step returns. This module answers only "can this exact step be completed
using the MCP tools already configured for the OperationsAgent?" — it never
decides to skip, insert, reorder, or retry a step, and it never starts a new
conversational clarification workflow (required inputs are already resolved
by LangGraph before this is ever called).

One bounded `Runner.run(...)` call per step: this is not a long-lived
autonomous procedure agent that receives all steps at once, and it is not the
main chat interlocutor (`OperationsAgent` keeps that role).
"""
import logging
from typing import Any

from agents import (
    Agent, AgentOutputSchema, ModelSettings, OpenAIChatCompletionsModel,
    RunConfig, RunHooks, Runner, ToolCallOutputItem,
)
from agents.mcp import MCPServerManager
from openai import AsyncOpenAI

from app.config import MAX_TURNS, Config
from app.diagnostics import ProtocolTextError, log_failure, log_model_response
from app.mcp import MCPConnectionError, create_mcp_servers, mcp_tool_error
from app.procedure.models import ProcedureStep, StepResult
from app.procedure.risk import (
    READ,
    ApprovalRequired,
    DuplicateToolCallError,
    ScopeViolationError,
    classify_tool_risk,
    describe_operation,
    fingerprint_call,
    is_in_scope,
)
from app.procedure.runtime import StepExecutor

logger = logging.getLogger(__name__)

# Deliberately separate from `app.prompts.SYSTEM_PROMPT`: this agent executes
# one fixed step, not an open-ended conversation, and must not plan, invent,
# or alter the procedure, ask the user anything, or act beyond the step.
STEP_EXECUTOR_INSTRUCTIONS = """You execute exactly one step of a fixed IT operations procedure.

Complete only the supplied step. Do not perform remediation unless the step requires it. Do not
execute an operation merely because it may be useful. Do not execute later procedure steps. Do not
invent procedure steps. Do not alter the procedure.

Prefer the least invasive tool operation that satisfies the step. For example, to verify whether
something exists, use a read/list/get/search operation; do not create, change, or delete anything
to find that out.

Use the available MCP tools as needed. You may make multiple tool calls if required to complete the step.

Do not perform remediation or additional actions (for example restarting, deleting, or scaling a
resource) unless explicitly required by the current step's instruction.

Some tool calls require a separate approval before they run, and a change that already succeeded
will not be repeated. If a tool call is rejected or blocked, do not retry the exact same call;
choose a different, in-scope approach, or report that the step could not be completed.

Use only live MCP tool results for operational facts. Never report that a resource exists, is
running, or that an action succeeded without a successful MCP tool result confirming it.

If the step instruction says the procedure should stop under a condition and that condition is
observed, return outcome STOP. If the step cannot be completed, return outcome FAILED. Otherwise
return outcome SUCCESS.

If completing the step would require information you were not given, return outcome FAILED
describing what is missing. Do not ask a question.
"""

UNVERIFIED_OUTCOME_SUMMARY = (
    "The step reported a result without a successful MCP tool result to support it."
)
INVALID_MODEL_OUTPUT_SUMMARY = (
    "The model returned invalid tool-protocol text instead of a usable result."
)
EXECUTOR_ERROR_SUMMARY = "The step could not be completed due to an unexpected error."
INVALID_STRUCTURED_OUTPUT_SUMMARY = "The step executor returned an invalid structured result."


class _StepExecutorHooks(RunHooks):
    async def on_llm_end(self, context, agent, response) -> None:
        log_model_response(logger, response)


def _format_step_prompt(
    step: ProcedureStep, inputs: dict[str, Any], previous_results: dict[str, StepResult] | None,
) -> str:
    """The model receives only the current step's own wording plus already-
    resolved values. It never receives the full KB article and is never
    asked to reinterpret or re-plan the procedure.
    """
    lines = [f"Step title:\n{step.title}", "", f"Step instruction:\n{step.instruction}"]
    if inputs:
        lines += ["", "Resolved inputs:"]
        lines += [f"{name} = {value}" for name, value in inputs.items()]
    if previous_results:
        lines += ["", "Previous step results:"]
        for step_id, result in previous_results.items():
            entry = f"- {step_id}: outcome={result.outcome}, summary={result.summary}"
            if result.data:
                entry += f", data={result.data}"
            lines.append(entry)
    return "\n".join(lines)


def _tool_call_counts(result) -> tuple[int, int]:
    outputs = [item for item in result.new_items if isinstance(item, ToolCallOutputItem)]
    attempted = len(outputs)
    succeeded = sum(
        bool((item.custom_data or {}).get("mcp_executed") and (item.custom_data or {}).get("mcp_success"))
        for item in outputs
    )
    return attempted, succeeded


_BLOCKED_STATUSES = ("failed", "blocked_scope", "blocked_duplicate")

# Safe, specific, model-visible messages for the two tool-call-boundary
# rejections that are meant to be recoverable within the same step (unlike
# `ApprovalRequired`, which must pause the run instead). Deliberately do not
# describe the tool's arguments.
_OUT_OF_SCOPE_MESSAGE = (
    "This operation is outside the scope of the current step and was not executed. "
    "Only perform actions required by the current step's instruction."
)
_DUPLICATE_BLOCKED_MESSAGE = (
    "This exact operation already completed successfully earlier in this step. "
    "It was not executed again."
)


class _ToolCallGate:
    """The tool-call-boundary policy for exactly one Step Executor attempt:
    bounded scope, risk classification plus approval, and duplicate
    side-effect protection. Not a tool binder or capability registry: it
    decides nothing about which tool to call, only whether a proposed call
    may proceed.
    """

    def __init__(self, step: ProcedureStep, approved_calls: frozenset[str]):
        self.step = step
        self._approved_calls = approved_calls
        self._executed_risky_fingerprints: set[str] = set()
        self.audit: list[dict[str, str]] = []

    def before_call(self, tool, arguments: dict) -> tuple[str, str]:
        """Returns `(risk, fingerprint)` to allow the call, or raises."""
        risk = classify_tool_risk(tool)
        fingerprint = fingerprint_call(tool.name, arguments)
        if risk == READ:
            return risk, fingerprint
        if fingerprint in self._executed_risky_fingerprints:
            self._record(tool.name, risk, "granted", "blocked_duplicate")
            raise DuplicateToolCallError(tool_name=tool.name)
        if not is_in_scope(self.step, tool, risk):
            self._record(tool.name, risk, "not_required", "blocked_scope")
            raise ScopeViolationError(tool_name=tool.name, step_title=self.step.title)
        if fingerprint not in self._approved_calls:
            self._record(tool.name, risk, "pending", "approval_required")
            raise ApprovalRequired(
                tool_name=tool.name, risk=risk,
                operation=describe_operation(tool), fingerprint=fingerprint,
            )
        return risk, fingerprint

    def after_success(self, tool, risk: str, fingerprint: str) -> None:
        if risk != READ:
            self._executed_risky_fingerprints.add(fingerprint)
        self._record(tool.name, risk, "not_required" if risk == READ else "granted", "succeeded")

    def after_failure(self, tool, risk: str) -> None:
        self._record(tool.name, risk, "not_required" if risk == READ else "granted", "failed")

    def _record(self, tool_name: str, risk: str, approval: str, status: str) -> None:
        self.audit.append({"tool": tool_name, "risk": risk, "approval": approval, "status": status})
        # Tool name, risk, approval decision, and outcome only: never
        # argument values, tool-call text, or model reasoning.
        logger.info(
            "Procedure step_id=%s tool=%s risk=%s approval=%s status=%s",
            self.step.id, tool_name, risk, approval, status,
        )


def _wrap_call_tool(server, gate: _ToolCallGate, tools: list) -> None:
    """Intercept exactly this server's `call_tool`, at the same granularity
    the Agents SDK itself dispatches tool calls, so every proposed MCP tool
    call (from any server) passes through the gate before the real MCP
    tool ever runs.
    """
    tool_lookup = {tool.name: tool for tool in tools}
    original_call_tool = server.call_tool

    async def call_tool(tool_name: str, arguments: dict, *args, **kwargs):
        tool = tool_lookup.get(tool_name)
        if tool is None:  # pragma: no cover - defensive: only discovered tools are ever proposed
            return await original_call_tool(tool_name, arguments, *args, **kwargs)
        risk, fingerprint = gate.before_call(tool, arguments or {})
        try:
            result = await original_call_tool(tool_name, arguments, *args, **kwargs)
        except Exception:
            gate.after_failure(tool, risk)
            raise
        gate.after_success(tool, risk, fingerprint)
        return result

    server.call_tool = call_tool


def _make_failure_error_function(secrets):
    """Extends the shared default MCP failure formatter (`app.mcp.mcp_tool_error`)
    with the Step Executor's own tool-call-boundary signals, without
    changing that default for any other caller.

    Returning `None` tells the Agents SDK not to convert the exception into
    a model-visible string — it propagates instead, which is exactly what
    `ApprovalRequired` needs in order to reach `app.procedure.runtime` as a
    LangGraph pause rather than a tool error the agent could try to retry.
    """
    def failure_error_function(context, error: Exception):
        if isinstance(error, ApprovalRequired):
            return None
        if isinstance(error, ScopeViolationError):
            return _OUT_OF_SCOPE_MESSAGE
        if isinstance(error, DuplicateToolCallError):
            return _DUPLICATE_BLOCKED_MESSAGE
        return mcp_tool_error(context, error, secrets=secrets)

    return failure_error_function


def _log_audit_summary(step: ProcedureStep, gate: _ToolCallGate) -> None:
    attempted = len(gate.audit)
    succeeded = sum(1 for entry in gate.audit if entry["status"] == "succeeded")
    failed = sum(1 for entry in gate.audit if entry["status"] in _BLOCKED_STATUSES)
    logger.info(
        "Procedure step_id=%s tools_attempted=%d tools_succeeded=%d tools_failed=%d",
        step.id, attempted, succeeded, failed,
    )


def make_step_executor(config: Config) -> StepExecutor:
    """Build a Step Executor bound to this request's model/MCP configuration.

    Reuses the exact same MCP server construction (`create_mcp_servers`) and
    Agents SDK model wiring as the direct `OperationsAgent` path: no new MCP
    client, auth, TLS, or transport code, and no duplicate capability
    registry/tool binder.
    """
    secrets = (config.model_api_key, *(server.token for server in config.mcp_servers))

    async def execute_step(
        step: ProcedureStep, inputs: dict[str, Any], previous_results: dict[str, StepResult] | None = None,
        *, approved_calls: frozenset[str] = frozenset(),
    ) -> StepResult:
        manager = MCPServerManager(create_mcp_servers(
            config.mcp_servers, failure_error_function=_make_failure_error_function(secrets),
        ))
        gate = _ToolCallGate(step, approved_calls)
        logger.info("Procedure step_id=%s executor=started", step.id)
        async with manager, AsyncOpenAI(
            base_url=config.model_base_url, api_key=config.model_api_key
        ) as client:
            if manager.errors:
                for server, error in manager.errors.items():
                    log_failure(logger, f"MCP connection failed server={server.name}", error, secrets)
                failures = ", ".join(
                    f"{server.name} ({type(error).__name__})" for server, error in manager.errors.items()
                )
                raise MCPConnectionError(f"MCP connection/initialization failed: {failures}")

            for server in manager.active_servers:
                _wrap_call_tool(server, gate, await server.list_tools())

            agent = Agent(
                name="ProcedureStepExecutor",
                instructions=STEP_EXECUTOR_INSTRUCTIONS,
                model=OpenAIChatCompletionsModel(
                    model=config.model_name, openai_client=client,
                    should_replay_reasoning_content=lambda context: False,
                ),
                # No local tools (notably no `request_user_input`): pause/resume for
                # missing information stays owned by LangGraph, never by this agent.
                mcp_servers=manager.active_servers,
                mcp_config={"include_server_in_tool_names": True},
                model_settings=ModelSettings(tool_choice="auto"),
                output_type=AgentOutputSchema(StepResult, strict_json_schema=False),
            )
            prompt = _format_step_prompt(step, inputs, previous_results)
            try:
                try:
                    result = await Runner.run(
                        agent, prompt, max_turns=MAX_TURNS, hooks=_StepExecutorHooks(),
                        run_config=RunConfig(tracing_disabled=True),
                    )
                except ProtocolTextError as error:
                    log_failure(logger, f"Invalid model response step={step.id}", error, secrets)
                    return StepResult(
                        outcome="FAILED", summary=INVALID_MODEL_OUTPUT_SUMMARY,
                        error="protocol_in_assistant_text",
                    )
                except ApprovalRequired:
                    # Not a failure: a tool call needs explicit approval before
                    # it can run. `app.procedure.runtime` turns this into a
                    # LangGraph pause and, on resume, re-attempts this exact
                    # step with the approval recorded.
                    raise
                except Exception as error:
                    log_failure(logger, f"Step executor failed step={step.id}", error, secrets)
                    raise

                attempted, succeeded = _tool_call_counts(result)
                logger.info(
                    "Procedure step_id=%s mcp_attempted=%d mcp_succeeded=%d mcp_failed=%d",
                    step.id, attempted, succeeded, attempted - succeeded,
                )

                step_result = result.final_output
                if not isinstance(step_result, StepResult):  # pragma: no cover - defensive
                    return StepResult(outcome="FAILED", summary=INVALID_STRUCTURED_OUTPUT_SUMMARY,
                                      error="invalid_structured_output")

                if step_result.outcome in ("SUCCESS", "STOP") and succeeded == 0:
                    # Never let the model claim an operational fact that no live MCP
                    # tool result actually confirmed.
                    logger.info(
                        "Procedure step_id=%s executor=rejected_unverified_outcome claimed=%s",
                        step.id, step_result.outcome,
                    )
                    return StepResult(outcome="FAILED", summary=UNVERIFIED_OUTCOME_SUMMARY,
                                       error="no_live_mcp_evidence")
                return step_result
            finally:
                _log_audit_summary(step, gate)

    return execute_step
