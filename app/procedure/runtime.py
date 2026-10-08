"""LangGraph runtime for fixed, deterministically-parsed procedures.

LangGraph is the deterministic procedure runtime. It alone controls:
procedure state, required-input collection, pause/resume, the current step,
step ordering, procedure-level confirmation, step outcomes, completion,
failure, and cancellation.

The injected Step Executor contract (`StepExecutor`) only answers "given
this exact step, its resolved inputs, and prior step results, can you
complete it?" — it never controls sequencing. The real Agents SDK/MCP Step
Executor lives in `app.procedure.step_executor`; `placeholder_step_executor`
below is only a safe fallback/test default.

No LLM is involved in any control-flow decision in this module.
"""
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.procedure.models import ProcedureDefinition, ProcedureInput, ProcedureStep, StepResult
from app.procedure.risk import ApprovalRequired

logger = logging.getLogger(__name__)

# --- Statuses ----------------------------------------------------------------
# A small, explicit, fixed set. Do not add more states unless clearly required.
PENDING = "PENDING"
WAITING_FOR_INPUT = "WAITING_FOR_INPUT"
WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"

_TERMINAL_STATUSES = (COMPLETED, FAILED, CANCELLED)

# `StepExecutor`s also accept a keyword-only `approved_calls: frozenset[str]`
# (fingerprints of tool calls already approved for this step's remaining
# attempts); `Callable` cannot express keyword-only parameters, so this
# signature is approximate.
StepExecutor = Callable[
    [ProcedureStep, dict[str, Any], "dict[str, StepResult] | None"], Awaitable[StepResult]
]


class ProcedureRunNotFound(KeyError):
    """No active run exists for the given run_id (unknown, or already finished)."""


class ProcedureRunState(TypedDict):
    """The complete LangGraph-checkpointed state for one procedure run.

    Deliberately small: it never holds chat history (that stays in the
    OpenAI Agents SDK session) and never holds the `ProcedureDefinition`
    itself (that is supplied per-call via `config["configurable"]`, since it
    is immutable for the run and does not need to be replayed/checkpointed).
    """
    run_id: str
    procedure_id: str
    procedure_version: int
    status: str
    inputs: dict[str, Any]
    current_step_index: int
    step_results: dict[str, Any]
    final_message: str | None
    # A risky tool call this step is waiting on explicit approval for (set
    # by `_execute_current_step_node`, consumed by `_await_tool_approval_node`),
    # or None when nothing is pending.
    pending_approval: dict[str, Any] | None
    # Fingerprints of tool calls already approved for the current step's
    # remaining attempts. Cleared whenever the step finishes (any outcome)
    # or advances; never carried across steps.
    approved_tool_calls: list[str]
    # Append-only record of every approval requested/decided during this
    # run (never cleared), for the run-level audit summary only (never
    # shown to the user by default): `{"step_id", "risk", "decision"}` with
    # `decision` in `requested`/`granted`/`rejected`.
    approval_log: list[dict[str, str]]


async def placeholder_step_executor(
    step: ProcedureStep, inputs: dict[str, Any], previous_results: dict[str, StepResult] | None = None,
) -> StepResult:
    """A safe fallback Step Executor. Always reports that the step itself
    could not be completed, so a run fails in a controlled way instead of
    silently pretending to execute anything. `ProcedureRuntime`'s production
    default is overridden with the real Agents SDK/MCP executor (see
    `app.procedure.step_executor.make_step_executor`); this remains the
    class-level default for direct/defensive use and for runtime unit tests.
    """
    return StepResult(
        outcome="FAILED",
        summary="Step execution is not implemented yet.",
        error="placeholder_step_executor",
    )


@dataclass
class ProcedureOutcome:
    """What callers (the chat/procedure integration layer) see.

    Never exposes LangGraph internals (thread IDs, checkpoints, interrupt
    objects): just a status, a user-safe message, and the structural data a
    caller needs to continue (e.g. which fields are being requested).
    """
    run_id: str
    status: str
    message: str
    step_results: dict[str, Any] = field(default_factory=dict)
    waiting_fields: list[dict[str, str | None]] = field(default_factory=list)
    # An internal structured summary (procedure/run identifiers, final
    # status, step outcomes, approval counts), populated only once a run
    # reaches a terminal status. For diagnostics/evaluation only: never
    # rendered into `message`, and never sent to the user by default.
    audit: dict[str, Any] | None = None


@dataclass
class _RunContext:
    definition: ProcedureDefinition
    step_executor: StepExecutor
    last_outcome: ProcedureOutcome | None = None


def _missing_required_inputs(definition: ProcedureDefinition, inputs: dict[str, Any]) -> list[ProcedureInput]:
    return [item for item in definition.inputs if item.required and item.name not in inputs]


def _apply_defaults(definition: ProcedureDefinition, inputs: dict[str, Any]) -> None:
    # Declared defaults are applied deterministically; neither the Step
    # Executor nor an LLM is ever asked to invent one.
    for item in definition.inputs:
        if item.name not in inputs and item.default is not None:
            inputs[item.name] = item.default


def _resolve_step_inputs(step: ProcedureStep, inputs: dict[str, Any]) -> dict[str, Any]:
    # Only pass values the step's instruction actually references.
    return {name: inputs[name] for name in step.input_refs if name in inputs}


def _collect_inputs_node(state: ProcedureRunState, config: RunnableConfig) -> dict:
    definition: ProcedureDefinition = config["configurable"]["definition"]
    # Everything here is pure/deterministic: safe to replay before `interrupt()`.
    inputs = dict(state["inputs"])
    _apply_defaults(definition, inputs)
    missing = _missing_required_inputs(definition, inputs)
    while missing:
        payload = {
            "type": "input_required",
            "run_id": state["run_id"],
            "fields": [
                {"name": item.name, "label": item.label, "description": item.description}
                for item in missing
            ],
        }
        resumed = interrupt(payload)
        if not isinstance(resumed, dict):
            raise TypeError("Resume value for input collection must be a mapping of input values")
        for item in missing:
            if item.name in resumed:
                inputs[item.name] = resumed[item.name]
        missing = _missing_required_inputs(definition, inputs)
    return {"inputs": inputs}


def _confirm_if_required_node(state: ProcedureRunState, config: RunnableConfig) -> dict:
    definition: ProcedureDefinition = config["configurable"]["definition"]
    if not definition.confirmation_required:
        return {}
    approved = interrupt({
        "type": "confirmation_required",
        "run_id": state["run_id"],
        "procedure_id": definition.id,
        "title": definition.title,
        "risk": definition.risk,
        "step_titles": [step.title for step in definition.steps],
    })
    if not approved:
        return {"status": CANCELLED, "final_message": "Procedure cancelled by the user."}
    return {}


def _route_after_confirmation(state: ProcedureRunState) -> str:
    return END if state["status"] == CANCELLED else "execute_current_step"


async def _execute_current_step_node(state: ProcedureRunState, config: RunnableConfig) -> dict:
    definition: ProcedureDefinition = config["configurable"]["definition"]
    step_executor: StepExecutor = config["configurable"]["step_executor"]
    step = definition.steps[state["current_step_index"]]
    resolved_inputs = _resolve_step_inputs(step, state["inputs"])
    # Structured results from steps already executed in this run, keyed by
    # step ID; never the raw chat/conversation history.
    previous_results = {
        step_id: StepResult.model_validate(data) for step_id, data in state["step_results"].items()
    }
    approved_calls = frozenset(state["approved_tool_calls"])

    logger.info(
        "Procedure procedure_run_id=%s procedure_id=%s step_id=%s status=starting",
        state["run_id"], definition.id, step.id,
    )
    try:
        result = await step_executor(step, resolved_inputs, previous_results, approved_calls=approved_calls)
    except ApprovalRequired as approval:
        # A WRITE/DESTRUCTIVE/UNKNOWN tool call was proposed with no
        # matching approval yet. This is not a step failure: it pauses the
        # run (via `_await_tool_approval_node`) and, once approved, retries
        # this exact step with that one call now authorized. The tool was
        # never executed.
        logger.info(
            "Procedure procedure_run_id=%s procedure_id=%s step_id=%s tool=%s risk=%s status=approval_requested",
            state["run_id"], definition.id, step.id, approval.tool_name, approval.risk,
        )
        return {
            "status": RUNNING,
            "pending_approval": {
                "step_id": step.id, "step_title": step.title,
                "risk": approval.risk, "operation": approval.operation,
                "fingerprint": approval.fingerprint,
            },
            "approval_log": [*state["approval_log"], {
                "step_id": step.id, "risk": approval.risk, "decision": "requested",
            }],
        }
    except Exception as error:
        # The Step Executor itself failed unexpectedly: a controlled FAILED
        # outcome, never a silent continuation and never a fallback to the
        # normal OperationsAgent.
        result = StepResult(
            outcome="FAILED",
            summary="The step could not be completed due to an unexpected error.",
            error=type(error).__name__,
        )
    logger.info(
        "Procedure procedure_run_id=%s procedure_id=%s step_id=%s status=%s",
        state["run_id"], definition.id, step.id, result.outcome,
    )

    step_results = dict(state["step_results"])
    step_results[step.id] = result.model_dump()
    return {
        "status": RUNNING, "step_results": step_results,
        "pending_approval": None, "approved_tool_calls": [],
    }


def _route_after_execute_step(state: ProcedureRunState) -> str:
    return "await_tool_approval" if state.get("pending_approval") else "handle_step_result"


def _await_tool_approval_node(state: ProcedureRunState, config: RunnableConfig) -> dict:
    pending = state["pending_approval"]
    decision = interrupt({
        "type": "tool_approval_required",
        "run_id": state["run_id"],
        "step_id": pending["step_id"],
        "step_title": pending["step_title"],
        "risk": pending["risk"],
        "operation": pending["operation"],
    })
    approval_log = [*state["approval_log"], {
        "step_id": pending["step_id"], "risk": pending["risk"],
        "decision": "granted" if decision else "rejected",
    }]
    logger.info(
        "Procedure procedure_run_id=%s step_id=%s status=approval_%s",
        state["run_id"], pending["step_id"], "granted" if decision else "rejected",
    )
    if not decision:
        # The safer interpretation for the prototype: an explicit rejection
        # cancels the run rather than failing it, and the tool is never
        # executed.
        return {
            "status": CANCELLED, "pending_approval": None, "approval_log": approval_log,
            "final_message": _approval_rejected_message(pending),
        }
    return {
        "status": RUNNING, "pending_approval": None, "approval_log": approval_log,
        "approved_tool_calls": [*state["approved_tool_calls"], pending["fingerprint"]],
    }


def _route_after_approval(state: ProcedureRunState) -> str:
    return END if state["status"] == CANCELLED else "execute_current_step"


def _handle_step_result_node(state: ProcedureRunState, config: RunnableConfig) -> dict:
    definition: ProcedureDefinition = config["configurable"]["definition"]
    index = state["current_step_index"]
    step = definition.steps[index]
    result = state["step_results"][step.id]
    outcome = result["outcome"]

    if outcome == "SUCCESS":
        next_index = index + 1
        if next_index >= len(definition.steps):
            return {"status": COMPLETED, "final_message": _completion_message(definition)}
        return {"current_step_index": next_index}
    if outcome == "STOP":
        return {"status": COMPLETED, "final_message": _stop_message(definition, index, step, result)}
    return {"status": FAILED, "final_message": _failure_message(definition, index, step)}


def _route_after_step(state: ProcedureRunState) -> str:
    return "execute_current_step" if state["status"] == RUNNING else END


def _completion_message(definition: ProcedureDefinition) -> str:
    lines = ["Procedure completed successfully.", ""]
    lines += [f"✓ {step.title}" for step in definition.steps]
    return "\n".join(lines)


def _stop_message(definition: ProcedureDefinition, index: int, step: ProcedureStep, result: dict) -> str:
    lines = ["Procedure stopped as instructed.", ""]
    lines += [f"✓ {s.title}" for s in definition.steps[:index + 1]]
    lines += ["", "Reason:", result["summary"], "", "No later steps were executed."]
    return "\n".join(lines)


def _failure_message(definition: ProcedureDefinition, index: int, step: ProcedureStep) -> str:
    lines = ["Procedure failed.", ""]
    lines += [f"✓ {s.title}" for s in definition.steps[:index]]
    lines.append(f"✗ {step.title}")
    lines += ["", "No later steps were executed."]
    return "\n".join(lines)


def _audit_summary(run_id: str, definition: ProcedureDefinition, status: str, result: dict) -> dict[str, Any]:
    """An internal, structured record of one finished run, for diagnostics
    and evaluation only (never rendered into a user-facing message).

    MCP tool-level attempt/success/failure counts are not duplicated here:
    they are already recorded per step by `app.procedure.step_executor`'s
    own structured logs, keyed by the same `step_id` this summary reports
    outcomes for.
    """
    approval_log = result.get("approval_log", [])
    return {
        "procedure_id": definition.id,
        "procedure_version": definition.version,
        "run_id": run_id,
        "status": status,
        "step_outcomes": {
            step_id: data.get("outcome") for step_id, data in result.get("step_results", {}).items()
        },
        "approvals_requested": sum(1 for entry in approval_log if entry["decision"] == "requested"),
        "approvals_accepted": sum(1 for entry in approval_log if entry["decision"] == "granted"),
        "approvals_rejected": sum(1 for entry in approval_log if entry["decision"] == "rejected"),
    }


def _approval_rejected_message(pending: dict) -> str:
    return "\n".join([
        "Procedure cancelled: approval was not granted.", "",
        f"Step: {pending['step_title']}",
        f"Operation: {pending['operation']}",
    ])


def _build_graph():
    graph = StateGraph(ProcedureRunState)
    graph.add_node("collect_inputs", _collect_inputs_node)
    graph.add_node("confirm_if_required", _confirm_if_required_node)
    graph.add_node("execute_current_step", _execute_current_step_node)
    graph.add_node("await_tool_approval", _await_tool_approval_node)
    graph.add_node("handle_step_result", _handle_step_result_node)

    graph.add_edge(START, "collect_inputs")
    graph.add_edge("collect_inputs", "confirm_if_required")
    graph.add_conditional_edges("confirm_if_required", _route_after_confirmation, {
        "execute_current_step": "execute_current_step", END: END,
    })
    graph.add_conditional_edges("execute_current_step", _route_after_execute_step, {
        "await_tool_approval": "await_tool_approval", "handle_step_result": "handle_step_result",
    })
    graph.add_conditional_edges("await_tool_approval", _route_after_approval, {
        "execute_current_step": "execute_current_step", END: END,
    })
    graph.add_conditional_edges("handle_step_result", _route_after_step, {
        "execute_current_step": "execute_current_step", END: END,
    })
    return graph.compile(checkpointer=InMemorySaver())


def _missing_fields_payload(interrupt_value: dict) -> list[dict[str, str | None]]:
    return interrupt_value.get("fields", [])


def _message_for_interrupt(interrupt_value: dict) -> tuple[str, str]:
    """Return (status, user-facing message) for a pending interrupt. Never
    exposes the raw interrupt/thread internals to the caller.
    """
    kind = interrupt_value.get("type")
    if kind == "input_required":
        lines = ["I need the following information to continue.", ""]
        lines += [f"- {item['label']}" for item in interrupt_value["fields"]]
        return WAITING_FOR_INPUT, "\n".join(lines)
    if kind == "confirmation_required":
        lines = [
            f"Procedure: {interrupt_value['title']}",
            f"Risk: {interrupt_value['risk']}", "", "Steps:",
        ]
        lines += [f"{i}. {title}" for i, title in enumerate(interrupt_value["step_titles"], start=1)]
        lines += ["", "Proceed?"]
        return WAITING_FOR_CONFIRMATION, "\n".join(lines)
    if kind == "tool_approval_required":
        # Deliberately not the raw MCP tool name/arguments: a semantic
        # description of the operation and its risk level only.
        message = (
            "PROCEDURE · APPROVAL REQUIRED\n\n"
            f"Step:\n{interrupt_value['step_title']}\n\n"
            f"This step is about to perform a {interrupt_value['risk']} operation.\n\n"
            f"Operation:\n{interrupt_value['operation']}\n\n"
            "Proceed?"
        )
        return WAITING_FOR_APPROVAL, message
    raise ValueError(f"Unknown interrupt type: {kind!r}")  # pragma: no cover - defensive


class ProcedureRuntime:
    """Owns the compiled LangGraph, its checkpointer, and the registry of
    currently-active runs. One shared instance is enough for this
    application; tests can construct their own instance with a fake step
    executor to stay independent of production wiring.
    """

    def __init__(self, *, step_executor: StepExecutor = placeholder_step_executor):
        self._graph = _build_graph()
        self._default_step_executor = step_executor
        self._runs: dict[str, _RunContext] = {}

    def _config(self, run_id: str, context: _RunContext) -> dict:
        return {"configurable": {
            "thread_id": run_id, "definition": context.definition, "step_executor": context.step_executor,
        }}

    def _outcome_from_result(self, run_id: str, context: _RunContext, result: dict) -> ProcedureOutcome:
        interrupts = result.get("__interrupt__")
        if interrupts:
            status, message = _message_for_interrupt(interrupts[0].value)
            return ProcedureOutcome(
                run_id=run_id, status=status, message=message,
                step_results=result.get("step_results", {}),
                waiting_fields=_missing_fields_payload(interrupts[0].value),
            )
        status = result["status"]
        audit = _audit_summary(run_id, context.definition, status, result) if status in _TERMINAL_STATUSES else None
        return ProcedureOutcome(
            run_id=run_id, status=status,
            message=result.get("final_message") or "",
            step_results=result.get("step_results", {}),
            audit=audit,
        )

    async def start(self, definition: ProcedureDefinition, *,
                     step_executor: StepExecutor | None = None) -> ProcedureOutcome:
        run_id = str(uuid4())
        context = _RunContext(definition=definition, step_executor=step_executor or self._default_step_executor)
        self._runs[run_id] = context
        logger.info(
            "Procedure procedure_run_id=%s procedure_id=%s status=created",
            run_id, definition.id,
        )

        initial_state: ProcedureRunState = {
            "run_id": run_id, "procedure_id": definition.id, "procedure_version": definition.version,
            "status": PENDING, "inputs": {}, "current_step_index": 0,
            "step_results": {}, "final_message": None,
            "pending_approval": None, "approved_tool_calls": [], "approval_log": [],
        }
        result = await self._graph.ainvoke(initial_state, config=self._config(run_id, context))
        return self._finish(run_id, context, result)

    async def resume(self, run_id: str, resume_value: Any) -> ProcedureOutcome:
        context = self._runs.get(run_id)
        if context is None:
            raise ProcedureRunNotFound(run_id)
        logger.info("Procedure procedure_run_id=%s status=resumed", run_id)
        if context.last_outcome is not None and context.last_outcome.status == WAITING_FOR_CONFIRMATION and resume_value:
            logger.info("Procedure procedure_run_id=%s status=confirmed", run_id)
        if context.last_outcome is not None and context.last_outcome.status == WAITING_FOR_APPROVAL:
            logger.info(
                "Procedure procedure_run_id=%s status=approval_%s",
                run_id, "granted" if resume_value else "rejected",
            )
        result = await self._graph.ainvoke(Command(resume=resume_value), config=self._config(run_id, context))
        return self._finish(run_id, context, result)

    async def cancel(self, run_id: str) -> ProcedureOutcome:
        context = self._runs.get(run_id)
        if context is None:
            raise ProcedureRunNotFound(run_id)
        del self._runs[run_id]
        logger.info("Procedure procedure_run_id=%s status=%s", run_id, CANCELLED)
        # Cancelling does not touch the graph, so this reflects whatever was
        # last known (e.g. steps already completed before the user cancelled
        # while waiting for input/confirmation/approval).
        last_step_results = context.last_outcome.step_results if context.last_outcome else {}
        audit = _audit_summary(run_id, context.definition, CANCELLED, {"step_results": last_step_results})
        return ProcedureOutcome(run_id=run_id, status=CANCELLED, message="Procedure cancelled.", audit=audit)

    def peek(self, run_id: str) -> ProcedureOutcome | None:
        """The last known outcome for an active run, without touching the graph."""
        context = self._runs.get(run_id)
        return context.last_outcome if context else None

    def _finish(self, run_id: str, context: _RunContext, result: dict) -> ProcedureOutcome:
        outcome = self._outcome_from_result(run_id, context, result)
        context.last_outcome = outcome
        logger.info("Procedure procedure_run_id=%s status=%s", run_id, outcome.status)
        if outcome.audit is not None:
            logger.info(
                "Procedure audit procedure_run_id=%s procedure_id=%s status=%s "
                "step_outcomes=%s approvals_requested=%d approvals_accepted=%d approvals_rejected=%d",
                run_id, context.definition.id, outcome.status, outcome.audit["step_outcomes"],
                outcome.audit["approvals_requested"], outcome.audit["approvals_accepted"],
                outcome.audit["approvals_rejected"],
            )
        if outcome.status in _TERMINAL_STATUSES:
            self._runs.pop(run_id, None)
        return outcome


# A single shared runtime is enough for this application; the production
# step executor is the placeholder until a real one exists.
default_runtime = ProcedureRuntime()
