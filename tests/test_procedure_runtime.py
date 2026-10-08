"""Tests for the LangGraph procedure runtime, using fake step executors only.

No real MCP server and no real LLM API is called anywhere in this file: the
runtime depends only on the injected `StepExecutor` contract.
"""
import asyncio

import pytest

from app.procedure.models import ProcedureDefinition, ProcedureInput, ProcedureStep, StepResult
from app.procedure.risk import ApprovalRequired
from app.procedure.runtime import (
    CANCELLED,
    COMPLETED,
    FAILED,
    WAITING_FOR_APPROVAL,
    WAITING_FOR_CONFIRMATION,
    WAITING_FOR_INPUT,
    ProcedureRunNotFound,
    ProcedureRuntime,
)


def make_definition(*, confirmation_required=False, inputs=None, risk="low", step_count=3):
    if inputs is None:
        inputs = [ProcedureInput(name="namespace", label="Namespace", required=True)]
    input_refs = [item.name for item in inputs if item.required]
    return ProcedureDefinition(
        id="inspect-namespace-health", version=1, title="Inspect namespace health",
        risk=risk, confirmation_required=confirmation_required,
        inputs=inputs,
        steps=[
            ProcedureStep(
                id=f"step{i}", title=f"Step {i}", instruction=f"Do step {i}.", input_refs=input_refs,
            )
            for i in range(1, step_count + 1)
        ],
    )


def fake_executor(outcomes_by_step=None, default_outcome="SUCCESS"):
    calls = []

    async def executor(step, inputs, previous_results=None, *, approved_calls=frozenset()):
        calls.append((step.id, dict(inputs)))
        outcome = (outcomes_by_step or {}).get(step.id, default_outcome)
        if outcome == "SUCCESS":
            return StepResult(outcome="SUCCESS", summary=f"{step.title} done")
        if outcome == "STOP":
            return StepResult(outcome="STOP", summary="Namespace does not exist.")
        return StepResult(outcome="FAILED", summary="boom")

    executor.calls = calls
    return executor


def run(coro):
    return asyncio.run(coro)


# --- Missing input -----------------------------------------------------------

def test_missing_required_input_interrupts_without_calling_the_executor():
    executor = fake_executor()
    runtime = ProcedureRuntime(step_executor=executor)
    outcome = run(runtime.start(make_definition()))

    assert outcome.status == WAITING_FOR_INPUT
    assert [field["name"] for field in outcome.waiting_fields] == ["namespace"]
    assert "Namespace" in outcome.message
    assert executor.calls == []


# --- Resume -------------------------------------------------------------------

def test_resume_with_input_continues_the_same_run():
    executor = fake_executor()
    runtime = ProcedureRuntime(step_executor=executor)
    first = run(runtime.start(make_definition()))

    second = run(runtime.resume(first.run_id, {"namespace": "payments"}))

    assert second.run_id == first.run_id
    assert second.status == COMPLETED
    assert all(inputs == {"namespace": "payments"} for _, inputs in executor.calls)
    assert [step_id for step_id, _ in executor.calls] == ["step1", "step2", "step3"]


# --- Default input -------------------------------------------------------------

def test_input_with_default_does_not_interrupt():
    inputs = [ProcedureInput(name="minimum_pod_count", label="Minimum pod count", required=False, default=1)]
    definition = make_definition(inputs=inputs)
    executor = fake_executor()
    runtime = ProcedureRuntime(step_executor=executor)

    outcome = run(runtime.start(definition))

    assert outcome.status == COMPLETED
    assert len(executor.calls) == 3
    assert all(inputs == {} for _, inputs in executor.calls)  # steps only reference "namespace"


# --- Confirmation --------------------------------------------------------------

def test_confirmation_required_interrupts_before_the_first_step():
    executor = fake_executor()
    runtime = ProcedureRuntime(step_executor=executor)
    definition = make_definition(confirmation_required=True, risk="medium", inputs=[])

    outcome = run(runtime.start(definition))

    assert outcome.status == WAITING_FOR_CONFIRMATION
    assert "Risk: medium" in outcome.message
    assert executor.calls == []


def test_rejecting_confirmation_cancels_with_zero_executor_calls():
    executor = fake_executor()
    runtime = ProcedureRuntime(step_executor=executor)
    first = run(runtime.start(make_definition(confirmation_required=True, inputs=[])))

    second = run(runtime.resume(first.run_id, False))

    assert second.status == CANCELLED
    assert executor.calls == []


def test_approving_confirmation_proceeds_to_execution():
    executor = fake_executor()
    runtime = ProcedureRuntime(step_executor=executor)
    first = run(runtime.start(make_definition(confirmation_required=True, inputs=[])))

    second = run(runtime.resume(first.run_id, True))

    assert second.status == COMPLETED
    assert [step_id for step_id, _ in executor.calls] == ["step1", "step2", "step3"]


# --- Sequential success --------------------------------------------------------

def test_sequential_success_executes_steps_in_exact_order_and_completes():
    executor = fake_executor()
    runtime = ProcedureRuntime(step_executor=executor)

    outcome = run(runtime.start(make_definition(inputs=[])))

    assert [step_id for step_id, _ in executor.calls] == ["step1", "step2", "step3"]
    assert outcome.status == COMPLETED
    assert "✓ Step 1" in outcome.message and "✓ Step 2" in outcome.message and "✓ Step 3" in outcome.message


# --- STOP outcome ---------------------------------------------------------------

def test_stop_outcome_terminates_cleanly_without_running_later_steps():
    executor = fake_executor(outcomes_by_step={"step1": "STOP"})
    runtime = ProcedureRuntime(step_executor=executor)

    outcome = run(runtime.start(make_definition(inputs=[])))

    assert [step_id for step_id, _ in executor.calls] == ["step1"]
    assert outcome.status == COMPLETED
    assert "Namespace does not exist." in outcome.message


# --- FAILED outcome ---------------------------------------------------------------

def test_failed_outcome_stops_before_later_steps():
    executor = fake_executor(outcomes_by_step={"step2": "FAILED"})
    runtime = ProcedureRuntime(step_executor=executor)

    outcome = run(runtime.start(make_definition(inputs=[])))

    assert [step_id for step_id, _ in executor.calls] == ["step1", "step2"]
    assert outcome.status == FAILED
    assert "✗ Step 2" in outcome.message


# --- Executor exception ------------------------------------------------------------

def test_unexpected_executor_exception_causes_a_controlled_failure():
    async def exploding_executor(step, inputs, previous_results=None, *, approved_calls=frozenset()):
        raise RuntimeError("boom")

    runtime = ProcedureRuntime(step_executor=exploding_executor)
    outcome = run(runtime.start(make_definition(inputs=[])))

    assert outcome.status == FAILED
    assert "boom" not in outcome.message  # no raw exception text leaks to the user
    assert "✗ Step 1" in outcome.message


# --- Cancellation --------------------------------------------------------------

def test_cancelling_an_interrupted_run_sets_cancelled_status():
    executor = fake_executor()
    runtime = ProcedureRuntime(step_executor=executor)
    first = run(runtime.start(make_definition()))
    assert first.status == WAITING_FOR_INPUT

    outcome = run(runtime.cancel(first.run_id))

    assert outcome.status == CANCELLED
    assert runtime.peek(first.run_id) is None


def test_cancelling_an_unknown_run_raises():
    runtime = ProcedureRuntime(step_executor=fake_executor())
    with pytest.raises(ProcedureRunNotFound):
        run(runtime.cancel("not-a-real-run"))


def test_resuming_an_unknown_run_raises():
    runtime = ProcedureRuntime(step_executor=fake_executor())
    with pytest.raises(ProcedureRunNotFound):
        run(runtime.resume("not-a-real-run", {"namespace": "x"}))


# --- Run/session isolation -----------------------------------------------------

def test_two_runs_of_the_same_procedure_are_independent():
    executor = fake_executor()
    runtime = ProcedureRuntime(step_executor=executor)
    definition = make_definition()

    run_a = run(runtime.start(definition))
    run_b = run(runtime.start(definition))
    assert run_a.run_id != run_b.run_id

    outcome_a = run(runtime.resume(run_a.run_id, {"namespace": "payments"}))
    outcome_b = run(runtime.resume(run_b.run_id, {"namespace": "billing"}))

    assert outcome_a.status == COMPLETED and outcome_b.status == COMPLETED
    payments_calls = [i for sid, i in executor.calls if i.get("namespace") == "payments"]
    billing_calls = [i for sid, i in executor.calls if i.get("namespace") == "billing"]
    assert len(payments_calls) == 3 and len(billing_calls) == 3


# --- Step results & step input resolution --------------------------------------

def test_step_results_are_stored_by_step_id_with_structured_data():
    async def executor(step, inputs, previous_results=None, *, approved_calls=frozenset()):
        return StepResult(outcome="SUCCESS", summary="ok", data={"pod_count": 3})

    runtime = ProcedureRuntime(step_executor=executor)
    outcome = run(runtime.start(make_definition(inputs=[])))

    assert outcome.status == COMPLETED
    assert outcome.step_results["step1"]["data"] == {"pod_count": 3}
    assert outcome.step_results["step3"]["outcome"] == "SUCCESS"


def test_previous_step_results_are_passed_to_later_steps():
    seen_previous = []

    async def executor(step, inputs, previous_results=None, *, approved_calls=frozenset()):
        seen_previous.append((step.id, dict(previous_results or {})))
        return StepResult(outcome="SUCCESS", summary=f"{step.title} done", data={"from": step.id})

    runtime = ProcedureRuntime(step_executor=executor)
    run(runtime.start(make_definition(inputs=[])))

    assert seen_previous[0] == ("step1", {})
    assert set(seen_previous[1][1]) == {"step1"}
    assert seen_previous[1][1]["step1"].data == {"from": "step1"}
    assert set(seen_previous[2][1]) == {"step1", "step2"}
    # Never a `StepResult` dict passed as the raw chat/conversation history.
    assert all(isinstance(result, StepResult) for _, previous in seen_previous for result in previous.values())


# --- Internal audit summary (never exposed via `.message`) --------------------

def test_completed_outcome_carries_an_internal_audit_summary():
    executor = fake_executor()
    runtime = ProcedureRuntime(step_executor=executor)

    outcome = run(runtime.start(make_definition(inputs=[])))

    assert outcome.status == COMPLETED
    assert outcome.audit == {
        "procedure_id": "inspect-namespace-health", "procedure_version": 1,
        "run_id": outcome.run_id, "status": COMPLETED,
        "step_outcomes": {"step1": "SUCCESS", "step2": "SUCCESS", "step3": "SUCCESS"},
        "approvals_requested": 0, "approvals_accepted": 0, "approvals_rejected": 0,
    }
    # Internal diagnostics only: never rendered into the user-facing message.
    assert "audit" not in outcome.message.lower()
    assert "approvals_requested" not in outcome.message


def test_audit_summary_is_absent_for_non_terminal_outcomes():
    executor = fake_executor()
    runtime = ProcedureRuntime(step_executor=executor)

    outcome = run(runtime.start(make_definition()))

    assert outcome.status == WAITING_FOR_INPUT
    assert outcome.audit is None


def test_audit_summary_counts_approval_decisions():
    attempts = {"count": 0}

    async def executor(step, inputs, previous_results=None, *, approved_calls=frozenset()):
        if step.id == "step1" and not approved_calls:
            attempts["count"] += 1
            raise ApprovalRequired(
                tool_name="launch_job", risk="WRITE", operation="Launch a job",
                fingerprint="launch_job:abc123",
            )
        return StepResult(outcome="SUCCESS", summary=f"{step.title} done")

    runtime = ProcedureRuntime(step_executor=executor)
    first = run(runtime.start(make_definition(inputs=[])))
    assert first.status == WAITING_FOR_APPROVAL
    assert first.audit is None

    second = run(runtime.resume(first.run_id, True))

    assert second.status == COMPLETED
    assert second.audit["approvals_requested"] == 1
    assert second.audit["approvals_accepted"] == 1
    assert second.audit["approvals_rejected"] == 0


def test_audit_summary_counts_a_rejected_approval_on_cancellation():
    async def executor(step, inputs, previous_results=None, *, approved_calls=frozenset()):
        raise ApprovalRequired(
            tool_name="launch_job", risk="WRITE", operation="Launch a job",
            fingerprint="launch_job:abc123",
        )

    runtime = ProcedureRuntime(step_executor=executor)
    first = run(runtime.start(make_definition(inputs=[])))

    second = run(runtime.resume(first.run_id, False))

    assert second.status == CANCELLED
    assert second.audit["approvals_requested"] == 1
    assert second.audit["approvals_accepted"] == 0
    assert second.audit["approvals_rejected"] == 1


def test_cancelling_a_waiting_run_still_produces_an_audit_summary():
    executor = fake_executor()
    runtime = ProcedureRuntime(step_executor=executor)
    first = run(runtime.start(make_definition()))

    outcome = run(runtime.cancel(first.run_id))

    assert outcome.status == CANCELLED
    assert outcome.audit == {
        "procedure_id": "inspect-namespace-health", "procedure_version": 1,
        "run_id": first.run_id, "status": CANCELLED,
        "step_outcomes": {}, "approvals_requested": 0, "approvals_accepted": 0, "approvals_rejected": 0,
    }


def test_only_referenced_inputs_are_passed_to_a_step():
    inputs = [
        ProcedureInput(name="namespace", label="Namespace", required=True),
        ProcedureInput(name="application_name", label="Application name", required=True),
    ]
    definition = ProcedureDefinition(
        id="p", version=1, title="T", risk="low", confirmation_required=False,
        inputs=inputs,
        steps=[ProcedureStep(id="s1", title="S1", instruction="do", input_refs=["namespace"])],
    )
    seen = {}

    async def executor(step, resolved, previous_results=None, *, approved_calls=frozenset()):
        seen.update(resolved)
        return StepResult(outcome="SUCCESS", summary="ok")

    runtime = ProcedureRuntime(step_executor=executor)
    first = run(runtime.start(definition))
    run(runtime.resume(first.run_id, {"namespace": "payments", "application_name": "billing"}))

    assert seen == {"namespace": "payments"}
