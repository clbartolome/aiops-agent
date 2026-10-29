"""Data shapes for the deterministic `/procedure` path.

Routing/retrieval types are plain dataclasses (unchanged from earlier
iterations). The parsed-procedure types are small Pydantic models, kept
intentionally free of any MCP-specific concepts (no tool/server/system
names) so the KB stays independent from concrete MCP implementations.

Every `procedure.steps` entry is executed in fixed KB order, one at a
time, continuing only while each step returns SUCCESS (see
`app.procedure.executor` and the generic loop in `app.procedure`).
"""
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

# The only primitive types a declared input/value may hold.
PrimitiveValue = str | int | float | bool


@dataclass(frozen=True)
class KBResult:
    """The minimal knowledge-base fields the procedure path needs to keep."""

    title: str
    content: str


@dataclass(frozen=True)
class ProcedureRequest:
    """Internal record of one `/procedure` invocation and its KB match, if any."""

    original_message: str
    query: str
    kb: KBResult | None = None


@dataclass(frozen=True)
class ProcedureStatus:
    """The terminal, UI-facing outcome of one `/procedure` turn.

    `state` is a small machine-readable tag (for example "kb_found",
    "collecting_inputs", "completed", "stopped", "failed", "cancelled",
    "parse_error", "no_result", "not_executable", "empty_query",
    "mcp_error"). `message` is the final human-readable text for this
    turn (whatever is still useful to a non-streaming caller, e.g.
    `/api/chat`); the chat UI itself renders every permanent message as
    its own `procedure_message` event while the procedure runs (see
    `app.procedure`'s module docstring) rather than from this field.

    This is always the *last* item an `app.procedure` event generator
    yields (every earlier yielded item is a wire-shaped progress/message
    event dict); see `app.web._stream_procedure`.

    `context` carries the active `ProcedureContext` to persist on the chat
    session for the next turn (or `None` once there is no procedure left
    waiting on this session, e.g. after a terminal error or cancellation).
    It is never serialized to the client.
    """

    state: str
    message: str
    title: str | None = None
    context: "ProcedureContext | None" = None


class ProcedureInput(BaseModel):
    """One declared, named input the procedure needs (values are not extracted yet)."""

    name: str
    label: str
    description: str | None = None
    required: bool = True
    default: str | int | float | bool | None = None


class ProcedureStep(BaseModel):
    """One ordered, numbered step parsed from the KB `## Steps` section."""

    id: str
    title: str
    instruction: str


class ProcedureDefinition(BaseModel):
    """The deterministically parsed structure of one KB procedure.

    Deliberately free of MCP-specific fields (no tool_name, server_name,
    system, action, resource, or capability): the KB stays independent of
    any concrete MCP implementation.
    """

    id: str
    version: int
    title: str
    description: str | None = None
    risk: Literal["low", "medium", "high"]
    confirmation_required: bool

    inputs: list[ProcedureInput] = []
    steps: list[ProcedureStep] = []

    success_criteria: str | None = None


class StepResult(BaseModel):
    """The deliberately small, self-contained outcome of one executed step.

    The model only ever returns `status` and a human-readable `summary`;
    it never copies or serializes raw MCP payloads here (see
    `app.procedure.executor`, which tracks tool-call evidence separately
    in plain Python and rejects a self-reported SUCCESS that has no
    successful live MCP evidence behind it).

    - SUCCESS: the step completed as instructed.
    - STOP: the step's own instruction explicitly says to stop the
      procedure under some observed condition, and that condition held.
    - FAILED: the step could not be completed or verified.
    """

    status: Literal["SUCCESS", "STOP", "FAILED"]
    summary: str


class StepExecutionContext:
    """Minimal previous-step context passed to the Step Executor for
    chaining: just enough for the model to understand what already
    happened (title, status, summary) -- never raw MCP payloads. Real MCP
    evidence stays in application code (see `app.procedure.executor`).

    A plain object (not a Pydantic model): this is internal plumbing
    between orchestration and the Step Executor, never serialized.
    """

    __slots__ = ("step", "result")

    def __init__(self, step: ProcedureStep, result: StepResult):
        self.step = step
        self.result = result


class ProcedureContext(BaseModel):
    """The active `/procedure` run retained on a chat session between turns.

    `inputs` holds the input values collected so far (defaults applied,
    plus anything deterministically validated from extraction).

    `status` tracks progress: input collection (`COLLECTING_INPUTS`,
    `WAITING_FOR_CONFIRMATION` for every run, `READY`), step execution (`RUNNING`, set just before the first step
    starts), a terminal outcome (`COMPLETED` once every step has
    succeeded, `STOPPED`, `FAILED`), or `CANCELLED`. Steps run one at a
    time, in fixed KB order, continuing only on SUCCESS; `current_step_index`
    starts at 0 and advances past a step only once it succeeds, so on
    STOP/FAILED it points at the step that stopped/failed the procedure.
    `step_results` holds every executed step's result so far, keyed by
    step id, in execution order, and never overwrites an earlier step's
    stored result.
    """

    source: Literal["user", "incident"] = "user"
    source_id: str | None = None
    # Application-level approval is mandatory before collection or execution.
    confirmed: bool = False
    run_id: str
    original_request: str
    kb_title: str
    kb_content: str
    procedure: ProcedureDefinition
    inputs: dict[str, PrimitiveValue] = {}
    status: Literal[
        "COLLECTING_INPUTS", "WAITING_FOR_CONFIRMATION", "READY", "RUNNING", "COMPLETED", "STOPPED", "FAILED", "CANCELLED",
    ] = "COLLECTING_INPUTS"
    step_results: dict[str, StepResult] = {}
    current_step_index: int = 0
