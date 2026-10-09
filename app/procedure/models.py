"""Data shapes for the deterministic `/procedure` path.

Routing/retrieval types are plain dataclasses (unchanged from earlier
iterations). The parsed-procedure types are small Pydantic models, kept
intentionally free of any MCP-specific concepts (no tool/server/system
names) so the KB stays independent from concrete MCP implementations.

Only the first procedure step is executed so far (see
`app.procedure.executor`); the full step loop is not implemented yet.
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
    """A single, UI-facing snapshot of `/procedure` processing.

    `state` is a small machine-readable tag (for example "kb_found",
    "collecting_inputs", "first_step_completed", "stopped", "failed",
    "cancelled", "parse_error", "no_result", "not_executable",
    "empty_query", "mcp_error"). `message` is the human-readable text the
    chat UI renders as a normal chat message once procedure processing
    finishes; it is always the *last* entry of `messages` when more than
    one is produced (for example "KB found: ..." then the parsed summary,
    then the collected-inputs/ready outcome, then the first step's
    completed/stopped/failed result).

    `progress_stages`, when present, has exactly `len(messages) - 1`
    entries: the transient progress-bubble label to show right before
    revealing `messages[i]` for `i > 0`.

    `context` carries the active `ProcedureContext` to persist on the chat
    session for the next turn (or `None` once there is no procedure left
    waiting on this session, e.g. after a terminal error or cancellation).
    It is never serialized to the client.
    """

    state: str
    message: str
    title: str | None = None
    messages: tuple[str, ...] | None = None
    progress_stages: tuple[str, ...] | None = None
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


class ProcedureContext(BaseModel):
    """The active `/procedure` run retained on a chat session between turns.

    `inputs` holds the input values collected so far (defaults applied,
    plus anything deterministically validated from extraction).

    `status` tracks progress: input collection (`COLLECTING_INPUTS`,
    `READY`), the outcome of executing the first step
    (`FIRST_STEP_COMPLETED`, `STOPPED`, `FAILED`), or `CANCELLED`. Only the
    first step is ever executed so far; `current_step_index` stays `0` and
    `step_results` has at most one entry this iteration.
    """

    run_id: str
    original_request: str
    kb_title: str
    kb_content: str
    procedure: ProcedureDefinition
    inputs: dict[str, PrimitiveValue] = {}
    status: Literal[
        "COLLECTING_INPUTS", "READY", "FIRST_STEP_COMPLETED", "STOPPED", "FAILED", "CANCELLED",
    ] = "COLLECTING_INPUTS"
    step_results: dict[str, StepResult] = {}
    current_step_index: int = 0
