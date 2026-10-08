"""Data shapes for the deterministic `/procedure` path.

Routing/retrieval types are plain dataclasses (unchanged from earlier
iterations). The parsed-procedure types are small Pydantic models, kept
intentionally free of any MCP-specific concepts (no tool/server/system
names) so the KB stays independent from concrete MCP implementations.

No parameter extraction, missing-input prompting, or step execution is
implemented yet.
"""
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


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
    "parsed", "parse_error", "no_result", "not_executable", "empty_query",
    "mcp_error"). `message` is the human-readable text the chat UI renders
    as a normal chat message once procedure processing finishes; it is
    always the *last* entry of `messages` when more than one is produced
    (for example "KB found: ..." followed by the parsed summary).
    """

    state: str
    message: str
    title: str | None = None
    messages: tuple[str, ...] | None = None


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


class ProcedureContext(BaseModel):
    """Everything retained in memory for one `/procedure` invocation so far.

    No execution state yet — this is only carried through the current
    request for this iteration.
    """

    original_request: str
    kb_title: str
    kb_content: str
    procedure: ProcedureDefinition
