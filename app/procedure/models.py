"""The fixed internal procedure structure.

Markdown defines the procedure structure; the parser (``app.procedure.parser``)
extracts it deterministically into these models. The LLM never invents or
edits this structure, and these models intentionally do not bind steps to any
MCP tool, server, or argument mapping — that belongs to a later iteration.
"""
from typing import Literal

from pydantic import BaseModel

RiskLevel = Literal["low", "medium", "high"]


class ProcedureInput(BaseModel):
    """A single declared input, parsed from "## Required information"."""
    name: str
    label: str
    description: str | None = None
    required: bool = True
    default: str | int | float | bool | None = None


class ProcedureStep(BaseModel):
    """A single fixed step, parsed from a "### N. Title" heading under "## Steps".

    `instruction` is the step's source wording, kept close to the original KB
    text: a later runtime step will hand it, verbatim, to a step executor.
    """
    id: str
    title: str
    instruction: str
    input_refs: list[str] = []


class ProcedureDefinition(BaseModel):
    """The complete, fixed structure parsed from one KB procedure article."""
    id: str
    version: int
    title: str
    description: str | None = None
    risk: RiskLevel
    confirmation_required: bool
    inputs: list[ProcedureInput]
    steps: list[ProcedureStep]
    success_criteria: str | None = None


class StepResult(BaseModel):
    """What a Step Executor reports back for exactly one step.

    The Step Executor only answers "given this exact step, can you complete
    it?" — it never decides what happens next. LangGraph alone interprets
    this result to control sequencing (advance, stop, or fail the run).
    """
    outcome: Literal["SUCCESS", "STOP", "FAILED"]
    summary: str
    data: dict[str, object] | None = None
    error: str | None = None
