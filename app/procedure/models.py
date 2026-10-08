"""Data shapes for the deterministic `/procedure` path.

No procedure Markdown parsing, metadata extraction, parameter extraction, or
execution model is implemented yet. These are intentionally minimal containers
for routing and knowledge-base retrieval only.
"""
from dataclasses import dataclass


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
    "no_result", "not_executable", "empty_query", "mcp_error").
    `message` is the human-readable text the chat UI renders as a normal
    chat message once procedure processing finishes.
    """

    state: str
    message: str
    title: str | None = None
