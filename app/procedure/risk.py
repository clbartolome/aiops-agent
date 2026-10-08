"""Runtime tool-risk policy for the procedure Step Executor.

This is deliberately NOT a capability registry, a semantic
system/action/resource model, or a static tool binder: it classifies and
gates individual proposed MCP tool calls at the moment the Step Executor's
agent proposes them, using only the tool's own name/description/annotations
and the current step's own wording. Nothing here is stored in, or derived
from, the procedure knowledge base/Markdown.

Classification sources, in order of preference:
  1. MCP tool annotations (`readOnlyHint`/`destructiveHint`), when present.
  2. A small override map for exceptional tools, if one is ever needed.
  3. Lightweight, deterministic inference from the tool's name/description.
  4. `UNKNOWN`, when none of the above establishes a risk level safely.

The conservative default is `UNKNOWN`, never `READ`: an unclassifiable tool
is treated the same as a `WRITE`/`DESTRUCTIVE` one for approval purposes.
"""
import hashlib
import json
import re

from agents.exceptions import UserError

READ = "READ"
WRITE = "WRITE"
DESTRUCTIVE = "DESTRUCTIVE"
UNKNOWN = "UNKNOWN"

# Deliberately small and explainable: whole-word intent hints, not an
# ontology. Checked most-dangerous-first so an ambiguous name/description
# is never classified as less risky than it might be.
_DESTRUCTIVE_HINTS = (
    "delete", "destroy", "terminate", "remove", "wipe", "revoke", "purge", "drop", "kill",
)
_WRITE_HINTS = (
    "create", "launch", "start", "update", "patch", "assign", "add", "set", "modify",
    "run", "trigger", "restart", "scale", "approve", "execute", "deploy",
)
_READ_HINTS = (
    "list", "get", "read", "search", "inspect", "status", "describe", "find", "fetch",
    "show", "check", "view", "count",
)

# Exceptional tools whose name/description would otherwise be misclassified
# by the deterministic inference below. Empty by default: this is a safety
# valve, not a per-tool maintenance burden, and is only ever populated when
# a specific tool is demonstrably misclassified.
RISK_OVERRIDES: dict[str, str] = {}

_TOKEN_RE = re.compile(r"[a-z]+")


class ApprovalRequired(UserError):
    """Raised at the tool-call boundary when a non-`READ` tool call has no
    matching approval yet.

    Subclasses the Agents SDK's own `UserError` so the existing MCP
    tool-call exception pipeline (`agents.mcp.util.invoke_mcp_tool`)
    re-raises it unchanged instead of wrapping it into a generic
    `AgentsException`. The Step Executor's own `failure_error_function`
    additionally recognizes this exact type and returns `None` for it,
    which the Agents SDK treats as "do not convert to a tool-visible
    string, let it propagate" — so it reaches `app.procedure.runtime`
    rather than becoming a model-visible tool error the agent could try to
    work around on its own.
    """

    def __init__(self, *, tool_name: str, risk: str, operation: str, fingerprint: str):
        self.tool_name = tool_name
        self.risk = risk
        self.operation = operation
        self.fingerprint = fingerprint
        super().__init__(f"Approval required for tool {tool_name!r} ({risk})")


class ScopeViolationError(UserError):
    """The proposed tool call is clearly unrelated to the current step."""

    def __init__(self, *, tool_name: str, step_title: str):
        self.tool_name = tool_name
        super().__init__(f"Tool {tool_name!r} is out of scope for step {step_title!r}")


class DuplicateToolCallError(UserError):
    """The exact same risky tool call already succeeded earlier in this step."""

    def __init__(self, *, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool {tool_name!r} already completed successfully in this step")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _matches_any(text: str, hints: tuple[str, ...]) -> bool:
    tokens = _tokens(text)
    return any(hint in tokens for hint in hints)


def _risk_from_annotations(tool) -> str | None:
    annotations = getattr(tool, "annotations", None)
    if annotations is None:
        return None
    if getattr(annotations, "read_only_hint", None) is True:
        return READ
    if getattr(annotations, "destructive_hint", None) is True:
        return DESTRUCTIVE
    if getattr(annotations, "read_only_hint", None) is False:
        return WRITE
    return None


def _risk_from_text(name: str, description: str) -> str:
    text = f"{name} {description}"
    if _matches_any(text, _DESTRUCTIVE_HINTS):
        return DESTRUCTIVE
    if _matches_any(text, _WRITE_HINTS):
        return WRITE
    if _matches_any(text, _READ_HINTS):
        return READ
    return UNKNOWN


def classify_tool_risk(tool) -> str:
    """Classify one MCP tool's risk. Never consults, and never writes to,
    the procedure knowledge base.
    """
    override = RISK_OVERRIDES.get(tool.name)
    if override is not None:
        return override
    from_annotations = _risk_from_annotations(tool)
    if from_annotations is not None:
        return from_annotations
    return _risk_from_text(tool.name, tool.description or "")


def describe_operation(tool) -> str:
    """A short, user-facing description of what a tool does, for the
    approval prompt. Never exposes raw provider protocol structures.
    """
    description = (tool.description or "").strip()
    if description:
        return description.rstrip(".")
    return tool.name.replace("_", " ").replace("-", " ").strip().capitalize()


def is_in_scope(step, tool, risk: str) -> bool:
    """A deterministic, conservative bounded-scope guardrail.

    This does not try to predict the single correct tool for a step (the
    Step Executor still chooses tools agentically); it only rejects tool
    calls that are clearly unrelated to the current step's own wording, so
    that risk approval is never asked to validate obviously out-of-scope
    behavior (for example a destructive "delete" call during an
    observational "inspect" step).
    """
    if risk == READ:
        return True
    step_text = f"{step.title} {step.instruction}"
    if risk == DESTRUCTIVE:
        return _matches_any(step_text, _DESTRUCTIVE_HINTS)
    # WRITE or UNKNOWN: in scope only if the step's own wording expresses
    # some write or destructive intent itself.
    return _matches_any(step_text, _WRITE_HINTS) or _matches_any(step_text, _DESTRUCTIVE_HINTS)


def fingerprint_call(tool_name: str, arguments: dict) -> str:
    """A short, deterministic key for "this exact tool call", used for both
    approval matching and duplicate-side-effect detection. Never logged or
    stored with the raw argument values themselves.
    """
    normalized = json.dumps(arguments or {}, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{tool_name}:{normalized}".encode()).hexdigest()[:16]
    return f"{tool_name}:{digest}"
