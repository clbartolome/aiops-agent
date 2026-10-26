"""Deterministic parser for the fixed `/procedure` knowledge-base Markdown format.

Pure Python, regex-based parsing: no LLM is used to determine procedure
structure. The retrieved KB Markdown (already fetched once via
`rag_search_kb`) is the single source of truth; this module never calls any
MCP tool and never re-searches the knowledge base.

Supported fixed structure (see the KB sample in the project's procedure
iteration notes):

    # <Title>

    <optional free-text description>

    ## Procedure

    **ID:** <id>
    **Version:** <positive integer>
    **Risk:** low|medium|high
    **Confirmation required:** yes|no (or true|false)

    ## Required information

    - **<Label>** — required|optional — <description>[. Defaults to `<value>`.]

    ## Steps

    ### 1. <Title>

    <instruction body, preserved verbatim>

    ## Success

    <success criteria text, preserved verbatim>
"""
import re

from app.procedure.models import ProcedureDefinition, ProcedureInput, ProcedureStep

RISK_LEVELS = ("low", "medium", "high")

_TRUE_WORDS = {"yes", "true"}
_FALSE_WORDS = {"no", "false"}

# Headings: exactly N '#' characters (never matched by a heading with more).
_TITLE_RE = re.compile(r"^#(?!#)\s+(.+)$", re.MULTILINE)
_SECTION_RE = re.compile(r"^##(?!#)\s+(.+)$", re.MULTILINE)
_STEP_HEADING_RE = re.compile(r"^###(?!#)\s+(\d+)\.\s+(.+)$", re.MULTILINE)

_METADATA_LINE_RE = re.compile(r"^\*\*([^*]+):\*\*\s*(.*)$")
# "- **Label** — required|optional — description" (em dash separators).
_REQUIRED_INPUT_RE = re.compile(r"^-\s+\*\*([^*]+)\*\*\s+—\s+(required|optional)\s+—\s+(.*)$")
_DEFAULT_RE = re.compile(r"Defaults to `([^`]+)`")

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


class ProcedureParseError(RuntimeError):
    """A controlled, credential-free failure parsing a KB procedure."""


def normalize_name(label: str) -> str:
    """Deterministically derive an internal identifier from a human label.

    "Minimum pod count" -> "minimum_pod_count". Pure Python string logic;
    no LLM is involved.
    """
    return _NON_ALNUM_RE.sub("_", label.strip().lower()).strip("_")


def _section_bodies(content: str) -> tuple[dict[str, str], list[re.Match]]:
    """Map each top-level '##' section name to its body text (first occurrence wins)."""
    matches = list(_SECTION_RE.finditer(content))
    bodies: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        bodies.setdefault(match.group(1).strip(), content[start:end].strip())
    return bodies, matches


def _parse_metadata(body: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in body.splitlines():
        match = _METADATA_LINE_RE.match(line.strip())
        if match:
            metadata[match.group(1).strip().lower()] = match.group(2).strip()
    return metadata


def _parse_bool(value: str, field_label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_WORDS:
        return True
    if normalized in _FALSE_WORDS:
        return False
    raise ProcedureParseError(f"Invalid {field_label}: {value!r}. Expected yes/no or true/false.")


def coerce_primitive(raw: str) -> str | int | float | bool:
    """Deterministically coerce raw text into one of the supported primitives.

    Tries boolean words, then integer, then float, falling back to the
    trimmed original string. Used both for declared KB defaults and for the
    single-missing-input reply shortcut (see `app.procedure`); no LLM is
    involved.
    """
    stripped = raw.strip()
    lowered = stripped.lower()
    if lowered in _TRUE_WORDS:
        return True
    if lowered in _FALSE_WORDS:
        return False
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped


def _parse_metadata_section(procedure_body: str) -> tuple[str, int, str, bool]:
    metadata = _parse_metadata(procedure_body)

    proc_id = metadata.get("id", "").strip()
    if not proc_id:
        raise ProcedureParseError("Missing required metadata: ID.")

    version_raw = metadata.get("version", "").strip()
    if not version_raw:
        raise ProcedureParseError("Missing required metadata: Version.")
    try:
        version = int(version_raw)
    except ValueError:
        raise ProcedureParseError(
            f"Invalid version: {version_raw!r}. Version must be a positive integer."
        ) from None
    if version <= 0:
        raise ProcedureParseError(
            f"Invalid version: {version_raw!r}. Version must be a positive integer."
        )

    risk_raw = metadata.get("risk", "").strip()
    if not risk_raw:
        raise ProcedureParseError("Missing required metadata: Risk.")
    risk = risk_raw.lower()
    if risk not in RISK_LEVELS:
        raise ProcedureParseError(
            f"Invalid risk: {risk_raw!r}. Risk must be one of: {', '.join(RISK_LEVELS)}."
        )

    confirmation_raw = metadata.get("confirmation required", "").strip()
    if not confirmation_raw:
        raise ProcedureParseError("Missing required metadata: Confirmation required.")
    confirmation_required = _parse_bool(confirmation_raw, "confirmation required")

    return proc_id, version, risk, confirmation_required


def _parse_inputs(body: str) -> list[ProcedureInput]:
    inputs: list[ProcedureInput] = []
    seen_names: dict[str, str] = {}
    for line in body.splitlines():
        match = _REQUIRED_INPUT_RE.match(line.strip())
        if not match:
            continue
        raw_label, requirement, description = match.groups()
        label = raw_label.strip()
        description = description.strip()

        name = normalize_name(label)
        if not name:
            raise ProcedureParseError(f"Unable to derive an input name from label: {label!r}")
        if name in seen_names:
            raise ProcedureParseError(
                f"Duplicate input name {name!r} from labels {seen_names[name]!r} and {label!r}."
            )
        seen_names[name] = label

        default = None
        default_match = _DEFAULT_RE.search(description)
        if default_match:
            try:
                default = coerce_primitive(default_match.group(1))
            except Exception as error:
                raise ProcedureParseError(
                    f"Unable to parse default for input {label!r}: {default_match.group(1)!r}"
                ) from error

        inputs.append(ProcedureInput(
            name=name, label=label, description=description or None,
            required=(requirement == "required"), default=default,
        ))
    return inputs


def _parse_steps(body: str) -> list[ProcedureStep]:
    matches = list(_STEP_HEADING_RE.finditer(body))
    steps: list[ProcedureStep] = []
    seen_ids: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        instruction = body[start:end].strip()

        step_id = normalize_name(title)
        if not step_id:
            raise ProcedureParseError(f"Unable to derive a step id from title: {title!r}")
        if step_id in seen_ids:
            raise ProcedureParseError(
                f"Duplicate step id {step_id!r} from titles {seen_ids[step_id]!r} and {title!r}."
            )
        seen_ids[step_id] = title

        steps.append(ProcedureStep(id=step_id, title=title, instruction=instruction))
    return steps


def parse_procedure(content: str) -> ProcedureDefinition:
    """Deterministically parse the fixed KB Markdown procedure format.

    Pure Python/regex parsing of the already-retrieved KB Markdown, which is
    the single source of truth. No LLM is used and no MCP tool is called.
    Raises `ProcedureParseError` on any structural or validation failure.
    """
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")

    title_match = _TITLE_RE.search(normalized)
    if not title_match:
        raise ProcedureParseError("Missing top-level title heading ('# Title').")
    title = title_match.group(1).strip()

    bodies, section_matches = _section_bodies(normalized)
    description_end = section_matches[0].start() if section_matches else len(normalized)
    description = normalized[title_match.end():description_end].strip() or None

    procedure_body = bodies.get("Procedure")
    if procedure_body is None:
        raise ProcedureParseError("Missing '## Procedure' section.")
    proc_id, version, risk, confirmation_required = _parse_metadata_section(procedure_body)

    inputs = _parse_inputs(bodies.get("Required information", ""))

    steps = _parse_steps(bodies.get("Steps", ""))
    if not steps:
        raise ProcedureParseError("No executable steps were found.")

    success_criteria = bodies.get("Success") or None

    return ProcedureDefinition(
        id=proc_id, version=version, title=title, description=description,
        risk=risk, confirmation_required=confirmation_required,
        inputs=inputs, steps=steps, success_criteria=success_criteria,
    )
