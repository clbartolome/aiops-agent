"""Deterministic Markdown -> ProcedureDefinition parser.

Markdown defines the procedure structure. This module extracts that structure
using plain Python (regular expressions and string handling) only. No model
call is made here, and none of the structural facts below are inferred by an
LLM: procedure id/version/risk/confirmation, input labels/order/defaults,
step count/order/titles/ids, and step instruction text are all read directly
from the article's existing wording.

This module does not map steps to MCP tools and does not execute anything.
"""
import re

from app.procedure.models import ProcedureDefinition, ProcedureInput, ProcedureStep

_SUPPORTED_RISKS = ("low", "medium", "high")
_TRUE_WORDS = ("yes", "true", "1")
_FALSE_WORDS = ("no", "false", "0")

_NON_ALNUM_RUN = re.compile(r"[^a-z0-9]+")
_H1_RE = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
_H2_RE = re.compile(r"^##(?!#)\s+(?P<name>.+?)\s*$", re.MULTILINE)
_STEP_HEADING_RE = re.compile(r"^###\s+\d+\.\s*(?P<title>.+?)\s*$", re.MULTILINE)
_METADATA_LINE_RE = re.compile(r"^\*\*(?P<key>[^*:]+):\*\*\s*(?P<value>.+?)\s*$")
_INPUT_LINE_RE = re.compile(
    r"^-\s*\*\*(?P<label>[^*]+)\*\*\s*[-\u2013\u2014]\s*"
    r"(?P<requirement>required|optional)\s*[-\u2013\u2014]\s*(?P<description>.*)$",
    re.IGNORECASE,
)
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_DEFAULT_PHRASE_RE = re.compile(r"defaults?\s+to", re.IGNORECASE)
_DEFAULT_VALUE_RE = re.compile(r"defaults?\s+to\s*`(?P<value>[^`]*)`", re.IGNORECASE)


class ProcedureParseError(ValueError):
    """A KB procedure article does not follow the supported, deterministic
    Markdown convention. This is a controlled error, not a crash: callers
    should report it to the user instead of guessing at the missing structure.
    """


def normalize_identifier(text: str) -> str:
    """Deterministically normalize a human-readable label/title into a
    lower_snake_case identifier. The only normalization helper used for both
    input names and step ids, so identical wording always normalizes the
    same way.
    """
    normalized = _NON_ALNUM_RUN.sub("_", text.strip().lower()).strip("_")
    if not normalized:
        raise ProcedureParseError(f"Cannot derive an identifier from {text!r}")
    return normalized


def _split_sections(markdown: str) -> tuple[str, str | None, dict[str, str]]:
    """Split the article into (title, description, {section_name: body})."""
    h1_matches = list(_H1_RE.finditer(markdown))
    if not h1_matches:
        raise ProcedureParseError("Procedure article is missing a top-level '# Title' heading")
    title_match = h1_matches[0]
    title = title_match.group("title").strip()

    h2_matches = list(_H2_RE.finditer(markdown))
    preamble_end = h2_matches[0].start() if h2_matches else len(markdown)
    description = markdown[title_match.end():preamble_end].strip() or None

    sections: dict[str, str] = {}
    for index, match in enumerate(h2_matches):
        name = match.group("name").strip().lower()
        start = match.end()
        end = h2_matches[index + 1].start() if index + 1 < len(h2_matches) else len(markdown)
        sections[name] = markdown[start:end].strip()
    return title, description, sections


def _require_section(sections: dict[str, str], name: str) -> str:
    if name not in sections:
        raise ProcedureParseError(f"Procedure article is missing the required '## {name.title()}' section")
    return sections[name]


def _parse_bool(value: str, *, field: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE_WORDS:
        return True
    if lowered in _FALSE_WORDS:
        return False
    raise ProcedureParseError(f"Unsupported {field} value: {value!r}")


def _parse_metadata(procedure_section: str) -> dict[str, object]:
    values: dict[str, str] = {}
    for line in procedure_section.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _METADATA_LINE_RE.match(line)
        if match is None:
            raise ProcedureParseError(f"Unrecognized line in '## Procedure' metadata: {line!r}")
        values[match.group("key").strip().lower()] = match.group("value").strip()

    procedure_id = values.get("id", "").strip()
    if not procedure_id:
        raise ProcedureParseError("Procedure metadata is missing a non-empty 'ID'")

    version_text = values.get("version", "").strip()
    if not re.fullmatch(r"\d+", version_text or ""):
        raise ProcedureParseError(f"Procedure 'Version' must be a positive integer, got {version_text!r}")
    version = int(version_text)
    if version < 1:
        raise ProcedureParseError("Procedure 'Version' must be a positive integer")

    risk = values.get("risk", "").strip().lower()
    if risk not in _SUPPORTED_RISKS:
        raise ProcedureParseError(f"Unsupported procedure 'Risk': {risk!r}; expected one of {_SUPPORTED_RISKS}")

    confirmation_text = values.get("confirmation required")
    if confirmation_text is None:
        raise ProcedureParseError("Procedure metadata is missing 'Confirmation required'")
    confirmation_required = _parse_bool(confirmation_text, field="'Confirmation required'")

    return {
        "id": procedure_id, "version": version, "risk": risk,
        "confirmation_required": confirmation_required,
    }


def _coerce_default(raw: str) -> str | int | float | bool:
    if raw == "":
        raise ProcedureParseError("Default value must not be empty")
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return raw


def _parse_default(description: str) -> str | int | float | bool | None:
    match = _DEFAULT_VALUE_RE.search(description)
    if match is None:
        if _DEFAULT_PHRASE_RE.search(description):
            raise ProcedureParseError(
                f"Default value is not expressed as the supported `value` form: {description!r}"
            )
        return None
    return _coerce_default(match.group("value").strip())


def _parse_inputs(section: str | None) -> list[ProcedureInput]:
    if not section:
        return []
    inputs: list[ProcedureInput] = []
    for line in section.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _INPUT_LINE_RE.match(line)
        if match is None:
            raise ProcedureParseError(f"Unrecognized line in '## Required information': {line!r}")
        label = match.group("label").strip()
        description = match.group("description").strip() or None
        inputs.append(ProcedureInput(
            name=normalize_identifier(label),
            label=label,
            description=description,
            required=match.group("requirement").lower() == "required",
            default=_parse_default(description or ""),
        ))
    return inputs


def _validate_inputs(inputs: list[ProcedureInput]) -> None:
    names = [item.name for item in inputs]
    if len(names) != len(set(names)):
        raise ProcedureParseError(f"Duplicate input name(s) in '## Required information': {_duplicates(names)}")
    labels = [item.label for item in inputs]
    if len(labels) != len(set(labels)):
        raise ProcedureParseError(f"Duplicate input label(s) in '## Required information': {_duplicates(labels)}")


def _duplicates(values: list[str]) -> str:
    seen, dupes = set(), []
    for value in values:
        if value in seen and value not in dupes:
            dupes.append(value)
        seen.add(value)
    return ", ".join(sorted(dupes))


def _extract_input_refs(instruction: str, valid_names: set[str], *, step_title: str) -> list[str]:
    """Extract declared-input references from bold spans in a step instruction.

    Bold text is the only supported reference convention: every bold span
    must normalize to a declared input. This never guesses an undeclared
    reference; an explicit reference to something that is not a declared
    input fails parsing instead of being silently ignored.
    """
    refs: list[str] = []
    for match in _BOLD_RE.finditer(instruction):
        name = normalize_identifier(match.group(1))
        if name not in valid_names:
            raise ProcedureParseError(
                f"Step '{step_title}' references undeclared input '{match.group(1).strip()}'"
            )
        if name not in refs:
            refs.append(name)
    return refs


def _parse_steps(steps_section: str, valid_names: set[str]) -> list[ProcedureStep]:
    headings = list(_STEP_HEADING_RE.finditer(steps_section))
    if not headings:
        raise ProcedureParseError("Procedure '## Steps' section has no '### N. Title' step headings")

    steps: list[ProcedureStep] = []
    for index, match in enumerate(headings):
        title = match.group("title").strip()
        if not title:
            raise ProcedureParseError("A step heading is missing its title")
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(steps_section)
        instruction = steps_section[start:end].strip()
        if not instruction:
            raise ProcedureParseError(f"Step '{title}' has no instruction text")
        step_id = normalize_identifier(title)
        input_refs = _extract_input_refs(instruction, valid_names, step_title=title)
        steps.append(ProcedureStep(id=step_id, title=title, instruction=instruction, input_refs=input_refs))
    return steps


def _validate_steps(steps: list[ProcedureStep]) -> None:
    if not steps:
        raise ProcedureParseError("Procedure must declare at least one step")
    ids = [step.id for step in steps]
    if len(ids) != len(set(ids)):
        raise ProcedureParseError(f"Duplicate step id(s) in '## Steps': {_duplicates(ids)}")


def parse_procedure_markdown(markdown: str) -> ProcedureDefinition:
    """Deterministically parse a KB procedure article into a ProcedureDefinition.

    Raises ProcedureParseError (a controlled, user-safe error) if the article
    does not follow the supported Markdown convention. Never falls back to an
    LLM to fill in missing or ambiguous structure.
    """
    title, description, sections = _split_sections(markdown)
    metadata = _parse_metadata(_require_section(sections, "procedure"))

    inputs = _parse_inputs(sections.get("required information"))
    _validate_inputs(inputs)

    valid_names = {item.name for item in inputs}
    steps = _parse_steps(_require_section(sections, "steps"), valid_names)
    _validate_steps(steps)

    success_criteria = sections.get("success")
    success_criteria = success_criteria.strip() if success_criteria else None

    return ProcedureDefinition(
        id=metadata["id"], version=metadata["version"], title=title, description=description,
        risk=metadata["risk"], confirmation_required=metadata["confirmation_required"],
        inputs=inputs, steps=steps, success_criteria=success_criteria,
    )
