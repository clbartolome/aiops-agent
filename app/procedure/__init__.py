"""Deterministic `/procedure` routing, knowledge-base retrieval, parsing, and
input collection.

Routing is a pure prefix check: only messages that start with the explicit
`/procedure` command enter this module as a *new* procedure. No LLM is used
to classify or route messages, and natural-language mentions of "procedure"
never reach here. Separately, while a procedure is actively waiting on
input, the *next* chat message belongs to that procedure instead of normal
chat — see `app.web` for the deterministic routing priority that uses the
helpers in this module.

The flow for this iteration is:

    /procedure request -> rag_search_kb -> KB found -> parse that exact KB
        -> apply declared defaults -> extract input values from the
        original request -> determine missing required inputs
        -> ask only for what is missing -> repeat on each reply
        -> READY (no step execution yet)

The KB is retrieved exactly once via the fixed `rag_search_kb` MCP tool; the
retrieved title/content is parsed deterministically with plain Python (see
`app.procedure.parser`). Input *values* may be extracted from free text with
a small, tool-less, structured-output model call (see
`app.procedure.extractor`), but which inputs are declared, which are
missing, and when the procedure is READY is always decided by plain Python
— never by the model.
"""
import json
import logging
import re
from uuid import uuid4

from agents.mcp import MCPServerManager

from app.config import Config
from app.diagnostics import log_failure
from app.mcp import create_mcp_servers
from app.procedure.extractor import extract_procedure_inputs
from app.procedure.models import (
    KBResult, PrimitiveValue, ProcedureContext, ProcedureDefinition, ProcedureInput,
    ProcedureRequest, ProcedureStatus,
)
from app.procedure.parser import ProcedureParseError, coerce_primitive, parse_procedure

logger = logging.getLogger(__name__)

PROCEDURE_PREFIX = "/procedure"
PROCEDURE_MARKER = "## Procedure"
RAG_SEARCH_TOOL = "rag_search_kb"
ITSM_SERVER_NAME = "itsm"
CANCEL_COMMAND = "/cancel"

EMPTY_QUERY_MESSAGE = "Please provide a procedure request after /procedure."
NO_RESULT_MESSAGE = "No matching procedure was found."
NOT_EXECUTABLE_MESSAGE = "KB found, but it is not an executable procedure."
MCP_FAILURE_MESSAGE = "Unable to retrieve a matching procedure from the knowledge base."
CANCELLED_MESSAGE = "Procedure cancelled."

# Matches "/procedure" alone, or "/procedure" followed by whitespace and the
# rest of the request. Anything else (e.g. "/proceduresomething") is rejected.
_COMMAND_PATTERN = re.compile(rf"^{re.escape(PROCEDURE_PREFIX)}(?:\s+(.*))?$", re.DOTALL)


class ProcedureKBError(RuntimeError):
    """A credential-free failure retrieving a KB procedure via rag_search_kb."""


def is_procedure_command(message: str) -> bool:
    """True only for messages explicitly starting with the `/procedure` command."""
    return _COMMAND_PATTERN.match(message.strip()) is not None


def is_cancel_command(message: str) -> bool:
    """True only for the exact, literal `/cancel` command."""
    return message.strip() == CANCEL_COMMAND


def extract_procedure_query(message: str) -> str:
    """Return the trimmed text after `/procedure` (empty if none was supplied)."""
    match = _COMMAND_PATTERN.match(message.strip())
    if not match:
        return ""
    return (match.group(1) or "").strip()


def is_executable_procedure(content: str) -> bool:
    """A KB article is an executable procedure only if it has a `## Procedure` section."""
    return PROCEDURE_MARKER in content


def _find_itsm_config(config: Config):
    return next((server for server in config.mcp_servers if server.name == ITSM_SERVER_NAME), None)


async def _call_rag_search_kb(query: str, config: Config) -> KBResult | None:
    """Call the fixed `rag_search_kb` MCP tool directly, bypassing the LLM.

    No other KB search tool is discovered or considered.
    """
    itsm_config = _find_itsm_config(config)
    if itsm_config is None:
        raise ProcedureKBError("The ITSM MCP server is not configured.")

    servers = create_mcp_servers((itsm_config,))
    async with MCPServerManager(servers, strict=True) as manager:
        server = manager.active_servers[0]
        result = await server.call_tool(RAG_SEARCH_TOOL, {"query": query})

    if result.is_error:
        raise ProcedureKBError("rag_search_kb returned an error result.")

    text = next((part.text for part in result.content if getattr(part, "text", None)), None)
    if text is None:
        return None

    payload = json.loads(text)
    results = payload.get("results") or []
    if not results:
        return None

    best = max(results, key=lambda item: item.get("score", 0))
    title = str(best.get("title") or "").strip()
    if not title:
        return None
    return KBResult(title=title, content=str(best.get("description") or ""))


def _format_parsed_summary(procedure: ProcedureDefinition) -> str:
    """Build the user-visible, human-readable parsed-procedure summary.

    Never renders raw Markdown or raw Pydantic/JSON output.
    """
    required = [item for item in procedure.inputs if item.required]
    optional = [item for item in procedure.inputs if not item.required]

    lines = ["Procedure parsed successfully.", ""]
    if required:
        lines.append("Required inputs:")
        lines.extend(f"- {item.label}" for item in required)
    else:
        lines.append("No required inputs.")
    if optional:
        lines.append("")
        lines.append("Optional inputs:")
        for item in optional:
            suffix = f" (default: {item.default})" if item.default is not None else ""
            lines.append(f"- {item.label}{suffix}")
    lines.append("")
    lines.append("Steps:")
    lines.extend(f"{index}. {step.title}" for index, step in enumerate(procedure.steps, start=1))
    return "\n".join(lines)


# --- Deterministic input bookkeeping (plain Python; no LLM decides this) ----

def _apply_defaults(procedure: ProcedureDefinition) -> dict[str, PrimitiveValue]:
    """Explicit KB defaults are applied before anything is extracted or asked."""
    return {item.name: item.default for item in procedure.inputs if item.default is not None}


def _filter_declared(extracted: dict, declared_names: set[str]) -> dict[str, PrimitiveValue]:
    """Drop any field the extractor returned that is not a declared input.

    This is the deterministic rejection of "unknown fields" required by the
    extraction contract; the model never decides what counts as valid.
    """
    return {name: value for name, value in extracted.items() if name in declared_names}


def _missing_required(procedure: ProcedureDefinition, collected: dict[str, PrimitiveValue]) -> list[ProcedureInput]:
    """Required inputs not yet present in `collected`, in declared order."""
    return [item for item in procedure.inputs if item.required and item.name not in collected]


def _ask_for_missing_message(missing: list[ProcedureInput]) -> str:
    if len(missing) == 1:
        label = missing[0].label.lower()
        return f"What {label} should I use?"
    lines = ["I still need:", ""]
    lines.extend(f"- {item.label}" for item in missing)
    return "\n".join(lines)


def _extraction_failed_message(targets: list[ProcedureInput]) -> str:
    """Recovery wording when the structured extractor call itself fails.

    The context/collected values are never lost; this only re-asks for what
    is still needed, directly from the procedure definition.
    """
    if len(targets) == 1:
        label = targets[0].label.lower()
        return f"I couldn't extract the {label} from that message.\n\nWhat {label} should I use?"
    return _ask_for_missing_message(targets)


def _ready_message(procedure: ProcedureDefinition, collected: dict[str, PrimitiveValue]) -> str:
    ordered = [item for item in procedure.inputs if item.name in collected]
    lines = ["Inputs collected:", ""]
    if ordered:
        lines.extend(f"- {item.label}: {collected[item.name]}" for item in ordered)
    else:
        lines.append("No inputs required.")
    lines.append("")
    lines.append("Procedure is ready to execute.")
    return "\n".join(lines)


def _secrets(config: Config) -> tuple[str | None, ...]:
    return (config.model_api_key, *(server.token for server in config.mcp_servers))


async def _extract_and_merge(
    run_id: str, text: str, targets: list[ProcedureInput], collected: dict[str, PrimitiveValue],
    config: Config,
) -> tuple[dict[str, PrimitiveValue], bool]:
    """Run the structured extractor restricted to `targets`, merge any valid,
    declared values into a *new* dict (existing values are preserved unless
    the extractor clearly supplies a replacement), and report whether
    extraction itself failed (as opposed to simply finding nothing).
    """
    logger.info("Procedure run=%s input_extraction=started", run_id)
    try:
        extracted = await extract_procedure_inputs(text, targets, config)
    except Exception as error:
        log_failure(logger, f"Procedure run={run_id} input extraction failed", error, _secrets(config))
        return collected, True

    declared_names = {item.name for item in targets}
    filtered = _filter_declared(extracted, declared_names)
    logger.info("Procedure run=%s inputs_extracted=%s", run_id, sorted(filtered))
    merged = {**collected, **filtered}
    return merged, False


def _settle(context: ProcedureContext, collected: dict[str, PrimitiveValue],
            extraction_failed: bool, failed_targets: list[ProcedureInput]) -> ProcedureStatus:
    """Compute missing inputs deterministically and build the final status
    (asking for what remains, or READY) plus the context to persist.
    """
    procedure = context.procedure
    missing = _missing_required(procedure, collected)
    logger.info("Procedure run=%s missing_inputs=%s", context.run_id, [item.name for item in missing])

    if missing:
        status_value = "COLLECTING_INPUTS"
        message = (_extraction_failed_message(failed_targets) if extraction_failed
                   else _ask_for_missing_message(missing))
    else:
        status_value = "READY"
        message = _ready_message(procedure, collected)

    logger.info("Procedure run=%s status=%s", context.run_id, status_value)
    new_context = context.model_copy(update={"inputs": collected, "status": status_value})
    return ProcedureStatus(
        state=status_value.lower(), message=message, title=context.kb_title, context=new_context,
    )


def _parse_kb_procedure(kb: KBResult) -> tuple[ProcedureDefinition | None, str, str]:
    """Parse the already-retrieved KB and build the "KB found" + parsed/
    error messages. Returns (procedure_or_None, kb_found_message, outcome).

    `outcome` is the parsed summary on success, or the parse-error text on
    failure; `procedure` is None exactly when parsing failed.
    """
    kb_found_message = f"KB found: {kb.title}"
    logger.info("Procedure parsing started title=%s", kb.title)
    try:
        procedure = parse_procedure(kb.content)
    except ProcedureParseError as error:
        logger.error("Procedure parsing failed reason=%s", error)
        return None, kb_found_message, f"Unable to parse the procedure.\n\n{error}"

    logger.info(
        "Procedure parsed procedure=%s version=%d inputs=%s steps=%s",
        procedure.id, procedure.version,
        [item.name for item in procedure.inputs], [step.id for step in procedure.steps],
    )
    return procedure, kb_found_message, _format_parsed_summary(procedure)


async def handle_procedure(message: str, config: Config) -> ProcedureStatus:
    """Start a brand-new `/procedure` run and return a UI-facing status.

    Flow: extract the query -> call `rag_search_kb` once -> parse the KB
    deterministically (no LLM, no second MCP call) -> apply declared
    defaults -> extract any remaining input values from the original
    request text -> ask for whatever required input is still missing, or
    report READY. Does not execute any step yet.
    """
    query = extract_procedure_query(message)
    if not query:
        return ProcedureStatus(state="empty_query", message=EMPTY_QUERY_MESSAGE)

    try:
        kb = await _call_rag_search_kb(query, config)
    except Exception as error:
        log_failure(logger, "Procedure KB retrieval failed", error, _secrets(config))
        return ProcedureStatus(state="mcp_error", message=MCP_FAILURE_MESSAGE)

    request = ProcedureRequest(original_message=message, query=query, kb=kb)

    if request.kb is None:
        return ProcedureStatus(state="no_result", message=NO_RESULT_MESSAGE)

    if not is_executable_procedure(request.kb.content):
        return ProcedureStatus(
            state="not_executable", message=NOT_EXECUTABLE_MESSAGE, title=request.kb.title,
        )

    procedure, kb_found_message, outcome = _parse_kb_procedure(request.kb)
    if procedure is None:
        return ProcedureStatus(
            state="parse_error", message=outcome, title=request.kb.title,
            messages=(kb_found_message, outcome),
        )
    parsed_summary = outcome

    run_id = str(uuid4())
    collected = _apply_defaults(procedure)
    remaining = [item for item in procedure.inputs if item.name not in collected]

    extraction_failed = False
    if remaining and query:
        collected, extraction_failed = await _extract_and_merge(run_id, query, remaining, collected, config)

    context = ProcedureContext(
        run_id=run_id, original_request=query, kb_title=request.kb.title, kb_content=request.kb.content,
        procedure=procedure, inputs=collected, status="COLLECTING_INPUTS",
    )
    outcome_status = _settle(context, collected, extraction_failed, remaining)

    return ProcedureStatus(
        state=outcome_status.state, message=outcome_status.message, title=request.kb.title,
        messages=(kb_found_message, parsed_summary, outcome_status.message),
        progress_stages=("Parsing procedure", "Extracting inputs"),
        context=outcome_status.context,
    )


async def handle_procedure_input_reply(message: str, context: ProcedureContext, config: Config) -> ProcedureStatus:
    """Continue an active procedure that is waiting on input.

    Only called while `context.status == "COLLECTING_INPUTS"`. When exactly
    one input is missing, a plain-text reply is stored directly for that
    field with no LLM call (the deterministic single-field shortcut).
    Otherwise, a small structured extractor is used, restricted to the
    fields that are still missing; existing collected values are always
    preserved unless this reply clearly supplies a replacement.
    """
    logger.info("Procedure run=%s input_reply_received", context.run_id)
    missing = _missing_required(context.procedure, context.inputs)

    if len(missing) == 1:
        reply = message.strip()
        if not reply:
            return _settle(context, context.inputs, extraction_failed=False, failed_targets=missing)
        item = missing[0]
        collected = {**context.inputs, item.name: coerce_primitive(reply)}
        logger.info("Procedure run=%s inputs_extracted=%s", context.run_id, [item.name])
        return _settle(context, collected, extraction_failed=False, failed_targets=missing)

    collected, extraction_failed = await _extract_and_merge(
        context.run_id, message, missing, context.inputs, config,
    )
    return _settle(context, collected, extraction_failed, missing)


def handle_cancel(context: ProcedureContext) -> ProcedureStatus:
    """Cancel the active procedure. The context is cleared (not persisted)."""
    logger.info("Procedure run=%s status=CANCELLED", context.run_id)
    return ProcedureStatus(state="cancelled", message=CANCELLED_MESSAGE, title=context.kb_title, context=None)
