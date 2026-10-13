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
        -> READY -> RUNNING -> execute procedure.steps in fixed order,
           one at a time, each step receiving every earlier successful
           step's result as compact previous-step context
        -> continue only while each step returns SUCCESS
        -> stop immediately on STOP or FAILED, preserving every result
           already stored
        -> COMPLETED once every step has succeeded
        -> store each result -> show each in chat

The KB is retrieved exactly once via the fixed `rag_search_kb` MCP tool; the
retrieved title/content is parsed deterministically with plain Python (see
`app.procedure.parser`). Input *values* may be extracted from free text with
a small, tool-less, structured-output model call (see
`app.procedure.extractor`), but which inputs are declared, which are
missing, and when the procedure is READY is always decided by plain Python
— never by the model. Once READY, every step is executed in turn by the
same dedicated Step Executor (see `app.procedure.executor`), in a single
generic loop over `procedure.steps`; which step runs next, and whether
execution continues at all, is always decided here in plain Python from
`StepResult.status`, never by the model. The model never reorders, skips,
or invents a step, and is only ever given the one current step.
"""
import json
import logging
import re
from uuid import uuid4

from agents.mcp import MCPServerManager

from app.config import Config
from app.diagnostics import log_failure
from app.mcp import create_mcp_servers
from app.procedure.executor import execute_step
from app.procedure.extractor import extract_procedure_inputs
from app.procedure.models import (
    KBResult, PrimitiveValue, ProcedureContext, ProcedureDefinition, ProcedureInput, ProcedureStep,
    ProcedureRequest, ProcedureStatus, StepExecutionContext, StepResult,
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


# --- Step execution: a generic loop over every `procedure.steps` entry, in
# fixed KB order. The model never chooses, skips, reorders, or invents a
# step; this module alone decides which step runs next, and whether
# execution continues at all, purely from `StepResult.status`.

STEP_RESULT_MESSAGE_PREFIX = {
    "SUCCESS": "Step completed", "STOP": "Step stopped the procedure", "FAILED": "Step failed",
}
STEP_EXECUTION_ERROR_SUMMARY = "Unable to complete this step."
PROCEDURE_COMPLETED_MESSAGE = "Procedure completed successfully."
# Shown as the transient progress bubble right before the final completion
# message -- there is no "next step" to run, so there is nothing step-like
# to name; this keeps the existing one-stage-per-revealed-message UI
# contract intact without any frontend change.
PROCEDURE_COMPLETING_STAGE = "Finishing up..."

# STOP/FAILED map directly onto the matching terminal context status; SUCCESS
# has no terminal status of its own here -- the loop only reaches COMPLETED
# once every step has returned SUCCESS.
_TERMINAL_CONTEXT_STATUS = {"STOP": "STOPPED", "FAILED": "FAILED"}


def _step_result_message(step: ProcedureStep, result: StepResult) -> str:
    return f"{STEP_RESULT_MESSAGE_PREFIX[result.status]}: {step.title}\n\n{result.summary}"


async def _run_one_step(
    context: ProcedureContext, step: ProcedureStep, previous_results: list[StepExecutionContext], config: Config,
) -> tuple[str, str, StepResult, ProcedureContext]:
    """Execute exactly the one given step and return (progress label, user
    message, its `StepResult`, updated context). `previous_results` is
    compact already-completed-step context (title/status/summary only);
    it never changes which step runs, only what this one step is told.

    Only `step_results` is updated here; `current_step_index` and the
    procedure's own `status` are owned by the calling loop (`_execute_steps`),
    since only it knows whether this was the last step, whether execution
    should continue, and -- on STOP/FAILED -- that no later step may start.
    """
    procedure = context.procedure
    progress_stage = f"Running: {step.title}..."

    try:
        result = await execute_step(
            context.run_id, step, procedure.inputs, context.inputs, context.original_request, config,
            previous_results=previous_results,
        )
    except Exception as error:
        log_failure(logger, f"Procedure run={context.run_id} step={step.id} execution failed", error, _secrets(config))
        result = StepResult(status="FAILED", summary=STEP_EXECUTION_ERROR_SUMMARY)

    message = _step_result_message(step, result)
    new_context = context.model_copy(update={
        # Never overwrite an earlier step's stored result; preserve order.
        "step_results": {**context.step_results, step.id: result},
    })
    return progress_stage, message, result, new_context


async def _execute_steps(
    context: ProcedureContext, config: Config,
) -> tuple[tuple[str, ...], tuple[str, ...], ProcedureContext]:
    """Execute `procedure.steps` in fixed KB order, one at a time, stopping
    immediately on the first non-SUCCESS result. Only called once
    `context.status == "READY"`.

    Each step receives every earlier *successful* step's result as compact
    previous-step context (title/status/summary only), always in original
    execution order. `current_step_index` starts at 0 and only advances
    past a step once it has returned SUCCESS, so on STOP/FAILED it is left
    pointing at the step that stopped/failed the procedure.
    """
    procedure = context.procedure
    logger.info("Procedure run=%s status=RUNNING steps=%d", context.run_id, len(procedure.steps))
    current_context = context.model_copy(update={"status": "RUNNING", "current_step_index": 0})
    stages: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()
    completed: list[StepExecutionContext] = []

    while current_context.current_step_index < len(procedure.steps):
        index = current_context.current_step_index
        step = procedure.steps[index]
        logger.info("Procedure run=%s step_index=%d step=%s status=starting", context.run_id, index, step.id)

        stage, message, result, current_context = await _run_one_step(
            current_context, step, list(completed), config,
        )
        stages += (stage,)
        messages += (message,)
        logger.info("Procedure run=%s step=%s status=%s", context.run_id, step.id, result.status)

        if result.status == "SUCCESS":
            completed.append(StepExecutionContext(step=step, result=result))
            current_context = current_context.model_copy(update={"current_step_index": index + 1})
            continue

        terminal_status = _TERMINAL_CONTEXT_STATUS[result.status]
        current_context = current_context.model_copy(update={"status": terminal_status})
        logger.info("Procedure run=%s status=%s step=%s", context.run_id, terminal_status, step.id)
        return stages, messages, current_context

    current_context = current_context.model_copy(update={"status": "COMPLETED"})
    logger.info("Procedure run=%s status=COMPLETED", context.run_id)
    return stages, messages, current_context


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


async def _settle(context: ProcedureContext, collected: dict[str, PrimitiveValue],
                   extraction_failed: bool, failed_targets: list[ProcedureInput], config: Config) -> ProcedureStatus:
    """Compute missing inputs deterministically and build the final status:
    asking for what remains, or -- once READY -- running every procedure
    step in order (see `_execute_steps`) and folding each step's result
    into the response as an additional sequential chat message.
    """
    procedure = context.procedure
    missing = _missing_required(procedure, collected)
    logger.info("Procedure run=%s missing_inputs=%s", context.run_id, [item.name for item in missing])

    if missing:
        logger.info("Procedure run=%s status=COLLECTING_INPUTS", context.run_id)
        message = (_extraction_failed_message(failed_targets) if extraction_failed
                   else _ask_for_missing_message(missing))
        new_context = context.model_copy(update={"inputs": collected, "status": "COLLECTING_INPUTS"})
        return ProcedureStatus(
            state="collecting_inputs", message=message, title=context.kb_title, context=new_context,
        )

    logger.info("Procedure run=%s status=READY", context.run_id)
    ready_message = _ready_message(procedure, collected)
    ready_context = context.model_copy(update={"inputs": collected, "status": "READY"})

    stages, step_messages, final_context = await _execute_steps(ready_context, config)
    if final_context.status == "COMPLETED":
        stages += (PROCEDURE_COMPLETING_STAGE,)
        step_messages += (PROCEDURE_COMPLETED_MESSAGE,)

    return ProcedureStatus(
        state=final_context.status.lower(), message=step_messages[-1], title=context.kb_title,
        messages=(ready_message, *step_messages), progress_stages=stages,
        context=final_context,
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
    -- once READY -- execute `procedure.steps[0]` and report its result.
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
    outcome_status = await _settle(context, collected, extraction_failed, remaining, config)

    # `_settle` may itself produce more than one sequential message/stage
    # once READY (the "ready" message followed by the step-1 result); fold
    # those onto this request's own KB-found/parsed sequence.
    tail_messages = outcome_status.messages or (outcome_status.message,)
    tail_stages = outcome_status.progress_stages or ()

    return ProcedureStatus(
        state=outcome_status.state, message=outcome_status.message, title=request.kb.title,
        messages=(kb_found_message, parsed_summary, *tail_messages),
        progress_stages=("Parsing procedure", "Extracting inputs", *tail_stages),
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
            return await _settle(context, context.inputs, extraction_failed=False, failed_targets=missing, config=config)
        item = missing[0]
        collected = {**context.inputs, item.name: coerce_primitive(reply)}
        logger.info("Procedure run=%s inputs_extracted=%s", context.run_id, [item.name])
        return await _settle(context, collected, extraction_failed=False, failed_targets=missing, config=config)

    collected, extraction_failed = await _extract_and_merge(
        context.run_id, message, missing, context.inputs, config,
    )
    return await _settle(context, collected, extraction_failed, missing, config)


def handle_cancel(context: ProcedureContext) -> ProcedureStatus:
    """Cancel the active procedure. The context is cleared (not persisted)."""
    logger.info("Procedure run=%s status=CANCELLED", context.run_id)
    return ProcedureStatus(state="cancelled", message=CANCELLED_MESSAGE, title=context.kb_title, context=None)
