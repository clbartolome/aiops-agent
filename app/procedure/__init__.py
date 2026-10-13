"""Deterministic `/procedure` routing, knowledge-base retrieval, parsing,
input collection, and sequential step execution -- delivered live.

Routing is a pure prefix check: only messages that start with the explicit
`/procedure` command enter this module as a *new* procedure. No LLM is used
to classify or route messages, and natural-language mentions of "procedure"
never reach here. Separately, while a procedure is actively waiting on
input, the *next* chat message belongs to that procedure instead of normal
chat — see `app.web` for the deterministic routing priority that uses the
helpers in this module.

The flow is:

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
        -> once terminal (COMPLETED/STOPPED/FAILED), generate one final
           natural-language summary (see `app.procedure.summarizer`) and
           show it as the last chat message

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

-- Live progress delivery --

`handle_procedure` and `handle_procedure_input_reply` (and every private
helper they call that can take a noticeable amount of time: KB retrieval,
parsing, parameter extraction, each step, and final-summary generation)
are plain `async def` generator functions. Each one `yield`s, in exact
execution order:

  - zero or more wire-shaped event dicts (`_progress`/`_PROGRESS_CLEAR`/
    `_message` below) -- these are sent to the client *immediately*, as
    they happen, interleaved with the real `await`s doing the actual work
    (an MCP call, a model call, the Step Executor); nothing is batched up
    and returned only once the whole procedure finishes.
  - exactly one `ProcedureStatus` as the very last item, carrying the
    terminal `state`/`message`/`context` for this turn (see
    `app.web._stream_procedure`, which drains one of these generators
    over a streaming HTTP response and persists `.context` on the chat
    session once the generator is exhausted).

This is deliberately *not* a generic event bus: every event is produced
by the one coroutine that is also doing the corresponding work, in a
plain top-to-bottom generator -- no separate event loop, task, or queue
is introduced. A lower-level generator's own non-dict "return value" is
threaded up to its caller using the small `_Collector` helper below, since
Python async generators cannot both `yield` and meaningfully `return` a
value to an `async for` loop.
"""
import json
import logging
import re
from collections.abc import AsyncIterator
from typing import TypeVar
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
from app.procedure.summarizer import generate_procedure_summary

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

SEARCHING_KB_STAGE = "Searching KB"
PARSING_PROCEDURE_STAGE = "Parsing procedure"
EXTRACTING_PARAMETERS_STAGE = "Extracting parameters"
SUMMARY_STAGE = "Generating procedure summary"

# Matches "/procedure" alone, or "/procedure" followed by whitespace and the
# rest of the request. Anything else (e.g. "/proceduresomething") is rejected.
_COMMAND_PATTERN = re.compile(rf"^{re.escape(PROCEDURE_PREFIX)}(?:\s+(.*))?$", re.DOTALL)


class ProcedureKBError(RuntimeError):
    """A credential-free failure retrieving a KB procedure via rag_search_kb."""


# --- Wire-shaped progress/message events -------------------------------------
# Exactly the three event shapes `app.web`/`app.index.html` understand:
# a transient progress label, clearing it, and one permanent chat message.
# `ProcedureStatus` (the only non-dict item ever yielded) is never sent to
# the wire directly -- `app.web._stream_procedure` turns it into the one
# terminal `procedure_status` line instead.
ProcedureEvent = dict[str, object]


def _progress(text: str) -> ProcedureEvent:
    return {"type": "procedure_progress", "message": text}


_PROGRESS_CLEAR: ProcedureEvent = {"type": "procedure_progress_clear"}


def _message(text: str) -> ProcedureEvent:
    return {"type": "procedure_message", "message": text}


_T = TypeVar("_T")


class _Collector:
    """Mutable, single-use sink for one lower-level event generator's final
    non-dict "return value", filled in by `forward()`.

    This only exists because Python async generators cannot both `yield`
    events and `return` a value an `async for` loop can see; it is not a
    general-purpose event bus, just the one small plumbing trick every
    generator below needs to call another one -- and it is deliberately
    *not* a function that drains the inner generator into a list before
    returning: `forward()` re-yields each dict the instant the inner
    generator produces it, which is exactly what keeps every event live
    all the way up to `app.web`'s streaming response instead of being
    buffered and released only once the whole procedure finishes.
    """

    def __init__(self):
        self.value: _T | None = None

    async def forward(self, events: AsyncIterator[ProcedureEvent | _T]) -> AsyncIterator[ProcedureEvent]:
        async for item in events:
            if isinstance(item, dict):
                yield item
            else:
                self.value = item


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


def _parsed_summary_message(procedure: ProcedureDefinition) -> str:
    """A compact, user-visible confirmation that the KB was parsed.

    Deliberately terse: full required/optional input labels and the step
    list are internal/log detail (see `_parse_kb_procedure`'s logging),
    not a permanent chat message. Keeps the default chat UX free of
    technical/debug-shaped noise.
    """
    count = len(procedure.steps)
    noun = "step" if count == 1 else "steps"
    return f"Procedure loaded: {count} {noun}."


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
    """A compact, presentation-only confirmation of the inputs that will be
    used, shown right before execution begins (never the full verbose
    per-field breakdown -- that would be unnecessary chat noise once every
    step result is about to follow anyway).
    """
    ordered = [item for item in procedure.inputs if item.name in collected]
    if not ordered:
        return "Procedure is ready to execute."
    pairs = ", ".join(f"{item.label}: {collected[item.name]}" for item in ordered)
    return f"Using {pairs}."


def _secrets(config: Config) -> tuple[str | None, ...]:
    return (config.model_api_key, *(server.token for server in config.mcp_servers))


# --- Step execution: a generic loop over every `procedure.steps` entry, in
# fixed KB order. The model never chooses, skips, reorders, or invents a
# step; this module alone decides which step runs next, and whether
# execution continues at all, purely from `StepResult.status`.

STEP_EXECUTION_ERROR_SUMMARY = "Unable to complete this step."

# STOP/FAILED map directly onto the matching terminal context status; SUCCESS
# has no terminal status of its own here -- the loop only reaches COMPLETED
# once every step has returned SUCCESS.
_TERMINAL_CONTEXT_STATUS = {"STOP": "STOPPED", "FAILED": "FAILED"}


def _step_result_message(result: StepResult) -> str:
    """The permanent chat message for one step's outcome: just the concise
    `StepResult.summary` itself (no "Step completed:"/id-shaped prefix --
    the preceding "Running step N: <title>" progress bubble already named
    the step, so repeating it here would be redundant chat noise).
    """
    return result.summary


def _fallback_summary(
    procedure: ProcedureDefinition, step_results: dict[str, StepResult], final_status: str,
) -> str:
    """Deterministic, LLM-free summary used only if `generate_procedure_summary`
    fails. Presentation-only: it never changes `final_status` itself, and
    never invents anything beyond already-stored step results/counts.
    """
    succeeded = sum(
        1 for step in procedure.steps
        if step_results.get(step.id) is not None and step_results[step.id].status == "SUCCESS"
    )
    if final_status == "COMPLETED":
        total = len(procedure.steps)
        return f"Procedure completed successfully.\n\n{succeeded} of {total} steps completed."

    stopping_step = next(
        (step for step in procedure.steps
         if step_results.get(step.id) is not None and step_results[step.id].status in ("STOP", "FAILED")),
        None,
    )
    verb = "stopped" if final_status == "STOPPED" else "failed"
    where = f" at: {stopping_step.title}." if stopping_step is not None else "."
    plural = "" if succeeded == 1 else "s"
    return f"Procedure {verb}{where}\n\n{succeeded} previous step{plural} completed successfully."


async def _generate_summary(
    context: ProcedureContext, final_context: ProcedureContext, config: Config,
) -> str:
    """Run the final-summary LLM call and fall back to a deterministic
    message (never changing `final_context.status`) if it fails or
    returns nothing usable. A plain `async def` (not a generator): the
    caller (`_settle`) is the one that shows/clears the "Generating
    summary" progress bubble around this one `await`.
    """
    logger.info("Procedure run=%s summary_generation=started", context.run_id)
    try:
        summary = await generate_procedure_summary(final_context, config)
    except Exception as error:
        log_failure(logger, f"Procedure run={context.run_id} summary generation failed", error, _secrets(config))
        summary = None

    if summary is None:
        logger.info("Procedure run=%s summary_generation=fallback", context.run_id)
        return _fallback_summary(context.procedure, final_context.step_results, final_context.status)

    logger.info("Procedure run=%s summary_generation=llm", context.run_id)
    return summary


async def _run_one_step(
    context: ProcedureContext, step: ProcedureStep, previous_results: list[StepExecutionContext], config: Config,
) -> AsyncIterator[ProcedureEvent | tuple[StepResult, ProcedureContext]]:
    """Execute exactly the one given step, yielding its progress/clear/
    result events live around the (potentially slow) Step Executor call,
    and ending with `(StepResult, updated_context)` as its one "return
    value" (see `_Collector`). `previous_results` is compact already-
    completed-step context (title/status/summary only); it never changes
    which step runs, only what this one step is told.

    Only `step_results` is updated here; `current_step_index` and the
    procedure's own `status` are owned by the calling loop
    (`_execute_steps`), since only it knows whether this was the last
    step, whether execution should continue, and -- on STOP/FAILED --
    that no later step may start.

    `execute_step`'s own exceptions are always converted to a FAILED
    `StepResult` here (never left to propagate), so the progress-clear
    event below is always reached, success or failure.
    """
    procedure = context.procedure
    # Human-friendly, 1-based step numbering; internal step IDs are never
    # shown to the user.
    step_number = procedure.steps.index(step) + 1
    yield _progress(f"Running step {step_number}: {step.title}")

    try:
        result = await execute_step(
            context.run_id, step, procedure.inputs, context.inputs, context.original_request, config,
            previous_results=previous_results,
        )
    except Exception as error:
        log_failure(logger, f"Procedure run={context.run_id} step={step.id} execution failed", error, _secrets(config))
        result = StepResult(status="FAILED", summary=STEP_EXECUTION_ERROR_SUMMARY)

    yield _PROGRESS_CLEAR
    yield _message(_step_result_message(result))

    new_context = context.model_copy(update={
        # Never overwrite an earlier step's stored result; preserve order.
        "step_results": {**context.step_results, step.id: result},
    })
    yield (result, new_context)


async def _execute_steps(
    context: ProcedureContext, config: Config,
) -> AsyncIterator[ProcedureEvent | ProcedureContext]:
    """Execute `procedure.steps` in fixed KB order, one at a time, stopping
    immediately on the first non-SUCCESS result, yielding every step's
    progress/clear/result events live as each step runs (see
    `_run_one_step`) and ending with the final `ProcedureContext` as this
    generator's one "return value". Only called once `context.status ==
    "READY"`.

    Each step receives every earlier *successful* step's result as compact
    previous-step context (title/status/summary only), always in original
    execution order. `current_step_index` starts at 0 and only advances
    past a step once it has returned SUCCESS, so on STOP/FAILED it is left
    pointing at the step that stopped/failed the procedure.
    """
    procedure = context.procedure
    logger.info("Procedure run=%s status=RUNNING steps=%d", context.run_id, len(procedure.steps))
    current_context = context.model_copy(update={"status": "RUNNING", "current_step_index": 0})
    completed: list[StepExecutionContext] = []

    while current_context.current_step_index < len(procedure.steps):
        index = current_context.current_step_index
        step = procedure.steps[index]
        logger.info("Procedure run=%s step_index=%d step=%s status=starting", context.run_id, index, step.id)

        collector = _Collector()
        async for event in collector.forward(_run_one_step(current_context, step, list(completed), config)):
            yield event
        result, current_context = collector.value
        logger.info("Procedure run=%s step=%s status=%s", context.run_id, step.id, result.status)

        if result.status == "SUCCESS":
            completed.append(StepExecutionContext(step=step, result=result))
            current_context = current_context.model_copy(update={"current_step_index": index + 1})
            continue

        terminal_status = _TERMINAL_CONTEXT_STATUS[result.status]
        current_context = current_context.model_copy(update={"status": terminal_status})
        logger.info("Procedure run=%s status=%s step=%s", context.run_id, terminal_status, step.id)
        yield current_context
        return

    current_context = current_context.model_copy(update={"status": "COMPLETED"})
    logger.info("Procedure run=%s status=COMPLETED", context.run_id)
    yield current_context


async def _extract_and_merge(
    run_id: str, text: str, targets: list[ProcedureInput], collected: dict[str, PrimitiveValue],
    config: Config,
) -> tuple[dict[str, PrimitiveValue], bool]:
    """Run the structured extractor restricted to `targets`, merge any valid,
    declared values into a *new* dict (existing values are preserved unless
    the extractor clearly supplies a replacement), and report whether
    extraction itself failed (as opposed to simply finding nothing). A
    plain `async def` (not a generator): the caller shows/clears the
    "Extracting parameters" progress bubble around this one `await`.
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
                   extraction_failed: bool, failed_targets: list[ProcedureInput],
                   config: Config) -> AsyncIterator[ProcedureEvent | ProcedureStatus]:
    """Compute missing inputs deterministically and settle the turn live:
    asking for what remains, or -- once READY -- running every procedure
    step in order (see `_execute_steps`), then generating and showing one
    natural-language summary once the procedure reaches a terminal state
    (COMPLETED/STOPPED/FAILED). Ends with the terminal `ProcedureStatus`.
    """
    procedure = context.procedure
    missing = _missing_required(procedure, collected)
    logger.info("Procedure run=%s missing_inputs=%s", context.run_id, [item.name for item in missing])

    if missing:
        logger.info("Procedure run=%s status=COLLECTING_INPUTS", context.run_id)
        message = (_extraction_failed_message(failed_targets) if extraction_failed
                   else _ask_for_missing_message(missing))
        new_context = context.model_copy(update={"inputs": collected, "status": "COLLECTING_INPUTS"})
        yield _message(message)
        yield ProcedureStatus(state="collecting_inputs", message=message, title=context.kb_title, context=new_context)
        return

    logger.info("Procedure run=%s status=READY", context.run_id)
    ready_context = context.model_copy(update={"inputs": collected, "status": "READY"})
    yield _message(_ready_message(procedure, collected))

    collector = _Collector()
    async for event in collector.forward(_execute_steps(ready_context, config)):
        yield event
    final_context = collector.value

    yield _progress(SUMMARY_STAGE)
    summary_message = await _generate_summary(context, final_context, config)
    yield _PROGRESS_CLEAR
    yield _message(summary_message)

    yield ProcedureStatus(
        state=final_context.status.lower(), message=summary_message, title=context.kb_title, context=final_context,
    )


def _parse_kb_procedure(kb: KBResult) -> tuple[ProcedureDefinition | None, str]:
    """Parse the already-retrieved KB. Returns (procedure_or_None, outcome):
    `outcome` is the compact parsed-summary message on success, or the
    parse-error text on failure; `procedure` is `None` exactly when
    parsing failed. A plain function (synchronous, no `await`): parsing
    is effectively instant, so there is nothing to show progress for
    other than the "Parsing procedure" bubble the caller shows/clears
    immediately around this call.
    """
    logger.info("Procedure parsing started title=%s", kb.title)
    try:
        procedure = parse_procedure(kb.content)
    except ProcedureParseError as error:
        logger.error("Procedure parsing failed reason=%s", error)
        return None, f"Unable to parse the procedure.\n\n{error}"

    logger.info(
        "Procedure parsed procedure=%s version=%d inputs=%s steps=%s",
        procedure.id, procedure.version,
        [item.name for item in procedure.inputs], [step.id for step in procedure.steps],
    )
    return procedure, _parsed_summary_message(procedure)


async def handle_procedure(message: str, config: Config) -> AsyncIterator[ProcedureEvent | ProcedureStatus]:
    """Start a brand-new `/procedure` run, yielding every progress/message
    event live as it happens (see this module's docstring), and ending
    with the terminal `ProcedureStatus` for this turn.

    Flow: extract the query -> call `rag_search_kb` once -> parse the KB
    deterministically (no LLM, no second MCP call) -> apply declared
    defaults -> extract any remaining input values from the original
    request text -> ask for whatever required input is still missing, or
    -- once READY -- execute every step in turn and report each result.
    """
    query = extract_procedure_query(message)
    if not query:
        yield _message(EMPTY_QUERY_MESSAGE)
        yield ProcedureStatus(state="empty_query", message=EMPTY_QUERY_MESSAGE)
        return

    yield _progress(SEARCHING_KB_STAGE)
    try:
        kb = await _call_rag_search_kb(query, config)
    except Exception as error:
        log_failure(logger, "Procedure KB retrieval failed", error, _secrets(config))
        yield _PROGRESS_CLEAR
        yield _message(MCP_FAILURE_MESSAGE)
        yield ProcedureStatus(state="mcp_error", message=MCP_FAILURE_MESSAGE)
        return
    yield _PROGRESS_CLEAR

    request = ProcedureRequest(original_message=message, query=query, kb=kb)

    if request.kb is None:
        yield _message(NO_RESULT_MESSAGE)
        yield ProcedureStatus(state="no_result", message=NO_RESULT_MESSAGE)
        return

    yield _message(f"KB found: {request.kb.title}")

    if not is_executable_procedure(request.kb.content):
        yield ProcedureStatus(state="not_executable", message=NOT_EXECUTABLE_MESSAGE, title=request.kb.title)
        return

    yield _progress(PARSING_PROCEDURE_STAGE)
    procedure, outcome = _parse_kb_procedure(request.kb)
    yield _PROGRESS_CLEAR
    yield _message(outcome)

    if procedure is None:
        yield ProcedureStatus(state="parse_error", message=outcome, title=request.kb.title)
        return

    run_id = str(uuid4())
    collected = _apply_defaults(procedure)
    remaining = [item for item in procedure.inputs if item.name not in collected]

    extraction_failed = False
    if remaining and query:
        yield _progress(EXTRACTING_PARAMETERS_STAGE)
        collected, extraction_failed = await _extract_and_merge(run_id, query, remaining, collected, config)
        yield _PROGRESS_CLEAR

    context = ProcedureContext(
        run_id=run_id, original_request=query, kb_title=request.kb.title, kb_content=request.kb.content,
        procedure=procedure, inputs=collected, status="COLLECTING_INPUTS",
    )
    collector = _Collector()
    async for event in collector.forward(_settle(context, collected, extraction_failed, remaining, config)):
        yield event
    yield collector.value


async def handle_procedure_input_reply(
    message: str, context: ProcedureContext, config: Config,
) -> AsyncIterator[ProcedureEvent | ProcedureStatus]:
    """Continue an active procedure that is waiting on input, yielding
    every progress/message event live and ending with the terminal
    `ProcedureStatus` for this turn.

    Only called while `context.status == "COLLECTING_INPUTS"`. When exactly
    one input is missing, a plain-text reply is stored directly for that
    field with no LLM call (the deterministic single-field shortcut, which
    shows no "Extracting parameters" progress at all since nothing is
    actually extracted). Otherwise, a small structured extractor is used,
    restricted to the fields that are still missing; existing collected
    values are always preserved unless this reply clearly supplies a
    replacement.
    """
    logger.info("Procedure run=%s input_reply_received", context.run_id)
    missing = _missing_required(context.procedure, context.inputs)

    if not missing:
        # Nothing left to extract -- settle directly with no extraction
        # progress at all (there would be nothing for it to do).
        collector = _Collector()
        async for event in collector.forward(
            _settle(context, context.inputs, extraction_failed=False, failed_targets=missing, config=config),
        ):
            yield event
        yield collector.value
        return

    if len(missing) == 1:
        reply = message.strip()
        if reply:
            item = missing[0]
            collected = {**context.inputs, item.name: coerce_primitive(reply)}
            logger.info("Procedure run=%s inputs_extracted=%s", context.run_id, [item.name])
        else:
            collected = context.inputs
        collector = _Collector()
        async for event in collector.forward(
            _settle(context, collected, extraction_failed=False, failed_targets=missing, config=config),
        ):
            yield event
        yield collector.value
        return

    yield _progress(EXTRACTING_PARAMETERS_STAGE)
    collected, extraction_failed = await _extract_and_merge(context.run_id, message, missing, context.inputs, config)
    yield _PROGRESS_CLEAR

    collector = _Collector()
    async for event in collector.forward(_settle(context, collected, extraction_failed, missing, config)):
        yield event
    yield collector.value


def handle_cancel(context: ProcedureContext) -> ProcedureStatus:
    """Cancel the active procedure. The context is cleared (not persisted).

    Instantaneous and synchronous -- no progress bubble is ever needed
    for this, so (unlike `handle_procedure`/`handle_procedure_input_reply`)
    this stays a plain function, not a generator.
    """
    logger.info("Procedure run=%s status=CANCELLED", context.run_id)
    return ProcedureStatus(state="cancelled", message=CANCELLED_MESSAGE, title=context.kb_title, context=None)
