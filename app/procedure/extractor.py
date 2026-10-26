"""Focused structured extraction of declared procedure input values.

This is not an autonomous agent: it has no tools (MCP or otherwise), never
executes anything, and never answers the user conversationally. It is given
the user's text and only the currently-declared/missing procedure inputs,
and must return a small structured object naming only values it is
confident about. All validation of the result (declared-field membership,
supported types, missing-value detection) happens afterward in plain
Python — see `app.procedure` — never in the model.
"""
import re

from agents import AgentOutputSchema, Agent, ModelSettings, OpenAIChatCompletionsModel, RunConfig, Runner
from openai import AsyncOpenAI

from pydantic import BaseModel, ConfigDict, Field, create_model

from app.config import Config
from app.procedure.models import PrimitiveValue, ProcedureInput
from app.procedure.parser import coerce_primitive

EXTRACTION_INSTRUCTIONS = """Extract values only for the declared procedure inputs.

Use the internal field names exactly as provided.

Use only values explicitly stated or unambiguously implied by the user's message.

Do not guess missing values.
Do not invent fields that are not declared.
If a value is not clearly provided, omit it from the result.

You have no tools. Never execute anything and never answer the user
conversationally; only return the structured extraction result."""


def explicit_input_values(text: str, declared: list[ProcedureInput]) -> dict[str, PrimitiveValue]:
    """Recognize only complete lines with an explicit declared label and value."""
    values = {}
    for item in declared:
        pattern = re.compile(
            rf"^\s*{re.escape(item.label)}\s*(?::|=|\bis\b)\s*(\S[^\n]*)$",
            re.IGNORECASE | re.MULTILINE,
        )
        for match in pattern.finditer(text):
            values[item.name] = coerce_primitive(match.group(1))
    return values


def extraction_schema(declared: list[ProcedureInput]) -> type[BaseModel]:
    """Constrain model keys to this turn's fields; KB inputs have no type metadata."""
    return create_model(
        "ExtractedProcedureInputs",
        __config__=ConfigDict(extra="forbid"),
        **{item.name: (PrimitiveValue | None, Field(default=None, description=(
            f"{item.label}: {item.description or ''}"
        ))) for item in declared},
    )


def _declared_inputs_block(declared: list[ProcedureInput]) -> str:
    lines = []
    for item in declared:
        lines.append(f"- {item.name}")
        lines.append(f"  label: {item.label}")
        if item.description:
            lines.append(f"  description: {item.description}")
    return "\n".join(lines)


async def extract_procedure_inputs(
    text: str, declared: list[ProcedureInput], config: Config,
) -> dict[str, PrimitiveValue]:
    """Extract values for `declared` inputs from `text` using a small,
    tool-less, structured-output model call.

    Returns an empty dict if there is nothing to extract from. Raises on a
    genuine model/provider failure; callers must handle that explicitly
    (see `app.procedure`'s extraction-failure recovery) rather than silently
    returning an empty/guessed result.
    """
    if not declared or not text.strip():
        return {}

    explicit = explicit_input_values(text, declared)
    # A reply consisting entirely of explicit pairs needs no model call.
    if explicit and all(
        explicit_input_values(line, declared) for line in text.splitlines() if line.strip()
    ):
        return explicit
    declared = [item for item in declared if item.name not in explicit]
    schema = extraction_schema(declared)
    prompt = f"User text:\n{text}\n\nDeclared inputs:\n{_declared_inputs_block(declared)}"

    async with AsyncOpenAI(base_url=config.model_base_url, api_key=config.model_api_key) as client:
        agent = Agent(
            name="ProcedureInputExtractor",
            instructions=EXTRACTION_INSTRUCTIONS,
            model=OpenAIChatCompletionsModel(
                model=config.model_name, openai_client=client,
                should_replay_reasoning_content=lambda context: False,
            ),
            tools=[],
            mcp_servers=[],
            output_type=AgentOutputSchema(schema),
            model_settings=ModelSettings(tool_choice="none"),
        )
        result = await Runner.run(
            agent, prompt, max_turns=1, run_config=RunConfig(tracing_disabled=True),
        )

    output = result.final_output
    if isinstance(output, schema):
        return {**output.model_dump(exclude_none=True), **explicit}
    return explicit
