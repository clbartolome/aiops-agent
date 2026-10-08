"""Focused structured extraction of declared procedure input values.

This is not an autonomous agent: it has no tools (MCP or otherwise), never
executes anything, and never answers the user conversationally. It is given
the user's text and only the currently-declared/missing procedure inputs,
and must return a small structured object naming only values it is
confident about. All validation of the result (declared-field membership,
supported types, missing-value detection) happens afterward in plain
Python — see `app.procedure` — never in the model.
"""
from agents import AgentOutputSchema, Agent, ModelSettings, OpenAIChatCompletionsModel, RunConfig, Runner
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import Config
from app.procedure.models import PrimitiveValue, ProcedureInput

EXTRACTION_INSTRUCTIONS = """Extract values only for the declared procedure inputs.

Use only values explicitly stated or unambiguously implied by the user's message.

Do not guess missing values.
Do not invent fields that are not declared.
If a value is not clearly provided, omit it from the result.

You have no tools. Never execute anything and never answer the user
conversationally; only return the structured extraction result."""


class ExtractedProcedureInputs(BaseModel):
    """Strictly structured extraction output: declared input name -> value."""

    values: dict[str, PrimitiveValue] = {}


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
            # This provider's strict-JSON-schema enforcement rejects the
            # open `dict[str, ...]` values mapping; relax schema strictness
            # only (the SDK still validates the Pydantic model locally).
            output_type=AgentOutputSchema(ExtractedProcedureInputs, strict_json_schema=False),
            model_settings=ModelSettings(tool_choice="none"),
        )
        result = await Runner.run(
            agent, prompt, max_turns=1, run_config=RunConfig(tracing_disabled=True),
        )

    output = result.final_output
    if isinstance(output, ExtractedProcedureInputs):
        return dict(output.values)
    return {}
