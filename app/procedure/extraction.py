"""Natural-language extraction for multi-field procedure input replies.

Deterministic parsing (`app.procedure._parse_field_values`'s "name: value" /
"name=value" / bare-single-value conventions) is tried first and handles
every structured reply without ever calling a model. This module is only
reached when a reply to a *multiple*-field request does not follow that
convention — for example "namespace is payments and application is
checkout" — and performs one small, constrained, bounded LLM call instead
of extending the deterministic grammar indefinitely.

The extractor can only ever return values for the exact field names it is
given: the structured output schema has no other fields, so it cannot
invent an undeclared one. It never chooses what happens next (LangGraph
still owns pause/resume and decides which fields remain missing); it only
turns one natural-language sentence into a `{field_name: value}` mapping.
"""
import logging

from agents import Agent, AgentOutputSchema, OpenAIChatCompletionsModel, RunConfig, Runner
from openai import AsyncOpenAI
from pydantic import create_model

from app.config import Config

logger = logging.getLogger(__name__)

_EXTRACTOR_INSTRUCTIONS = """Extract values for the requested fields from the user's message.

Return a value only for a field the user clearly provided. Leave a field null if it was not
provided or is ambiguous. Never invent a value. Never return information for anything other
than the requested fields.
"""


def _build_extraction_model(field_names: list[str]):
    fields = {name: (str | None, None) for name in field_names}
    return create_model("ExtractedProcedureInputs", **fields)


async def extract_field_values(message: str, field_names: list[str], config: Config) -> dict[str, str]:
    """One bounded, constrained extraction call. Returns `{}` on any model,
    protocol, or connectivity failure: the caller falls back to asking the
    user again rather than guessing.
    """
    if not field_names:
        return {}
    extraction_model = _build_extraction_model(field_names)
    try:
        async with AsyncOpenAI(base_url=config.model_base_url, api_key=config.model_api_key) as client:
            agent = Agent(
                name="ProcedureInputExtractor",
                instructions=_EXTRACTOR_INSTRUCTIONS,
                model=OpenAIChatCompletionsModel(
                    model=config.model_name, openai_client=client,
                    should_replay_reasoning_content=lambda context: False,
                ),
                output_type=AgentOutputSchema(extraction_model, strict_json_schema=False),
            )
            result = await Runner.run(
                agent, message, max_turns=1, run_config=RunConfig(tracing_disabled=True),
            )
    except Exception as error:  # pragma: no cover - defensive: never block input collection
        logger.info("Procedure input extraction failed error=%s", type(error).__name__)
        return {}
    extracted = result.final_output
    if not isinstance(extracted, extraction_model):  # pragma: no cover - defensive
        return {}
    return {name: value for name in field_names if (value := getattr(extracted, name, None))}
