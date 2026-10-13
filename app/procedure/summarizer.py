"""Final natural-language summary of an already-finished `/procedure` run.

This is not an autonomous agent: it has no MCP tools, no procedure
execution capabilities, and no ability to ask the user for more input. It
only receives already-stored procedure state (title, original request,
resolved inputs, step titles/statuses/summaries, and the final procedure
status) and returns a short natural-language synthesis of what already
happened.

This is presentation-only. Generating -- or failing to generate -- this
text never changes the procedure's own already-final status; see
`app.procedure`'s deterministic, LLM-free fallback for when this call
fails.
"""
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, RunConfig, Runner
from openai import AsyncOpenAI

from app.config import Config
from app.procedure.models import PrimitiveValue, ProcedureContext

SUMMARY_INSTRUCTIONS = """Summarize the outcome of one already-executed IT operations procedure.

You have no tools. Never call a tool, never execute anything, and never
ask the user for more information; only return a short natural-language
summary of the procedure state given to you below.

Base your summary only on the procedure title, the resolved inputs, and
the listed step results. Never invent details, reasons, or outcomes that
are not present in the step summaries below.

Keep it concise: two or three sentences at most. Clearly convey whether
the procedure completed successfully, was intentionally stopped, or
failed. Do not repeat every step verbatim if the outcome is already clear."""


def _resolved_inputs_block(inputs: dict[str, PrimitiveValue]) -> str:
    if not inputs:
        return "(none)"
    return "\n".join(f"{name} = {value}" for name, value in inputs.items())


def _results_block(context: ProcedureContext) -> str:
    lines = []
    for index, step in enumerate(context.procedure.steps, start=1):
        result = context.step_results.get(step.id)
        if result is None:
            continue
        lines.append(f"{index}. {step.title}")
        lines.append(f"   {result.status}")
        lines.append(f"   {result.summary}")
    return "\n".join(lines)


def _summary_prompt(context: ProcedureContext) -> str:
    return "\n\n".join([
        f"Procedure:\n{context.kb_title}",
        f"Original request:\n{context.original_request or '(none)'}",
        f"Inputs:\n{_resolved_inputs_block(context.inputs)}",
        f"Results:\n{_results_block(context)}",
        f"Final status:\n{context.status}",
    ])


async def generate_procedure_summary(context: ProcedureContext, config: Config) -> str | None:
    """Generate a short natural-language summary of an already-finished
    procedure run (`context.status` is one of COMPLETED/STOPPED/FAILED).

    Returns `None` on any failure (model/provider error, or an unusable
    response); callers must fall back to a deterministic message built
    directly from `context.step_results` rather than ever letting this
    call change the procedure's own already-final status.
    """
    prompt = _summary_prompt(context)
    try:
        async with AsyncOpenAI(base_url=config.model_base_url, api_key=config.model_api_key) as client:
            agent = Agent(
                name="ProcedureSummarizer",
                instructions=SUMMARY_INSTRUCTIONS,
                model=OpenAIChatCompletionsModel(
                    model=config.model_name, openai_client=client,
                    should_replay_reasoning_content=lambda context: False,
                ),
                tools=[],
                mcp_servers=[],
                model_settings=ModelSettings(tool_choice="none"),
            )
            result = await Runner.run(
                agent, prompt, max_turns=1, run_config=RunConfig(tracing_disabled=True),
            )
    except Exception:
        return None

    output = result.final_output
    return output.strip() if isinstance(output, str) and output.strip() else None
