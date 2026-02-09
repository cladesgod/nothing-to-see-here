"""Content Reviewer Agent: evaluates content validity of items.

Uses the Colquitt et al. (2019) method: rates each item's relevance to
the target dimension and two orbiting (related) dimensions on a 1-7 scale.
Outputs natural language text with a markdown rating table.

When calculator tool is enabled (agents.toml), the agent uses a tool for
precise c-value/d-value computation instead of LLM mental math.
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.config import get_agent_settings
from src.models import create_llm
from src.prompts.templates import (
    CONTENT_REVIEWER_SYSTEM,
    CONTENT_REVIEWER_SYSTEM_WITH_TOOL,
    CONTENT_REVIEWER_TASK,
)
from src.schemas.state import ReviewChainState
from src.utils.console import print_agent_message

logger = structlog.get_logger(__name__)

_MAX_TOOL_ROUNDS = 10  # Safety limit to prevent infinite tool-calling loops


def _extract_expression(tool_call: dict) -> str | None:
    """Extract the arithmetic expression from a tool call, even if the model
    uses the wrong parameter name (e.g. 'c' instead of 'expression')."""
    args = tool_call.get("args", {})
    if "expression" in args:
        return args["expression"]
    # Model sometimes sends a single value under a wrong key — use it
    for value in args.values():
        if isinstance(value, str):
            return value
    return None


async def content_reviewer_node(state: ReviewChainState) -> dict:
    """Evaluate all items' content validity using the Colquitt method."""
    items_text = state.get("items_text", "")
    dimension_info = state.get("dimension_info", "Not specified.")

    logger.info("content_reviewer_start")

    # Check if calculator tool is enabled in agents.toml
    agent_cfg = get_agent_settings().get_agent_config("content_reviewer")
    use_calculator = getattr(agent_cfg, "calculator", False)

    if use_calculator:
        from src.tools.calculator import calculate

        llm = create_llm("content_reviewer", tools=[calculate])
        system_prompt = CONTENT_REVIEWER_SYSTEM_WITH_TOOL
        logger.info("content_reviewer_calculator_enabled")
    else:
        llm = create_llm("content_reviewer")
        system_prompt = CONTENT_REVIEWER_SYSTEM

    prompt = CONTENT_REVIEWER_TASK.format(
        items_text=items_text,
        dimension_info=dimension_info,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)

    # Tool calling loop — execute tool calls and feed results back to LLM
    if use_calculator:
        rounds = 0
        while response.tool_calls and rounds < _MAX_TOOL_ROUNDS:
            rounds += 1
            messages.append(response)
            for tc in response.tool_calls:
                tool_call_id = tc.get("id", "")
                expr = _extract_expression(tc)
                if expr is not None:
                    result_str = calculate.invoke({"expression": expr})
                    logger.info(
                        "content_reviewer_tool_call",
                        expression=expr,
                        result=result_str,
                    )
                    messages.append(
                        ToolMessage(content=result_str, tool_call_id=tool_call_id)
                    )
                else:
                    # Cannot parse — send error back so LLM can retry
                    error_msg = (
                        f"Error: could not parse expression from args {tc.get('args')}. "
                        "Please call calculate with: calculate(expression=\"<math>\")"
                    )
                    logger.warning("content_reviewer_tool_parse_error", args=tc.get("args"))
                    messages.append(
                        ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                    )
            response = await llm.ainvoke(messages)

        if rounds >= _MAX_TOOL_ROUNDS:
            logger.warning("content_reviewer_max_tool_rounds_reached", rounds=rounds)

    review_text = response.content

    logger.info("content_reviewer_done")

    print_agent_message("ContentReviewer", "Critic", review_text)

    return {"content_review": review_text}
