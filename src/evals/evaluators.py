"""LLM-as-a-Judge evaluators for independent quality assessment.

Each evaluator is a standalone function that uses an LLM to score generated
items on a specific quality dimension. These are independent from the
in-pipeline reviewers — they assess the final output, not guide the generation.

Evaluators return a dict: {"score": float, "reasoning": str}
  - score: 0.0 to 1.0 (normalized)
  - reasoning: LLM's explanation of the score
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_agent_settings
from src.models import create_llm

logger = structlog.get_logger(__name__)


def _get_judge_llm():
    """Create the judge LLM from eval config."""
    agent_settings = get_agent_settings()
    eval_cfg = getattr(agent_settings, "eval", None)
    model = getattr(eval_cfg, "judge_model", None) if eval_cfg else None
    temp = getattr(eval_cfg, "judge_temperature", 0.0) if eval_cfg else 0.0

    # Use eval_judge as agent name; falls back to defaults.model if no eval config
    return create_llm("eval_judge", temperature=temp)


def _parse_score(response_text: str) -> dict:
    """Extract score and reasoning from judge response.

    Expected format from judge:
    SCORE: X.XX
    REASONING: ...
    """
    score = 0.0
    reasoning = response_text.strip()

    for line in response_text.split("\n"):
        line_upper = line.strip().upper()
        if line_upper.startswith("SCORE:"):
            try:
                raw = line.split(":", 1)[1].strip()
                # Handle both "0.85" and "85/100" and "6/7" formats
                if "/" in raw:
                    parts = raw.split("/")
                    score = float(parts[0].strip()) / float(parts[1].strip())
                else:
                    score = float(raw)
                # Clamp to 0-1
                score = max(0.0, min(1.0, score))
            except (ValueError, IndexError, ZeroDivisionError):
                pass
        elif line_upper.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return {"score": score, "reasoning": reasoning}


CONTENT_VALIDITY_PROMPT = """\
You are an independent psychometric judge evaluating item quality.

Your task: Assess whether the following Likert-scale item accurately measures
the target construct dimension, using the Colquitt et al. (2019) content validity method.

Target Dimension: {dimension_name}
Definition: {dimension_definition}

Orbiting Dimension 1: {orbiting_1_name}
Definition: {orbiting_1_definition}

Orbiting Dimension 2: {orbiting_2_name}
Definition: {orbiting_2_definition}

Item to evaluate:
"{item_text}"

Instructions:
1. Rate how well this item measures the TARGET dimension (1-7 scale, 1=extremely bad, 7=extremely good)
2. Rate how well this item measures ORBITING dimension 1 (1-7 scale)
3. Rate how well this item measures ORBITING dimension 2 (1-7 scale)
4. Compute c-value = target_rating / 6 (threshold: >= {c_threshold})
5. Compute d-value = (target - mean(orbiting)) / 6 (threshold: >= {d_threshold})
6. Provide a final normalized score (0.0 to 1.0) and reasoning.

Respond in this exact format:
SCORE: <number between 0.0 and 1.0>
REASONING: <your assessment>"""


async def content_validity_evaluator(
    item_text: str,
    dimension_name: str,
    dimension_definition: str,
    orbiting_dimensions: list[tuple[str, str]],
) -> dict:
    """Evaluate content validity of a single item.

    Args:
        item_text: The Likert-scale item to evaluate.
        dimension_name: Target dimension name.
        dimension_definition: Target dimension definition.
        orbiting_dimensions: List of (name, definition) tuples for orbiting dims.

    Returns:
        {"score": float, "reasoning": str}
    """
    agent_settings = get_agent_settings()
    eval_cfg = getattr(agent_settings, "eval", None)
    c_threshold = getattr(eval_cfg, "content_validity_threshold", 0.83) if eval_cfg else 0.83
    d_threshold = getattr(eval_cfg, "distinctiveness_threshold", 0.35) if eval_cfg else 0.35

    orb_1_name, orb_1_def = orbiting_dimensions[0] if len(orbiting_dimensions) > 0 else ("N/A", "N/A")
    orb_2_name, orb_2_def = orbiting_dimensions[1] if len(orbiting_dimensions) > 1 else ("N/A", "N/A")

    prompt = CONTENT_VALIDITY_PROMPT.format(
        dimension_name=dimension_name,
        dimension_definition=dimension_definition,
        orbiting_1_name=orb_1_name,
        orbiting_1_definition=orb_1_def,
        orbiting_2_name=orb_2_name,
        orbiting_2_definition=orb_2_def,
        c_threshold=c_threshold,
        d_threshold=d_threshold,
        item_text=item_text,
    )

    llm = _get_judge_llm()
    messages = [
        SystemMessage(content="You are a psychometric evaluation judge. Be precise and objective."),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    result = _parse_score(response.content)
    logger.info("content_eval", item=item_text[:50], score=result["score"])
    return result


LINGUISTIC_QUALITY_PROMPT = """\
You are an independent linguistic quality judge for psychological test items.

Evaluate the following Likert-scale item on these criteria (paper Table 2):
1. Grammatical accuracy and stylistic consistency
2. Ease of understanding (7th-8th grade reading level)
3. Avoidance of unnecessary negative language
4. Clarity and directness (no double-barreled, tricky, or confusing items)

Item to evaluate:
"{item_text}"

Rate the overall linguistic quality as a score from 0.0 (very poor) to 1.0 (excellent).
Consider: Would a typical employee easily understand this item? Is it grammatically correct?
Is it clear, direct, and unambiguous?

Respond in this exact format:
SCORE: <number between 0.0 and 1.0>
REASONING: <your assessment covering all 4 criteria>"""


async def linguistic_quality_evaluator(item_text: str) -> dict:
    """Evaluate linguistic quality of a single item.

    Returns:
        {"score": float, "reasoning": str}
    """
    prompt = LINGUISTIC_QUALITY_PROMPT.format(item_text=item_text)

    llm = _get_judge_llm()
    messages = [
        SystemMessage(content="You are a linguistic quality judge. Be precise and objective."),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    result = _parse_score(response.content)
    logger.info("linguistic_eval", item=item_text[:50], score=result["score"])
    return result


BIAS_EVALUATION_PROMPT = """\
You are an independent bias evaluation judge for psychological test items.

Evaluate the following Likert-scale item for potential demographic biases:
1. Gender bias
2. Religious bias
3. Racial/ethnic bias
4. Age bias
5. Cultural bias

Item to evaluate:
"{item_text}"

Rate the item's freedom from bias as a score from 0.0 (highly biased) to 1.0 (completely unbiased).
An item scoring 1.0 is universally applicable across all demographic groups.
An item scoring 0.0 contains explicit bias that makes it inappropriate for use.

Respond in this exact format:
SCORE: <number between 0.0 and 1.0>
REASONING: <your assessment covering bias categories>"""


async def bias_evaluator(item_text: str) -> dict:
    """Evaluate bias in a single item.

    Returns:
        {"score": float, "reasoning": str}
    """
    prompt = BIAS_EVALUATION_PROMPT.format(item_text=item_text)

    llm = _get_judge_llm()
    messages = [
        SystemMessage(content="You are a bias evaluation judge. Be precise and objective."),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    result = _parse_score(response.content)
    logger.info("bias_eval", item=item_text[:50], score=result["score"])
    return result


OVERALL_QUALITY_PROMPT = """\
You are a senior psychometric judge providing a final quality assessment.

Given an item and its individual evaluation scores, provide an overall quality judgment.

Item: "{item_text}"
Dimension: {dimension_name}

Individual Scores:
- Content Validity: {content_score:.2f}
- Linguistic Quality: {linguistic_score:.2f}
- Bias Freedom: {bias_score:.2f}

Provide a final overall quality score (0.0 to 1.0) that considers all three dimensions.
Also provide a final decision: ACCEPT (score >= 0.8), REVISE (0.5-0.8), or REJECT (< 0.5).

Respond in this exact format:
SCORE: <number between 0.0 and 1.0>
REASONING: <ACCEPT/REVISE/REJECT — your overall assessment>"""


async def overall_quality_evaluator(
    item_text: str,
    dimension_name: str,
    content_score: float,
    linguistic_score: float,
    bias_score: float,
) -> dict:
    """Compute overall quality score from individual evaluator results.

    Returns:
        {"score": float, "reasoning": str}
    """
    prompt = OVERALL_QUALITY_PROMPT.format(
        item_text=item_text,
        dimension_name=dimension_name,
        content_score=content_score,
        linguistic_score=linguistic_score,
        bias_score=bias_score,
    )

    llm = _get_judge_llm()
    messages = [
        SystemMessage(content="You are a senior psychometric judge. Be precise and decisive."),
        HumanMessage(content=prompt),
    ]

    response = await llm.ainvoke(messages)
    result = _parse_score(response.content)
    logger.info("overall_eval", item=item_text[:50], score=result["score"])
    return result
