"""Helpers for strict structured output with JSON fixing retries."""

from __future__ import annotations

import json
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.json import parse_json_markdown
from pydantic import BaseModel

from src.config import get_agent_settings
from src.models import create_llm

T = TypeVar("T", bound=BaseModel)


def _schema_text(schema: type[T]) -> str:
    return json.dumps(schema.model_json_schema(), ensure_ascii=True)


def _parse_schema(content: str, schema: type[T]) -> T:
    data = parse_json_markdown(content)
    return schema.model_validate(data)


async def invoke_structured_with_fix(
    *,
    agent_name: str,
    messages: list,
    schema: type[T],
    llm: object | None = None,
    max_attempts: int | None = None,
    memory_window: int | None = None,
) -> T:
    """Invoke an agent and enforce a valid schema with fixer retries.

    Strategy:
    1) Primary model call
    2) Parse + validate
    3) If invalid, invoke fixer LLM with error memory and retry parse

    Args:
        llm: Optional pre-built LLM to use instead of creating one from
             agent_name. Useful when a specific provider is needed.
    """
    cfg = get_agent_settings().json_fix
    max_attempts = max_attempts or cfg.max_attempts
    memory_window = memory_window or cfg.memory_window

    if llm is None:
        llm = create_llm(agent_name)
    response = await llm.ainvoke(messages)
    content = response.content if response.content else ""
    errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        try:
            return _parse_schema(content, schema)
        except Exception as exc:  # parse or validation
            err = str(exc)
            errors.append(err)
            if attempt >= max_attempts:
                raise ValueError(
                    f"{agent_name} failed structured parsing after {max_attempts} attempts. "
                    f"Last error: {err}"
                ) from exc

            recent = errors[-memory_window:]
            fixer = create_llm(agent_name, temperature=0.0)
            fixer_messages = [
                SystemMessage(
                    content=(
                        "You are a JSON fixer. Return ONLY valid JSON that matches the given schema. "
                        "Do not include markdown, explanations, or extra keys."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Schema JSON:\n{_schema_text(schema)}\n\n"
                        f"Invalid output:\n{content}\n\n"
                        f"Recent parsing/validation errors:\n- " + "\n- ".join(recent)
                    )
                ),
            ]
            fixed = await fixer.ainvoke(fixer_messages)
            content = fixed.content if fixed.content else ""
