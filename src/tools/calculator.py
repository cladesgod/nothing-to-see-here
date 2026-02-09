"""Calculator tool for precise arithmetic in content validity assessment.

Provides the Content Reviewer agent with a tool for computing c-value,
d-value, means, and threshold comparisons using Python arithmetic instead
of relying on LLM mental math.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Use this for computing c-value, d-value, means, and threshold comparisons.
    Only supports numbers and basic arithmetic operators (+, -, *, /, parentheses).

    Examples:
        "5/6" → "0.8333"
        "((6-2)+(6-3))/2/6" → "0.5833"
        "6/6" → "1.0"
    """
    allowed = set("0123456789.+-*/() ")
    if not all(c in allowed for c in expression):
        return f"Error: Invalid characters in expression: {expression}"
    try:
        result = eval(expression)  # noqa: S307 — safe: only digits and operators
        return str(round(result, 4))
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {e}"
