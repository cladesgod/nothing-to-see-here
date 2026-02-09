"""Tests for the calculator tool."""

from __future__ import annotations

from src.agents.content_reviewer import _extract_expression
from src.tools.calculator import calculate


class TestCalculateTool:
    """Tests for the calculate tool function."""

    def test_basic_division(self):
        result = calculate.invoke({"expression": "5/6"})
        assert result == "0.8333"

    def test_cvalue_perfect(self):
        """c-value for a perfect rating of 6 on 7-point scale."""
        result = calculate.invoke({"expression": "6/6"})
        assert result == "1.0"

    def test_cvalue_threshold(self):
        """c-value for rating 5 → 0.8333 (meets >= 0.83 threshold)."""
        result = calculate.invoke({"expression": "5/6"})
        assert result == "0.8333"

    def test_dvalue_computation(self):
        """d-value: target=7, orb1=3, orb2=2 → ((7-3)+(7-2))/2/6 = 0.75."""
        result = calculate.invoke({"expression": "((7-3)+(7-2))/2/6"})
        assert result == "0.75"

    def test_dvalue_low(self):
        """d-value when orbiting ratings are close to target."""
        result = calculate.invoke({"expression": "((5-4)+(5-4))/2/6"})
        assert result == "0.1667"

    def test_mean_computation(self):
        result = calculate.invoke({"expression": "(4+5)/2"})
        assert result == "4.5"

    def test_integer_result(self):
        result = calculate.invoke({"expression": "3+4"})
        assert result == "7"

    def test_rejects_letters(self):
        result = calculate.invoke({"expression": "import os"})
        assert result.startswith("Error:")

    def test_rejects_dunder(self):
        result = calculate.invoke({"expression": "__builtins__"})
        assert result.startswith("Error:")

    def test_rejects_semicolons(self):
        result = calculate.invoke({"expression": "1; print('x')"})
        assert result.startswith("Error:")

    def test_division_by_zero(self):
        result = calculate.invoke({"expression": "1/0"})
        assert result == "Error: Division by zero"

    def test_empty_expression(self):
        result = calculate.invoke({"expression": ""})
        # Empty string eval raises SyntaxError
        assert result.startswith("Error:")

    def test_whitespace_handling(self):
        result = calculate.invoke({"expression": " 5 / 6 "})
        assert result == "0.8333"


class TestExtractExpression:
    """Tests for _extract_expression that handles wrong param names from LLM."""

    def test_correct_param_name(self):
        tc = {"args": {"expression": "5/6"}}
        assert _extract_expression(tc) == "5/6"

    def test_wrong_param_name_c(self):
        """Model sends 'c' instead of 'expression'."""
        tc = {"args": {"c": "6/6"}}
        assert _extract_expression(tc) == "6/6"

    def test_wrong_param_name_arbitrary(self):
        """Model sends some other key name."""
        tc = {"args": {"formula": "((7-3)+(7-2))/2/6"}}
        assert _extract_expression(tc) == "((7-3)+(7-2))/2/6"

    def test_empty_args(self):
        tc = {"args": {}}
        assert _extract_expression(tc) is None

    def test_no_args_key(self):
        tc = {}
        assert _extract_expression(tc) is None

    def test_non_string_value_skipped(self):
        """Non-string values should be skipped."""
        tc = {"args": {"count": 42}}
        assert _extract_expression(tc) is None

    def test_expression_preferred_over_other_keys(self):
        """When 'expression' exists, it should be preferred."""
        tc = {"args": {"expression": "5/6", "c": "6/6"}}
        assert _extract_expression(tc) == "5/6"
