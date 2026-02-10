"""Tests for the LLM factory with fallback provider chain."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSequence, RunnableWithFallbacks
from langchain_openai import ChatOpenAI

from src.config import AgentSettings
from src.models import _make_length_validator, create_llm


def _make_settings(**overrides):
    """Create a mock Settings object for testing."""
    from src.config import Settings

    defaults = {
        "openrouter_api_key": "test-key",
        "tavily_api_key": "test-key",
        "groq_api_key": "test-groq-key",
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# Response Length Validator
# ---------------------------------------------------------------------------


class TestLengthValidator:
    """Test the _make_length_validator pipe function."""

    def test_passes_long_response(self):
        validator = _make_length_validator(50)
        response = AIMessage(content="A" * 100)
        result = validator.invoke(response)
        assert result.content == "A" * 100

    def test_raises_on_short_response(self):
        validator = _make_length_validator(50)
        response = AIMessage(content="Too short")
        with pytest.raises(ValueError, match="too short"):
            validator.invoke(response)

    def test_raises_on_empty_response(self):
        validator = _make_length_validator(50)
        response = AIMessage(content="")
        with pytest.raises(ValueError, match="too short"):
            validator.invoke(response)

    def test_raises_on_whitespace_only(self):
        validator = _make_length_validator(50)
        response = AIMessage(content="   \n\t  ")
        with pytest.raises(ValueError, match="too short"):
            validator.invoke(response)

    def test_exact_threshold_passes(self):
        validator = _make_length_validator(10)
        response = AIMessage(content="A" * 10)
        result = validator.invoke(response)
        assert result.content == "A" * 10

    def test_one_below_threshold_raises(self):
        validator = _make_length_validator(10)
        response = AIMessage(content="A" * 9)
        with pytest.raises(ValueError, match="too short"):
            validator.invoke(response)

    def test_zero_threshold_passes_anything(self):
        validator = _make_length_validator(0)
        response = AIMessage(content="")
        result = validator.invoke(response)
        assert result.content == ""


# ---------------------------------------------------------------------------
# Basic create_llm
# ---------------------------------------------------------------------------


class TestCreateLlmBasic:
    """Test basic create_llm behavior."""

    def test_returns_piped_chain_when_no_fallbacks(self):
        agent_settings = AgentSettings()  # providers disabled by default
        settings = _make_settings()
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", settings=settings)
        # Now returns RunnableSequence (LLM | validator), not plain ChatOpenAI
        assert isinstance(llm, RunnableSequence)
        assert isinstance(llm.first, ChatOpenAI)

    def test_primary_has_timeout(self):
        agent_settings = AgentSettings.model_validate({
            "defaults": {"timeout": 60}
        })
        settings = _make_settings()
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", settings=settings)
        primary = llm.first if isinstance(llm, RunnableSequence) else llm
        assert primary.request_timeout == 60


# ---------------------------------------------------------------------------
# Fallback Chain
# ---------------------------------------------------------------------------


class TestFallbackChain:
    """Test fallback provider chain construction."""

    def test_groq_fallback_when_enabled(self):
        agent_settings = AgentSettings.model_validate({
            "providers": {"groq": {"enabled": True, "default_model": "llama-3.3-70b-versatile"}}
        })
        settings = _make_settings(groq_api_key="test-groq-key")
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", settings=settings)
        assert isinstance(llm, RunnableWithFallbacks)
        assert len(llm.fallbacks) == 1

    def test_full_chain_groq_and_ollama(self):
        agent_settings = AgentSettings.model_validate({
            "providers": {
                "groq": {"enabled": True, "default_model": "llama-3.3-70b-versatile"},
                "ollama": {"enabled": True, "default_model": "llama3.2:latest"},
            }
        })
        settings = _make_settings(groq_api_key="test-groq-key")
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", settings=settings)
        assert isinstance(llm, RunnableWithFallbacks)
        assert len(llm.fallbacks) == 2

    def test_groq_skipped_without_api_key(self):
        agent_settings = AgentSettings.model_validate({
            "providers": {
                "groq": {"enabled": True, "default_model": "llama-3.3-70b-versatile"},
                "ollama": {"enabled": True, "default_model": "llama3.2:latest"},
            }
        })
        settings = _make_settings(groq_api_key="")
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", settings=settings)
        assert isinstance(llm, RunnableWithFallbacks)
        assert len(llm.fallbacks) == 1  # Only Ollama

    def test_ollama_only_fallback(self):
        agent_settings = AgentSettings.model_validate({
            "providers": {"ollama": {"enabled": True, "default_model": "llama3.2:latest"}}
        })
        settings = _make_settings()
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", settings=settings)
        assert isinstance(llm, RunnableWithFallbacks)
        assert len(llm.fallbacks) == 1

    def test_fallbacks_are_piped_chains(self):
        """Each fallback should be a RunnableSequence (LLM | validator)."""
        agent_settings = AgentSettings.model_validate({
            "providers": {
                "groq": {"enabled": True, "default_model": "llama-3.3-70b-versatile"},
                "ollama": {"enabled": True, "default_model": "llama3.2:latest"},
            }
        })
        settings = _make_settings(groq_api_key="test-groq-key")
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", settings=settings)
        for fb in llm.fallbacks:
            assert isinstance(fb, RunnableSequence)


# ---------------------------------------------------------------------------
# Max Tokens Propagation
# ---------------------------------------------------------------------------


class TestMaxTokensPropagation:
    """FIX 2: max_tokens should propagate to all fallback providers."""

    def test_primary_gets_max_tokens(self):
        agent_settings = AgentSettings()
        settings = _make_settings()
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", max_tokens=2048, settings=settings)
        primary = llm.first if isinstance(llm, RunnableSequence) else llm
        assert primary.max_tokens == 2048

    def test_groq_gets_max_tokens(self):
        agent_settings = AgentSettings.model_validate({
            "providers": {"groq": {"enabled": True, "default_model": "llama-3.3-70b-versatile"}}
        })
        settings = _make_settings(groq_api_key="test-groq-key")
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", max_tokens=2048, settings=settings)
        assert isinstance(llm, RunnableWithFallbacks)
        groq_chain = llm.fallbacks[0]
        groq_llm = groq_chain.first if isinstance(groq_chain, RunnableSequence) else groq_chain
        assert groq_llm.max_tokens == 2048

    def test_ollama_gets_num_predict(self):
        agent_settings = AgentSettings.model_validate({
            "providers": {"ollama": {"enabled": True, "default_model": "llama3.2:latest"}}
        })
        settings = _make_settings()
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", max_tokens=2048, settings=settings)
        assert isinstance(llm, RunnableWithFallbacks)
        ollama_chain = llm.fallbacks[0]
        ollama_llm = ollama_chain.first if isinstance(ollama_chain, RunnableSequence) else ollama_chain
        assert ollama_llm.num_predict == 2048

    def test_no_max_tokens_when_not_specified(self):
        agent_settings = AgentSettings.model_validate({
            "providers": {"groq": {"enabled": True, "default_model": "llama-3.3-70b-versatile"}}
        })
        settings = _make_settings(groq_api_key="test-groq-key")
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", settings=settings)
        assert isinstance(llm, RunnableWithFallbacks)
        primary = llm.runnable.first if isinstance(llm.runnable, RunnableSequence) else llm.runnable
        assert primary.max_tokens is None
