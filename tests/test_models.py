"""Tests for the LLM factory with fallback provider chain."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.runnables import RunnableWithFallbacks
from langchain_openai import ChatOpenAI

from src.config import AgentSettings
from src.models import create_llm


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


class TestCreateLlmFallbacks:
    """Test create_llm() fallback chain behavior."""

    def test_no_fallbacks_when_providers_disabled(self):
        """When providers are disabled, create_llm returns plain ChatOpenAI."""
        agent_settings = AgentSettings()  # defaults: providers disabled
        settings = _make_settings()
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", settings=settings)
        assert isinstance(llm, ChatOpenAI)
        assert not isinstance(llm, RunnableWithFallbacks)

    def test_groq_fallback_when_enabled(self):
        """When Groq is enabled, create_llm returns RunnableWithFallbacks."""
        agent_settings = AgentSettings.model_validate({
            "providers": {
                "groq": {"enabled": True, "default_model": "llama-3.3-70b-versatile"},
            }
        })
        settings = _make_settings(groq_api_key="test-groq-key")
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", settings=settings)
        assert isinstance(llm, RunnableWithFallbacks)
        assert len(llm.fallbacks) == 1

    def test_full_chain_groq_and_ollama(self):
        """When both Groq and Ollama enabled, create_llm has 2 fallbacks."""
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
        """Groq enabled but no API key → skip Groq, only Ollama fallback."""
        agent_settings = AgentSettings.model_validate({
            "providers": {
                "groq": {"enabled": True, "default_model": "llama-3.3-70b-versatile"},
                "ollama": {"enabled": True, "default_model": "llama3.2:latest"},
            }
        })
        settings = _make_settings(groq_api_key="")  # No Groq key
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", settings=settings)
        assert isinstance(llm, RunnableWithFallbacks)
        assert len(llm.fallbacks) == 1  # Only Ollama

    def test_ollama_only_fallback(self):
        """Only Ollama enabled, no Groq."""
        agent_settings = AgentSettings.model_validate({
            "providers": {
                "ollama": {"enabled": True, "default_model": "llama3.2:latest"},
            }
        })
        settings = _make_settings()
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("item_writer", settings=settings)
        assert isinstance(llm, RunnableWithFallbacks)
        assert len(llm.fallbacks) == 1


class TestCreateLlmWithTools:
    """Test create_llm() tool binding behavior."""

    def test_no_tools_returns_plain_model(self):
        """Without tools, create_llm returns standard model (no bind_tools)."""
        agent_settings = AgentSettings()  # providers disabled
        settings = _make_settings()
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("content_reviewer", settings=settings)
        assert isinstance(llm, ChatOpenAI)
        # ChatOpenAI without bind_tools has no 'tools' in kwargs
        assert not getattr(llm, "kwargs", {}).get("tools")

    def test_tools_bind_to_primary(self):
        """When tools are passed, primary LLM should have tools bound."""
        from langchain_core.tools import tool

        @tool
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        agent_settings = AgentSettings()  # providers disabled
        settings = _make_settings()
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("content_reviewer", tools=[dummy_tool], settings=settings)
        # bind_tools returns a RunnableBinding, not plain ChatOpenAI
        assert hasattr(llm, "bound")  # RunnableBinding wraps the model

    def test_tools_with_fallbacks(self):
        """When tools + fallbacks, all providers should have tools bound."""
        from langchain_core.tools import tool

        @tool
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        agent_settings = AgentSettings.model_validate({
            "providers": {
                "ollama": {"enabled": True, "default_model": "llama3.2:latest"},
            }
        })
        settings = _make_settings()
        with patch("src.models.get_agent_settings", return_value=agent_settings):
            llm = create_llm("content_reviewer", tools=[dummy_tool], settings=settings)
        # Should be RunnableWithFallbacks with bound primary
        assert isinstance(llm, RunnableWithFallbacks)
        # Primary (runnable) should have tools bound (RunnableBinding has 'bound')
        assert hasattr(llm.runnable, "bound")
        # Fallback should also have tools bound
        assert len(llm.fallbacks) == 1
        assert hasattr(llm.fallbacks[0], "bound")
