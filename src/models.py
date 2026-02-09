"""LLM factory with fallback provider chain.

Primary: OpenRouter (Llama models)
Fallback 1: Groq (cloud, fast inference)
Fallback 2: Ollama (local)

Models, temperatures, and fallback providers are configured in agents.toml.
"""

from __future__ import annotations

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config import Settings, get_agent_settings, get_settings

logger = structlog.get_logger(__name__)


def create_llm(
    agent_name: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    settings: Settings | None = None,
    tools: list | None = None,
) -> BaseChatModel:
    """Create an LLM with optional tool binding and fallback provider chain.

    Returns a ChatOpenAI (OpenRouter) instance. If fallback providers are
    enabled in agents.toml, wraps it with with_fallbacks() so that failures
    automatically cascade: OpenRouter → Groq → Ollama.

    If tools are provided, bind_tools() is applied to each provider BEFORE
    with_fallbacks(), since RunnableWithFallbacks does not expose bind_tools().

    Args:
        agent_name: Agent identifier used to look up config in agents.toml.
        temperature: Sampling temperature. None = read from agents.toml.
        max_tokens: Maximum tokens in response.
        settings: Optional Settings instance; loads from env if not provided.
        tools: Optional list of LangChain tools to bind to the LLM.

    Returns:
        A BaseChatModel — either plain ChatOpenAI or a RunnableWithFallbacks.
    """
    if settings is None:
        settings = get_settings()

    agent_settings = get_agent_settings()
    model = agent_settings.get_model(agent_name)

    # Resolve temperature: explicit param > agents.toml > 0.7 fallback
    if temperature is None:
        temperature = agent_settings.get_temperature(agent_name)

    kwargs = dict(
        model=model,
        temperature=temperature,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    primary = ChatOpenAI(**kwargs)

    # Bind tools before fallbacks (RunnableWithFallbacks has no bind_tools)
    if tools:
        primary = primary.bind_tools(tools)

    # Build fallback chain
    fallbacks = []

    # Fallback 1: Groq
    if agent_settings.providers.groq.enabled and settings.groq_api_key:
        from langchain_groq import ChatGroq

        groq_model = agent_settings.get_groq_model(agent_name)
        groq_llm = ChatGroq(
            model=groq_model,
            temperature=temperature,
            api_key=settings.groq_api_key,
        )
        if tools:
            groq_llm = groq_llm.bind_tools(tools)
        fallbacks.append(groq_llm)
        logger.debug("groq_fallback_configured", agent=agent_name, model=groq_model)

    # Fallback 2: Ollama (local — no API key needed)
    if agent_settings.providers.ollama.enabled:
        from langchain_ollama import ChatOllama

        ollama_model = agent_settings.get_ollama_model(agent_name)
        base_url = (
            agent_settings.providers.ollama.base_url or "http://localhost:11434"
        )
        ollama_llm = ChatOllama(
            model=ollama_model,
            temperature=temperature,
            base_url=base_url,
        )
        if tools:
            ollama_llm = ollama_llm.bind_tools(tools)
        fallbacks.append(ollama_llm)
        logger.debug(
            "ollama_fallback_configured", agent=agent_name, model=ollama_model
        )

    if fallbacks:
        return primary.with_fallbacks(fallbacks)
    return primary
