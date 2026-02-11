"""LLM factory with fallback provider chain.

Primary: OpenRouter (Llama models)
Fallback 1: Groq (cloud, fast inference)
Fallback 2: Ollama (local)

Models, temperatures, and fallback providers are configured in agents.toml.

Each provider is piped with a response-length validator so that suspiciously
short responses (<min_response_length chars) trigger a cascade to the next
provider via with_fallbacks().
"""

from __future__ import annotations

import structlog
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI

from src.config import Settings, get_agent_settings, get_settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Response length validator
# ---------------------------------------------------------------------------


def _make_length_validator(min_chars: int) -> RunnableLambda:
    """Create a Runnable that raises if the LLM response is too short.

    When piped after an LLM (``llm | validator``), a short response raises
    a ``ValueError`` which ``with_fallbacks()`` catches to try the next
    provider in the chain.
    """

    def _validate(response):  # noqa: ANN001
        content = response.content if response.content else ""
        stripped = content.strip()
        if len(stripped) < min_chars:
            raise ValueError(
                f"Response too short ({len(stripped)} chars, minimum {min_chars}). "
                "Falling back to next provider."
            )
        return response

    return RunnableLambda(_validate)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def create_llm(
    agent_name: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    settings: Settings | None = None,
) -> Runnable:
    """Create an LLM with optional fallback provider chain.

    Returns a Runnable (plain LLM, piped chain, or chain with fallbacks).
    If fallback providers are enabled in agents.toml, each provider is piped
    with a response-length validator and chained via ``with_fallbacks()`` so
    that failures (exceptions, timeouts, or short responses) automatically
    cascade: OpenRouter → Groq → Ollama.

    Args:
        agent_name: Agent identifier used to look up config in agents.toml.
        temperature: Sampling temperature. None = read from agents.toml.
        max_tokens: Maximum tokens in response.
        settings: Optional Settings instance; loads from env if not provided.

    Returns:
        A Runnable — either a plain LLM, a piped chain, or a chain with fallbacks.
    """
    if settings is None:
        settings = get_settings()

    agent_settings = get_agent_settings()
    model = agent_settings.get_model(agent_name)
    timeout = agent_settings.defaults.timeout
    min_chars = agent_settings.defaults.min_response_length

    # Resolve temperature: explicit param > agents.toml > 0.7 fallback
    if temperature is None:
        temperature = agent_settings.get_temperature(agent_name)

    kwargs = dict(
        model=model,
        temperature=temperature,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        timeout=timeout,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    primary = ChatOpenAI(**kwargs)

    # Response length validator — piped after each provider
    validator = _make_length_validator(min_chars)
    primary_chain: Runnable = primary | validator

    # Build fallback chain
    fallbacks: list[Runnable] = []

    # Fallback 1: Groq
    if agent_settings.providers.groq.enabled and settings.groq_api_key:
        from langchain_groq import ChatGroq

        groq_model = agent_settings.get_groq_model(agent_name)
        groq_kwargs = dict(
            model=groq_model,
            temperature=temperature,
            api_key=settings.groq_api_key,
            timeout=timeout,
        )
        if max_tokens is not None:
            groq_kwargs["max_tokens"] = max_tokens
        groq_llm = ChatGroq(**groq_kwargs)
        fallbacks.append(groq_llm | validator)
        logger.debug("groq_fallback_configured", agent=agent_name, model=groq_model)

    # Fallback 2: Ollama (local — no API key needed)
    if agent_settings.providers.ollama.enabled:
        from langchain_ollama import ChatOllama

        ollama_model = agent_settings.get_ollama_model(agent_name)
        base_url = (
            agent_settings.providers.ollama.base_url or "http://localhost:11434"
        )
        ollama_kwargs = dict(
            model=ollama_model,
            temperature=temperature,
            base_url=base_url,
            timeout=timeout,
        )
        if max_tokens is not None:
            ollama_kwargs["num_predict"] = max_tokens
        ollama_llm = ChatOllama(**ollama_kwargs)
        fallbacks.append(ollama_llm | validator)
        logger.debug(
            "ollama_fallback_configured", agent=agent_name, model=ollama_model
        )

    if fallbacks:
        return primary_chain.with_fallbacks(fallbacks)
    return primary_chain
