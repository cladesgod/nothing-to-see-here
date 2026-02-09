"""Application configuration using pydantic-settings.

Loads settings from environment variables and .env file.
Agent behavior config (temperatures, models, parameters) loaded from agents.toml.

Priority: CLI args > Environment variables (.env) > agents.toml > hardcoded defaults
"""

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Agent settings from agents.toml
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Base configuration for a single agent."""

    model: str | None = None
    temperature: float | None = None
    hf_model: str = ""       # Agent-specific HuggingFace model override
    groq_model: str = ""     # Agent-specific Groq model override
    ollama_model: str = ""   # Agent-specific Ollama model override


class WebSurferConfig(AgentConfig):
    """WebSurfer-specific configuration."""

    temperature: float = 0.0
    max_results: int = 5
    search_depth: str = "advanced"


class ItemWriterConfig(AgentConfig):
    """Item Writer-specific configuration."""

    temperature: float = 1.0
    num_items: int = 8


class ContentReviewerConfig(AgentConfig):
    temperature: float = 0.0
    calculator: bool = False  # Enable calculator tool for c-value/d-value computation


class LinguisticReviewerConfig(AgentConfig):
    temperature: float = 0.0


class BiasReviewerConfig(AgentConfig):
    temperature: float = 0.0


class MetaEditorConfig(AgentConfig):
    temperature: float = 0.3


class LewModConfig(AgentConfig):
    temperature: float = 0.3


class AgentsTable(BaseModel):
    """The [agents] table from agents.toml."""

    websurfer: WebSurferConfig = Field(default_factory=WebSurferConfig)
    item_writer: ItemWriterConfig = Field(default_factory=ItemWriterConfig)
    content_reviewer: ContentReviewerConfig = Field(
        default_factory=ContentReviewerConfig
    )
    linguistic_reviewer: LinguisticReviewerConfig = Field(
        default_factory=LinguisticReviewerConfig
    )
    bias_reviewer: BiasReviewerConfig = Field(default_factory=BiasReviewerConfig)
    meta_editor: MetaEditorConfig = Field(default_factory=MetaEditorConfig)
    lewmod: LewModConfig = Field(default_factory=LewModConfig)


class DefaultsTable(BaseModel):
    """The [defaults] table from agents.toml."""

    model: str = "meta-llama/llama-4-maverick"


class WorkflowTable(BaseModel):
    """The [workflow] table from agents.toml."""

    max_revisions: int = 3


class RetryConfig(BaseModel):
    """The [retry] table from agents.toml."""

    max_attempts: int = 3
    initial_interval: float = 1.0
    backoff_factor: float = 2.0


class ProviderConfig(BaseModel):
    """Configuration for a single fallback provider."""

    enabled: bool = False
    default_model: str = ""
    base_url: str = ""


class ProvidersTable(BaseModel):
    """The [providers] table from agents.toml."""

    huggingface: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    ollama: ProviderConfig = Field(default_factory=ProviderConfig)


class AgentSettings(BaseModel):
    """Configuration loaded from agents.toml."""

    defaults: DefaultsTable = Field(default_factory=DefaultsTable)
    agents: AgentsTable = Field(default_factory=AgentsTable)
    workflow: WorkflowTable = Field(default_factory=WorkflowTable)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    providers: ProvidersTable = Field(default_factory=ProvidersTable)

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get the config for a specific agent."""
        return getattr(self.agents, agent_name, AgentConfig())

    def get_model(self, agent_name: str) -> str:
        """Get the resolved model for an agent (agent-specific > defaults)."""
        agent_cfg = self.get_agent_config(agent_name)
        return agent_cfg.model or self.defaults.model

    def get_temperature(self, agent_name: str) -> float:
        """Get the resolved temperature for an agent."""
        agent_cfg = self.get_agent_config(agent_name)
        if agent_cfg.temperature is not None:
            return agent_cfg.temperature
        return 0.7  # fallback

    def get_hf_model(self, agent_name: str) -> str:
        """Get HuggingFace model: agent-specific > providers.huggingface.default_model."""
        agent_cfg = self.get_agent_config(agent_name)
        return agent_cfg.hf_model or self.providers.huggingface.default_model

    def get_groq_model(self, agent_name: str) -> str:
        """Get Groq model: agent-specific > providers.groq.default_model."""
        agent_cfg = self.get_agent_config(agent_name)
        return agent_cfg.groq_model or self.providers.groq.default_model

    def get_ollama_model(self, agent_name: str) -> str:
        """Get Ollama model: agent-specific > providers.ollama.default_model."""
        agent_cfg = self.get_agent_config(agent_name)
        return agent_cfg.ollama_model or self.providers.ollama.default_model


_AGENT_SETTINGS_CACHE: AgentSettings | None = None


def get_agent_settings() -> AgentSettings:
    """Load and cache agent settings from agents.toml."""
    global _AGENT_SETTINGS_CACHE
    if _AGENT_SETTINGS_CACHE is not None:
        return _AGENT_SETTINGS_CACHE

    toml_path = Path(__file__).parent.parent / "agents.toml"
    if toml_path.exists():
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        _AGENT_SETTINGS_CACHE = AgentSettings.model_validate(data)
    else:
        _AGENT_SETTINGS_CACHE = AgentSettings()

    return _AGENT_SETTINGS_CACHE


# ---------------------------------------------------------------------------
# Environment settings from .env (API keys, secrets, env-var overrides)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    openrouter_api_key: str
    tavily_api_key: str
    hf_token: str = ""      # Optional — HuggingFace fallback provider
    groq_api_key: str = ""  # Optional — Groq fallback provider

    # LangSmith (set LANGCHAIN_TRACING_V2=true to enable)
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "lm-aig"

    # OpenRouter base URL
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Logging
    log_level: str = "INFO"

    # Rate limiting
    rate_limit_rpm: int = 60  # requests per minute


def get_settings() -> Settings:
    """Create and return a Settings instance.

    Also exports LangSmith env vars so the LangChain SDK
    picks them up automatically for tracing.
    """
    settings = Settings()

    # LangSmith tracing is driven by env vars read by langchain-core.
    # We mirror them from our pydantic-settings into os.environ.
    if settings.langchain_tracing_v2:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        if settings.langchain_api_key:
            os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)

    return settings
