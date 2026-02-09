"""Tests for agent configuration loading from agents.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

from src.config import AgentSettings


class TestAgentSettingsDefaults:
    """Test that all defaults match paper-recommended values."""

    def test_default_temperatures(self):
        settings = AgentSettings()
        assert settings.agents.websurfer.temperature == 0.0
        assert settings.agents.item_writer.temperature == 1.0
        assert settings.agents.content_reviewer.temperature == 0.0
        assert settings.agents.linguistic_reviewer.temperature == 0.0
        assert settings.agents.bias_reviewer.temperature == 0.0
        assert settings.agents.meta_editor.temperature == 0.3
        assert settings.agents.lewmod.temperature == 0.3

    def test_default_model(self):
        settings = AgentSettings()
        assert settings.defaults.model == "meta-llama/llama-4-maverick"

    def test_websurfer_specific_defaults(self):
        settings = AgentSettings()
        assert settings.agents.websurfer.max_results == 5
        assert settings.agents.websurfer.search_depth == "advanced"

    def test_item_writer_num_items(self):
        settings = AgentSettings()
        assert settings.agents.item_writer.num_items == 8

    def test_workflow_max_revisions(self):
        settings = AgentSettings()
        assert settings.workflow.max_revisions == 3


class TestAgentSettingsFromDict:
    """Test loading from parsed TOML data (dict)."""

    def test_override_temperature(self):
        data = {"agents": {"item_writer": {"temperature": 0.5}}}
        settings = AgentSettings.model_validate(data)
        assert settings.agents.item_writer.temperature == 0.5
        # Others keep defaults
        assert settings.agents.websurfer.temperature == 0.0

    def test_override_model(self):
        data = {"agents": {"websurfer": {"model": "some/other-model"}}}
        settings = AgentSettings.model_validate(data)
        assert settings.get_model("websurfer") == "some/other-model"
        # Fallback to defaults for others
        assert settings.get_model("item_writer") == "meta-llama/llama-4-maverick"

    def test_override_num_items(self):
        data = {"agents": {"item_writer": {"num_items": 12}}}
        settings = AgentSettings.model_validate(data)
        assert settings.agents.item_writer.num_items == 12

    def test_empty_dict_uses_all_defaults(self):
        settings = AgentSettings.model_validate({})
        assert settings.defaults.model == "meta-llama/llama-4-maverick"
        assert settings.agents.item_writer.num_items == 8
        assert settings.agents.websurfer.temperature == 0.0
        assert settings.workflow.max_revisions == 3


class TestRetryConfig:
    """Test retry configuration defaults and overrides."""

    def test_retry_defaults(self):
        settings = AgentSettings()
        assert settings.retry.max_attempts == 3
        assert settings.retry.initial_interval == 1.0
        assert settings.retry.backoff_factor == 2.0

    def test_retry_override(self):
        data = {"retry": {"max_attempts": 5, "backoff_factor": 3.0}}
        settings = AgentSettings.model_validate(data)
        assert settings.retry.max_attempts == 5
        assert settings.retry.backoff_factor == 3.0
        assert settings.retry.initial_interval == 1.0  # default kept


class TestProviderConfig:
    """Test fallback provider configuration."""

    def test_provider_defaults_disabled(self):
        settings = AgentSettings()
        assert settings.providers.groq.enabled is False
        assert settings.providers.ollama.enabled is False
        assert settings.providers.groq.default_model == ""
        assert settings.providers.ollama.default_model == ""

    def test_provider_from_dict(self):
        data = {
            "providers": {
                "groq": {"enabled": True, "default_model": "llama-3.3-70b-versatile"},
                "ollama": {
                    "enabled": True,
                    "default_model": "llama3.2:latest",
                    "base_url": "http://localhost:11434",
                },
            }
        }
        settings = AgentSettings.model_validate(data)
        assert settings.providers.groq.enabled is True
        assert settings.providers.groq.default_model == "llama-3.3-70b-versatile"
        assert settings.providers.ollama.enabled is True
        assert settings.providers.ollama.default_model == "llama3.2:latest"
        assert settings.providers.ollama.base_url == "http://localhost:11434"

    def test_get_groq_model_agent_override(self):
        data = {
            "providers": {"groq": {"default_model": "default-groq"}},
            "agents": {"websurfer": {"groq_model": "custom-groq"}},
        }
        settings = AgentSettings.model_validate(data)
        assert settings.get_groq_model("websurfer") == "custom-groq"

    def test_get_groq_model_falls_back_to_default(self):
        data = {
            "providers": {"groq": {"default_model": "default-groq"}},
        }
        settings = AgentSettings.model_validate(data)
        assert settings.get_groq_model("content_reviewer") == "default-groq"

    def test_get_ollama_model_agent_override(self):
        data = {
            "providers": {"ollama": {"default_model": "default-ollama"}},
            "agents": {"item_writer": {"ollama_model": "custom-ollama"}},
        }
        settings = AgentSettings.model_validate(data)
        assert settings.get_ollama_model("item_writer") == "custom-ollama"

    def test_get_ollama_model_falls_back_to_default(self):
        data = {
            "providers": {"ollama": {"default_model": "default-ollama"}},
        }
        settings = AgentSettings.model_validate(data)
        assert settings.get_ollama_model("bias_reviewer") == "default-ollama"


class TestCalculatorConfig:
    """Test calculator tool configuration for content reviewer."""

    def test_calculator_default_false(self):
        settings = AgentSettings()
        assert settings.agents.content_reviewer.calculator is False

    def test_calculator_enabled_from_dict(self):
        data = {"agents": {"content_reviewer": {"calculator": True}}}
        settings = AgentSettings.model_validate(data)
        assert settings.agents.content_reviewer.calculator is True

    def test_calculator_from_toml_file(self):
        toml_path = Path(__file__).parent.parent / "agents.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        settings = AgentSettings.model_validate(data)
        assert settings.agents.content_reviewer.calculator is True


class TestAgentsTomlFile:
    """Test that the actual agents.toml file parses correctly."""

    def test_agents_toml_exists_and_parses(self):
        toml_path = Path(__file__).parent.parent / "agents.toml"
        assert toml_path.exists(), "agents.toml should exist in project root"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        settings = AgentSettings.model_validate(data)
        assert settings.defaults.model is not None
        assert settings.agents.websurfer.temperature == 0.0
        assert settings.agents.item_writer.num_items == 8

    def test_agents_toml_has_retry_config(self):
        toml_path = Path(__file__).parent.parent / "agents.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        settings = AgentSettings.model_validate(data)
        assert settings.retry.max_attempts == 3
        assert settings.retry.initial_interval == 1.0

    def test_agents_toml_has_providers(self):
        toml_path = Path(__file__).parent.parent / "agents.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        settings = AgentSettings.model_validate(data)
        assert settings.providers.groq.enabled is True
        assert settings.providers.groq.default_model == "llama-3.3-70b-versatile"
        assert settings.providers.ollama.enabled is True
