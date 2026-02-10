"""Tests for agent configuration loading from agents.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

from src.config import AgentSettings


class TestAgentSettingsDefaults:
    """Test that defaults match paper-recommended values."""

    def test_default_temperatures(self):
        s = AgentSettings()
        assert s.agents.websurfer.temperature == 0.0
        assert s.agents.item_writer.temperature == 1.0
        assert s.agents.content_reviewer.temperature == 0.0
        assert s.agents.linguistic_reviewer.temperature == 0.0
        assert s.agents.bias_reviewer.temperature == 0.0
        assert s.agents.meta_editor.temperature == 0.3
        assert s.agents.lewmod.temperature == 0.3

    def test_default_model(self):
        assert AgentSettings().defaults.model == "meta-llama/llama-4-maverick"

    def test_default_timeout_and_min_response_length(self):
        s = AgentSettings()
        assert s.defaults.timeout == 120
        assert s.defaults.min_response_length == 50

    def test_websurfer_defaults(self):
        s = AgentSettings()
        assert s.agents.websurfer.max_results == 5
        assert s.agents.websurfer.search_depth == "advanced"

    def test_item_writer_num_items(self):
        assert AgentSettings().agents.item_writer.num_items == 8

    def test_workflow_max_revisions(self):
        assert AgentSettings().workflow.max_revisions == 3

    def test_workflow_memory_defaults(self):
        s = AgentSettings()
        assert s.workflow.memory_enabled is True
        assert s.workflow.memory_limit == 5


class TestAgentSettingsOverrides:
    """Test loading from parsed TOML data (dict)."""

    def test_override_temperature(self):
        data = {"agents": {"item_writer": {"temperature": 0.5}}}
        s = AgentSettings.model_validate(data)
        assert s.agents.item_writer.temperature == 0.5
        assert s.agents.websurfer.temperature == 0.0  # unchanged

    def test_override_model(self):
        data = {"agents": {"websurfer": {"model": "other/model"}}}
        s = AgentSettings.model_validate(data)
        assert s.get_model("websurfer") == "other/model"
        assert s.get_model("item_writer") == "meta-llama/llama-4-maverick"

    def test_override_num_items(self):
        data = {"agents": {"item_writer": {"num_items": 12}}}
        assert AgentSettings.model_validate(data).agents.item_writer.num_items == 12

    def test_override_memory_settings(self):
        data = {"workflow": {"memory_enabled": False, "memory_limit": 10}}
        s = AgentSettings.model_validate(data)
        assert s.workflow.memory_enabled is False
        assert s.workflow.memory_limit == 10

    def test_override_timeout_and_min_response_length(self):
        data = {"defaults": {"timeout": 30, "min_response_length": 100}}
        s = AgentSettings.model_validate(data)
        assert s.defaults.timeout == 30
        assert s.defaults.min_response_length == 100

    def test_empty_dict_uses_defaults(self):
        s = AgentSettings.model_validate({})
        assert s.defaults.model == "meta-llama/llama-4-maverick"
        assert s.agents.item_writer.num_items == 8
        assert s.workflow.max_revisions == 3


class TestRetryConfig:
    """Test retry configuration."""

    def test_retry_defaults(self):
        s = AgentSettings()
        assert s.retry.max_attempts == 3
        assert s.retry.initial_interval == 1.0
        assert s.retry.backoff_factor == 2.0

    def test_retry_override(self):
        data = {"retry": {"max_attempts": 5, "backoff_factor": 3.0}}
        s = AgentSettings.model_validate(data)
        assert s.retry.max_attempts == 5
        assert s.retry.backoff_factor == 3.0
        assert s.retry.initial_interval == 1.0


class TestProviderConfig:
    """Test fallback provider configuration."""

    def test_providers_disabled_by_default(self):
        s = AgentSettings()
        assert s.providers.groq.enabled is False
        assert s.providers.ollama.enabled is False

    def test_provider_from_dict(self):
        data = {
            "providers": {
                "groq": {"enabled": True, "default_model": "llama-3.3-70b-versatile"},
                "ollama": {"enabled": True, "default_model": "llama3.2:latest", "base_url": "http://localhost:11434"},
            }
        }
        s = AgentSettings.model_validate(data)
        assert s.providers.groq.enabled is True
        assert s.providers.groq.default_model == "llama-3.3-70b-versatile"
        assert s.providers.ollama.base_url == "http://localhost:11434"

    def test_groq_model_agent_override(self):
        data = {
            "providers": {"groq": {"default_model": "default-groq"}},
            "agents": {"websurfer": {"groq_model": "custom-groq"}},
        }
        s = AgentSettings.model_validate(data)
        assert s.get_groq_model("websurfer") == "custom-groq"

    def test_groq_model_falls_back_to_default(self):
        data = {"providers": {"groq": {"default_model": "default-groq"}}}
        assert AgentSettings.model_validate(data).get_groq_model("content_reviewer") == "default-groq"

    def test_ollama_model_agent_override(self):
        data = {
            "providers": {"ollama": {"default_model": "default-ollama"}},
            "agents": {"item_writer": {"ollama_model": "custom-ollama"}},
        }
        assert AgentSettings.model_validate(data).get_ollama_model("item_writer") == "custom-ollama"

    def test_ollama_model_falls_back_to_default(self):
        data = {"providers": {"ollama": {"default_model": "default-ollama"}}}
        assert AgentSettings.model_validate(data).get_ollama_model("bias_reviewer") == "default-ollama"


class TestEvalConfig:
    """Test eval configuration."""

    def test_eval_defaults(self):
        s = AgentSettings()
        assert s.eval.enabled is True
        assert s.eval.judge_temperature == 0.0
        assert s.eval.content_validity_threshold == 0.83
        assert s.eval.distinctiveness_threshold == 0.35
        assert s.eval.linguistic_threshold == 0.8
        assert s.eval.bias_threshold == 0.9

    def test_eval_judge_model_resolution(self):
        data = {"eval": {"judge_model": "custom/judge"}}
        assert AgentSettings.model_validate(data).get_model("eval_judge") == "custom/judge"

    def test_eval_judge_falls_back_to_default(self):
        assert AgentSettings().get_model("eval_judge") == "meta-llama/llama-4-maverick"


class TestAgentsTomlFile:
    """Test that the actual agents.toml file parses correctly."""

    def test_agents_toml_exists_and_parses(self):
        toml_path = Path(__file__).parent.parent / "agents.toml"
        assert toml_path.exists()
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        s = AgentSettings.model_validate(data)
        assert s.defaults.model is not None
        assert s.agents.websurfer.temperature == 0.0

    def test_agents_toml_has_retry_config(self):
        toml_path = Path(__file__).parent.parent / "agents.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        s = AgentSettings.model_validate(data)
        assert s.retry.max_attempts == 3

    def test_agents_toml_has_providers(self):
        toml_path = Path(__file__).parent.parent / "agents.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        s = AgentSettings.model_validate(data)
        assert s.providers.groq.enabled is True
        assert s.providers.groq.default_model == "llama-3.3-70b-versatile"
        assert s.providers.ollama.enabled is True

    def test_agents_toml_has_eval_config(self):
        toml_path = Path(__file__).parent.parent / "agents.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        s = AgentSettings.model_validate(data)
        assert s.eval.enabled is True
        assert s.eval.judge_model == "meta-llama/llama-4-maverick"
