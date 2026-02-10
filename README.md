# LM-AIG: Multi-Agent Automatic Item Generation

LLM-based multi-agent system for generating psychological test items (Likert-scale), based on **Lee et al. (2025)** - *"LLM-Based Multi-Agent AIG"*.

Built with **LangGraph** for orchestration, **Llama models** via **OpenRouter**, and **Tavily** for web research.

## Architecture

```
                    ┌──→ WebSurfer ────┐
                    │                  │
  START → Critic ──┼──→ Item Writer ──┼──→ Critic ──→ END
            ↑       │                  │       ↑
            │       ├──→ Review Chain ─┤       │
            │       │   Content  ─┐   │       │
            │       │   Linguistic ──→ Meta Editor
            │       │   Bias ─────┘   │       │
            │       └──→ Human (HITL) ─┘       │
            └──────────────────────────────────┘
```

**7 Agents:**

| Agent | Role | Temperature |
|-------|------|-------------|
| **Critic** | Central orchestrator (deterministic routing) | - |
| **WebSurfer** | Researches construct via Tavily web search | 0.0 |
| **Item Writer** | Generates/revises Likert-scale items | 1.0 |
| **Content Reviewer** | Evaluates content validity (c-value, d-value) | 0.0 |
| **Linguistic Reviewer** | Grammar, readability, ambiguity check | 0.0 |
| **Bias Reviewer** | Gender, cultural, socioeconomic fairness | 0.0 |
| **Meta Editor** | Synthesizes reviews → keep/revise/discard | 0.3 |

## Setup

### 1. Prerequisites

- Python >= 3.11
- [OpenRouter API key](https://openrouter.ai/keys)
- [Tavily API key](https://app.tavily.com)
- (Optional) [LangSmith API key](https://smith.langchain.com) for tracing

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Run

**CLI (human feedback):**
```bash
python run.py
```

**CLI (automated feedback via LewMod):**
```bash
python run.py --lewmod
```

**LangGraph Studio (web UI):**
```bash
langgraph dev
# Open: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

**Jupyter Notebook:**
```bash
jupyter notebook notebooks/demo.ipynb
```

## Agent Configuration

Agent models, temperatures, and parameters are configured in `agents.toml`:

```toml
[defaults]
model = "meta-llama/llama-4-maverick"

[agents.websurfer]
model = "meta-llama/llama-4-scout"
temperature = 0.0
max_results = 5
search_depth = "advanced"

[agents.item_writer]
temperature = 1.0
num_items = 8

[agents.content_reviewer]
temperature = 0.0

# See agents.toml for full config
```

API keys are stored in `.env` (secrets only).

## Reliability: Retry & Fallback Providers

The system has a two-layer reliability mechanism to handle API failures:

**Layer 1 — LangGraph RetryPolicy (node level):**
When an agent node fails (network timeout, 503, rate limit), LangGraph automatically retries it with exponential backoff. Configured in `agents.toml`:

```toml
[retry]
max_attempts = 3        # 1 initial + 2 retries
initial_interval = 1.0  # Seconds before first retry
backoff_factor = 2.0    # Exponential backoff multiplier
```

**Layer 2 — Fallback Provider Chain (LLM level):**
If the primary provider (OpenRouter) fails, the LLM call cascades to fallback providers:

```
OpenRouter (primary) → Groq (fallback 1) → Ollama (fallback 2, local)
```

Configured in `agents.toml`:

```toml
[providers.groq]
enabled = true
default_model = "llama-3.3-70b-versatile"

[providers.ollama]
enabled = true
default_model = "gpt-oss:20b"
base_url = "http://localhost:11434"
```

Each agent can override the fallback model:

```toml
[agents.websurfer]
groq_model = "llama-3.3-70b-versatile"
ollama_model = "gpt-oss:20b"
```

Requires `GROQ_API_KEY` in `.env` (Ollama needs no key — runs locally).

## Project Structure

```
agents.toml             # Agent config (models, temperatures, parameters)
src/
  config.py             # pydantic-settings (.env) + AgentSettings (agents.toml)
  models.py             # LLM factory with fallback chain (OpenRouter → Groq → Ollama)
  schemas/              # TypedDict states + construct definitions
  agents/               # 7 agent implementations
  graphs/               # LangGraph workflow + review subgraph
  prompts/              # Prompt templates (from paper Table 2)
  utils/                # Console output (rich)
tests/                  # 61 tests (schemas, agents, config, models, graph structure)
notebooks/demo.ipynb    # End-to-end demo
run.py                  # CLI entry point
langgraph.json          # LangGraph Studio configuration
```

## Testing

```bash
pytest tests/ -v
```

## Observability

Enable LangSmith tracing by setting in `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-key
LANGCHAIN_PROJECT=lm-aig
```

All LLM calls and graph executions will be traced at [smith.langchain.com](https://smith.langchain.com).

## Reference

Lee, H., et al. (2025). *LLM-Based Multi-Agent Automatic Item Generation for Psychological Scale Development.* Springer.
