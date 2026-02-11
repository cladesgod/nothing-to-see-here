# LM-AIG: Multi-Agent Automatic Item Generation

LLM-based multi-agent system for generating psychological test items (Likert-scale), based on **Lee et al. (2025)**.

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

8 agents. Critic routes, 3 reviewers run in parallel, MetaEditor synthesizes. LewMod can replace human feedback for fully automated runs.

## Setup

**1. Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
```

**2. Install dependencies**

```bash
pip install -e ".[dev]"
```

**3. Configure API keys**

```bash
cp .env.example .env
```

Edit `.env` and set:

| Key | Required | Source |
|-----|----------|--------|
| `OPENROUTER_API_KEY` | Yes | [openrouter.ai/keys](https://openrouter.ai/keys) |
| `TAVILY_API_KEY` | Yes | [app.tavily.com](https://app.tavily.com) |
| `GROQ_API_KEY` | Optional | [console.groq.com/keys](https://console.groq.com/keys) (fallback provider) |
| `LANGCHAIN_API_KEY` | Optional | [smith.langchain.com](https://smith.langchain.com) (tracing) |

## Configuration

All agent behavior is controlled via `agents.toml` in the project root:

```toml
[defaults]
model = "meta-llama/llama-4-maverick"   # Default LLM for all agents

[agents.item_writer]
temperature = 1.0                       # Creative diversity
num_items = 8                           # Items per generation cycle

[workflow]
max_revisions = 5                       # Max revision rounds

[providers.groq]
enabled = true                          # Fallback: OpenRouter → Groq → Ollama
```

See `agents.toml` for full options (models, temperatures, retry policy, fallback providers, JSON fixer settings).

## Usage

```bash
# Default: human-in-the-loop feedback, AAAW construct
python run.py

# Automated feedback via LewMod (no human input needed)
python run.py --lewmod

# Custom psychological construct from JSON file
python run.py --construct-file examples/custom_construct.json

# Show raw JSON output from reviewers
python run.py --verbose-json

# LangGraph Studio (web UI)
langgraph dev
```

## Testing

```bash
pytest tests/ -v    # 238 tests, no real API calls
```

## Observability

Set in `.env` to enable LangSmith tracing:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-key
LANGCHAIN_PROJECT=lm-aig
```

## Reference

Lee, H., et al. (2025). *LLM-Based Multi-Agent Automatic Item Generation for Psychological Scale Development.* Springer.
