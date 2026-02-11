# CLAUDE.md - LM-AIG Multi-Agent System

#RULES

1- Talk turkish to the user but do jobs in english.
2- Always check langchain docs with mcp for better understanding.
3- Check best practicies with context7 mcp.

## Project Summary

LLM-based Multi-Agent Automatic Item Generation (AIG) system for psychological test development. Based on Lee et al. (2025). Generates, reviews, and refines Likert-scale items through a multi-agent pipeline using LangGraph.

## Quick Reference

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run CLI (human feedback, AAAW preset — default)
python run.py

# Run CLI (explicit preset)
python run.py --preset aaaw

# Run CLI (custom construct from JSON)
python run.py --construct-file examples/custom_construct.json

# Run CLI (LewMod — automated expert feedback)
python run.py --lewmod

# Run API server (multi-user, scalable)
pip install -e ".[api]"
uvicorn src.api.app:app --reload

# Run with Docker Compose (SQLite, dev)
docker compose up

# Run with Docker Compose (PostgreSQL, production)
docker compose --profile production up

# LangGraph Studio (web UI)
langgraph dev

# LangGraph Studio (Safari)
langgraph dev --tunnel
```

## Tech Stack

- **Python 3.13** (requires >=3.11)
- **LangGraph** - graph orchestration, StateGraph, subgraphs, interrupt()
- **LangChain OpenAI** - ChatOpenAI pointed at OpenRouter
- **LangChain Groq** - ChatGroq fallback provider
- **LangChain Ollama** - ChatOllama local fallback provider
- **OpenRouter** - LLM provider (Llama 4 models, primary)
- **Tavily** - web search for WebSurfer agent
- **Pydantic / pydantic-settings** - config
- **structlog** - structured logging
- **rich** - colored console output (agent communication, phase transitions)
- **LangSmith** - tracing/observability (opt-in via LANGCHAIN_TRACING_V2=true)
- **FastAPI** - REST API server (optional, `pip install -e ".[api]"`)
- **Prometheus** - metrics export (optional, via prometheus_client)
- **PostgreSQL** - production database (optional, via psycopg + langgraph-checkpoint-postgres)
- **Docker** - containerized deployment (Dockerfile + docker-compose.yml)

## Architecture

### Outer Graph (main_workflow.py)

Hub-spoke pattern with Critic as central node:

```
START → critic → web_surfer → critic → item_writer → critic → review_chain → critic → human_feedback → critic → END
```

All worker nodes return to `critic`. The `critic_node` is the visible node; `critic_router` is the conditional edge function that reads `current_phase` from state.

### Inner Subgraph (review_chain.py)

Fan-out / fan-in pattern:

```
START → content_reviewer ──┐
START → linguistic_reviewer ──→ meta_editor → END
START → bias_reviewer ─────┘
```

Three reviewers run in **parallel**, results converge at Meta Editor.

### Phase Flow (current_phase values)

`web_research` → `item_generation` → `review` → `human_feedback` → `revision` → (loop or `done`)

## File Structure

```
agents.toml             # Agent behavior config (models, temperatures, parameters, json_fix)
examples/
  custom_construct.json # Example custom construct definition (Job Satisfaction)
src/
  config.py             # pydantic-settings (.env) + AgentSettings (agents.toml)
  models.py             # create_llm() factory with fallback chain (OpenRouter → Groq → Ollama)
  logging_config.py     # structlog setup
  schemas/
    state.py            # MainState (outer), ReviewChainState (inner) - TypedDict
    constructs.py       # Construct presets (AAAW built-in), registry, JSON loader, fingerprint
    agent_outputs.py    # Pydantic output schemas for ALL agents (structured JSON)
    phases.py           # Phase StrEnum (web_research, item_generation, review, etc.)
  agents/
    critic.py           # critic_node (visible node) + critic_router (conditional edge)
    web_surfer.py       # Tavily search + LLM summary (temp=0)
    item_writer.py      # generates/revises items via structured JSON (temp=1.0), item freeze logic
    content_reviewer.py # Colquitt method: structured JSON ratings (temp=0)
    linguistic_reviewer.py  # 4 criteria, structured JSON ratings (temp=0)
    bias_reviewer.py    # 5-point bias scale, structured JSON ratings (temp=0)
    meta_editor.py      # synthesizes reviews → structured JSON decisions (temp=0.3)
    lewmod.py           # LewMod: automated expert feedback, structured JSON (temp=0.3)
  graphs/
    review_chain.py     # inner subgraph (parallel reviewers → meta_editor)
    main_workflow.py    # outer graph (critic hub-spoke), deterministic scoring layer
  api/                  # REST API layer (optional, pip install -e ".[api]")
    app.py              # FastAPI application + routes (/api/v1/runs, health, metrics)
    schemas.py          # Request/response Pydantic models
    auth.py             # API key authentication (SHA-256 hashed, timing-safe)
    queue.py            # WorkerPool — async worker pool with bounded concurrency
    rate_limiter.py     # Token bucket rate limiter + per-user concurrency limiter
    metrics.py          # Prometheus metrics (runs, queue depth, LLM calls, latency)
    dependencies.py     # FastAPI dependency injection (auth, rate limiter, pool)
  persistence/
    db.py               # SQLite + PostgreSQL connection abstraction (WAL / psycopg)
    repository.py       # CRUD functions for all pipeline data + get_previous_items()
  prompts/
    templates.py        # all system/task prompts per agent
  utils/
    console.py          # rich console output helpers (agent messages, structured display)
    structured_output.py    # invoke_structured_with_fix() — JSON fixer retry loop
    deterministic_scoring.py # build_deterministic_meta_review() — code-computed decisions
    injection_defense.py    # Two-layer LLM prompt injection defense for human feedback
Dockerfile              # Container build (Python 3.13 + API deps)
docker-compose.yml      # Dev (SQLite) + production (PostgreSQL) profiles
```

## Key Conventions

### Naming
- Agent config in `agents.toml`: `[agents.websurfer]`, `[agents.item_writer]`, etc.
- Phases use `Phase` StrEnum (`src/schemas/phases.py`): `web_research`, `item_generation`, `review`, `human_feedback`, `revision`, `done`

### State Pattern
- Outer graph: `MainState` (TypedDict, `total=False`). Key fields:
  - `items_text` / `active_items_text` / `review_text` — agent outputs (structured JSON strings)
  - `frozen_item_numbers: list[int]` — KEEP items preserved across revision rounds
  - `human_item_decisions: dict[str, str]` — per-item KEEP/REVISE from human/LewMod
  - `human_global_note: str` — optional free-text note from human
  - `dimension_info` / `construct_definition` / `construct_fingerprint` — construct context
  - `messages: Annotated[list[str], operator.add]` — log accumulation
  - `run_id` / `db_path` — persistence identifiers
  - `previously_approved_items: Annotated[list[str], operator.add]` — cross-round diversity
- Inner graph: `ReviewChainState` (TypedDict, `total=False`). Fields: `items_text`, `construct_name`, `construct_definition`, `dimension_info`, `content_review`, `linguistic_review`, `bias_review`, `meta_review`.
- Nodes return partial dicts. Only returned keys get updated.
- DB persistence: each node opens via `get_connection(db_path)` (WAL + foreign keys).

### Agent Communication
- **Structured JSON output** — all agents (reviewers, item writer, meta editor, LewMod) return Pydantic-validated JSON via `invoke_structured_with_fix()`. Schemas defined in `src/schemas/agent_outputs.py`.
- **JSON fixer retry loop** — if LLM output fails parsing, a fixer LLM re-attempts with error context. Config: `[json_fix]` in `agents.toml` (max_attempts=8, memory_window=4).
- **Deterministic scoring** — review chain wrapper applies `build_deterministic_meta_review()` to compute KEEP/REVISE/DISCARD from raw reviewer ratings in code (not LLM). Content c/d-value thresholds, linguistic min score, bias score all code-computed.
- Reviewers process all items as a batch (3 reviewer + 1 meta = 4 LLM calls, not per-item)
- Rich console shows paper-style `AgentName (to Target):` formatted messages with compact structured summaries via `format_structured_agent_output()`

### Graph Compilation
- `build_main_workflow(checkpointer=None, lewmod=False)` → MemorySaver + human feedback (CLI/standalone)
- `build_main_workflow(checkpointer=False)` → no checkpointer (LangGraph Platform)
- `build_main_workflow(lewmod=True)` → LewMod replaces human feedback node
- Module-level `graph = build_main_workflow(checkpointer=False)` is what langgraph.json references

### Human-in-the-Loop / LewMod
- **Human mode (default):** `interrupt(payload)` in `human_feedback_node` pauses the graph. CLI presents item-by-item KEEP/REVISE selection (not just free text). Resume with `Command(resume=dict)` or `Command(resume="approve")`. Requires a checkpointer and `thread_id` in config.
- **Item freeze mechanism:** Items marked KEEP become frozen (`frozen_item_numbers`). Only active (non-frozen) items get reviewed and revised in subsequent rounds. `active_items_text` tracks the subset being worked on.
- **LewMod mode (`--lewmod`):** `lewmod_node` replaces `human_feedback_node`. Returns structured `LewModOutput` with per-item keep/revise/discard lists. No interrupt — graph runs continuously. LewMod decides when to approve (no max_revisions limit).
- **`--verbose-json` CLI flag:** Shows raw JSON blocks for reviewer/meta outputs instead of compact summaries.

## Config

### `agents.toml` — Agent Behavior Config

Centralized configuration for all agent parameters. Located in project root.

```toml
[defaults]
model = "meta-llama/llama-4-maverick"
timeout = 120                # Request timeout per provider (seconds)
min_response_length = 50     # Min chars — below this, try next provider

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

# ... (see agents.toml for full config)

[workflow]
max_revisions = 5
memory_enabled = true      # Use previous run items for diversity (anti-homogeneity)
memory_limit = 5           # How many prior runs' items to include

[json_fix]
max_attempts = 8           # Fixer retry attempts for structured output parsing
memory_window = 4          # Recent errors shown to fixer LLM
```

Uses `tomllib` (Python stdlib). Loaded via `get_agent_settings()` in `src/config.py`. Pydantic-validated.

### Retry & Fallback

Two-layer reliability configured in `agents.toml`:

```toml
[retry]
max_attempts = 3           # LangGraph RetryPolicy (node level)
initial_interval = 1.0
backoff_factor = 2.0

[providers.groq]
enabled = true
default_model = "llama-3.3-70b-versatile"

[providers.ollama]
enabled = true
default_model = "gpt-oss:20b"
base_url = "http://localhost:11434"
```

- **Layer 1:** LangGraph `RetryPolicy` on all LLM nodes (not critic — deterministic, not human_feedback — interrupt)
- **Layer 2:** `create_llm()` uses `with_fallbacks()`: OpenRouter → Groq → Ollama. Each provider is piped with a response-length validator (`LLM | validator`).
- **Fallback triggers:** Exceptions (network errors, rate limits), timeouts (`defaults.timeout`), and short responses (`defaults.min_response_length` chars).
- Per-agent fallback model overrides: `groq_model` / `ollama_model` in `[agents.X]`
- Both layers compose: each retry runs the full fallback chain

### `.env` — Secrets Only

Required keys: `OPENROUTER_API_KEY`, `TAVILY_API_KEY`

Optional: `GROQ_API_KEY` (for Groq fallback), `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`

Model selection is done exclusively via `agents.toml` (no env var overrides).

## Quality Thresholds (Colquitt et al., 2019)

LLM reviewers output structured JSON ratings. Final KEEP/REVISE/DISCARD decisions are computed deterministically in code (`src/utils/deterministic_scoring.py`):

| Metric | Formula | Threshold | Decision Impact |
|--------|---------|-----------|-----------------|
| c-value (content validity) | target_rating / 6 (7-point scale, a-1=6) | >= 0.83 | Must pass for KEEP |
| d-value (distinctiveness) | mean(target - orbiting) / 6 | >= 0.35 | Must pass for KEEP |
| Linguistic min | min(grammar, understanding, negative_free, clarity) | >= 4 | Below 2 → DISCARD |
| Bias score | single 1-5 score | >= 4 | Below 2 → DISCARD |

Decision logic: `content_ok AND bias>=4 AND ling_min>=4` → KEEP; `bias<=2 OR ling_min<=2` → DISCARD; otherwise → REVISE.

## Testing

```bash
pytest tests/ -v                        # all tests (238 total)
pytest tests/test_schemas.py            # construct, state, agent_outputs, phases
pytest tests/test_agents.py             # critic node + router + lewmod + web search caching
pytest tests/test_workflow.py           # graph structure + retry policy + human feedback + item freeze
pytest tests/test_config.py             # agents.toml config, retry, provider, json_fix, api
pytest tests/test_models.py             # LLM factory + fallback chain tests
pytest tests/test_persistence.py        # SQLite DB, CRUD, anti-homogeneity, research cache
pytest tests/test_deterministic_scoring.py  # deterministic KEEP/REVISE/DISCARD rules
pytest tests/test_prompts.py            # prompt template formatting
pytest tests/test_run_report.py         # run report / KEEP metrics extraction
pytest tests/test_api.py               # API layer: auth, rate limiter, concurrency, schemas, DB abstraction
pytest tests/test_injection_defense.py # prompt injection defense: schema, layers, fail-open, config
```

Tests use mock API keys (`OPENROUTER_API_KEY=test-key`) set in `conftest.py`. No real API calls in tests.

## Related Documentation

- `PAPER_VS_IMPLEMENTATION.md` - Detailed comparison: what the paper does vs what we changed and why
- `IMPROVEMENTS.md` - Karsilasilan sorunlar, cozumler ve sonuclar (gelistirme raporu)
- `COMPARISON.md` - LM-AIG vs MAPIG (Psynalytics) feature comparison

## Changelog

All significant additions and changes to the codebase are logged here.

**Iyilestirme sonuclari:** Her iyilestirme sonrasinda sonuclar `IMPROVEMENTS.md`'ye eklenmeli (sorun → cozum → sonuc formati).

### 2026-02-10: Dual-LLM Prompt Injection Defense for Human Feedback
- **New feature:** Dual-LLM prompt injection defense for human feedback `global_note` input. Two independent LLMs perform the same injection check — model diversity means if one LLM is fooled by a sophisticated injection, the other catches it.
- **Architecture:** Layer 1 = Primary LLM (OpenRouter, `llama-4-scout`), Layer 2 = Cross-validation LLM (Groq, `llama-3.3-70b-versatile`). Same prompt, different models. Both must PASS. If either returns STOP with confidence >= threshold (default 0.7), run terminates (`Phase.DONE`) with generic rejection message. Layer 1 STOP skips Layer 2 (token savings). If Groq not configured, Layer 2 skipped (fail-open). Defense LLM errors → input passes through.
- **Configuration:** `[prompt_injection]` in `agents.toml` (enabled, threshold, min_input_length). `[agents.injection_classifier]` for model/temperature + Groq fallback model.
- **Injection vector:** `global_note` free text flows unsanitized into `ITEM_WRITER_REVISE` template via `_format_human_feedback_for_prompt()`. Primary injection surface — `item_decisions` are validated (KEEP/REVISE only).
- **invoke_structured_with_fix:** Added optional `llm` parameter to support pre-built LLM instances (used by Layer 2 to pass Groq LLM directly).
- **human_feedback_node:** Converted from `def` to `async def` for `await check_prompt_injection()`. LangGraph compatible.
- **New file:** `src/utils/injection_defense.py` (InjectionCheckResult schema, shared prompt, _create_groq_llm(), check_prompt_injection())
- **New test file:** `tests/test_injection_defense.py` — 17 tests: schema validation, config bypass, both-pass, layer1-stop, layer2-stop, below-threshold, groq-not-configured, fail-open, same-messages verification.
- **Tests:** 238 total (was 221). All existing workflow tests updated for async human_feedback_node.
- **Files changed:** `src/config.py`, `agents.toml`, `src/utils/structured_output.py`, `src/graphs/main_workflow.py`, `tests/test_workflow.py`, `CLAUDE.md`

### 2026-02-10: Enterprise API Layer — Queue System, Rate Limiting, Multi-User Support
- **FastAPI REST API:** New `src/api/` package with 7 modules. Endpoints: POST /api/v1/runs (submit), GET /api/v1/runs/{id} (status), GET /api/v1/runs (list), POST /api/v1/runs/{id}/feedback, DELETE /api/v1/runs/{id} (cancel), GET /api/v1/health, GET /api/v1/metrics (Prometheus).
- **Async worker pool:** `WorkerPool` with `asyncio.Semaphore` — bounded concurrency (configurable `max_workers`). Each pipeline run executes as an independent asyncio task. Tier 2 upgrade path to Celery+Redis documented.
- **API key authentication:** SHA-256 hashed keys with constant-time comparison (hmac.compare_digest). Keys loaded from `AIG_API_KEYS` env var. Dev key auto-registered when no keys configured.
- **Token bucket rate limiter:** Per-user minute and daily rate limits. Configurable via `agents.toml` `[api]` section.
- **Per-user concurrency limiter:** Limits concurrent pipeline runs per API key (default: 3). Prevents single user from monopolizing workers.
- **Prometheus metrics:** Runs submitted/completed counters, queue depth gauge, active workers gauge, LLM call counters, HTTP request latency histograms.
- **Database abstraction:** `get_connection()` now routes to SQLite or PostgreSQL based on URL scheme. `PG_SCHEMA_SQL` with SERIAL columns. psycopg lazy import (only when PostgreSQL URL provided).
- **PostgreSQL checkpointer:** `build_main_workflow(checkpointer="postgres", db_url=...)` uses `PostgresSaver` from `langgraph-checkpoint-postgres`.
- **Docker deployment:** `Dockerfile` (Python 3.13, non-root user) + `docker-compose.yml` with dev (SQLite) and production (PostgreSQL) profiles.
- **Config additions:** `APIConfig` model in `src/config.py`, `[api]` section in `agents.toml`, `DATABASE_URL` in `Settings`.
- **New dependencies (optional):** `fastapi`, `uvicorn`, `prometheus-client`, `langgraph-checkpoint-postgres`, `psycopg` — all under `[api]` extras group.
- **Tests:** 221 total (was 183). Added 38 API tests: auth (10), rate limiter (4), token bucket (5), concurrency limiter (6), schemas (9), config (2), DB abstraction (2).
- **Zero regression:** CLI mode (`python run.py`) works identically. API is additive — same graph, same agents, same DB.
- **New files:** `src/api/__init__.py`, `src/api/app.py`, `src/api/schemas.py`, `src/api/auth.py`, `src/api/queue.py`, `src/api/rate_limiter.py`, `src/api/metrics.py`, `src/api/dependencies.py`, `Dockerfile`, `docker-compose.yml`, `tests/test_api.py`
- **Modified files:** `src/config.py`, `agents.toml`, `src/persistence/db.py`, `src/graphs/main_workflow.py`, `pyproject.toml`, `CLAUDE.md`

### 2026-02-10: Structured JSON Output + Deterministic Scoring + Item Freeze
- **Structured JSON output restored:** All agents now return Pydantic-validated JSON via `invoke_structured_with_fix()`. New `src/schemas/agent_outputs.py` defines output schemas for all 8 agents (WebSurferOutput, ItemWriterOutput, ContentReviewerOutput, LinguisticReviewerOutput, BiasReviewerOutput, MetaEditorOutput, LewModOutput). JSON fixer retry loop with error memory (`[json_fix]` config in agents.toml).
- **Deterministic scoring layer:** `src/utils/deterministic_scoring.py` — `build_deterministic_meta_review()` computes KEEP/REVISE/DISCARD from raw reviewer ratings in code. Content c/d-value, linguistic min, bias score thresholds are code-enforced. Review chain wrapper applies this after LLM meta editor output.
- **Item freeze mechanism:** KEEP items are frozen across revision rounds (`frozen_item_numbers` in MainState). Only active items get reviewed/revised. `active_items_text` tracks the subset. Item numbers aligned via `_align_generated_to_targets()`. Frozen items enforced via `_enforce_keep_locks()`.
- **Structured human feedback:** CLI now presents item-by-item KEEP/REVISE selection (not just free text). `human_item_decisions: dict[str, str]` and `human_global_note: str` in MainState. LewMod also returns per-item decisions.
- **Run report with metrics:** Final output shows KEEP items with their deterministic metrics (c-value, d-value, ling_min, bias). Backfills from earlier rounds if current round metrics are missing.
- **`--verbose-json` CLI flag:** Shows raw JSON blocks for reviewer/meta outputs.
- **New files:** `src/schemas/agent_outputs.py`, `src/utils/deterministic_scoring.py`, `src/utils/structured_output.py`, `tests/test_deterministic_scoring.py`, `tests/test_prompts.py`, `tests/test_run_report.py`
- **Tests:** 209 total (was 173). Added deterministic scoring tests, prompt tests, run report tests, item freeze workflow tests.
- **Files changed:** All agent files, `src/schemas/state.py`, `src/config.py`, `agents.toml`, `src/graphs/main_workflow.py`, `src/utils/console.py`, `run.py`, all test files.

### 2026-02-10: Complexity Cleanup + Consistency Pass
- **Run stability:** `run.py` now resolves `max_revisions` at runtime (not argparse parse-time), uses safer DB path resolution from `PRAGMA database_list` row fields with fallback to `DB_PATH`, and wraps DB lifecycle in `try/finally` with failure status persistence.
- **Phase consistency:** Added `Phase` StrEnum (`src/schemas/phases.py`) and replaced scattered raw phase strings across core workflow/agent nodes.
- **DB access consistency:** Replaced scattered `sqlite3.connect(db_path)` calls in agents/graph wrapper with `get_connection(db_path)` to keep WAL/foreign-key behavior consistent.
- **Within-run memory fix:** `previously_approved_items` now actually accumulates prior revision-round item sets in Item Writer (used by anti-homogeneity prompt context).
- **Config simplification:** Removed unused config fields (`rate_limit_rpm`).
- **Docs drift fixes:** Updated `README.md`, `COMPARISON.md`, and `PAPER_VS_IMPLEMENTATION.md` to match current construct-agnostic implementation and current project/test reality.
- **Files changed:** `run.py`, `agents.toml`, `src/config.py`, `src/schemas/phases.py`, `src/schemas/state.py`, `src/agents/critic.py`, `src/agents/web_surfer.py`, `src/agents/item_writer.py`, `src/agents/lewmod.py`, `src/graphs/main_workflow.py`, `README.md`, `COMPARISON.md`, `PAPER_VS_IMPLEMENTATION.md`, `tests/test_config.py`, `CLAUDE.md`

### 2026-02-10: Fingerprint-Based Research Caching
- **New feature:** Two-layer caching for WebSurfer research, keyed by construct fingerprint (SHA-256).
- **Layer 1 — DB research reuse:** Before running Tavily + LLM, checks if a prior run with the same exact construct already has a research_summary in the DB. If found (within TTL), skips the entire WebSurfer pipeline (0 API calls). New function: `get_cached_research()` in repository.py.
- **Layer 2 — File cache key fix:** Tavily search result file cache now uses `fingerprint[:12]` instead of `construct_name` slug. Prevents cache pollution between different constructs that share a name.
- **TTL:** Both layers respect `cache_ttl_hours` from `agents.toml` (default: 24 hours).
- **No schema changes:** Uses existing `research` table joined through `runs.construct_fingerprint`.
- **Tests:** 173 total (was 167). Added 5 DB research cache tests + 1 file cache fingerprint isolation test.
- **Files changed:** `src/persistence/repository.py`, `src/agents/web_surfer.py`, `tests/test_persistence.py`, `CLAUDE.md`

### 2026-02-10: Construct Fingerprint — Memory Correctness
- **New feature:** SHA-256 fingerprint of the full construct definition (name + dimensions + orbiting). Used for both anti-homogeneity memory filtering and research caching.
- **Problem solved:** With construct-agnostic design, `get_previous_items()` filtered by `construct_name` — two different "Job Satisfaction" constructs would share memory incorrectly. Now filters by exact fingerprint.
- **Architecture:** `compute_fingerprint(construct)` uses `model_dump_json()` + `hashlib.sha256()`. Stored in `runs.construct_fingerprint` DB column with auto-migration for existing DBs.
- **State change:** `MainState` gets `construct_fingerprint: str` field.
- **Tests:** 167 total (was ~156). Added 5 fingerprint hash tests + 2 persistence fingerprint tests + 1 null fingerprint test.
- **Files changed:** `src/schemas/constructs.py`, `src/schemas/state.py`, `src/persistence/db.py`, `src/persistence/repository.py`, `run.py`, `src/agents/item_writer.py`, `tests/test_schemas.py`, `tests/test_persistence.py`, `CLAUDE.md`

### 2026-02-10: Construct-Agnostic Design
- **New feature:** The system now accepts any psychological construct at runtime, not just AAAW. Built-in presets (currently AAAW) plus custom construct loading from JSON files.
- **CLI:** `--preset aaaw` (default) or `--construct-file path/to/construct.json`. Mutually exclusive. `--construct` flag replaced by `--preset`.
- **Architecture:** Dimension info is now pre-built at the entry point via `build_dimension_info(construct)` and passed through `MainState.dimension_info`. The `review_chain_wrapper` in `main_workflow.py` no longer imports `AAAW_CONSTRUCT` — it reads dimension info from state.
- **Preset registry:** `CONSTRUCT_PRESETS` dict in `constructs.py` maps preset names to `Construct` objects. `get_preset()`, `list_presets()` helpers.
- **JSON loader:** `load_construct_from_file(path)` loads and Pydantic-validates a custom construct from a JSON file.
- **Backward compatible:** `python run.py` with no args defaults to `--preset aaaw` — identical behavior to before.
- **State change:** `MainState` gets `dimension_info: str` field (pre-formatted dimension + orbiting text).
- **Tests:** ~156 total (was 144). Added 16 new tests: preset registry (6), build_dimension_info (5), load_construct_from_file (4), MainState dimension_info (1).
- **New files:** `examples/custom_construct.json`
- **Files changed:** `src/schemas/constructs.py`, `src/schemas/state.py`, `src/graphs/main_workflow.py`, `run.py`, `tests/test_schemas.py`, `CLAUDE.md`

### 2026-02-10: Smart Fallback — Timeout + Min Response Length
- **New feature:** Two additional fallback triggers for the provider chain (OpenRouter → Groq → Ollama). Previously only exceptions triggered a fallback.
- **Timeout:** Configurable per-provider request timeout (`defaults.timeout = 120`). When exceeded, the request raises an exception and `with_fallbacks()` cascades to the next provider.
- **Min response length:** Each provider is piped with a response-length validator (`LLM | validator`). Responses below `defaults.min_response_length` chars raise a `ValueError`, triggering the next provider. This catches "suspiciously short" responses that would otherwise pass through.
- **Architecture:** Uses `RunnableLambda` pipe pattern: `primary | validator` → each fallback also piped. `with_fallbacks()` catches both timeout exceptions and short-response ValueErrors.
- **Config:** `agents.toml` `[defaults]` section: `timeout = 120`, `min_response_length = 50`.
- **Tests:** 144 total (was 133). Added 7 validator tests + 2 chain structure tests + 2 config tests.
- **Files changed:** `agents.toml`, `src/config.py`, `src/models.py`, `tests/test_models.py`, `tests/test_config.py`, `CLAUDE.md`

### 2026-02-10: SQLite Persistence + Anti-Homogeneity Guard
- **New feature:** SQLite persistence layer (`src/persistence/`) for full pipeline state. 5 tables: `runs`, `research`, `generation_rounds`, `reviews`, `feedback`. Zero new dependencies (`sqlite3` is Python stdlib).
- **Anti-homogeneity:** `get_previous_items()` fetches final items from completed runs for the same construct. Item Writer reads these from DB and injects into prompts to push for diversity. `previously_approved_items` state field accumulates across revision rounds within a run. Configurable via `agents.toml`: `memory_enabled` (on/off) and `memory_limit` (how many prior runs).
- **Prompt changes:** `ITEM_WRITER_GENERATE` and `ITEM_WRITER_REVISE` get `{previously_approved_items}` section. `META_EDITOR_TASK` gets Rule 6 (inter-item similarity check).
- **Agent persistence:** All agents now persist to DB — WebSurfer (research), ItemWriter (generation rounds), ReviewChain (reviews), Human/LewMod (feedback). Each node opens its own `sqlite3.connect(db_path)`.
- **State changes:** `MainState` gets `run_id: str`, `db_path: str`, `previously_approved_items: Annotated[list[str], operator.add]`.
- **Tests:** 133 total (was 103). Added 28 persistence tests (DB schema, CRUD, anti-homogeneity queries, `_format_item_history` helper) + 2 memory config tests.
- **New files:** `src/persistence/__init__.py`, `src/persistence/db.py`, `src/persistence/repository.py`, `tests/test_persistence.py`
- **Files changed:** `src/schemas/state.py`, `src/agents/item_writer.py`, `src/agents/web_surfer.py`, `src/agents/lewmod.py`, `src/graphs/main_workflow.py`, `src/prompts/templates.py`, `run.py`, `.gitignore`, `FUTURE.md`, `CLAUDE.md`

### 2026-02-09: Retry Mechanism + Fallback Provider Chain
- **New feature:** Two-layer reliability for all LLM calls. (1) LangGraph `RetryPolicy` on all LLM nodes with exponential backoff. (2) `with_fallbacks()` provider chain: OpenRouter → Groq → Ollama.
- **Architecture:** Both layers compose — each retry attempt runs the full fallback chain. `create_llm()` returns `BaseChatModel` (either plain `ChatOpenAI` or `RunnableWithFallbacks`).
- **Config:** `agents.toml` `[retry]` section (max_attempts, initial_interval, backoff_factor). `[providers.groq]` and `[providers.ollama]` sections (enabled, default_model, base_url). Per-agent `groq_model`/`ollama_model` overrides.
- **Lazy imports:** `langchain_groq` and `langchain_ollama` only imported when their provider is enabled — no ImportError if not installed.
- **RetryPolicy assignment:** web_surfer, item_writer, review_chain, lewmod get retry. Critic (deterministic) and human_feedback (interrupt) do not.
- **Tests:** 61 total (was 40). Added 8 config tests (RetryConfig, ProviderConfig), 5 model tests (fallback chain), 6 workflow tests (RetryPolicy on nodes), 2 TOML tests.
- **New dependencies:** `langchain-groq`, `langchain-ollama`
- **New file:** `tests/test_models.py`
- **Files changed:** `pyproject.toml`, `.env.example`, `agents.toml`, `src/config.py`, `src/models.py`, `src/graphs/main_workflow.py`, `src/graphs/review_chain.py`, `src/utils/console.py`, `tests/conftest.py`, `tests/test_config.py`, `tests/test_workflow.py`, `README.md`, `IMPROVEMENTS.md`, `PAPER_VS_IMPLEMENTATION.md`, `CLAUDE.md`

### 2026-02-09: Project Cleanup & Best Practice
- **Removed unused dependencies:** `langchain-community`, `httpx`, `python-dotenv` from `pyproject.toml` (zero imports in src/).
- **Removed dead code:** `src/utils/rate_limiter.py` and `src/utils/retry.py` — neither was imported anywhere. Available in git history if needed.
- **Fixed broken notebook:** `notebooks/demo.ipynb` updated to current API (plain text state, `get_agent_settings()`, async graph streaming). Removed references to deleted `LikertItem`, `settings.get_model()`, and JSON structured output.
- **Improved exception handling:** `run.py` now catches `GraphInterrupt` directly instead of fragile `str(type(e))` string matching.
- **Updated .gitignore:** Added `.DS_Store`, `.langgraph_api/`, `.ruff_cache/`, `.mypy_cache/`.
- **Updated all documentation:** README.md (test count, LewMod, agents.toml config), IMPROVEMENTS.md (marked structured output as historical, added new features), PAPER_VS_IMPLEMENTATION.md (agents.toml references, removed stale file references).
- **Files changed:** `pyproject.toml`, `run.py`, `.gitignore`, `notebooks/demo.ipynb`, `README.md`, `IMPROVEMENTS.md`, `PAPER_VS_IMPLEMENTATION.md`, `CLAUDE.md`
- **Files deleted:** `src/utils/rate_limiter.py`, `src/utils/retry.py`

### 2026-02-09: Centralized Agent Config (`agents.toml`)
- **New feature:** `agents.toml` in project root — centralized configuration for all agent parameters (models, temperatures, num_items, max_results, search_depth, max_revisions).
- **Format:** TOML — parsed via `tomllib` (Python stdlib, zero dependencies). Pydantic-validated via `AgentSettings` model.
- **Priority chain:** CLI args > env vars (.env) > agents.toml > hardcoded defaults.
- **Architecture:** `get_agent_settings()` singleton in `src/config.py`. `create_llm()` auto-resolves temperature from TOML when not explicitly passed.
- **Agent changes:** All 7 agent files no longer hardcode temperature. WebSurfer reads `max_results`/`search_depth` from config. ItemWriter reads `num_items` from config.
- **Tests:** 40 total (was 30). Added 10 config tests (defaults, overrides, TOML file parsing).
- **Files changed:** `agents.toml` (new), `tests/test_config.py` (new), `src/config.py`, `src/models.py`, all 7 agent files, `run.py`, `CLAUDE.md`, `.env.example`

### 2026-02-09: LewMod — Automated Expert Feedback Agent
- **New feature:** LewMod replaces human-in-the-loop with an automated senior psychometrician LLM agent. Activated via `python run.py --lewmod`.
- **Character:** Dr. LewMod — senior psychometrician persona. Evaluates items holistically and decides APPROVE (done) or REVISE (with specific feedback). Prompt guides toward approval after 2-3 revision rounds.
- **Architecture:** Same graph node name `"human_feedback"` — only the backing function changes (`lewmod_node` vs `human_feedback_node`). No changes to critic, routing, or edges.
- **No max_revisions limit:** LewMod controls its own termination (max_revisions set to 999 internally).
- **New file:** `src/agents/lewmod.py`
- **Tests:** 30 total (was 25). Added 3 LewMod decision parsing tests + 2 LewMod graph structure tests.
- **Files changed:** `src/agents/lewmod.py` (new), `src/prompts/templates.py`, `src/config.py`, `src/utils/console.py`, `src/graphs/main_workflow.py`, `run.py`, `tests/test_agents.py`, `tests/test_workflow.py`, `CLAUDE.md`

### 2026-02-09: Prompt Alignment with Paper Table 2
- **Item Writer:** All 10 guidelines from paper Table 2 incorporated. Reverse-coded items now avoided (paper recommendation). Academic citations removed from prompts.
- **Content Reviewer:** Scale corrected from 1-6 to 7-point (1-7) matching paper. c-value threshold adjusted to >= 0.83. Formulas use a-1=6 divisor.
- **Linguistic Reviewer:** Reduced from 5 to 4 criteria (paper Table 2): grammatical accuracy, ease of understanding, negative language, clarity/directness. Removed extra "conciseness" criterion.
- **Bias Reviewer:** Aligned to paper categories (gender, religion, race, age, culture). Removed extra categories.
- **Meta Editor:** Simplified to paper's 4 responsibilities: synthesize, edit/discard, integrate human feedback, identify remaining issues.
- **Files changed:** `src/prompts/templates.py`, `CLAUDE.md`, `PAPER_VS_IMPLEMENTATION.md`

### 2026-02-09: Natural Language Communication + Rich Console Output
- **Change:** Removed all JSON structured output (`json_schema` mode, `with_structured_output`, `invoke_with_repair`). Agents now communicate via natural language text, matching the paper's AutoGen chat-based approach.
- **Rich console:** Added `rich` library for colored, formatted console output. Each agent has a distinct color. Messages displayed as `AgentName (to Target):` panels.
- **Batch review:** Review chain now processes all items in a single batch (4 LLM calls) instead of per-item (32 LLM calls).
- **State simplification:** `MainState` uses `items_text: str` and `review_text: str` instead of `list[LikertItem]` and `list[MetaEditorRecommendation]`. `ReviewChainState` uses `content_review: str`, `linguistic_review: str`, `bias_review: str`, `meta_review: str`.
- **Deleted files:** `src/schemas/items.py`, `src/utils/structured_output.py`, `src/utils/validators.py` (no longer needed).
- **New file:** `src/utils/console.py` (rich console helpers).
- **Tests:** 25 tests (was 42). Removed Pydantic model tests and validator tests.
- **Files changed:** All agent files, state.py, templates.py, review_chain.py, main_workflow.py, run.py, all test files, pyproject.toml, CLAUDE.md

### 2026-02-08: Repair Layer — Missing Brace + Quote Fix
- **Change:** `repair_json()` now handles OpenRouter stripping both `{` and `"` from JSON start. If text doesn't start with `"`, inserts `{"` instead of just `{`.
- **Reason:** OpenRouter sometimes strips `{"` together, producing `gender_neutrality": 5, ...}` instead of `{"gender_neutrality": 5, ...}`. The previous repair only added `{`, leaving invalid JSON.
- **Files changed:** `src/utils/structured_output.py`

### 2026-02-08: max_tokens 8192 + Few-Shot Examples in Reviewer Prompts
- **Change:** `max_tokens` increased from 4096 → 8192 on all 4 agents (content_reviewer, linguistic_reviewer, bias_reviewer, meta_editor). Added few-shot JSON output examples to CONTENT_REVIEWER_TASK, LINGUISTIC_REVIEWER_TASK, BIAS_REVIEWER_TASK prompts.
- **Reason:** meta_editor and linguistic_reviewer were hitting the 4096 token limit (JSON cut off mid-generation). Few-shot examples guide the model to produce compact, correctly-structured JSON.
- **Files changed:** `src/agents/content_reviewer.py`, `src/agents/linguistic_reviewer.py`, `src/agents/bias_reviewer.py`, `src/agents/meta_editor.py`, `src/prompts/templates.py`

### 2026-02-08: Paper Uyumluluk Duzeltmeleri
- **Construct:** AAAW = "Attitudes Toward the Use of AI in the Workplace" (Park et al., 2024). Previously wrong ("Adjustment to Academic Work"). 6 dimensions: AI Use Anxiety, Personal Utility, Perceived Humanlikeness of AI, Perceived Adaptability of AI, Perceived Quality of AI, Job Insecurity.
- **Content Validity:** Colquitt et al. (2019) method. 3 ratings (target + 2 orbiting) instead of 4 subscales. c-value=target/6 (>= 0.88), d-value=mean(target-orb)/6 (>= 0.35). Each dimension has 2 pre-defined orbiting dimensions.
- **State Accumulation Bug:** `items` and `review_results` now use `replace_list` reducer (replace on revision, not append 8→16→24). `messages` stays with `operator.add`.
- **Linguistic Criteria:** Added `stylistic_consistency` and `negative_language_free` (paper criteria). Now 6 criteria total (5 from paper + our `conciseness`). Average /6.
- **Bias Categories:** Added `ethnic_neutrality`, `racial_neutrality`, `sexual_orientation_neutrality` (paper categories). Now 7 total (6 from paper + our `socioeconomic_neutrality`). Average /7.
- **Tests:** 42 tests (was 31). New tests for orbiting dimensions, Colquitt formulas, new criteria/categories.
- **Files changed:** `src/schemas/constructs.py` (rewrite), `src/schemas/items.py`, `src/schemas/state.py`, `src/utils/validators.py`, `src/prompts/templates.py`, `src/agents/content_reviewer.py`, `src/agents/linguistic_reviewer.py`, `src/agents/bias_reviewer.py`, `src/agents/meta_editor.py`, `src/graphs/main_workflow.py`, `tests/conftest.py`, `tests/test_schemas.py`, `IMPROVEMENTS.md`, `CLAUDE.md`

### 2026-02-08: JSON Repair → LangChain Built-in parse_json_markdown
- **Change:** Replaced custom markdown stripping and closing-brace fix in `repair_json()` with LangChain's built-in `parse_json_markdown` from `langchain_core.utils.json`. Only custom code remaining is the missing opening `{` fix (1 line), which LangChain doesn't handle.
- **Repair pipeline:** `repair_json()` (brace fix) → `parse_json_markdown()` (markdown strip + partial JSON) → `schema.model_validate(dict)` (Pydantic validation)
- **Removed:** `import json`, `import re` (no longer needed). Custom markdown stripping and closing-brace logic.
- **Files changed:** `src/utils/structured_output.py`

### 2026-02-08: JSON Repair Layer for Structured Output
- **Problem:** Maverick on OpenRouter's `json_schema` mode sometimes returns JSON without the opening `{` brace (e.g., `  "relevance": 6,...}`), causing ~30-50% parsing failures.
- **Solution:** Created `src/utils/structured_output.py` with `invoke_with_repair()` helper. Uses `include_raw=True` to get raw LLM text on failure, then `repair_json()` fixes common issues (missing braces, markdown code blocks) before re-parsing with Pydantic.
- **Files changed:** `src/utils/structured_output.py` (new), `src/agents/content_reviewer.py`, `src/agents/linguistic_reviewer.py`, `src/agents/bias_reviewer.py`, `src/agents/meta_editor.py`

### 2026-02-08: Reviewer Models → Llama 4 Maverick + Pydantic json_schema_extra
- **Change:** All 3 reviewer agents (content, linguistic, bias) switched from `llama-3.3-70b-instruct` to `llama-4-maverick`
- **Reason:** Llama 3.3 70B doesn't natively support `structured_outputs` (json_schema), causing ~20-30% structured output failures. Maverick supports it natively → ~0% failure rate.
- **Also:** Added `json_schema_extra` with example outputs to `ContentValidityScore`, `LinguisticScore`, `BiasScore`, `MetaEditorRecommendation` schemas. This embeds examples directly in the JSON Schema sent to the LLM, improving output quality.
- **Files changed:** `.env`, `.env.example`, `src/schemas/items.py`
- **Cost impact:** Negligible (~$0.002 per full run of 24 reviewer calls)

### 2026-02-08: Structured Output Fix
- **Problem:** `llama-3.3-70b-instruct` generated 16K+ tokens without producing valid JSON using `method="function_calling"` (default). `llama-4-maverick` doesn't support `tools` on OpenRouter at all.
- **Investigation:** Tested `json_mode` (model outputs wrong field names/nesting), `function_calling` + `max_tokens=1024` (too low), `function_calling` + `max_tokens=4096` (still fails ~30% for linguistic reviewer).
- **Solution:** Switched ALL agents to `method="json_schema"` with `strict=True`. This uses OpenRouter's `response_format.json_schema` which both Maverick and Llama 3.3 support. Added `max_tokens=4096` as safety net.
- **Files changed:** `src/agents/content_reviewer.py`, `src/agents/linguistic_reviewer.py`, `src/agents/bias_reviewer.py`, `src/agents/meta_editor.py`, `src/agents/item_writer.py`, `src/models.py`
- **Residual issue:** Llama 3.3 still occasionally fails (~20% of calls). Fallback scores are used when this happens.

### 2026-02-08: max_tokens Parameter Added to create_llm
- **Change:** `src/models.py` `create_llm()` now accepts optional `max_tokens` parameter
- **Reason:** Prevent token runaway in reviewer agents (Llama 3.3 generating 16K tokens without valid output)

### 2026-02-08: Model Change - Reviewer Agents
- **Change:** `meta-llama/llama-3.2-90b-vision-instruct` → `meta-llama/llama-3.3-70b-instruct`
- **Reason:** The 3.2-90b model doesn't exist on OpenRouter (404 error). Listed available Llama models and selected 3.3-70b as the closest replacement.
- **Files changed:** `.env`, `.env.example`

### 2026-02-08: Architecture Refactor - Parallel Reviewers
- **Change:** Review chain changed from sequential (content → linguistic → bias → meta) to parallel fan-out/fan-in
- **Reason:** Paper Figure 2 shows reviewers running in parallel. LangGraph supports this natively with multiple edges from START.
- **Files changed:** `src/graphs/review_chain.py`

### 2026-02-08: Architecture Refactor - Critic as Visible Node
- **Change:** Critic changed from invisible conditional edge to visible graph node + conditional edge
- **Reason:** Paper shows Critic as central orchestrator. Making it a visible node means: (a) it appears in graph visualization, (b) it can emit messages/log, (c) it handles max_revisions logic.
- **Files changed:** `src/agents/critic.py`, `src/graphs/main_workflow.py`

### 2026-02-08: LangGraph Platform Compatibility
- **Change:** `build_main_workflow(checkpointer=False)` → no checkpointer (for LangGraph Platform/Studio)
- **Reason:** LangGraph Platform provides its own checkpointer. Custom `MemorySaver` causes `ValueError`.
- **Files changed:** `src/graphs/main_workflow.py`, `langgraph.json`

### 2026-02-08: LikertItem.construct → construct_name
- **Change:** `LikertItem.construct` field renamed to `construct_name`
- **Reason:** `construct` shadows Pydantic `BaseModel.construct()` class method, causing deprecation warning
- **Files changed:** `src/schemas/items.py` + all files referencing `item.construct`

### 2026-02-08: Initial Implementation
- Full project scaffold: pyproject.toml, .env, config, models, logging
- All 7 agents implemented
- Review chain subgraph (parallel) + main workflow (critic hub-spoke)
- 31 unit tests (schemas, agents, graph structure)
- CLI entry point (run.py) + demo notebook
- LangSmith tracing integration
- CLAUDE.md, README.md documentation
