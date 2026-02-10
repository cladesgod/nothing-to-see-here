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

# Run CLI (human feedback)
python run.py

# Run CLI (LewMod — automated expert feedback)
python run.py --lewmod

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
agents.toml             # Agent behavior config (models, temperatures, parameters)
src/
  config.py             # pydantic-settings (.env) + AgentSettings (agents.toml)
  models.py             # create_llm() factory with fallback chain (OpenRouter → Groq → Ollama)
  logging_config.py     # structlog setup
  schemas/
    state.py            # MainState (outer), ReviewChainState (inner) - TypedDict, text fields
    constructs.py       # AAAW (AI in Workplace) 6-dimension construct + orbiting dimensions
  agents/
    critic.py           # critic_node (visible node) + critic_router (conditional edge)
    web_surfer.py       # Tavily search + LLM summary (temp=0)
    item_writer.py      # generates/revises items as natural language text (temp=1.0)
    content_reviewer.py # Colquitt method: markdown table output (temp=0)
    linguistic_reviewer.py  # 4 criteria, narrative evaluation (temp=0)
    bias_reviewer.py    # 5-point bias scale, narrative evaluation (temp=0)
    meta_editor.py      # synthesizes reviews → keep/revise/discard text (temp=0.3)
    lewmod.py           # LewMod: automated expert feedback, replaces human-in-the-loop (temp=0.3)
  graphs/
    review_chain.py     # inner subgraph (parallel reviewers → meta_editor)
    main_workflow.py    # outer graph (critic hub-spoke), module-level `graph` for langgraph dev
  persistence/
    db.py               # SQLite connection + 6-table schema (WAL mode)
    repository.py       # CRUD functions for all pipeline data + get_previous_items()
  prompts/
    templates.py        # all system/task prompts per agent
  utils/
    console.py          # rich console output helpers (agent messages, phase transitions, panels)
```

## Key Conventions

### Naming
- Agent config in `agents.toml`: `[agents.websurfer]`, `[agents.item_writer]`, etc.
- Phases are lowercase strings: `web_research`, `item_generation`, `review`, `human_feedback`, `revision`, `done`

### State Pattern
- Outer graph: `MainState` (TypedDict, `total=False`). `items_text` and `review_text` are plain strings. `messages` uses `operator.add` (log accumulation). `run_id` and `db_path` for persistence. `previously_approved_items` uses `operator.add` for cross-round diversity.
- Inner graph: `ReviewChainState` (TypedDict, `total=False`). All fields are plain strings (content_review, linguistic_review, bias_review, meta_review).
- Nodes return partial dicts. Only returned keys get updated.
- DB persistence: each node opens its own `sqlite3.connect(db_path)` (not serializable in state).

### Agent Communication
- **Natural language text** — agents communicate via plain text strings, matching the paper's AutoGen chat-based approach (Lee et al., 2025)
- No JSON structured output (`json_schema` mode removed entirely)
- All agents use `llm.ainvoke(messages)` → `response.content` (plain text)
- Reviewers process all items as a batch (3 reviewer + 1 meta = 4 LLM calls, not per-item)
- Rich console shows paper-style `AgentName (to Target):` formatted messages

### Graph Compilation
- `build_main_workflow(checkpointer=None, lewmod=False)` → MemorySaver + human feedback (CLI/standalone)
- `build_main_workflow(checkpointer=False)` → no checkpointer (LangGraph Platform)
- `build_main_workflow(lewmod=True)` → LewMod replaces human feedback node
- Module-level `graph = build_main_workflow(checkpointer=False)` is what langgraph.json references

### Human-in-the-Loop / LewMod
- **Human mode (default):** `interrupt(payload)` in `human_feedback_node` pauses the graph. Resume with `Command(resume="feedback text")` or `Command(resume="approve")`. Requires a checkpointer and `thread_id` in config.
- **LewMod mode (`--lewmod`):** `lewmod_node` replaces `human_feedback_node`. Senior psychometrician LLM agent provides automated feedback. No interrupt — graph runs continuously. LewMod decides when to approve (no max_revisions limit).

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
max_revisions = 3
memory_enabled = true      # Use previous run items for diversity (anti-homogeneity)
memory_limit = 5           # How many prior runs' items to include
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

Quality assessment is done by LLM reviewers in natural language. Reviewers are prompted with these criteria:

| Metric | Formula | Threshold |
|--------|---------|-----------|
| c-value (content validity) | target_rating / 6 (7-point scale, a-1=6) | >= 0.83 |
| d-value (distinctiveness) | mean(target - orbiting) / 6 | >= 0.35 |
| Linguistic | 4 criteria, 5-point scale | qualitative assessment |
| Bias | 5-point scale (gender, religion, race, age, culture) | qualitative assessment |

## Testing

```bash
pytest tests/ -v               # all tests (144 total)
pytest tests/test_schemas.py   # construct + state schema tests
pytest tests/test_agents.py    # critic node + router + lewmod tests
pytest tests/test_workflow.py  # graph structure + retry policy tests
pytest tests/test_config.py    # agents.toml config, retry, provider tests
pytest tests/test_models.py    # LLM factory + fallback chain tests
pytest tests/test_evals.py     # golden dataset, score parsing, caching
pytest tests/test_persistence.py  # SQLite DB, CRUD, anti-homogeneity queries
```

Tests use mock API keys (`OPENROUTER_API_KEY=test-key`) set in `conftest.py`. No real API calls in tests.

## Related Documentation

- `PAPER_VS_IMPLEMENTATION.md` - Detailed comparison: what the paper does vs what we changed and why
- `IMPROVEMENTS.md` - Karsilasilan sorunlar, cozumler ve sonuclar (gelistirme raporu)

## Changelog

All significant additions and changes to the codebase are logged here.

**Iyilestirme sonuclari:** Her iyilestirme sonrasinda sonuclar `IMPROVEMENTS.md`'ye eklenmeli (sorun → cozum → sonuc formati).

### 2026-02-10: Smart Fallback — Timeout + Min Response Length
- **New feature:** Two additional fallback triggers for the provider chain (OpenRouter → Groq → Ollama). Previously only exceptions triggered a fallback.
- **Timeout:** Configurable per-provider request timeout (`defaults.timeout = 120`). When exceeded, the request raises an exception and `with_fallbacks()` cascades to the next provider.
- **Min response length:** Each provider is piped with a response-length validator (`LLM | validator`). Responses below `defaults.min_response_length` chars raise a `ValueError`, triggering the next provider. This catches "suspiciously short" responses that would otherwise pass through.
- **Architecture:** Uses `RunnableLambda` pipe pattern: `primary | validator` → each fallback also piped. `with_fallbacks()` catches both timeout exceptions and short-response ValueErrors.
- **Config:** `agents.toml` `[defaults]` section: `timeout = 120`, `min_response_length = 50`.
- **Tests:** 144 total (was 133). Added 7 validator tests + 2 chain structure tests + 2 config tests.
- **Files changed:** `agents.toml`, `src/config.py`, `src/models.py`, `tests/test_models.py`, `tests/test_config.py`, `CLAUDE.md`

### 2026-02-10: SQLite Persistence + Anti-Homogeneity Guard
- **New feature:** SQLite persistence layer (`src/persistence/`) for full pipeline state. 6 tables: `runs`, `research`, `generation_rounds`, `reviews`, `feedback`, `eval_results`. Zero new dependencies (`sqlite3` is Python stdlib).
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
