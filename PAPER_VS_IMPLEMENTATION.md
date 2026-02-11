# Paper vs Implementation: Differences & Design Decisions

**Paper:** Lee, H., et al. (2025). *LLM-Based Multi-Agent Automatic Item Generation for Psychological Scale Development.* Springer.

**Implementation:** LM-AIG Multi-Agent System (this repository)

---

## 1. Framework & Orchestration

| Aspect | Paper | Our Implementation |
|--------|-------|--------------------|
| **Framework** | Microsoft AutoGen (nested chats) | LangGraph (StateGraph, subgraphs) |
| **Agent communication** | AutoGen `GroupChat` — agents exchange natural language messages | LangGraph state passing — agents write/read plain text fields in shared state |
| **Message format** | `AgentName (to Target): <text>` chat messages | Same style reproduced via `rich` console output |
| **State management** | AutoGen chat history | TypedDict state with `operator.add` reducer for message log |
| **Subgraph pattern** | Nested chat (inner chat within outer) | Compiled subgraph invoked from wrapper node |

**Why LangGraph?** AutoGen's nested chat pattern maps well to LangGraph's subgraph pattern, but LangGraph offers: native `interrupt()` for HITL, built-in checkpointing, streaming, and LangSmith observability. LangGraph also makes the graph topology explicit and inspectable.

**Communication style:** Both systems use LLM-generated content for agent communication. Our implementation adds a **structured JSON output layer** — all agents return Pydantic-validated JSON via `invoke_structured_with_fix()`, with a fixer retry loop for robustness. Console display uses `format_structured_agent_output()` for compact human-readable summaries.

---

## 2. LLM Models

| Aspect | Paper | Our Implementation |
|--------|-------|--------------------|
| **Provider** | OpenAI (direct) | OpenRouter (proxy) |
| **Primary model** | GPT-4 | Llama 4 Maverick (MoE, 400B total / 17B active) |
| **Reviewer models** | GPT-4 (same for all) | Llama 4 Maverick |
| **Research model** | GPT-4 + Bing Search | Llama 4 Scout + Tavily |
| **Per-agent config** | Not mentioned | Each agent can use a different model via `agents.toml` config |

**Why Llama?** The assignment context requires demonstrating multi-model orchestration. Using Llama models via OpenRouter shows: (a) model-agnostic architecture, (b) per-agent model optimization, (c) cost management (smaller models for simpler tasks).

---

## 3. Agent Output Format

| Aspect | Paper | Our Implementation |
|--------|-------|--------------------|
| **Output format** | Natural language chat messages | Structured JSON (Pydantic-validated) |
| **Reviewer output** | Text tables + narrative feedback | JSON with numeric ratings per item (`agent_outputs.py`) |
| **Meta editor output** | KEEP/REVISE/DISCARD recommendations as text | JSON with per-item decision + reason + revised stem |
| **Item writer output** | Numbered list of items | JSON with item_number, stem, rationale per item |
| **Structured output** | None (AutoGen uses chat messages) | `invoke_structured_with_fix()` — fixer retry loop with error memory |
| **Decision computation** | LLM decides KEEP/REVISE/DISCARD | **Deterministic scoring in code** from raw reviewer ratings |

**Key design decision:** The paper uses AutoGen's GroupChat where agents produce free text. Our implementation uses **structured JSON output** — all agents return Pydantic models (`ContentReviewerOutput`, `ItemWriterOutput`, etc.) via `invoke_structured_with_fix()`. If the LLM output fails JSON parsing, a fixer LLM re-attempts with error context (configurable: max_attempts=8, memory_window=4). This enables the **deterministic scoring layer** (`build_deterministic_meta_review()`) which computes KEEP/REVISE/DISCARD from raw numeric ratings using code-enforced thresholds — not LLM judgment.

---

## 4. Critic Agent (Orchestrator)

| Aspect | Paper | Our Implementation |
|--------|-------|--------------------|
| **Type** | LLM-based orchestrator | Deterministic router (no LLM call) |
| **Routing logic** | LLM decides next step | Python function reads `current_phase` from state |
| **Visibility** | Central agent in the group | Visible graph node (`critic_node`) + conditional edge (`critic_router`) |

**Why deterministic?** The paper's critic doesn't need LLM intelligence for routing - the workflow order is fixed (research → generate → review → feedback → revise). Using a deterministic router is: (a) faster (no API call), (b) cheaper, (c) more reliable, (d) testable without mocking.

The critic still handles important logic: checking `revision_count >= max_revisions` to prevent infinite loops.

---

## 5. Review Chain Architecture

| Aspect | Paper | Our Implementation |
|--------|-------|--------------------|
| **Reviewer execution** | Parallel (Figure 2 in paper) | Parallel (LangGraph fan-out / fan-in) |
| **Pattern** | 3 reviewers → meta editor | START → {content, linguistic, bias} → meta_editor → END |
| **State isolation** | Nested chat context | `ReviewChainState` (separate TypedDict from `MainState`) |
| **Item processing** | Each item reviewed as part of group discussion | **Batch review** — all items sent to each reviewer at once (4 LLM calls total) |
| **Orbiting dimensions** | Passed to content reviewer | Entry point builds `dimension_info` from selected construct (preset or JSON) and passes it through state |

**Batch review (4 LLM calls):** The paper's AutoGen setup sends items to reviewers as part of a group chat. Our implementation sends all items as a batch to each of the 3 reviewers + 1 meta editor = 4 LLM calls total. This is more efficient than per-item review (which would be 8 items × 4 agents = 32 calls).

**Fan-out/fan-in:** LangGraph's `add_edge(START, "content_reviewer")` + `add_edge(START, "linguistic_reviewer")` + `add_edge(START, "bias_reviewer")` creates automatic fan-out. All three `add_edge("reviewer", "meta_editor")` creates fan-in. The meta_editor waits for all three reviewers to complete before running.

---

## 6. Web Research

| Aspect | Paper | Our Implementation |
|--------|-------|--------------------|
| **Search API** | Bing Search | Tavily |
| **Integration** | AutoGen tool registration | Direct Tavily Python client in agent node |
| **Summary** | GPT-4 summarizes results | Llama 4 Scout summarizes results (temp=0) |

**Why Tavily?** Tavily is purpose-built for LLM research tasks, returns cleaner results than Bing, and has a generous free tier. The LangChain ecosystem has better Tavily integration than Bing.

---

## 7. Human-in-the-Loop (HITL)

| Aspect | Paper | Our Implementation |
|--------|-------|--------------------|
| **Mechanism** | AutoGen human proxy agent | LangGraph `interrupt()` + `Command(resume=...)` |
| **Persistence** | Chat history | `MemorySaver` checkpointer (in-memory) |
| **Resume** | Continue chat | `graph.astream(Command(resume=dict), config)` |
| **Actions** | Approve or provide feedback | Item-by-item KEEP/REVISE + optional global note, or `"approve"` |
| **Item freeze** | Not mentioned | KEEP items frozen across rounds (`frozen_item_numbers`) |
| **LewMod** | Not mentioned | Automated expert agent with structured per-item decisions |

**Advantage:** LangGraph's interrupt is more robust - the graph state is fully checkpointed. Our structured feedback (`human_item_decisions` dict) enables per-item control with item freezing — KEEP items are preserved unchanged across revision rounds, and only active items get re-reviewed. In production, `PostgresSaver` would replace `MemorySaver`.

---

## 8. Quality Assessment (Colquitt et al., 2019)

| Metric | Paper Formula | Threshold | Implementation |
|--------|--------------|-----------|----------------|
| **c-value** (content validity) | target_rating / (a-1), where a=7 | >= 0.83 | Content reviewer outputs JSON ratings → c-value computed in code |
| **d-value** (distinctiveness) | mean(target - orbiting) / (a-1) | >= 0.35 | Content reviewer outputs JSON ratings → d-value computed in code |
| **Linguistic** | 4 criteria, 5-point scale | Score of 5 = sound | Linguistic reviewer outputs JSON ratings → min score computed in code |
| **Bias** | 5-point scale (gender, religion, race, age, culture) | Score of 5 = unbiased | Bias reviewer outputs JSON score → threshold checked in code |

**Content validity uses the Colquitt method:** The content reviewer outputs structured JSON with `target_rating`, `orbiting_1_rating`, `orbiting_2_rating` per item. c-value = target / 6, d-value = mean(target - orbiting) / 6. Each construct dimension has two pre-defined orbiting dimensions in `constructs.py`.

**Deterministic scoring layer:** Unlike the paper where the meta editor LLM decides KEEP/REVISE/DISCARD, our implementation uses **code-computed decisions** in `build_deterministic_meta_review()`:
- `content_ok AND bias>=4 AND ling_min>=4` → **KEEP**
- `bias<=2 OR ling_min<=2` → **DISCARD**
- Otherwise → **REVISE**

The LLM meta editor still runs and provides synthesis, but final decisions come from deterministic code.

---

## 9. Temperature Settings

| Agent | Paper | Our Implementation |
|-------|-------|--------------------|
| **WebSurfer** | 0 (factual) | 0.0 |
| **Item Writer** | 1.0 (creative diversity) | 1.0 |
| **Content Reviewer** | 0 (consistent evaluation) | 0.0 |
| **Linguistic Reviewer** | 0 | 0.0 |
| **Bias Reviewer** | 0 | 0.0 |
| **Meta Editor** | Low (0.3 in paper context) | 0.3 |

All temperatures match the paper's recommendations exactly.

---

## 10. Construct Definition

| Aspect | Paper | Our Implementation |
|--------|-------|--------------------|
| **Test construct** | AAAW (Park et al., 2024) | AAAW - Attitudes Toward the Use of AI in the Workplace |
| **Dimensions** | 6 dimensions | AI Use Anxiety, Personal Utility, Perceived Humanlikeness of AI, Perceived Adaptability of AI, Perceived Quality of AI, Job Insecurity |
| **Orbiting dimensions** | 2 per dimension (for Colquitt d-value) | Pre-defined in `constructs.py`, looked up by `get_orbiting_definitions()` |
| **Item count** | Variable | 8 items per generation cycle |

---

## 11. What We Added (Not in Paper)

1. **Structured JSON output + fixer retry** — All agents return Pydantic-validated JSON. `invoke_structured_with_fix()` handles parse failures with a fixer LLM retry loop (max_attempts=8, error memory).
2. **Deterministic scoring layer** — Code-computed KEEP/REVISE/DISCARD from raw reviewer ratings using Colquitt thresholds. Reproducible, testable, not dependent on LLM judgment for final decisions.
3. **Item freeze mechanism** — KEEP items frozen across revision rounds. Only active items re-reviewed/revised. `frozen_item_numbers` in state, `active_items_text` subset tracking.
4. **Structured human feedback** — Per-item KEEP/REVISE selection in CLI (not just free text). `human_item_decisions` dict in state.
5. **Construct-agnostic design** — Any construct at runtime via `--preset` or `--construct-file`. SHA-256 fingerprint for memory correctness.
6. **Anti-homogeneity guard** — SQLite persistence + cross-run item memory. Previous items injected into prompts for diversity.
7. **Research caching** — Two-layer cache (DB + file) keyed by construct fingerprint with TTL.
8. **LewMod (automated feedback)** — `--lewmod` replaces human-in-the-loop with LLM senior psychometrician. Returns structured per-item decisions.
9. **Retry + Fallback provider chain** — LangGraph RetryPolicy + `with_fallbacks()`: OpenRouter → Groq → Ollama. Timeout + min response length triggers.
10. **Per-agent model configuration** — `agents.toml` for all agent parameters. Pydantic-validated.
11. **Rich console output** — Colored agent messages with compact structured summaries.
12. **LangSmith observability** — All LLM calls traced. LangGraph Studio support.
13. **Run report with metrics** — Final output shows KEEP items with deterministic metric values.

---

## 12. Known Limitations vs Paper

1. **Preset-first construct model**: System supports preset or JSON-based custom constructs, but lacks API-level rich metadata controls (target population, constraints, exclusions) found in production systems like MAPIG.

2. **In-memory checkpointer**: `MemorySaver` is in-memory only. Session state is lost if the process restarts. Production would need `PostgresSaver` or `AsyncSqliteSaver`.

3. **Deterministic critic**: Paper uses an LLM-based critic that can make nuanced routing decisions. Our deterministic router follows a fixed phase sequence, which is simpler but less flexible.

4. **Llama vs GPT-4**: Llama 4 Maverick is capable but may produce different quality evaluations compared to GPT-4. The structured JSON output + fixer retry mitigates format issues.
