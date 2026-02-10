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

**Communication style:** Both the paper and our implementation use **natural language text** for all agent-to-agent communication. No JSON structured output — reviewers produce markdown tables and narrative feedback, the meta editor synthesizes them as text, and the item writer reads text feedback for revisions. This matches the paper's AutoGen GroupChat approach exactly.

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
| **Output format** | Natural language chat messages | Natural language text (plain strings in state) |
| **Reviewer output** | Text tables + narrative feedback | Markdown tables + narrative feedback |
| **Meta editor output** | KEEP/REVISE/DISCARD recommendations as text | Same — text-based recommendations |
| **Item writer output** | Numbered list of items | Numbered list with rationale per item |
| **Structured output** | None (AutoGen uses chat messages) | None — `llm.ainvoke(messages)` → `response.content` |

**Key design decision:** The paper uses AutoGen's GroupChat where agents communicate via natural language messages — there is no JSON parsing or structured output extraction. Our implementation matches this: all agents produce plain text, and the state holds string fields (`items_text`, `review_text`, `content_review`, etc.). This eliminates JSON parsing failures entirely and makes the agent communication visible and debuggable.

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
| **Orbiting dimensions** | Passed to content reviewer | `review_chain_wrapper` looks up orbiting dims from `AAAW_CONSTRUCT` and injects as `dimension_info` text |

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
| **Resume** | Continue chat | `graph.astream(Command(resume="feedback"), config)` |
| **Actions** | Approve or provide feedback | `"approve"` → done, any other text → revision |

**Advantage:** LangGraph's interrupt is more robust - the graph state is fully checkpointed, so the process can restart and resume from exactly where it paused. In production, `PostgresSaver` would replace `MemorySaver` for persistence across process restarts.

---

## 8. Quality Assessment (Colquitt et al., 2019)

| Metric | Paper Formula | Threshold | Implementation |
|--------|--------------|-----------|----------------|
| **c-value** (content validity) | target_rating / (a-1), where a=7 | >= 0.83 | Content reviewer evaluates in text (7-point scale) |
| **d-value** (distinctiveness) | mean(target - orbiting) / (a-1) | >= 0.35 | Content reviewer evaluates in text |
| **Linguistic** | 4 criteria, 5-point scale | Score of 5 = sound | Linguistic reviewer narrative assessment |
| **Bias** | 5-point scale (gender, religion, race, age, culture) | Score of 5 = unbiased | Bias reviewer narrative assessment |

**Content validity uses the Colquitt method:** The content reviewer is prompted to rate each item on a 7-point scale (1 = extremely bad job of measuring the concept; 7 = extremely good job) for (1) the target dimension and (2) two orbiting (related but distinct) dimensions. c-value = rating / 6, d-value = mean(target - orbiting) / 6. Each AAAW dimension has two pre-defined orbiting dimensions in `constructs.py`.

**Approach:** Matching the paper, reviewers produce text-based evaluations including ratings, and the meta editor synthesizes all reviews into KEEP/REVISE/DISCARD recommendations. No deterministic quality gates.

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

1. **Per-agent model configuration** - Each agent can use a different LLM model, controlled via `agents.toml`
2. **LangSmith observability** - All LLM calls and graph executions are traced
3. **LangGraph Studio support** - Visual web UI for debugging via `langgraph dev`
4. **Structured logging** - structlog with agent-level context (timings, phases)
5. **Rich console output** - Colored, formatted agent messages matching paper's `AgentName (to Target):` style
6. **Checkpointer abstraction** - `checkpointer=False` pattern for LangGraph Platform compatibility
7. **LewMod (automated feedback)** - `--lewmod` flag replaces human-in-the-loop with an LLM-based senior psychometrician agent. Evaluates items holistically and decides APPROVE or REVISE. No max_revisions limit — LewMod controls its own termination. Same graph structure, only the feedback node function is swapped.
8. **Centralized config** - `agents.toml` for all agent parameters (models, temperatures, num_items, max_results, etc.). Pydantic-validated via `AgentSettings` model.
9. **Retry + Fallback provider chain** - Two-layer reliability: (a) LangGraph `RetryPolicy` for node-level automatic retry with exponential backoff. (b) `with_fallbacks()` for LLM-level fallback: OpenRouter → Groq → Ollama. Both layers compose — each retry attempt runs the full fallback chain. Configured via `agents.toml` `[retry]` and `[providers]` sections. Per-agent fallback model overrides supported.

---

## 12. Known Limitations vs Paper

1. **Single construct**: Paper demonstrates multiple constructs; our implementation is configured for AAAW but extensible via `constructs.py`.

2. **No persistent storage**: Paper doesn't mention this, but our MemorySaver is in-memory only. Production would need PostgresSaver.

3. **Deterministic critic**: Paper uses an LLM-based critic that can make nuanced routing decisions. Our deterministic router follows a fixed phase sequence, which is simpler but less flexible.

4. **Llama vs GPT-4**: Llama 4 Maverick is capable but may produce different quality evaluations compared to GPT-4. The natural language approach mitigates this — there's no strict format to break, so the model can express its assessment freely.
