"""WebSurfer Agent: researches the target construct via web search.

Uses Tavily search tool for web research, then summarizes findings
with temperature=0 for factual accuracy (per paper guidelines).

Includes construct-based caching to avoid redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_agent_settings, get_settings
from src.models import create_llm
from src.persistence.repository import save_research
from src.prompts.templates import WEBSURFER_SYSTEM, WEBSURFER_TASK
from src.schemas.state import MainState
from src.utils.console import print_agent_message, validate_llm_response

logger = structlog.get_logger(__name__)

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "web_search"


def _cache_path(construct_name: str, query: str) -> Path:
    """Build a deterministic cache file path for a query."""
    slug = construct_name.lower().replace(" ", "_")
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{slug}__{query_hash}.json"


def _read_cache(construct_name: str, query: str, ttl_hours: int) -> str | None:
    """Return cached results if present and not expired, else None."""
    path = _cache_path(construct_name, query)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(data["timestamp"])
        age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
        if age_hours < ttl_hours:
            logger.info("cache_hit", query=query[:60], age_hours=round(age_hours, 1))
            return data["results"]
        logger.info("cache_expired", query=query[:60], age_hours=round(age_hours, 1))
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning("cache_corrupt_removing", path=str(path))
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return None


def _write_cache(construct_name: str, query: str, results: str) -> None:
    """Persist search results to the cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(construct_name, query)
    payload = {
        "query": query,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("cache_write", path=str(path))


async def _search_tavily(query: str) -> str:
    """Run a Tavily search and return formatted results."""
    from tavily import TavilyClient

    settings = get_settings()
    client = TavilyClient(api_key=settings.tavily_api_key)
    agent_cfg = get_agent_settings().get_agent_config("websurfer")
    response = client.search(
        query=query,
        max_results=agent_cfg.max_results,
        search_depth=agent_cfg.search_depth,
    )

    results = []
    for r in response.get("results", []):
        results.append(f"**{r['title']}**\n{r['url']}\n{r['content']}\n")
    return "\n---\n".join(results) if results else "No results found."


async def _search_with_cache(construct_name: str, query: str) -> str:
    """Search with caching: return cached results if available, else fetch and cache."""
    agent_cfg = get_agent_settings().get_agent_config("websurfer")
    cache_enabled = getattr(agent_cfg, "cache_enabled", True)
    cache_ttl = getattr(agent_cfg, "cache_ttl_hours", 24)

    if cache_enabled:
        cached = _read_cache(construct_name, query, cache_ttl)
        if cached is not None:
            return cached

    results = await _search_tavily(query)

    if cache_enabled:
        _write_cache(construct_name, query, results)

    return results


async def web_surfer_node(state: MainState) -> dict:
    """WebSurfer agent node: searches and summarizes construct information."""
    construct_name = state.get("construct_name", "")
    construct_definition = state.get("construct_definition", "")

    logger.info("web_surfer_start", construct=construct_name)

    # Build search queries from construct info
    search_queries = [
        f"{construct_name} psychological scale validated items",
        f"{construct_name} Likert scale measurement psychometrics",
    ]

    # Gather search results (with caching)
    all_results = []
    for query in search_queries:
        try:
            result = await _search_with_cache(construct_name, query)
            all_results.append(f"### Query: {query}\n{result}")
        except Exception as e:
            logger.warning("tavily_search_error", query=query, error=str(e))
            all_results.append(f"### Query: {query}\nSearch failed: {e}")

    raw_research = "\n\n".join(all_results)

    llm = create_llm("websurfer")

    messages = [
        SystemMessage(content=WEBSURFER_SYSTEM),
        HumanMessage(
            content=WEBSURFER_TASK.format(
                construct_name=construct_name,
                construct_definition=construct_definition,
                dimension_name="(all dimensions)",
                dimension_definition="See research results below.",
            )
            + f"\n\n**Raw Search Results:**\n{raw_research}"
        ),
    ]

    response = await llm.ainvoke(messages)
    summary = validate_llm_response(response.content, "WebSurfer")

    logger.info("web_surfer_done", summary_length=len(summary))

    print_agent_message("WebSurfer", "Critic", summary)

    # Persist research summary to DB
    db_path = state.get("db_path")
    run_id = state.get("run_id")
    if db_path and run_id:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            save_research(conn, run_id, summary)
            conn.close()
        except Exception:
            logger.warning("web_surfer_db_write_failed", exc_info=True)

    return {
        "research_summary": summary,
        "current_phase": "item_generation",
        "messages": [f"[WebSurfer] Research completed for {construct_name}"],
    }
