"""WebSurfer Agent: researches the target construct via web search.

Uses Tavily search tool for web research, then summarizes findings
with temperature=0 for factual accuracy (per paper guidelines).
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_agent_settings, get_settings
from src.models import create_llm
from src.prompts.templates import WEBSURFER_SYSTEM, WEBSURFER_TASK
from src.schemas.state import MainState
from src.utils.console import print_agent_message

logger = structlog.get_logger(__name__)


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

    # Gather search results
    all_results = []
    for query in search_queries:
        try:
            result = await _search_tavily(query)
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
    summary = response.content

    logger.info("web_surfer_done", summary_length=len(summary))

    print_agent_message("WebSurfer", "Critic", summary)

    return {
        "research_summary": summary,
        "current_phase": "item_generation",
        "messages": [f"[WebSurfer] Research completed for {construct_name}"],
    }
