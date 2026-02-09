"""Rich console output for agent communication visualization.

Displays agent-to-agent messages in the paper's communication style:
  AgentName (to Target):
  <message content>

Each agent has a distinct color for quick visual identification.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console()

# Agent color mapping
AGENT_COLORS = {
    "Critic": "bold white",
    "WebSurfer": "cyan",
    "ItemWriter": "green",
    "ContentReviewer": "yellow",
    "LinguisticReviewer": "magenta",
    "BiasReviewer": "red",
    "MetaEditor": "blue",
    "Human": "bold bright_white",
    "LewMod": "bold bright_cyan",
}


def print_header(
    construct: str, model: str, max_revisions: int, lewmod: bool = False
) -> None:
    """Print the startup banner with retry/fallback status."""
    from src.config import get_agent_settings, get_settings

    agent_settings = get_agent_settings()
    settings = get_settings()

    feedback_mode = (
        "[bold bright_cyan]LewMod (automated)[/bold bright_cyan]"
        if lewmod
        else "Human-in-the-loop"
    )
    revisions_text = "Unlimited (LewMod decides)" if lewmod else str(max_revisions)

    # Build fallback chain display
    fallback_parts = []
    if agent_settings.providers.groq.enabled and settings.groq_api_key:
        fallback_parts.append("Groq")
    if agent_settings.providers.ollama.enabled:
        fallback_parts.append("Ollama")
    fallback_text = " → ".join(fallback_parts) if fallback_parts else "None"

    retry_cfg = agent_settings.retry
    retry_text = f"{retry_cfg.max_attempts} attempts (backoff: {retry_cfg.backoff_factor}x)"

    console.print()
    console.print(
        Panel(
            f"[bold]LM-AIG Multi-Agent Item Generation System[/bold]\n\n"
            f"  Construct: [cyan]{construct}[/cyan]\n"
            f"  Model: [cyan]{model}[/cyan]\n"
            f"  Max Revisions: [cyan]{revisions_text}[/cyan]\n"
            f"  Feedback Mode: {feedback_mode}\n"
            f"  Fallback: [cyan]{fallback_text}[/cyan]\n"
            f"  Retry: [cyan]{retry_text}[/cyan]",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )
    console.print()


def print_agent_message(from_agent: str, to_agent: str, content: str) -> None:
    """Print a message from one agent to another in the paper's format.

    Example output:
      ══════════════════════════════════════════
      ContentReviewer (to Critic):

      <message content rendered as markdown>
      ══════════════════════════════════════════
    """
    color = AGENT_COLORS.get(from_agent, "white")

    console.print()
    console.print(Rule(style=color))

    header = Text(f"  {from_agent} (to {to_agent}):", style=color)
    console.print(header)
    console.print()

    # Render content as markdown for tables, bold, etc.
    console.print(Markdown(content))

    console.print(Rule(style=color))


def print_phase_transition(phase: str) -> None:
    """Print a phase transition indicator."""
    phase_labels = {
        "web_research": "Web Research",
        "item_generation": "Item Generation",
        "review": "Review",
        "human_feedback": "Human Feedback",
        "revision": "Revision",
        "done": "Done",
    }
    label = phase_labels.get(phase, phase)
    console.print()
    console.print(f"  [bold bright_yellow]→ Phase: {label}[/bold bright_yellow]")
    console.print()


def print_human_prompt(review_summary: str) -> None:
    """Print the human feedback prompt with the review summary."""
    console.print()
    console.print(
        Panel(
            Markdown(review_summary),
            title="[bold bright_white]Items for Review[/bold bright_white]",
            border_style="bright_white",
            padding=(1, 2),
        )
    )
    console.print()


def print_final_results(review_text: str) -> None:
    """Print the final results."""
    console.print()
    console.print(
        Panel(
            Markdown(review_text),
            title="[bold green]Final Results[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


def print_info(message: str) -> None:
    """Print an informational message."""
    console.print(f"  [dim]{message}[/dim]")


def print_langsmith_status(enabled: bool) -> None:
    """Print LangSmith tracing status."""
    if enabled:
        console.print("  [green]LangSmith tracing: enabled[/green]")
    else:
        console.print("  [dim]LangSmith tracing: disabled[/dim]")
