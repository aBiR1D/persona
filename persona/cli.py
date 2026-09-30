"""Typer CLI — interactive persona creation, chat, training, and stats."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from persona.engine import PersonaEngine
from persona.knowledge import KnowledgeBase
from persona.marketplace import Marketplace
from persona.models import (
    CommunicationStyle,
    PersonaProfile,
    QAPair,
    TrainingData,
)

app = typer.Typer(
    name="persona",
    help="Persona Market — Clone your expertise into an AI persona and sell access.",
    add_completion=False,
)
console = Console()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ------------------------------------------------------------------
# create
# ------------------------------------------------------------------

@app.command()
def create():
    """Interactive wizard to create a new AI persona."""
    console.print(Panel("[bold cyan]✨ Persona Creation Wizard[/bold cyan]", expand=False))

    name = typer.prompt("Persona name")
    title = typer.prompt("Professional title (e.g. 'Senior Cardiologist')")
    areas_raw = typer.prompt("Expertise areas (comma-separated)")
    expertise_areas = [a.strip() for a in areas_raw.split(",") if a.strip()]

    traits_raw = typer.prompt("Personality traits (comma-separated)", default="helpful, knowledgeable")
    personality_traits = [t.strip() for t in traits_raw.split(",") if t.strip()]

    styles = [s.value for s in CommunicationStyle]
    style_str = typer.prompt(
        f"Communication style ({', '.join(styles)})",
        default="professional",
    )
    try:
        communication_style = CommunicationStyle(style_str)
    except ValueError:
        communication_style = CommunicationStyle.PROFESSIONAL

    summary = typer.prompt("Knowledge base summary", default="")
    pricing = typer.prompt("Price per query ($)", default="5.0", type=float)

    profile = PersonaProfile(
        name=name,
        title=title,
        expertise_areas=expertise_areas,
        personality_traits=personality_traits,
        communication_style=communication_style,
        knowledge_base_summary=summary,
        pricing_per_query=pricing,
    )

    engine = PersonaEngine()
    _run(engine.create_persona(profile))

    console.print(Panel(
        f"[bold green]✅ Persona created![/bold green]\n\n"
        f"[bold]Name:[/bold] {profile.name}\n"
        f"[bold]ID:[/bold] {profile.id}\n"
        f"[bold]Title:[/bold] {profile.title}\n"
        f"[bold]Expertise:[/bold] {', '.join(profile.expertise_areas)}\n"
        f"[bold]Price:[/bold] ${profile.pricing_per_query:.2f}/query",
        title="Persona Profile",
        expand=False,
    ))


# ------------------------------------------------------------------
# train
# ------------------------------------------------------------------

@app.command()
def train(
    persona_name: str = typer.Argument(..., help="Name of the persona to train"),
    file: Path = typer.Option(..., "--file", "-f", help="Path to knowledge file (text or .yaml)"),
):
    """Add knowledge to a persona from a file."""
    kb = KnowledgeBase()
    persona_data = _run(kb.find_persona_by_name(persona_name))
    if not persona_data:
        console.print(f"[red]Persona '{persona_name}' not found.[/red]")
        raise typer.Exit(1)

    if not file.exists():
        console.print(f"[red]File '{file}' does not exist.[/red]")
        raise typer.Exit(1)

    text = file.read_text(encoding="utf-8")
    engine = PersonaEngine(kb=kb)

    if file.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
        qa_pairs = [QAPair(**p) for p in data.get("qa_pairs", [])]
        td = TrainingData(
            text_samples=data.get("text_samples", []),
            qa_pairs=qa_pairs,
            expertise_docs=data.get("expertise_docs", []),
            style_examples=data.get("style_examples", []),
        )
    else:
        td = TrainingData(text_samples=[text])

    count = _run(engine.train(persona_data["id"], td))
    console.print(f"[green]✅ Ingested {count} knowledge entries for '{persona_name}'.[/green]")


# ------------------------------------------------------------------
# chat
# ------------------------------------------------------------------

@app.command()
def chat(
    persona_name: str = typer.Argument(..., help="Name of the persona to chat with"),
):
    """Interactive chat session with a persona."""
    kb = KnowledgeBase()
    persona_data = _run(kb.find_persona_by_name(persona_name))
    if not persona_data:
        console.print(f"[red]Persona '{persona_name}' not found.[/red]")
        raise typer.Exit(1)

    profile = PersonaProfile(**persona_data)
    console.print(Panel(
        f"[bold cyan]💬 Chatting with {profile.name}[/bold cyan]\n"
        f"{profile.title}\n"
        f"${profile.pricing_per_query:.2f} per query  •  Type 'quit' to exit",
        expand=False,
    ))

    engine = PersonaEngine(kb=kb)
    while True:
        try:
            query = console.input("[bold green]You:[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("quit", "exit", "q"):
            break
        if not query.strip():
            continue

        try:
            conv = _run(engine.chat(profile.id, query))
            console.print(Panel(conv.response, title=profile.name, border_style="cyan"))
            console.print(f"  [dim]tokens: {conv.tokens_used} | cost: ${conv.cost:.2f}[/dim]\n")
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")


# ------------------------------------------------------------------
# list
# ------------------------------------------------------------------

@app.command(name="list")
def list_personas():
    """List all available personas on the marketplace."""
    marketplace = Marketplace()
    personas = _run(marketplace.list_personas())

    if not personas:
        console.print("[yellow]No personas found. Create one with 'persona create'.[/yellow]")
        return

    table = Table(title="🏪 Persona Marketplace", show_lines=True)
    table.add_column("Name", style="bold cyan")
    table.add_column("Title")
    table.add_column("Expertise")
    table.add_column("Style")
    table.add_column("Price", justify="right")
    table.add_column("Active")

    for p in personas:
        table.add_row(
            p.name,
            p.title,
            ", ".join(p.expertise_areas[:3]),
            p.communication_style.value,
            f"${p.pricing_per_query:.2f}",
            "✅" if p.is_active else "❌",
        )
    console.print(table)


# ------------------------------------------------------------------
# stats
# ------------------------------------------------------------------

@app.command()
def stats(
    persona_name: str = typer.Argument(..., help="Name of the persona"),
):
    """Show earnings and usage statistics for a persona."""
    kb = KnowledgeBase()
    marketplace = Marketplace(kb=kb)

    persona_data = _run(kb.find_persona_by_name(persona_name))
    if not persona_data:
        console.print(f"[red]Persona '{persona_name}' not found.[/red]")
        raise typer.Exit(1)

    profile = PersonaProfile(**persona_data)
    persona_stats = _run(marketplace.get_stats(profile.id))
    earnings = _run(marketplace.calculate_earnings(profile.id))

    console.print(Panel(
        f"[bold]📊 Stats for {profile.name}[/bold]\n\n"
        f"Total conversations: {persona_stats.total_conversations}\n"
        f"Average rating:      {persona_stats.avg_rating:.1f} / 5.0\n"
        f"Total revenue:       ${persona_stats.total_revenue:.2f}\n"
        f"Platform commission: ${earnings['platform_commission']:.2f} ({earnings['commission_rate']})\n"
        f"Creator earnings:    ${persona_stats.creator_earnings:.2f}",
        title="Persona Statistics",
        expand=False,
    ))


# ------------------------------------------------------------------
# __main__
# ------------------------------------------------------------------

def main():
    app()


if __name__ == "__main__":
    main()
