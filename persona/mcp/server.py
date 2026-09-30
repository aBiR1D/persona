"""MCP server exposing Persona Market tools via FastMCP."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from persona.engine import PersonaEngine
from persona.knowledge import KnowledgeBase
from persona.marketplace import Marketplace
from persona.models import (
    CommunicationStyle,
    PersonaProfile,
    QAPair,
    TrainingData,
)

mcp = FastMCP("persona-market")

kb = KnowledgeBase()
engine = PersonaEngine(kb=kb)
marketplace = Marketplace(kb=kb)


@mcp.tool()
async def create_persona(
    name: str,
    title: str,
    expertise_areas: list[str],
    personality_traits: list[str] | None = None,
    communication_style: str = "professional",
    pricing_per_query: float = 5.0,
) -> dict:
    """Create a new AI persona that clones someone's expertise.

    Args:
        name: Display name for the persona (e.g. 'Dr. Sarah Chen')
        title: Professional title (e.g. 'Board-Certified Cardiologist')
        expertise_areas: List of expertise domains
        personality_traits: Personality descriptors
        communication_style: One of formal, casual, academic, friendly, professional, socratic
        pricing_per_query: Price per consultation in USD
    """
    style = CommunicationStyle(communication_style)
    profile = PersonaProfile(
        name=name,
        title=title,
        expertise_areas=expertise_areas,
        personality_traits=personality_traits or ["helpful", "knowledgeable"],
        communication_style=style,
        pricing_per_query=pricing_per_query,
    )
    created = await engine.create_persona(profile)
    return created.model_dump(mode="json")


@mcp.tool()
async def chat_with_persona(persona_id: str, query: str) -> dict:
    """Chat with an AI persona — ask questions and get expert answers.

    Args:
        persona_id: ID of the persona to chat with
        query: Your question or message
    """
    conv = await engine.chat(persona_id, query)
    return {
        "response": conv.response,
        "tokens_used": conv.tokens_used,
        "cost": conv.cost,
    }


@mcp.tool()
async def list_personas(query: str = "", category: str = "") -> list[dict]:
    """Browse available AI personas on the marketplace.

    Args:
        query: Optional search term
        category: Optional expertise category filter
    """
    if query or category:
        personas = await marketplace.search_personas(query=query, category=category)
    else:
        personas = await marketplace.list_personas()
    return [p.model_dump(mode="json") for p in personas]


@mcp.tool()
async def train_persona(
    persona_id: str,
    text_samples: list[str] | None = None,
    qa_pairs: list[dict] | None = None,
) -> dict:
    """Add knowledge to a persona to make it smarter.

    Args:
        persona_id: ID of the persona to train
        text_samples: Free-text knowledge to ingest
        qa_pairs: List of {"question": ..., "answer": ...} pairs
    """
    parsed_qa = [QAPair(**p) for p in (qa_pairs or [])]
    td = TrainingData(
        text_samples=text_samples or [],
        qa_pairs=parsed_qa,
    )
    count = await engine.train(persona_id, td)
    return {"persona_id": persona_id, "entries_added": count}


if __name__ == "__main__":
    mcp.run()
