"""FastAPI application — REST endpoints for the Persona Market platform."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from persona.engine import PersonaEngine
from persona.knowledge import KnowledgeBase
from persona.marketplace import Marketplace
from persona.models import (
    CommunicationStyle,
    PersonaProfile,
    PersonaStats,
    QAPair,
    TrainingData,
)

app = FastAPI(
    title="Persona Market API",
    description="Clone your expertise into an AI persona and sell access.",
    version="0.1.0",
)

kb = KnowledgeBase()
engine = PersonaEngine(kb=kb)
marketplace = Marketplace(kb=kb)


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class CreatePersonaRequest(BaseModel):
    name: str
    title: str
    expertise_areas: list[str]
    personality_traits: list[str] = []
    communication_style: CommunicationStyle = CommunicationStyle.PROFESSIONAL
    knowledge_base_summary: str = ""
    pricing_per_query: float = 5.0


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    cost: float


class TrainRequest(BaseModel):
    text_samples: list[str] = []
    qa_pairs: list[dict] = []
    expertise_docs: list[str] = []
    style_examples: list[str] = []


class TrainResponse(BaseModel):
    entries_added: int


class SearchRequest(BaseModel):
    query: str = ""
    category: str = ""


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "persona-market"}


# ------------------------------------------------------------------
# Persona CRUD
# ------------------------------------------------------------------

@app.post("/api/v1/personas", response_model=dict)
async def create_persona(req: CreatePersonaRequest):
    profile = PersonaProfile(**req.model_dump())
    created = await engine.create_persona(profile)
    return created.model_dump(mode="json")


@app.get("/api/v1/personas", response_model=list[dict])
async def list_personas(query: Optional[str] = None, category: Optional[str] = None):
    if query or category:
        personas = await marketplace.search_personas(query=query or "", category=category or "")
    else:
        personas = await marketplace.list_personas()
    return [p.model_dump(mode="json") for p in personas]


@app.get("/api/v1/personas/{persona_id}", response_model=dict)
async def get_persona(persona_id: str):
    data = await kb.get_persona(persona_id)
    if not data:
        raise HTTPException(status_code=404, detail="Persona not found")
    return data


# ------------------------------------------------------------------
# Chat
# ------------------------------------------------------------------

@app.post("/api/v1/personas/{persona_id}/chat", response_model=ChatResponse)
async def chat_with_persona(persona_id: str, req: ChatRequest):
    try:
        conv = await engine.chat(persona_id, req.query)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ChatResponse(
        response=conv.response,
        tokens_used=conv.tokens_used,
        cost=conv.cost,
    )


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------

@app.post("/api/v1/personas/{persona_id}/train", response_model=TrainResponse)
async def train_persona(persona_id: str, req: TrainRequest):
    data = await kb.get_persona(persona_id)
    if not data:
        raise HTTPException(status_code=404, detail="Persona not found")

    qa_pairs = [QAPair(**p) for p in req.qa_pairs]
    td = TrainingData(
        text_samples=req.text_samples,
        qa_pairs=qa_pairs,
        expertise_docs=req.expertise_docs,
        style_examples=req.style_examples,
    )
    count = await engine.train(persona_id, td)
    return TrainResponse(entries_added=count)


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------

@app.get("/api/v1/personas/{persona_id}/stats", response_model=dict)
async def persona_stats(persona_id: str):
    data = await kb.get_persona(persona_id)
    if not data:
        raise HTTPException(status_code=404, detail="Persona not found")
    stats = await marketplace.get_stats(persona_id)
    return stats.model_dump()
