"""Domain models for the Persona Market platform."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class CommunicationStyle(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    ACADEMIC = "academic"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    SOCRATIC = "socratic"


class PersonaProfile(BaseModel):
    """Public-facing profile of a cloned-expertise persona."""

    id: str = Field(default_factory=_new_id)
    name: str = Field(..., min_length=1, max_length=120)
    title: str = Field(..., min_length=1, max_length=200)
    expertise_areas: list[str] = Field(default_factory=list, min_length=1)
    personality_traits: list[str] = Field(default_factory=list)
    communication_style: CommunicationStyle = CommunicationStyle.PROFESSIONAL
    knowledge_base_summary: str = ""
    pricing_per_query: float = Field(default=5.0, ge=0.0)
    creator_id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True


class KnowledgeEntry(BaseModel):
    """A single piece of extracted knowledge stored in the knowledge base."""

    id: str = Field(default_factory=_new_id)
    persona_id: str = ""
    topic: str
    content: str
    source: str = "manual"
    category: str = "general"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class QAPair(BaseModel):
    """A question-answer pair used for persona training."""

    question: str
    answer: str


class TrainingData(BaseModel):
    """Corpus supplied by the creator to shape the persona."""

    text_samples: list[str] = Field(default_factory=list)
    qa_pairs: list[QAPair] = Field(default_factory=list)
    expertise_docs: list[str] = Field(default_factory=list)
    style_examples: list[str] = Field(default_factory=list)


class Conversation(BaseModel):
    """A single query-response exchange with a persona."""

    id: str = Field(default_factory=_new_id)
    persona_id: str
    user_query: str
    response: str = ""
    tokens_used: int = 0
    cost: float = 0.0
    rating: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PersonaStats(BaseModel):
    """Aggregated statistics for a persona."""

    persona_id: str = ""
    total_conversations: int = 0
    avg_rating: float = 0.0
    total_revenue: float = 0.0
    creator_earnings: float = 0.0
