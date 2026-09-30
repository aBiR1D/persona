"""Tests for Pydantic domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from persona.models import (
    CommunicationStyle,
    Conversation,
    KnowledgeEntry,
    PersonaProfile,
    PersonaStats,
    QAPair,
    TrainingData,
)


class TestPersonaProfile:
    def test_create_minimal(self):
        p = PersonaProfile(
            name="Dr. Alice",
            title="Cardiologist",
            expertise_areas=["cardiology"],
        )
        assert p.name == "Dr. Alice"
        assert p.title == "Cardiologist"
        assert len(p.id) == 12
        assert p.pricing_per_query == 5.0
        assert p.is_active is True

    def test_create_full(self):
        p = PersonaProfile(
            name="Bob",
            title="Senior Developer",
            expertise_areas=["python", "debugging"],
            personality_traits=["patient", "thorough"],
            communication_style=CommunicationStyle.SOCRATIC,
            knowledge_base_summary="10 yrs Python experience",
            pricing_per_query=10.0,
            creator_id="creator123",
        )
        assert p.communication_style == CommunicationStyle.SOCRATIC
        assert p.pricing_per_query == 10.0
        assert p.creator_id == "creator123"

    def test_name_required(self):
        with pytest.raises(ValidationError):
            PersonaProfile(name="", title="X", expertise_areas=["a"])

    def test_negative_pricing_rejected(self):
        with pytest.raises(ValidationError):
            PersonaProfile(
                name="X", title="Y", expertise_areas=["a"], pricing_per_query=-1
            )

    def test_unique_ids(self):
        a = PersonaProfile(name="A", title="T", expertise_areas=["x"])
        b = PersonaProfile(name="B", title="T", expertise_areas=["x"])
        assert a.id != b.id


class TestKnowledgeEntry:
    def test_defaults(self):
        e = KnowledgeEntry(topic="heart", content="The heart pumps blood.")
        assert e.source == "manual"
        assert e.category == "general"
        assert e.confidence == 1.0

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            KnowledgeEntry(topic="x", content="y", confidence=1.5)
        with pytest.raises(ValidationError):
            KnowledgeEntry(topic="x", content="y", confidence=-0.1)


class TestTrainingData:
    def test_empty(self):
        td = TrainingData()
        assert td.text_samples == []
        assert td.qa_pairs == []

    def test_with_qa(self):
        td = TrainingData(
            qa_pairs=[QAPair(question="What is X?", answer="X is Y.")]
        )
        assert len(td.qa_pairs) == 1
        assert td.qa_pairs[0].question == "What is X?"


class TestConversation:
    def test_basic(self):
        c = Conversation(persona_id="abc", user_query="hello")
        assert c.persona_id == "abc"
        assert c.response == ""
        assert c.rating is None

    def test_rating_bounds(self):
        c = Conversation(persona_id="a", user_query="q", rating=4.5)
        assert c.rating == 4.5
        with pytest.raises(ValidationError):
            Conversation(persona_id="a", user_query="q", rating=0.5)
        with pytest.raises(ValidationError):
            Conversation(persona_id="a", user_query="q", rating=6.0)


class TestPersonaStats:
    def test_defaults(self):
        s = PersonaStats()
        assert s.total_conversations == 0
        assert s.avg_rating == 0.0
        assert s.total_revenue == 0.0
        assert s.creator_earnings == 0.0

    def test_values(self):
        s = PersonaStats(
            persona_id="p1",
            total_conversations=100,
            avg_rating=4.8,
            total_revenue=500.0,
            creator_earnings=400.0,
        )
        assert s.total_conversations == 100
        assert s.creator_earnings == 400.0
