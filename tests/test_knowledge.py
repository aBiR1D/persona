"""Tests for the SQLite-backed knowledge base."""

from __future__ import annotations

import os
import tempfile

import pytest

from persona.knowledge import KnowledgeBase
from persona.models import PersonaProfile, QAPair


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def kb(tmp_db):
    return KnowledgeBase(db_path=tmp_db)


@pytest.fixture
def sample_profile():
    return PersonaProfile(
        name="Dr. Test",
        title="Testing Expert",
        expertise_areas=["testing", "quality"],
    )


class TestPersonaCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(self, kb, sample_profile):
        data = sample_profile.model_dump(mode="json")
        await kb.save_persona(data)
        result = await kb.get_persona(sample_profile.id)
        assert result is not None
        assert result["name"] == "Dr. Test"

    @pytest.mark.asyncio
    async def test_get_missing(self, kb):
        result = await kb.get_persona("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_personas(self, kb, sample_profile):
        data = sample_profile.model_dump(mode="json")
        await kb.save_persona(data)
        results = await kb.list_personas()
        assert len(results) == 1
        assert results[0]["name"] == "Dr. Test"

    @pytest.mark.asyncio
    async def test_find_by_name(self, kb, sample_profile):
        data = sample_profile.model_dump(mode="json")
        await kb.save_persona(data)
        found = await kb.find_persona_by_name("Dr. Test")
        assert found is not None
        assert found["id"] == sample_profile.id

    @pytest.mark.asyncio
    async def test_find_by_name_case_insensitive(self, kb, sample_profile):
        data = sample_profile.model_dump(mode="json")
        await kb.save_persona(data)
        found = await kb.find_persona_by_name("dr. test")
        assert found is not None


class TestKnowledgeIngestion:
    @pytest.mark.asyncio
    async def test_ingest_text(self, kb, sample_profile):
        text = "First paragraph about testing.\n\nSecond paragraph about quality."
        entries = await kb.ingest_text(sample_profile.id, text)
        assert len(entries) == 2
        assert entries[0].persona_id == sample_profile.id

    @pytest.mark.asyncio
    async def test_ingest_qa(self, kb, sample_profile):
        pairs = [
            QAPair(question="What is unit testing?", answer="Testing individual units of code."),
            QAPair(question="What is TDD?", answer="Test-driven development."),
        ]
        entries = await kb.ingest_qa(sample_profile.id, pairs)
        assert len(entries) == 2
        assert "Q: What is unit testing?" in entries[0].content

    @pytest.mark.asyncio
    async def test_search(self, kb, sample_profile):
        await kb.ingest_text(sample_profile.id, "Python is a programming language.")
        results = await kb.search("python", persona_id=sample_profile.id)
        assert len(results) >= 1
        assert "Python" in results[0].content

    @pytest.mark.asyncio
    async def test_build_context(self, kb, sample_profile):
        await kb.ingest_text(sample_profile.id, "Debugging requires patience and methodical thinking.")
        ctx = await kb.build_context("debugging", sample_profile.id)
        assert "Relevant knowledge" in ctx
        assert "Debugging" in ctx

    @pytest.mark.asyncio
    async def test_build_context_empty(self, kb, sample_profile):
        ctx = await kb.build_context("nonexistent topic xyz", sample_profile.id)
        assert ctx == ""


class TestConversations:
    @pytest.mark.asyncio
    async def test_save_and_get(self, kb, sample_profile):
        conv = {
            "id": "conv1",
            "persona_id": sample_profile.id,
            "user_query": "hello",
            "response": "Hi there!",
            "tokens_used": 50,
            "cost": 5.0,
            "rating": 4.0,
            "created_at": "2026-01-01T00:00:00",
        }
        await kb.save_conversation(conv)
        convs = await kb.get_conversations(sample_profile.id)
        assert len(convs) == 1
        assert convs[0]["user_query"] == "hello"
        assert convs[0]["rating"] == 4.0

    @pytest.mark.asyncio
    async def test_get_all_knowledge(self, kb, sample_profile):
        await kb.ingest_text(sample_profile.id, "Some knowledge content here.")
        entries = await kb.get_all_knowledge(sample_profile.id)
        assert len(entries) == 1
