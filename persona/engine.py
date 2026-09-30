"""Core persona engine — creation, chat, training, and quality evaluation."""

from __future__ import annotations

from persona.ai import LLMClient, LLMResponse
from persona.config import settings
from persona.knowledge import KnowledgeBase
from persona.models import (
    Conversation,
    PersonaProfile,
    PersonaStats,
    TrainingData,
)


def _build_system_prompt(profile: PersonaProfile, context: str = "") -> str:
    traits = ", ".join(profile.personality_traits) if profile.personality_traits else "knowledgeable"
    areas = ", ".join(profile.expertise_areas) if profile.expertise_areas else "various topics"

    prompt = (
        f"You are {profile.name}, {profile.title}.\n\n"
        f"Your expertise spans: {areas}.\n"
        f"Your personality traits: {traits}.\n"
        f"Communication style: {profile.communication_style.value}.\n\n"
        "Instructions:\n"
        "- Always stay in character.\n"
        "- Draw on your knowledge base to give accurate, helpful answers.\n"
        "- If something is outside your expertise, say so honestly.\n"
        "- Match the communication style described above.\n"
    )
    if profile.knowledge_base_summary:
        prompt += f"\nKnowledge summary: {profile.knowledge_base_summary}\n"
    if context:
        prompt += f"\n{context}\n"
    return prompt


class PersonaEngine:
    """High-level operations on personas."""

    def __init__(self, kb: KnowledgeBase | None = None, llm: LLMClient | None = None) -> None:
        self.kb = kb or KnowledgeBase()
        self.llm = llm or LLMClient()

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_persona(
        self, profile: PersonaProfile, training_data: TrainingData | None = None
    ) -> PersonaProfile:
        await self.kb.save_persona(profile.model_dump(mode="json"))

        if training_data:
            await self.train(profile.id, training_data)

        return profile

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat(self, persona_id: str, user_query: str) -> Conversation:
        persona_data = await self.kb.get_persona(persona_id)
        if not persona_data:
            raise ValueError(f"Persona '{persona_id}' not found")

        profile = PersonaProfile(**persona_data)
        context = await self.kb.build_context(user_query, persona_id)
        system_prompt = _build_system_prompt(profile, context)

        llm_resp: LLMResponse = await self.llm.generate(system_prompt, user_query)

        cost = profile.pricing_per_query
        conv = Conversation(
            persona_id=persona_id,
            user_query=user_query,
            response=llm_resp.text,
            tokens_used=llm_resp.tokens_used,
            cost=cost,
        )
        await self.kb.save_conversation(conv.model_dump(mode="json"))
        return conv

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    async def train(self, persona_id: str, data: TrainingData) -> int:
        count = 0
        if data.text_samples:
            for text in data.text_samples:
                entries = await self.kb.ingest_text(persona_id, text, source="training")
                count += len(entries)
        if data.qa_pairs:
            entries = await self.kb.ingest_qa(persona_id, data.qa_pairs, source="training")
            count += len(entries)
        if data.expertise_docs:
            for doc in data.expertise_docs:
                entries = await self.kb.ingest_text(persona_id, doc, source="expertise_doc")
                count += len(entries)
        if data.style_examples:
            for ex in data.style_examples:
                entries = await self.kb.ingest_text(persona_id, ex, source="style_example")
                count += len(entries)
        return count

    # ------------------------------------------------------------------
    # Evaluate quality
    # ------------------------------------------------------------------

    async def evaluate_quality(self, persona_id: str) -> float:
        conversations = await self.kb.get_conversations(persona_id)
        if not conversations:
            return 0.0

        rated = [c for c in conversations if c.get("rating") is not None]
        if not rated:
            return 0.5

        avg_rating = sum(c["rating"] for c in rated) / len(rated)
        knowledge = await self.kb.get_all_knowledge(persona_id)
        knowledge_factor = min(len(knowledge) / 20.0, 1.0)

        return round((avg_rating / 5.0) * 0.7 + knowledge_factor * 0.3, 3)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self, persona_id: str) -> PersonaStats:
        conversations = await self.kb.get_conversations(persona_id)
        total = len(conversations)
        rated = [c for c in conversations if c.get("rating") is not None]
        avg_rating = (sum(c["rating"] for c in rated) / len(rated)) if rated else 0.0
        total_revenue = sum(c.get("cost", 0.0) for c in conversations)
        creator_earnings = total_revenue * (1 - settings.platform_commission)

        return PersonaStats(
            persona_id=persona_id,
            total_conversations=total,
            avg_rating=round(avg_rating, 2),
            total_revenue=round(total_revenue, 2),
            creator_earnings=round(creator_earnings, 2),
        )
