"""Marketplace logic — browsing, searching, and earnings calculations."""

from __future__ import annotations

from persona.config import settings
from persona.knowledge import KnowledgeBase
from persona.models import PersonaProfile, PersonaStats


class Marketplace:
    """Public marketplace operations."""

    def __init__(self, kb: KnowledgeBase | None = None) -> None:
        self.kb = kb or KnowledgeBase()

    async def list_personas(self) -> list[PersonaProfile]:
        rows = await self.kb.list_personas()
        return [PersonaProfile(**r) for r in rows]

    async def search_personas(
        self, query: str = "", category: str = ""
    ) -> list[PersonaProfile]:
        all_personas = await self.list_personas()
        results: list[PersonaProfile] = []
        q_lower = query.lower()

        for p in all_personas:
            if not p.is_active:
                continue
            if query:
                searchable = " ".join(
                    [p.name, p.title, " ".join(p.expertise_areas)]
                ).lower()
                if q_lower not in searchable:
                    continue
            if category:
                if category.lower() not in [a.lower() for a in p.expertise_areas]:
                    continue
            results.append(p)
        return results

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

    async def calculate_earnings(self, persona_id: str) -> dict:
        stats = await self.get_stats(persona_id)
        return {
            "persona_id": persona_id,
            "total_revenue": stats.total_revenue,
            "platform_commission": round(stats.total_revenue * settings.platform_commission, 2),
            "creator_earnings": stats.creator_earnings,
            "commission_rate": f"{settings.platform_commission * 100:.0f}%",
        }
