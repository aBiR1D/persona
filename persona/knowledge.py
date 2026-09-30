"""SQLite-backed knowledge base for persona expertise storage and retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import aiosqlite

from persona.config import settings
from persona.models import KnowledgeEntry, QAPair


_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    id          TEXT PRIMARY KEY,
    persona_id  TEXT NOT NULL,
    topic       TEXT NOT NULL,
    content     TEXT NOT NULL,
    source      TEXT DEFAULT 'manual',
    category    TEXT DEFAULT 'general',
    confidence  REAL DEFAULT 1.0,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_persona ON knowledge(persona_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_topic   ON knowledge(topic);

CREATE TABLE IF NOT EXISTS personas (
    id          TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    persona_id  TEXT NOT NULL,
    user_query  TEXT NOT NULL,
    response    TEXT NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    cost        REAL DEFAULT 0.0,
    rating      REAL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_persona ON conversations(persona_id);
"""


class KnowledgeBase:
    """Async SQLite-backed knowledge store."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.db_path

    async def _connect(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        await db.executescript(_SCHEMA)
        return db

    # ------------------------------------------------------------------
    # Persona CRUD
    # ------------------------------------------------------------------

    async def save_persona(self, data: dict) -> None:
        db = await self._connect()
        try:
            await db.execute(
                "INSERT OR REPLACE INTO personas (id, data, created_at) VALUES (?, ?, ?)",
                (data["id"], json.dumps(data), data.get("created_at", "")),
            )
            await db.commit()
        finally:
            await db.close()

    async def get_persona(self, persona_id: str) -> Optional[dict]:
        db = await self._connect()
        try:
            cur = await db.execute("SELECT data FROM personas WHERE id = ?", (persona_id,))
            row = await cur.fetchone()
            return json.loads(row[0]) if row else None
        finally:
            await db.close()

    async def list_personas(self) -> list[dict]:
        db = await self._connect()
        try:
            cur = await db.execute("SELECT data FROM personas ORDER BY created_at DESC")
            rows = await cur.fetchall()
            return [json.loads(r[0]) for r in rows]
        finally:
            await db.close()

    async def find_persona_by_name(self, name: str) -> Optional[dict]:
        personas = await self.list_personas()
        name_lower = name.lower()
        for p in personas:
            if p.get("name", "").lower() == name_lower:
                return p
        return None

    # ------------------------------------------------------------------
    # Knowledge ingestion
    # ------------------------------------------------------------------

    async def ingest_text(self, persona_id: str, text: str, *, source: str = "text") -> list[KnowledgeEntry]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        entries: list[KnowledgeEntry] = []
        db = await self._connect()
        try:
            for para in paragraphs:
                topic = para[:80].replace("\n", " ")
                entry = KnowledgeEntry(
                    persona_id=persona_id,
                    topic=topic,
                    content=para,
                    source=source,
                    category="general",
                )
                await db.execute(
                    "INSERT INTO knowledge (id, persona_id, topic, content, source, category, confidence, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entry.id,
                        entry.persona_id,
                        entry.topic,
                        entry.content,
                        entry.source,
                        entry.category,
                        entry.confidence,
                        entry.created_at.isoformat(),
                    ),
                )
                entries.append(entry)
            await db.commit()
        finally:
            await db.close()
        return entries

    async def ingest_qa(
        self, persona_id: str, pairs: list[QAPair], *, source: str = "qa"
    ) -> list[KnowledgeEntry]:
        entries: list[KnowledgeEntry] = []
        db = await self._connect()
        try:
            for pair in pairs:
                entry = KnowledgeEntry(
                    persona_id=persona_id,
                    topic=pair.question[:80],
                    content=f"Q: {pair.question}\nA: {pair.answer}",
                    source=source,
                    category="qa",
                )
                await db.execute(
                    "INSERT INTO knowledge (id, persona_id, topic, content, source, category, confidence, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entry.id,
                        entry.persona_id,
                        entry.topic,
                        entry.content,
                        entry.source,
                        entry.category,
                        entry.confidence,
                        entry.created_at.isoformat(),
                    ),
                )
                entries.append(entry)
            await db.commit()
        finally:
            await db.close()
        return entries

    # ------------------------------------------------------------------
    # Search & context assembly
    # ------------------------------------------------------------------

    async def search(self, query: str, persona_id: Optional[str] = None, limit: int = 10) -> list[KnowledgeEntry]:
        db = await self._connect()
        try:
            terms = [f"%{t}%" for t in query.lower().split() if len(t) > 2]
            if not terms:
                terms = [f"%{query.lower()}%"]

            conditions = " OR ".join(
                ["LOWER(topic) LIKE ? OR LOWER(content) LIKE ?"] * len(terms)
            )
            params: list[str] = []
            for t in terms:
                params.extend([t, t])

            sql = f"SELECT * FROM knowledge WHERE ({conditions})"
            if persona_id:
                sql += " AND persona_id = ?"
                params.append(persona_id)
            sql += f" LIMIT {limit}"

            cur = await db.execute(sql, params)
            rows = await cur.fetchall()
            return [
                KnowledgeEntry(
                    id=r[0],
                    persona_id=r[1],
                    topic=r[2],
                    content=r[3],
                    source=r[4],
                    category=r[5],
                    confidence=r[6],
                )
                for r in rows
            ]
        finally:
            await db.close()

    async def build_context(self, query: str, persona_id: str, *, max_entries: int = 8) -> str:
        entries = await self.search(query, persona_id=persona_id, limit=max_entries)
        if not entries:
            return ""
        parts = ["Relevant knowledge from your expertise:"]
        for e in entries:
            parts.append(f"- [{e.category}] {e.content}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def save_conversation(self, conv: dict) -> None:
        db = await self._connect()
        try:
            await db.execute(
                "INSERT INTO conversations (id, persona_id, user_query, response, tokens_used, cost, rating, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    conv["id"],
                    conv["persona_id"],
                    conv["user_query"],
                    conv["response"],
                    conv.get("tokens_used", 0),
                    conv.get("cost", 0.0),
                    conv.get("rating"),
                    conv.get("created_at", ""),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    async def get_conversations(self, persona_id: str) -> list[dict]:
        db = await self._connect()
        try:
            cur = await db.execute(
                "SELECT id, persona_id, user_query, response, tokens_used, cost, rating, created_at "
                "FROM conversations WHERE persona_id = ? ORDER BY created_at DESC",
                (persona_id,),
            )
            rows = await cur.fetchall()
            return [
                {
                    "id": r[0],
                    "persona_id": r[1],
                    "user_query": r[2],
                    "response": r[3],
                    "tokens_used": r[4],
                    "cost": r[5],
                    "rating": r[6],
                    "created_at": r[7],
                }
                for r in rows
            ]
        finally:
            await db.close()

    async def get_all_knowledge(self, persona_id: str) -> list[KnowledgeEntry]:
        db = await self._connect()
        try:
            cur = await db.execute(
                "SELECT * FROM knowledge WHERE persona_id = ? ORDER BY created_at DESC",
                (persona_id,),
            )
            rows = await cur.fetchall()
            return [
                KnowledgeEntry(
                    id=r[0],
                    persona_id=r[1],
                    topic=r[2],
                    content=r[3],
                    source=r[4],
                    category=r[5],
                    confidence=r[6],
                )
                for r in rows
            ]
        finally:
            await db.close()
