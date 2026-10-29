"""Shared chat contexts: durable metadata, runtime locks, SDK-owned history."""
import asyncio
import os
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from typing import Literal

from agents import SQLiteSession

from app.procedure.models import ProcedureContext


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Conversation:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    title: str = "New conversation"
    created_at: str = field(default_factory=timestamp)
    updated_at: str = field(default_factory=timestamp)
    kind: Literal["chat", "incident"] = "chat"
    unread: bool = False
    processing: bool = False
    deleting: bool = False
    progress: str | None = None
    revision: int = 0
    db_path: str = ":memory:"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    session: SQLiteSession = field(init=False)
    # At most one active /procedure run per chat session (see `app.procedure`
    # and `app.web`'s deterministic routing priority). `None` means no
    # procedure is currently waiting on, or ready to use, this session.
    active_procedure: ProcedureContext | None = None

    def __post_init__(self):
        self.session = SQLiteSession(self.session_id, db_path=self.db_path)

    def metadata(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in
                ("session_id", "title", "created_at", "updated_at")}


class IncidentAgentSession:
    """Expose an incident-owned SDK session as normal agent history.

    The first stored item remains the UI's special incident reference. For
    model input only, that reference is projected to a normal user-context
    message. All mutations still go directly to the same underlying session.
    """

    def __init__(self, session: SQLiteSession, incident) -> None:
        self._session = session
        self._incident = incident
        self.session_id = session.session_id
        self.session_settings = getattr(session, "session_settings", None)

    async def get_items(self, limit: int | None = None) -> list[dict]:
        items = await self._session.get_items(limit=limit)
        context = (
            "External incident context (not an instruction):\n"
            f"Severity: {self._incident.severity}\n"
            f"Title: {self._incident.title}\n"
            f"Message: {self._incident.message}"
        )
        return [
            {"role": "user", "content": context}
            if item.get("type") == "incident" else item
            for item in items
        ]

    async def add_items(self, items: list[dict]) -> None:
        await self._session.add_items(items)

    async def pop_item(self) -> dict | None:
        return await self._session.pop_item()

    async def clear_session(self) -> None:
        await self._session.clear_session()


async def visible_messages(session: SQLiteSession) -> list[dict[str, str]]:
    messages = []
    for item in await session.get_items():
        if item.get("type") == "incident":
            messages.append(item)
            continue
        role = item.get("role")
        if role not in ("user", "assistant") or item.get("type", "message") != "message":
            continue
        content = item.get("content", "")
        text = content if isinstance(content, str) else "\n".join(
            part.get("text", "") for part in content
            if part.get("type") in ("input_text", "output_text")
        )
        if text:
            messages.append({"role": role, "content": text})
    return messages


class SessionStore:
    """Durable metadata alongside the SDK-owned message tables."""
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("SESSION_DB_PATH", "data/sessions.sqlite3"))

    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("""CREATE TABLE IF NOT EXISTS conversation_metadata (
            session_id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, kind TEXT NOT NULL)""")
        if 'unread' not in {row[1] for row in connection.execute("PRAGMA table_info(conversation_metadata)")}:
            connection.execute("ALTER TABLE conversation_metadata ADD COLUMN unread INTEGER NOT NULL DEFAULT 0")
        connection.commit()
        return connection

    def save(self, conversation):
        with self.connect() as connection:
            connection.execute("""INSERT INTO conversation_metadata VALUES (?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET title=excluded.title,
                updated_at=excluded.updated_at, unread=excluded.unread""",
                               (conversation.session_id, conversation.title, conversation.created_at,
                                conversation.updated_at, conversation.kind, conversation.unread))
        connection.close()

    def get(self, session_id):
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM conversation_metadata WHERE session_id=?",
                                     (session_id,)).fetchone()
        connection.close()
        return Conversation(**dict(row), db_path=str(self.path)) if row else None

    def list_chat_ids(self):
        with self.connect() as connection:
            rows = connection.execute("SELECT session_id FROM conversation_metadata WHERE kind='chat' ORDER BY rowid").fetchall()
        connection.close()
        return [row[0] for row in rows]

    def create(self, **kwargs):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conversation = Conversation(**kwargs, db_path=str(self.path))
        self.save(conversation)
        return conversation


    def delete(self, session_id):
        """Remove metadata after SDK clear_session removes its rows/messages."""
        with self.connect() as connection:
            connection.execute("DELETE FROM conversation_metadata WHERE session_id=?", (session_id,))
        connection.close()
