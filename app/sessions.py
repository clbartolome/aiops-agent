"""Development conversations: metadata and locks, with SDK-owned history."""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

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
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    session: SQLiteSession = field(init=False)
    # At most one active /procedure run per chat session (see `app.procedure`
    # and `app.web`'s deterministic routing priority). `None` means no
    # procedure is currently waiting on, or ready to use, this session.
    active_procedure: ProcedureContext | None = None

    def __post_init__(self):
        self.session = SQLiteSession(self.session_id)

    def metadata(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in
                ("session_id", "title", "created_at", "updated_at")}


async def visible_messages(session: SQLiteSession) -> list[dict[str, str]]:
    messages = []
    for item in await session.get_items():
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
