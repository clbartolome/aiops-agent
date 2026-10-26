"""External intake storage, deliberately independent of agent/procedure routing.

Execution invariant: source=incident ALWAYS requires explicit human
confirmation before the first procedure step, regardless of KB confirmation
metadata. source=user continues to use normal procedure confirmation rules.
Opening an incident must never invoke an agent, KB search, or procedure.
"""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, StringConstraints, field_validator

Severity = Literal["critical", "warning", "info"]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class IncidentRequest(BaseModel):
    title: NonEmpty
    message: NonEmpty
    severity: Severity

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, value):
        return value.strip().lower() if isinstance(value, str) else value


class IncidentSummary(BaseModel):
    id: str
    title: str
    severity: Severity
    status: Literal["NEW", "IN_PROGRESS", "DONE", "DISMISSED"]
    created_at: datetime


class Incident(IncidentSummary):
    message: str
    session_id: str
    procedure_run_id: str | None = None


class IncidentStore:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("INCIDENT_DB_PATH", "data/incidents.sqlite3"))

    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("""CREATE TABLE IF NOT EXISTS incidents (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, message TEXT NOT NULL,
            severity TEXT NOT NULL CHECK(severity IN ('critical','warning','info')),
            status TEXT NOT NULL CHECK(status IN ('NEW','IN_PROGRESS','DONE','DISMISSED')),
            created_at TEXT NOT NULL)""")
        # Backfill old inbox records with stable session identities.
        columns = {row[1] for row in connection.execute("PRAGMA table_info(incidents)")}
        if "session_id" not in columns:
            connection.execute("ALTER TABLE incidents ADD COLUMN session_id TEXT")
        if "procedure_run_id" not in columns:
            connection.execute("ALTER TABLE incidents ADD COLUMN procedure_run_id TEXT")
        connection.execute("UPDATE incidents SET session_id='incident-' || id WHERE session_id IS NULL")
        connection.commit()
        return connection

    def create(self, request: IncidentRequest) -> Incident:
        incident = Incident(**request.model_dump(), id=str(uuid4()), status="NEW",
                            created_at=datetime.now(timezone.utc), session_id=str(uuid4()))
        with self.connect() as connection:
            connection.execute("INSERT INTO incidents (id,title,message,severity,status,created_at,session_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (incident.id, incident.title, incident.message, incident.severity,
                                incident.status, incident.created_at.isoformat(), incident.session_id))
        connection.close()
        return incident

    def list(self) -> list[IncidentSummary]:
        with self.connect() as connection:
            rows = connection.execute("SELECT id,title,severity,status,created_at FROM incidents "
                                      "ORDER BY created_at DESC, rowid DESC").fetchall()
        connection.close()
        return [IncidentSummary(**dict(row)) for row in rows]

    def get(self, incident_id: str) -> Incident | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        connection.close()
        return Incident(**dict(row)) if row else None


    def delete(self, incident_id):
        with self.connect() as connection:
            connection.execute("DELETE FROM incidents WHERE id=?", (incident_id,))
        connection.close()


    def for_session(self, session_id):
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM incidents WHERE session_id=?", (session_id,)).fetchone()
        connection.close()
        return Incident(**dict(row)) if row else None

    def update_workflow(self, incident_id, status, run_id=None):
        with self.connect() as connection:
            connection.execute("UPDATE incidents SET status=?, procedure_run_id=COALESCE(?,procedure_run_id) WHERE id=?",
                               (status, run_id, incident_id))
        connection.close()
