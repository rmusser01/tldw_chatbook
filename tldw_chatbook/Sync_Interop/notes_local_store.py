"""Local notes store interface for Sync v2 apply, plus an in-memory implementation.

The protocol is the seam the apply adapter writes through. P2 ships an in-memory
implementation for round-trip verification; a ChaChaNotes-backed implementation is a
later integration step.
"""

from __future__ import annotations

from typing import Any, Protocol


class NotesMaterializationUnsupported(ValueError):
    """The materializer cannot retain supplied metadata; no mutation occurred."""


class NotesSyncLocalStore(Protocol):
    """Retain every accepted payload field, or reject before any mutation.

    Stores lacking backlink storage must raise NotesMaterializationUnsupported
    for non-null backlinks before changing canonical state. Explicit null needs
    no additional metadata storage when no backlinks can exist locally.
    """

    def upsert_note(
        self, object_id: str, payload: dict[str, Any], *, object_revision: int
    ) -> None: ...
    def soft_delete_note(self, object_id: str, *, object_revision: int) -> None: ...
    def get(self, object_id: str) -> dict[str, Any] | None: ...


class InMemoryNotesStore:
    """Minimal in-memory notes store implementing NotesSyncLocalStore."""

    def __init__(self) -> None:
        self._notes: dict[str, dict[str, Any]] = {}
        self.upsert_calls = 0
        self.delete_calls = 0

    def upsert_note(
        self, object_id: str, payload: dict[str, Any], *, object_revision: int
    ) -> None:
        self.upsert_calls += 1
        record = {
            "title": payload.get("title", ""),
            "content": payload.get("content", payload.get("body", "")),
            "deleted": False,
        }
        existing = self._notes.get(object_id, {})
        if "content_language" in payload:
            record["content_language"] = payload["content_language"]
        elif "content_language" in existing:
            record["content_language"] = existing["content_language"]
        for backlink in ("conversation_id", "message_id"):
            if backlink in payload:
                record[backlink] = payload[backlink]
            elif backlink in existing:
                record[backlink] = existing[backlink]
        self._notes[object_id] = record

    def soft_delete_note(self, object_id: str, *, object_revision: int) -> None:
        self.delete_calls += 1
        existing = self._notes.get(object_id, {"title": "", "content": ""})
        existing["deleted"] = True
        self._notes[object_id] = existing

    def get(self, object_id: str) -> dict[str, Any] | None:
        return self._notes.get(object_id)
