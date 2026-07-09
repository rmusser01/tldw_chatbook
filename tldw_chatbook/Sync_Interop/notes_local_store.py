"""Local notes store interface for Sync v2 apply, plus an in-memory implementation.

The protocol is the seam the apply adapter writes through. P2 ships an in-memory
implementation for round-trip verification; a ChaChaNotes-backed implementation is a
later integration step.
"""
from __future__ import annotations

from typing import Any, Protocol


class NotesSyncLocalStore(Protocol):
    def upsert_note(self, object_id: str, payload: dict[str, Any], *, object_revision: int) -> None: ...
    def soft_delete_note(self, object_id: str, *, object_revision: int) -> None: ...
    def get(self, object_id: str) -> dict[str, Any] | None: ...


class InMemoryNotesStore:
    """Minimal in-memory notes store implementing NotesSyncLocalStore."""

    def __init__(self) -> None:
        self._notes: dict[str, dict[str, Any]] = {}
        self.upsert_calls = 0
        self.delete_calls = 0

    def upsert_note(self, object_id: str, payload: dict[str, Any], *, object_revision: int) -> None:
        self.upsert_calls += 1
        record = {"title": payload.get("title", ""), "content": payload.get("content", payload.get("body", "")), "deleted": False}
        self._notes[object_id] = record

    def soft_delete_note(self, object_id: str, *, object_revision: int) -> None:
        self.delete_calls += 1
        existing = self._notes.get(object_id, {"title": "", "content": ""})
        existing["deleted"] = True
        self._notes[object_id] = existing

    def get(self, object_id: str) -> dict[str, Any] | None:
        return self._notes.get(object_id)
