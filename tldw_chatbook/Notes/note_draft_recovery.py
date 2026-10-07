"""Pending Database Note drafts in the existing device-private store."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tldw_chatbook.Library.library_notes_state import DatabaseNoteDraft
from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore


@dataclass(frozen=True)
class NoteDraftScope:
    """Captured persistence authority, independent of today's remote selection."""

    profile: str
    principal: str
    workspace: str | None

    def key(self) -> str:
        """Return a collision-free private scope key after validating identity."""
        if not self.profile or not self.principal:
            raise ValueError("Draft recovery requires a proven local owner.")
        return json.dumps((self.profile, self.principal, self.workspace))

    @classmethod
    def local(cls, database_path: str | Path, principal: str) -> NoteDraftScope:
        """Capture an opaque identity for a durable local database path."""
        path = Path(database_path)
        if str(path) == ":memory:" or not path.is_absolute():
            raise ValueError("Draft recovery requires a durable local database.")
        identity = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
        return cls(identity, principal, None)


class NoteDraftRecovery:
    """A narrow facade; connection, permissions and transactions stay store-owned."""

    def __init__(
        self,
        store: NotesDeviceStateStore,
        run: Callable[..., Awaitable[Any]] = asyncio.to_thread,
    ) -> None:
        self._store = store
        self._run = run

    async def save(
        self, scope: NoteDraftScope, draft: DatabaseNoteDraft, base_version: int
    ) -> None:
        """Durably retain one exact pending revision, never replacing a newer one."""
        await self._run(self._save, scope, draft, base_version)

    def _save(
        self, scope: NoteDraftScope, draft: DatabaseNoteDraft, base_version: int
    ) -> None:
        with self._store.transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO note_drafts VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, note_id) DO UPDATE SET
                    title=excluded.title, body=excluded.body,
                    keywords_text=excluded.keywords_text,
                    content_language=excluded.content_language,
                    revision=excluded.revision, base_version=excluded.base_version
                WHERE excluded.revision >= note_drafts.revision""",
                (
                    scope.key(),
                    draft.note_id,
                    draft.title,
                    draft.body,
                    draft.keywords_text,
                    draft.content_language,
                    draft.revision,
                    base_version,
                ),
            )

    async def load(
        self, scope: NoteDraftScope, note_id: str
    ) -> tuple[DatabaseNoteDraft, int] | None:
        """Read a pending draft and the version it originally edited."""
        return await self._run(self._load, scope, note_id)

    def _load(
        self, scope: NoteDraftScope, note_id: str
    ) -> tuple[DatabaseNoteDraft, int] | None:
        with self._store.transaction() as connection:
            row = connection.execute(
                "SELECT title, body, keywords_text, content_language, revision, base_version "
                "FROM note_drafts WHERE scope=? AND note_id=?",
                (scope.key(), note_id),
            ).fetchone()
        if row is None:
            return None
        return (
            DatabaseNoteDraft(note_id, row[0], row[1], row[2], row[4], row[3]),
            row[5],
        )

    async def acknowledge(
        self, scope: NoteDraftScope, note_id: str, saved_revision: int
    ) -> None:
        """Remove only the matching fully saved or explicitly discarded revision."""
        await self._run(self._acknowledge, scope, note_id, saved_revision)

    def _acknowledge(
        self, scope: NoteDraftScope, note_id: str, saved_revision: int
    ) -> None:
        with self._store.transaction(immediate=True) as connection:
            connection.execute(
                "DELETE FROM note_drafts WHERE scope=? AND note_id=? AND revision=?",
                (scope.key(), note_id, saved_revision),
            )
