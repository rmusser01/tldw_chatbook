"""Host-independent Database Note session and serialized-save coordinator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Callable, Protocol

from tldw_chatbook.Library.library_notes_state import (
    DatabaseNoteDraft,
    DatabaseNoteSavePayload,
    LibraryNoteSessionSnapshot,
    NormalizedDatabaseNote,
    NoteValidationVeto,
    validate_database_note_draft,
)


class DatabaseNoteSessionPort(Protocol):
    """Minimum asynchronous persistence boundary required by a session."""

    async def load_note(self, note_id: str) -> DatabaseNotePortLoadReply:
        """Load one complete normalized note detail."""
        ...

    async def save_note(
        self,
        note_id: str,
        expected_version: int,
        payload: DatabaseNoteSavePayload,
    ) -> DatabaseNotePortSaveReply:
        """Persist one exact payload against an optimistic-lock version."""
        ...


class PortLoadKind(StrEnum):
    """Normalized port-level detail-fetch result kinds."""

    LOADED = "loaded"
    MISSING = "missing"
    FAILED = "failed"


class PortSaveKind(StrEnum):
    """Normalized port-level versioned-save result kinds."""

    SAVED = "saved"
    CONFLICT = "conflict"
    FAILED = "failed"


class NoteLoadOutcomeKind(StrEnum):
    """Public session-opening result kinds."""

    LOADED = "loaded"
    MISSING = "missing"
    FAILED = "failed"
    STALE = "stale"


class NoteSaveOutcomeKind(StrEnum):
    """Public save-request result kinds."""

    SAVED = "saved"
    ACKNOWLEDGED = "acknowledged"
    VALIDATION_VETO = "validation_veto"
    FAILED = "failed"
    CONFLICTED = "conflicted"
    BLOCKED = "blocked"
    STALE = "stale"


@dataclass(frozen=True)
class DatabaseNotePortLoadReply:
    """Normalized result from the injected detail-fetch port."""

    kind: PortLoadKind
    detail: NormalizedDatabaseNote | None = None
    message: str = ""

    def __post_init__(self) -> None:
        """Reject ambiguous loaded/missing/failed reply shapes."""
        if self.kind is PortLoadKind.LOADED and self.detail is None:
            raise ValueError("A loaded Database Note reply requires detail.")
        if self.kind is not PortLoadKind.LOADED and self.detail is not None:
            raise ValueError("A non-loaded Database Note reply cannot carry detail.")

    @classmethod
    def loaded(cls, detail: NormalizedDatabaseNote) -> DatabaseNotePortLoadReply:
        """Build a coherent loaded-detail reply."""
        return cls(kind=PortLoadKind.LOADED, detail=detail)

    @classmethod
    def missing(
        cls, message: str = "Note no longer exists."
    ) -> DatabaseNotePortLoadReply:
        """Build a genuinely missing-note reply."""
        return cls(kind=PortLoadKind.MISSING, message=message)

    @classmethod
    def failed(cls, message: str = "Unable to load note.") -> DatabaseNotePortLoadReply:
        """Build an ordinary load-failure reply."""
        return cls(kind=PortLoadKind.FAILED, message=message)


@dataclass(frozen=True)
class DatabaseNotePortSaveReply:
    """Normalized result from the injected versioned-save port."""

    kind: PortSaveKind
    version: int | None = None
    modified_at: str = ""
    keywords: tuple[str, ...] | None = None
    message: str = ""

    @classmethod
    def saved(
        cls,
        *,
        version: int,
        modified_at: str = "",
        keywords: tuple[str, ...] | None = None,
    ) -> DatabaseNotePortSaveReply:
        """Build a successful save reply."""
        return cls(
            kind=PortSaveKind.SAVED,
            version=version,
            modified_at=modified_at,
            keywords=keywords,
        )

    @classmethod
    def conflict(
        cls, message: str = "Note changed elsewhere."
    ) -> DatabaseNotePortSaveReply:
        """Build an optimistic-lock conflict reply."""
        return cls(kind=PortSaveKind.CONFLICT, message=message)

    @classmethod
    def failed(
        cls, message: str = "Save failed — edits kept. Press Save to retry."
    ) -> DatabaseNotePortSaveReply:
        """Build an ordinary save-failure reply."""
        return cls(kind=PortSaveKind.FAILED, message=message)


@dataclass(frozen=True)
class NoteLoadOutcome:
    """Typed public result of opening a Database Note session."""

    kind: NoteLoadOutcomeKind
    note_id: str
    message: str = ""


@dataclass(frozen=True)
class NoteSaveOutcome:
    """Typed public result of a serialized save request."""

    kind: NoteSaveOutcomeKind
    note_id: str = ""
    revision: int | None = None
    version: int | None = None
    message: str = ""
    veto: NoteValidationVeto | None = None


class DatabaseNoteSessionCoordinator:
    """Own one canonical draft and serialize all saves admitted for it."""

    def __init__(
        self,
        port: DatabaseNoteSessionPort,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the coordinator with its minimal async port.

        Args:
            port: Complete-detail load and optimistic-version save boundary.
            clock: Timestamp source used for saved status and baseline metadata.
        """
        self._port = port
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._snapshot: LibraryNoteSessionSnapshot | None = None
        self._session_request_token = 0
        self._session_generation = 0
        self._save_task: asyncio.Task[NoteSaveOutcome] | None = None
        self._pending_save_requested = False
        self._untouched_create_token: str | None = None
        self._destructive: object | None = None

    @property
    def snapshot(self) -> LibraryNoteSessionSnapshot | None:
        """Return the current immutable session view, if one is active."""
        return self._snapshot

    @property
    def untouched_create_token(self) -> str | None:
        """Return the active untouched-create token, if still eligible."""
        return self._untouched_create_token

    def invalidate_session_request(self) -> None:
        """Make every currently pending detail-load completion stale."""
        self._session_request_token += 1

    async def open_session(
        self,
        note_id: str,
        *,
        untouched_create_token: str | None = None,
    ) -> NoteLoadOutcome:
        """Load and start a session only if this request remains current.

        Args:
            note_id: Database Note identity to load.
            untouched_create_token: Matching creation token for discard
                eligibility until the first edit or explicit Save.

        Returns:
            A typed loaded, missing, failed, or stale outcome.
        """
        self._session_request_token += 1
        request_token = self._session_request_token
        try:
            reply = await self._port.load_note(note_id)
        except Exception as error:
            if request_token != self._session_request_token:
                return NoteLoadOutcome(NoteLoadOutcomeKind.STALE, note_id)
            return NoteLoadOutcome(
                NoteLoadOutcomeKind.FAILED,
                note_id,
                str(error) or "Unable to load note.",
            )

        if request_token != self._session_request_token:
            return NoteLoadOutcome(NoteLoadOutcomeKind.STALE, note_id)
        if reply.kind is PortLoadKind.MISSING:
            return NoteLoadOutcome(
                NoteLoadOutcomeKind.MISSING,
                note_id,
                reply.message or "Note no longer exists.",
            )
        if reply.kind is PortLoadKind.FAILED:
            return NoteLoadOutcome(
                NoteLoadOutcomeKind.FAILED,
                note_id,
                reply.message or "Unable to load note.",
            )
        if reply.detail is None:
            return NoteLoadOutcome(
                NoteLoadOutcomeKind.FAILED,
                note_id,
                "Note detail was incomplete.",
            )
        if reply.detail.note_id != note_id:
            return NoteLoadOutcome(
                NoteLoadOutcomeKind.FAILED,
                note_id,
                "Loaded note identity did not match the request.",
            )

        self._start_loaded_session(
            reply.detail,
            untouched_create_token=untouched_create_token,
        )
        return NoteLoadOutcome(NoteLoadOutcomeKind.LOADED, note_id)

    def _start_loaded_session(
        self,
        detail: NormalizedDatabaseNote,
        *,
        untouched_create_token: str | None,
    ) -> None:
        """Replace the active session from one coherent normalized detail."""
        self._session_generation += 1
        draft = DatabaseNoteDraft(
            note_id=detail.note_id,
            title=detail.title,
            body=detail.body,
            keywords_text=", ".join(detail.keywords),
            revision=0,
        )
        self._snapshot = LibraryNoteSessionSnapshot(
            baseline=detail,
            draft=draft,
            session_generation=self._session_generation,
            saved_revision=0,
            dirty=False,
            saving=False,
            in_conflict=False,
            conflict_generation=0,
            status_message="",
        )
        self._save_task = None
        self._pending_save_requested = False
        self._untouched_create_token = untouched_create_token
        self._destructive = None

    def mutate(
        self,
        *,
        title: str | None = None,
        body: str | None = None,
        keywords_text: str | None = None,
    ) -> bool:
        """Apply one genuine value change to the canonical raw draft.

        Args:
            title: Replacement raw title, or None to leave unchanged.
            body: Replacement raw body, or None to leave unchanged.
            keywords_text: Replacement raw keyword input, or None to leave
                unchanged.

        Returns:
            True when at least one supplied value genuinely changed.
        """
        snapshot = self._snapshot
        if snapshot is None or self._destructive is not None:
            return False

        draft = snapshot.draft
        next_title = draft.title if title is None else title
        next_body = draft.body if body is None else body
        next_keywords = draft.keywords_text if keywords_text is None else keywords_text
        if (
            next_title == draft.title
            and next_body == draft.body
            and next_keywords == draft.keywords_text
        ):
            return False

        next_draft = DatabaseNoteDraft(
            note_id=draft.note_id,
            title=next_title,
            body=next_body,
            keywords_text=next_keywords,
            revision=draft.revision + 1,
        )
        self._snapshot = replace(
            snapshot,
            draft=next_draft,
            dirty=True,
            status_message=(
                snapshot.status_message
                if snapshot.saving or snapshot.in_conflict
                else "Unsaved changes"
            ),
        )
        self._untouched_create_token = None
        if snapshot.saving:
            self._pending_save_requested = True
        return True

    async def request_save(self, *, explicit: bool) -> NoteSaveOutcome:
        """Request a serialized save and await the complete coalesced chain.

        Args:
            explicit: Whether the user explicitly acknowledged Save.

        Returns:
            A typed saved, acknowledged, vetoed, failed, conflicted, blocked,
            or stale outcome.
        """
        snapshot = self._snapshot
        if snapshot is None:
            return NoteSaveOutcome(
                NoteSaveOutcomeKind.BLOCKED,
                message="No note is open.",
            )
        if self._destructive is not None:
            return NoteSaveOutcome(
                NoteSaveOutcomeKind.BLOCKED,
                note_id=snapshot.note_id,
                revision=snapshot.draft_revision,
                message="A destructive action is in progress.",
            )
        if snapshot.in_conflict:
            return NoteSaveOutcome(
                NoteSaveOutcomeKind.CONFLICTED,
                note_id=snapshot.note_id,
                revision=snapshot.draft_revision,
                version=snapshot.version,
                message="Conflict — review the choices below.",
            )
        if not snapshot.dirty:
            if explicit:
                self._untouched_create_token = None
            return NoteSaveOutcome(
                NoteSaveOutcomeKind.ACKNOWLEDGED,
                note_id=snapshot.note_id,
                revision=snapshot.draft_revision,
                version=snapshot.version,
                message="Saved — no changes.",
            )

        self._pending_save_requested = True
        task = self._save_task
        if task is None or task.done():
            task = asyncio.create_task(self._drive_saves())
            self._save_task = task
        return await asyncio.shield(task)

    async def _drive_saves(self) -> NoteSaveOutcome:
        """Persist successive latest revisions with one active port call."""
        while True:
            snapshot = self._snapshot
            if snapshot is None:
                return NoteSaveOutcome(
                    NoteSaveOutcomeKind.STALE,
                    message="The note session changed before save completed.",
                )
            if snapshot.in_conflict:
                self._pending_save_requested = False
                return NoteSaveOutcome(
                    NoteSaveOutcomeKind.CONFLICTED,
                    note_id=snapshot.note_id,
                    revision=snapshot.draft_revision,
                    version=snapshot.version,
                    message="Conflict — review the choices below.",
                )
            if not snapshot.dirty:
                self._pending_save_requested = False
                return NoteSaveOutcome(
                    NoteSaveOutcomeKind.ACKNOWLEDGED,
                    note_id=snapshot.note_id,
                    revision=snapshot.draft_revision,
                    version=snapshot.version,
                    message="Saved — no changes.",
                )

            note_id = snapshot.note_id
            session_generation = snapshot.session_generation
            expected_version = snapshot.version
            draft_revision = snapshot.draft_revision
            payload = validate_database_note_draft(snapshot.draft)
            self._pending_save_requested = False
            if isinstance(payload, NoteValidationVeto):
                self._snapshot = replace(
                    snapshot,
                    saving=False,
                    dirty=True,
                    status_message=payload.message,
                )
                return NoteSaveOutcome(
                    NoteSaveOutcomeKind.VALIDATION_VETO,
                    note_id=note_id,
                    revision=draft_revision,
                    version=expected_version,
                    message=payload.message,
                    veto=payload,
                )

            self._snapshot = replace(
                snapshot,
                saving=True,
                status_message="Saving…",
            )
            try:
                reply = await self._port.save_note(
                    note_id,
                    expected_version,
                    payload,
                )
            except Exception as error:
                reply = DatabaseNotePortSaveReply.failed(
                    str(error) or "Save failed — edits kept. Press Save to retry."
                )

            current = self._snapshot
            if (
                current is None
                or current.note_id != note_id
                or current.session_generation != session_generation
            ):
                return NoteSaveOutcome(
                    NoteSaveOutcomeKind.STALE,
                    note_id=note_id,
                    revision=draft_revision,
                    version=expected_version,
                    message="The note session changed before save completed.",
                )

            if reply.kind is PortSaveKind.FAILED:
                self._pending_save_requested = False
                message = self._actionable_save_failure(reply.message)
                self._snapshot = replace(
                    current,
                    saving=False,
                    dirty=True,
                    status_message=message,
                )
                return NoteSaveOutcome(
                    NoteSaveOutcomeKind.FAILED,
                    note_id=note_id,
                    revision=current.draft_revision,
                    version=current.version,
                    message=message,
                )

            if reply.kind is PortSaveKind.CONFLICT:
                self._pending_save_requested = False
                message = "Conflict — review the choices below."
                self._snapshot = replace(
                    current,
                    saving=False,
                    dirty=True,
                    in_conflict=True,
                    conflict_generation=current.conflict_generation + 1,
                    status_message=message,
                )
                return NoteSaveOutcome(
                    NoteSaveOutcomeKind.CONFLICTED,
                    note_id=note_id,
                    revision=current.draft_revision,
                    version=current.version,
                    message=message,
                )

            saved_at = self._clock()
            saved_version = (
                reply.version if reply.version is not None else expected_version + 1
            )
            baseline = NormalizedDatabaseNote(
                note_id=note_id,
                title=payload.title,
                body=payload.body,
                keywords=(
                    reply.keywords if reply.keywords is not None else payload.keywords
                ),
                version=saved_version,
                created_at=current.baseline.created_at,
                modified_at=reply.modified_at or saved_at.isoformat(),
            )
            has_newer_draft = current.draft_revision > draft_revision
            self._snapshot = replace(
                current,
                baseline=baseline,
                saved_revision=draft_revision,
                dirty=has_newer_draft,
                saving=False,
                in_conflict=False,
                status_message=(
                    "Unsaved changes"
                    if has_newer_draft
                    else f"Saved {saved_at.strftime('%H:%M')}"
                ),
            )
            if has_newer_draft:
                self._pending_save_requested = True
                continue

            self._pending_save_requested = False
            return NoteSaveOutcome(
                NoteSaveOutcomeKind.SAVED,
                note_id=note_id,
                revision=draft_revision,
                version=saved_version,
                message=self._snapshot.status_message,
            )

    @staticmethod
    def _actionable_save_failure(detail: str) -> str:
        """Keep every ordinary save failure explicit about safe recovery."""
        message = detail.strip()
        if "Press Save to retry" in message:
            return message
        if message:
            return (
                f"Save failed — {message.rstrip('.')}. Edits kept. Press Save to retry."
            )
        return "Save failed — edits kept. Press Save to retry."
