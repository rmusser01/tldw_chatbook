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


class ConflictAction(StrEnum):
    """User-selected optimistic-conflict recovery actions."""

    OVERWRITE = "overwrite"
    RELOAD = "reload"


class ConflictOutcomeKind(StrEnum):
    """Typed conflict-resolution result kinds."""

    OVERWRITTEN = "overwritten"
    RELOADED = "reloaded"
    DRAFT_CHANGED = "draft_changed"
    ALREADY_RUNNING = "already_running"
    RENEWED_CONFLICT = "renewed_conflict"
    VALIDATION_VETO = "validation_veto"
    MISSING = "missing"
    FAILED = "failed"
    STALE = "stale"
    NOT_IN_CONFLICT = "not_in_conflict"


class NoteFlushOutcomeKind(StrEnum):
    """Typed navigation-barrier result kinds."""

    PERMITTED = "permitted"
    VALIDATION_VETO = "validation_veto"
    FAILED = "failed"
    CONFLICTED = "conflicted"
    BLOCKED = "blocked"
    STALE = "stale"


class DestructiveKind(StrEnum):
    """Coordinator-gated destructive Database Note actions."""

    DISCARD_NEW_NOTE = "discard_new_note"
    DELETE = "delete"


class DestructiveAdmissionOutcomeKind(StrEnum):
    """Typed destructive-admission result kinds."""

    ADMITTED = "admitted"
    FLUSH_VETOED = "flush_vetoed"
    ALREADY_RUNNING = "already_running"
    NOT_ELIGIBLE = "not_eligible"
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


@dataclass(frozen=True)
class ConflictOutcome:
    """Typed public result of one gated conflict action."""

    kind: ConflictOutcomeKind
    action: ConflictAction
    note_id: str = ""
    message: str = ""
    save_outcome: NoteSaveOutcome | None = None


@dataclass(frozen=True)
class NoteFlushOutcome:
    """Typed public navigation-barrier result."""

    kind: NoteFlushOutcomeKind
    message: str = ""
    save_outcome: NoteSaveOutcome | None = None


@dataclass(frozen=True)
class DestructiveAdmission:
    """Immutable authority token revalidated immediately before deletion."""

    kind: DestructiveKind
    note_id: str
    session_generation: int
    expected_version: int
    operation_token: int
    create_token: str | None = None


@dataclass(frozen=True)
class DestructiveAdmissionOutcome:
    """Typed public result of requesting destructive admission."""

    kind: DestructiveAdmissionOutcomeKind
    message: str = ""
    admission: DestructiveAdmission | None = None
    flush_outcome: NoteFlushOutcome | None = None


@dataclass(frozen=True)
class _ConflictOperation:
    action: ConflictAction
    note_id: str
    session_generation: int
    conflict_generation: int
    draft_revision: int
    operation_token: int


@dataclass(frozen=True)
class _DestructiveState:
    admission: DestructiveAdmission
    running: bool = False


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
        self._conflict_operation_counter = 0
        self._active_conflict_operation: _ConflictOperation | None = None
        self._destructive_operation_counter = 0
        self._destructive: _DestructiveState | None = None

    @property
    def snapshot(self) -> LibraryNoteSessionSnapshot | None:
        """Return the current immutable session view, if one is active."""
        return self._snapshot

    @property
    def untouched_create_token(self) -> str | None:
        """Return the active untouched-create token, if still eligible."""
        return self._untouched_create_token

    @property
    def conflict_resolution_running(self) -> bool:
        """Return whether one conflict action currently owns the gate."""
        return self._active_conflict_operation is not None

    @property
    def destructive_admission(self) -> DestructiveAdmission | None:
        """Return the active destructive authority token, if admitted."""
        return self._destructive.admission if self._destructive is not None else None

    @property
    def destructive_running(self) -> bool:
        """Return whether the admitted destructive service call has begun."""
        return self._destructive is not None and self._destructive.running

    def invalidate_session_request(self) -> None:
        """Make every currently pending detail-load completion stale."""
        self._session_request_token += 1

    def close_session(self) -> None:
        """End the active session and invalidate every pending completion."""
        self._close_session()

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
        self._active_conflict_operation = None
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

    async def resolve_conflict(self, action: ConflictAction) -> ConflictOutcome:
        """Run one token-gated Reload or revision-safe Overwrite operation.

        Args:
            action: Recovery action selected by the user.

        Returns:
            A typed terminal or already-running conflict outcome.
        """
        snapshot = self._snapshot
        if self._active_conflict_operation is not None:
            return ConflictOutcome(
                ConflictOutcomeKind.ALREADY_RUNNING,
                action,
                note_id=snapshot.note_id if snapshot is not None else "",
                message="A conflict action is already running.",
            )
        if snapshot is None or not snapshot.in_conflict:
            return ConflictOutcome(
                ConflictOutcomeKind.NOT_IN_CONFLICT,
                action,
                note_id=snapshot.note_id if snapshot is not None else "",
                message="No note conflict is active.",
            )

        self._conflict_operation_counter += 1
        operation = _ConflictOperation(
            action=action,
            note_id=snapshot.note_id,
            session_generation=snapshot.session_generation,
            conflict_generation=snapshot.conflict_generation,
            draft_revision=snapshot.draft_revision,
            operation_token=self._conflict_operation_counter,
        )
        self._active_conflict_operation = operation

        try:
            reply = await self._port.load_note(operation.note_id)
        except asyncio.CancelledError:
            self._finish_conflict_operation(operation)
            raise
        except Exception:
            if not self._conflict_operation_is_current(operation):
                self._finish_conflict_operation(operation)
                return ConflictOutcome(
                    ConflictOutcomeKind.STALE,
                    action,
                    note_id=operation.note_id,
                    message=(
                        "The note session changed before conflict recovery completed."
                    ),
                )
            return self._finish_conflict_failure(
                operation,
                "Conflict refresh failed — try again.",
            )

        if not self._conflict_operation_is_current(operation):
            self._finish_conflict_operation(operation)
            return ConflictOutcome(
                ConflictOutcomeKind.STALE,
                action,
                note_id=operation.note_id,
                message="The note session changed before conflict recovery completed.",
            )

        current = self._snapshot
        assert current is not None
        if reply.kind is PortLoadKind.FAILED:
            failure_detail = reply.message.strip()
            message = (
                f"Conflict refresh failed — {failure_detail.rstrip('.')}. Try again."
                if failure_detail
                else "Conflict refresh failed — try again."
            )
            return self._finish_conflict_failure(operation, message)
        if reply.kind is PortLoadKind.MISSING:
            if action is ConflictAction.RELOAD:
                if current.draft_revision != operation.draft_revision:
                    return self._reload_draft_changed(operation)
                self._close_session()
                return ConflictOutcome(
                    ConflictOutcomeKind.MISSING,
                    action,
                    note_id=operation.note_id,
                    message="Note no longer exists; local conflict draft was discarded.",
                )
            message = "Note no longer exists — local draft kept."
            self._snapshot = replace(current, status_message=message)
            self._finish_conflict_operation(operation)
            return ConflictOutcome(
                ConflictOutcomeKind.MISSING,
                action,
                note_id=operation.note_id,
                message=message,
            )

        detail = reply.detail
        if detail is None or detail.note_id != operation.note_id:
            return self._finish_conflict_failure(
                operation,
                "Conflict refresh returned the wrong note — try again.",
            )

        if action is ConflictAction.RELOAD:
            if current.draft_revision != operation.draft_revision:
                return self._reload_draft_changed(operation)
            revision = current.draft_revision + 1
            draft = DatabaseNoteDraft(
                note_id=detail.note_id,
                title=detail.title,
                body=detail.body,
                keywords_text=", ".join(detail.keywords),
                revision=revision,
            )
            self._snapshot = replace(
                current,
                baseline=detail,
                draft=draft,
                saved_revision=revision,
                dirty=False,
                saving=False,
                in_conflict=False,
                status_message="Reloaded latest saved note.",
            )
            self._finish_conflict_operation(operation)
            return ConflictOutcome(
                ConflictOutcomeKind.RELOADED,
                action,
                note_id=operation.note_id,
                message="Reloaded latest saved note.",
            )

        self._snapshot = replace(
            current,
            baseline=detail,
            dirty=True,
            saving=False,
            in_conflict=False,
            status_message="Unsaved changes",
        )
        try:
            save_outcome = await self.request_save(explicit=True)
        except asyncio.CancelledError:
            self._finish_conflict_operation(operation)
            raise
        if not self._conflict_operation_matches_session(operation):
            self._finish_conflict_operation(operation)
            return ConflictOutcome(
                ConflictOutcomeKind.STALE,
                action,
                note_id=operation.note_id,
                message="The note session changed before overwrite completed.",
                save_outcome=save_outcome,
            )

        outcome_kind = {
            NoteSaveOutcomeKind.SAVED: ConflictOutcomeKind.OVERWRITTEN,
            NoteSaveOutcomeKind.ACKNOWLEDGED: ConflictOutcomeKind.OVERWRITTEN,
            NoteSaveOutcomeKind.CONFLICTED: ConflictOutcomeKind.RENEWED_CONFLICT,
            NoteSaveOutcomeKind.VALIDATION_VETO: ConflictOutcomeKind.VALIDATION_VETO,
            NoteSaveOutcomeKind.FAILED: ConflictOutcomeKind.FAILED,
            NoteSaveOutcomeKind.STALE: ConflictOutcomeKind.STALE,
            NoteSaveOutcomeKind.BLOCKED: ConflictOutcomeKind.FAILED,
        }[save_outcome.kind]
        self._finish_conflict_operation(operation)
        return ConflictOutcome(
            outcome_kind,
            action,
            note_id=operation.note_id,
            message=save_outcome.message,
            save_outcome=save_outcome,
        )

    def _conflict_operation_is_current(self, operation: _ConflictOperation) -> bool:
        snapshot = self._snapshot
        return (
            self._active_conflict_operation == operation
            and snapshot is not None
            and snapshot.note_id == operation.note_id
            and snapshot.session_generation == operation.session_generation
            and snapshot.in_conflict
            and snapshot.conflict_generation == operation.conflict_generation
        )

    def _conflict_operation_matches_session(
        self, operation: _ConflictOperation
    ) -> bool:
        snapshot = self._snapshot
        return (
            self._active_conflict_operation == operation
            and snapshot is not None
            and snapshot.note_id == operation.note_id
            and snapshot.session_generation == operation.session_generation
        )

    def _finish_conflict_operation(self, operation: _ConflictOperation) -> None:
        if self._active_conflict_operation == operation:
            self._active_conflict_operation = None

    def _finish_conflict_failure(
        self,
        operation: _ConflictOperation,
        message: str,
    ) -> ConflictOutcome:
        if self._conflict_operation_is_current(operation):
            assert self._snapshot is not None
            self._snapshot = replace(self._snapshot, status_message=message)
        self._finish_conflict_operation(operation)
        return ConflictOutcome(
            ConflictOutcomeKind.FAILED,
            operation.action,
            note_id=operation.note_id,
            message=message,
        )

    def _reload_draft_changed(self, operation: _ConflictOperation) -> ConflictOutcome:
        message = "Draft changed — Reload not applied. Choose again."
        if self._conflict_operation_is_current(operation):
            assert self._snapshot is not None
            self._snapshot = replace(self._snapshot, status_message=message)
        self._finish_conflict_operation(operation)
        return ConflictOutcome(
            ConflictOutcomeKind.DRAFT_CHANGED,
            operation.action,
            note_id=operation.note_id,
            message=message,
        )

    async def flush(self) -> NoteFlushOutcome:
        """Cross the pending-work barrier without inferring from widget state."""
        while True:
            snapshot = self._snapshot
            if snapshot is None:
                return NoteFlushOutcome(NoteFlushOutcomeKind.PERMITTED)
            if self._destructive is not None:
                return NoteFlushOutcome(
                    NoteFlushOutcomeKind.BLOCKED,
                    "A destructive action is in progress.",
                )
            if self._active_conflict_operation is not None:
                return NoteFlushOutcome(
                    NoteFlushOutcomeKind.BLOCKED,
                    "A conflict action is in progress.",
                )
            if snapshot.in_conflict:
                return NoteFlushOutcome(
                    NoteFlushOutcomeKind.CONFLICTED,
                    "Conflict — review the choices before leaving.",
                )
            if not snapshot.dirty and not snapshot.saving:
                return NoteFlushOutcome(NoteFlushOutcomeKind.PERMITTED)

            save_outcome = await self.request_save(explicit=False)
            if save_outcome.kind in {
                NoteSaveOutcomeKind.SAVED,
                NoteSaveOutcomeKind.ACKNOWLEDGED,
            }:
                continue
            outcome_kind = {
                NoteSaveOutcomeKind.VALIDATION_VETO: NoteFlushOutcomeKind.VALIDATION_VETO,
                NoteSaveOutcomeKind.FAILED: NoteFlushOutcomeKind.FAILED,
                NoteSaveOutcomeKind.CONFLICTED: NoteFlushOutcomeKind.CONFLICTED,
                NoteSaveOutcomeKind.BLOCKED: NoteFlushOutcomeKind.BLOCKED,
                NoteSaveOutcomeKind.STALE: NoteFlushOutcomeKind.STALE,
            }[save_outcome.kind]
            return NoteFlushOutcome(
                outcome_kind,
                save_outcome.message,
                save_outcome=save_outcome,
            )

    async def request_destructive_admission(
        self,
        kind: DestructiveKind,
        *,
        note_id: str,
        session_generation: int,
        expected_version: int,
        create_token: str | None = None,
    ) -> DestructiveAdmissionOutcome:
        """Flush, revalidate, then atomically block mutation and save admission."""
        if self._destructive is not None:
            return DestructiveAdmissionOutcome(
                DestructiveAdmissionOutcomeKind.ALREADY_RUNNING,
                "A destructive action is already admitted.",
            )

        initial = self._snapshot
        if (
            initial is None
            or initial.note_id != note_id
            or initial.session_generation != session_generation
            or initial.version != expected_version
        ):
            return DestructiveAdmissionOutcome(
                DestructiveAdmissionOutcomeKind.STALE,
                "The note changed before the destructive action was admitted.",
            )
        if kind is DestructiveKind.DISCARD_NEW_NOTE and (
            create_token is None or create_token != self._untouched_create_token
        ):
            return DestructiveAdmissionOutcome(
                DestructiveAdmissionOutcomeKind.NOT_ELIGIBLE,
                "This note is no longer eligible for discard.",
            )

        flush_outcome = await self.flush()
        if flush_outcome.kind is not NoteFlushOutcomeKind.PERMITTED:
            return DestructiveAdmissionOutcome(
                DestructiveAdmissionOutcomeKind.FLUSH_VETOED,
                flush_outcome.message,
                flush_outcome=flush_outcome,
            )
        if self._destructive is not None:
            return DestructiveAdmissionOutcome(
                DestructiveAdmissionOutcomeKind.ALREADY_RUNNING,
                "A destructive action is already admitted.",
            )

        snapshot = self._snapshot
        if (
            snapshot is None
            or snapshot.note_id != note_id
            or snapshot.session_generation != session_generation
        ):
            return DestructiveAdmissionOutcome(
                DestructiveAdmissionOutcomeKind.STALE,
                "The note changed before the destructive action was admitted.",
            )
        if kind is DestructiveKind.DISCARD_NEW_NOTE and (
            create_token is None or create_token != self._untouched_create_token
        ):
            return DestructiveAdmissionOutcome(
                DestructiveAdmissionOutcomeKind.NOT_ELIGIBLE,
                "This note is no longer eligible for discard.",
            )

        self._destructive_operation_counter += 1
        admission = DestructiveAdmission(
            kind=kind,
            note_id=snapshot.note_id,
            session_generation=snapshot.session_generation,
            expected_version=snapshot.version,
            operation_token=self._destructive_operation_counter,
            create_token=create_token,
        )
        self._destructive = _DestructiveState(admission)
        return DestructiveAdmissionOutcome(
            DestructiveAdmissionOutcomeKind.ADMITTED,
            admission=admission,
        )

    def mark_destructive_running(self, admission: DestructiveAdmission) -> bool:
        """Revalidate the full admission immediately before the service call."""
        state = self._destructive
        snapshot = self._snapshot
        if (
            state is None
            or state.running
            or state.admission != admission
            or snapshot is None
            or snapshot.note_id != admission.note_id
            or snapshot.session_generation != admission.session_generation
            or snapshot.version != admission.expected_version
            or (
                admission.kind is DestructiveKind.DISCARD_NEW_NOTE
                and admission.create_token != self._untouched_create_token
            )
        ):
            return False
        self._destructive = replace(state, running=True)
        return True

    def cancel_destructive(self, admission: DestructiveAdmission) -> bool:
        """Cancel only an admitted operation whose service call has not begun."""
        state = self._destructive
        if state is None or state.running or state.admission != admission:
            return False
        self._destructive = None
        return True

    def finish_destructive(
        self,
        admission: DestructiveAdmission,
        *,
        success: bool,
    ) -> bool:
        """Finish a running destructive action, closing only on success."""
        state = self._destructive
        if state is None or not state.running or state.admission != admission:
            return False
        if success:
            self._close_session()
            return True

        self._destructive = None
        if self._snapshot is not None:
            action = (
                "Discard"
                if admission.kind is DestructiveKind.DISCARD_NEW_NOTE
                else "Delete"
            )
            self._snapshot = replace(
                self._snapshot,
                status_message=f"{action} failed — edits kept. Try again.",
            )
        return True

    def _close_session(self) -> None:
        """Invalidate all tokens and end the active in-memory session."""
        self._session_request_token += 1
        self._session_generation += 1
        self._snapshot = None
        self._save_task = None
        self._pending_save_requested = False
        self._untouched_create_token = None
        self._active_conflict_operation = None
        self._destructive = None
