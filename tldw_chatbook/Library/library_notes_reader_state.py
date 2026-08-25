"""Presentation-only state transitions for the Library Notes reader."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .library_notes_state import LibraryNoteSessionSnapshot


NotesReaderMode = Literal["edit", "preview", "info"]


@dataclass(frozen=True)
class NotesReaderRequest:
    """One Notes detail request fenced against stale settlement."""

    destination: Literal["notes"]
    note_id: str
    version: int
    generation: int


@dataclass(frozen=True)
class NotesReaderState:
    """Reader presentation that references, but never copies, the Notes draft."""

    selected_id: str | None = None
    loaded_id: str | None = None
    loaded_version: int | None = None
    generation: int = 0
    mode: NotesReaderMode = "edit"
    session: LibraryNoteSessionSnapshot | None = None
    error: str | None = None

    @property
    def preview_body(self) -> str:
        """Return the current canonical draft body for Preview."""
        return self.session.body if self.session is not None else ""

    @property
    def info_status(self) -> str:
        """Distinguish persisted metadata from an unsaved current draft."""
        if self.session is None:
            return "No note loaded."
        if self.session.dirty:
            return f"Unsaved draft · based on saved v{self.session.version}."
        return f"Saved v{self.session.version}."


def _valid_identity(note_id: str, version: int) -> None:
    if not isinstance(note_id, str) or not note_id.strip():
        raise ValueError("note id must be non-blank text.")
    if type(version) is not int or version < 0:
        raise ValueError("note version must be a non-negative integer.")


def _request_matches_selection(
    state: NotesReaderState, request: NotesReaderRequest
) -> bool:
    return (
        request.destination == "notes"
        and request.note_id == state.selected_id
        and request.generation == state.generation
    )


def _request_matches_loaded(
    state: NotesReaderState, request: NotesReaderRequest
) -> bool:
    return (
        _request_matches_selection(state, request)
        and request.note_id == state.loaded_id
        and request.version == state.loaded_version
    )


def select_notes_reader_item(
    state: NotesReaderState,
    note_id: str,
    *,
    version: int,
) -> tuple[NotesReaderState, NotesReaderRequest]:
    """Select a note and create its next generation-fenced detail request."""
    _valid_identity(note_id, version)
    generation = state.generation + 1
    request = NotesReaderRequest("notes", note_id, version, generation)
    return (
        replace(
            state,
            selected_id=note_id,
            generation=generation,
            mode=state.mode if note_id == state.loaded_id else "edit",
            error=None,
        ),
        request,
    )


def retry_notes_reader_load(
    state: NotesReaderState, *, version: int
) -> tuple[NotesReaderState, NotesReaderRequest]:
    """Retry the selected note with a fresh generation."""
    if state.selected_id is None:
        raise ValueError("a selected note is required for retry.")
    return select_notes_reader_item(state, state.selected_id, version=version)


def set_notes_reader_mode(
    state: NotesReaderState, mode: NotesReaderMode
) -> NotesReaderState:
    """Change only the Edit/Preview/Info projection."""
    if mode not in {"edit", "preview", "info"}:
        raise ValueError("mode must be edit, preview, or info.")
    return replace(state, mode=mode)


def settle_notes_reader_load(
    state: NotesReaderState,
    request: NotesReaderRequest,
    session: LibraryNoteSessionSnapshot,
) -> NotesReaderState:
    """Apply a matching loaded session without taking ownership of its draft."""
    if (
        not _request_matches_selection(state, request)
        or session.note_id != request.note_id
        or session.version != request.version
    ):
        return state
    return replace(
        state,
        loaded_id=session.note_id,
        loaded_version=session.version,
        session=session,
        error=None,
    )


def fail_notes_reader_load(
    state: NotesReaderState,
    request: NotesReaderRequest,
    message: str,
) -> NotesReaderState:
    """Record a matching detail failure while retaining the previous draft."""
    if not _request_matches_selection(state, request):
        return state
    error = message.strip() if isinstance(message, str) else ""
    return replace(state, error=error or "Unable to load note.")


def update_notes_reader_session(
    state: NotesReaderState,
    fence: NotesReaderRequest,
    session: LibraryNoteSessionSnapshot,
) -> NotesReaderState:
    """Project a conflict/save update captured against the loaded version."""
    if not _request_matches_loaded(state, fence) or session.note_id != state.loaded_id:
        return state
    return replace(
        state,
        loaded_version=session.version,
        session=session,
        error=None,
    )


def delete_notes_reader_item(
    state: NotesReaderState, fence: NotesReaderRequest
) -> NotesReaderState:
    """Clear a matching deleted item and invalidate every late settlement."""
    if not _request_matches_loaded(state, fence):
        return state
    return NotesReaderState(generation=state.generation + 1)
