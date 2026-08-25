from __future__ import annotations

from dataclasses import fields, replace

import pytest

from tldw_chatbook.Library.library_notes_reader_state import (
    NotesReaderState,
    delete_notes_reader_item,
    fail_notes_reader_load,
    retry_notes_reader_load,
    select_notes_reader_item,
    set_notes_reader_mode,
    settle_notes_reader_load,
    update_notes_reader_session,
)
from tldw_chatbook.Library.library_notes_state import (
    DatabaseNoteDraft,
    LibraryNoteSessionSnapshot,
    NormalizedDatabaseNote,
)


def _snapshot(
    note_id: str = "n-1",
    *,
    version: int = 4,
    revision: int = 0,
    dirty: bool = False,
    conflict: bool = False,
    title: str = "Saved title",
    body: str = "Saved body",
) -> LibraryNoteSessionSnapshot:
    baseline = NormalizedDatabaseNote(
        note_id=note_id,
        title="Saved title",
        body="Saved body",
        keywords=("saved",),
        version=version,
        created_at="2026-07-01T00:00:00+00:00",
        modified_at="2026-07-02T00:00:00+00:00",
    )
    return LibraryNoteSessionSnapshot(
        baseline=baseline,
        draft=DatabaseNoteDraft(
            note_id,
            title,
            body,
            "saved, draft" if dirty else "saved",
            revision,
        ),
        session_generation=9,
        saved_revision=0,
        dirty=dirty,
        saving=False,
        in_conflict=conflict,
        conflict_generation=1 if conflict else 0,
        status_message="Conflict" if conflict else "Ready",
    )


def _loaded_state(
    snapshot: LibraryNoteSessionSnapshot | None = None,
) -> tuple[NotesReaderState, object]:
    snapshot = snapshot or _snapshot()
    selected, request = select_notes_reader_item(
        NotesReaderState(), snapshot.note_id, version=snapshot.version
    )
    return settle_notes_reader_load(selected, request, snapshot), request


def test_default_state_is_clean_edit_presentation_without_a_second_draft_owner():
    state = NotesReaderState()

    assert state.mode == "edit"
    assert state.session is None
    assert state.preview_body == ""
    assert state.info_status == "No note loaded."
    assert {field.name for field in fields(NotesReaderState)} == {
        "selected_id",
        "loaded_id",
        "loaded_version",
        "generation",
        "mode",
        "session",
        "error",
    }


def test_loading_a_clean_snapshot_keeps_the_exact_snapshot_and_draft_unchanged():
    snapshot = _snapshot()
    selected, request = select_notes_reader_item(NotesReaderState(), "n-1", version=4)

    state = settle_notes_reader_load(selected, request, snapshot)

    assert state.selected_id == state.loaded_id == "n-1"
    assert state.loaded_version == 4
    assert state.session is snapshot
    assert state.session.dirty is False
    assert state.session.draft_revision == 0
    assert state.info_status == "Saved v4."


def test_modes_are_presentation_only_and_preview_reads_the_current_unsaved_draft():
    snapshot = _snapshot(revision=3, dirty=True, body="Current unsaved body")
    state, _request = _loaded_state(snapshot)

    preview = set_notes_reader_mode(state, "preview")
    info = set_notes_reader_mode(preview, "info")

    assert preview.preview_body == "Current unsaved body"
    assert info.info_status == "Unsaved draft · based on saved v4."
    assert preview.session is snapshot
    assert info.session is snapshot
    assert info.session.draft_revision == 3
    with pytest.raises(ValueError, match="edit, preview, or info"):
        set_notes_reader_mode(state, "context")  # type: ignore[arg-type]


def test_load_settlement_requires_matching_note_version_and_generation():
    selected, request = select_notes_reader_item(NotesReaderState(), "n-1", version=4)

    assert (
        settle_notes_reader_load(
            selected, replace(request, note_id="n-2"), _snapshot("n-2")
        )
        is selected
    )
    assert (
        settle_notes_reader_load(
            selected, replace(request, generation=request.generation + 1), _snapshot()
        )
        is selected
    )
    assert settle_notes_reader_load(selected, request, _snapshot(version=5)) is selected


def test_error_and_retry_retain_the_previous_loaded_draft_until_matching_success():
    old_snapshot = _snapshot("n-1")
    loaded, _old_request = _loaded_state(old_snapshot)
    selected, request = select_notes_reader_item(loaded, "n-2", version=7)

    failed = fail_notes_reader_load(selected, request, "Temporary failure")
    retrying, retry = retry_notes_reader_load(failed, version=7)

    assert failed.selected_id == "n-2"
    assert failed.loaded_id == "n-1"
    assert failed.session is old_snapshot
    assert failed.error == "Temporary failure"
    assert retrying.session is old_snapshot
    assert retrying.error is None
    assert retry.generation == request.generation + 1
    assert fail_notes_reader_load(retrying, request, "late") is retrying


def test_conflict_and_save_updates_require_the_loaded_identity_version_and_generation():
    state, request = _loaded_state()
    conflicted = _snapshot(revision=2, dirty=True, conflict=True, body="Keep me")

    updated = update_notes_reader_session(state, request, conflicted)

    assert updated.session is conflicted
    assert updated.session.in_conflict is True
    assert updated.preview_body == "Keep me"
    assert (
        update_notes_reader_session(
            updated, replace(request, generation=request.generation + 1), conflicted
        )
        is updated
    )

    saved = _snapshot(version=5, revision=2, dirty=False, body="Keep me")
    settled_save = update_notes_reader_session(updated, request, saved)
    assert settled_save.session is saved
    assert settled_save.loaded_version == 5
    assert settled_save.info_status == "Saved v5."


def test_matching_delete_clears_identity_and_invalidates_late_results():
    state, request = _loaded_state()

    deleted = delete_notes_reader_item(state, request)

    assert deleted.selected_id is None
    assert deleted.loaded_id is None
    assert deleted.session is None
    assert deleted.mode == "edit"
    assert deleted.generation == request.generation + 1
    assert settle_notes_reader_load(deleted, request, _snapshot()) is deleted
    assert delete_notes_reader_item(state, replace(request, version=5)) is state
