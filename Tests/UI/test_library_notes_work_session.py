import pytest

from tldw_chatbook.UI.Library_Modules.library_notes_work_session import (
    NotesWorkSessionEvent,
    NotesWorkSessionPhase,
    reduce_notes_work_session,
)


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (None, NotesWorkSessionPhase.INACTIVE),
        (119, NotesWorkSessionPhase.INACTIVE),
        (120, NotesWorkSessionPhase.ACTIVE),
        (121, NotesWorkSessionPhase.ACTIVE),
    ],
)
def test_editable_item_opened_activates_only_at_reader_width_120(
    width: int | None, expected: NotesWorkSessionPhase
) -> None:
    assert (
        reduce_notes_work_session(
            NotesWorkSessionPhase.INACTIVE,
            NotesWorkSessionEvent.EDITABLE_ITEM_OPENED,
            reader_width=width,
        )
        is expected
    )


@pytest.mark.parametrize(
    "event",
    [
        NotesWorkSessionEvent.ITEM_CHANGED,
        NotesWorkSessionEvent.SELECTION_CHANGED,
        NotesWorkSessionEvent.EDIT_MODE_CHANGED,
        NotesWorkSessionEvent.PREVIEW_MODE_CHANGED,
        NotesWorkSessionEvent.INFO_MODE_CHANGED,
        NotesWorkSessionEvent.MANAGE_MODE_CHANGED,
        NotesWorkSessionEvent.SAVE,
        NotesWorkSessionEvent.CONFLICT,
        NotesWorkSessionEvent.RECOVERY,
        NotesWorkSessionEvent.RESIZE,
        NotesWorkSessionEvent.FOLDER_BACK_TO_NAVIGATOR,
    ],
)
def test_non_boundary_events_preserve_session_phase(
    event: NotesWorkSessionEvent,
) -> None:
    for phase in NotesWorkSessionPhase:
        assert reduce_notes_work_session(phase, event) is phase


@pytest.mark.parametrize(
    "event",
    [
        NotesWorkSessionEvent.DATABASE_IDENTITY_CLEARED,
        NotesWorkSessionEvent.FOLDER_IDENTITY_CLEARED,
        NotesWorkSessionEvent.AUTHORITY_CHANGED,
        NotesWorkSessionEvent.FOLDER_ROOT_CHANGED,
        NotesWorkSessionEvent.LEFT_NOTES,
    ],
)
def test_identity_and_scope_changes_reset_to_inactive(
    event: NotesWorkSessionEvent,
) -> None:
    for phase in NotesWorkSessionPhase:
        assert reduce_notes_work_session(phase, event) is NotesWorkSessionPhase.INACTIVE


def test_manual_library_expand_marks_active_session_cancelled_and_is_idempotent() -> (
    None
):
    assert (
        reduce_notes_work_session(
            NotesWorkSessionPhase.ACTIVE,
            NotesWorkSessionEvent.MANUAL_LIBRARY_EXPAND,
        )
        is NotesWorkSessionPhase.MANUALLY_CANCELLED
    )
    for phase in NotesWorkSessionPhase:
        assert reduce_notes_work_session(
            phase,
            NotesWorkSessionEvent.MANUAL_LIBRARY_EXPAND,
        ) is (
            NotesWorkSessionPhase.MANUALLY_CANCELLED
            if phase is NotesWorkSessionPhase.ACTIVE
            else phase
        )


def test_another_item_open_does_not_rearm_cancelled_session() -> None:
    assert (
        reduce_notes_work_session(
            NotesWorkSessionPhase.MANUALLY_CANCELLED,
            NotesWorkSessionEvent.EDITABLE_ITEM_OPENED,
            reader_width=120,
        )
        is NotesWorkSessionPhase.MANUALLY_CANCELLED
    )
