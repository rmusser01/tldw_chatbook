"""Pure state transitions for the Library Notes reader work session."""

from enum import StrEnum


class NotesWorkSessionPhase(StrEnum):
    """The lifecycle phase of the Notes reader work session."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    MANUALLY_CANCELLED = "manually_cancelled"


class NotesWorkSessionEvent(StrEnum):
    """Explicit events that can change a Notes work-session phase."""

    EDITABLE_ITEM_OPENED = "editable_item_opened"
    MANUAL_LIBRARY_EXPAND = "manual_library_expand"
    ITEM_CHANGED = "item_changed"
    SELECTION_CHANGED = "selection_changed"
    EDIT_MODE_CHANGED = "edit_mode_changed"
    PREVIEW_MODE_CHANGED = "preview_mode_changed"
    INFO_MODE_CHANGED = "info_mode_changed"
    MANAGE_MODE_CHANGED = "manage_mode_changed"
    SAVE = "save"
    CONFLICT = "conflict"
    RECOVERY = "recovery"
    RESIZE = "resize"
    FOLDER_BACK_TO_NAVIGATOR = "folder_back_to_navigator"
    DATABASE_IDENTITY_CLEARED = "database_identity_cleared"
    FOLDER_IDENTITY_CLEARED = "folder_identity_cleared"
    AUTHORITY_CHANGED = "authority_changed"
    FOLDER_ROOT_CHANGED = "folder_root_changed"
    LEFT_NOTES = "left_notes"


_RESET_EVENTS = frozenset(
    {
        NotesWorkSessionEvent.DATABASE_IDENTITY_CLEARED,
        NotesWorkSessionEvent.FOLDER_IDENTITY_CLEARED,
        NotesWorkSessionEvent.AUTHORITY_CHANGED,
        NotesWorkSessionEvent.FOLDER_ROOT_CHANGED,
        NotesWorkSessionEvent.LEFT_NOTES,
    }
)


def reduce_notes_work_session(
    phase: NotesWorkSessionPhase,
    event: NotesWorkSessionEvent,
    *,
    reader_width: int | None = None,
) -> NotesWorkSessionPhase:
    """Return the next work-session phase for an explicit event."""
    if event in _RESET_EVENTS:
        return NotesWorkSessionPhase.INACTIVE
    if event is NotesWorkSessionEvent.MANUAL_LIBRARY_EXPAND:
        return (
            NotesWorkSessionPhase.MANUALLY_CANCELLED
            if phase is NotesWorkSessionPhase.ACTIVE
            else phase
        )
    if (
        event is NotesWorkSessionEvent.EDITABLE_ITEM_OPENED
        and phase is NotesWorkSessionPhase.INACTIVE
        and reader_width is not None
        and reader_width >= 120
    ):
        return NotesWorkSessionPhase.ACTIVE
    return phase
