# tldw_chatbook/Event_Handlers/ingest_events.py
#
# Main ingest events module - imports and exports from modularized components
#
# This file maintains backward compatibility by re-exporting all functions
# that were previously defined here. The actual implementations have been
# moved to more focused modules for better organization and maintainability.
#

# Import all utilities and constants
from .ingest_utils import (
    MAX_PROMPT_PREVIEWS,
    PROMPT_FILE_FILTERS,
    MAX_CHARACTER_PREVIEWS,
    CHARACTER_FILE_FILTERS,
    MAX_NOTE_PREVIEWS,
    NOTE_FILE_FILTERS,
    _truncate_text,
)

# Import character ingestion functions
from .character_ingest_events import (
    _update_character_preview_display,
    _parse_single_character_file_for_preview,
    _handle_character_file_selected_callback,
    handle_ingest_characters_select_file_button_pressed,
    handle_ingest_characters_clear_files_button_pressed,
    handle_ingest_characters_import_now_button_pressed,
)

# Import prompt ingestion functions
from .prompt_ingest_events import (
    _update_prompt_preview_display,
    _parse_single_prompt_file_for_preview,
    _handle_prompt_file_selected_callback,
    handle_ingest_prompts_select_file_button_pressed,
    handle_ingest_prompts_clear_files_button_pressed,
    handle_ingest_prompts_import_now_button_pressed,
)

# Import note ingestion functions
from .note_ingest_events import (
    _update_note_preview_display,
    _parse_single_note_file_for_preview,
    _handle_note_file_selected_callback,
    handle_ingest_notes_select_file_button_pressed,
    handle_ingest_notes_clear_files_button_pressed,
    handle_ingest_notes_import_now_button_pressed,
)


# Import worker handlers
from .media_ingest_workers import (
    handle_tldw_api_worker_failure,
    handle_tldw_api_worker_success,
)


# --- Button Handler Map ---
# This dictionary maps button IDs to their handler functions
# It's used by the main app to route button click events
INGEST_BUTTON_HANDLERS = {
    # Prompts
    "ingest-prompts-select-file-button": handle_ingest_prompts_select_file_button_pressed,
    "ingest-prompts-clear-files-button": handle_ingest_prompts_clear_files_button_pressed,
    "ingest-prompts-import-now-button": handle_ingest_prompts_import_now_button_pressed,
    # Characters
    "ingest-characters-select-file-button": handle_ingest_characters_select_file_button_pressed,
    "ingest-characters-clear-files-button": handle_ingest_characters_clear_files_button_pressed,
    "ingest-characters-import-now-button": handle_ingest_characters_import_now_button_pressed,
    # Notes
    "ingest-notes-select-file-button": handle_ingest_notes_select_file_button_pressed,
    "ingest-notes-clear-files-button": handle_ingest_notes_clear_files_button_pressed,
    "ingest-notes-import-now-button": handle_ingest_notes_import_now_button_pressed,
}

# Export all symbols for backward compatibility
__all__ = [
    # Constants
    "MAX_PROMPT_PREVIEWS",
    "PROMPT_FILE_FILTERS",
    "MAX_CHARACTER_PREVIEWS",
    "CHARACTER_FILE_FILTERS",
    "MAX_NOTE_PREVIEWS",
    "NOTE_FILE_FILTERS",
    # Utilities
    "_truncate_text",
    # Character functions
    "_update_character_preview_display",
    "_parse_single_character_file_for_preview",
    "_handle_character_file_selected_callback",
    "handle_ingest_characters_select_file_button_pressed",
    "handle_ingest_characters_clear_files_button_pressed",
    "handle_ingest_characters_import_now_button_pressed",
    # Prompt functions
    "_update_prompt_preview_display",
    "_parse_single_prompt_file_for_preview",
    "_handle_prompt_file_selected_callback",
    "handle_ingest_prompts_select_file_button_pressed",
    "handle_ingest_prompts_clear_files_button_pressed",
    "handle_ingest_prompts_import_now_button_pressed",
    # Note functions
    "_update_note_preview_display",
    "_parse_single_note_file_for_preview",
    "_handle_note_file_selected_callback",
    "handle_ingest_notes_select_file_button_pressed",
    "handle_ingest_notes_clear_files_button_pressed",
    "handle_ingest_notes_import_now_button_pressed",
    # TLDW API functions
    # Worker handlers
    "handle_tldw_api_worker_failure",
    "handle_tldw_api_worker_success",
    # Local ingestion handlers
    # Main export
    "INGEST_BUTTON_HANDLERS",
]
