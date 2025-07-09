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
    _truncate_text
)

# Import character ingestion functions
from .character_ingest_events import (
    _update_character_preview_display,
    _parse_single_character_file_for_preview,
    _handle_character_file_selected_callback,
    handle_ingest_characters_select_file_button_pressed,
    handle_ingest_characters_clear_files_button_pressed,
    handle_ingest_characters_import_now_button_pressed
)

# Import prompt ingestion functions
from .prompt_ingest_events import (
    _update_prompt_preview_display,
    _parse_single_prompt_file_for_preview,
    _handle_prompt_file_selected_callback,
    handle_ingest_prompts_select_file_button_pressed,
    handle_ingest_prompts_clear_files_button_pressed,
    handle_ingest_prompts_import_now_button_pressed
)

# Import note ingestion functions
from .note_ingest_events import (
    _update_note_preview_display,
    _parse_single_note_file_for_preview,
    _handle_note_file_selected_callback,
    handle_ingest_notes_select_file_button_pressed,
    handle_ingest_notes_clear_files_button_pressed,
    handle_ingest_notes_import_now_button_pressed
)

# Import TLDW API functions
from .tldw_api_events import (
    handle_tldw_api_auth_method_changed,
    handle_tldw_api_media_type_changed,
    _collect_common_form_data,
    _collect_video_specific_data,
    _collect_audio_specific_data,
    _collect_pdf_specific_data,
    _collect_ebook_specific_data,
    _collect_document_specific_data,
    _collect_plaintext_specific_data,
    _collect_xml_specific_data,
    _collect_mediawiki_specific_data,
    handle_tldw_api_submit_button_pressed
)

# Import worker handlers
from .media_ingest_workers import (
    handle_tldw_api_worker_failure,
    handle_tldw_api_worker_success
)

# Import local ingestion handlers
from .local_ingest_events import (
    handle_ingest_local_web_button_pressed,
    handle_local_pdf_ebook_submit_button_pressed,
    handle_local_audio_video_submit_button_pressed
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
    # TLDW API
    "tldw-api-submit-video": handle_tldw_api_submit_button_pressed,
    "tldw-api-submit-audio": handle_tldw_api_submit_button_pressed,
    "tldw-api-submit-pdf": handle_tldw_api_submit_button_pressed,
    "tldw-api-submit-ebook": handle_tldw_api_submit_button_pressed,
    "tldw-api-submit-document": handle_tldw_api_submit_button_pressed,
    "tldw-api-submit-xml": handle_tldw_api_submit_button_pressed,
    "tldw-api-submit-mediawiki_dump": handle_tldw_api_submit_button_pressed,
    # Local Web Article buttons
    "ingest-local-web-clear-urls": handle_ingest_local_web_button_pressed,
    "ingest-local-web-import-urls": handle_ingest_local_web_button_pressed,
    "ingest-local-web-remove-duplicates": handle_ingest_local_web_button_pressed,
    "ingest-local-web-process": handle_ingest_local_web_button_pressed,
    "ingest-local-web-stop": handle_ingest_local_web_button_pressed,
    "ingest-local-web-retry": handle_ingest_local_web_button_pressed,
    # Local PDF and Ebook buttons
    "local-submit-pdf": handle_local_pdf_ebook_submit_button_pressed,
    "local-submit-ebook": handle_local_pdf_ebook_submit_button_pressed,
    # Local Audio and Video buttons
    "local-submit-audio": handle_local_audio_video_submit_button_pressed,
    "local-submit-video": handle_local_audio_video_submit_button_pressed,
}

# Export all symbols for backward compatibility
__all__ = [
    # Constants
    'MAX_PROMPT_PREVIEWS',
    'PROMPT_FILE_FILTERS',
    'MAX_CHARACTER_PREVIEWS',
    'CHARACTER_FILE_FILTERS',
    'MAX_NOTE_PREVIEWS',
    'NOTE_FILE_FILTERS',
    # Utilities
    '_truncate_text',
    # Character functions
    '_update_character_preview_display',
    '_parse_single_character_file_for_preview',
    '_handle_character_file_selected_callback',
    'handle_ingest_characters_select_file_button_pressed',
    'handle_ingest_characters_clear_files_button_pressed',
    'handle_ingest_characters_import_now_button_pressed',
    # Prompt functions
    '_update_prompt_preview_display',
    '_parse_single_prompt_file_for_preview',
    '_handle_prompt_file_selected_callback',
    'handle_ingest_prompts_select_file_button_pressed',
    'handle_ingest_prompts_clear_files_button_pressed',
    'handle_ingest_prompts_import_now_button_pressed',
    # Note functions
    '_update_note_preview_display',
    '_parse_single_note_file_for_preview',
    '_handle_note_file_selected_callback',
    'handle_ingest_notes_select_file_button_pressed',
    'handle_ingest_notes_clear_files_button_pressed',
    'handle_ingest_notes_import_now_button_pressed',
    # TLDW API functions
    'handle_tldw_api_auth_method_changed',
    'handle_tldw_api_media_type_changed',
    '_collect_common_form_data',
    '_collect_video_specific_data',
    '_collect_audio_specific_data',
    '_collect_pdf_specific_data',
    '_collect_ebook_specific_data',
    '_collect_document_specific_data',
    '_collect_plaintext_specific_data',
    '_collect_xml_specific_data',
    '_collect_mediawiki_specific_data',
    'handle_tldw_api_submit_button_pressed',
    # Worker handlers
    'handle_tldw_api_worker_failure',
    'handle_tldw_api_worker_success',
    # Local ingestion handlers
    'handle_ingest_local_web_button_pressed',
    'handle_local_pdf_ebook_submit_button_pressed',
    # Main export
    'INGEST_BUTTON_HANDLERS'
]