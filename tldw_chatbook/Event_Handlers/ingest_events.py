# tldw_chatbook/Event_Handlers/ingest_events.py
#
#
# Imports
import json
from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Any, Dict, Callable, Union
#
# 3rd-party Libraries
from loguru import logger
from textual.widgets import Select, Input, TextArea, Checkbox, Label, Static, Markdown, ListItem, \
    ListView, Collapsible, LoadingIndicator, Button, RadioSet, RadioButton
from textual.css.query import QueryError
from textual.containers import Container, VerticalScroll
from textual.worker import Worker

from ..Constants import ALL_TLDW_API_OPTION_CONTAINERS
#
# Local Imports
from ..UI.Ingest_Window import IngestWindow # Added for IngestWindow access
import tldw_chatbook.Event_Handlers.conv_char_events as ccp_handlers
from .Chat_Events import chat_events as chat_handlers
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import populate_chat_conversation_character_filter_select
from ..config import get_cli_setting
from ..tldw_api import (
    TLDWAPIClient, ProcessVideoRequest, ProcessAudioRequest,
    APIConnectionError, APIRequestError, APIResponseError, AuthenticationError,
    MediaItemProcessResult, ProcessedMediaWikiPage, BatchMediaProcessResponse,
    ProcessPDFRequest, ProcessEbookRequest, ProcessDocumentRequest,
    ProcessXMLRequest, ProcessMediaWikiRequest, ProcessPlaintextRequest,
    BatchProcessXMLResponse
)
# Prompts Interop (existing)
from ..Prompt_Management.Prompts_Interop import (
    parse_yaml_prompts_from_content, parse_json_prompts_from_content,
    parse_markdown_prompts_from_content, parse_txt_prompts_from_content,
    is_initialized as prompts_db_initialized,  # Renamed for clarity
    import_prompts_from_files, _get_file_type as _get_prompt_file_type  # Renamed for clarity
)
from ..DB.ChaChaNotes_DB import CharactersRAGDBError
# Character Chat Lib for parsing and importing character cards
from ..Character_Chat import Character_Chat_Lib as ccl
from ..DB.ChaChaNotes_DB import ConflictError as ChaChaConflictError  # For character import conflict
from ..Third_Party.textual_fspicker import Filters, FileOpen
# Note importers for multi-format support
from ..Utils.note_importers import note_importer_registry, ParsedNote
#
if TYPE_CHECKING:
    from ..app import TldwCli
########################################################################################################################
#
# Functions:




# --- Prompt Ingest Constants (existing) ---
MAX_PROMPT_PREVIEWS = 10
PROMPT_FILE_FILTERS = Filters(
    ("Markdown", lambda p: p.suffix.lower() == ".md"),
    ("JSON", lambda p: p.suffix.lower() == ".json"),
    ("YAML", lambda p: p.suffix.lower() in (".yaml", ".yml")),
    ("Text", lambda p: p.suffix.lower() == ".txt"),
    ("All Supported", lambda p: p.suffix.lower() in (".md", ".json", ".yaml", ".yml", ".txt")),
    ("All Files", lambda _: True),
)

# --- Character Ingest Constants ---
MAX_CHARACTER_PREVIEWS = 5  # Show fewer character previews as they can be larger
CHARACTER_FILE_FILTERS = Filters(
    ("Character Cards (JSON, YAML, PNG, WebP, MD)",
     lambda p: p.suffix.lower() in (".json", ".yaml", ".yml", ".png", ".webp", ".md")),
    ("JSON (*.json)", lambda p: p.suffix.lower() == ".json"),
    ("YAML (*.yaml, *.yml)", lambda p: p.suffix.lower() in (".yaml", ".yml")),
    ("PNG (*.png)", lambda p: p.suffix.lower() == ".png"),
    ("WebP (*.webp)", lambda p: p.suffix.lower() == ".webp"),
    ("Markdown (*.md)", lambda p: p.suffix.lower() == ".md"),
    ("All Files", lambda _: True),
)

# --- Notes Ingest Constants ---
MAX_NOTE_PREVIEWS = 10
NOTE_FILE_FILTERS = Filters(
    ("All Supported", lambda p: p.suffix.lower() in (".json", ".yaml", ".yml", ".txt", ".md", ".markdown", ".rst", ".csv")),
    ("JSON (*.json)", lambda p: p.suffix.lower() == ".json"),
    ("YAML (*.yaml, *.yml)", lambda p: p.suffix.lower() in (".yaml", ".yml")),
    ("Markdown (*.md)", lambda p: p.suffix.lower() in (".md", ".markdown")),
    ("Text (*.txt, *.rst)", lambda p: p.suffix.lower() in (".txt", ".text", ".rst")),
    ("CSV (*.csv)", lambda p: p.suffix.lower() == ".csv"),
    ("All Files", lambda _: True),
)

def _truncate_text(text: Optional[str], max_len: int) -> str:
    """
    Truncates a string to a maximum length, adding ellipsis if truncated.
    Returns 'N/A' if the input text is None or empty.
    """
    if not text: # Handles None or empty string
        return "N/A"
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text


# --- Character Preview Functions (NEW) ---
async def _update_character_preview_display(app: 'TldwCli') -> None:
    """Updates the character preview area in the UI."""
    try:
        preview_area = app.query_one("#ingest-characters-preview-area", VerticalScroll)
        await preview_area.remove_children()

        if not app.parsed_characters_for_preview:
            await preview_area.mount(
                Static("Select files to see a preview, or no characters found.",
                       id="ingest-characters-preview-placeholder"))
            return

        num_to_display = len(app.parsed_characters_for_preview)
        chars_to_show = app.parsed_characters_for_preview[:MAX_CHARACTER_PREVIEWS]

        for idx, char_data in enumerate(chars_to_show):
            name = char_data.get("name", f"Unnamed Character {idx + 1}")
            description = _truncate_text(char_data.get("description"), 150)
            creator = char_data.get("creator", "N/A")
            # Add more fields as relevant for previewing a character

            md_content = f"""### {name}
**Creator:** {creator}
**Description:**
```text
{description}
```
---
"""
            # For PNG/WebP, you might not show much textual preview other than filename or basic metadata if extracted.
            # For JSON/YAML/MD, you can parse more.
            # This is a simplified preview.
            if "error" in char_data:  # If parsing produced an error message
                md_content = f"""### Error parsing {char_data.get("filename", "file")}
 ```text
 {char_data["error"]}
 ```
 ---
 """
            await preview_area.mount(
                Markdown(md_content, classes="prompt-preview-item"))  # Reusing class, can make specific

        if num_to_display > MAX_CHARACTER_PREVIEWS:
            await preview_area.mount(
                Static(f"...and {num_to_display - MAX_CHARACTER_PREVIEWS} more characters loaded (not shown)."))

    except QueryError as e:
        logger.error(f"UI component not found for character preview update: {e}")
        app.notify("Error updating character preview UI.", severity="error")
    except Exception as e:
        logger.error(f"Unexpected error updating character preview: {e}", exc_info=True)
        app.notify("Unexpected error during character preview update.", severity="error")


def _parse_single_character_file_for_preview(file_path: Path, app_ref: 'TldwCli') -> List[Dict[str, Any]]:
    """
    Parses a single character file for preview.
    Returns a list containing one dict (or an error dict).
    """
    logger.debug(f"Parsing character file for preview: {file_path}")
    # For preview, we might not need the full DB interaction of ccl.load_character_card_from_file
    # We primarily need 'name' and maybe 'description'.
    # ccl.load_character_card_from_file handles JSON, YAML, MD, PNG, WebP.
    # It returns a dictionary of the character card data.

    # Minimal DB for ccl.load_character_card_from_file to work if it expects one
    # For preview, we might not want to pass a real DB.
    # Let's assume ccl.load_character_card_from_file can work without a db for simple parsing,
    # or we make a lightweight parser.
    # For simplicity, let's try calling it. If it strictly needs a DB, this part needs adjustment.

    # Placeholder: In a real scenario, ccl.load_character_card_from_file might need a dummy DB object
    # or a refactor to separate parsing from DB saving.
    # For now, we'll attempt a simplified parsing for preview.

    preview_data = {"filename": file_path.name}
    file_suffix = file_path.suffix.lower()

    try:
        if file_suffix in (".json", ".yaml", ".yml", ".md"):
            # ccl.load_character_card_from_file can take a path and parse these
            # It doesn't strictly need a DB for parsing if the file doesn't refer to external DB lookups during parse.
            # Let's assume it's primarily about loading the structure.
            # This function is in Character_Chat_Lib.py
            # `load_character_card_from_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:`
            char_dict = ccl.load_character_card_from_file(str(file_path))  # No DB passed
            if char_dict:
                preview_data.update(char_dict)
                # Ensure 'name' is present for valid preview item
                if not preview_data.get("name"):
                    preview_data["name"] = file_path.stem  # Fallback to filename without ext
            else:
                preview_data["error"] = f"Could not parse character data from {file_path.name}."
                preview_data["name"] = f"Error: {file_path.name}"

        elif file_suffix in (".png", ".webp"):
            # For images, ccl.load_character_card_from_file tries to extract metadata.
            char_dict = ccl.load_character_card_from_file(str(file_path))
            if char_dict:
                preview_data.update(char_dict)
                if not preview_data.get("name"):
                    preview_data["name"] = file_path.stem
            else:
                preview_data["name"] = file_path.name  # Just show filename
                preview_data["description"] = "Image file (binary data not shown in preview)"
        else:
            preview_data["error"] = f"Unsupported file type for character preview: {file_path.name}"
            preview_data["name"] = f"Error: {file_path.name}"

        return [preview_data]

    except Exception as e:
        logger.error(f"Error parsing character file {file_path} for preview: {e}", exc_info=True)
        app_ref.notify(f"Error previewing {file_path.name}.", severity="error")
        return [{"filename": file_path.name, "name": f"Error: {file_path.name}", "error": str(e)}]


async def _handle_character_file_selected_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    """Callback for character file selection."""
    if selected_path:
        logger.info(f"Character file selected via dialog: {selected_path}")
        if selected_path in app.selected_character_files_for_import:
            app.notify(f"File '{selected_path.name}' is already in the character selection.", severity="warning")
            return

        app.selected_character_files_for_import.append(selected_path)
        app.last_character_import_dir = selected_path.parent

        try:
            list_view = app.query_one("#ingest-characters-selected-files-list", ListView)

            # Check if the list view contains only the "No files selected." placeholder
            # This is safer than assuming it's always the first child.
            placeholder_exists = False
            if list_view.children:  # Check if there are any children
                first_child = list_view.children[0]
                if isinstance(first_child, ListItem) and first_child.children:
                    first_label_of_first_item = first_child.children[0]
                    if isinstance(first_label_of_first_item, Label):
                        # Convert Label's renderable (Rich Text) to plain string for comparison
                        if str(first_label_of_first_item.renderable).strip() == "No files selected.":
                            placeholder_exists = True

            if placeholder_exists:
                await list_view.clear()
                logger.debug("Cleared 'No files selected.' placeholder from character list.")

            await list_view.append(ListItem(Label(str(selected_path))))
            logger.debug(f"Appended '{selected_path}' to character list view.")

        except QueryError:
            logger.error("Could not find #ingest-characters-selected-files-list ListView to update.")
        except Exception as e_lv:
            logger.error(f"Error updating character list view: {e_lv}", exc_info=True)

        parsed_chars_from_file = _parse_single_character_file_for_preview(selected_path, app)
        app.parsed_characters_for_preview.extend(parsed_chars_from_file)

        await _update_character_preview_display(app)
    else:
        logger.info("Character file selection cancelled.")
        app.notify("File selection cancelled.")


# --- Prompt Ingest Handlers (existing, ensure they use renamed constants/functions if any) ---
async def handle_ingest_prompts_select_file_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger.debug("Select Prompt File(s) button pressed. Opening file dialog.")
    current_dir = app.last_prompt_import_dir or Path(".")
    await app.push_screen(
        FileOpen(
            location=str(current_dir),
            title="Select Prompt File (.md, .json, .yaml, .txt)",
            filters=PROMPT_FILE_FILTERS
        ),
        lambda path: app.call_after_refresh(lambda: _handle_prompt_file_selected_callback(app, path))
        # path type here is Optional[Path]
    )



# --- Character Ingest Handlers (NEW) ---
async def handle_ingest_characters_select_file_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Select Character File(s)' button press."""
    logger.debug("Select Character File(s) button pressed. Opening file dialog.")
    current_dir = app.last_character_import_dir or Path(".")  # Use new state var

    await app.push_screen(
        FileOpen(
            location=str(current_dir),
            title="Select Character File (.json, .yaml, .png, .webp, .md)",
            filters=CHARACTER_FILE_FILTERS
        ),
        lambda path: app.call_after_refresh(lambda: _handle_character_file_selected_callback(app, path))
        # path type here is Optional[Path]
    )


async def handle_ingest_characters_clear_files_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles 'Clear Selection' for character import."""
    logger.info("Clearing selected character files and preview.")
    app.selected_character_files_for_import.clear()
    app.parsed_characters_for_preview.clear()

    try:
        selected_list_view = app.query_one("#ingest-characters-selected-files-list", ListView)
        await selected_list_view.clear()
        await selected_list_view.append(ListItem(Label("No files selected.")))

        preview_area = app.query_one("#ingest-characters-preview-area", VerticalScroll)
        await preview_area.remove_children()
        await preview_area.mount(Static("Select files to see a preview.", id="ingest-characters-preview-placeholder"))

        status_area = app.query_one("#ingest-character-import-status-area", TextArea)
        status_area.clear()
        app.notify("Character selection and preview cleared.")
    except QueryError as e:
        logger.error(f"UI component not found for clearing character selection: {e}")
        app.notify("Error clearing character UI.", severity="error")


async def handle_ingest_characters_import_now_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles 'Import Selected Characters Now' button press."""
    logger.info("Import Selected Character Files Now button pressed.")

    if not app.selected_character_files_for_import:
        app.notify("No character files selected to import.", severity="warning")
        return

    if not app.notes_service:  # Character cards are stored via NotesService (ChaChaNotesDB)
        msg = "Notes/Character database service is not initialized. Cannot import characters."
        app.notify(msg, severity="error", timeout=7)
        logger.error(msg + " Aborting character import.")
        return

    try:
        status_area = app.query_one("#ingest-character-import-status-area", TextArea)
        status_area.clear()  # Clear previous status
        status_area.load_text("Starting character import process...\n")  # Use load_text to set initial content
    except QueryError:
        logger.error("Could not find #ingest-character-import-status-area TextArea.")
        app.notify("Status display area not found.", severity="error")
        return

    app.notify("Importing characters...")

    db = app.notes_service._get_db(app.notes_user_id)

    async def import_worker_char():
        results = []
        for file_path in app.selected_character_files_for_import:
            try:
                # Ensure file_path is a string for ccl.import_and_save_character_from_file
                char_id = ccl.import_and_save_character_from_file(db, str(file_path))
                if char_id is not None:
                    char_name = file_path.stem
                    try:
                        card_data = ccl.load_character_card_from_file(str(file_path))
                        if card_data and card_data.get("name"):
                            char_name = card_data.get("name")
                    except Exception:
                        pass

                    results.append({
                        "file_path": str(file_path),
                        "character_name": char_name,
                        "status": "success",
                        "message": f"Character imported successfully. ID: {char_id}",
                        "char_id": char_id
                    })
                else:
                    results.append({
                        "file_path": str(file_path),
                        "character_name": file_path.stem,
                        "status": "failure",
                        "message": "Failed to import (see logs for details)."
                    })
            except ChaChaConflictError as ce:
                results.append({
                    "file_path": str(file_path),
                    "character_name": file_path.stem,
                    "status": "conflict",
                    "message": str(ce)
                })
            except ImportError as ie:
                results.append({
                    "file_path": str(file_path),
                    "character_name": file_path.stem,
                    "status": "failure",
                    "message": f"Import error: {ie}. A required library might be missing."
                })
            except Exception as e:
                logger.error(f"Error importing character from {file_path}: {e}", exc_info=True)
                results.append({
                    "file_path": str(file_path),
                    "character_name": file_path.stem,
                    "status": "failure",
                    "message": f"Unexpected error: {type(e).__name__}"
                })
        return results

    def on_import_success_char(results: List[Dict[str, Any]]):
        log_text_parts = ["Character import process finished.\n\nResults:\n"]  # Renamed to avoid conflict
        successful_imports = 0
        failed_imports = 0
        for res in results:
            status = res.get("status", "unknown")
            file_path_str = res.get("file_path", "N/A")
            char_name = res.get("character_name", "N/A")
            message = res.get("message", "")

            log_text_parts.append(f"File: {Path(file_path_str).name}\n")
            log_text_parts.append(f"  Character: '{char_name}'\n")
            log_text_parts.append(f"  Status: {status.upper()}\n")
            if message:
                log_text_parts.append(f"  Message: {message}\n")
            log_text_parts.append("-" * 30 + "\n")

            if status == "success":
                successful_imports += 1
            else:
                failed_imports += 1

        summary = f"\nSummary: {successful_imports} characters imported, {failed_imports} failed/conflicts."
        log_text_parts.append(summary)

        try:
            status_area_widget = app.query_one("#ingest-character-import-status-area", TextArea)
            status_area_widget.load_text("".join(log_text_parts))  # Use load_text for the final result
        except QueryError:
            logger.error("Could not find #ingest-character-import-status-area to update with results.")

        app.notify(f"Character import finished. Success: {successful_imports}, Failed: {failed_imports}", timeout=8)
        logger.info(summary)

        app.call_later(populate_chat_conversation_character_filter_select, app)
        app.call_later(ccp_handlers.populate_ccp_character_select, app)

    def on_import_failure_char(error: Exception):
        logger.error(f"Character import worker failed critically: {error}", exc_info=True)
        try:
            status_area_widget = app.query_one("#ingest-character-import-status-area", TextArea)
            # Append error to existing text or load new text
            current_text = status_area_widget.text
            status_area_widget.load_text(
                current_text + f"\nCharacter import process failed critically: {error}\nCheck logs.\n")
        except QueryError:
            logger.error("Could not find #ingest-character-import-status-area to report critical failure.")

        app.notify(f"Character import CRITICALLY failed: {error}", severity="error", timeout=10)

    app.run_worker(
        import_worker_char,
        name="character_import_worker",
        group="file_operations",
        description="Importing selected character files."
    )


async def _update_prompt_preview_display(app: 'TldwCli') -> None:
    """Updates the prompt preview area in the UI."""
    try:
        preview_area = app.query_one("#ingest-prompts-preview-area", VerticalScroll)
        await preview_area.remove_children()

        if not app.parsed_prompts_for_preview:
            await preview_area.mount(
                Static("Select files to see a preview, or no prompts found.", id="ingest-prompts-preview-placeholder"))
            return

        num_to_display = len(app.parsed_prompts_for_preview)
        prompts_to_show = app.parsed_prompts_for_preview[:MAX_PROMPT_PREVIEWS]

        for idx, prompt_data in enumerate(prompts_to_show):
            name = prompt_data.get("name", f"Unnamed Prompt {idx + 1}")
            author = prompt_data.get("author", "N/A")
            details = _truncate_text(prompt_data.get("details"), 150)
            system_prompt = _truncate_text(prompt_data.get("system_prompt"), 200)
            user_prompt = _truncate_text(prompt_data.get("user_prompt"), 200)
            keywords_list = prompt_data.get("keywords", [])
            keywords = ", ".join(keywords_list) if keywords_list else "N/A"

            md_content = f"""### {name}
**Author:** {author}
**Keywords:** {keywords}

**Details:**
```text
{details}
```

**System Prompt:**
```text
{system_prompt}
```

**User Prompt:**
```text
{user_prompt}
```
---
"""
            await preview_area.mount(Markdown(md_content, classes="prompt-preview-item"))

        if num_to_display > MAX_PROMPT_PREVIEWS:
            await preview_area.mount(
                Static(f"...and {num_to_display - MAX_PROMPT_PREVIEWS} more prompts loaded (not shown)."))

    except QueryError as e:
        logger.error(f"UI component not found for prompt preview update: {e}")
        app.notify("Error updating prompt preview UI.", severity="error")
    except Exception as e:
        logger.error(f"Unexpected error updating prompt preview: {e}", exc_info=True)
        app.notify("Unexpected error during preview update.", severity="error")


def _parse_single_prompt_file_for_preview(file_path: Path, app_ref: 'TldwCli') -> List[Dict[str, Any]]:
    """Parses a single prompt file and returns a list of prompt data dicts."""
    file_type = _get_prompt_file_type(file_path)  # Use helper from interop
    if not file_type:
        logger.warning(f"Unsupported file type for preview: {file_path}")
        return [{"name": f"Error: Unsupported type {file_path.name}",
                 "details": "File type not recognized for prompt import."}]

    parser_map: Dict[str, Callable[[str], List[Dict[str, Any]]]] = {
        "json": parse_json_prompts_from_content,
        "yaml": parse_yaml_prompts_from_content,
        "markdown": parse_markdown_prompts_from_content,
        "txt": parse_txt_prompts_from_content,
    }
    parser = parser_map.get(file_type)
    if not parser:
        logger.error(f"No parser found for file type {file_type} (preview)")
        return [
            {"name": f"Error: No parser for {file_path.name}", "details": f"File type '{file_type}' has no parser."}]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        parsed = parser(content)
        if not parsed:  # If parser returns empty list (e.g. empty file or no valid prompts)
            logger.info(f"No prompts found in {file_path.name} by parser for preview.")
            # Not necessarily an error, could be an empty file.
            # Return an empty list, or a specific message if preferred.
            return []
        return parsed
    except RuntimeError as e:
        logger.error(f"Parser dependency missing for {file_path}: {e}")
        app_ref.notify(f"Cannot preview {file_path.name}: Required library missing ({e}).", severity="error", timeout=7)
        return [{"name": f"Error processing {file_path.name}", "details": str(e)}]
    except ValueError as e:
        logger.error(f"Failed to parse {file_path} for preview: {e}")
        app_ref.notify(f"Error parsing {file_path.name}: Invalid format.", severity="warning", timeout=7)
        return [{"name": f"Error parsing {file_path.name}", "details": str(e)}]
    except Exception as e:
        logger.error(f"Unexpected error reading/parsing {file_path} for preview: {e}", exc_info=True)
        app_ref.notify(f"Error reading {file_path.name}.", severity="error", timeout=7)
        return [{"name": f"Error reading {file_path.name}", "details": str(e)}]


async def _handle_prompt_file_selected_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    """
    Callback function executed after the FileOpen dialog for prompt selection returns.
    """
    if selected_path:
        logger.info(f"Prompt file selected via dialog: {selected_path}")
        if selected_path in app.selected_prompt_files_for_import:
            app.notify(f"File '{selected_path.name}' is already in the selection.", severity="warning")
            return

        app.selected_prompt_files_for_import.append(selected_path)
        app.last_prompt_import_dir = selected_path.parent

        try:
            list_view = app.query_one("#ingest-prompts-selected-files-list", ListView)

            placeholder_exists = False
            if list_view.children:  # Check if there are any children
                first_child = list_view.children[0]
                # Ensure the first child is a ListItem and it has children (the Label)
                if isinstance(first_child, ListItem) and first_child.children:
                    first_label_of_first_item = first_child.children[0]
                    if isinstance(first_label_of_first_item, Label):
                        # Convert Label's renderable (Rich Text) to plain string for comparison
                        if str(first_label_of_first_item.renderable).strip() == "No files selected.":
                            placeholder_exists = True

            if placeholder_exists:
                await list_view.clear()
                logger.debug("Cleared 'No files selected.' placeholder from prompt list.")

            await list_view.append(ListItem(Label(str(selected_path))))
            logger.debug(f"Appended '{selected_path}' to prompt list view.")

        except QueryError:
            logger.error("Could not find #ingest-prompts-selected-files-list ListView to update.")
        except Exception as e_lv:
            logger.error(f"Error updating prompt list view: {e_lv}", exc_info=True)

        # Parse this file and add to overall preview list
        parsed_prompts_from_file = _parse_single_prompt_file_for_preview(selected_path, app)
        app.parsed_prompts_for_preview.extend(parsed_prompts_from_file)

        await _update_prompt_preview_display(app)  # Update the preview display
    else:
        logger.info("Prompt file selection cancelled by user.")
        app.notify("File selection cancelled.")


async def handle_ingest_prompts_clear_files_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Clear Selection' button press for prompt import."""
    logger.info("Clearing selected prompt files and preview.")
    app.selected_prompt_files_for_import.clear()
    app.parsed_prompts_for_preview.clear()

    try:
        selected_list_view = app.query_one("#ingest-prompts-selected-files-list", ListView)
        await selected_list_view.clear()
        await selected_list_view.append(ListItem(Label("No files selected.")))

        preview_area = app.query_one("#ingest-prompts-preview-area", VerticalScroll)
        await preview_area.remove_children()
        await preview_area.mount(Static("Select files to see a preview.", id="ingest-prompts-preview-placeholder"))

        status_area = app.query_one("#prompt-import-status-area", TextArea)
        status_area.clear()
        app.notify("Selection and preview cleared.")
    except QueryError as e:
        logger.error(f"UI component not found for clearing prompt selection: {e}")
        app.notify("Error clearing UI.", severity="error")


async def handle_ingest_prompts_import_now_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Import Selected Files Now' button press."""
    logger.info("Import Selected Prompt Files Now button pressed.")

    if not app.selected_prompt_files_for_import:
        app.notify("No prompt files selected to import.", severity="warning")
        return

    if not prompts_db_initialized():
        msg = "Prompts database is not initialized. Cannot import."
        app.notify(msg, severity="error", timeout=7)
        logger.error(msg + " Aborting import.")
        return

    try:
        status_area = app.query_one("#prompt-import-status-area", TextArea)
        status_area.text = ""
        status_area.text = "Starting import process...\n"
    except QueryError:
        logger.error("Could not find #prompt-import-status-area TextArea.")
        app.notify("Status display area not found.", severity="error")
        return

    app.notify("Importing prompts... This may take a moment.")

    # The worker function itself remains the same
    async def import_worker_target():  # Renamed to avoid confusion with Worker class
        logger.info("--- import_worker_target (Prompts) RUNNING ---")
        try:
            results = import_prompts_from_files(app.selected_prompt_files_for_import)
            logger.info(f"--- import_worker_target (Prompts) FINISHED, results count: {len(results)} ---")
            return results  # Return the results
        except Exception as e_worker:
            logger.error(f"Exception inside import_worker_target (Prompts): {e_worker}", exc_info=True)
            # To signal an error to the worker system, you should re-raise the exception
            # or return a specific error indicator if you want to handle it differently
            # in on_worker_state_changed. For now, re-raising is simpler.
            raise e_worker

    # Define the functions that will handle success and failure,
    # these will be called by your app's on_worker_state_changed handler.
    # We pass the worker_name to identify which worker completed.

    def process_prompt_import_success(results: List[Dict[str, Any]], worker_name: str):
        if worker_name != "prompt_import_worker":  # Ensure this is for the correct worker
            return

        logger.info(f"--- process_prompt_import_success CALLED for worker: {worker_name} ---")
        logger.debug(f"Import results received: {results}")

        log_text_parts = ["Import process finished.\n\nResults:\n"]
        successful_imports = 0
        failed_imports = 0

        if not results:
            log_text_parts.append("No results returned from import worker.\n")
            logger.warning("process_prompt_import_success: Received empty results list.")
        else:
            for res_idx, res in enumerate(results):
                logger.debug(f"Processing result item {res_idx}: {res}")
                status = res.get("status", "unknown")
                file_path_str = res.get("file_path", "N/A")
                prompt_name = res.get("prompt_name", "N/A")
                message = res.get("message", "")

                log_text_parts.append(f"File: {Path(file_path_str).name}\n")
                if prompt_name and prompt_name != "N/A":
                    log_text_parts.append(f"  Prompt: '{prompt_name}'\n")
                log_text_parts.append(f"  Status: {status.upper()}\n")
                if message:
                    log_text_parts.append(f"  Message: {message}\n")
                log_text_parts.append("-" * 30 + "\n")

                if status == "success":
                    successful_imports += 1
                else:
                    failed_imports += 1

        summary = f"\nSummary: {successful_imports} prompts imported successfully, {failed_imports} failed."
        log_text_parts.append(summary)
        final_log_text_to_display = "".join(log_text_parts)
        logger.debug(f"Final text for status_area:\n{final_log_text_to_display}")

        try:
            status_area_cb = app.query_one("#prompt-import-status-area", TextArea)
            logger.info("Successfully queried #prompt-import-status-area in process_prompt_import_success.")
            status_area_cb.load_text(final_log_text_to_display)
            logger.info("Called load_text on #prompt-import-status-area.")
            status_area_cb.refresh(layout=True)
            logger.info("Called refresh() on status_area_cb.")
        except QueryError:
            logger.error("Failed to find #prompt-import-status-area in process_prompt_import_success.")
        except Exception as e_load_text:
            logger.error(f"Error during status_area_cb.load_text in process_prompt_import_success: {e_load_text}",
                         exc_info=True)

        app.notify(f"Prompt import finished. Success: {successful_imports}, Failed: {failed_imports}", timeout=8)
        logger.info(f"Prompt import summary: {summary.strip()}")

        app.call_later(ccp_handlers.populate_ccp_prompts_list_view, app)
        app.call_later(chat_handlers.handle_chat_sidebar_prompt_search_changed, app, "")
        logger.info("--- process_prompt_import_success FINISHED ---")

    def process_prompt_import_failure(error: Exception, worker_name: str):
        if worker_name != "prompt_import_worker":
            return

        logger.error(f"--- process_prompt_import_failure CALLED for worker {worker_name}: {error} ---", exc_info=True)
        try:
            status_area_cb_fail = app.query_one("#prompt-import-status-area", TextArea)
            current_text = status_area_cb_fail.text
            status_area_cb_fail.load_text(
                current_text + f"\nImport process failed critically: {error}\nCheck logs for details.\n")
        except QueryError:
            logger.error("Failed to find #prompt-import-status-area in process_prompt_import_failure.")
        app.notify(f"Prompt import failed: {str(error)[:100]}", severity="error", timeout=10)

    # Store these handlers on the app instance temporarily or pass them via a different mechanism
    # For simplicity here, we'll assume app.py's on_worker_state_changed will call them.
    # A more robust way is to make these methods of a class or use a dispatch dictionary in app.py.
    app.prompt_import_success_handler = process_prompt_import_success
    app.prompt_import_failure_handler = process_prompt_import_failure

    # Run the worker
    app.run_worker(
        import_worker_target,  # The async callable
        name="prompt_import_worker",  # Crucial for identifying the worker later
        group="file_operations",
        description="Importing selected prompt files."
        # No on_success or on_failure here
    )


# --- TLDW API Form Handlers ---
async def handle_tldw_api_auth_method_changed(app: 'TldwCli', event: Union[Select.Changed, str]) -> None:
    event_value = str(event.value) if isinstance(event, Select.Changed) else event
    logger.debug(f"TLDW API Auth method changed to: {event_value}")
    try:
        custom_token_input = app.query_one("#tldw-api-custom-token", Input)
        custom_token_label = app.query_one("#tldw-api-custom-token-label", Label)
        if event_value == "custom_token":
            custom_token_input.display = True
            custom_token_label.display = True
            custom_token_input.focus()
        else:
            custom_token_input.display = False
            custom_token_label.display = False
    except QueryError as e:
        logger.error(f"UI component not found for TLDW API auth method change: {e}")

async def handle_tldw_api_media_type_changed(app: 'TldwCli', event: Union[Select.Changed, str]) -> None:
    """Shows/hides media type specific option containers."""
    event_value = str(event.value) if isinstance(event, Select.Changed) else event
    logger.debug(f"TLDW API Media Type changed to: {event_value}")
    try:
        # Hide all specific option containers first
        for container_id in ALL_TLDW_API_OPTION_CONTAINERS:
            try:
                app.query_one(f"#{container_id}", Container).display = False
            except QueryError:
                pass # Container might not exist if not composed yet, or for all types

        # Show the relevant one
        target_container_id = f"tldw-api-{event_value.lower().replace('_', '-')}-options"
        if event_value: # If a type is selected
            try:
                container_to_show = app.query_one(f"#{target_container_id}", Container)
                container_to_show.display = True
                logger.info(f"Displaying options container: {target_container_id}")
            except QueryError:
                logger.warning(f"Options container #{target_container_id} not found for media type {event_value}.")

    except QueryError as e:
        logger.error(f"UI component not found for TLDW API media type change: {e}")
    except Exception as ex:
        logger.error(f"Unexpected error handling media type change: {ex}", exc_info=True)

def _collect_common_form_data(app: 'TldwCli', media_type: str) -> Dict[str, Any]:
    """Collects common data fields from the TLDW API form for a given media_type."""
    data = {}
    # Keep track of which field was being processed for better error messages
    # The f-string will be used in the actual query_one call.
    current_field_template_for_error = "Unknown Field-{media_type}"

    # Get the IngestWindow instance to access selected_local_files
    try:
        ingest_window = app.query_one(IngestWindow)
    except QueryError:
        logger.error("Could not find IngestWindow instance to retrieve selected files.")
        # Decide how to handle this: raise error, return empty, or notify.
        # For now, let's log and proceed, which means local_files might be empty.
        # A more robust solution might involve ensuring IngestWindow is always available.
        ingest_window = None # Or handle error more strictly

    try:
        current_field_template_for_error = f"#tldw-api-urls-{media_type}"
        data["urls"] = [url.strip() for url in app.query_one(f"#tldw-api-urls-{media_type}", TextArea).text.splitlines() if url.strip()]

        # Try to get local files from the individual window's selected_local_files
        data["local_files"] = []
        # First try the old way with IngestWindow
        if ingest_window and hasattr(ingest_window, 'selected_local_files') and media_type in ingest_window.selected_local_files:
            # Convert Path objects to strings as expected by the API client processing functions
            data["local_files"] = [str(p) for p in ingest_window.selected_local_files[media_type]]
        else:
            # Try to find the specific media type window
            try:
                # Map media type to window class
                window_map = {
                    "video": "IngestTldwApiVideoWindow",
                    "audio": "IngestTldwApiAudioWindow", 
                    "pdf": "IngestTldwApiPdfWindow",
                    "ebook": "IngestTldwApiEbookWindow",
                    "document": "IngestTldwApiDocumentWindow",
                    "xml": "IngestTldwApiXmlWindow",
                    "mediawiki_dump": "IngestTldwApiMediaWikiWindow"
                }
                if media_type in window_map:
                    from ..Widgets import IngestTldwApiVideoWindow, IngestTldwApiAudioWindow, IngestTldwApiPdfWindow, IngestTldwApiEbookWindow, IngestTldwApiDocumentWindow, IngestTldwApiXmlWindow, IngestTldwApiMediaWikiWindow
                    window_classes = {
                        "video": IngestTldwApiVideoWindow,
                        "audio": IngestTldwApiAudioWindow,
                        "pdf": IngestTldwApiPdfWindow,
                        "ebook": IngestTldwApiEbookWindow,
                        "document": IngestTldwApiDocumentWindow,
                        "xml": IngestTldwApiXmlWindow,
                        "mediawiki_dump": IngestTldwApiMediaWikiWindow
                    }
                    window_class = window_classes.get(media_type)
                    if window_class:
                        media_window = app.query_one(window_class)
                        if hasattr(media_window, 'selected_local_files'):
                            data["local_files"] = [str(p) for p in media_window.selected_local_files]
            except:
                # If we can't find the window, just use empty list
                pass

        current_field_template_for_error = f"#tldw-api-title-{media_type}"
        data["title"] = app.query_one(f"#tldw-api-title-{media_type}", Input).value or None

        current_field_template_for_error = f"#tldw-api-author-{media_type}"
        data["author"] = app.query_one(f"#tldw-api-author-{media_type}", Input).value or None

        current_field_template_for_error = f"#tldw-api-keywords-{media_type}"
        data["keywords_str"] = app.query_one(f"#tldw-api-keywords-{media_type}", TextArea).text

        current_field_template_for_error = f"#tldw-api-custom-prompt-{media_type}"
        data["custom_prompt"] = app.query_one(f"#tldw-api-custom-prompt-{media_type}", TextArea).text or None

        current_field_template_for_error = f"#tldw-api-system-prompt-{media_type}"
        data["system_prompt"] = app.query_one(f"#tldw-api-system-prompt-{media_type}", TextArea).text or None

        current_field_template_for_error = f"#tldw-api-perform-analysis-{media_type}"
        data["perform_analysis"] = app.query_one(f"#tldw-api-perform-analysis-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-overwrite-db-{media_type}"
        data["overwrite_existing_db"] = app.query_one(f"#tldw-api-overwrite-db-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-perform-chunking-{media_type}"
        data["perform_chunking"] = app.query_one(f"#tldw-api-perform-chunking-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-chunk-method-{media_type}"
        chunk_method_select = app.query_one(f"#tldw-api-chunk-method-{media_type}", Select)
        data["chunk_method"] = chunk_method_select.value if chunk_method_select.value != Select.BLANK else None

        current_field_template_for_error = f"#tldw-api-chunk-size-{media_type}"
        data["chunk_size"] = int(app.query_one(f"#tldw-api-chunk-size-{media_type}", Input).value or "500")

        current_field_template_for_error = f"#tldw-api-chunk-overlap-{media_type}"
        data["chunk_overlap"] = int(app.query_one(f"#tldw-api-chunk-overlap-{media_type}", Input).value or "200")

        current_field_template_for_error = f"#tldw-api-chunk-lang-{media_type}"
        data["chunk_language"] = app.query_one(f"#tldw-api-chunk-lang-{media_type}", Input).value or None

        current_field_template_for_error = f"#tldw-api-adaptive-chunking-{media_type}"
        data["use_adaptive_chunking"] = app.query_one(f"#tldw-api-adaptive-chunking-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-multi-level-chunking-{media_type}"
        data["use_multi_level_chunking"] = app.query_one(f"#tldw-api-multi-level-chunking-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-custom-chapter-pattern-{media_type}"
        data["custom_chapter_pattern"] = app.query_one(f"#tldw-api-custom-chapter-pattern-{media_type}", Input).value or None

        current_field_template_for_error = f"#tldw-api-analysis-api-name-{media_type}"
        analysis_api_select = app.query_one(f"#tldw-api-analysis-api-name-{media_type}", Select)
        data["api_name"] = analysis_api_select.value if analysis_api_select.value != Select.BLANK else None

        current_field_template_for_error = f"#tldw-api-analysis-api-key-{media_type}"
        data["api_key"] = app.query_one(f"#tldw-api-analysis-api-key-{media_type}", Input).value or None

        current_field_template_for_error = f"#tldw-api-use-cookies-{media_type}"
        data["use_cookies"] = app.query_one(f"#tldw-api-use-cookies-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-cookies-{media_type}"
        data["cookies"] = app.query_one(f"#tldw-api-cookies-{media_type}", TextArea).text or None

        current_field_template_for_error = f"#tldw-api-summarize-recursively-{media_type}"
        data["summarize_recursively"] = app.query_one(f"#tldw-api-summarize-recursively-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-perform-rolling-summarization-{media_type}"
        data["perform_rolling_summarization"] = app.query_one(f"#tldw-api-perform-rolling-summarization-{media_type}", Checkbox).value

    except QueryError as e:
        # Log the specific query that failed if possible, or the last attempted field ID
        logger.error(f"Error querying TLDW API form field (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Missing form field. Details: {e}", severity="error")
        raise # Re-raise to stop further processing
    except ValueError as e: # For int() conversion errors
        logger.error(f"Error converting TLDW API form field value (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Invalid value in form field (around {current_field_template_for_error.format(media_type=media_type)}). Check numbers.", severity="error")
        raise # Re-raise
    return data


def _collect_video_specific_data(app: 'TldwCli', common_data: Dict[str, Any], media_type: str) -> ProcessVideoRequest:
    current_field_template_for_error = "Unknown Video Field-{media_type}"
    try:
        current_field_template_for_error = f"#tldw-api-video-transcription-model-{media_type}"
        common_data["transcription_model"] = app.query_one(f"#tldw-api-video-transcription-model-{media_type}",
                                                           Input).value or "deepdml/faster-whisper-large-v3-turbo-ct2"

        current_field_template_for_error = f"#tldw-api-video-transcription-language-{media_type}"
        common_data["transcription_language"] = app.query_one(f"#tldw-api-video-transcription-language-{media_type}",
                                                              Input).value or "en"

        current_field_template_for_error = f"#tldw-api-video-diarize-{media_type}"
        common_data["diarize"] = app.query_one(f"#tldw-api-video-diarize-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-video-timestamp-{media_type}"
        common_data["timestamp_option"] = app.query_one(f"#tldw-api-video-timestamp-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-video-vad-{media_type}"
        common_data["vad_use"] = app.query_one(f"#tldw-api-video-vad-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-video-confab-check-{media_type}"
        common_data["perform_confabulation_check_of_analysis"] = app.query_one(f"#tldw-api-video-confab-check-{media_type}",
                                                                               Checkbox).value

        current_field_template_for_error = f"#tldw-api-video-start-time-{media_type}"
        common_data["start_time"] = app.query_one(f"#tldw-api-video-start-time-{media_type}", Input).value or None

        current_field_template_for_error = f"#tldw-api-video-end-time-{media_type}"
        common_data["end_time"] = app.query_one(f"#tldw-api-video-end-time-{media_type}", Input).value or None

        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]

        return ProcessVideoRequest(**common_data)
    except QueryError as e:
        logger.error(f"Error querying video-specific TLDW API form field (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Missing video form field. Details: {e}", severity="error")
        raise
    except ValueError as e: # For Pydantic validation or other conversion errors
        logger.error(
            f"Error converting video-specific TLDW API form field value or creating request model (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Invalid value in video form field (around {current_field_template_for_error.format(media_type=media_type)}).", severity="error")
        raise

def _collect_audio_specific_data(app: 'TldwCli', common_data: Dict[str, Any], media_type: str) -> ProcessAudioRequest:
    current_field_template_for_error = "Unknown Audio Field-{media_type}"
    try:
        current_field_template_for_error = f"#tldw-api-audio-transcription-model-{media_type}"
        common_data["transcription_model"] = app.query_one(f"#tldw-api-audio-transcription-model-{media_type}", Input).value or "deepdml/faster-distil-whisper-large-v3.5"

        current_field_template_for_error = f"#tldw-api-audio-transcription-language-{media_type}"
        common_data["transcription_language"] = app.query_one(f"#tldw-api-audio-transcription-language-{media_type}", Input).value or "en"

        current_field_template_for_error = f"#tldw-api-audio-diarize-{media_type}"
        common_data["diarize"] = app.query_one(f"#tldw-api-audio-diarize-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-audio-timestamp-{media_type}"
        common_data["timestamp_option"] = app.query_one(f"#tldw-api-audio-timestamp-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-audio-vad-{media_type}"
        common_data["vad_use"] = app.query_one(f"#tldw-api-audio-vad-{media_type}", Checkbox).value
        # TODO: Add confab check if UI element is added: id=f"tldw-api-audio-confab-check-{media_type}"

        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]
        return ProcessAudioRequest(**common_data)
    except QueryError as e:
        logger.error(f"Error querying audio-specific TLDW API form field (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Missing audio form field. Details: {e}", severity="error")
        raise
    except ValueError as e: # For Pydantic validation or other conversion errors
        logger.error(
            f"Error converting audio-specific TLDW API form field value or creating request model (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Invalid value in audio form field (around {current_field_template_for_error.format(media_type=media_type)}).", severity="error")
        raise


def _collect_pdf_specific_data(app: 'TldwCli', common_data: Dict[str, Any], media_type: str) -> ProcessPDFRequest:
    current_field_template_for_error = "Unknown PDF Field-{media_type}"
    try:
        current_field_template_for_error = f"#tldw-api-pdf-engine-{media_type}"
        pdf_engine_select = app.query_one(f"#tldw-api-pdf-engine-{media_type}", Select)
        common_data["pdf_parsing_engine"] = pdf_engine_select.value if pdf_engine_select.value != Select.BLANK else "pymupdf4llm"

        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]
        return ProcessPDFRequest(**common_data)
    except QueryError as e:
        logger.error(f"Error querying PDF-specific TLDW API form field (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Missing PDF form field. Details: {e}", severity="error")
        raise
    except ValueError as e:
        logger.error(f"Error creating PDF request model (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Invalid value in PDF form field (around {current_field_template_for_error.format(media_type=media_type)}).", severity="error")
        raise

def _collect_ebook_specific_data(app: 'TldwCli', common_data: Dict[str, Any], media_type: str) -> ProcessEbookRequest:
    current_field_template_for_error = "Unknown Ebook Field-{media_type}"
    try:
        current_field_template_for_error = f"#tldw-api-ebook-extraction-method-{media_type}"
        extraction_method_select = app.query_one(f"#tldw-api-ebook-extraction-method-{media_type}", Select)
        common_data["extraction_method"] = extraction_method_select.value if extraction_method_select.value != Select.BLANK else "filtered"

        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]
        return ProcessEbookRequest(**common_data)
    except QueryError as e:
        logger.error(f"Error querying Ebook-specific TLDW API form field (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Missing Ebook form field. Details: {e}", severity="error")
        raise
    except ValueError as e:
        logger.error(f"Error creating Ebook request model (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Invalid value in Ebook form field (around {current_field_template_for_error.format(media_type=media_type)}).", severity="error")
        raise

def _collect_document_specific_data(app: 'TldwCli', common_data: Dict[str, Any], media_type: str) -> ProcessDocumentRequest:
    # No document-specific fields in UI yet, so it's just converting common_data
    try:
        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]
        # Add any document-specific fields here if they are added to the UI, using f"...-{media_type}"
        return ProcessDocumentRequest(**common_data)
    except Exception as e: # Catch potential Pydantic validation errors
        logger.error(f"Error creating ProcessDocumentRequest for media_type {media_type}: {e}")
        app.notify("Error: Could not prepare document request data.", severity="error")
        raise

def _collect_plaintext_specific_data(app: 'TldwCli', common_data: Dict[str, Any], media_type: str) -> ProcessPlaintextRequest:
    """Collect plaintext-specific form data and create request model."""
    current_field_template_for_error = "Unknown Plaintext Field-{media_type}"
    try:
        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]
        
        # Plaintext-specific fields
        current_field_template_for_error = f"#tldw-api-encoding-{media_type}"
        common_data["encoding"] = str(app.query_one(f"#tldw-api-encoding-{media_type}", Select).value)
        
        current_field_template_for_error = f"#tldw-api-line-ending-{media_type}"
        common_data["line_ending"] = str(app.query_one(f"#tldw-api-line-ending-{media_type}", Select).value)
        
        current_field_template_for_error = f"#tldw-api-remove-whitespace-{media_type}"
        common_data["remove_extra_whitespace"] = app.query_one(f"#tldw-api-remove-whitespace-{media_type}", Checkbox).value
        
        current_field_template_for_error = f"#tldw-api-convert-paragraphs-{media_type}"
        common_data["convert_to_paragraphs"] = app.query_one(f"#tldw-api-convert-paragraphs-{media_type}", Checkbox).value
        
        current_field_template_for_error = f"#tldw-api-split-pattern-{media_type}"
        split_pattern = app.query_one(f"#tldw-api-split-pattern-{media_type}", Input).value.strip()
        common_data["split_pattern"] = split_pattern if split_pattern else None
        
        return ProcessPlaintextRequest(**common_data)
    except QueryError as e:
        logger.error(f"Error querying Plaintext-specific TLDW API form field (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Missing Plaintext form field. Details: {e}", severity="error")
        raise
    except Exception as e:
        logger.error(f"Error creating ProcessPlaintextRequest for media_type {media_type}: {e}")
        app.notify("Error: Could not prepare plaintext request data.", severity="error")
        raise

def _collect_xml_specific_data(app: 'TldwCli', common_api_data: Dict[str, Any], media_type: str) -> ProcessXMLRequest:
    data = {}
    current_field_template_for_error = "Unknown XML Field-{media_type}"
    try:
        data["title"] = common_api_data.get("title")
        data["author"] = common_api_data.get("author")
        data["keywords"] = [k.strip() for k in common_api_data.get("keywords_str", "").split(',') if k.strip()]
        data["system_prompt"] = common_api_data.get("system_prompt")
        data["custom_prompt"] = common_api_data.get("custom_prompt")
        data["api_name"] = common_api_data.get("api_name")
        data["api_key"] = common_api_data.get("api_key")

        current_field_template_for_error = f"#tldw-api-xml-auto-summarize-{media_type}"
        data["auto_summarize"] = app.query_one(f"#tldw-api-xml-auto-summarize-{media_type}", Checkbox).value
        return ProcessXMLRequest(**data)
    except QueryError as e:
        logger.error(f"Error querying XML-specific TLDW API form field (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Missing XML form field. Details: {e}", severity="error")
        raise
    except ValueError as e:
        logger.error(f"Error creating XML request model (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Invalid value in XML form field (around {current_field_template_for_error.format(media_type=media_type)}).", severity="error")
        raise

def _collect_mediawiki_specific_data(app: 'TldwCli', common_api_data: Dict[str, Any], media_type: str) -> ProcessMediaWikiRequest:
    data = {}
    current_field_template_for_error = "Unknown MediaWiki Field-{media_type}"
    try:
        current_field_template_for_error = f"#tldw-api-mediawiki-wiki-name-{media_type}"
        data["wiki_name"] = app.query_one(f"#tldw-api-mediawiki-wiki-name-{media_type}", Input).value or "default_wiki"
        current_field_template_for_error = f"#tldw-api-mediawiki-namespaces-{media_type}"
        data["namespaces_str"] = app.query_one(f"#tldw-api-mediawiki-namespaces-{media_type}", Input).value or None
        current_field_template_for_error = f"#tldw-api-mediawiki-skip-redirects-{media_type}"
        data["skip_redirects"] = app.query_one(f"#tldw-api-mediawiki-skip-redirects-{media_type}", Checkbox).value
        
        current_field_template_for_error = f"#tldw-api-mediawiki-chunk-max-size-{media_type}"
        data["chunk_max_size"] = int(app.query_one(f"#tldw-api-mediawiki-chunk-max-size-{media_type}", Input).value or "1000")
        
        current_field_template_for_error = f"#tldw-api-mediawiki-api-name-vector-db-{media_type}"
        data["api_name_vector_db"] = app.query_one(f"#tldw-api-mediawiki-api-name-vector-db-{media_type}", Input).value or None
        
        current_field_template_for_error = f"#tldw-api-mediawiki-api-key-vector-db-{media_type}"
        data["api_key_vector_db"] = app.query_one(f"#tldw-api-mediawiki-api-key-vector-db-{media_type}", Input).value or None
        
        return ProcessMediaWikiRequest(**data)
    except QueryError as e:
        logger.error(f"Error querying MediaWiki-specific TLDW API form field (around {current_field_template_for_error.format(media_type=media_type)}): {e}")
        app.notify(f"Error: Missing MediaWiki form field. Details: {e}", severity="error")
        raise


async def handle_tldw_api_submit_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    if not event.button.id:
        logger.error("Submit button pressed but has no ID. Cannot determine media_type.")
        app.notify("Critical error: Submit button has no ID.", severity="error")
        return

    logger.info(f"TLDW API Submit button pressed: {event.button.id}")

    selected_media_type = event.button.id.replace("tldw-api-submit-", "")
    logger.info(f"Extracted media_type: {selected_media_type} from button ID.")

    app.notify(f"Processing {selected_media_type} request via tldw API...")

    try:
        loading_indicator = app.query_one(f"#tldw-api-loading-indicator-{selected_media_type}", LoadingIndicator)
        status_area = app.query_one(f"#tldw-api-status-area-{selected_media_type}", TextArea)
        submit_button = event.button # This is already the correct button
        endpoint_url_input = app.query_one(f"#tldw-api-endpoint-url-{selected_media_type}", Input)
        auth_method_select = app.query_one(f"#tldw-api-auth-method-{selected_media_type}", Select)
    except QueryError as e:
        logger.error(f"Critical UI component missing for media_type '{selected_media_type}': {e}")
        app.notify(f"Error: UI component missing for {selected_media_type}: {e.widget.id if hasattr(e, 'widget') and e.widget else 'Unknown'}. Cannot proceed.", severity="error")
        return

    endpoint_url = endpoint_url_input.value.strip()
    auth_method = str(auth_method_select.value) # Ensure it's a string

    # --- Input Validation ---
    if not endpoint_url:
        app.notify("API Endpoint URL is required.", severity="error")
        endpoint_url_input.focus()
        # No need to revert UI state as it hasn't been changed yet
        return

    if not (endpoint_url.startswith("http://") or endpoint_url.startswith("https://")):
        app.notify("API Endpoint URL must start with http:// or https://.", severity="error")
        endpoint_url_input.focus()
        # No need to revert UI state
        return

    if auth_method == str(Select.BLANK):
        app.notify("Please select an Authentication Method.", severity="error")
        auth_method_select.focus()
        return

    # --- Set UI to Loading State ---
    loading_indicator.display = True
    status_area.clear()
    status_area.load_text("Validating inputs and preparing request...")
    status_area.display = True
    submit_button.disabled = True
    # app.notify is already called at the start of the function

    def _reset_ui():
        """Return the widgets to their idle state after a hard failure."""
        loading_indicator.display = False
        submit_button.disabled = False
        status_area.load_text("Submission halted.")

    # --- Get Auth Token (after basic validations pass) ---
    auth_token: Optional[str] = None
    try:
        if auth_method == "custom_token":
            custom_token_input = app.query_one(f"#tldw-api-custom-token-{selected_media_type}", Input)
            auth_token = custom_token_input.value.strip()
            if not auth_token:
                app.notify("Custom Auth Token is required for selected method.", severity="error")
                custom_token_input.focus()
                # Revert UI loading state
                loading_indicator.display = False
                submit_button.disabled = False
                status_area.load_text("Custom token required. Submission halted.")
                return

        elif auth_method == "config_token":
            # 1.  Look in the active config, then in the environment.
            auth_token = (
                    get_cli_setting("tldw_api", "auth_token")  # ~/.config/tldw_cli/config.toml
                    or get_cli_setting("tldw_api", "api_key")  # Alternative config key
                    or getenv("TDLW_AUTH_TOKEN")  # optional override
                    or getenv("TLDW_API_KEY")  # Alternative env var
            )

            # 2. Abort early if we still have nothing.
            if not auth_token:
                msg = (
                    "Auth token not found — add it to the [tldw_api] section as "
                    "`auth_token = \"<your token>\"` or `api_key = \"<your key>\"`, "
                    "or export TDLW_AUTH_TOKEN or TLDW_API_KEY."
                )
                logger.error(msg)
                app.notify(msg, severity="error")
                _reset_ui()
                return
    except QueryError as e:
        logger.error(f"UI component not found for TLDW API auth token for {selected_media_type}: {e}")
        app.notify(f"Error: Missing UI field for auth for {selected_media_type}: {e.widget.id if hasattr(e, 'widget') and e.widget else 'Unknown'}", severity="error")
        loading_indicator.display = False
        submit_button.disabled = False
        status_area.load_text("Error accessing auth fields. Submission halted.")
        return

    status_area.load_text("Collecting form data and building request...")
    request_model: Optional[Any] = None
    local_file_paths: Optional[List[str]] = None
    try:
        common_data = _collect_common_form_data(app, selected_media_type) # Pass selected_media_type
        local_file_paths = common_data.pop("local_files", [])

        if selected_media_type == "video":
            request_model = _collect_video_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "audio":
            request_model = _collect_audio_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "pdf":
            request_model = _collect_pdf_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "ebook":
            request_model = _collect_ebook_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "document":
            request_model = _collect_document_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "plaintext":
            request_model = _collect_plaintext_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "xml":
            request_model = _collect_xml_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "mediawiki_dump":
            request_model = _collect_mediawiki_specific_data(app, common_data, selected_media_type)
        else:
            app.notify(f"Media type '{selected_media_type}' not yet supported by this client form.", severity="warning")
            loading_indicator.display = False
            submit_button.disabled = False
            status_area.load_text("Unsupported media type selected. Submission halted.")
            return
    except (QueryError, ValueError) as e:
        logger.error(f"Error collecting form data for {selected_media_type}: {e}", exc_info=True)
        app.notify(f"Error in form data for {selected_media_type}: {str(e)[:100]}. Please check fields.", severity="error")
        loading_indicator.display = False
        submit_button.disabled = False
        status_area.load_text(f"Error processing form data: {str(e)[:100]}. Submission halted.")
        return
    except Exception as e:
        logger.error(f"Unexpected error preparing request model for TLDW API ({selected_media_type}): {e}", exc_info=True)
        app.notify("Error: Could not prepare data for API request.", severity="error")
        loading_indicator.display = False
        submit_button.disabled = False
        status_area.load_text("Unexpected error preparing request. Submission halted.")
        return

    if not request_model:
        app.notify("Failed to create request model.", severity="error")
        loading_indicator.display = False
        submit_button.disabled = False
        status_area.load_text("Internal error: Failed to create request model. Submission halted.")
        return

    # URL/Local file validation (adjust for XML/MediaWiki which primarily use local_file_paths)
    if not getattr(request_model, 'urls', None) and not local_file_paths:
        # This check might be specific to certain request models, adjust if necessary
        # For XML and MediaWiki, local_file_paths is primary and urls might not exist on model
        is_xml_or_mediawiki = selected_media_type in ["xml", "mediawiki_dump"]
        if not is_xml_or_mediawiki or (is_xml_or_mediawiki and not local_file_paths):
            app.notify("Please provide at least one URL or one local file path.", severity="warning")
            try:
                app.query_one(f"#tldw-api-urls-{selected_media_type}", TextArea).focus()
            except QueryError: pass
            loading_indicator.display = False
            submit_button.disabled = False
            status_area.load_text("Missing URL or local file. Submission halted.")
            return

    status_area.load_text("Connecting to TLDW API and sending request...")
    # Determine if auth_token is a Bearer token or API key based on auth_method
    if auth_method == "custom_token":
        # Custom token is treated as Bearer token
        api_client = TLDWAPIClient(base_url=endpoint_url)
        api_client.bearer_token = auth_token
    else:
        # Config token is treated as API key
        api_client = TLDWAPIClient(base_url=endpoint_url, token=auth_token)
    overwrite_db = common_data.get("overwrite_existing_db", False) # From common_data

    # Worker and callbacks remain largely the same but need to use the correct UI element IDs for this tab
    # The on_worker_success and on_worker_failure need to know which loading_indicator/submit_button/status_area to update.
    # This is implicitly handled as they are queried again using the selected_media_type.

    async def process_media_worker(): # This worker is fine
        nonlocal request_model
        try:
            if selected_media_type == "video":
                return await api_client.process_video(request_model, local_file_paths)
            elif selected_media_type == "audio":
                return await api_client.process_audio(request_model, local_file_paths)
            elif selected_media_type == "pdf":
                return await api_client.process_pdf(request_model, local_file_paths)
            elif selected_media_type == "ebook":
                return await api_client.process_ebook(request_model, local_file_paths)
            elif selected_media_type == "document":
                return await api_client.process_document(request_model, local_file_paths)
            elif selected_media_type == "plaintext":
                return await api_client.process_plaintext(request_model, local_file_paths)
            elif selected_media_type == "xml":
                if not local_file_paths: raise ValueError("XML processing requires a local file path.")
                return await api_client.process_xml(request_model, local_file_paths[0])
            elif selected_media_type == "mediawiki_dump":
                if not local_file_paths: raise ValueError("MediaWiki processing requires a local file path.")
                # For streaming, the worker should yield, not return directly.
                # This example shows how to initiate and collect, actual handling of stream in on_success would differ.
                results = []
                async for item in api_client.process_mediawiki_dump(request_model, local_file_paths[0]):
                    results.append(item)
                return results
            else:
                raise NotImplementedError(f"Client-side processing for {selected_media_type} not implemented.")
        finally:
            await api_client.close()

    def on_worker_success(response_data: Any):
        # Query the specific UI elements for this tab
        try:
            current_loading_indicator = app.query_one(f"#tldw-api-loading-indicator-{selected_media_type}", LoadingIndicator)
            current_loading_indicator.display = False
            # current_submit_button = app.query_one(f"#tldw-api-submit-{selected_media_type}", Button) # Button instance is already event.button
            submit_button.disabled = False # submit_button is already defined from event.button
        except QueryError as e_ui:
            logger.error(f"UI component not found in on_worker_success for {selected_media_type}: {e_ui}")

        app.notify(f"TLDW API request for {selected_media_type} successful. Processing results...", timeout=2)
        logger.info(f"TLDW API Response for {selected_media_type}: {response_data}")

        try:
            current_status_area = app.query_one(f"#tldw-api-status-area-{selected_media_type}", TextArea)
            current_status_area.clear()
        except QueryError:
            logger.error(f"Could not find status_area for {selected_media_type} in on_worker_success.")
            return # Cannot display results


        if not app.media_db:
            logger.error("Media_DB_v2 not initialized. Cannot ingest API results.")
            app.notify("Error: Local media database not available.", severity="error")
            current_status_area.load_text("## Error\n\nLocal media database not available.")
            return

        processed_count = 0
        error_count = 0
        successful_ingestions_details = [] # To store details of successful items

        # Handle different response types
        results_to_ingest: List[MediaItemProcessResult] = []
        if isinstance(response_data, BatchMediaProcessResponse):
            results_to_ingest = response_data.results
        elif isinstance(response_data, BatchProcessXMLResponse):
            # Convert ProcessXMLResponseItem to MediaItemProcessResult
            for xml_item in response_data.results:
                results_to_ingest.append(MediaItemProcessResult(
                    status=xml_item.status,
                    input_ref=xml_item.input_ref,
                    processing_source=xml_item.input_ref,
                    media_type="xml",
                    metadata={"title": xml_item.title, "author": xml_item.author, "keywords": xml_item.keywords},
                    content=xml_item.content,
                    summary=xml_item.summary,
                    segments=xml_item.segments,
                    error=xml_item.error
                ))
        elif isinstance(response_data, dict) and "results" in response_data:
            if "processed_count" in response_data:
                raw_results = response_data.get("results", [])
                for item_dict in raw_results:
                    # Try to coerce into MediaItemProcessResult, might need specific mapping for XML
                    # For now, assume XML result items can be mostly mapped.
                    results_to_ingest.append(MediaItemProcessResult(**item_dict))

        elif isinstance(response_data, list) and all(isinstance(item, ProcessedMediaWikiPage) for item in response_data):
            # MediaWiki dump (if collected into a list by worker)
            for mw_page in response_data:
                if mw_page.status == "Error":
                    error_count +=1
                    logger.error(f"MediaWiki page '{mw_page.title}' processing error: {mw_page.error_message}")
                    continue
                # Adapt ProcessedMediaWikiPage to MediaItemProcessResult structure for ingestion
                results_to_ingest.append(MediaItemProcessResult(
                    status="Success", # Assume success if no error status
                    input_ref=mw_page.input_ref or mw_page.title,
                    processing_source=mw_page.title, # or another identifier
                    media_type="mediawiki_article", # or "mediawiki_page"
                    metadata={"title": mw_page.title, "page_id": mw_page.page_id, "namespace": mw_page.namespace, "revision_id": mw_page.revision_id, "timestamp": mw_page.timestamp},
                    content=mw_page.content,
                    chunks=[{"text": chunk.get("text", ""), "metadata": chunk.get("metadata", {})} for chunk in mw_page.chunks] if hasattr(mw_page, 'chunks') and mw_page.chunks else None,
                ))
        else:
            logger.error(f"Unexpected TLDW API response data type for {selected_media_type}: {type(response_data)}.")
            current_status_area.load_text(f"## API Request Processed\n\nUnexpected response format. Raw response logged.")
            current_status_area.display = True
            app.notify("Error: Received unexpected data format from API.", severity="error")
            return
        # Add elif for XML if it returns a single ProcessXMLResponseItem or similar

        for item_result in results_to_ingest:
            if item_result.status == "Success":
                media_id_ingested = None # For storing the ID if ingestion is successful
                try:
                    # Prepare data for add_media_with_keywords
                    # Keywords: API response might not have 'keywords'. Use originally submitted ones if available.
                    # For simplicity, let's assume API response's metadata *might* have keywords.
                    api_keywords = item_result.metadata.get("keywords", []) if item_result.metadata else []
                    if isinstance(api_keywords, str): # If server returns comma-sep string
                        api_keywords = [k.strip() for k in api_keywords.split(',') if k.strip()]

                    # Chunks for UnvectorizedMediaChunks
                    # Ensure chunks are in the format: [{'text': str, 'start_char': int, ...}, ...]
                    unvectorized_chunks_to_save = None
                    if item_result.chunks:
                        unvectorized_chunks_to_save = []
                        for idx, chunk_item in enumerate(item_result.chunks):
                            if isinstance(chunk_item, dict) and "text" in chunk_item:
                                unvectorized_chunks_to_save.append({
                                    "text": chunk_item.get("text"),
                                    "start_char": chunk_item.get("metadata", {}).get("start_char"), # Assuming metadata structure
                                    "end_char": chunk_item.get("metadata", {}).get("end_char"),
                                    "chunk_type": chunk_item.get("metadata", {}).get("type", selected_media_type),
                                    "metadata": chunk_item.get("metadata", {}) # Store full chunk metadata
                                })
                            elif isinstance(chunk_item, str): # If chunks are just strings
                                 unvectorized_chunks_to_save.append({"text": chunk_item})
                            else:
                                logger.warning(f"Skipping malformed chunk item: {chunk_item}")


                    media_id, media_uuid, msg = app.media_db.add_media_with_keywords(
                        url=item_result.input_ref, # Original URL/filename
                        title=item_result.metadata.get("title", item_result.input_ref) if item_result.metadata else item_result.input_ref,
                        media_type=item_result.media_type,
                        content=item_result.content or item_result.transcript, # Use transcript if content is empty
                        keywords=api_keywords or (request_model.keywords if hasattr(request_model, "keywords") else []), # Fallback to request
                        prompt=request_model.custom_prompt if hasattr(request_model, "custom_prompt") else None, # From original request
                        analysis_content=item_result.analysis or item_result.summary,
                        transcription_model=item_result.analysis_details.get("transcription_model") if item_result.analysis_details else (request_model.transcription_model if hasattr(request_model, "transcription_model") else None),
                        author=item_result.metadata.get("author") if item_result.metadata else (request_model.author if hasattr(request_model, "author") else None),
                        # ingestion_date: use current time,
                        overwrite=overwrite_db, # Use the specific DB overwrite flag
                        chunks=unvectorized_chunks_to_save # Pass prepared chunks
                    )
                    if media_id:
                        logger.info(f"Successfully ingested '{item_result.input_ref}' into local DB for {selected_media_type}. Media ID: {media_id}. Message: {msg}")
                        processed_count += 1
                        media_id_ingested = media_id # Store the ID
                    else:
                        logger.error(f"Failed to ingest '{item_result.input_ref}' into local DB for {selected_media_type}. Message: {msg}")
                        error_count += 1
                except Exception as e_ingest:
                    logger.error(f"Error ingesting item '{item_result.input_ref}' for {selected_media_type} into local DB: {e_ingest}", exc_info=True)
                    error_count += 1

                if media_id_ingested: # Only add to details if successfully ingested
                    successful_ingestions_details.append({
                        "input_ref": item_result.input_ref,
                        "title": item_result.metadata.get("title", "N/A") if item_result.metadata else "N/A",
                        "media_type": item_result.media_type,
                        "db_id": media_id_ingested
                    })
            else:
                logger.error(f"API processing error for '{item_result.input_ref}' ({selected_media_type}): {item_result.error}")
                error_count += 1

        summary_parts = [f"## TLDW API Request Successful ({selected_media_type.title()})\n\n"]
        # ... (rest of summary construction similar to before) ...
        if processed_count == 0 and error_count == 0 and not results_to_ingest:
             summary_parts.append("API request successful, but no items were provided or found for processing.\n")
        elif processed_count == 0 and error_count > 0:
            summary_parts.append(f"API request successful, but no new items were ingested due to errors.\n")
            summary_parts.append(f"- Successfully processed items by API: {processed_count}\n") # This might be confusing if API said success but ingest failed
            summary_parts.append(f"- Items with errors during API processing or local ingestion: {error_count}\n")
        else:
            summary_parts.append(f"- Successfully processed and ingested items: {processed_count}\n")
            summary_parts.append(f"- Items with errors during API processing or local ingestion: {error_count}\n\n")

        if error_count > 0:
            summary_parts.append("**Please check the application logs for details on any errors.**\n\n")

        if successful_ingestions_details:
            if len(successful_ingestions_details) <= 5:
                summary_parts.append("### Successfully Ingested Items:\n")
                for detail in successful_ingestions_details:
                    title_str = f" (Title: {detail['title']})" if detail['title'] != 'N/A' else ""
                    summary_parts.append(f"- **Input:** `{detail['input_ref']}`{title_str}\n") # Use backticks for input ref
                    summary_parts.append(f"  - **Type:** {detail['media_type']}, **DB ID:** {detail['db_id']}\n")
            else:
                summary_parts.append(f"Details for {len(successful_ingestions_details)} successfully ingested items are available in the logs.\n")
        elif processed_count > 0 : # Processed but no details (should not happen if logic is correct)
             summary_parts.append("Successfully processed items, but details are unavailable.\n")


        current_status_area.load_text("".join(summary_parts))
        current_status_area.display = True
        current_status_area.scroll_home(animate=False)

        notify_msg = f"{selected_media_type.title()} Ingestion: {processed_count} done, {error_count} errors."
        app.notify(notify_msg, severity="information" if error_count == 0 and processed_count > 0 else "warning", timeout=6)


    def on_worker_failure(error: Exception):
        try:
            current_loading_indicator = app.query_one(f"#tldw-api-loading-indicator-{selected_media_type}", LoadingIndicator)
            current_loading_indicator.display = False
            # current_submit_button = app.query_one(f"#tldw-api-submit-{selected_media_type}", Button)
            submit_button.disabled = False # submit_button is already defined from event.button
        except QueryError as e_ui:
            logger.error(f"UI component not found in on_worker_failure for {selected_media_type}: {e_ui}")

        logger.error(f"TLDW API request worker failed for {selected_media_type}: {error}", exc_info=True)

        error_message_parts = [f"## API Request Failed! ({selected_media_type.title()})\n\n"]
        brief_notify_message = f"{selected_media_type.title()} API Request Failed."
        if isinstance(error, APIResponseError):
            error_type = "API Error"
            error_message_parts.append(f"**Type:** API Error\n**Status Code:** {error.status_code}\n**Message:** `{str(error)}`\n")
            if error.detail:
                error_message_parts.append(f"**Details:**\n```\n{error.detail}\n```\n")
            brief_notify_message = f"{selected_media_type.title()} API Error {error.status_code}: {str(error)[:50]}"
            if error.response_data:
                try:
                    # Try to pretty-print if it's JSON, otherwise just str
                    response_data_str = json.dumps(error.response_data, indent=2)
                except (TypeError, ValueError):
                    response_data_str = str(error.response_data)
                error_message_parts.append(f"**Response Data:**\n```json\n{response_data_str}\n```\n")
            brief_notify_message = f"API Error {error.status_code}: {str(error)[:100]}"
        elif isinstance(error, AuthenticationError):
            error_type = "Authentication Error"
            error_message_parts.append(f"**Type:** {error_type}\n")
            error_message_parts.append(f"**Message:** `{str(error)}`\n")
            brief_notify_message = f"Auth Error: {str(error)[:100]}"
        elif isinstance(error, APIConnectionError):
            error_type = "Connection Error"
            error_message_parts.append(f"**Type:** {error_type}\n")
            error_message_parts.append(f"**Message:** `{str(error)}`\n")
            brief_notify_message = f"Connection Error: {str(error)[:100]}"
        elif isinstance(error, APIRequestError):
            error_type = "API Request Error"
            error_message_parts.append(f"**Type:** {error_type}\n")
            error_message_parts.append(f"**Message:** `{str(error)}`\n")
            brief_notify_message = f"Request Error: {str(error)[:100]}"
        else:
            error_type = "General Error"
            error_message_parts.append(f"**Type:** {error_type}\n")
            error_message_parts.append(f"**Message:** `{str(error)}`\n")
            brief_notify_message = f"Processing failed: {str(error)[:100]}"

        try:
            current_status_area = app.query_one(f"#tldw-api-status-area-{selected_media_type}", TextArea)
            current_status_area.clear()
            current_status_area.load_text("".join(error_message_parts))
            current_status_area.display = True
            current_status_area.scroll_home(animate=False)
        except QueryError:
            logger.error(f"Could not find status_area for {selected_media_type} to display error.")
            app.notify(f"Critical: Status area for {selected_media_type} not found. Error: {brief_notify_message}", severity="error", timeout=10)
            return

        app.notify(brief_notify_message, severity="error", timeout=8)

    # STORE THE CONTEXT
    app._last_tldw_api_request_context = {
        "request_model": request_model,
        "overwrite_db": overwrite_db,
    }

    app.run_worker(
        process_media_worker,
        name=f"tldw_api_processing_{selected_media_type}", # Unique worker name per tab
        group="api_calls",
        description=f"Processing {selected_media_type} media via TLDW API",
        exit_on_error=False
    )


async def _update_note_preview_display(app: 'TldwCli') -> None:
    """Updates the note preview area in the UI."""
    try:
        preview_area = app.query_one("#ingest-notes-preview-area", VerticalScroll)
        await preview_area.remove_children()

        if not app.parsed_notes_for_preview:
            await preview_area.mount(
                Static("Select files to see a preview, or no notes found.", id="ingest-notes-preview-placeholder"))
            return

        num_to_display = len(app.parsed_notes_for_preview)
        notes_to_show = app.parsed_notes_for_preview[:MAX_NOTE_PREVIEWS]

        for idx, note_data in enumerate(notes_to_show):
            title = note_data.get("title", f"Untitled Note {idx + 1}")
            content_preview = _truncate_text(note_data.get("content"), 200)

            md_content = f"""### {title}
{content_preview}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

"""
            if "error" in note_data:
                md_content = f"""### Error parsing {note_data.get("filename", "file")}
                {note_data["error"]}
                IGNORE_WHEN_COPYING_START
                content_copy
                download
                Use code with caution.
                Text
                IGNORE_WHEN_COPYING_END                
                """
                await preview_area.mount(Markdown(md_content, classes="prompt-preview-item"))

            if num_to_display > MAX_NOTE_PREVIEWS:
                    await preview_area.mount(
                        Static(f"...and {num_to_display - MAX_NOTE_PREVIEWS} more notes loaded (not shown)."))

    except QueryError as e:
        logger.error(f"UI component not found for note preview update: {e}")
        app.notify("Error updating note preview UI.", severity="error")
    except Exception as e:
        logger.error(f"Unexpected error updating note preview: {e}", exc_info=True)
        app.notify("Unexpected error during note preview update.", severity="error")


def _parse_single_note_file_for_preview(file_path: Path, app_ref: 'TldwCli', import_as_template: bool = False) -> List[Dict[str, Any]]:
    """
    Parses a single note file using the appropriate importer.
    Returns a list of note data dicts for preview.
    """
    logger.debug(f"Parsing note file for preview: {file_path}")
    preview_notes = []

    try:
        # Use the note importer registry to parse the file
        parsed_notes = note_importer_registry.parse_file(file_path, import_as_template=import_as_template)
        
        for note in parsed_notes:
            note_data = {
                "filename": file_path.name,
                "title": note.title,
                "content": note.content,
                "is_template": note.is_template,
                "template": note.template,
                "keywords": note.keywords
            }
            
            # Add type indicator to title for templates
            if note.is_template:
                note_data["title"] = f"[TEMPLATE] {note.title}"
            elif note.template:
                note_data["title"] = f"[{note.template}] {note.title}"
                
            preview_notes.append(note_data)
            
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        preview_notes.append({
            "filename": file_path.name, 
            "title": f"Error: {file_path.name}",
            "error": f"File not found: {e}"
        })
    except ValueError as e:
        logger.error(f"Error parsing {file_path.name}: {e}")
        preview_notes.append({
            "filename": file_path.name,
            "title": f"Error: {file_path.name}",
            "error": str(e)
        })
    except Exception as e:
        logger.error(f"Unexpected error parsing note file {file_path}: {e}", exc_info=True)
        preview_notes.append({
            "filename": file_path.name,
            "title": f"Error: {file_path.name}",
            "error": f"Unexpected error: {e}"
        })

    if not preview_notes:  # If parsing yielded nothing
        preview_notes.append({
            "filename": file_path.name,
            "title": file_path.name,
            "content": "No notes found in file."
        })

    return preview_notes

async def _handle_note_file_selected_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    """Callback for note file selection."""
    if selected_path:
        logger.info(f"Note file selected via dialog: {selected_path}")
        if selected_path in app.selected_note_files_for_import:
            app.notify(f"File '{selected_path.name}' is already in the note selection.", severity="warning")
            return

        app.selected_note_files_for_import.append(selected_path)
        app.last_note_import_dir = selected_path.parent

        try:
            list_view = app.query_one("#ingest-notes-selected-files-list", ListView)
            placeholder_exists = False
            if list_view.children:
                first_child = list_view.children[0]
                if isinstance(first_child, ListItem) and first_child.children:
                    first_label = first_child.children[0]
                    if isinstance(first_label, Label) and str(first_label.renderable).strip() == "No files selected.":
                        placeholder_exists = True
            if placeholder_exists:
                await list_view.clear()
            await list_view.append(ListItem(Label(str(selected_path))))
        except QueryError:
            logger.error("Could not find #ingest-notes-selected-files-list ListView to update.")

        # Check if importing as templates
        import_as_template = False
        try:
            radio_set = app.query_one("#ingest-notes-import-type", RadioSet)
            import_as_template = radio_set.pressed_index == 1  # Second option is "Import as Templates"
        except QueryError:
            logger.debug("Could not find import type RadioSet, defaulting to notes")
        
        parsed_notes_from_file = _parse_single_note_file_for_preview(selected_path, app, import_as_template=import_as_template)
        app.parsed_notes_for_preview.extend(parsed_notes_from_file)

        await _update_note_preview_display(app)
    else:
        logger.info("Note file selection cancelled.")
        app.notify("File selection cancelled.")


# --- Notes Ingest Handlers (NEW) ---
async def handle_ingest_notes_select_file_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger.debug("Select Notes File(s) button pressed. Opening file dialog.")
    current_dir = app.last_note_import_dir or Path(".")

    def post_file_open_action(selected_file_path: Optional[Path]) -> None:
        """This function matches the expected callback signature for push_screen more directly."""
        # It's okay for this outer function to be synchronous if all it does
        # is schedule an async task via call_after_refresh.
        if selected_file_path is not None: # Or however you want to handle None path
            # The lambda passed to call_after_refresh captures selected_file_path
            app.call_after_refresh(lambda: _handle_note_file_selected_callback(app, selected_file_path))
        else:
            # Handle the case where selection was cancelled (path is None)
            app.call_after_refresh(lambda: _handle_note_file_selected_callback(app, None))
    # The screen you're pushing
    file_open_screen = FileOpen(
        location=str(current_dir),
        title="Select Notes File (.json)",
        filters=NOTE_FILE_FILTERS
    )
    # Push the screen with the defined callback
    # await app.push_screen(file_open_screen, post_file_open_action) # This should work
    # If you need to call an async method from a sync context (like a button press handler that isn't async itself)
    # and push_screen itself needs to be awaited, then the button handler must be async.
    # Your handle_ingest_notes_select_file_button_pressed is already async, so this is fine:
    await app.push_screen(file_open_screen, post_file_open_action)


async def handle_ingest_notes_clear_files_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles 'Clear Selection' for note import."""
    logger.info("Clearing selected note files and preview.")
    app.selected_note_files_for_import.clear()
    app.parsed_notes_for_preview.clear()

    try:
        selected_list_view = app.query_one("#ingest-notes-selected-files-list", ListView)
        await selected_list_view.clear()
        await selected_list_view.append(ListItem(Label("No files selected.")))

        preview_area = app.query_one("#ingest-notes-preview-area", VerticalScroll)
        await preview_area.remove_children()
        await preview_area.mount(Static("Select files to see a preview.", id="ingest-notes-preview-placeholder"))

        status_area = app.query_one("#ingest-notes-import-status-area", TextArea)
        status_area.clear()
        app.notify("Note selection and preview cleared.")
    except QueryError as e:
        logger.error(f"UI component not found for clearing note selection: {e}")
        app.notify("Error clearing note UI.", severity="error")


async def handle_ingest_notes_import_now_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles 'Import Selected Notes Now' button press."""
    logger.info("Import Selected Note Files Now button pressed.")

    if not app.selected_note_files_for_import:
        app.notify("No note files selected to import.", severity="warning")
        return

    # Check if importing as templates or notes
    try:
        import_type_radio = app.query_one("#import-as-templates-radio", RadioButton)
        import_as_templates = import_type_radio.value
    except QueryError:
        # Default to importing as notes if radio button not found
        import_as_templates = False

    if not import_as_templates and not app.notes_service:
        msg = "Notes database service is not initialized. Cannot import notes."
        app.notify(msg, severity="error", timeout=7)
        logger.error(msg + " Aborting note import.")
        return

    try:
        # Use query_one to get the widget directly or raise QueryError
        status_area = app.query_one("#ingest-notes-import-status-area", TextArea)
    except QueryError:
        logger.error("Could not find #ingest-notes-import-status-area TextArea.")
        app.notify("Status display area not found.", severity="error")
        return

    status_area.text = ""  # Clear the TextArea
    status_area.text = f"Starting {'template' if import_as_templates else 'note'} import process...\n"  # Set initial text
    app.notify(f"Importing {'templates' if import_as_templates else 'notes'}...")

    user_id = app.notes_user_id

    async def import_worker_notes():
        results = []
        
        if import_as_templates:
            # Import as templates
            import json
            from pathlib import Path
            
            # Load existing templates
            user_config_dir = Path.home() / ".config" / "tldw_cli"
            user_templates_path = user_config_dir / "note_templates.json"
            
            # Create directory if needed
            user_config_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing templates if any
            templates = {}
            if user_templates_path.exists():
                try:
                    with open(user_templates_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        templates = data.get('templates', {})
                except Exception as e:
                    logger.error(f"Error loading existing templates: {e}")
            
            # Process each file
            for file_path in app.selected_note_files_for_import:
                notes_in_file = _parse_single_note_file_for_preview(file_path, app, import_as_template=True)
                for note_data in notes_in_file:
                    if "error" in note_data or not note_data.get("title") or not note_data.get("content"):
                        results.append({
                            "file_path": str(file_path),
                            "note_title": note_data.get("title", file_path.stem),
                            "status": "failure",
                            "message": note_data.get("error", "Missing title or content.")
                        })
                        continue
                    
                    try:
                        # Generate a unique template key from the title
                        base_key = note_data["title"].lower().replace(" ", "_").replace("-", "_")
                        # Remove non-alphanumeric characters except underscores
                        base_key = ''.join(c for c in base_key if c.isalnum() or c == '_')
                        
                        # Ensure unique key
                        key = base_key
                        counter = 1
                        while key in templates:
                            key = f"{base_key}_{counter}"
                            counter += 1
                        
                        # Create template entry
                        templates[key] = {
                            "title": note_data["title"],
                            "content": note_data["content"],
                            "keywords": note_data.get("keywords", ""),
                            "description": f"Imported template: {note_data['title']}"
                        }
                        
                        results.append({
                            "file_path": str(file_path),
                            "note_title": note_data["title"],
                            "status": "success",
                            "message": f"Template imported successfully. Key: {key}",
                            "template_key": key
                        })
                    except Exception as e:
                        logger.error(f"Error importing template '{note_data['title']}' from {file_path}: {e}",
                                     exc_info=True)
                        results.append({
                            "file_path": str(file_path),
                            "note_title": note_data["title"],
                            "status": "failure",
                            "message": f"Error: {type(e).__name__}"
                        })
            
            # Save all templates
            if any(r["status"] == "success" for r in results):
                try:
                    output = {"templates": templates}
                    with open(user_templates_path, 'w', encoding='utf-8') as f:
                        json.dump(output, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved {len(templates)} templates to {user_templates_path}")
                except Exception as e:
                    logger.error(f"Error saving templates file: {e}")
                    # Update all success results to failure
                    for r in results:
                        if r["status"] == "success":
                            r["status"] = "failure"
                            r["message"] = f"Template imported but failed to save: {e}"
        
        else:
            # Import as notes (existing logic)
            for file_path in app.selected_note_files_for_import:
                notes_in_file = _parse_single_note_file_for_preview(file_path, app)
                for note_data in notes_in_file:
                    if "error" in note_data or not note_data.get("title") or not note_data.get("content"):
                        results.append({
                            "file_path": str(file_path),
                            "note_title": note_data.get("title", file_path.stem),
                            "status": "failure",
                            "message": note_data.get("error", "Missing title or content.")
                        })
                        continue
                    try:
                        note_id = app.notes_service.add_note(
                            user_id=user_id,
                            title=note_data["title"],
                            content=note_data["content"]
                        )
                        results.append({
                            "file_path": str(file_path),
                            "note_title": note_data["title"],
                            "status": "success",
                            "message": f"Note imported successfully. ID: {note_id}",
                            "note_id": note_id
                        })
                    except (ChaChaConflictError, CharactersRAGDBError, ValueError) as e:
                        logger.error(f"Error importing note '{note_data['title']}' from {file_path}: {e}", exc_info=True)
                        results.append({
                            "file_path": str(file_path),
                            "note_title": note_data["title"],
                            "status": "failure",
                            "message": f"DB/Input error: {type(e).__name__} - {str(e)[:100]}"
                        })
                    except Exception as e:
                        logger.error(f"Unexpected error importing note '{note_data['title']}' from {file_path}: {e}",
                                     exc_info=True)
                        results.append({
                            "file_path": str(file_path),
                            "note_title": note_data["title"],
                            "status": "failure",
                            "message": f"Unexpected error: {type(e).__name__}"
                        })
        return results

    def on_import_success_notes(results: List[Dict[str, Any]]):
        import_type = "template" if import_as_templates else "note"
        log_text_parts = [f"{import_type.capitalize()} import process finished.\n\nResults:\n"]  # Renamed to avoid conflict
        successful_imports = 0
        failed_imports = 0
        for res in results:
            status = res.get("status", "unknown")
            file_path_str = res.get("file_path", "N/A")
            note_title = res.get("note_title", "N/A")
            message = res.get("message", "")

            log_text_parts.append(f"File: {Path(file_path_str).name} ({import_type.capitalize()}: '{note_title}')\n")
            log_text_parts.append(f"  Status: {status.upper()}\n")
            if message:
                log_text_parts.append(f"  Message: {message}\n")
            log_text_parts.append("-" * 30 + "\n")

            if status == "success":
                successful_imports += 1
            else:
                failed_imports += 1

        summary = f"\nSummary: {successful_imports} {import_type}s imported, {failed_imports} failed."
        log_text_parts.append(summary)

        try:
            status_area_cb = app.query_one("#ingest-notes-import-status-area", TextArea)
            status_area_cb.load_text("".join(log_text_parts))
        except QueryError:
            logger.error("Failed to find #ingest-notes-import-status-area in on_import_success_notes.")

        app.notify(f"{import_type.capitalize()} import finished. Success: {successful_imports}, Failed: {failed_imports}", timeout=8)
        logger.info(summary)

        if import_as_templates:
            # For templates, we need to reload the templates in the notes event handler
            app.notify("Templates will be available after restarting the application.", severity="information", timeout=10)
        else:
            #app.call_later(load_and_display_notes_handler, app)
            app.call_later(app.refresh_notes_tab_after_ingest)
            try:
                # Make sure to query the collapsible before creating the Toggled event instance
                chat_notes_collapsible_widget = app.query_one("#chat-notes-collapsible", Collapsible)
                app.call_later(app.on_chat_notes_collapsible_toggle, Collapsible.Toggled(chat_notes_collapsible_widget))
            except QueryError:
                logger.error("Failed to find #chat-notes-collapsible widget for refresh after note import.")

    def on_import_failure_notes(error: Exception):
        import_type = "template" if import_as_templates else "note"
        logger.error(f"{import_type.capitalize()} import worker failed critically: {error}", exc_info=True)
        try:
            status_area_cb_fail = app.query_one("#ingest-notes-import-status-area", TextArea)
            current_text = status_area_cb_fail.text
            status_area_cb_fail.load_text(
                current_text + f"\n{import_type.capitalize()} import process failed critically: {error}\nCheck logs.\n")
        except QueryError:
            logger.error("Failed to find #ingest-notes-import-status-area in on_import_failure_notes.")
        app.notify(f"{import_type.capitalize()} import CRITICALLY failed: {error}", severity="error", timeout=10)

    app.run_worker(
        import_worker_notes,
        name="note_import_worker",
        group="file_operations",
        description="Importing selected note files."
    )

async def handle_tldw_api_worker_failure(app: 'TldwCli', event: 'Worker.StateChanged'):
    """Handles the failure of a TLDW API worker and updates the UI."""
    worker_name = event.worker.name or ""
    media_type = worker_name.replace("tldw_api_processing_", "")
    error = event.worker.error

    logger.error(f"TLDW API request worker failed for {media_type}: {error}", exc_info=True)

    try:
        loading_indicator = app.query_one(f"#tldw-api-loading-indicator-{media_type}", LoadingIndicator)
        submit_button = app.query_one(f"#tldw-api-submit-{media_type}", Button)
        status_area = app.query_one(f"#tldw-api-status-area-{media_type}", TextArea)

        loading_indicator.display = False
        submit_button.disabled = False
    except QueryError as e_ui:
        logger.error(f"UI component not found in on_worker_failure for {media_type}: {e_ui}")
        return

    error_message_parts = [f"## API Request Failed! ({media_type.title()})\n\n"]
    brief_notify_message = f"{media_type.title()} API Request Failed."

    # This logic is copied from your original local on_worker_failure function
    if isinstance(error, APIConnectionError):
        error_type = "Connection Error"
        error_message_parts.append(f"**Type:** {error_type}\n")
        error_message_parts.append(f"**Message:** `{str(error)}`\n")
        brief_notify_message = f"Connection Error: {str(error)[:100]}"
    elif isinstance(error, APIResponseError):
        error_type = "API Error"
        error_message_parts.append(f"**Type:** API Error\n**Status Code:** {error.status_code}\n**Message:** `{str(error)}`\n")
        brief_notify_message = f"API Error {error.status_code}: {str(error)[:100]}"
    # ... add other specific error types from your original function if needed ...
    else:
        error_type = "General Error"
        error_message_parts.append(f"**Type:** {type(error).__name__}\n")
        error_message_parts.append(f"**Message:** `{str(error)}`\n")
        brief_notify_message = f"Processing failed: {str(error)[:100]}"

    status_area.clear()
    status_area.load_text("".join(error_message_parts))
    status_area.display = True
    app.notify(brief_notify_message, severity="error", timeout=8)


# async def handle_tldw_api_worker_success(app: 'TldwCli', event: 'Worker.StateChanged'):
#     """Handles the success of a TLDW API worker and ingests the results."""
#     # This function would contain the logic from your original 'on_worker_success'
#     # It needs to be made async and re-query UI elements.
#     # The logic is complex, so for brevity, I'll show the skeleton.
#     # The key is that you have access to `app` and `event` (which has the result).
#
#     worker_name = event.worker.name or ""
#     media_type = worker_name.replace("tldw_api_processing_", "")
#     response_data = event.worker.result
#
#     logger.info(f"TLDW API worker for {media_type} succeeded. Processing results.")
#
#     try:
#         # Reset UI state (disable loading, enable button)
#         app.query_one(f"#tldw-api-loading-indicator-{media_type}", LoadingIndicator).display = False
#         app.query_one(f"#tldw-api-submit-{media_type}", Button).disabled = False
#         status_area = app.query_one(f"#tldw-api-status-area-{media_type}", TextArea)
#         status_area.clear()
#
#         # The rest of your success logic goes here. For example:
#         if not app.media_db:
#              # ... handle missing db ...
#              return
#
#         # ... process response_data, ingest to db, build summary_parts ...
#         # (This logic is already well-written in your original `on_worker_success`)
#         # You will need to retrieve `overwrite_db` and `request_model` if they
#         # are needed for the success logic, potentially by storing them on the app
#         # instance temporarily before starting the worker.
#         status_area.load_text("## Success!\n\n(Full success logic from original function would go here)")
#         status_area.display = True
#         app.notify(f"{media_type.title()} processing complete.", severity="information")
#
#     except QueryError as e_ui:
#         logger.error(f"UI component not found in on_worker_success for {media_type}: {e_ui}")
#     except Exception as e:
#         logger.error(f"Error handling TLDW API worker success for {media_type}: {e}", exc_info=True)
#         app.notify("Error processing successful API response.", severity="error")
async def handle_tldw_api_worker_success(app: 'TldwCli', event: 'Worker.StateChanged'):
    """Handles the success of a TLDW API worker and ingests the results."""
    worker_name = event.worker.name or ""
    media_type = worker_name.replace("tldw_api_processing_", "")
    response_data = event.worker.result

    logger.info(f"TLDW API worker for {media_type} succeeded. Processing results.")

    try:
        # Reset UI state (disable loading, enable button)
        app.query_one(f"#tldw-api-loading-indicator-{media_type}", LoadingIndicator).display = False
        app.query_one(f"#tldw-api-submit-{media_type}", Button).disabled = False
        status_area = app.query_one(f"#tldw-api-status-area-{media_type}", TextArea)
        status_area.clear()

    except QueryError as e_ui:
        logger.error(f"UI component not found in on_worker_success for {media_type}: {e_ui}")
        return
    except Exception as e:
        logger.error(f"Error resetting UI state in worker success handler: {e}", exc_info=True)
        return

    # --- Pre-flight Checks and Context Retrieval ---
    if not app.media_db:
        logger.error("Media_DB_v2 not initialized. Cannot ingest API results.")
        app.notify("Error: Local media database not available.", severity="error")
        status_area.load_text("## Error\n\nLocal media database is not available. Cannot save results.")
        return

    # Retrieve the context we saved before starting the worker
    request_context = getattr(app, "_last_tldw_api_request_context", {})
    request_model = request_context.get("request_model")
    overwrite_db = request_context.get("overwrite_db", False)

    if not request_model:
        logger.error("Could not retrieve request_model from app context. Cannot properly ingest results.")
        status_area.load_text("## Internal Error\n\nCould not retrieve original request context. Ingestion aborted.")
        return

    # --- Data Processing and Ingestion ---
    processed_count = 0
    error_count = 0
    successful_ingestions_details = []
    results_to_ingest: List[MediaItemProcessResult] = []

    # Normalize different response types into a single list of MediaItemProcessResult
    if isinstance(response_data, BatchMediaProcessResponse):
        results_to_ingest = response_data.results
    elif isinstance(response_data, list) and all(isinstance(item, ProcessedMediaWikiPage) for item in response_data):
        for mw_page in response_data:
            if mw_page.status == "Error":
                error_count += 1
                logger.error(f"MediaWiki page '{mw_page.title}' processing error: {mw_page.error_message}")
                continue
            # Adapt ProcessedMediaWikiPage to the common result structure
            results_to_ingest.append(MediaItemProcessResult(
                status="Success",
                input_ref=mw_page.input_ref or mw_page.title,
                processing_source=mw_page.title,
                media_type="mediawiki_page",
                metadata={"title": mw_page.title, "page_id": mw_page.page_id, "namespace": mw_page.namespace},
                content=mw_page.content,
                chunks=[{"text": chunk.get("text", ""), "metadata": chunk.get("metadata", {})} for chunk in mw_page.chunks] if mw_page.chunks else None,
            ))
    elif isinstance(response_data, BatchProcessXMLResponse):
        for xml_item in response_data.results:
            if xml_item.status == "Error":
                error_count +=1; continue
            results_to_ingest.append(MediaItemProcessResult(
                status="Success", input_ref=xml_item.input_ref, media_type="xml",
                metadata={"title": xml_item.title, "author": xml_item.author, "keywords": xml_item.keywords},
                content=xml_item.content, analysis=xml_item.summary,
            ))
    else:
        logger.error(f"Unexpected TLDW API response data type for {media_type}: {type(response_data)}.")
        status_area.load_text(f"## API Request Processed\n\nUnexpected response format. Raw response logged.")
        app.notify("Error: Received unexpected data format from API.", severity="error")
        return

    # --- Ingestion Loop ---
    for item_result in results_to_ingest:
        if item_result.status == "Success":
            try:
                # Prepare chunks for database insertion
                unvectorized_chunks_to_save = []
                if item_result.chunks:
                    for chunk_item in item_result.chunks:
                        if isinstance(chunk_item, dict) and "text" in chunk_item:
                            unvectorized_chunks_to_save.append({
                                "text": chunk_item.get("text"), "metadata": chunk_item.get("metadata", {})
                            })
                        elif isinstance(chunk_item, str):
                            unvectorized_chunks_to_save.append({"text": chunk_item, "metadata": {}})

                # Call the DB function with data from both the API response and original request
                media_id, _, msg = app.media_db.add_media_with_keywords(
                    url=item_result.input_ref,
                    title=item_result.metadata.get("title", item_result.input_ref),
                    media_type=item_result.media_type,
                    content=item_result.content or item_result.transcript,
                    keywords=item_result.metadata.get("keywords", []) or request_model.keywords,
                    prompt=request_model.custom_prompt,
                    analysis_content=item_result.analysis or item_result.summary,
                    author=item_result.metadata.get("author") or request_model.author,
                    overwrite=overwrite_db,
                    chunks=unvectorized_chunks_to_save
                )

                if media_id:
                    logger.info(f"Successfully ingested '{item_result.input_ref}' into local DB. Media ID: {media_id}. Msg: {msg}")
                    processed_count += 1
                    successful_ingestions_details.append({
                        "input_ref": item_result.input_ref,
                        "title": item_result.metadata.get("title", "N/A"),
                        "media_type": item_result.media_type,
                        "db_id": media_id
                    })
                else:
                    logger.error(f"Failed to ingest '{item_result.input_ref}' into local DB. Message: {msg}")
                    error_count += 1

            except Exception as e_ingest:
                logger.error(f"Error ingesting item '{item_result.input_ref}' into local DB: {e_ingest}", exc_info=True)
                error_count += 1
        else:
            logger.error(f"API processing error for '{item_result.input_ref}': {item_result.error}")
            error_count += 1

    # --- Build and Display Summary ---
    summary_parts = [f"## TLDW API Request Successful ({media_type.title()})\n\n"]
    if not results_to_ingest and error_count == 0:
        summary_parts.append("API request successful, but no items were provided or found for processing.\n")
    else:
        summary_parts.append(f"- **Successfully Processed & Ingested:** {processed_count}\n")
        summary_parts.append(f"- **Errors (API or DB):** {error_count}\n\n")

    if error_count > 0:
        summary_parts.append("**Please check the application logs for details on any errors.**\n\n")

    if successful_ingestions_details:
        summary_parts.append("### Ingested Items:\n")
        for detail in successful_ingestions_details[:10]: # Show max 10 details
            title_str = f" (Title: `{detail['title']}`)" if detail['title'] != 'N/A' else ""
            summary_parts.append(f"- **Input:** `{detail['input_ref']}`{title_str}\n")
            summary_parts.append(f"  - **Type:** {detail['media_type']}, **DB ID:** {detail['db_id']}\n")
        if len(successful_ingestions_details) > 10:
            summary_parts.append(f"\n...and {len(successful_ingestions_details) - 10} more items.")

    status_area.load_text("".join(summary_parts))
    status_area.display = True
    status_area.scroll_home(animate=False)

    notify_msg = f"{media_type.title()} Ingestion: {processed_count} done, {error_count} errors."
    app.notify(notify_msg, severity="information" if error_count == 0 else "warning", timeout=7)

async def handle_ingest_local_web_button_pressed(app: 'TldwCli', event: 'Button.Pressed') -> None:
    """Handle local web article button presses by delegating to IngestWindow methods."""
    button_id = event.button.id
    try:
        ingest_window = app.query_one("#ingest-window", IngestWindow)
        
        if button_id == "ingest-local-web-clear-urls":
            await ingest_window._handle_clear_urls()
        elif button_id == "ingest-local-web-import-urls":
            await ingest_window._handle_import_urls_from_file()
        elif button_id == "ingest-local-web-remove-duplicates":
            await ingest_window._handle_remove_duplicate_urls()
        elif button_id == "ingest-local-web-process":
            await ingest_window.handle_local_web_article_process()
        elif button_id == "ingest-local-web-stop":
            await ingest_window._handle_stop_web_scraping()
        elif button_id == "ingest-local-web-retry":
            await ingest_window._handle_retry_failed_urls()
    except QueryError:
        logger.error(f"Could not find IngestWindow to handle button {button_id}")
    except Exception as e:
        logger.error(f"Error handling local web button {button_id}: {e}")
        app.notify(f"Error: {str(e)}", severity="error")

async def handle_local_pdf_ebook_submit_button_pressed(app: 'TldwCli', event: 'Button.Pressed') -> None:
    """Handle local PDF and ebook processing button presses."""
    button_id = event.button.id
    media_type = "pdf" if button_id == "local-submit-pdf" else "ebook"
    
    try:
        ingest_window = app.query_one("#ingest-window", IngestWindow)
        
        if media_type == "pdf":
            await ingest_window.handle_local_pdf_process()
        else:  # ebook
            await ingest_window.handle_local_ebook_process()
            
    except QueryError:
        logger.error(f"Could not find IngestWindow to handle button {button_id}")
    except Exception as e:
        logger.error(f"Error handling local {media_type} button: {e}")
        app.notify(f"Error: {str(e)}", severity="error")

# --- Button Handler Map ---
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
}


#
# End of tldw_chatbook/Event_Handlers/ingest_events.py
#######################################################################################################################