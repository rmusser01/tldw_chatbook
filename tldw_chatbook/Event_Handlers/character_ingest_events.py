# tldw_chatbook/Event_Handlers/character_ingest_events.py
#
# Character ingestion event handlers and related functions
#
# Imports
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Any, Dict

# 3rd-party Libraries
from loguru import logger
from textual.widgets import Button, Label, ListItem, ListView, Static, Markdown, TextArea
from textual.css.query import QueryError
from textual.containers import VerticalScroll

# Local Imports
from ..Character_Chat import Character_Chat_Lib as ccl
from ..DB.ChaChaNotes_DB import ConflictError as ChaChaConflictError
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
from .ingest_utils import (
    CHARACTER_FILE_FILTERS, 
    MAX_CHARACTER_PREVIEWS,
    _truncate_text
)

if TYPE_CHECKING:
    from ..app import TldwCli

# --- Character Preview Functions ---
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


# --- Character Ingest Handlers ---
async def handle_ingest_characters_select_file_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Select Character File(s)' button press."""
    logger.debug("Select Character File(s) button pressed. Opening file dialog.")
    current_dir = app.last_character_import_dir or Path(".")  # Use new state var

    await app.push_screen(
        FileOpen(location=str(current_dir, context="character_ingest"),
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

    # Import these for worker-related functionality
    import tldw_chatbook.Event_Handlers.conv_char_events as ccp_handlers
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import populate_chat_conversation_character_filter_select

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