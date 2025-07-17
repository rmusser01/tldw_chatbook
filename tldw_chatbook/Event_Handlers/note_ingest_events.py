# tldw_chatbook/Event_Handlers/note_ingest_events.py
#
# Note ingestion event handlers and related functions
#
# Imports
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Any, Dict

# 3rd-party Libraries
from loguru import logger
from textual.widgets import Button, Label, ListItem, ListView, Static, Markdown, TextArea, RadioButton, RadioSet, Collapsible
from textual.css.query import QueryError
from textual.containers import VerticalScroll

# Local Imports
from ..DB.ChaChaNotes_DB import ConflictError as ChaChaConflictError, CharactersRAGDBError
from ..Utils.note_importers import note_importer_registry, ParsedNote
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
from .ingest_utils import (
    NOTE_FILE_FILTERS,
    MAX_NOTE_PREVIEWS,
    _truncate_text
)

if TYPE_CHECKING:
    from ..app import TldwCli

# --- Note Preview Functions ---
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


# --- Notes Ingest Handlers ---
async def handle_ingest_notes_select_file_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger.debug("Select Notes File(s) button pressed. Opening file dialog.")
    current_dir = app.last_note_import_dir or Path(".")

    def post_file_open_action(selected_file_path: Optional[Path]) -> None:
        """This function matches the expected callback signature for push_screen more directly."""
        # It's okay for this outer function to be synchronous if all it does
        # is schedule an async task via call_after_refresh.
        if selected_file_path is not None:  # Or however you want to handle None path
            # The lambda passed to call_after_refresh captures selected_file_path
            app.call_after_refresh(lambda: _handle_note_file_selected_callback(app, selected_file_path))
        else:
            # Handle the case where selection was cancelled (path is None)
            app.call_after_refresh(lambda: _handle_note_file_selected_callback(app, None))
    # The screen you're pushing
    file_open_screen = FileOpen(location=str(current_dir, context="note_ingest"),
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