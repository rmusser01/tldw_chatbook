# notes_events.py
# Description:
#
# Imports
import logging
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional, Dict, Tuple, TYPE_CHECKING
#
# 3rd-Party Imports
from loguru import logger
from textual.widgets import Input, ListView, TextArea, Label, Button, ListItem, Select, Static
from textual.css.query import QueryError  # For try-except
import yaml
#
# Local Imports
from ..Widgets.notes_sidebar_right import NotesSidebarRight
from ..DB.ChaChaNotes_DB import ConflictError, CharactersRAGDBError
from ..Widgets.notes_sidebar_left import NotesSidebarLeft
from ..Third_Party.textual_fspicker import FileOpen, FileSave, Filters
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Functions:

async def handle_notes_tab_sidebar_toggle(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles sidebar toggles specific to the Notes tab."""
    button_id = event.button.id
    logger_instance = getattr(app, 'loguru_logger', logger) # Use app's logger or global
    if button_id == "toggle-notes-sidebar-left":
        app.notes_sidebar_left_collapsed = not app.notes_sidebar_left_collapsed
        logger_instance.debug("Notes left sidebar now %s", "collapsed" if app.notes_sidebar_left_collapsed else "expanded")
    elif button_id == "toggle-notes-sidebar-right":
        app.notes_sidebar_right_collapsed = not app.notes_sidebar_right_collapsed
        logger_instance.debug("Notes right sidebar now %s", "collapsed" if app.notes_sidebar_right_collapsed else "expanded")
    else:
        logger_instance.warning(f"Unhandled sidebar toggle button ID '{button_id}' in Notes tab handler.")


########################################################################################################################
#
# Helper Functions (specific to Notes tab logic, moved from app.py)
#
########################################################################################################################

async def save_current_note_handler(app: 'TldwCli') -> bool:
    """Saves the currently selected note's title, content, and keywords to the database."""
    logger = getattr(app, 'loguru_logger', logging) # Use app's logger or global
    if not app.notes_service:
        logger.error("Notes service not available. Cannot save note.")
        app.notify("Notes service unavailable.", severity="error")
        return False
    if not app.current_selected_note_id or app.current_selected_note_version is None:
        logger.warning("No note selected or version missing. Cannot save.")
        app.notify("No note selected to save, or version is missing.", severity="warning")
        return False

    try:
        editor = app.query_one("#notes-editor-area", TextArea)
        keywords_area = app.query_one("#notes-keywords-area", TextArea)

        try:
            # Query for the NotesSidebarRight widget instance first
            notes_sidebar_right_instance = app.query_one(NotesSidebarRight)  # You'll need to import NotesSidebarRight
            # Then query for the input within that sidebar instance
            title_input = notes_sidebar_right_instance.query_one("#notes-title-input", Input)
        except QueryError as e_query:
            logger.error(f"UI component (NotesSidebarRight or #notes-title-input) not found: {e_query}", exc_info=True)
            app.notify("UI error: Title input not found.", severity="error")
            return False

        current_content_from_ui = editor.text
        current_title_from_ui = title_input.value.strip()

        logger.info(
            f"Attempting to save note ID: {app.current_selected_note_id}, Version: {app.current_selected_note_version}")

        # Save note content and title
        success = app.notes_service.update_note(
            user_id=app.notes_user_id,
            note_id=app.current_selected_note_id,
            update_data={'title': current_title_from_ui, 'content': current_content_from_ui},
            expected_version=app.current_selected_note_version
        )
        
        if success:
            logger.info(f"Note {app.current_selected_note_id} saved successfully via notes_service.")
            
            # Now save keywords
            input_keyword_texts = {kw.strip().lower() for kw in keywords_area.text.split(',') if kw.strip()}
            logger.info(f"Saving keywords for note {app.current_selected_note_id}: {input_keyword_texts}")
            
            # Get existing keywords
            existing_linked_keywords_data = app.notes_service.get_keywords_for_note(
                user_id=app.notes_user_id, note_id=app.current_selected_note_id
            )
            existing_linked_keyword_map = {kw['keyword'].lower(): kw['id'] for kw in existing_linked_keywords_data}
            
            # Add/Link new keywords
            for kw_text_to_add in input_keyword_texts:
                if kw_text_to_add not in existing_linked_keyword_map:
                    keyword_detail = app.notes_service.get_keyword_by_text(app.notes_user_id, kw_text_to_add)
                    kw_id_to_link = None
                    if not keyword_detail:
                        new_kw_id = app.notes_service.add_keyword(app.notes_user_id, kw_text_to_add)
                        if new_kw_id is not None: 
                            kw_id_to_link = new_kw_id
                    else:
                        kw_id_to_link = keyword_detail['id']
                    
                    if kw_id_to_link:
                        app.notes_service.link_note_to_keyword(
                            user_id=app.notes_user_id, note_id=app.current_selected_note_id, keyword_id=kw_id_to_link
                        )
            
            # Unlink removed keywords
            for existing_kw_text, existing_kw_id in existing_linked_keyword_map.items():
                if existing_kw_text not in input_keyword_texts:
                    app.notes_service.unlink_note_from_keyword(
                        user_id=app.notes_user_id, note_id=app.current_selected_note_id, keyword_id=existing_kw_id
                    )
            
            # Get updated note details
            updated_note_details = app.notes_service.get_note_by_id(
                user_id=app.notes_user_id,
                note_id=app.current_selected_note_id
            )
            if updated_note_details:
                app.current_selected_note_version = updated_note_details.get('version')
                app.current_selected_note_title = updated_note_details.get('title')
                app.current_selected_note_content = updated_note_details.get('content')
                title_input.value = app.current_selected_note_title or "" # Update UI from DB confirmed state
                
                # Refresh keywords display with canonical casing
                refreshed_keywords_data = app.notes_service.get_keywords_for_note(
                    user_id=app.notes_user_id, note_id=app.current_selected_note_id
                )
                keywords_area.text = ", ".join([kw['keyword'] for kw in refreshed_keywords_data]) if refreshed_keywords_data else ""
            else:
                logger.warning(f"Note {app.current_selected_note_id} not found after presumably successful save.")
                app.notify("Note saved, but failed to refresh details.", severity="warning")

            await load_and_display_notes_handler(app)  # Refresh list in left sidebar
            app.notify("Note and keywords saved!", severity="information")
            # Reset unsaved changes flag
            app.notes_unsaved_changes = False
            return True
        else:
            logger.warning(
                f"notes_service.update_note for {app.current_selected_note_id} returned False without error.")
            app.notify("Failed to save note (unknown reason).", severity="error")
            return False

    except ConflictError as e_conflict:
        logger.error(f"Conflict saving note {app.current_selected_note_id}: {e_conflict}", exc_info=True)
        app.notify(f"Save conflict: {e_conflict}. Please reload the note.", severity="error")
        return False
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error saving note {app.current_selected_note_id}: {e_db}", exc_info=True)
        app.notify("Error saving note to database.", severity="error")
        return False
    except QueryError as e_query:
        logger.error(f"UI component not found while saving note: {e_query}", exc_info=True)
        app.notify("UI error while saving note.", severity="error")
        return False
    except Exception as e_unexp:
        logger.error(f"Unexpected error saving note {app.current_selected_note_id}: {e_unexp}", exc_info=True)
        app.notify("Unexpected error saving note.", severity="error")
        return False


async def load_and_display_notes_handler(app: 'TldwCli') -> None:
    """Loads notes from the database and populates the left sidebar list."""
    logger = getattr(app, 'loguru_logger', logging)
    sidebar_left_instance: Optional['NotesSidebarLeft'] = None
    try:
        sidebar_left_instance = app.query_one("#notes-sidebar-left", NotesSidebarLeft)
    except QueryError:
        logger.error("Failed to find #notes-sidebar-left to populate notes.")
        return # Cannot proceed if sidebar isn't there

    if not app.notes_service:
        logger.error("Notes service not available, cannot load notes.")
        try:
            # Get the ListView within the sidebar to display the error
            list_view_in_sidebar = sidebar_left_instance.query_one("#notes-list-view", ListView)
            await list_view_in_sidebar.clear()
            await list_view_in_sidebar.mount(ListItem(Label("Notes service unavailable.")))
        except QueryError:
            logger.error("Failed to find #notes-list-view within #notes-sidebar-left to show service error.")
        return

    try:
        notes_list_data = app.notes_service.list_notes(user_id=app.notes_user_id, limit=200)
        
        # Sort notes based on current settings
        if hasattr(app, 'notes_sort_by') and hasattr(app, 'notes_sort_ascending'):
            if app.notes_sort_by == "title":
                notes_list_data.sort(key=lambda n: (n.get('title', '') or '').lower(), reverse=not app.notes_sort_ascending)
            elif app.notes_sort_by == "date_modified":
                notes_list_data.sort(key=lambda n: n.get('updated_at', ''), reverse=not app.notes_sort_ascending)
            else:  # date_created (default)
                notes_list_data.sort(key=lambda n: n.get('created_at', ''), reverse=not app.notes_sort_ascending)
        
        # Update note count in sidebar title
        try:
            sidebar_title = sidebar_left_instance.query_one("#notes-sidebar-title-main", Static)
            sidebar_title.update(f"My Notes ({len(notes_list_data)})")
        except QueryError:
            pass
        
        # Call the method on the sidebar instance, which handles its internal ListView
        await sidebar_left_instance.populate_notes_list(notes_list_data)
        logger.info(f"Loaded {len(notes_list_data)} notes into the sidebar.")
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error loading notes: {e_db}", exc_info=True)
        try:
            list_view_in_sidebar_db_err = sidebar_left_instance.query_one("#notes-list-view", ListView)
            await list_view_in_sidebar_db_err.clear()
            await list_view_in_sidebar_db_err.mount(ListItem(Label("Error loading notes from DB.")))
        except QueryError: pass # If ListView itself is the problem
    except Exception as e_unexp: # Catch other errors during populate_notes_list or list_notes
        logger.error(f"Unexpected error loading or populating notes: {e_unexp}", exc_info=True)
        try:
            list_view_in_sidebar_unexp_err = sidebar_left_instance.query_one("#notes-list-view", ListView)
            await list_view_in_sidebar_unexp_err.clear()
            await list_view_in_sidebar_unexp_err.mount(ListItem(Label("Unexpected error loading notes.")))
        except QueryError: pass


########################################################################################################################
#
# Helper Functions for Note Import
#
########################################################################################################################

def _parse_note_from_file_content(file_path: Path, file_content_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses note title and content from string content.
    Tries JSON (expects "title", "content" keys), then YAML (same keys),
    then plain text (first line as title, rest as content).
    For .txt and .md files, it directly uses the filename as title and full content.
    For other files, if JSON/YAML parsing fails or doesn't find a "title",
    it falls back to using the filename as title and the full file content.

    Returns:
        A tuple (title, content). Title might be None if unparsable and no filename fallback.
        Content will be the full string if unparsable.
    """
    logger_instance = logger  # Use global logger or pass app.loguru_logger
    filename_as_title = file_path.stem
    full_file_content = file_content_str.strip()

    if not full_file_content:  # Handle truly empty files after stripping
        logger_instance.debug(f"Note file '{file_path.name}' is empty. Using filename as title.")
        return filename_as_title, ""

    parsed_title: Optional[str] = None
    parsed_content: str = full_file_content  # Default to full content

    file_suffix = file_path.suffix.lower()

    # 1. Specific handling for .txt and .md files
    if file_suffix in ['.txt', '.md']:
        logger_instance.debug(f"Parsing note file '{file_path.name}' as TXT/MD. Using filename as title.")
        return filename_as_title, full_file_content

    # 1. Try JSON
    try:
        data = json.loads(file_content_str)
        if isinstance(data, dict):
            json_title = data.get("title")
            json_content = data.get("content")
            if json_title is not None:  # JSON has a "title" key
                logger_instance.debug(f"Parsed note file '{file_path.name}' as JSON.")
                # Use JSON content if present, otherwise use the full file content
                return json_title, json_content if json_content is not None else full_file_content
                # If JSON is valid dict but no "title", fall through to filename fallback logic
    except json.JSONDecodeError:
        logger_instance.debug(f"Note file '{file_path.name}' is not valid JSON. Trying YAML.")
    except Exception as e_json_other:
        logger_instance.warning(f"Unexpected error during JSON parsing of note '{file_path.name}': {e_json_other}")
        # Fall through

    # 2. Try YAML
    try:
        data = yaml.safe_load(file_content_str)
        if isinstance(data, dict):
            yaml_title = data.get("title")
            yaml_content = data.get("content")
            if yaml_title is not None:  # YAML has a "title" key
                logger_instance.debug(f"Parsed note file '{file_path.name}' as YAML.")
                return parsed_title, parsed_content
            return yaml_title, yaml_content if yaml_content is not None else full_file_content
            # If YAML is valid dict but no "title", fall through to filename fallback logic
    except yaml.YAMLError:
        logger_instance.debug(f"Note file '{file_path.name}' is not valid YAML. Using filename as title.")
    except Exception as e_yaml_other:
        logger_instance.warning(f"Unexpected error during YAML parsing of note '{file_path.name}': {e_yaml_other}")
    # Fall through

    # 3. Fallback for non-txt/md files if JSON/YAML parsing failed or didn't yield a title
    logger_instance.debug(
        f"Note file '{file_path.name}' (not TXT/MD) failed structured parsing or lacked 'title'. Using filename as title.")
    return filename_as_title, full_file_content


async def _note_import_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    logger_instance = getattr(app, 'loguru_logger', logger)
    if selected_path:
        logger_instance.info(f"Note import selected: {selected_path}")
        if not app.notes_service:
            app.notify("Database service not available.", severity="error")
            logger_instance.error("Notes service not available for note import.")
            return

        try:
            with open(selected_path, 'r', encoding='utf-8') as f:
                content_str = f.read()

            title, content = _parse_note_from_file_content(selected_path, content_str)

            if title is None:  # Should ideally not happen if filename fallback is used
                app.notify("Failed to determine title for imported note.", severity="error")
                logger_instance.error(f"Could not determine title for note from file: {selected_path}")
                return

            new_note_id = app.notes_service.add_note(
                user_id=app.notes_user_id,
                title=title,
                content=content if content is not None else ""  # Ensure content is not None
            )

            if new_note_id:
                app.notify(f"Note '{title}' imported successfully (ID: {new_note_id}).", severity="information")
                await load_and_display_notes_handler(app)  # Refresh notes list
            else:
                app.notify("Failed to import note into database. Check logs.", severity="error")
        except FileNotFoundError:
            app.notify(f"Note file not found: {selected_path}", severity="error")
            logger_instance.error(f"Note file not found: {selected_path}")
        except Exception as e:
            app.notify(f"Error importing note: {type(e).__name__}", severity="error", timeout=6)
            logger_instance.error(f"Error importing note from '{selected_path}': {e}", exc_info=True)
    else:
        logger_instance.info("Note import cancelled.")
        app.notify("Note import cancelled.", severity="information", timeout=2)


async def handle_notes_import_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger_instance = getattr(app, 'loguru_logger', logger)
    logger_instance.info("Notes 'Import Note' button pressed.")

    defined_filters = Filters(
        ("Note files (TXT, MD, JSON, YAML)", lambda p: p.suffix.lower() in (".txt", ".md", ".json", ".yaml", ".yml")),
        ("Text/Markdown (*.txt, *.md)", lambda p: p.suffix.lower() in (".txt", ".md")),
        ("JSON files (*.json)", lambda p: p.suffix.lower() == ".json"),
        ("YAML files (*.yaml, *.yml)", lambda p: p.suffix.lower() in (".yaml", ".yml")),
        ("All files (*.*)", lambda p: True)
    )
    await app.push_screen(
        FileOpen(location=str(Path.home()), title="Select Note File to Import", filters=defined_filters),
        callback=lambda path: _note_import_callback(app, path))


async def handle_notes_create_new_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Create New Note' button press in the notes sidebar."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("Notes 'Create New Note' button pressed.")
    if not app.notes_service:
        logger.error("Notes service not available, cannot create new note.")
        app.notify("Notes service unavailable.", severity="error")
        return
    try:
        new_note_title = f"New Note {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        new_note_id = app.notes_service.add_note(
            user_id=app.notes_user_id,
            title=new_note_title,
            content=""
        )
        if new_note_id:
            logger.info(f"New note created with ID: {new_note_id}")
            await load_and_display_notes_handler(app)  # This refreshes the list_view
            app.notify(f"Note '{new_note_title}' created.", severity="information")

            # After refreshing, find the newly added item in the ListView
            list_view = app.query_one("#notes-list-view", ListView)
            newly_selected_list_item: Optional[ListItem] = None
            target_index: Optional[int] = None

            for index, child_widget in enumerate(list_view.children):
                # Ensure it's a ListItem and has our custom attribute
                if isinstance(child_widget, ListItem) and hasattr(child_widget, 'note_id'):
                    # The hasattr check is crucial for runtime safety
                    # Type checkers might still warn unless you cast or use a custom ListItem type
                    if child_widget.note_id == new_note_id:  # type: ignore
                        newly_selected_list_item = child_widget
                        target_index = index
                        break

            if newly_selected_list_item is not None and target_index is not None:
                list_view.index = target_index  # Set the highlighted item by index
                list_view.scroll_to_widget(newly_selected_list_item, animate=False)  # Scroll to it
                # Now call the selection handler with the item that was just highlighted
                await handle_notes_list_view_selected(app, list_view.id, newly_selected_list_item)
            else:
                logger.warning(
                    f"Could not find newly created note item (ID: {new_note_id}) in the list view after refresh to auto-select it.")

        else:
            logger.error("Failed to create new note (ID was None).")
            app.notify("Failed to create new note.", severity="error")
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error creating new note: {e_db}", exc_info=True)
        app.notify("Database error creating note.", severity="error")
    except QueryError as e_query:  # Catch if #notes-list-view itself is not found
        logger.error(f"UI component error during new note creation: {e_query}", exc_info=True)
        app.notify("UI error processing new note.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error creating new note: {e_unexp}", exc_info=True)
        app.notify("Unexpected error creating note.", severity="error")


async def handle_notes_edit_selected_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Focuses the notes editor if a note is selected."""
    logging.info("Notes 'Edit Selected Note' button pressed.")
    try:
        if app.current_selected_note_id:
            app.query_one("#notes-editor-area", TextArea).focus()
            logging.info("Focused notes editor for editing selected note.")
        else:
            logging.info("Edit selected note: No note is currently selected.")
            app.notify("No note selected to edit.", severity="warning")
    except QueryError as e_query:
        logging.error(f"UI component not found for 'notes-edit-selected-button': {e_query}", exc_info=True)
        app.notify("UI error: Cannot focus editor.", severity="error")


async def handle_notes_search_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Performs a combined search with both content and keyword filters."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("Notes 'Search / Filter' button pressed.")
    
    try:
        search_input = app.query_one("#notes-search-input", Input)
        keyword_filter_input = app.query_one("#notes-keyword-filter-input", Input)
        
        search_term = search_input.value.strip()
        keyword_filter = keyword_filter_input.value.strip()
        
        await _perform_combined_search(app, search_term, keyword_filter)
        
    except QueryError as e_query:
        logger.error(f"UI component not found for 'notes-search-button': {e_query}", exc_info=True)
        app.notify("UI error: Cannot perform search.", severity="error")


async def handle_notes_load_selected_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Loads the highlighted note from the list into the editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("Notes 'Load Selected Note' button pressed.")
    if not app.notes_service:
        logger.error("Notes service not available, cannot load selected note.")
        app.notify("Notes service unavailable.", severity="error")
        return
    try:
        notes_list_view = app.query_one("#notes-list-view", ListView) # Query directly from app
        selected_item = notes_list_view.highlighted_child

        if selected_item and hasattr(selected_item, 'note_id') and hasattr(selected_item, 'note_version'):
            # Re-delegate to the selection handler for consistency
            await handle_notes_list_view_selected(app, notes_list_view.id, selected_item)
        else:
            logger.info("No item highlighted in notes list to load.")
            app.notify("No note selected in the list to load.", severity="warning")
    except QueryError as e_query:
        logger.error(f"UI component not found for 'notes-load-selected-button': {e_query}", exc_info=True)
        app.notify("UI error loading note.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error in 'notes-load-selected-button': {e_unexp}", exc_info=True)
        app.notify("Unexpected error loading note.", severity="error")


async def handle_notes_save_current_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Save Current Note' button in the notes sidebar."""
    logging.info("Notes 'Save Current Note' (sidebar) button pressed.")
    await save_current_note_handler(app)


async def handle_notes_main_save_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Save Note' button in the notes main content area controls."""
    logging.info("Notes 'Save Note' (main controls) button pressed.")
    await save_current_note_handler(app)


async def handle_notes_delete_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Delete Selected Note' button press."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("Notes 'Delete Selected Note' button pressed.")
    if not app.notes_service:
        logger.error("Notes service not available. Cannot delete note.")
        app.notify("Notes service unavailable.", severity="error")
        return
    if not app.current_selected_note_id or app.current_selected_note_version is None:
        logger.warning("No note selected to delete or version missing.")
        app.notify("No note selected to delete, or version is missing.", severity="warning")
        return

    logger.info(
        f"Attempting to delete note ID: {app.current_selected_note_id}, Version: {app.current_selected_note_version}")

    try:
        success = app.notes_service.soft_delete_note(
            user_id=app.notes_user_id,
            note_id=app.current_selected_note_id,
            expected_version=app.current_selected_note_version
        )
        if success:
            logger.info(f"Note {app.current_selected_note_id} soft-deleted successfully.")
            app.notify("Note deleted.", severity="information")

            app.current_selected_note_id = None
            app.current_selected_note_version = None
            app.current_selected_note_title = ""
            app.current_selected_note_content = ""

            app.query_one("#notes-editor-area", TextArea).text = ""
            app.query_one("#notes-title-input", Input).value = ""
            app.query_one("#notes-keywords-area", TextArea).text = ""

            await load_and_display_notes_handler(app)
        else:
            logger.warning(f"notes_service.soft_delete_note for {app.current_selected_note_id} returned False.")
            app.notify("Failed to delete note (may have been changed or already deleted).", severity="warning")
            await load_and_display_notes_handler(app)

    except ConflictError as e_conflict:
        logger.error(f"Conflict deleting note {app.current_selected_note_id}: {e_conflict}", exc_info=True)
        app.notify(f"Delete conflict: {e_conflict}. Note may have been changed. Please reload.", severity="error")
        await load_and_display_notes_handler(app)
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error deleting note {app.current_selected_note_id}: {e_db}", exc_info=True)
        app.notify("Error deleting note from database.", severity="error")
    except QueryError as e_query:
        logger.error(f"UI component not found while deleting note: {e_query}", exc_info=True)
        app.notify("UI error while deleting note.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error deleting note {app.current_selected_note_id}: {e_unexp}", exc_info=True)
        app.notify("Unexpected error deleting note.", severity="error")


async def handle_notes_save_keywords_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles saving keywords for the currently selected note."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("Notes 'Save Keywords' button pressed.")
    if not app.notes_service:
        logger.error("Notes service not available. Cannot save keywords.")
        app.notify("Notes service unavailable.", severity="error")
        return
    if not app.current_selected_note_id:
        logger.warning("No note selected. Cannot save keywords.")
        app.notify("No note selected to save keywords for.", severity="warning")
        return

    try:
        keywords_area = app.query_one("#notes-keywords-area", TextArea)
        input_keyword_texts = {kw.strip().lower() for kw in keywords_area.text.split(',') if kw.strip()}
        logger.info(
            f"Attempting to save keywords for note {app.current_selected_note_id}. Input: {input_keyword_texts}")

        existing_linked_keywords_data = app.notes_service.get_keywords_for_note(
            user_id=app.notes_user_id, note_id=app.current_selected_note_id
        )
        existing_linked_keyword_map = {kw['keyword'].lower(): kw['id'] for kw in existing_linked_keywords_data}

        keywords_actually_changed = False

        # Add/Link new keywords
        for kw_text_to_add in input_keyword_texts:
            if kw_text_to_add not in existing_linked_keyword_map:  # If not already linked (case-insensitively)
                # Get or create the keyword globally
                keyword_detail = app.notes_service.get_keyword_by_text(app.notes_user_id, kw_text_to_add)
                kw_id_to_link: Optional[int] = None
                if not keyword_detail:
                    new_kw_id = app.notes_service.add_keyword(app.notes_user_id, kw_text_to_add)  # Returns int ID
                    if new_kw_id is not None: kw_id_to_link = new_kw_id
                else:
                    kw_id_to_link = keyword_detail['id']

                if kw_id_to_link:
                    app.notes_service.link_note_to_keyword(
                        user_id=app.notes_user_id, note_id=app.current_selected_note_id, keyword_id=kw_id_to_link
                    )
                    keywords_actually_changed = True
                    logger.debug(
                        f"Linked keyword ID {kw_id_to_link} ('{kw_text_to_add}') to note {app.current_selected_note_id}")

        # Unlink removed keywords
        for existing_kw_text, existing_kw_id in existing_linked_keyword_map.items():
            if existing_kw_text not in input_keyword_texts:
                app.notes_service.unlink_note_from_keyword(
                    user_id=app.notes_user_id, note_id=app.current_selected_note_id, keyword_id=existing_kw_id
                )
                keywords_actually_changed = True
                logger.debug(
                    f"Unlinked keyword ID {existing_kw_id} ('{existing_kw_text}') from note {app.current_selected_note_id}")

        if keywords_actually_changed:
            # Refresh the displayed keywords to show canonical casing from DB
            refreshed_keywords_data = app.notes_service.get_keywords_for_note(
                user_id=app.notes_user_id, note_id=app.current_selected_note_id
            )
            keywords_area.text = ", ".join(
                [kw['keyword'] for kw in refreshed_keywords_data]) if refreshed_keywords_data else ""
            app.notify("Keywords saved successfully!", severity="information")
            logger.info(f"Keywords for note {app.current_selected_note_id} updated and refreshed.")
        else:
            app.notify("No changes to keywords.", severity="info")


    except CharactersRAGDBError as e_db:
        logger.error(f"Database error saving keywords for note {app.current_selected_note_id}: {e_db}", exc_info=True)
        app.notify("Error saving keywords to database.", severity="error")
    except QueryError as e_query:
        logger.error(f"UI component #notes-keywords-area not found: {e_query}", exc_info=True)
        app.notify("UI error while saving keywords.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error saving keywords for note {app.current_selected_note_id}: {e_unexp}",
                      exc_info=True)
        app.notify("Unexpected error saving keywords.", severity="error")


# --- Input/List View Changed Handlers for Notes Tab ---


async def _perform_combined_search(app: 'TldwCli', search_term: str, keyword_filter: str) -> None:
    """Performs a combined search with content and keyword filtering."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.debug(f"Combined search - Content: '{search_term}', Keywords: '{keyword_filter}'")
    
    if not app.notes_service:
        logger.error("Notes service not available for search.")
        app.notify("Notes service unavailable.", severity="error")
        return
        
    try:
        sidebar_left: NotesSidebarLeft = app.query_one("#notes-sidebar-left", NotesSidebarLeft)
        
        # If both are empty, show all notes
        if not search_term and not keyword_filter:
            await load_and_display_notes_handler(app)
            return
        
        # Get all notes first (we'll filter in memory for now)
        all_notes = []
        
        # If there's a search term, search by content
        if search_term:
            all_notes = app.notes_service.search_notes(
                user_id=app.notes_user_id, search_term=search_term, limit=500
            )
        else:
            # If no search term, get all notes
            all_notes = app.notes_service.list_notes(
                user_id=app.notes_user_id, limit=500
            )
        
        # Filter by keywords if specified
        if keyword_filter:
            # Parse keyword filter (comma-separated)
            filter_keywords = {kw.strip().lower() for kw in keyword_filter.split(',') if kw.strip()}
            
            filtered_notes = []
            for note in all_notes:
                # Get keywords for this note
                note_keywords_data = app.notes_service.get_keywords_for_note(
                    user_id=app.notes_user_id, note_id=note['id']
                )
                note_keywords = {kw['keyword'].lower() for kw in note_keywords_data}
                
                # Check if any filter keyword matches
                if any(fk in note_keywords for fk in filter_keywords):
                    filtered_notes.append(note)
            
            all_notes = filtered_notes
        
        # Update the UI with results
        await sidebar_left.populate_notes_list(all_notes)
        
        result_count = len(all_notes)
        if search_term and keyword_filter:
            logger.info(f"Combined search found {result_count} notes")
            app.notify(f"Found {result_count} notes matching content and keywords", severity="information")
        elif search_term:
            logger.info(f"Content search found {result_count} notes for '{search_term}'")
        elif keyword_filter:
            logger.info(f"Keyword filter found {result_count} notes")
            app.notify(f"Found {result_count} notes with specified keywords", severity="information")
            
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error during combined search: {e_db}", exc_info=True)
        app.notify("Database error during search.", severity="error")
    except QueryError as e_query:
        logger.error(f"UI component not found during combined search: {e_query}", exc_info=True)
        app.notify("UI error during search.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error during combined search: {e_unexp}", exc_info=True)
        app.notify("Unexpected error during search.", severity="error")


async def _actual_notes_search(app: 'TldwCli', search_term: str) -> None:
    """Performs the actual notes search and updates the UI."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.debug(f"Debounced notes search executing for term: '{search_term}'")

    if not app.notes_service:
        logger.error("Notes service not available for actual search.")
        # Optionally notify or update UI to show error, though less critical for debounced search
        return
    try:
        sidebar_left_search: NotesSidebarLeft = app.query_one("#notes-sidebar-left", NotesSidebarLeft)
        if not search_term: # Should typically be caught by the caller, but good to re-check
            await load_and_display_notes_handler(app)
        else:
            notes_list_results = app.notes_service.search_notes(
                user_id=app.notes_user_id, search_term=search_term, limit=200
            )
            await sidebar_left_search.populate_notes_list(notes_list_results)
            logger.info(f"Debounced search found {len(notes_list_results)} notes for term '{search_term}'.")
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error during debounced notes search for '{search_term}': {e_db}", exc_info=True)
    except QueryError as e_query:
        logger.error(f"UI component not found during debounced note search: {e_query}", exc_info=True)
    except Exception as e_unexp:
        logger.error(f"Unexpected error during debounced note search: {e_unexp}", exc_info=True)


async def handle_notes_search_input_changed(app: 'TldwCli', event_value: str) -> None:
    """Handles input changes in the notes search bar with debouncing."""
    logger = getattr(app, 'loguru_logger', logging)
    search_term = event_value.strip()
    # Log that input changed, but not that search is *immediately* executing
    logger.debug(f"Notes search input changed to: '{search_term}'. Debouncing search...")

    if app._notes_search_timer is not None:
        app._notes_search_timer.stop()
        logger.debug("Previous notes search timer stopped.")

    # If the search term is empty, we want to load all notes immediately,
    # not after a delay.
    if not search_term:
        logger.debug("Search term is empty, loading all notes immediately.")
        app._notes_search_timer = None # Clear any pending timer
        await load_and_display_notes_handler(app)
        return

    # Define a new callable for set_timer that captures the current app and search_term
    async def debounced_search_action():
        await _actual_notes_search(app, search_term)

    app._notes_search_timer = app.set_timer(
        0.5,  # 500ms delay
        debounced_search_action # Pass the async callable
    )
    logger.debug(f"Notes search timer started for term '{search_term}'.")


async def handle_notes_keyword_filter_input_changed(app: 'TldwCli', event_value: str) -> None:
    """Handles input changes in the keyword filter field."""
    logger = getattr(app, 'loguru_logger', logging)
    keyword_filter = event_value.strip()
    logger.debug(f"Notes keyword filter input changed to: '{keyword_filter}'")
    
    # Get current search term
    try:
        search_input = app.query_one("#notes-search-input", Input)
        search_term = search_input.value.strip()
    except QueryError:
        search_term = ""
    
    # Perform combined search immediately for keyword filter changes
    # (no debouncing for keyword filter to provide immediate feedback)
    await _perform_combined_search(app, search_term, keyword_filter)


async def handle_notes_list_view_selected(app: 'TldwCli', list_view_id: str, item: Any) -> None:
    """Handles selecting a note from the list in the notes left sidebar."""
    if not app.notes_service:
        logger.error("Notes service not available, cannot load selected note details.")
        app.notify("Notes service unavailable.", severity="error")
        return

    selected_list_item = item  # This is the ListItem widget
    if selected_list_item and hasattr(selected_list_item, 'note_id') and hasattr(selected_list_item, 'note_version'):
        note_id_selected = selected_list_item.note_id
        logger.info(f"Note selected in UI: ID={note_id_selected}")

        try:
            note_details_from_db = app.notes_service.get_note_by_id(
                user_id=app.notes_user_id, note_id=note_id_selected
            )
            if note_details_from_db:
                app.current_selected_note_id = note_id_selected
                app.current_selected_note_version = note_details_from_db.get('version')  # Use fresh version from DB
                app.current_selected_note_title = note_details_from_db.get('title')
                app.current_selected_note_content = note_details_from_db.get('content', "")

                editor_widget = app.query_one("#notes-editor-area", TextArea)
                editor_widget.load_text(app.current_selected_note_content) # Use load_text for TextAreas

                title_input_widget = app.query_one("#notes-title-input", Input)
                title_input_widget.value = app.current_selected_note_title or ""

                keywords_area_widget = app.query_one("#notes-keywords-area", TextArea)
                keywords_for_note_list = app.notes_service.get_keywords_for_note(
                    user_id=app.notes_user_id, note_id=note_id_selected
                )
                keywords_area_widget.load_text(", ".join(
                    [kw['keyword'] for kw in keywords_for_note_list]) if keywords_for_note_list else "")

                logger.info(
                    f"Loaded note '{app.current_selected_note_title}' (v{app.current_selected_note_version}) into editor.")
                app.notify(f"Note '{app.current_selected_note_title}' loaded.", severity="information", timeout=2)
                
                # Reset unsaved changes flag when loading a note
                app.notes_unsaved_changes = False

            else:
                logger.warning(f"Could not retrieve details for note ID: {note_id_selected}. It may have been deleted.")
                app.notify(f"Failed to load note ID: {note_id_selected}. It may no longer exist.", severity="error")
                # Clear UI and reactive vars
                app.current_selected_note_id = None
                app.current_selected_note_version = None
                app.current_selected_note_title = ""
                app.current_selected_note_content = ""
                app.query_one("#notes-editor-area", TextArea).load_text("")
                app.query_one("#notes-title-input", Input).value = ""
                app.query_one("#notes-keywords-area", TextArea).load_text("")
                await load_and_display_notes_handler(app)

        except CharactersRAGDBError as e_db:
            logger.error(f"Database error loading note {note_id_selected}: {e_db}", exc_info=True)
            app.notify("Database error loading note.", severity="error")
        except QueryError as e_query:
            logger.error(f"UI component not found while loading note details: {e_query}", exc_info=True)
            app.notify("UI error loading note details.", severity="error")
        except Exception as e_unexp:
            logger.error(f"Unexpected error loading note {note_id_selected}: {e_unexp}", exc_info=True)
            app.notify("Unexpected error loading note.", severity="error")
    else:
        logger.debug("Notes ListView selection was empty or item lacked note_id/note_version.")


# --- Export Handlers ---

async def _note_export_callback(app: 'TldwCli', selected_path: Optional[Path], export_format: str) -> None:
    """Callback for exporting a note to a file."""
    logger_instance = getattr(app, 'loguru_logger', logger)
    
    if not selected_path:
        logger_instance.info("Note export cancelled.")
        app.notify("Note export cancelled.", severity="information", timeout=2)
        return
        
    if not app.current_selected_note_id:
        app.notify("No note selected to export.", severity="warning")
        logger_instance.warning("Export attempted with no note selected.")
        return
    
    try:
        # Ensure we have the latest content from the editor
        editor = app.query_one("#notes-editor-area", TextArea)
        title_input = app.query_one("#notes-title-input", Input)
        
        current_title = title_input.value.strip() or "Untitled Note"
        current_content = editor.text
        
        # Get keywords for the note
        keywords_text = ""
        if app.notes_service:
            keywords_data = app.notes_service.get_keywords_for_note(
                user_id=app.notes_user_id, 
                note_id=app.current_selected_note_id
            )
            if keywords_data:
                keywords_text = ", ".join([kw['keyword'] for kw in keywords_data])
        
        # Format content based on export type
        if export_format == "markdown":
            # Add metadata as YAML frontmatter for markdown
            export_content = f"""---
title: {current_title}
date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
keywords: {keywords_text}
note_id: {app.current_selected_note_id}
---

# {current_title}

{current_content}"""
        else:  # Plain text
            export_content = f"""Title: {current_title}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Keywords: {keywords_text}
Note ID: {app.current_selected_note_id}

{'=' * 50}

{current_content}"""
        
        # Write to file
        with open(selected_path, 'w', encoding='utf-8') as f:
            f.write(export_content)
        
        app.notify(f"Note exported successfully to {selected_path.name}", severity="information")
        logger_instance.info(f"Note '{current_title}' exported to {selected_path}")
        
    except Exception as e:
        app.notify(f"Error exporting note: {type(e).__name__}", severity="error")
        logger_instance.error(f"Error exporting note to '{selected_path}': {e}", exc_info=True)


async def handle_notes_export_markdown_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Export as Markdown' button press."""
    logger_instance = getattr(app, 'loguru_logger', logger)
    logger_instance.info("Notes 'Export as Markdown' button pressed.")
    
    if not app.current_selected_note_id:
        app.notify("No note selected to export.", severity="warning")
        return
    
    # Get current title for default filename
    try:
        title_input = app.query_one("#notes-title-input", Input)
        current_title = title_input.value.strip() or "Untitled Note"
        # Sanitize title for filename
        safe_title = "".join(c for c in current_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        default_filename = f"{safe_title}.md"
    except QueryError:
        default_filename = "note.md"
    
    await app.push_screen(
        FileSave(
            location=str(Path.home()),
            default_file=default_filename,
            title="Export Note as Markdown"
        ),
        callback=lambda path: _note_export_callback(app, path, "markdown")
    )


async def handle_notes_export_text_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Export as Text' button press."""
    logger_instance = getattr(app, 'loguru_logger', logger)
    logger_instance.info("Notes 'Export as Text' button pressed.")
    
    if not app.current_selected_note_id:
        app.notify("No note selected to export.", severity="warning")
        return
    
    # Get current title for default filename
    try:
        title_input = app.query_one("#notes-title-input", Input)
        current_title = title_input.value.strip() or "Untitled Note"
        # Sanitize title for filename
        safe_title = "".join(c for c in current_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        default_filename = f"{safe_title}.txt"
    except QueryError:
        default_filename = "note.txt"
    
    await app.push_screen(
        FileSave(
            location=str(Path.home()),
            default_file=default_filename,
            title="Export Note as Text"
        ),
        callback=lambda path: _note_export_callback(app, path, "text")
    )


async def handle_notes_copy_markdown_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Copy as Markdown' button press."""
    logger_instance = getattr(app, 'loguru_logger', logger)
    logger_instance.info("Notes 'Copy as Markdown' button pressed.")
    
    if not app.current_selected_note_id:
        app.notify("No note selected to copy.", severity="warning")
        return
    
    await _copy_note_to_clipboard(app, "markdown")


async def handle_notes_copy_text_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Copy as Text' button press."""
    logger_instance = getattr(app, 'loguru_logger', logger)
    logger_instance.info("Notes 'Copy as Text' button pressed.")
    
    if not app.current_selected_note_id:
        app.notify("No note selected to copy.", severity="warning")
        return
    
    await _copy_note_to_clipboard(app, "text")


async def handle_notes_sidebar_emoji_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the sidebar emoji button press."""
    from ..Widgets.emoji_picker import EmojiPickerScreen, EmojiSelected
    
    async def emoji_callback(emoji_char: str) -> None:
        if emoji_char:
            # Insert emoji at cursor in notes editor
            try:
                editor = app.query_one("#notes-editor-area", TextArea)
                editor.insert_text_at_cursor(emoji_char)
                editor.focus()
            except QueryError:
                pass
    
    await app.push_screen(EmojiPickerScreen(), emoji_callback)


async def _copy_note_to_clipboard(app: 'TldwCli', format_type: str) -> None:
    """Copy the current note to clipboard in the specified format."""
    logger_instance = getattr(app, 'loguru_logger', logger)
    
    try:
        # Ensure we have the latest content from the editor
        editor = app.query_one("#notes-editor-area", TextArea)
        title_input = app.query_one("#notes-title-input", Input)
        
        current_title = title_input.value.strip() or "Untitled Note"
        current_content = editor.text
        
        # Get keywords for the note
        keywords_text = ""
        if app.notes_service:
            keywords_data = app.notes_service.get_keywords_for_note(
                user_id=app.notes_user_id, 
                note_id=app.current_selected_note_id
            )
            if keywords_data:
                keywords_text = ", ".join([kw['keyword'] for kw in keywords_data])
        
        # Format content based on format type
        if format_type == "markdown":
            # Add metadata as YAML frontmatter for markdown
            clipboard_content = f"""---
title: {current_title}
date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
keywords: {keywords_text}
note_id: {app.current_selected_note_id}
---

# {current_title}

{current_content}"""
        else:  # Plain text
            clipboard_content = f"""Title: {current_title}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Keywords: {keywords_text}
Note ID: {app.current_selected_note_id}

{'=' * 50}

{current_content}"""
        
        # Copy to clipboard
        try:
            import pyperclip
            pyperclip.copy(clipboard_content)
            app.notify(f"Note copied to clipboard as {format_type}!", severity="information")
            logger_instance.info(f"Note '{current_title}' copied to clipboard as {format_type}")
        except ImportError:
            # Fallback to Textual's clipboard if pyperclip is not available
            try:
                await app.copy_to_clipboard(clipboard_content)
                app.notify(f"Note copied to clipboard as {format_type}!", severity="information")
                logger_instance.info(f"Note '{current_title}' copied to clipboard as {format_type}")
            except Exception as e:
                app.notify("Clipboard copy failed. Please install 'pyperclip' for better clipboard support.", severity="error")
                logger_instance.error(f"Failed to copy to clipboard: {e}")
        
    except QueryError as e:
        app.notify("UI error while copying note.", severity="error")
        logger_instance.error(f"UI component not found: {e}")
    except Exception as e:
        app.notify(f"Error copying note: {type(e).__name__}", severity="error")
        logger_instance.error(f"Error copying note to clipboard: {e}", exc_info=True)


# Button Handler Map will be defined at the end of the file

#
# --- Template Definitions ---

import json
import os
from pathlib import Path

def load_note_templates():
    """Load note templates from JSON file or use defaults."""
    # Try to load from user's config directory first
    user_config_path = Path.home() / ".config" / "tldw_cli" / "note_templates.json"
    
    # Fallback to app's config directory
    app_config_path = Path(__file__).parent.parent / "Config_Files" / "note_templates.json"
    
    # Use hardcoded defaults as last resort
    default_templates = {
        "blank": {
            "title": "New Note",
            "content": "",
            "keywords": ""
        },
        "meeting": {
            "title": "Meeting Notes - {date}",
            "content": """## Meeting Notes

**Date:** {date}
**Time:** {time}
**Attendees:** 

### Agenda
- 

### Discussion Points
- 

### Action Items
- [ ] 
- [ ] 

### Next Steps
- 

### Notes
""",
            "keywords": "meeting, notes"
        }
    }
    
    # Try to load templates
    templates_data = None
    loaded_from = None
    
    # First try user config
    if user_config_path.exists():
        try:
            with open(user_config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                templates_data = data.get('templates', data)
                loaded_from = "user config"
        except Exception as e:
            logger.warning(f"Failed to load user templates from {user_config_path}: {e}")
    
    # Then try app config
    if templates_data is None and app_config_path.exists():
        try:
            with open(app_config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                templates_data = data.get('templates', data)
                loaded_from = "app config"
        except Exception as e:
            logger.warning(f"Failed to load app templates from {app_config_path}: {e}")
    
    # Use defaults if nothing loaded
    if templates_data is None:
        templates_data = default_templates
        loaded_from = "defaults"
    
    logger.info(f"Loaded {len(templates_data)} note templates from {loaded_from}")
    return templates_data

# Load templates on module import
NOTE_TEMPLATES = load_note_templates()

# --- New UX Enhancement Handlers ---

async def handle_notes_editor_changed(app: 'TldwCli', event) -> None:
    """Handles text changes in the notes editor to track unsaved changes and word count."""
    if hasattr(app, 'notes_unsaved_changes') and app.current_selected_note_id:
        # Mark as having unsaved changes if content differs from original
        current_content = event.text_area.text
        if current_content != app.current_selected_note_content:
            app.notes_unsaved_changes = True
        else:
            app.notes_unsaved_changes = False
    
    # Update word count
    try:
        word_count_label = app.query_one("#notes-word-count", Label)
        text = event.text_area.text
        word_count = len(text.split()) if text else 0
        word_count_label.update(f"Words: {word_count}")
    except QueryError:
        pass


async def handle_notes_title_changed(app: 'TldwCli', event) -> None:
    """Handles title input changes to track unsaved changes."""
    if hasattr(app, 'notes_unsaved_changes') and app.current_selected_note_id:
        current_title = event.input.value
        if current_title != app.current_selected_note_title:
            app.notes_unsaved_changes = True
        else:
            app.notes_unsaved_changes = False


async def handle_notes_preview_toggle(app: 'TldwCli', event: Button.Pressed) -> None:
    """Toggles between edit and preview mode for markdown notes."""
    logger = getattr(app, 'loguru_logger', logging)
    
    try:
        editor = app.query_one("#notes-editor-area", TextArea)
        preview_button = app.query_one("#notes-preview-toggle", Button)
        
        if app.notes_preview_mode:
            # Switch back to edit mode
            app.notes_preview_mode = False
            editor.read_only = False
            preview_button.label = "Preview"
            logger.debug("Switched to edit mode")
        else:
            # Switch to preview mode
            app.notes_preview_mode = True
            editor.read_only = True
            preview_button.label = "Edit"
            logger.debug("Switched to preview mode")
            
            # TODO: In the future, render markdown to rich text
            # For now, just make it read-only
            
    except QueryError as e:
        logger.error(f"UI component not found for preview toggle: {e}")


async def handle_notes_sort_changed(app: 'TldwCli', event) -> None:
    """Handles changes to the sort dropdown."""
    logger = getattr(app, 'loguru_logger', logging)
    
    if hasattr(event, 'select'):
        app.notes_sort_by = event.select.value
        logger.debug(f"Sort by changed to: {app.notes_sort_by}")
        await load_and_display_notes_handler(app)


async def handle_notes_sort_order_toggle(app: 'TldwCli', event: Button.Pressed) -> None:
    """Toggles between ascending and descending sort order."""
    logger = getattr(app, 'loguru_logger', logging)
    
    app.notes_sort_ascending = not app.notes_sort_ascending
    
    # Update button label
    try:
        sort_button = app.query_one("#notes-sort-order-button", Button)
        if app.notes_sort_ascending:
            sort_button.label = "↑ Oldest First"
        else:
            sort_button.label = "↓ Newest First"
    except QueryError:
        pass
    
    logger.debug(f"Sort order toggled to: {'ascending' if app.notes_sort_ascending else 'descending'}")
    await load_and_display_notes_handler(app)


async def handle_notes_create_from_template(app: 'TldwCli', event: Button.Pressed) -> None:
    """Creates a new note from the selected template."""
    logger = getattr(app, 'loguru_logger', logging)
    
    if not app.notes_service:
        app.notify("Notes service is not available.", severity="error")
        logger.error("Notes service not available in handle_notes_create_from_template.")
        return
    
    try:
        # Get selected template
        template_select = app.query_one("#notes-template-select", Select)
        template_key = template_select.value
        
        if not template_key or template_key not in NOTE_TEMPLATES:
            app.notify("Please select a template.", severity="warning")
            return
        
        template = NOTE_TEMPLATES[template_key]
        
        # Format template with current date/time
        from datetime import datetime
        now = datetime.now()
        template_data = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "datetime": now.strftime("%Y-%m-%d %H:%M")
        }
        
        title = template["title"].format(**template_data)
        content = template["content"].format(**template_data)
        keywords = template["keywords"]
        
        logger.info(f"Creating new note from template: {template_key}")
        
        # Create the note
        new_note_id = app.notes_service.add_note(
            user_id=app.notes_user_id,
            title=title,
            content=content
        )
        
        if new_note_id:
            # Store Note ID and Version
            app.current_selected_note_id = new_note_id
            app.current_selected_note_version = 1
            app.current_selected_note_title = title
            app.current_selected_note_content = content
            
            logger.info(f"New note created from template with ID: {new_note_id}")
            
            # Update UI
            editor = app.query_one("#notes-editor-area", TextArea)
            editor.load_text(content)
            
            title_input = app.query_one("#notes-title-input", Input)
            title_input.value = title
            
            keywords_area = app.query_one("#notes-keywords-area", TextArea)
            keywords_area.load_text(keywords)
            
            # Reset unsaved changes since this is a new note
            app.notes_unsaved_changes = False
            
            # Refresh the notes list
            await load_and_display_notes_handler(app)
            
            # Focus the editor
            editor.focus()
            
            app.notify(f"Note created from '{template_key}' template!", severity="information")
        else:
            app.notify("Failed to create note from template.", severity="error")
            logger.error("notes_service.add_note returned None for template note.")
            
    except QueryError as e:
        logger.error(f"UI component not found: {e}")
        app.notify("UI error creating note from template.", severity="error")
    except Exception as e:
        logger.error(f"Unexpected error creating note from template: {e}", exc_info=True)
        app.notify("Error creating note from template.", severity="error")


# --- Button Handler Map ---
# This must be defined after all handler functions
NOTES_BUTTON_HANDLERS = {
    "toggle-notes-sidebar-left": handle_notes_tab_sidebar_toggle,
    "toggle-notes-sidebar-right": handle_notes_tab_sidebar_toggle,
    "notes-import-button": handle_notes_import_button_pressed,
    "notes-create-new-button": handle_notes_create_new_button_pressed,
    "notes-create-from-template-button": handle_notes_create_from_template,
    "notes-edit-selected-button": handle_notes_edit_selected_button_pressed,
    "notes-search-button": handle_notes_search_button_pressed,
    "notes-load-selected-button": handle_notes_load_selected_button_pressed,
    "notes-save-current-button": handle_notes_save_current_button_pressed,
    "notes-save-button": handle_notes_main_save_button_pressed,  # Main save button in notes editor area
    "notes-delete-button": handle_notes_delete_button_pressed,
    "notes-export-markdown-button": handle_notes_export_markdown_button_pressed,
    "notes-export-text-button": handle_notes_export_text_button_pressed,
    "notes-copy-markdown-button": handle_notes_copy_markdown_button_pressed,
    "notes-copy-text-button": handle_notes_copy_text_button_pressed,
    "notes-preview-toggle": handle_notes_preview_toggle,
    "notes-sort-order-button": handle_notes_sort_order_toggle,
    "notes-sidebar-emoji-button": handle_notes_sidebar_emoji_button_pressed,
}

# End of notes_events.py
########################################################################################################################
