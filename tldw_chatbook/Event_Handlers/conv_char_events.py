# conv_char_events.py
# Description:
#
# Imports
import json  # For export
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any, List, Dict, cast
import yaml
#
# 3rd-Party Imports
from loguru import logger as loguru_logger
from textual.widgets import (
    Input, ListView, TextArea, Label, Collapsible, Select, Static, ListItem, Button
)
from textual.containers import VerticalScroll
from textual.css.query import QueryError
from rich.text import Text # For displaying messages if needed
#
# Local Imports
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, Filters # For File Picker
from ..Widgets.chat_message import ChatMessage # If CCP tab displays ChatMessage widgets
from ..Character_Chat import Character_Chat_Lib as ccl
from ..Prompt_Management import Prompts_Interop as prompts_interop
from ..DB.ChaChaNotes_DB import ConflictError, CharactersRAGDBError # For specific error handling
from .Chat_Events.chat_events import load_branched_conversation_history_ui
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Functions:

########################################################################################################################

########################################################################################################################
#
# Helper Functions (specific to CCP tab logic, moved from app.py)
#
########################################################################################################################

async def populate_ccp_character_select(app: 'TldwCli') -> None:
    """Populates the character selection dropdown in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("Attempting to populate #conv-char-character-select dropdown.")
    if not app.notes_service:
        logger.error("Notes service not available, cannot populate character select for CCP tab.")
        try:
            char_select_widget = app.query_one("#conv-char-character-select", Select)
            char_select_widget.set_options([("Service Unavailable", Select.BLANK)])
            char_select_widget.value = Select.BLANK
            char_select_widget.prompt = "Service Unavailable"
        except QueryError:
            logger.error("Failed to find #conv-char-character-select to show service error.")
        return

    try:
        db = app.notes_service._get_db(app.notes_user_id)
        character_cards = db.list_character_cards(limit=1000)

        options = []
        if character_cards:
            options = [(char['name'], char['id']) for char in character_cards if
                       char.get('name') and char.get('id') is not None]

        char_select_widget = app.query_one("#conv-char-character-select", Select)

        if options:
            char_select_widget.set_options(options)
            char_select_widget.prompt = "Select Character..."
            logger.info(f"Populated #conv-char-character-select with {len(options)} characters.")
        else:
            char_select_widget.set_options([("No characters found", Select.BLANK)])
            char_select_widget.value = Select.BLANK
            char_select_widget.prompt = "No characters available"
            logger.info("No characters found to populate #conv-char-character-select.")

    except QueryError as e_query:
        logger.error(f"Failed to find #conv-char-character-select widget: {e_query}", exc_info=True)
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error populating #conv-char-character-select: {e_db}", exc_info=True)
        try:
            char_select_widget_err = app.query_one("#conv-char-character-select", Select)
            char_select_widget_err.set_options([("Error Loading Characters", Select.BLANK)])
            char_select_widget_err.prompt = "Error Loading"
        except QueryError:
            pass
    except Exception as e_unexp:
        logger.error(f"Unexpected error populating #conv-char-character-select: {e_unexp}", exc_info=True)
        try:
            char_select_widget_unexp = app.query_one("#conv-char-character-select", Select)
            char_select_widget_unexp.set_options([("Error Loading (Unexpected)", Select.BLANK)])
            char_select_widget_unexp.prompt = "Error Loading"
        except QueryError:
            pass


async def populate_ccp_prompts_list_view(app: 'TldwCli', search_term: Optional[str] = None) -> None:
    """Populates the prompts list view in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    if not app.prompts_service_initialized:
        try:
            list_view_prompt_err = app.query_one("#ccp-prompts-listview", ListView)
            await list_view_prompt_err.clear()
            await list_view_prompt_err.append(ListItem(Label("Prompts service not available.")))
        except QueryError:
            logger.error("Failed to find #ccp-prompts-listview to show service error.")
        return

    try:
        list_view_prompt = app.query_one("#ccp-prompts-listview", ListView)
        await list_view_prompt.clear()
        results: List[Dict[str, Any]] = []

        if search_term:
            results_tuple = prompts_interop.search_prompts(
                search_query=search_term,
                search_fields=["name", "details", "keywords"],
                page=1, results_per_page=100,
                include_deleted=False
            )
            results = results_tuple[0] if results_tuple else []
        else:
            results_tuple = prompts_interop.list_prompts(page=1, per_page=100, include_deleted=False)
            results = results_tuple[0] if results_tuple else []

        if not results:
            await list_view_prompt.append(ListItem(Label("No prompts found.")))
        else:
            for prompt_data in results:
                display_name = prompt_data.get('name', 'Unnamed Prompt')
                item = ListItem(Label(display_name))
                item.prompt_id = prompt_data.get('id')
                item.prompt_uuid = prompt_data.get('uuid')
                await list_view_prompt.append(item)
        logger.info(f"Populated CCP prompts list. Search: '{search_term}', Found: {len(results)}")
    except QueryError as e_query:
        logger.error(f"UI component error populating CCP prompts list: {e_query}", exc_info=True)
    except (prompts_interop.DatabaseError, RuntimeError) as e_prompt_service:
        logger.error(f"Error populating CCP prompts list: {e_prompt_service}", exc_info=True)
        try:
            list_view_err_detail = app.query_one("#ccp-prompts-listview", ListView)
            await list_view_err_detail.clear()
            await list_view_err_detail.append(
                ListItem(Label(f"Error loading prompts: {type(e_prompt_service).__name__}")))
        except QueryError:
            pass
    except Exception as e_unexp:
        logger.error(f"Unexpected error populating CCP prompts list: {e_unexp}", exc_info=True)


def clear_ccp_prompt_fields(app: 'TldwCli') -> None:
    """Clears prompt input fields in the CCP right pane."""
    logger = getattr(app, 'loguru_logger', logging)
    app._clear_prompt_fields()


########################################################################################################################
#
# Event Handlers for Character Card Import
#
########################################################################################################################
async def _character_import_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    logger = getattr(app, 'loguru_logger', logging)
    if selected_path:
        logger.info(f"Character card import selected: {selected_path}")
        if not app.notes_service:
            app.notify("Database service not available.", severity="error")
            logger.error("Notes service not available for character import.")
            return

        db = app.notes_service._get_db(app.notes_user_id)
        try:
            # import_and_save_character_from_file can take Path object directly
            char_id = ccl.import_and_save_character_from_file(db, str(selected_path))
            if char_id is not None:
                app.notify(f"Character card imported successfully (ID: {char_id}).", severity="information")
                await populate_ccp_character_select(app) # Refresh character dropdown
            else:
                # import_and_save_character_from_file logs errors, but we can notify too
                app.notify("Failed to import character card. Check logs.", severity="error")
        except ConflictError as ce:
            app.notify(f"Import conflict: {ce}", severity="warning", timeout=6)
            logger.warning(f"Conflict importing character card '{selected_path}': {ce}")
            await populate_ccp_character_select(app) # Refresh in case it was a duplicate name
        except ImportError as ie: # E.g., PyYAML missing for Markdown with frontmatter
            app.notify(f"Import error: {ie}. A required library might be missing.", severity="error", timeout=8)
            logger.error(f"Import error for character card '{selected_path}': {ie}", exc_info=True)
        except Exception as e:
            app.notify(f"Error importing character card: {type(e).__name__}", severity="error", timeout=6)
            logger.error(f"Error importing character card '{selected_path}': {e}", exc_info=True)
    else:
        logger.info("Character card import cancelled.")
        app.notify("Character import cancelled.", severity="information", timeout=2)

async def handle_ccp_import_character_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Import Character Card button pressed.")

    defined_filters = Filters(
        ("Character Card (JSON, PNG, WebP, MD)", lambda p: p.suffix.lower() in (".json", ".png", ".webp", ".md", ".yaml", ".yml")),
        ("JSON files (*.json)", lambda p: p.suffix.lower() == ".json"),
        ("PNG images (*.png)", lambda p: p.suffix.lower() == ".png"),
        ("WebP images (*.webp)", lambda p: p.suffix.lower() == ".webp"),
        ("Markdown files (*.md)", lambda p: p.suffix.lower() == ".md"),
        ("YAML files (*.yaml, *.yml)", lambda p: p.suffix.lower() in (".yaml", ".yml")),
        ("All files (*.*)", lambda p: True)
    )
    await app.push_screen(FileOpen(location=str(Path.home()), title="Select Character Card", filters=defined_filters),
                          # Use a lambda to capture `app` for the callback
                          callback=lambda path: _character_import_callback(app, path))


async def handle_ccp_left_load_character_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger = getattr(app, 'loguru_logger', logging) # Or however logger is typically obtained
    logger.info("CCP Load Selected Character button (left pane) pressed.")
    try:
        # Get the Select widget from the left pane
        char_select_widget = app.query_one("#conv-char-character-select", Select)
        selected_character_id = char_select_widget.value

        if selected_character_id == Select.BLANK or selected_character_id is None:
            app.notify("No character selected from the dropdown.", severity="warning")
            logger.warning("Load Character (left pane): No character selected.")
            return

        logger.info(f"Attempting to load character ID: {selected_character_id} into center editor view.")

        # Store the selected character ID (optional, but good for state tracking if needed later)
        # app.current_editing_character_id = selected_character_id

        # Switch the center pane view to the character editor
        app.ccp_active_view = "character_editor_view"
        # This will trigger the watcher in app.py to make #ccp-character-editor-view visible.
        # Population of fields will be handled in a later phase.

        # app.notify(f"Character (ID: {selected_character_id}) view activated. Editor population pending.", severity="information")

        if not app.notes_service:
            app.notify("Database service not available.", severity="error")
            logger.error("Notes service not available for loading character.")
            return

        db = app.notes_service._get_db(app.notes_user_id)

        try:
            # Load character data and image
            char_data, initial_ui_history, img = ccl.load_character_and_image(db, selected_character_id, app.notes_user_id)

            if char_data:
                app.current_ccp_character_details = char_data
                app.current_ccp_character_image = img  # Store the PIL Image object

                # Switch the center pane view to the character card display
                app.ccp_active_view = "character_card_view"

                app.notify(f"Displaying character: {char_data.get('name', 'Unknown')}", severity="information")
                logger.info(f"Character card for '{char_data.get('name', 'Unknown')}' loaded and view switched to card display.")
            else:
                app.notify("Failed to load character details.", severity="error")
                logger.warning(f"Failed to load character details for ID: {selected_character_id}")

        except CharactersRAGDBError as e_db_rag:
            logger.error(f"Database error loading character {selected_character_id}: {e_db_rag}", exc_info=True)
            app.notify(f"Database error loading character: {e_db_rag}", severity="error")
        except Exception as e_load_char: # Catch other potential errors from ccl.load_character_and_image
            logger.error(f"Error loading character data/image for ID {selected_character_id}: {e_load_char}", exc_info=True)
            app.notify(f"Error loading character: {type(e_load_char).__name__}", severity="error")

    except QueryError as e_query:
        logger.error(f"UI component not found during load character (left pane): {e_query}", exc_info=True)
        app.notify("Error: UI component missing for loading character.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error during load character (left pane): {e_unexp}", exc_info=True)
        app.notify("An unexpected error occurred while trying to load character view.", severity="error")


async def handle_ccp_create_character_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles creating a new character from scratch in the CCP tab."""
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.info("CCP Create Character button pressed.")
    
    # Clear the character editor fields
    await _helper_ccp_clear_center_pane_character_editor_fields(app)
    
    # Clear the current editing state
    app.current_editing_character_id = None
    app.current_editing_character_data = None
    
    # Switch to the character editor view
    app.ccp_active_view = "character_editor_view"
    
    # Make the Cancel Edit button visible
    try:
        cancel_button = app.query_one("#ccp-editor-char-cancel-button", Button)
        cancel_button.remove_class("hidden")
    except QueryError:
        logger.error("Failed to find #ccp-editor-char-cancel-button to remove 'hidden' class.")
    
    # Focus the character name input field
    try:
        name_input = app.query_one("#ccp-editor-char-name-input", Input)
        name_input.focus()
    except QueryError:
        logger.error("Failed to find #ccp-editor-char-name-input to focus.")
    
    app.notify("Create a new character by filling in the details.", severity="information")
    logger.info("Character editor cleared and ready for new character creation.")


async def handle_ccp_refresh_character_list_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the Refresh List button press to reload the character select dropdown."""
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.info("CCP Refresh Character List button pressed.")
    
    try:
        await populate_ccp_character_select(app)
        app.notify("Character list refreshed.", severity="information")
        logger.info("Character list refreshed successfully.")
    except Exception as e:
        logger.error(f"Error refreshing character list: {e}", exc_info=True)
        app.notify("Failed to refresh character list.", severity="error")


async def handle_ccp_card_save_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the Save Changes button press on the CCP Character Card view."""
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.info("CCP Card Save Character button pressed.")

    if not app.notes_service: # This should be app.chachanotes_db for character ops
        app.notify("Database service not available.", severity="error")
        logger.error("ChaChaNotes DB (via notes_service or direct) not available for saving character card.")
        return

    # Use cast to help the type checker understand the type of the reactive's value
    current_details_value = cast(Optional[Dict[str, Any]], app.current_ccp_character_details)

    if not current_details_value:
        app.notify("No character loaded or details missing. Cannot save.", severity="warning")
        logger.warning("Save Character Card: app.current_ccp_character_details value is None or empty dict.")
        return

    char_id = current_details_value.get("id")
    char_version = current_details_value.get("version")

    if not char_id or char_version is None:
        app.notify("Character ID or version missing from loaded details. Cannot save.", severity="warning")
        logger.warning(f"Save Character Card: ID ({char_id}) or Version ({char_version}) is missing.")
        return

    try:
        name_input = app.query_one("#ccp-card-name-input", Input)
        # For TextAreas on the card view, if they are read-only, their values
        # should reflect app.current_ccp_character_details. If they are editable,
        # then reading from them is correct. Assuming they are editable for this "Save" action.
        description_textarea = app.query_one("#ccp-card-description-display", TextArea)
        personality_textarea = app.query_one("#ccp-card-personality-display", TextArea)
        scenario_textarea = app.query_one("#ccp-card-scenario-display", TextArea)
        first_message_textarea = app.query_one("#ccp-card-first-message-display", TextArea)

        original_name = current_details_value.get("name", "") # Use the casted value

        update_data: Dict[str, Any] = { # Ensure type hint for update_data
            "name": name_input.value.strip(),
            "description": description_textarea.text.strip(),
            "personality": personality_textarea.text.strip(),
            "scenario": scenario_textarea.text.strip(),
            "first_message": first_message_textarea.text.strip(),
        }

        if not update_data["name"]:
            app.notify("Character Name cannot be empty.", severity="error")
            name_input.focus()
            return

        # db = app.notes_service._get_db(app.notes_user_id) # Old way
        if not app.chachanotes_db:
            app.notify("Character database not properly initialized.", severity="error")
            logger.error("app.chachanotes_db is not initialized for save.")
            return
        db = app.chachanotes_db


        updated_character_data = db.update_character_card(
            character_id=char_id, # Must be str
            update_data=update_data,
            expected_version=char_version # Must be int
        )

        if updated_character_data:
            app.notify(f"Character '{update_data['name']}' updated successfully.", severity="information")
            logger.info(f"Character ID {char_id} updated successfully. New version: {updated_character_data.get('version')}")
            app.current_ccp_character_details = updated_character_data

            if original_name != update_data["name"]:
                logger.info(f"Character name changed from '{original_name}' to '{update_data['name']}'. Refreshing dropdown.")
                await populate_ccp_character_select(app) # Ensure this is imported/defined
        else:
            app.notify("Failed to update character. DB operation did not confirm success.", severity="error")
            logger.error(f"Update for character ID {char_id} did not return updated data or confirmation.")

    except ConflictError as e_conflict:
        logger.warning(f"ConflictError saving character ID {char_id}: {e_conflict}", exc_info=True)
        app.notify("Save conflict: Data has been modified elsewhere. Please reload and try again.", severity="error", timeout=7)
    except QueryError as e_query:
        logger.error(f"UI component not found during save character card: {e_query}", exc_info=True)
        app.notify("Error: UI component missing for saving character.", severity="error")
    except CharactersRAGDBError as e_db:
        logger.error(f"CharactersRAGDBError saving character ID {char_id}: {e_db}", exc_info=True)
        app.notify(f"Error saving character: {type(e_db).__name__}. Check logs.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error saving character ID {char_id}: {e_unexp}", exc_info=True)
        app.notify(f"An unexpected error occurred: {type(e_unexp).__name__}. Check logs.", severity="error")


async def handle_ccp_card_clone_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the Clone Character button press on the CCP Character Card view."""
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.info("CCP Card Clone Character button pressed.")

    if not app.chachanotes_db:  # Check the correct DB instance
        app.notify("Database service not available.", severity="error")
        logger.error("ChaChaNotes DB (via app.chachanotes_db) not available for cloning character card.")
        return

    current_details_value = cast(Optional[Dict[str, Any]], app.current_ccp_character_details)

    if not current_details_value:
        app.notify("No character loaded to clone.", severity="warning")
        logger.warning("Clone Character Card: app.current_ccp_character_details value is None.")
        return

    try:
        # Get current data from UI fields to capture any unsaved edits for the clone
        # These UI elements are part of the "Character Card View"
        name_input_val = app.query_one("#ccp-card-name-display", Static).renderable  # If it's a Static for display
        # If #ccp-card-name-input is an Input field for editing on the card:
        # name_input_val = app.query_one("#ccp-card-name-input", Input).value

        # Check if name_input_val is Text or str for Static, or just str for Input
        original_name_from_ui = ""
        if isinstance(name_input_val, Text):
            original_name_from_ui = str(name_input_val).strip()
        elif isinstance(name_input_val, str):
            original_name_from_ui = name_input_val.strip()
        else:  # Fallback if it's neither, though unlikely for these widgets
            original_name_from_ui = str(current_details_value.get("name", "Unnamed")).strip()

        description_from_ui = app.query_one("#ccp-card-description-display", TextArea).text.strip()
        personality_from_ui = app.query_one("#ccp-card-personality-display", TextArea).text.strip()
        scenario_from_ui = app.query_one("#ccp-card-scenario-display", TextArea).text.strip()
        first_message_from_ui = app.query_one("#ccp-card-first-message-display", TextArea).text.strip()

        if not original_name_from_ui:
            app.notify("Original character name is empty. Cannot clone.", severity="error")
            # app.query_one("#ccp-card-name-input", Input).focus() # If it's an Input
            return

        cloned_name = f"{original_name_from_ui} Clone {datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Base data from UI (captures unsaved edits if card view is editable)
        cloned_character_data: Dict[str, Any] = {
            "name": cloned_name,
            "description": description_from_ui,
            "personality": personality_from_ui,
            "scenario": scenario_from_ui,
            "first_message": first_message_from_ui,
        }

        # Copy other fields from the currently loaded character details (current_details_value)
        # These fields are not directly editable on the card view, so use stored values.
        fields_to_copy_from_original = [
            'creator_notes', 'system_prompt', 'post_history_instructions',
            'alternate_greetings', 'tags', 'creator', 'character_version',
            # character_version might be reset by DB on add
            'extensions',  # 'image' needs special handling if it's bytes
        ]

        for field in fields_to_copy_from_original:
            if field in current_details_value:
                original_field_value = current_details_value[field]
                # Deep copy for mutable types, direct assignment for immutable
                if isinstance(original_field_value, list):
                    cloned_character_data[field] = list(original_field_value)
                elif isinstance(original_field_value, dict):
                    cloned_character_data[field] = dict(original_field_value)
                else:
                    cloned_character_data[field] = original_field_value

        # Handle 'image' field separately if it exists and is bytes
        if 'image' in current_details_value and isinstance(current_details_value['image'], bytes):
            cloned_character_data['image'] = current_details_value['image']  # Bytes are immutable enough for this copy

        db = app.chachanotes_db  # Already checked it's not None

        new_character_id = db.add_character_card(cloned_character_data)

        if new_character_id:
            app.notify(f"Character cloned as '{cloned_name}' successfully (ID: {new_character_id}).", severity="information")
            logger.info(
                f"Character cloned from ID {current_details_value.get('id')} to new ID {new_character_id} with name '{cloned_name}'.")

            await populate_ccp_character_select(app)  # Ensure this is imported/defined
        else:
            app.notify(f"Failed to clone character '{original_name_from_ui}'. DB operation did not confirm success.",
                       severity="error")
            logger.error(f"Cloning character '{original_name_from_ui}' did not return new character details or ID.")

    except ConflictError as e_conflict:
        logger.warning(f"ConflictError cloning character '{original_name_from_ui}': {e_conflict}", exc_info=True)
        app.notify(f"Save conflict while cloning: {e_conflict}. Please try again.", severity="error", timeout=7)
    except QueryError as e_query:
        logger.error(f"UI component not found during clone character card: {e_query}", exc_info=True)
        app.notify("Error: UI component missing for cloning character.", severity="error")
    except CharactersRAGDBError as e_db:
        logger.error(f"CharactersRAGDBError cloning character '{original_name_from_ui}': {e_db}", exc_info=True)
        app.notify(f"Error cloning character: {type(e_db).__name__}. Check logs.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error cloning character '{original_name_from_ui}': {e_unexp}", exc_info=True)
        app.notify(f"An unexpected error occurred while cloning: {type(e_unexp).__name__}. Check logs.",
                   severity="error")


async def handle_ccp_right_delete_character_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the Delete Character button press from the right pane (associated with card view)."""
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.info("CCP Right Pane Delete Character button pressed.")

    if not app.notes_service:
        app.notify("Database service not available.", severity="error")
        logger.error("Notes service not available for deleting character.")
        return

    if not app.current_ccp_character_details or not app.current_ccp_character_details.get('id'):
        app.notify("No character is currently loaded in the view to delete.", severity="warning")
        logger.warning("Delete Character (Right Pane): app.current_ccp_character_details is None or missing ID.")
        return

    character_id_to_delete = app.current_ccp_character_details.get('id')
    character_name_to_delete = app.current_ccp_character_details.get('name', 'The selected character')

    # Optional: Add a confirmation dialog here in a future iteration.
    # app.push_screen(ConfirmationDialog("Are you sure you want to delete...?"), callback=...)

    try:
        db = app.notes_service._get_db(app.notes_user_id)

        # Assuming db.delete_character_card returns True on success, False if not found/error.
        # Or it might raise an exception if character_id is not found, depending on DB layer design.
        # ChaChaNotes_DB.delete_character_card seems to return True/False.
        success = db.delete_character_card(character_id=character_id_to_delete)

        if success:
            app.notify(f"Character '{character_name_to_delete}' deleted successfully.", severity="information")
            logger.info(f"Character ID {character_id_to_delete} (Name: '{character_name_to_delete}') deleted successfully.")

            await populate_ccp_character_select(app) # Refresh dropdown

            # Clear the card view by resetting details and image, then switch view
            app.current_ccp_character_details = {} # Trigger watcher to clear UI
            app.current_ccp_character_image = None   # Trigger watcher for image

            # Switch to a default view. If the watcher for current_ccp_character_details
            # also hides/clears the #ccp-character-card-view, this might be redundant or a good fallback.
            app.ccp_active_view = "conversation_messages_view" # Or another appropriate default

            # Explicitly clear fields if watcher doesn't handle all of them for card view
            try:
                app.query_one("#ccp-card-name-input", Input).value = ""
                app.query_one("#ccp-card-description-display", TextArea).text = ""
                app.query_one("#ccp-card-personality-display", TextArea).text = ""
                app.query_one("#ccp-card-scenario-display", TextArea).text = ""
                app.query_one("#ccp-card-first-message-display", TextArea).text = ""
                # Image placeholder clearing is handled by app.current_ccp_character_image = None and its watcher.
            except QueryError as qe_clear:
                logger.warning(f"Could not explicitly clear some card UI fields after delete: {qe_clear}")

        else:
            # This case might be hit if db.delete_character_card returns False (e.g., char not found)
            app.notify(f"Failed to delete character '{character_name_to_delete}'. It might have already been deleted or an error occurred.", severity="error")
            logger.error(f"Delete operation for character ID {character_id_to_delete} (Name: '{character_name_to_delete}') returned False.")
            # Refresh dropdown anyway, in case the character was deleted by another process.
            await populate_ccp_character_select(app)


    except CharactersRAGDBError as e_db: # More specific DB errors
        logger.error(f"Database error deleting character ID {character_id_to_delete}: {e_db}", exc_info=True)
        app.notify(f"Error deleting character: {type(e_db).__name__}. Check logs.", severity="error")
    except QueryError as e_query: # If query_one for clearing fields fails
        logger.error(f"UI component error during delete character (likely clearing fields): {e_query}", exc_info=True)
        app.notify("UI error during character deletion process.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error deleting character ID {character_id_to_delete}: {e_unexp}", exc_info=True)
        app.notify(f"An unexpected error occurred while deleting: {type(e_unexp).__name__}. Check logs.", severity="error")


async def perform_ccp_conversation_search(app: 'TldwCli') -> None:
    """Performs conversation search for the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.debug("Performing CCP conversation search...")
    try:
        search_input = app.query_one("#conv-char-search-input", Input)
        search_term = search_input.value.strip()

        char_select_widget = app.query_one("#conv-char-character-select", Select)
        selected_character_id = char_select_widget.value
        if selected_character_id == Select.BLANK:  # Treat BLANK as None for filtering
            selected_character_id = None

        results_list_view = app.query_one("#conv-char-search-results-list", ListView)
        await results_list_view.clear()

        if not app.notes_service:
            logger.error("Notes service not available for CCP conversation search.")
            await results_list_view.append(ListItem(Label("Error: Notes service unavailable.")))
            return

        db = app.notes_service._get_db(app.notes_user_id)
        conversations: List[Dict[str, Any]] = []

        if selected_character_id:
            logger.debug(f"CCP Search: Filtering for character ID: {selected_character_id}")
            if search_term:
                logger.debug(f"CCP Search: Term '{search_term}', CharID {selected_character_id}")
                conversations = db.search_conversations_by_title(
                    title_query=search_term, character_id=selected_character_id, limit=200
                )
            else:
                logger.debug(f"CCP Search: All conversations for CharID {selected_character_id}")
                conversations = db.get_conversations_for_character(
                    character_id=selected_character_id, limit=200
                )
        else:
            logger.debug(f"CCP Search: No character selected. Global search with term: '{search_term}'")
            conversations = db.search_conversations_by_title(
                title_query=search_term, character_id=None, limit=200
            )

        if not conversations:
            msg = "Enter search term or select a character." if not search_term and not selected_character_id else "No items found matching your criteria."
            await results_list_view.append(ListItem(Label(msg)))
        else:
            for conv_data in conversations:
                base_title = conv_data.get('title') or f"Conversation ID: {conv_data['id'][:8]}..."
                display_title = base_title
                char_id_for_conv = conv_data.get('character_id')
                char_name_prefix = ""  # Initialize prefix

                if char_id_for_conv:  # If the conversation has an associated character
                    try:
                        # Always fetch character details from DB for consistent naming
                        char_details = db.get_character_card_by_id(char_id_for_conv)
                        if char_details and char_details.get('name'):
                            char_name_prefix = f"[{char_details['name']}] "
                    except Exception as e_char_name_fetch:
                        logger.warning(
                            f"Could not fetch char name for conv {conv_data['id']}, char_id {char_id_for_conv}: {e_char_name_fetch}",
                            exc_info=False)

                display_title = f"{char_name_prefix}{base_title}"
                item = ListItem(Label(display_title))
                item.details = conv_data
                await results_list_view.append(item)

        logger.info(
            f"CCP conversation search (Term: '{search_term}', CharID: {selected_character_id}) yielded {len(conversations)} results.")

    except QueryError as e_query:
        logger.error(f"UI component not found during CCP conversation search: {e_query}", exc_info=True)
        if 'results_list_view' in locals() and results_list_view.is_mounted:
            await results_list_view.clear()
            await results_list_view.append(ListItem(Label("Error: UI component missing.")))
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error during CCP conversation search: {e_db}", exc_info=True)
        if 'results_list_view' in locals() and results_list_view.is_mounted:
            await results_list_view.clear()
            await results_list_view.append(ListItem(Label("Error: Database search failed.")))
    except Exception as e_unexp:
        logger.error(f"Unexpected error during CCP conversation search: {e_unexp}", exc_info=True)
        if 'results_list_view' in locals() and results_list_view.is_mounted:
            await results_list_view.clear()
            await results_list_view.append(ListItem(Label("Error: Unexpected search failure.")))


#
########################################################################################################################
#
# Event Handlers (called by app.py)
#
########################################################################################################################
async def handle_ccp_conversation_search_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the search button press in the CCP tab's conversation section."""
    await perform_ccp_conversation_search(app)


async def handle_ccp_load_conversation_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles loading a selected conversation in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Load Conversation button pressed.")
    try:
        results_list_view = app.query_one("#conv-char-search-results-list", ListView)
        highlighted_item = results_list_view.highlighted_child

        if not (highlighted_item and hasattr(highlighted_item, 'details')):
            logger.warning("No conversation selected in CCP list or item has no details.")
            app.notify("Please select a conversation to load.", severity="warning")
            return

        conv_details_from_item = highlighted_item.details
        loaded_conversation_id = conv_details_from_item.get('id')

        if not loaded_conversation_id:
            logger.error("Selected item in CCP list is missing conversation ID.")
            app.notify("Selected item is invalid.", severity="error")
            return

        app.current_conv_char_tab_conversation_id = loaded_conversation_id
        logger.info(f"Current CCP tab conversation ID set to: {loaded_conversation_id}")

        title_input_ccp = app.query_one("#conv-char-title-input", Input)
        keywords_input_ccp = app.query_one("#conv-char-keywords-input", TextArea)
        title_input_ccp.value = conv_details_from_item.get('title', '')

        if app.notes_service:
            db = app.notes_service._get_db(app.notes_user_id)
            keywords_list_ccp = db.get_keywords_for_conversation(loaded_conversation_id)
            keywords_input_ccp.text = ", ".join(
                [kw['keyword'] for kw in keywords_list_ccp]) if keywords_list_ccp else ""
        else:
            keywords_input_ccp.text = "Service unavailable"

        app.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = False
        app.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True

        center_pane = app.query_one("#conv-char-center-pane", VerticalScroll)
        await center_pane.remove_children()

        if not app.notes_service:
            logger.error("Notes service not available for loading CCP messages.")
            await center_pane.mount(Static("Error: Notes service unavailable for messages."))
            return

        # FIXME/TODO - Conversation Branching
        await load_branched_conversation_history_ui(app, loaded_conversation_id, center_pane)

        center_pane.scroll_end(animate=False)
        logger.info(f"Loaded messages into #conv-char-center-pane for conversation {loaded_conversation_id}.")
        app.notify(f"Conversation '{title_input_ccp.value}' loaded.", severity="information")

    except QueryError as e_query:
        logger.error(f"UI component not found during CCP load conversation: {e_query}", exc_info=True)
        app.notify("Error: UI component missing for loading.", severity="error")
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error during CCP load conversation: {e_db}", exc_info=True)
        app.notify("Error loading conversation data from database.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error during CCP load conversation: {e_unexp}", exc_info=True)
        app.notify("An unexpected error occurred while loading conversation.", severity="error")


async def handle_ccp_save_conversation_details_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles saving conversation title/keywords in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Save Conversation Details button pressed.")
    if not app.current_conv_char_tab_conversation_id:
        logger.warning("No current conversation loaded in CCP tab to save details for.")
        app.notify("No conversation loaded to save details for.", severity="warning")
        return

    if not app.notes_service:
        logger.error("Notes service is not available for CCP save details.")
        app.notify("Database service not available.", severity="error")
        return

    try:
        title_input_ccp_save = app.query_one("#conv-char-title-input", Input)
        keywords_widget_ccp_save = app.query_one("#conv-char-keywords-input", TextArea)

        new_title_ccp = title_input_ccp_save.value.strip()
        new_keywords_str_ccp = keywords_widget_ccp_save.text.strip()
        target_conversation_id_ccp = app.current_conv_char_tab_conversation_id

        db = app.notes_service._get_db(app.notes_user_id)
        current_conv_details_ccp = db.get_conversation_by_id(target_conversation_id_ccp)

        if not current_conv_details_ccp:
            logger.error(f"CCP: Conversation {target_conversation_id_ccp} not found in DB for saving details.")
            app.notify("Error: Conversation not found in database.", severity="error")
            return

        current_db_version_ccp = current_conv_details_ccp.get('version')
        if current_db_version_ccp is None:
            logger.error(f"CCP: Conversation {target_conversation_id_ccp} is missing version information.")
            app.notify("Error: Conversation version information is missing.", severity="error")
            return

        title_updated_ccp = False
        if new_title_ccp != current_conv_details_ccp.get('title', ''):
            update_payload_ccp = {'title': new_title_ccp}
            db.update_conversation(conversation_id=target_conversation_id_ccp, update_data=update_payload_ccp,
                                   expected_version=current_db_version_ccp)
            title_updated_ccp = True
            current_db_version_ccp += 1
            await perform_ccp_conversation_search(app)

        existing_db_keywords_ccp = db.get_keywords_for_conversation(target_conversation_id_ccp)
        existing_kw_texts_set_ccp = {kw['keyword'].lower() for kw in existing_db_keywords_ccp}
        ui_kw_texts_set_ccp = {kw.strip().lower() for kw in new_keywords_str_ccp.split(',') if kw.strip()}

        keywords_to_add_ccp = ui_kw_texts_set_ccp - existing_kw_texts_set_ccp
        keywords_to_remove_details_ccp = [kw for kw in existing_db_keywords_ccp if
                                          kw['keyword'].lower() not in ui_kw_texts_set_ccp]
        keywords_changed_ccp = False

        for kw_text_add in keywords_to_add_ccp:
            # db.add_keyword returns keyword_id (int) for the given text (gets or creates)
            kw_id_add = db.add_keyword(kw_text_add)  # No user_id for global keywords
            if isinstance(kw_id_add, int):  # Check if it's an int ID
                db.link_conversation_to_keyword(conversation_id=target_conversation_id_ccp, keyword_id=kw_id_add)
                keywords_changed_ccp = True
            else:
                logger.error(f"Failed to get or add keyword '{kw_text_add}', received: {kw_id_add}")

        for kw_detail_remove in keywords_to_remove_details_ccp:
            db.unlink_conversation_from_keyword(conversation_id=target_conversation_id_ccp,
                                                keyword_id=kw_detail_remove['id'])
            keywords_changed_ccp = True

        if title_updated_ccp or keywords_changed_ccp:
            logger.info(f"CCP: Details saved for conversation {target_conversation_id_ccp}.")
            app.notify("Details saved successfully!", severity="information")
            final_keywords_list_ccp = db.get_keywords_for_conversation(target_conversation_id_ccp)
            keywords_widget_ccp_save.text = ", ".join(
                [kw['keyword'] for kw in final_keywords_list_ccp]) if final_keywords_list_ccp else ""
        else:
            app.notify("No changes to save.", severity="information")

    except ConflictError as e_conflict:
        logger.error(
            f"CCP: Conflict saving conversation details for {app.current_conv_char_tab_conversation_id}: {e_conflict}",
            exc_info=True)
        app.notify(f"Save conflict: {e_conflict}. Please reload.", severity="error")
    except QueryError as e_query:
        logger.error(f"CCP: UI component not found for saving details: {e_query}", exc_info=True)
        app.notify("Error accessing UI fields.", severity="error")
    except CharactersRAGDBError as e_db:
        logger.error(f"CCP: Database error saving details: {e_db}", exc_info=True)
        app.notify("Database error saving details.", severity="error")
    except Exception as e_unexp:
        logger.error(f"CCP: Unexpected error saving details: {e_unexp}", exc_info=True)
        app.notify("An unexpected error occurred.", severity="error")


async def handle_ccp_prompt_create_new_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None: # This one is for the RIGHT pane
    """Handles creating a new prompt in the CCP tab's RIGHT PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Create New Prompt button pressed.")
    clear_ccp_prompt_fields(app)
    try:
        app.query_one("#ccp-prompt-name-input", Input).value = "New Prompt"  # Default name in right pane
        author_name = app.app_config.get("user_defaults", {}).get("author_name", "User")
        app.query_one("#ccp-prompt-author-input", Input).value = author_name # In right pane

        app.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = False # Expand right pane prompt details
        app.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True # Collapse right pane conv details

        app.ccp_active_view = "conversation_details_view" # Ensure center pane is NOT on its editor view.
                                                        # Or, if you want to clear the center editor too:
                                                        # app.ccp_active_view = "prompt_editor_view"
                                                        # await app._clear_prompt_fields() # This clears center editor
        # Then perhaps switch back to conv view or focus right pane'
        app.current_prompt_id = None  # Signal that this is a new prompt

        app.query_one("#ccp-prompt-name-input", Input).focus() # Focus in right pane
        app.notify("Ready to create a new prompt in the right-side editor.", severity="information")
    except QueryError as e_query: # This can happen if the right sidebar is collapsed
        logger.error(f"CCP: UI error preparing for new prompt (right pane): {e_query}", exc_info=True)
        app.notify("UI error creating new prompt.", severity="error")

async def handle_ccp_center_pane_new_prompt_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """
    Handles the 'New Prompt' button that is intended to open the CENTER PANE editor.
    """
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Center Pane 'New Prompt' button pressed.")

    # 1. Signal the intent to edit a new prompt (clears any existing loaded prompt ID)
    app.current_prompt_id = None
    app.current_prompt_uuid = None

    # 2. Switch the center pane view to the prompt editor
    # The watcher `watch_ccp_active_view` in app.py will handle making the editor visible.
    app.ccp_active_view = "prompt_editor_view"
    # 3. Schedule the rest of the logic to run after the UI has updated.
    app.call_later(_finish_new_prompt_setup, app)


async def handle_ccp_prompt_load_selected_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles loading a selected prompt from the LEFT PANE list into the RIGHT PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Load Selected Prompt button pressed (loads into right pane editor).")
    try:
        list_view = app.query_one("#ccp-prompts-listview", ListView)
        selected_item = list_view.highlighted_child
        if selected_item and (hasattr(selected_item, 'prompt_id') or hasattr(selected_item, 'prompt_uuid')):
            prompt_id_to_load = getattr(selected_item, 'prompt_id', None)
            prompt_uuid_to_load = getattr(selected_item, 'prompt_uuid', None)
            # load_ccp_prompt_for_editing populates the RIGHT PANE editor
            await app._load_prompt_for_editing(prompt_id=prompt_id_to_load, prompt_uuid=prompt_uuid_to_load)
        else:
            app.notify("No prompt selected in the list.", severity="warning")
    except QueryError:
        app.notify("Prompt list not found.", severity="error")
    except Exception as e:  # Catch broader exceptions during load
        app.notify(f"Error loading prompt: {str(e)[:100]}", severity="error")  # Show part of error
        logger.error(f"Error loading prompt from list: {e}", exc_info=True)


async def handle_ccp_prompt_save_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles saving a new or existing prompt from the RIGHT PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Save Prompt button pressed (saves from right pane editor).")
    if not app.prompts_service_initialized:
        app.notify("Prompts service not available.", severity="error")
        return

    try:
        # Query fields from the RIGHT PANE editor
        name = app.query_one("#ccp-prompt-name-input", Input).value.strip()
        author = app.query_one("#ccp-prompt-author-input", Input).value.strip()
        details = app.query_one("#ccp-prompt-description-textarea", TextArea).text.strip()
        system_prompt = app.query_one("#ccp-prompt-system-textarea", TextArea).text.strip()
        user_prompt = app.query_one("#ccp-prompt-user-textarea", TextArea).text.strip()
        keywords_str = app.query_one("#ccp-prompt-keywords-textarea", TextArea).text.strip()
        keywords_list = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]

        if not name:
            app.notify("Prompt Name is required.", severity="error", timeout=4)
            app.query_one("#ccp-prompt-name-input", Input).focus() # Focus in right pane
            return

        saved_id: Optional[int] = None
        saved_uuid: Optional[str] = None
        message_from_save: str = ""

        if app.current_prompt_id is None: # This implies a new prompt being saved from the right pane editor
            logger.info(f"CCP: Saving new prompt from right pane: {name}")
            saved_id, saved_uuid, message_from_save = prompts_interop.add_prompt(
                name=name, author=author, details=details,
                system_prompt=system_prompt, user_prompt=user_prompt,
                keywords=keywords_list, overwrite=False
            )
        else: # Updating an existing prompt loaded into the right pane editor
            logger.info(f"CCP: Updating prompt ID (from right pane): {app.current_prompt_id}, Name: {name}")
            update_payload = {
                'name': name, 'author': author, 'details': details,
                'system_prompt': system_prompt, 'user_prompt': user_prompt,
                'keywords': keywords_list
            }
            # Assuming update_prompt_by_id takes the ID from app.current_prompt_id
            # and returns (uuid, message_str)
            updated_uuid, message_from_save = prompts_interop.get_db_instance().update_prompt_by_id(
                prompt_id=app.current_prompt_id,
                update_data=update_payload,
            ) # Removed version since it's not used by Prompts_DB update.
            saved_id = app.current_prompt_id
            saved_uuid = updated_uuid

        if saved_id is not None or saved_uuid is not None:
            app.notify(message_from_save, severity="information")
            await populate_ccp_prompts_list_view(app) # Refresh list in left pane
            # Reload the saved/updated prompt back into the RIGHT PANE editor to confirm changes and get new version
            await app._load_prompt_for_editing(prompt_id=saved_id, prompt_uuid=saved_uuid)
        else:
            app.notify(f"Failed to save prompt: {message_from_save or 'Unknown error'}", severity="error")

    except prompts_interop.InputError as e_in:
        app.notify(f"Input Error: {e_in}", severity="error", timeout=6)
    except prompts_interop.ConflictError as e_cf:
        app.notify(f"Save Conflict: {e_cf}", severity="error", timeout=6)
    except prompts_interop.DatabaseError as e_db:
        app.notify(f"Database Error: {e_db}", severity="error", timeout=6)
    except QueryError as e_query:
        logger.error(f"CCP Save Prompt (right pane): UI component error: {e_query}", exc_info=True)
        app.notify("UI Error saving prompt.", severity="error")
    except Exception as e_save:
        logger.error(f"CCP: Error saving prompt (right pane): {e_save}", exc_info=True)
        app.notify(f"Error saving prompt: {type(e_save).__name__}", severity="error")


async def handle_ccp_prompt_clone_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles cloning the currently loaded prompt in the RIGHT PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Clone Prompt button pressed (clones from right pane editor).")
    if not app.prompts_service_initialized or app.current_prompt_id is None:
        app.notify("No prompt loaded in right pane editor to clone or service unavailable.", severity="warning")
        return
    try:
        original_name = app.current_prompt_name or "Prompt"
        timestamp = datetime.now().strftime('%y%m%d%H%M%S')
        cloned_name = f"Clone of {original_name} ({timestamp})"[:100]

        cloned_id, cloned_uuid, msg_clone = prompts_interop.add_prompt(
            name=cloned_name,
            author=app.current_prompt_author or "",
            details=f"Clone of: {app.current_prompt_details or ''}",
            system_prompt=app.current_prompt_system or "",
            user_prompt=app.current_prompt_user or "",
            keywords=[kw.strip() for kw in (app.current_prompt_keywords_str or "").split(',') if kw.strip()],
            overwrite=False
        )
        if cloned_id:
            app.notify(f"Prompt cloned as '{cloned_name}'. {msg_clone}", severity="information")
            await populate_ccp_prompts_list_view(app) # Refresh list in left pane
            # Load the newly cloned prompt into the RIGHT PANE editor
            await app._load_prompt_for_editing(prompt_id=cloned_id, prompt_uuid=cloned_uuid)
        else:
            app.notify(f"Failed to clone prompt: {msg_clone}", severity="error")
    except Exception as e_clone:
        logger.error(f"CCP: Error cloning prompt (from right pane): {e_clone}", exc_info=True)
        app.notify(f"Error cloning prompt: {type(e_clone).__name__}", severity="error")


async def handle_ccp_prompt_delete_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles deleting (soft) the currently loaded prompt from the RIGHT PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Delete Prompt button pressed (deletes based on right pane editor state).")
    if not app.prompts_service_initialized or app.current_prompt_id is None:
        app.notify("No prompt loaded in right pane editor to delete or service unavailable.", severity="warning")
        return

    try:
        prompt_id_to_delete = app.current_prompt_id
        if prompt_id_to_delete is None:
            app.notify("Error: No prompt ID available for deletion.", severity="error")
            return

        # soft_delete_prompt in your DB class returns True on success, False if not found/already deleted
        # It raises ConflictError or DatabaseError on other issues.
        success = prompts_interop.soft_delete_prompt(prompt_id_to_delete) # client_id is handled internally

        if success:
            app.notify(f"Prompt '{app.current_prompt_name or 'selected'}' deleted.",
                       severity="information")
            clear_ccp_prompt_fields(app) # Clear right pane editor and reactives
            await populate_ccp_prompts_list_view(app) # Refresh list in left pane
            try: # Collapse details in right pane
                app.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True
            except QueryError:
                logger.warning("Could not find #ccp-prompt-details-collapsible to collapse after delete.")
        else:
            # This 'else' branch now correctly handles the "not found or already deleted" case
            app.notify(
                f"Failed to delete prompt '{app.current_prompt_name or 'selected'}'. It might have been already deleted or not found.",
                severity="error", timeout=7)
            # Refresh the list anyway to ensure UI consistency
            await populate_ccp_prompts_list_view(app)
            # If the prompt was not found, it makes sense to also clear the fields
            # as the currently displayed details might be for a non-existent prompt.
            # However, clear_ccp_prompt_fields also sets current_prompt_id to None,
            # which might be okay or you might want to just clear UI and keep ID if deletion failed for other reasons.
            # Given soft_delete_prompt returns False for "not found", clearing is appropriate.
            if app.current_prompt_id == prompt_id_to_delete: # Only clear if it was the one we tried to delete
                clear_ccp_prompt_fields(app)

    except prompts_interop.ConflictError as e_cf_del: # Specific exception from your DB class
        logger.error(f"CCP: Conflict error deleting prompt: {e_cf_del}", exc_info=True)
        app.notify(f"Conflict error deleting prompt: {e_cf_del}", severity="error")
        await populate_ccp_prompts_list_view(app) # Refresh list
    except prompts_interop.DatabaseError as e_db_del: # Specific exception from your DB class
        logger.error(f"CCP: Database error deleting prompt: {e_db_del}", exc_info=True)
        app.notify(f"Database error deleting prompt: {type(e_db_del).__name__}", severity="error")
    except Exception as e_del: # Catch any other unexpected exceptions
        logger.error(f"CCP: Unexpected error deleting prompt: {e_del}", exc_info=True)
        app.notify(f"Unexpected error deleting prompt: {type(e_del).__name__}", severity="error")


async def handle_ccp_conversation_search_input_changed(app: 'TldwCli', event: Input.Changed) -> None:
    """Handles input changes in the CCP conversation search bar with debouncing."""
    if app._conv_char_search_timer:
        app._conv_char_search_timer.stop()
    app._conv_char_search_timer = app.set_timer(0.5, lambda: perform_ccp_conversation_search(app))


async def handle_ccp_prompt_search_input_changed(app: 'TldwCli', event: Input.Changed) -> None:
    """Handles input changes in the CCP prompt search bar with debouncing."""
    search_term = event.value
    if app._prompt_search_timer:
        app._prompt_search_timer.stop()
    app._prompt_search_timer = app.set_timer(0.5, lambda: populate_ccp_prompts_list_view(app, search_term.strip()))


async def handle_ccp_character_select_changed(app: 'TldwCli', selected_value: Any) -> None:
    """Handles changes in the CCP character select dropdown."""
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.debug(f"CCP Character Select changed to: {selected_value}")
    await perform_ccp_conversation_search(app)


async def handle_ccp_prompts_list_view_selected(app: 'TldwCli', list_view_id: str, item: Any) -> None:
    """Handles selecting a prompt from the list in the CCP tab (loads to RIGHT pane)."""
    logger = getattr(app, 'loguru_logger', loguru_logger)
    if item and (hasattr(item, 'prompt_id') or hasattr(item, 'prompt_uuid')):
        prompt_id_to_load = getattr(item, 'prompt_id', None)
        prompt_uuid_to_load = getattr(item, 'prompt_uuid', None)
        logger.info(f"CCP Prompt selected from list: ID={prompt_id_to_load}, UUID={prompt_uuid_to_load}. Loading to right pane.")
        # load_ccp_prompt_for_editing loads into the RIGHT PANE editor
        await app._load_prompt_for_editing(prompt_id=prompt_id_to_load, prompt_uuid=prompt_uuid_to_load)
    else:
        logger.debug("CCP Prompts ListView selection was empty or item lacked ID/UUID.")


########################################################################################################################
#
# Event Handlers for Conversation Import
#
########################################################################################################################
async def _conversation_import_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    logger = getattr(app, 'loguru_logger', loguru_logger)
    if selected_path:
        logger.info(f"Conversation import selected: {selected_path}")
        if not app.notes_service:
            app.notify("Database service not available.", severity="error")
            logger.error("Notes service not available for conversation import.")
            return

        db = app.notes_service._get_db(app.notes_user_id)
        try:
            # ccl.load_chat_history_from_file_and_save_to_db expects a string path or BytesIO
            # It returns (conversation_id, character_id)
            conv_id, _ = ccl.load_chat_history_from_file_and_save_to_db(
                db,
                str(selected_path),
                user_name_for_placeholders=app.notes_user_id # Or a more appropriate placeholder
            )
            if conv_id:
                app.notify(f"Conversation imported successfully (ID: {conv_id}).", severity="information")
                await perform_ccp_conversation_search(app) # Refresh conversation list
            else:
                app.notify("Failed to import conversation. Check logs.", severity="error")
        except Exception as e:
            app.notify(f"Error importing conversation: {type(e).__name__}", severity="error", timeout=6)
            logger.error(f"Error importing conversation from '{selected_path}': {e}", exc_info=True)
    else:
        logger.info("Conversation import cancelled.")
        app.notify("Conversation import cancelled.", severity="information", timeout=2)

async def handle_ccp_import_conversation_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.info("CCP Import Conversation button pressed.")

    defined_filters = Filters(
        ("Chat Log (JSON)", lambda p: p.suffix.lower() == ".json"),
        ("All files (*.*)", lambda p: True)
    )
    await app.push_screen(
        FileOpen(location=str(Path.home()), title="Select Conversation File", filters=defined_filters),
        callback=lambda path: _conversation_import_callback(app, path))

########################################################################################################################
#
# Event Handlers for Prompt Import
#
########################################################################################################################
def _parse_prompt_from_file_content(file_content_str: str) -> Optional[Dict[str, Any]]:
    """
    Parses prompt data from string content.
    Tries JSON, then YAML, then a custom plain text format.
    """
    logger = loguru_logger
    if not file_content_str:
        return None
    # 1. Try JSON
    try:
        parsed_json = json.loads(file_content_str)
        if isinstance(parsed_json, dict) and parsed_json.get("name"):
            logger.debug("Successfully parsed prompt file content as JSON.")
            return parsed_json
        else:
            logger.debug("Content parsed as JSON but not a valid prompt structure (missing 'name' or not a dict).")
    except json.JSONDecodeError:
        logger.debug("Content is not valid JSON. Trying YAML.")
    except Exception as e_json_other:
        logger.warning(f"Unexpected error during JSON parsing attempt: {e_json_other}")

    # 2. Try YAML
    try:
        parsed_yaml = yaml.safe_load(file_content_str)
        if isinstance(parsed_yaml, dict) and parsed_yaml.get("name"):
            logger.debug("Successfully parsed prompt file content as YAML.")
            return parsed_yaml
        else:
            logger.debug("Content parsed as YAML but not a valid prompt structure (missing 'name' or not a dict).")
    except yaml.YAMLError:
        logger.debug("Content is not valid YAML. Trying custom plain text format.")
    except Exception as e_yaml_other:
        logger.warning(f"Unexpected error during YAML parsing attempt: {e_yaml_other}")

    # 3. Try Custom Plain Text Format
    logger.debug("Attempting to parse prompt file content as custom plain text format.")
    try:
        parsed_custom_data: Dict[str, Any] = {}
        # Regex to find sections: ### SECTION_NAME ###\nContent until next ### or EOF
        # Using re.DOTALL so . matches newlines.
        # Using re.IGNORECASE for section headers.
        # Capture group 1 is the section name, group 2 is the content.
        # This regex assumes sections are separated by ### SECTION_NAME ###
        # and the content is everything until the next ### or end of file.

        # A simpler approach based on your format:
        # Split by ###, then process pairs.
        # Example: "### TITLE ###\nSAMPLE PROMPT\n### AUTHOR ###\nrmusser01"
        # sections_raw = re.split(r'\s*###\s*([A-Z\s])\s*###\s*', file_content_str.strip(), flags=re.IGNORECASE)
        # Filter out empty strings that result from splitting at the start/end
        # sections_filtered = [s.strip() for s in sections_raw if s and s.strip()]

        # More robust parsing:
        header_map = {
            "TITLE": "name",
            "AUTHOR": "author",
            "SYSTEM": "system_prompt",
            "USER": "user_prompt",
            "KEYWORDS": "keywords_str",  # Will be split later
            "DETAILS": "details",
        }

        # Use re.finditer to find all section blocks
        # Pattern: ### SECTION_NAME ###\n(content up to next ### or EOF)
        pattern = r"^\s*###\s*("  "|".join(header_map.keys())
        r")\s*###\s*\n(.*?)(?=(?:\n\s*###|$))"
        for match in re.finditer(pattern, file_content_str, re.MULTILINE | re.DOTALL | re.IGNORECASE):
            section_name_from_file = match.group(1).upper()
            section_content = match.group(2).strip()
            dict_key = header_map.get(section_name_from_file)
            if dict_key:
                parsed_custom_data[dict_key] = section_content

        # Post-process keywords if they were read as a string
        if "keywords_str" in parsed_custom_data:
            parsed_custom_data["keywords"] = [
                kw.strip() for kw in parsed_custom_data["keywords_str"].split(',') if kw.strip()
            ]
            del parsed_custom_data["keywords_str"]  # remove the temp string version
        elif "keywords" not in parsed_custom_data:  # Ensure keywords key exists
            parsed_custom_data["keywords"] = []

        # If "name" (from TITLE) is missing after regex parsing, try to infer it from the first non-header line
        if "name" not in parsed_custom_data or not parsed_custom_data["name"]:
            first_few_lines = file_content_str.strip().split('\n', 5)
            potential_title_line = ""
            if first_few_lines:
                if first_few_lines[0].upper().strip().startswith("### TITLE ###"):
                    if len(first_few_lines) > 1 and not first_few_lines[1].upper().strip().startswith("###"):
                        potential_title_line = first_few_lines[1].strip()
                elif not first_few_lines[0].upper().strip().startswith("###"):
                    potential_title_line = first_few_lines[0].strip()
            if potential_title_line:
                parsed_custom_data["name"] = potential_title_line

        if "name" in parsed_custom_data and parsed_custom_data["name"]:
            logger.debug(
                f"Successfully parsed prompt file content as custom plain text. Data: {parsed_custom_data}")
            # Ensure all expected fields for add_prompt are present, defaulting to None or empty list
            final_data_for_add = {
                "name": parsed_custom_data.get("name"),
                "author": parsed_custom_data.get("author"),
                "details": parsed_custom_data.get("details"),
                "system_prompt": parsed_custom_data.get("system_prompt"),
                "user_prompt": parsed_custom_data.get("user_prompt"),
                "keywords": parsed_custom_data.get("keywords", []),
            }
            return final_data_for_add
        else:
            logger.debug("Custom plain text parsing did not yield a valid 'name' or was empty.")
    except Exception as e_custom:
        logger.error(f"Error parsing prompt file as custom plain text: {e_custom}", exc_info=True)
        return None
        # If all parsing attempts fail
    logger.error("All parsing attempts for prompt file failed.")
    return None

async def _prompt_import_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    logger = getattr(app, 'loguru_logger', logging)
    if selected_path:
        logger.info(f"Prompt import selected: {selected_path}")
        if not app.prompts_service_initialized:
            app.notify("Prompts service not available.", severity="error")
            return

        try:
            with open(selected_path, 'r', encoding='utf-8') as f:
                content = f.read()

            prompt_data = _parse_prompt_from_file_content(content)
            if not prompt_data:
                app.notify("Failed to parse prompt file. Invalid format or missing 'name'.", severity="error", timeout=7)
                return

            # prompts_interop.add_prompt expects individual arguments.
            # We need to extract them from the prompt_data dictionary.
            # It also returns (id, uuid, message_str)
            prompt_id, prompt_uuid, msg = prompts_interop.add_prompt(
                name=prompt_data.get("name", "Unnamed Prompt"), # name is required by _parse_prompt...
                author=prompt_data.get("author"),
                details=prompt_data.get("details"),
                system_prompt=prompt_data.get("system_prompt"),
                user_prompt=prompt_data.get("user_prompt"),
                keywords=prompt_data.get("keywords"), # Expects List[str] or None
                overwrite=False # Default to not overwriting, or make configurable
            )

            if prompt_id is not None or prompt_uuid is not None : # Check if save was successful (id or uuid exists)
                app.notify(f"Prompt imported: {msg}", severity="information")
                await populate_ccp_prompts_list_view(app) # Refresh prompt list
            else:
                app.notify(f"Failed to import prompt: {msg}", severity="error")
        except FileNotFoundError:
            app.notify(f"Prompt file not found: {selected_path}", severity="error")
            logger.error(f"Prompt file not found: {selected_path}")
        except Exception as e:
            app.notify(f"Error importing prompt: {type(e).__name__}", severity="error", timeout=6)
            logger.error(f"Error importing prompt from '{selected_path}': {e}", exc_info=True)
    else:
        logger.info("Prompt import cancelled.")
        app.notify("Prompt import cancelled.", severity="information", timeout=2)

async def handle_ccp_import_prompt_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Import Prompt button pressed.")

    defined_filters = Filters(
        ("Prompt files (TXT, MD, JSON, YAML)", lambda p: p.suffix.lower() in (".txt", ".md", ".json", ".yaml", ".yml")),
        ("JSON files (*.json)", lambda p: p.suffix.lower() == ".json"),
        ("YAML files (*.yaml, *.yml)", lambda p: p.suffix.lower() in (".yaml", ".yml")),
        ("Text files (*.txt)", lambda p: p.suffix.lower() == ".txt"),
        ("Markdown files (*.md)", lambda p: p.suffix.lower() == ".md"),
        ("All files (*.*)", lambda p: True)
    )
    await app.push_screen(FileOpen(location=str(Path.home()), title="Select Prompt File", filters=defined_filters),
                          callback=lambda path: _prompt_import_callback(app, path))

########################################################################################################################
# CCP Center Pane Editor Button Handlers (these operate on the #ccp-editor-* prefixed IDs)
########################################################################################################################
async def handle_ccp_editor_prompt_save_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles saving a new or existing prompt from the CENTER PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Editor Save Prompt button pressed (saves from center pane editor).")
    if not app.prompts_service_initialized:
        app.notify("Prompts service not available.", severity="error")
        return

    try:
        # Query fields from the CENTER PANE editor
        name = app.query_one("#ccp-editor-prompt-name-input", Input).value.strip()
        author = app.query_one("#ccp-editor-prompt-author-input", Input).value.strip()
        details = app.query_one("#ccp-editor-prompt-description-textarea", TextArea).text.strip()
        system_prompt = app.query_one("#ccp-editor-prompt-system-textarea", TextArea).text.strip()
        user_prompt = app.query_one("#ccp-editor-prompt-user-textarea", TextArea).text.strip()
        keywords_str = app.query_one("#ccp-editor-prompt-keywords-textarea", TextArea).text.strip()
        keywords_list = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]

        if not name:
            app.notify("Prompt Name is required for editor save.", severity="error", timeout=4)
            app.query_one("#ccp-editor-prompt-name-input", Input).focus()
            return

        saved_id: Optional[int] = None
        saved_uuid: Optional[str] = None
        message_from_save: str = ""

        # current_prompt_id, etc. are used to track the prompt loaded into *either* editor.
        # If the center pane editor is active, these reactives should reflect that.
        if app.current_prompt_id is None: # Saving a NEW prompt via center editor
            logger.info(f"CCP Editor: Saving new prompt: {name}")
            saved_id, saved_uuid, message_from_save = prompts_interop.add_prompt(
                name=name, author=author, details=details,
                system_prompt=system_prompt, user_prompt=user_prompt,
                keywords=keywords_list, overwrite=False
            )
        else: # Updating an EXISTING prompt via center editor
            logger.info(f"CCP Editor: Updating prompt ID: {app.current_prompt_id}, Name: {name}")
            update_payload = {
                'name': name, 'author': author, 'details': details,
                'system_prompt': system_prompt, 'user_prompt': user_prompt,
                'keywords': keywords_list
            }
            updated_uuid, message_from_save = prompts_interop.get_db_instance().update_prompt_by_id(
                prompt_id=app.current_prompt_id,
                update_data=update_payload,
            )
            saved_id = app.current_prompt_id
            saved_uuid = updated_uuid

        if saved_id is not None or saved_uuid is not None:
            app.notify(f"Prompt Editor: {message_from_save}", severity="information")
            await populate_ccp_prompts_list_view(app) # Refresh list in left pane
            # After saving from center editor, reload it into the center editor.
            # _load_prompt_for_editing already handles setting reactives and populating center editor.
            await app._load_prompt_for_editing(prompt_id=saved_id, prompt_uuid=saved_uuid)
        else:
            app.notify(f"Prompt Editor: Failed to save prompt: {message_from_save or 'Unknown error'}", severity="error")

    except prompts_interop.InputError as e_in:
        app.notify(f"Editor Input Error: {e_in}", severity="error", timeout=6)
    except prompts_interop.ConflictError as e_cf:
        app.notify(f"Editor Save Conflict: {e_cf}", severity="error", timeout=6)
    except prompts_interop.DatabaseError as e_db:
        app.notify(f"Editor Database Error: {e_db}", severity="error", timeout=6)
    except QueryError as e_query:
        logger.error(f"CCP Editor Save Prompt: UI component error: {e_query}", exc_info=True)
        app.notify("UI Error saving prompt from editor.", severity="error")
    except Exception as e_save:
        logger.error(f"CCP Editor: Error saving prompt: {e_save}", exc_info=True)
        app.notify(f"Editor: Error saving prompt: {type(e_save).__name__}", severity="error")


async def handle_ccp_editor_prompt_clone_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles cloning the prompt currently in the CENTER PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Editor Clone Prompt button pressed.")
    if not app.prompts_service_initialized or app.current_prompt_id is None:
        app.notify("No prompt loaded in editor to clone or service unavailable.", severity="warning")
        return

    try:
        # Assume current_prompt_* reactives hold the state of the editor
        original_name = app.current_prompt_name or "Prompt"
        timestamp = datetime.now().strftime('%y%m%d%H%M%S')
        cloned_name = f"Clone of {original_name} ({timestamp})"[:100]

        cloned_id, cloned_uuid, msg_clone = prompts_interop.add_prompt(
            name=cloned_name,
            author=app.current_prompt_author or "",
            details=f"Clone of: {app.current_prompt_details or ''}",
            system_prompt=app.current_prompt_system or "",
            user_prompt=app.current_prompt_user or "",
            keywords=[kw.strip() for kw in (app.current_prompt_keywords_str or "").split(',') if kw.strip()],
            overwrite=False
        )
        if cloned_id:
            app.notify(f"Editor: Prompt cloned as '{cloned_name}'. {msg_clone}", severity="information")
            await populate_ccp_prompts_list_view(app)
            # Load the newly cloned prompt into the CENTER PANE editor
            await app._load_prompt_for_editing(prompt_id=cloned_id, prompt_uuid=cloned_uuid)
        else:
            app.notify(f"Editor: Failed to clone prompt: {msg_clone}", severity="error")
    except Exception as e_clone:
        logger.error(f"CCP Editor: Error cloning prompt: {e_clone}", exc_info=True)
        app.notify(f"Editor: Error cloning prompt: {type(e_clone).__name__}", severity="error")


async def handle_ccp_editor_prompt_delete_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles deleting the prompt currently in the CENTER PANE editor."""
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.info("CCP Editor Delete Prompt button pressed.")
    if not app.prompts_service_initialized or app.current_prompt_id is None:
        app.notify("No prompt loaded in editor to delete or service unavailable.", severity="warning")
        return

    try:
        prompt_id_to_delete = app.current_prompt_id
        if prompt_id_to_delete is None:
            app.notify("Editor: No prompt ID available for deletion.", severity="error")
            return

        success = prompts_interop.soft_delete_prompt(prompt_id_to_delete)

        if success:
            app.notify(f"Editor: Prompt '{app.current_prompt_name or 'selected'}' deleted.", severity="information")
            app._clear_prompt_fields() # Clear center editor and reactives
            await populate_ccp_prompts_list_view(app)
            app.ccp_active_view = "conversation_details_view" # Switch view back
        else:
            app.notify(f"Editor: Failed to delete prompt. It might have been already deleted.", severity="error")
            await populate_ccp_prompts_list_view(app)
            if app.current_prompt_id == prompt_id_to_delete: # If it was the one we tried to delete
                app._clear_prompt_fields()
                app.ccp_active_view = "conversation_details_view"

    except prompts_interop.ConflictError as e_cf_del:
        logger.error(f"CCP Editor: Conflict deleting prompt: {e_cf_del}", exc_info=True)
        app.notify(f"Editor: Conflict error deleting prompt: {e_cf_del}", severity="error")
    except prompts_interop.DatabaseError as e_db_del:
        logger.error(f"CCP Editor: Database error deleting prompt: {e_db_del}", exc_info=True)
        app.notify(f"Editor: Database error deleting prompt: {type(e_db_del).__name__}", severity="error")
    except Exception as e_del:
        logger.error(f"CCP Editor: Unexpected error deleting prompt: {e_del}", exc_info=True)
        app.notify(f"Editor: Unexpected error deleting prompt: {type(e_del).__name__}", severity="error")

# ##############################################################
# --- CCP Center Pane Editor Button Handlers End ---
# ##############################################################

async def handle_ccp_editor_char_save_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles saving a new or existing character from the CENTER PANE editor."""
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.info("CCP Editor: Save Character button pressed.")

    if not app.chachanotes_db:
        app.notify("Database service not available for characters.", severity="error")
        logger.error("app.chachanotes_db not available for saving character.")
        return
    db = app.chachanotes_db

    try:
        # Retrieve character data from the UI input fields
        char_name = app.query_one("#ccp-editor-char-name-input", Input).value.strip()
        avatar_path = app.query_one("#ccp-editor-char-avatar-input", Input).value.strip()
        description = app.query_one("#ccp-editor-char-description-textarea", TextArea).text.strip()
        personality = app.query_one("#ccp-editor-char-personality-textarea", TextArea).text.strip()
        scenario = app.query_one("#ccp-editor-char-scenario-textarea", TextArea).text.strip()
        first_message = app.query_one("#ccp-editor-char-first-message-textarea", TextArea).text.strip()
        keywords_text = app.query_one("#ccp-editor-char-keywords-textarea", TextArea).text.strip()
        keywords_list = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
        
        # V2 Character Card fields
        creator_notes = app.query_one("#ccp-editor-char-creator-notes-textarea", TextArea).text.strip()
        system_prompt = app.query_one("#ccp-editor-char-system-prompt-textarea", TextArea).text.strip()
        post_history_instructions = app.query_one("#ccp-editor-char-post-history-instructions-textarea", TextArea).text.strip()
        alternate_greetings_text = app.query_one("#ccp-editor-char-alternate-greetings-textarea", TextArea).text.strip()
        alternate_greetings = [greeting.strip() for greeting in alternate_greetings_text.split('\n') if greeting.strip()]
        tags_text = app.query_one("#ccp-editor-char-tags-input", Input).value.strip()
        tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
        creator = app.query_one("#ccp-editor-char-creator-input", Input).value.strip()
        character_version = app.query_one("#ccp-editor-char-version-input", Input).value.strip()

        if not char_name:
            app.notify("Character Name cannot be empty.", severity="error", timeout=4)
            app.query_one("#ccp-editor-char-name-input", Input).focus()
            return

        character_data_for_db_op: Dict[str, Any] = {
            "name": char_name,
            "description": description,
            "personality": personality,
            "scenario": scenario,
            "first_message": first_message,
            "keywords": keywords_list,
            "image_path": avatar_path,  # Storing avatar path as image_path
            # V2 Character Card fields
            "creator_notes": creator_notes,
            "system_prompt": system_prompt,
            "post_history_instructions": post_history_instructions,
            "alternate_greetings": alternate_greetings,
            "tags": tags,
            "creator": creator,
            "character_version": character_version,
            "extensions": {}  # Empty dict for now, can be extended later
        }

        saved_character_details: Optional[Dict[str, Any]] = None
        current_editing_id = app.current_editing_character_id
        current_editing_data = cast(Optional[Dict[str, Any]], app.current_editing_character_data)

        if current_editing_id is None:  # New character
            logger.info(f"Attempting to add new character: {char_name}")
            new_character_id = db.add_character_card(card_data=character_data_for_db_op)
            if new_character_id:
                logger.info(f"New character '{char_name}' added. ID: {new_character_id}")
                app.notify(f"Character '{char_name}' saved successfully.", severity="information")
                # Update the app state with the new character ID
                app.current_editing_character_id = str(new_character_id)
                app.current_editing_character_data = character_data_for_db_op
                app.current_editing_character_data["id"] = str(new_character_id)
            else:
                logger.error(f"Failed to save new character '{char_name}'. DB response: {new_character_id}")
                app.notify(f"Failed to save new character '{char_name}'.", severity="error")
                return
        else:  # Existing character
            logger.info(f"Attempting to update character ID: {current_editing_id}, Name: {char_name}")
            if not current_editing_data:
                logger.error(f"Cannot update character {current_editing_id}: current editing data is missing.")
                app.notify("Error: Current character data is missing. Please reload.", severity="error")
                return

            current_version = current_editing_data.get('version')
            if current_version is None:
                logger.error(f"Cannot update character {current_editing_id}: version is missing from loaded data.")
                app.notify("Error: Character version is missing. Please reload and try again.", severity="error")
                return

            saved_character_details = db.update_character_card(
                character_id=current_editing_id,
                update_data=character_data_for_db_op,
                expected_version=current_version
            )
            if saved_character_details:
                logger.info(f"Character '{char_name}' (ID: {current_editing_id}) updated successfully.")
                app.notify(f"Character '{char_name}' updated successfully.", severity="information")
            else:
                logger.error(f"Failed to update character '{char_name}'. DB response: {saved_character_details}")
                app.notify(f"Failed to update character '{char_name}'.", severity="error")
                return

        if saved_character_details and saved_character_details.get("id"):
            new_char_id = saved_character_details["id"]
            # Reload the character into the editor to reflect any changes and update state
            await _helper_ccp_load_character_into_center_pane_editor(app, new_char_id)
            await populate_ccp_character_select(app)  # Refresh dropdown list
            try:
                cancel_button = app.query_one("#ccp-editor-char-cancel-button", Button)
                cancel_button.add_class("hidden")
            except QueryError:
                logger.error("Failed to find #ccp-editor-char-cancel-button to add 'hidden' class post-save.")
        else:
            # This case should ideally not be reached if errors are returned above.
            logger.warning("Save/Update operation completed but no valid character details received.")
            # Optionally, try to reload current if update failed but no specific error caught
            if current_editing_id:
                 await _helper_ccp_load_character_into_center_pane_editor(app, current_editing_id)


    except ConflictError as e_conflict:
        logger.warning(f"Conflict saving character '{char_name}': {e_conflict}", exc_info=True)
        app.notify(f"Save conflict: Data was modified elsewhere. Please reload and try again.", severity="error", timeout=7)
        if app.current_editing_character_id: # Reload to show current state from DB
            await _helper_ccp_load_character_into_center_pane_editor(app, app.current_editing_character_id)
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error saving character '{char_name}': {e_db}", exc_info=True)
        app.notify(f"Database error saving character: {type(e_db).__name__}", severity="error")
    except QueryError as e_query:
        logger.error(f"UI component error saving character: {e_query}", exc_info=True)
        app.notify("UI Error: Could not access character editor fields.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error saving character '{char_name}': {e_unexp}", exc_info=True)
        app.notify(f"An unexpected error occurred: {type(e_unexp).__name__}", severity="error")

# ##############################################################
# --- CCP Center Pane Editor Clearance ---
# ##############################################################
async def _helper_ccp_clear_center_pane_character_editor_fields(app: 'TldwCli') -> None:
    """Clears character input fields in the CCP CENTER PANE editor."""
    try:
        # Assuming these are the IDs for the CENTER PANE character editor
        app.query_one("#ccp-editor-char-name-input", Input).value = ""
        app.query_one("#ccp-editor-char-avatar-input", Input).value = ""
        app.query_one("#ccp-editor-char-description-textarea", TextArea).text = ""
        app.query_one("#ccp-editor-char-personality-textarea", TextArea).text = ""
        app.query_one("#ccp-editor-char-scenario-textarea", TextArea).text = ""
        app.query_one("#ccp-editor-char-first-message-textarea", TextArea).text = ""
        app.query_one("#ccp-editor-char-keywords-textarea", TextArea).text = ""
        # V2 Character Card fields
        app.query_one("#ccp-editor-char-creator-notes-textarea", TextArea).text = ""
        app.query_one("#ccp-editor-char-system-prompt-textarea", TextArea).text = ""
        app.query_one("#ccp-editor-char-post-history-instructions-textarea", TextArea).text = ""
        app.query_one("#ccp-editor-char-alternate-greetings-textarea", TextArea).text = ""
        app.query_one("#ccp-editor-char-tags-input", Input).value = ""
        app.query_one("#ccp-editor-char-creator-input", Input).value = ""
        app.query_one("#ccp-editor-char-version-input", Input).value = ""

        loguru_logger.debug("Cleared character editor fields in CCP center pane.")
    except QueryError as e:
        loguru_logger.error(f"Error clearing CCP center pane character editor fields: {e}")
    except Exception as e_clear:
        loguru_logger.error(f"Unexpected error clearing CCP center pane character editor fields: {e_clear}", exc_info=True)

async def _helper_ccp_load_character_into_center_pane_editor(app: 'TldwCli', character_id: str) -> None:
    """Loads character details into the CCP CENTER PANE editor and updates app state."""
    if not app.chachanotes_db: # Use the correct DB instance
        app.notify("Character database service not available.", severity="error")
        loguru_logger.error("ChaChaNotes DB not available for loading character into editor.")
        return

    # Ensure the Character Editor view in the center pane is active
    app.ccp_active_view = "character_editor_view" # This triggers the watcher in app.py

    try:
        # Use ccl (Character_Chat_Lib) to load the character data
        # Assuming load_character_and_image is the correct function from ccl
        # and it returns (character_data_dict, initial_ui_history_list, PIL_Image_object_or_None)
        char_data, _, char_image_pil = ccl.load_character_and_image(
            app.chachanotes_db,
            character_id,
            app.notes_user_id # Assuming this is the correct user context
        )

        if char_data:
            app.current_editing_character_id = char_data.get("id")
            app.current_editing_character_data = char_data
            # app.current_ccp_character_image = char_image_pil # If you want to display the image in the editor too

            # Populate UI elements in the CENTER PANE character editor
            app.query_one("#ccp-editor-char-name-input", Input).value = char_data.get("name", "")
            app.query_one("#ccp-editor-char-avatar-input", Input).value = char_data.get("image_path", char_data.get("avatar", "")) # Use image_path or avatar
            app.query_one("#ccp-editor-char-description-textarea", TextArea).text = char_data.get("description", "")
            app.query_one("#ccp-editor-char-personality-textarea", TextArea).text = char_data.get("personality", "")
            app.query_one("#ccp-editor-char-scenario-textarea", TextArea).text = char_data.get("scenario", "")
            app.query_one("#ccp-editor-char-first-message-textarea", TextArea).text = char_data.get("first_message", char_data.get("first_mes",""))
            keywords_list = char_data.get("keywords", [])
            app.query_one("#ccp-editor-char-keywords-textarea", TextArea).text = ", ".join(keywords_list) if keywords_list else ""

            # Populate V2 Character Card fields
            app.query_one("#ccp-editor-char-creator-notes-textarea", TextArea).text = char_data.get("creator_notes", "")
            app.query_one("#ccp-editor-char-system-prompt-textarea", TextArea).text = char_data.get("system_prompt", "")
            app.query_one("#ccp-editor-char-post-history-instructions-textarea", TextArea).text = char_data.get("post_history_instructions", "")
            alternate_greetings = char_data.get("alternate_greetings", [])
            app.query_one("#ccp-editor-char-alternate-greetings-textarea", TextArea).text = "\n".join(alternate_greetings) if alternate_greetings else ""
            tags_list = char_data.get("tags", [])
            app.query_one("#ccp-editor-char-tags-input", Input).value = ", ".join(tags_list) if tags_list else ""
            app.query_one("#ccp-editor-char-creator-input", Input).value = char_data.get("creator", "")
            app.query_one("#ccp-editor-char-version-input", Input).value = char_data.get("character_version", "")

            app.query_one("#ccp-editor-char-name-input", Input).focus()
            app.notify(f"Character '{char_data.get('name', 'Unknown')}' loaded into center editor.", severity="information")
            loguru_logger.info(f"Loaded character '{char_data.get('name', 'Unknown')}' (ID: {char_data.get('id')}) into CCP center editor.")
        else:
            app.notify(f"Failed to load character details for ID: {character_id} into editor.", severity="error")
            await _helper_ccp_clear_center_pane_character_editor_fields(app)
            app.current_editing_character_id = None
            app.current_editing_character_data = None
            loguru_logger.warning(f"Character details not found for ID {character_id} for editor.")

    except CharactersRAGDBError as e_db:
        loguru_logger.error(f"Database error loading character for CCP editing (ID: {character_id}): {e_db}", exc_info=True)
        app.notify(f"Database error loading character: {type(e_db).__name__}", severity="error")
        await _helper_ccp_clear_center_pane_character_editor_fields(app)
        app.current_editing_character_id = None
        app.current_editing_character_data = None
    except QueryError as e_query:
        loguru_logger.error(f"UI component error loading character for CCP editing (ID: {character_id}): {e_query}", exc_info=True)
        app.notify("UI Error: Could not populate character editor fields.", severity="error")
    except Exception as e:
        loguru_logger.error(f"Unexpected error loading character for CCP editing (ID: {character_id}): {e}", exc_info=True)
        app.notify(f"Error loading character into editor: {type(e).__name__}", severity="error")
        await _helper_ccp_clear_center_pane_character_editor_fields(app)
        app.current_editing_character_id = None
        app.current_editing_character_data = None


async def handle_ccp_editor_char_clone_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.info("CCP Editor: Clone Character button pressed.")

    if not app.chachanotes_db:
        app.notify("Database service not available.", severity="error")
        logger.error("ChaChaNotes DB not available for cloning character.")
        return

    # Use the reactive for the character currently in the EDITOR
    current_editing_id = app.current_editing_character_id
    current_editing_data_value = cast(Optional[Dict[str, Any]], app.current_editing_character_data)


    if current_editing_id is None or current_editing_data_value is None:
        app.notify("No character loaded in editor to clone. Please load a character first.", severity="warning", timeout=7)
        logger.warning("Clone Character from editor: No character data loaded in editor's state.")
        return

    try:
        # Data for cloning should come from the current state of app.current_editing_character_data,
        # which should reflect what's loaded in the editor.
        # If you want to clone based on potentially unsaved UI changes in the editor,
        # then you'd query the UI fields like in the save handler.
        # For simplicity, let's assume we clone the *saved* state held in current_editing_character_data.

        original_data = current_editing_data_value # This is already Optional[Dict[str, Any]]

        original_name = original_data.get("name", "Character")
        timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
        cloned_name = f"{original_name} Clone {timestamp}"[:255]

        # Prepare data for the new character, excluding 'id' and 'version'
        cloned_character_data_for_db: Dict[str, Any] = {
            key: value for key, value in original_data.items() if key not in ['id', 'version', 'uuid', 'created_at', 'last_modified']
        }
        cloned_character_data_for_db["name"] = cloned_name # Set the new name

        # Deep copy mutable fields if they exist
        if 'alternate_greetings' in cloned_character_data_for_db and isinstance(cloned_character_data_for_db['alternate_greetings'], list):
            cloned_character_data_for_db['alternate_greetings'] = list(cloned_character_data_for_db['alternate_greetings'])
        if 'tags' in cloned_character_data_for_db and isinstance(cloned_character_data_for_db['tags'], list):
            cloned_character_data_for_db['tags'] = list(cloned_character_data_for_db['tags'])
        if 'extensions' in cloned_character_data_for_db and isinstance(cloned_character_data_for_db['extensions'], dict):
            cloned_character_data_for_db['extensions'] = dict(cloned_character_data_for_db['extensions'])


        logger.info(f"Attempting to clone character '{original_name}' as '{cloned_name}' from editor state.")
        db = app.chachanotes_db # Already checked

        new_cloned_char_id = db.add_character_card(
            card_data=cloned_character_data_for_db
        )

        if new_cloned_char_id:
            logger.info(f"Character cloned successfully. New ID: {new_cloned_char_id}")
            app.notify(f"Character cloned as '{cloned_name}'.", severity="information")

            await _helper_ccp_load_character_into_center_pane_editor(app, str(new_cloned_char_id))
            await populate_ccp_character_select(app)
        else:
            logger.error(f"Failed to save cloned character '{cloned_name}'. DB returned: {new_cloned_char_id}")
            app.notify(f"Failed to clone character '{original_name}'.", severity="error")

    except CharactersRAGDBError as e_db:
        logger.error(f"Database error cloning character from editor: {e_db}", exc_info=True)
        app.notify(f"Database error cloning: {type(e_db).__name__}", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error cloning character from editor: {e_unexp}", exc_info=True)
        app.notify(f"Unexpected error cloning: {type(e_unexp).__name__}", severity="error")


async def handle_ccp_editor_char_cancel_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles cancelling an edit in the CCP CENTER PANE character editor."""
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.info("CCP Editor: Cancel Character Edit button pressed.")

    try:
        cancel_button = app.query_one("#ccp-editor-char-cancel-button", Button)
        cancel_button.add_class("hidden")

        if app.current_editing_character_id is not None:
            # An existing character was being edited. Restore card view with original data.
            stored_character_id = app.current_editing_character_id
            logger.info(f"Cancelling edit for existing character ID: {stored_character_id}. Restoring card view.")

            if not app.chachanotes_db:
                app.notify("Database service not available to restore character.", severity="error")
                logger.error("ChaChaNotes DB not available to restore character for card view.")
                # Attempt to clear editor and switch to a neutral view anyway
                await _helper_ccp_clear_center_pane_character_editor_fields(app)
                app.current_editing_character_id = None
                app.current_editing_character_data = None
                app.ccp_active_view = "conversation_messages_view" # Fallback view
                return

            try:
                # Fetch the original, unmodified character data for the card view
                original_char_data, _, original_char_image_pil = ccl.load_character_and_image(
                    app.chachanotes_db,
                    stored_character_id,
                    app.notes_user_id # Assuming this is the correct user context
                )

                if original_char_data:
                    # Update the state for the character card view
                    app.current_ccp_character_details = original_char_data
                    app.current_ccp_character_image = original_char_image_pil

                    # Switch view to the character card display
                    app.ccp_active_view = "character_card_view"
                    app.notify("Character editing cancelled. Displaying original card.", severity="information")
                else:
                    # Failed to reload original data, fallback to clearing editor and neutral view
                    app.notify("Could not reload original character data. Clearing editor.", severity="warning")
                    logger.warning(f"Failed to reload original data for char ID {stored_character_id} on cancel.")
                    await _helper_ccp_clear_center_pane_character_editor_fields(app)
                    app.ccp_active_view = "conversation_messages_view" # Fallback

            except Exception as e_load:
                logger.error(f"Error reloading original character data (ID: {stored_character_id}) on cancel: {e_load}", exc_info=True)
                app.notify("Error restoring character view. Clearing editor.", severity="error")
                await _helper_ccp_clear_center_pane_character_editor_fields(app)
                app.ccp_active_view = "conversation_messages_view" # Fallback

            # Clear the editor's specific state
            app.current_editing_character_id = None
            app.current_editing_character_data = None
            # Optionally, explicitly clear editor fields if not already done in error paths
            # await _helper_ccp_clear_center_pane_character_editor_fields(app) # This might be redundant if view always changes

        else:
            # A new character form was being edited. Clear fields and switch view.
            logger.info("Cancelling creation of new character. Clearing fields and switching view.")
            await _helper_ccp_clear_center_pane_character_editor_fields(app)
            # Ensure state reflects no character is being edited
            app.current_editing_character_id = None
            app.current_editing_character_data = None
            # Switch to a view that makes sense, e.g., where the user might have come from
            # If there was a previously viewed card, character_card_view might be okay,
            # otherwise, conversation_messages_view is a general default.
            if app.current_ccp_character_details and app.current_ccp_character_details.get("id"):
                app.ccp_active_view = "character_card_view" # Show previous card if one was loaded
            else:
                app.ccp_active_view = "conversation_messages_view" # General fallback
            app.notify("New character creation cancelled.", severity="information")

    except QueryError as e_query:
        logger.error(f"UI component error during cancel character edit (querying cancel button): {e_query}", exc_info=True)
        app.notify("UI Error: Could not properly cancel character edit.", severity="error")
        # Attempt to recover by switching to a default view
        app.ccp_active_view = "conversation_messages_view"
    except Exception as e_unexp:
        logger.error(f"Unexpected error during cancel character edit: {e_unexp}", exc_info=True)
        app.notify(f"An unexpected error occurred: {type(e_unexp).__name__}", severity="error")
        # Attempt to recover
        app.ccp_active_view = "conversation_messages_view"


async def handle_ccp_editor_char_delete_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.info("CCP Editor: Delete Character button pressed.")

    if not app.chachanotes_db:
        app.notify("Database service not available.", severity="error")
        logger.error("ChaChaNotes DB not available for deleting character.")
        return

    character_id_to_delete = app.current_editing_character_id
    current_editing_data_value = cast(Optional[Dict[str, Any]], app.current_editing_character_data)
    character_name_to_delete = current_editing_data_value.get("name", "the loaded character") if current_editing_data_value else "the loaded character"

    if character_id_to_delete is None:
        app.notify("No character loaded in the editor to delete.", severity="warning")
        logger.warning("Delete Character from editor: No character ID loaded.")
        return

    try:
        logger.info(f"Attempting to delete character ID from editor: {character_id_to_delete} (Name: '{character_name_to_delete}').")
        db = app.chachanotes_db # Already checked

        success = db.delete_character_card(character_id=character_id_to_delete) # delete_character_card expects string ID

        if success:
            logger.info(f"Character {character_id_to_delete} deleted successfully from editor.")
            app.notify(f"Character '{character_name_to_delete}' deleted.", severity="information")

            await _helper_ccp_clear_center_pane_character_editor_fields(app)
            app.current_editing_character_id = None # Reset reactives
            app.current_editing_character_data = None
            app.ccp_active_view = "conversation_messages_view"
            await populate_ccp_character_select(app)
        else:
            logger.error(f"Failed to delete character '{character_name_to_delete}' (ID: {character_id_to_delete}) from editor. DB returned False.")
            app.notify(f"Failed to delete character '{character_name_to_delete}'. It might have already been deleted.", severity="error", timeout=7)
            await populate_ccp_character_select(app)

    except ConflictError as e_conflict:
        logger.error(f"Conflict error deleting character {character_id_to_delete} from editor: {e_conflict}", exc_info=True)
        app.notify(f"Conflict deleting character: {e_conflict}", severity="error")
        await populate_ccp_character_select(app)
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error deleting character {character_id_to_delete} from editor: {e_db}", exc_info=True)
        app.notify(f"Database error deleting: {type(e_db).__name__}", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error deleting character {character_id_to_delete} from editor: {e_unexp}", exc_info=True)
        app.notify(f"Unexpected error deleting: {type(e_unexp).__name__}", severity="error")


async def _finish_new_prompt_setup(app: 'TldwCli') -> None:
    """
    This second-stage helper is called after the UI has had a chance to update.
    It safely populates the now-visible center pane editor.
    """
    # Now that the view is visible, we can safely clear and populate its fields.
    # The _load_prompt_for_editing method handles querying the correct editor fields.
    await app._load_prompt_for_editing(prompt_id=None) # Call with None to clear fields for a new prompt
    app.notify("Ready to create a new prompt in the center editor.", severity="information")
# ##############################################################
# --- CCP Right Pane Editor Button Handlers ---
# ##############################################################
async def handle_ccp_tab_sidebar_toggle(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles sidebar toggles specific to the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    button_id = event.button.id
    if button_id == "toggle-conv-char-left-sidebar":
        app.conv_char_sidebar_left_collapsed = not app.conv_char_sidebar_left_collapsed
        logger.debug("CCP left sidebar now %s", "collapsed" if app.conv_char_sidebar_left_collapsed else "expanded")
    elif button_id == "toggle-conv-char-right-sidebar":
        app.conv_char_sidebar_right_collapsed = not app.conv_char_sidebar_right_collapsed
        logger.debug("CCP right sidebar now %s", "collapsed" if app.conv_char_sidebar_right_collapsed else "expanded")
    else:
        logger.warning(f"Unhandled sidebar toggle button ID '{button_id}' in CCP tab handler.")


async def handle_ccp_card_edit_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger = getattr(app, 'loguru_logger', loguru_logger)
    logger.info("CCP Card Edit Character button pressed.")

    current_card_details = cast(Optional[Dict[str, Any]], app.current_ccp_character_details)
    if not current_card_details or not current_card_details.get("id"):
        app.notify("No character loaded on the card to edit.", severity="warning")
        return

    character_id_to_edit = current_card_details.get("id")
    if not character_id_to_edit:
         app.notify("Character ID missing from card details.", severity="error")
         return

    logger.info(f"Requesting to load character ID {character_id_to_edit} into editor.")
    await _helper_ccp_load_character_into_center_pane_editor(app, character_id_to_edit)
    # This helper will set app.current_editing_character_id, app.current_editing_character_data,
    # and app.ccp_active_view = "character_editor_view"
    try:
        cancel_button = app.query_one("#ccp-editor-char-cancel-button", Button)
        cancel_button.remove_class("hidden")
    except QueryError:
        logger.error("Failed to find #ccp-editor-char-cancel-button to remove 'hidden' class.")


# --- Button Handler Map ---
CCP_BUTTON_HANDLERS = {
    # Left Pane
    "ccp-import-character-button": handle_ccp_import_character_button_pressed,
    "ccp-create-character-button": handle_ccp_create_character_button_pressed,
    "ccp-refresh-character-list-button": handle_ccp_refresh_character_list_button_pressed,
    "ccp-import-conversation-button": handle_ccp_import_conversation_button_pressed,
    "ccp-import-prompt-button": handle_ccp_import_prompt_button_pressed,
    "conv-char-conversation-search-button": handle_ccp_conversation_search_button_pressed,
    "conv-char-load-button": handle_ccp_load_conversation_button_pressed,
    "ccp-prompt-create-new-button": handle_ccp_center_pane_new_prompt_button_pressed,
    "ccp-prompt-load-selected-button": handle_ccp_prompt_load_selected_button_pressed,
    "ccp-right-pane-load-character-button": handle_ccp_left_load_character_button_pressed,

    # Center Pane
    "ccp-card-edit-button": handle_ccp_card_edit_button_pressed,
    "ccp-card-save-button": handle_ccp_card_save_button_pressed,
    "ccp-card-clone-button": handle_ccp_card_clone_button_pressed,
    "ccp-editor-char-save-button": handle_ccp_editor_char_save_button_pressed,
    "ccp-editor-char-delete-button": handle_ccp_editor_char_delete_button_pressed,
    "ccp-editor-char-clone-button": handle_ccp_editor_char_clone_button_pressed,
    "ccp-editor-char-cancel-button": handle_ccp_editor_char_cancel_button_pressed,
    "ccp-editor-prompt-save-button": handle_ccp_editor_prompt_save_button_pressed,
    "ccp-prompt-clone-button": handle_ccp_prompt_clone_button_pressed,
    "ccp-prompt-delete-button": handle_ccp_prompt_delete_button_pressed,
    "ccp-editor-prompt-clone-button": handle_ccp_editor_prompt_clone_button_pressed,
    "ccp-prompt-save-button": handle_ccp_prompt_save_button_pressed,

    # Right Pane
    "conv-char-save-details-button": handle_ccp_save_conversation_details_button_pressed,
    "ccp-editor-prompt-delete-button": handle_ccp_editor_prompt_delete_button_pressed,
    "ccp-character-delete-button": handle_ccp_right_delete_character_button_pressed,

    # Sidebar Toggles
    "toggle-conv-char-left-sidebar": handle_ccp_tab_sidebar_toggle,
    "toggle-conv-char-right-sidebar": handle_ccp_tab_sidebar_toggle,
}

#
# End of conv_char_events.py
########################################################################################################################
