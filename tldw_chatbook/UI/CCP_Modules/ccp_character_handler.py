"""Handler for character-related operations in the CCP window."""

from typing import TYPE_CHECKING, Optional, Dict, Any, List
from loguru import logger
from textual import work
from textual.widgets import Select, Button, Input, TextArea, Static
import json
import base64
from pathlib import Path

from .ccp_messages import CharacterMessage, ViewChangeMessage

if TYPE_CHECKING:
    from ..Conv_Char_Window import CCPWindow

logger = logger.bind(module="CCPCharacterHandler")


class CCPCharacterHandler:
    """Handles all character-related operations for the CCP window."""
    
    def __init__(self, window: 'CCPWindow'):
        """Initialize the character handler.
        
        Args:
            window: Reference to the parent CCP window
        """
        self.window = window
        self.app_instance = window.app_instance
        self.current_character_id: Optional[int] = None
        self.current_character_data: Dict[str, Any] = {}
        self.pending_image_data: Optional[str] = None
        
        logger.debug("CCPCharacterHandler initialized")
    
    async def refresh_character_list(self) -> None:
        """Refresh the character select dropdown."""
        try:
            from ...Character_Chat.Character_Chat_Lib import fetch_character_names
            
            # Get character names
            characters = fetch_character_names()
            
            # Update the select widget
            character_select = self.window.query_one("#conv-char-character-select", Select)
            
            # Convert to Select options format
            options = [(name, str(char_id)) for char_id, name in characters.items()]
            
            # Update the select widget
            character_select.set_options(options)
            
            logger.info(f"Refreshed character list with {len(options)} characters")
            
        except Exception as e:
            logger.error(f"Error refreshing character list: {e}", exc_info=True)
    
    async def handle_load_character(self) -> None:
        """Load the selected character."""
        try:
            character_select = self.window.query_one("#conv-char-character-select", Select)
            
            if character_select.value:
                character_id = int(character_select.value)
                await self.load_character(character_id)
            else:
                logger.warning("No character selected to load")
                
        except Exception as e:
            logger.error(f"Error loading selected character: {e}", exc_info=True)
    
    @work(thread=True)
    async def load_character(self, character_id: int) -> None:
        """Load a character and display the card.
        
        Args:
            character_id: The ID of the character to load
        """
        logger.info(f"Loading character {character_id}")
        
        try:
            from ...Character_Chat.Character_Chat_Lib import fetch_character_card_by_id
            
            # Load the character card
            card_data = fetch_character_card_by_id(character_id)
            
            if card_data:
                self.current_character_id = character_id
                self.current_character_data = card_data
                
                # Post message for other components
                self.window.post_message(
                    CharacterMessage.Loaded(character_id, card_data)
                )
                
                # Switch view to show character card
                self.window.post_message(
                    ViewChangeMessage.Requested("character_card", {"character_id": character_id})
                )
                
                # Update UI on main thread
                self.window.call_from_thread(self._display_character_card)
                
                logger.info(f"Character {character_id} loaded successfully")
            else:
                logger.error(f"Failed to load character {character_id}")
                
        except Exception as e:
            logger.error(f"Error loading character {character_id}: {e}", exc_info=True)
    
    def _display_character_card(self) -> None:
        """Display character card in the UI."""
        try:
            if not self.current_character_data:
                return
            
            data = self.current_character_data
            
            # Update all the character card display fields
            self._update_field("#ccp-card-name-display", data.get("name", "N/A"))
            self._update_textarea("#ccp-card-description-display", data.get("description", ""))
            self._update_textarea("#ccp-card-personality-display", data.get("personality", ""))
            self._update_textarea("#ccp-card-scenario-display", data.get("scenario", ""))
            self._update_textarea("#ccp-card-first-message-display", data.get("first_message", ""))
            
            # V2 fields
            self._update_textarea("#ccp-card-creator-notes-display", data.get("creator_notes", ""))
            self._update_textarea("#ccp-card-system-prompt-display", data.get("system_prompt", ""))
            self._update_textarea("#ccp-card-post-history-instructions-display", 
                                data.get("post_history_instructions", ""))
            
            # Handle alternate greetings
            alternate_greetings = data.get("alternate_greetings", [])
            if alternate_greetings:
                greetings_text = "\n".join(alternate_greetings)
                self._update_textarea("#ccp-card-alternate-greetings-display", greetings_text)
            
            # Handle tags
            tags = data.get("tags", [])
            self._update_field("#ccp-card-tags-display", ", ".join(tags) if tags else "None")
            
            # Other metadata
            self._update_field("#ccp-card-creator-display", data.get("creator", "N/A"))
            self._update_field("#ccp-card-version-display", data.get("character_version", "N/A"))
            
            # Keywords
            keywords = data.get("keywords", [])
            self._update_field("#ccp-card-keywords-display", ", ".join(keywords) if keywords else "None")
            
            # Handle image display
            self._display_character_image(data)
            
            logger.debug(f"Displayed character card for {data.get('name', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Error displaying character card: {e}", exc_info=True)
    
    def _update_field(self, selector: str, value: str) -> None:
        """Update a Static field."""
        try:
            widget = self.window.query_one(selector, Static)
            widget.update(value)
        except Exception as e:
            logger.warning(f"Could not update field {selector}: {e}")
    
    def _update_textarea(self, selector: str, value: str) -> None:
        """Update a TextArea field."""
        try:
            widget = self.window.query_one(selector, TextArea)
            widget.text = value
        except Exception as e:
            logger.warning(f"Could not update textarea {selector}: {e}")
    
    def _display_character_image(self, data: Dict[str, Any]) -> None:
        """Display character image if available."""
        try:
            image_placeholder = self.window.query_one("#ccp-card-image-placeholder", Static)
            
            # Check for base64 image data
            if data.get("image"):
                # In a real implementation, we'd render the image
                # For now, just indicate an image is present
                image_placeholder.update("📷 Character Image")
            elif data.get("avatar"):
                # URL to avatar
                image_placeholder.update(f"🔗 Avatar: {data['avatar'][:50]}...")
            else:
                image_placeholder.update("No image")
                
        except Exception as e:
            logger.warning(f"Could not display character image: {e}")
    
    async def handle_edit_character(self) -> None:
        """Switch to character editor view."""
        if not self.current_character_data:
            logger.warning("No character loaded to edit")
            return
        
        try:
            # Switch view to editor
            self.window.post_message(
                ViewChangeMessage.Requested("character_editor", 
                                          {"character_id": self.current_character_id})
            )
            
            # Populate editor fields
            self._populate_editor_fields()
            
        except Exception as e:
            logger.error(f"Error switching to character editor: {e}", exc_info=True)
    
    def _populate_editor_fields(self) -> None:
        """Populate the character editor fields with current data."""
        try:
            data = self.current_character_data
            
            # Basic fields
            self._set_input_value("#ccp-editor-char-name-input", data.get("name", ""))
            self._set_textarea_value("#ccp-editor-char-description-textarea", data.get("description", ""))
            self._set_textarea_value("#ccp-editor-char-personality-textarea", data.get("personality", ""))
            self._set_textarea_value("#ccp-editor-char-scenario-textarea", data.get("scenario", ""))
            self._set_textarea_value("#ccp-editor-char-first-message-textarea", data.get("first_message", ""))
            
            # Keywords
            keywords = data.get("keywords", [])
            self._set_textarea_value("#ccp-editor-char-keywords-textarea", ", ".join(keywords))
            
            # V2 fields
            self._set_textarea_value("#ccp-editor-char-creator-notes-textarea", data.get("creator_notes", ""))
            self._set_textarea_value("#ccp-editor-char-system-prompt-textarea", data.get("system_prompt", ""))
            self._set_textarea_value("#ccp-editor-char-post-history-instructions-textarea", 
                                   data.get("post_history_instructions", ""))
            
            # Alternate greetings
            alternate_greetings = data.get("alternate_greetings", [])
            self._set_textarea_value("#ccp-editor-char-alternate-greetings-textarea", 
                                   "\n".join(alternate_greetings))
            
            # Tags
            tags = data.get("tags", [])
            self._set_input_value("#ccp-editor-char-tags-input", ", ".join(tags))
            
            # Metadata
            self._set_input_value("#ccp-editor-char-creator-input", data.get("creator", ""))
            self._set_input_value("#ccp-editor-char-version-input", data.get("character_version", ""))
            
            # Avatar URL
            self._set_input_value("#ccp-editor-char-avatar-input", data.get("avatar", ""))
            
        except Exception as e:
            logger.error(f"Error populating editor fields: {e}", exc_info=True)
    
    def _set_input_value(self, selector: str, value: str) -> None:
        """Set an Input widget's value."""
        try:
            widget = self.window.query_one(selector, Input)
            widget.value = value
        except Exception as e:
            logger.warning(f"Could not set input {selector}: {e}")
    
    def _set_textarea_value(self, selector: str, value: str) -> None:
        """Set a TextArea widget's value."""
        try:
            widget = self.window.query_one(selector, TextArea)
            widget.text = value
        except Exception as e:
            logger.warning(f"Could not set textarea {selector}: {e}")
    
    async def handle_save_character(self) -> None:
        """Save the character from editor."""
        try:
            # Gather all field values
            character_data = self._gather_editor_data()
            
            if self.current_character_id:
                # Update existing character
                await self._update_character(self.current_character_id, character_data)
            else:
                # Create new character
                await self._create_character(character_data)
                
        except Exception as e:
            logger.error(f"Error saving character: {e}", exc_info=True)
    
    def _gather_editor_data(self) -> Dict[str, Any]:
        """Gather all data from the editor fields."""
        data = {}
        
        try:
            # Basic fields
            data["name"] = self.window.query_one("#ccp-editor-char-name-input", Input).value
            data["description"] = self.window.query_one("#ccp-editor-char-description-textarea", TextArea).text
            data["personality"] = self.window.query_one("#ccp-editor-char-personality-textarea", TextArea).text
            data["scenario"] = self.window.query_one("#ccp-editor-char-scenario-textarea", TextArea).text
            data["first_message"] = self.window.query_one("#ccp-editor-char-first-message-textarea", TextArea).text
            
            # Keywords
            keywords_text = self.window.query_one("#ccp-editor-char-keywords-textarea", TextArea).text
            data["keywords"] = [k.strip() for k in keywords_text.split(",") if k.strip()]
            
            # V2 fields
            data["creator_notes"] = self.window.query_one("#ccp-editor-char-creator-notes-textarea", TextArea).text
            data["system_prompt"] = self.window.query_one("#ccp-editor-char-system-prompt-textarea", TextArea).text
            data["post_history_instructions"] = self.window.query_one("#ccp-editor-char-post-history-instructions-textarea", TextArea).text
            
            # Alternate greetings
            greetings_text = self.window.query_one("#ccp-editor-char-alternate-greetings-textarea", TextArea).text
            data["alternate_greetings"] = [g.strip() for g in greetings_text.split("\n") if g.strip()]
            
            # Tags
            tags_text = self.window.query_one("#ccp-editor-char-tags-input", Input).value
            data["tags"] = [t.strip() for t in tags_text.split(",") if t.strip()]
            
            # Metadata
            data["creator"] = self.window.query_one("#ccp-editor-char-creator-input", Input).value
            data["character_version"] = self.window.query_one("#ccp-editor-char-version-input", Input).value
            
            # Avatar URL
            data["avatar"] = self.window.query_one("#ccp-editor-char-avatar-input", Input).value
            
            # Include pending image data if available
            if self.pending_image_data:
                data["image"] = self.pending_image_data
            
        except Exception as e:
            logger.error(f"Error gathering editor data: {e}", exc_info=True)
        
        return data
    
    @work(thread=True)
    async def _update_character(self, character_id: int, data: Dict[str, Any]) -> None:
        """Update an existing character."""
        try:
            from ...Character_Chat.Character_Chat_Lib import update_character_card
            
            success = update_character_card(character_id, data)
            
            if success:
                logger.info(f"Updated character {character_id}")
                
                # Post update message
                self.window.post_message(
                    CharacterMessage.Updated(character_id, data)
                )
                
                # Refresh the character list
                self.window.call_from_thread(self.refresh_character_list)
            else:
                logger.error(f"Failed to update character {character_id}")
                
        except Exception as e:
            logger.error(f"Error updating character: {e}", exc_info=True)
    
    @work(thread=True)
    async def _create_character(self, data: Dict[str, Any]) -> None:
        """Create a new character."""
        try:
            from ...Character_Chat.Character_Chat_Lib import add_character_card
            
            character_id = add_character_card(data)
            
            if character_id:
                logger.info(f"Created new character with ID {character_id}")
                
                # Post creation message
                self.window.post_message(
                    CharacterMessage.Created(character_id, data.get("name", ""), data)
                )
                
                # Refresh the character list
                self.window.call_from_thread(self.refresh_character_list)
                
                # Set as current character
                self.current_character_id = character_id
                self.current_character_data = data
            else:
                logger.error("Failed to create new character")
                
        except Exception as e:
            logger.error(f"Error creating character: {e}", exc_info=True)
    
    async def handle_delete_character(self) -> None:
        """Delete the current character."""
        if not self.current_character_id:
            logger.warning("No character selected to delete")
            return
        
        try:
            from ...Character_Chat.Character_Chat_Lib import delete_character_card
            
            success = delete_character_card(self.current_character_id)
            
            if success:
                logger.info(f"Deleted character {self.current_character_id}")
                
                # Post deletion message
                self.window.post_message(
                    CharacterMessage.Deleted(self.current_character_id)
                )
                
                # Clear current character
                self.current_character_id = None
                self.current_character_data = {}
                
                # Refresh the character list
                await self.refresh_character_list()
                
                # Switch view back to main
                self.window.post_message(
                    ViewChangeMessage.Requested("conversations")
                )
            else:
                logger.error(f"Failed to delete character {self.current_character_id}")
                
        except Exception as e:
            logger.error(f"Error deleting character: {e}", exc_info=True)
    
    async def handle_import_character(self, file_path: str) -> None:
        """Import a character card from file.
        
        Args:
            file_path: Path to the character card file
        """
        try:
            from ...Character_Chat.ccv3_parser import import_character_card_json
            
            # Import the character card
            character_id = import_character_card_json(file_path)
            
            if character_id:
                logger.info(f"Imported character from {file_path}")
                
                # Refresh the character list
                await self.refresh_character_list()
                
                # Load the imported character
                await self.load_character(character_id)
            else:
                logger.error(f"Failed to import character from {file_path}")
                
        except Exception as e:
            logger.error(f"Error importing character: {e}", exc_info=True)
    
    async def handle_export_character(self) -> None:
        """Export the current character."""
        if not self.current_character_id:
            logger.warning("No character selected to export")
            return
        
        try:
            from ...Character_Chat.ccv3_parser import export_character_card_json
            
            # Generate export filename
            name = self.current_character_data.get("name", "character")
            safe_name = "".join(c for c in name if c.isalnum() or c in " -_").rstrip()
            file_path = f"exports/{safe_name}_card.json"
            
            # Export the character card
            success = export_character_card_json(self.current_character_id, file_path)
            
            if success:
                logger.info(f"Exported character to {file_path}")
                # Could show a notification here
            else:
                logger.error(f"Failed to export character {self.current_character_id}")
                
        except Exception as e:
            logger.error(f"Error exporting character: {e}", exc_info=True)
    
    async def handle_generate_field(self, field_name: str) -> None:
        """Generate a character field using AI.
        
        Args:
            field_name: Name of the field to generate
        """
        try:
            # Gather context for generation
            context = self._gather_editor_data()
            
            # Post message to trigger generation
            self.window.post_message(
                CharacterMessage.GenerateFieldRequested(field_name, context)
            )
            
            # The actual generation would be handled by the app or a dedicated AI handler
            logger.info(f"Requested AI generation for field: {field_name}")
            
        except Exception as e:
            logger.error(f"Error requesting field generation: {e}", exc_info=True)