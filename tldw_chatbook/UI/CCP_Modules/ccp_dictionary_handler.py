"""Handler for dictionary/world book operations in the CCP window."""

from typing import TYPE_CHECKING, Optional, Dict, Any, List
from loguru import logger
from textual import work
from textual.widgets import Select, ListView, ListItem, Input, TextArea, Button, Static

from .ccp_messages import DictionaryMessage, ViewChangeMessage

if TYPE_CHECKING:
    from ..Conv_Char_Window import CCPWindow

logger = logger.bind(module="CCPDictionaryHandler")


class CCPDictionaryHandler:
    """Handles all dictionary and world book operations for the CCP window."""
    
    def __init__(self, window: 'CCPWindow'):
        """Initialize the dictionary handler.
        
        Args:
            window: Reference to the parent CCP window
        """
        self.window = window
        self.app_instance = window.app_instance
        self.current_dictionary_id: Optional[int] = None
        self.current_dictionary_data: Dict[str, Any] = {}
        self.dictionary_entries: List[Dict[str, Any]] = []
        self.selected_entry_index: Optional[int] = None
        
        logger.debug("CCPDictionaryHandler initialized")
    
    async def refresh_dictionary_list(self) -> None:
        """Refresh the dictionary select dropdown."""
        try:
            from ...Character_Chat.Character_Chat_Lib import fetch_all_dictionaries
            
            # Get all dictionaries
            dictionaries = fetch_all_dictionaries()
            
            # Update the select widget
            dict_select = self.window.query_one("#ccp-dictionary-select", Select)
            
            # Convert to Select options format
            options = [(d['name'], str(d['id'])) for d in dictionaries]
            
            # Update the select widget
            dict_select.set_options(options)
            
            logger.info(f"Refreshed dictionary list with {len(options)} dictionaries")
            
        except Exception as e:
            logger.error(f"Error refreshing dictionary list: {e}", exc_info=True)
    
    async def handle_load_dictionary(self) -> None:
        """Load the selected dictionary."""
        try:
            dict_select = self.window.query_one("#ccp-dictionary-select", Select)
            
            if dict_select.value:
                dictionary_id = int(dict_select.value)
                await self.load_dictionary(dictionary_id)
            else:
                logger.warning("No dictionary selected to load")
                
        except Exception as e:
            logger.error(f"Error loading selected dictionary: {e}", exc_info=True)
    
    async def load_dictionary(self, dictionary_id: int) -> None:
        """Load a dictionary and display it (async wrapper).
        
        Args:
            dictionary_id: The ID of the dictionary to load
        """
        logger.info(f"Starting dictionary load for {dictionary_id}")
        
        # Run the sync database operation in a worker thread
        self.window.run_worker(
            self._load_dictionary_sync,
            dictionary_id,
            thread=True,
            exclusive=True,
            name=f"load_dictionary_{dictionary_id}"
        )
    
    @work(thread=True)
    def _load_dictionary_sync(self, dictionary_id: int) -> None:
        """Sync method to load dictionary data in a worker thread.
        
        Args:
            dictionary_id: The ID of the dictionary to load
        """
        logger.info(f"Loading dictionary {dictionary_id}")
        
        try:
            from ...Character_Chat.Character_Chat_Lib import fetch_dictionary_by_id
            
            # Load the dictionary (sync database operation)
            dict_data = fetch_dictionary_by_id(dictionary_id)
            
            if dict_data:
                self.current_dictionary_id = dictionary_id
                self.current_dictionary_data = dict_data
                self.dictionary_entries = dict_data.get('entries', [])
                
                # Post messages from worker thread using call_from_thread
                self.window.call_from_thread(
                    self.window.post_message,
                    DictionaryMessage.Loaded(dictionary_id, dict_data)
                )
                
                # Switch view to show dictionary
                self.window.call_from_thread(
                    self.window.post_message,
                    ViewChangeMessage.Requested("dictionary_view", {"dictionary_id": dictionary_id})
                )
                
                # Update UI on main thread
                self.window.call_from_thread(self._display_dictionary)
                
                logger.info(f"Dictionary {dictionary_id} loaded successfully")
            else:
                logger.error(f"Failed to load dictionary {dictionary_id}")
                
        except Exception as e:
            logger.error(f"Error loading dictionary {dictionary_id}: {e}", exc_info=True)
    
    def _display_dictionary(self) -> None:
        """Display dictionary in the UI."""
        try:
            if not self.current_dictionary_data:
                return
            
            data = self.current_dictionary_data
            
            # Update display fields
            self._update_field("#ccp-dict-name-display", data.get("name", "N/A"))
            self._update_textarea("#ccp-dict-description-display", data.get("description", ""))
            self._update_field("#ccp-dict-strategy-display", data.get("strategy", "sorted_evenly"))
            self._update_field("#ccp-dict-max-tokens-display", str(data.get("max_tokens", 1000)))
            
            # Display entries
            self._display_dictionary_entries()
            
            logger.debug(f"Displayed dictionary '{data.get('name', 'Unknown')}'")
            
        except Exception as e:
            logger.error(f"Error displaying dictionary: {e}", exc_info=True)
    
    def _display_dictionary_entries(self) -> None:
        """Display dictionary entries in the list."""
        try:
            entries_list = self.window.query_one("#ccp-dict-entries-list", ListView)
            entries_list.clear()
            
            for i, entry in enumerate(self.dictionary_entries):
                key = entry.get('key', 'Unknown')
                group = entry.get('group', '')
                probability = entry.get('probability', 100)
                
                # Format entry display
                entry_text = f"{key}"
                if group:
                    entry_text += f" [{group}]"
                if probability < 100:
                    entry_text += f" ({probability}%)"
                
                list_item = ListItem(Static(entry_text), id=f"dict-entry-{i}")
                entries_list.append(list_item)
            
            logger.debug(f"Displayed {len(self.dictionary_entries)} dictionary entries")
            
        except Exception as e:
            logger.error(f"Error displaying dictionary entries: {e}")
    
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
    
    async def handle_edit_dictionary(self) -> None:
        """Switch to dictionary editor view."""
        if not self.current_dictionary_data:
            logger.warning("No dictionary loaded to edit")
            return
        
        try:
            # Switch view to editor
            self.window.post_message(
                ViewChangeMessage.Requested("dictionary_editor", 
                                          {"dictionary_id": self.current_dictionary_id})
            )
            
            # Populate editor fields
            self._populate_editor_fields()
            
        except Exception as e:
            logger.error(f"Error switching to dictionary editor: {e}", exc_info=True)
    
    def _populate_editor_fields(self) -> None:
        """Populate the dictionary editor fields with current data."""
        try:
            data = self.current_dictionary_data
            
            # Basic fields
            self._set_input_value("#ccp-editor-dict-name-input", data.get("name", ""))
            self._set_textarea_value("#ccp-editor-dict-description-textarea", data.get("description", ""))
            
            # Strategy select
            strategy_select = self.window.query_one("#ccp-editor-dict-strategy-select", Select)
            strategy_select.value = data.get("strategy", "sorted_evenly")
            
            # Max tokens
            self._set_input_value("#ccp-editor-dict-max-tokens-input", str(data.get("max_tokens", 1000)))
            
            # Display entries in editor list
            self._display_editor_entries()
            
        except Exception as e:
            logger.error(f"Error populating editor fields: {e}", exc_info=True)
    
    def _display_editor_entries(self) -> None:
        """Display dictionary entries in the editor list."""
        try:
            entries_list = self.window.query_one("#ccp-editor-dict-entries-list", ListView)
            entries_list.clear()
            
            for i, entry in enumerate(self.dictionary_entries):
                key = entry.get('key', 'Unknown')
                list_item = ListItem(Static(key), id=f"editor-dict-entry-{i}")
                entries_list.append(list_item)
            
        except Exception as e:
            logger.error(f"Error displaying editor entries: {e}")
    
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
    
    async def handle_add_entry(self) -> None:
        """Add a new dictionary entry."""
        try:
            # Get entry data from inputs
            key = self.window.query_one("#ccp-dict-entry-key-input", Input).value
            value = self.window.query_one("#ccp-dict-entry-value-textarea", TextArea).text
            group = self.window.query_one("#ccp-dict-entry-group-input", Input).value
            probability_str = self.window.query_one("#ccp-dict-entry-probability-input", Input).value
            
            if not key or not value:
                logger.warning("Cannot add entry without key and value")
                return
            
            # Parse probability
            try:
                probability = int(probability_str) if probability_str else 100
                probability = max(0, min(100, probability))  # Clamp to 0-100
            except ValueError:
                probability = 100
            
            # Create entry
            entry = {
                'key': key,
                'value': value,
                'group': group,
                'probability': probability
            }
            
            # Add to entries list
            self.dictionary_entries.append(entry)
            
            # Update display
            self._display_editor_entries()
            
            # Clear input fields
            self.window.query_one("#ccp-dict-entry-key-input", Input).value = ""
            self.window.query_one("#ccp-dict-entry-value-textarea", TextArea).text = ""
            self.window.query_one("#ccp-dict-entry-group-input", Input).value = ""
            self.window.query_one("#ccp-dict-entry-probability-input", Input).value = "100"
            
            # Post message
            if self.current_dictionary_id:
                self.window.post_message(
                    DictionaryMessage.EntryAdded(self.current_dictionary_id, entry)
                )
            
            logger.info(f"Added dictionary entry: {key}")
            
        except Exception as e:
            logger.error(f"Error adding dictionary entry: {e}", exc_info=True)
    
    async def handle_remove_entry(self) -> None:
        """Remove the selected dictionary entry."""
        try:
            entries_list = self.window.query_one("#ccp-editor-dict-entries-list", ListView)
            
            if entries_list.highlighted_child:
                # Extract entry index from the list item ID
                item_id = entries_list.highlighted_child.id
                if item_id and item_id.startswith("editor-dict-entry-"):
                    index = int(item_id.replace("editor-dict-entry-", ""))
                    
                    if 0 <= index < len(self.dictionary_entries):
                        removed_entry = self.dictionary_entries.pop(index)
                        
                        # Update display
                        self._display_editor_entries()
                        
                        # Post message
                        if self.current_dictionary_id:
                            self.window.post_message(
                                DictionaryMessage.EntryRemoved(
                                    self.current_dictionary_id, 
                                    removed_entry['key']
                                )
                            )
                        
                        logger.info(f"Removed dictionary entry: {removed_entry['key']}")
            else:
                logger.warning("No entry selected to remove")
                
        except Exception as e:
            logger.error(f"Error removing dictionary entry: {e}", exc_info=True)
    
    async def handle_save_dictionary(self) -> None:
        """Save the dictionary from editor."""
        try:
            # Gather all field values
            dictionary_data = self._gather_editor_data()
            
            if not dictionary_data.get("name"):
                logger.warning("Cannot save dictionary without a name")
                return
            
            if self.current_dictionary_id:
                # Update existing dictionary
                await self._update_dictionary(self.current_dictionary_id, dictionary_data)
            else:
                # Create new dictionary
                await self._create_dictionary(dictionary_data)
                
        except Exception as e:
            logger.error(f"Error saving dictionary: {e}", exc_info=True)
    
    def _gather_editor_data(self) -> Dict[str, Any]:
        """Gather all data from the editor fields."""
        data = {}
        
        try:
            data["name"] = self.window.query_one("#ccp-editor-dict-name-input", Input).value
            data["description"] = self.window.query_one("#ccp-editor-dict-description-textarea", TextArea).text
            
            strategy_select = self.window.query_one("#ccp-editor-dict-strategy-select", Select)
            data["strategy"] = strategy_select.value or "sorted_evenly"
            
            max_tokens_str = self.window.query_one("#ccp-editor-dict-max-tokens-input", Input).value
            try:
                data["max_tokens"] = int(max_tokens_str) if max_tokens_str else 1000
            except ValueError:
                data["max_tokens"] = 1000
            
            data["entries"] = self.dictionary_entries
            
        except Exception as e:
            logger.error(f"Error gathering editor data: {e}", exc_info=True)
        
        return data
    
    @work(thread=True)
    def _create_dictionary(self, data: Dict[str, Any]) -> None:
        """Create a new dictionary (sync worker method)."""
        try:
            from ...Character_Chat.Character_Chat_Lib import create_dictionary
            
            dictionary_id = create_dictionary(data)
            
            if dictionary_id:
                logger.info(f"Created new dictionary with ID {dictionary_id}")
                
                # Update current dictionary info
                self.current_dictionary_id = dictionary_id
                self.current_dictionary_data = data
                
                # Post creation message from worker thread
                self.window.call_from_thread(
                    self.window.post_message,
                    DictionaryMessage.Created(dictionary_id, data["name"], data)
                )
                
                # Refresh the dictionary list on main thread
                self.window.call_from_thread(self.refresh_dictionary_list)
            else:
                logger.error("Failed to create new dictionary")
                
        except Exception as e:
            logger.error(f"Error creating dictionary: {e}", exc_info=True)
    
    @work(thread=True)
    def _update_dictionary(self, dictionary_id: int, data: Dict[str, Any]) -> None:
        """Update an existing dictionary (sync worker method)."""
        try:
            from ...Character_Chat.Character_Chat_Lib import update_dictionary
            
            success = update_dictionary(dictionary_id, data)
            
            if success:
                logger.info(f"Updated dictionary {dictionary_id}")
                
                # Update current dictionary data
                self.current_dictionary_data = data
                
                # Post update message from worker thread
                self.window.call_from_thread(
                    self.window.post_message,
                    DictionaryMessage.Updated(dictionary_id, data)
                )
                
                # Refresh the dictionary list on main thread
                self.window.call_from_thread(self.refresh_dictionary_list)
            else:
                logger.error(f"Failed to update dictionary {dictionary_id}")
                
        except Exception as e:
            logger.error(f"Error updating dictionary: {e}", exc_info=True)
    
    async def handle_delete_dictionary(self) -> None:
        """Delete the current dictionary."""
        if not self.current_dictionary_id:
            logger.warning("No dictionary selected to delete")
            return
        
        try:
            from ...Character_Chat.Character_Chat_Lib import delete_dictionary
            
            success = delete_dictionary(self.current_dictionary_id)
            
            if success:
                logger.info(f"Deleted dictionary {self.current_dictionary_id}")
                
                # Post deletion message
                self.window.post_message(
                    DictionaryMessage.Deleted(self.current_dictionary_id)
                )
                
                # Clear current dictionary
                self.current_dictionary_id = None
                self.current_dictionary_data = {}
                self.dictionary_entries = []
                
                # Refresh the dictionary list
                await self.refresh_dictionary_list()
                
                # Switch view back to main
                self.window.post_message(
                    ViewChangeMessage.Requested("conversations")
                )
            else:
                logger.error(f"Failed to delete dictionary {self.current_dictionary_id}")
                
        except Exception as e:
            logger.error(f"Error deleting dictionary: {e}", exc_info=True)
    
    async def handle_import(self) -> None:
        """Handle import request - prompts for file selection."""
        from ...Widgets.enhanced_file_picker import EnhancedFileOpen, Filters
        
        try:
            # Create filters for dictionary/world book files
            filters = Filters(
                ("Dictionary Files", "*.json;*.csv;*.yaml;*.yml"),
                ("JSON Files", "*.json"),
                ("CSV Files", "*.csv"),
                ("YAML Files", "*.yaml;*.yml"),
                ("All Files", "*.*")
            )
            
            # Create and show the file picker
            picker = EnhancedFileOpen(
                title="Import Dictionary/World Book",
                filters=filters,
                context="dictionary_import"
            )
            
            # Push the file picker screen
            file_path = await self.window.app.push_screen(picker, wait_for_dismiss=True)
            
            if file_path:
                await self.handle_import_dictionary(str(file_path))
        except Exception as e:
            logger.error(f"Error showing file picker: {e}")
    
    async def handle_import_dictionary(self, file_path: str) -> None:
        """Import a dictionary from file.
        
        Args:
            file_path: Path to the dictionary file
        """
        # TODO: Implement actual dictionary import logic
        logger.info(f"Would import dictionary from: {file_path}")
    
    async def handle_clone_dictionary(self) -> None:
        """Clone the current dictionary."""
        if not self.current_dictionary_data:
            logger.warning("No dictionary loaded to clone")
            return
        
        try:
            # Create a copy of the current data
            cloned_data = self.current_dictionary_data.copy()
            cloned_data["name"] = f"{cloned_data.get('name', 'Dictionary')} (Copy)"
            
            # Clear current dictionary ID to create new
            self.current_dictionary_id = None
            
            # Update editor with cloned data
            self._set_input_value("#ccp-editor-dict-name-input", cloned_data["name"])
            
            logger.info("Prepared dictionary clone in editor")
            
        except Exception as e:
            logger.error(f"Error cloning dictionary: {e}", exc_info=True)