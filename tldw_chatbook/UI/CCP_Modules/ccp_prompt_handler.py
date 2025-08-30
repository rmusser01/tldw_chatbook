"""Handler for prompt-related operations in the CCP window."""

from typing import TYPE_CHECKING, Optional, Dict, Any, List
from loguru import logger
from textual import work
from textual.widgets import ListView, ListItem, Input, TextArea, Button, Static

from .ccp_messages import PromptMessage, ViewChangeMessage

if TYPE_CHECKING:
    from ..Conv_Char_Window import CCPWindow

logger = logger.bind(module="CCPPromptHandler")


class CCPPromptHandler:
    """Handles all prompt-related operations for the CCP window."""
    
    def __init__(self, window: 'CCPWindow'):
        """Initialize the prompt handler.
        
        Args:
            window: Reference to the parent CCP window
        """
        self.window = window
        self.app_instance = window.app_instance
        self.current_prompt_id: Optional[int] = None
        self.current_prompt_data: Dict[str, Any] = {}
        self.search_results: List[Dict[str, Any]] = []
        
        logger.debug("CCPPromptHandler initialized")
    
    async def handle_search(self, search_term: str) -> None:
        """Search for prompts.
        
        Args:
            search_term: The term to search for in prompt names and content
        """
        logger.debug(f"Searching prompts for: '{search_term}'")
        
        try:
            from ...DB.Prompts_DB import fetch_all_prompts
            
            # Get all prompts
            all_prompts = fetch_all_prompts()
            
            if search_term:
                # Filter by search term
                search_lower = search_term.lower()
                self.search_results = [
                    prompt for prompt in all_prompts
                    if (search_lower in prompt.get('name', '').lower() or
                        search_lower in prompt.get('details', '').lower() or
                        search_lower in prompt.get('keywords', '').lower())
                ]
            else:
                # Show all prompts if no search term
                self.search_results = all_prompts
            
            # Update the UI
            await self._update_search_results_ui()
            
            logger.info(f"Found {len(self.search_results)} prompts matching '{search_term}'")
            
        except Exception as e:
            logger.error(f"Error searching prompts: {e}", exc_info=True)
    
    async def _update_search_results_ui(self) -> None:
        """Update the prompt search results ListView."""
        try:
            results_list = self.window.query_one("#ccp-prompts-listview", ListView)
            results_list.clear()
            
            for prompt in self.search_results:
                name = prompt.get('name', 'Untitled')
                prompt_id = prompt.get('id')
                author = prompt.get('author', 'Unknown')
                
                # Create a formatted list item
                item_text = f"{name} (by {author})"
                list_item = ListItem(Static(item_text), id=f"prompt-result-{prompt_id}")
                results_list.append(list_item)
                
        except Exception as e:
            logger.error(f"Error updating prompt search results: {e}")
    
    async def handle_load_selected(self) -> None:
        """Load the selected prompt."""
        try:
            results_list = self.window.query_one("#ccp-prompts-listview", ListView)
            
            if results_list.highlighted_child:
                # Extract prompt ID from the list item ID
                item_id = results_list.highlighted_child.id
                if item_id and item_id.startswith("prompt-result-"):
                    prompt_id = int(item_id.replace("prompt-result-", ""))
                    await self.load_prompt(prompt_id)
            else:
                logger.warning("No prompt selected to load")
                
        except Exception as e:
            logger.error(f"Error loading selected prompt: {e}", exc_info=True)
    
    async def load_prompt(self, prompt_id: int) -> None:
        """Load a prompt and display it in the editor (async wrapper).
        
        Args:
            prompt_id: The ID of the prompt to load
        """
        logger.info(f"Starting prompt load for {prompt_id}")
        
        # Run the sync database operation in a worker thread
        self.window.run_worker(
            self._load_prompt_sync,
            prompt_id,
            thread=True,
            exclusive=True,
            name=f"load_prompt_{prompt_id}"
        )
    
    @work(thread=True)
    def _load_prompt_sync(self, prompt_id: int) -> None:
        """Sync method to load prompt data in a worker thread.
        
        Args:
            prompt_id: The ID of the prompt to load
        """
        logger.info(f"Loading prompt {prompt_id}")
        
        try:
            from ...DB.Prompts_DB import fetch_prompt_by_id
            
            # Load the prompt (sync database operation)
            prompt_data = fetch_prompt_by_id(prompt_id)
            
            if prompt_data:
                self.current_prompt_id = prompt_id
                self.current_prompt_data = prompt_data
                
                # Post messages from worker thread using call_from_thread
                self.window.call_from_thread(
                    self.window.post_message,
                    PromptMessage.Loaded(prompt_id, prompt_data)
                )
                
                # Switch view to prompt editor
                self.window.call_from_thread(
                    self.window.post_message,
                    ViewChangeMessage.Requested("prompt_editor", {"prompt_id": prompt_id})
                )
                
                # Update UI on main thread
                self.window.call_from_thread(self._display_prompt_in_editor)
                
                logger.info(f"Prompt {prompt_id} loaded successfully")
            else:
                logger.error(f"Failed to load prompt {prompt_id}")
                
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_id}: {e}", exc_info=True)
    
    def _display_prompt_in_editor(self) -> None:
        """Display the loaded prompt in the editor."""
        try:
            if not self.current_prompt_data:
                return
            
            data = self.current_prompt_data
            
            # Update editor fields
            self._set_input_value("#ccp-editor-prompt-name-input", data.get("name", ""))
            self._set_input_value("#ccp-editor-prompt-author-input", data.get("author", ""))
            self._set_textarea_value("#ccp-editor-prompt-description-textarea", data.get("details", ""))
            self._set_textarea_value("#ccp-editor-prompt-system-textarea", data.get("system", ""))
            self._set_textarea_value("#ccp-editor-prompt-user-textarea", data.get("user", ""))
            self._set_textarea_value("#ccp-editor-prompt-keywords-textarea", data.get("keywords", ""))
            
            logger.debug(f"Displayed prompt '{data.get('name', 'Unknown')}' in editor")
            
        except Exception as e:
            logger.error(f"Error displaying prompt in editor: {e}", exc_info=True)
    
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
    
    async def handle_create_prompt(self) -> None:
        """Create a new prompt and switch to editor."""
        try:
            # Clear current prompt data
            self.current_prompt_id = None
            self.current_prompt_data = {}
            
            # Switch to editor view
            self.window.post_message(
                ViewChangeMessage.Requested("prompt_editor", {"new": True})
            )
            
            # Clear editor fields
            self._clear_editor_fields()
            
            logger.info("Switched to prompt editor for new prompt")
            
        except Exception as e:
            logger.error(f"Error creating new prompt: {e}", exc_info=True)
    
    def _clear_editor_fields(self) -> None:
        """Clear all prompt editor fields."""
        try:
            self._set_input_value("#ccp-editor-prompt-name-input", "")
            self._set_input_value("#ccp-editor-prompt-author-input", "")
            self._set_textarea_value("#ccp-editor-prompt-description-textarea", "")
            self._set_textarea_value("#ccp-editor-prompt-system-textarea", "")
            self._set_textarea_value("#ccp-editor-prompt-user-textarea", "")
            self._set_textarea_value("#ccp-editor-prompt-keywords-textarea", "")
        except Exception as e:
            logger.warning(f"Error clearing editor fields: {e}")
    
    async def handle_save_prompt(self) -> None:
        """Save the prompt from the editor."""
        try:
            # Gather data from editor
            prompt_data = self._gather_editor_data()
            
            if not prompt_data.get("name"):
                logger.warning("Cannot save prompt without a name")
                return
            
            if self.current_prompt_id:
                # Update existing prompt
                await self._update_prompt(self.current_prompt_id, prompt_data)
            else:
                # Create new prompt
                await self._create_prompt(prompt_data)
                
        except Exception as e:
            logger.error(f"Error saving prompt: {e}", exc_info=True)
    
    def _gather_editor_data(self) -> Dict[str, Any]:
        """Gather all data from the prompt editor fields."""
        data = {}
        
        try:
            data["name"] = self.window.query_one("#ccp-editor-prompt-name-input", Input).value
            data["author"] = self.window.query_one("#ccp-editor-prompt-author-input", Input).value
            data["details"] = self.window.query_one("#ccp-editor-prompt-description-textarea", TextArea).text
            data["system"] = self.window.query_one("#ccp-editor-prompt-system-textarea", TextArea).text
            data["user"] = self.window.query_one("#ccp-editor-prompt-user-textarea", TextArea).text
            data["keywords"] = self.window.query_one("#ccp-editor-prompt-keywords-textarea", TextArea).text
            
        except Exception as e:
            logger.error(f"Error gathering editor data: {e}", exc_info=True)
        
        return data
    
    @work(thread=True)
    def _create_prompt(self, data: Dict[str, Any]) -> None:
        """Create a new prompt in the database (sync worker method)."""
        try:
            from ...DB.Prompts_DB import add_prompt
            
            # Create the prompt (sync database operation)
            prompt_id = add_prompt(
                name=data["name"],
                details=data.get("details", ""),
                system=data.get("system", ""),
                user=data.get("user", ""),
                author=data.get("author", ""),
                keywords=data.get("keywords", "")
            )
            
            if prompt_id:
                logger.info(f"Created new prompt with ID {prompt_id}")
                
                # Update current prompt info
                self.current_prompt_id = prompt_id
                self.current_prompt_data = data
                
                # Post creation message from worker thread
                self.window.call_from_thread(
                    self.window.post_message,
                    PromptMessage.Created(prompt_id, data["name"], data)
                )
                
                # Refresh search results on main thread
                def refresh_search():
                    search_input = self.window.query_one("#ccp-prompt-search-input", Input)
                    self.window.run_worker(
                        self.handle_search,
                        search_input.value,
                        thread=True,
                        exclusive=True,
                        name="refresh_prompt_search"
                    )
                self.window.call_from_thread(refresh_search)
            else:
                logger.error("Failed to create new prompt")
                
        except Exception as e:
            logger.error(f"Error creating prompt: {e}", exc_info=True)
    
    @work(thread=True)
    def _update_prompt(self, prompt_id: int, data: Dict[str, Any]) -> None:
        """Update an existing prompt in the database (sync worker method)."""
        try:
            from ...DB.Prompts_DB import update_prompt
            
            # Update the prompt
            success = update_prompt(
                prompt_id=prompt_id,
                name=data["name"],
                details=data.get("details", ""),
                system=data.get("system", ""),
                user=data.get("user", ""),
                author=data.get("author", ""),
                keywords=data.get("keywords", "")
            )
            
            if success:
                logger.info(f"Updated prompt {prompt_id}")
                
                # Update current prompt data
                self.current_prompt_data = data
                
                # Post update message from worker thread
                self.window.call_from_thread(
                    self.window.post_message,
                    PromptMessage.Updated(prompt_id, data)
                )
                
                # Refresh search results on main thread
                def refresh_search():
                    search_input = self.window.query_one("#ccp-prompt-search-input", Input)
                    self.window.run_worker(
                        self.handle_search,
                        search_input.value,
                        thread=True,
                        exclusive=True,
                        name="refresh_prompt_search"
                    )
                self.window.call_from_thread(refresh_search)
            else:
                logger.error(f"Failed to update prompt {prompt_id}")
                
        except Exception as e:
            logger.error(f"Error updating prompt: {e}", exc_info=True)
    
    async def handle_clone_prompt(self) -> None:
        """Clone the current prompt."""
        if not self.current_prompt_data:
            logger.warning("No prompt loaded to clone")
            return
        
        try:
            # Create a copy of the current data
            cloned_data = self.current_prompt_data.copy()
            cloned_data["name"] = f"{cloned_data.get('name', 'Prompt')} (Copy)"
            
            # Clear current prompt ID to create new
            self.current_prompt_id = None
            
            # Update editor with cloned data
            self._set_input_value("#ccp-editor-prompt-name-input", cloned_data["name"])
            
            logger.info("Prepared prompt clone in editor")
            
        except Exception as e:
            logger.error(f"Error cloning prompt: {e}", exc_info=True)
    
    async def handle_delete_prompt(self) -> None:
        """Delete the current prompt."""
        if not self.current_prompt_id:
            logger.warning("No prompt selected to delete")
            return
        
        try:
            from ...DB.Prompts_DB import delete_prompt
            
            success = delete_prompt(self.current_prompt_id)
            
            if success:
                logger.info(f"Deleted prompt {self.current_prompt_id}")
                
                # Post deletion message
                self.window.post_message(
                    PromptMessage.Deleted(self.current_prompt_id)
                )
                
                # Clear current prompt
                self.current_prompt_id = None
                self.current_prompt_data = {}
                
                # Clear editor
                self._clear_editor_fields()
                
                # Refresh search results
                search_input = self.window.query_one("#ccp-prompt-search-input", Input)
                await self.handle_search(search_input.value)
            else:
                logger.error(f"Failed to delete prompt {self.current_prompt_id}")
                
        except Exception as e:
            logger.error(f"Error deleting prompt: {e}", exc_info=True)
    
    async def handle_import(self) -> None:
        """Handle import request - prompts for file selection."""
        from ...Widgets.enhanced_file_picker import EnhancedFileOpen, Filters
        
        try:
            # Create filters for prompt files
            filters = Filters(
                ("Prompt Files", "*.json;*.yaml;*.yml;*.txt"),
                ("JSON Files", "*.json"),
                ("YAML Files", "*.yaml;*.yml"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            )
            
            # Create and show the file picker
            picker = EnhancedFileOpen(
                title="Import Prompt",
                filters=filters,
                context="prompt_import"
            )
            
            # Push the file picker screen
            file_path = await self.window.app.push_screen(picker, wait_for_dismiss=True)
            
            if file_path:
                await self.handle_import_prompt(str(file_path))
        except Exception as e:
            logger.error(f"Error showing file picker: {e}")
    
    async def handle_import_prompt(self, file_path: str) -> None:
        """Import a prompt from file.
        
        Args:
            file_path: Path to the prompt file (JSON format expected)
        """
        try:
            import json
            from pathlib import Path
            
            # Read the prompt file
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Prompt file not found: {file_path}")
                return
            
            with open(path, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
            
            # Create the prompt
            await self._create_prompt(prompt_data)
            
            logger.info(f"Imported prompt from {file_path}")
            
        except Exception as e:
            logger.error(f"Error importing prompt: {e}", exc_info=True)