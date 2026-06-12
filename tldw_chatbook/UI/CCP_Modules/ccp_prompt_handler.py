"""Handler for prompt-related operations in the CCP window."""

from typing import TYPE_CHECKING, Optional, Dict, Any, List
from loguru import logger
from textual import work
from textual.widgets import ListView, ListItem, Input, TextArea, Button, Static

from .ccp_messages import PromptMessage, ViewChangeMessage

if TYPE_CHECKING:
    from ..Conv_Char_Window import CCPWindow

logger = logger.bind(module="CCPPromptHandler")

# Relocated from the retired CCPSidebarWidget.EMPTY_STATE_TEXT["prompts"].
PROMPTS_EMPTY_STATE_TEXT = (
    "No prompts yet. Create New Prompt to save reusable Chat instructions."
)


class CCPPromptHandler:
    """Handles all prompt-related operations for the CCP window."""
    
    def __init__(self, window: 'CCPWindow'):
        """Initialize the prompt handler.
        
        Args:
            window: Reference to the parent CCP window
        """
        self.window = window
        self.app_instance = window.app_instance
        self.current_prompt_id: Optional[Any] = None
        self.current_prompt_data: Dict[str, Any] = {}
        self.search_results: List[Dict[str, Any]] = []
        self._prompt_result_ids: Dict[str, Any] = {}
        
        logger.debug("CCPPromptHandler initialized")

    def _prompt_scope_service(self) -> Any:
        """Return the app-level prompt scope service when this build has one wired."""
        return getattr(self.app_instance, "prompt_scope_service", None)

    def _current_prompt_backend(self) -> str:
        backend = (
            getattr(self.app_instance, "current_runtime_backend", None)
            or getattr(self.app_instance, "runtime_backend", None)
            or "local"
        )
        normalized_backend = str(backend).strip().lower()
        if normalized_backend not in {"local", "server"}:
            return "local"
        return normalized_backend

    def _source_prompt_identifier(self, prompt_identifier: Any) -> Any:
        if isinstance(prompt_identifier, str) and ":prompt:" in prompt_identifier:
            return prompt_identifier.rsplit(":prompt:", 1)[1]
        return prompt_identifier

    def _notify(self, message: str, severity: str = "info") -> None:
        notify = getattr(self.app_instance, "notify", None) or getattr(self.window, "notify", None)
        if callable(notify):
            notify(message, severity=severity)

    def _set_static_value(self, selector: str, value: str) -> None:
        try:
            widget = self.window.query_one(selector, Static)
            widget.update(value)
        except Exception as e:
            logger.debug(f"Could not set static {selector}: {e}")

    def _current_source_prompt_identifier(self) -> Any:
        if not self.current_prompt_id:
            self._notify("Load a prompt before using prompt actions.", severity="warning")
            return None
        return self._source_prompt_identifier(self.current_prompt_id)

    def _update_prompt_usage_display(self, prompt_data: Dict[str, Any]) -> None:
        usage_count = prompt_data.get("usage_count", prompt_data.get("use_count"))
        usage_text = "Usage: -" if usage_count is None else f"Usage: {usage_count}"
        self._set_static_value("#ccp-editor-prompt-usage-display", usage_text)

    def _format_prompt_versions(self, versions: List[Dict[str, Any]]) -> str:
        if not versions:
            return "No server versions found."
        version_labels = []
        for version in versions:
            raw_version = version.get("version", version.get("version_number", "?"))
            name = version.get("name")
            label = f"v{raw_version}"
            if name:
                label = f"{label} {name}"
            version_labels.append(label)
        return "Versions: " + ", ".join(version_labels)

    def _filter_prompt_results(self, prompts: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
        if not search_term:
            return prompts

        search_lower = search_term.lower()

        def matches(prompt: Dict[str, Any]) -> bool:
            keywords = prompt.get("keywords", "")
            if isinstance(keywords, list):
                keywords_text = ", ".join(str(keyword) for keyword in keywords)
            else:
                keywords_text = str(keywords or "")

            haystack = " ".join(
                str(prompt.get(field, "") or "")
                for field in (
                    "name",
                    "author",
                    "details",
                    "system",
                    "user",
                    "system_prompt",
                    "user_prompt",
                )
            )
            return search_lower in haystack.lower() or search_lower in keywords_text.lower()

        return [prompt for prompt in prompts if matches(prompt)]

    def _legacy_prompt_list(self) -> List[Dict[str, Any]]:
        prompt_db = getattr(self.app_instance, "prompts_db", None)
        if prompt_db is not None and hasattr(prompt_db, "list_prompts"):
            result = prompt_db.list_prompts(page=1, per_page=100, include_deleted=False)
            if isinstance(result, tuple):
                return list(result[0] or [])
            if isinstance(result, dict):
                return list(result.get("items", []) or [])

        from ...Prompt_Management.Prompts_Interop import list_prompts

        result = list_prompts(page=1, per_page=100, include_deleted=False)
        if isinstance(result, tuple):
            return list(result[0] or [])
        return list(result.get("items", []) or [])
    
    async def handle_search(self, search_term: str) -> None:
        """Search for prompts.
        
        Args:
            search_term: The term to search for in prompt names and content
        """
        logger.debug(f"Searching prompts for: '{search_term}'")
        
        try:
            scope_service = self._prompt_scope_service()
            if scope_service is not None:
                response = await scope_service.list_prompts(
                    mode=self._current_prompt_backend(),
                    page=1,
                    per_page=100,
                    include_deleted=False,
                )
                all_prompts = list(response.get("items", []) or [])
            else:
                all_prompts = self._legacy_prompt_list()

            self.search_results = self._filter_prompt_results(all_prompts, search_term)
            
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
            self._prompt_result_ids = {}

            if not self.search_results:
                results_list.append(ListItem(Static(PROMPTS_EMPTY_STATE_TEXT)))
                return
            
            for index, prompt in enumerate(self.search_results):
                name = prompt.get('name', 'Untitled')
                prompt_id = prompt.get('id')
                author = prompt.get('author', 'Unknown')
                backend = prompt.get("backend")
                
                # Create a formatted list item
                source_prefix = f"[{backend}] " if backend else ""
                item_text = f"{source_prefix}{name} (by {author})"
                item_id = f"prompt-result-{index}"
                self._prompt_result_ids[item_id] = prompt_id
                list_item = ListItem(Static(item_text), id=item_id)
                setattr(list_item, "prompt_identifier", prompt_id)
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
                    prompt_id = getattr(results_list.highlighted_child, "prompt_identifier", None)
                    if prompt_id is None:
                        prompt_id = self._prompt_result_ids.get(item_id)
                    if prompt_id is None:
                        prompt_id = int(item_id.replace("prompt-result-", ""))
                    await self.load_prompt(prompt_id)
            else:
                logger.warning("No prompt selected to load")
                
        except Exception as e:
            logger.error(f"Error loading selected prompt: {e}", exc_info=True)
    
    async def load_prompt(self, prompt_id: Any) -> None:
        """Load a prompt and display it in the editor (async wrapper).
        
        Args:
            prompt_id: The ID of the prompt to load
        """
        logger.info(f"Starting prompt load for {prompt_id}")

        scope_service = self._prompt_scope_service()
        if scope_service is not None:
            await self._load_prompt_scoped(prompt_id)
            return
        
        # Run the sync database operation in a worker thread
        self.window.run_worker(
            self._load_prompt_sync,
            prompt_id,
            thread=True,
            exclusive=True,
            name=f"load_prompt_{prompt_id}"
        )

    async def _load_prompt_scoped(self, prompt_id: Any) -> None:
        """Load a prompt through the source-aware prompt service."""
        prompt_identifier = self._source_prompt_identifier(prompt_id)
        prompt_data = await self._prompt_scope_service().get_prompt(
            mode=self._current_prompt_backend(),
            prompt_identifier=prompt_identifier,
            include_deleted=False,
        )
        self.current_prompt_id = prompt_data.get("id", prompt_id)
        self.current_prompt_data = prompt_data
        self.window.post_message(PromptMessage.Loaded(self.current_prompt_id, prompt_data))
        self.window.post_message(
            ViewChangeMessage.Requested("prompt_editor", {"prompt_id": self.current_prompt_id})
        )
        self._display_prompt_in_editor()
        logger.info(f"Prompt {self.current_prompt_id} loaded successfully")
    
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
            self._set_textarea_value("#ccp-editor-prompt-system-textarea", data.get("system_prompt", data.get("system", "")))
            self._set_textarea_value("#ccp-editor-prompt-user-textarea", data.get("user_prompt", data.get("user", "")))
            keywords = data.get("keywords", "")
            if isinstance(keywords, list):
                keywords = ", ".join(str(keyword) for keyword in keywords)
            self._set_textarea_value("#ccp-editor-prompt-keywords-textarea", str(keywords or ""))
            self._update_prompt_usage_display(data)
            
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
                if self._prompt_scope_service() is not None:
                    await self._save_prompt_scoped(prompt_data, self.current_prompt_id)
                else:
                    await self._update_prompt(self.current_prompt_id, prompt_data)
            else:
                # Create new prompt
                if self._prompt_scope_service() is not None:
                    await self._save_prompt_scoped(prompt_data, None)
                else:
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

    def _keywords_from_editor_value(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            raw_keywords = value
        else:
            raw_keywords = str(value).split(",")
        return [str(keyword).strip() for keyword in raw_keywords if str(keyword).strip()]

    async def _save_prompt_scoped(self, data: Dict[str, Any], prompt_id: Any = None) -> None:
        """Create or update a prompt through the active prompt backend."""
        prompt_identifier = self._source_prompt_identifier(prompt_id) if prompt_id else None
        saved_prompt = await self._prompt_scope_service().save_prompt(
            mode=self._current_prompt_backend(),
            prompt_identifier=prompt_identifier,
            name=data["name"],
            author=data.get("author", ""),
            details=data.get("details", ""),
            system_prompt=data.get("system", ""),
            user_prompt=data.get("user", ""),
            keywords=self._keywords_from_editor_value(data.get("keywords", "")),
        )

        saved_prompt_id = saved_prompt.get("id", prompt_id)
        was_update = prompt_id not in (None, "")
        self.current_prompt_id = saved_prompt_id
        self.current_prompt_data = saved_prompt

        if was_update:
            self.window.post_message(PromptMessage.Updated(saved_prompt_id, saved_prompt))
        else:
            self.window.post_message(PromptMessage.Created(saved_prompt_id, saved_prompt.get("name", data["name"]), saved_prompt))

        try:
            search_input = self.window.query_one("#ccp-prompt-search-input", Input)
            await self.handle_search(search_input.value)
        except Exception:
            logger.debug("Prompt search refresh skipped after scoped save", exc_info=True)
    
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
            if self._prompt_scope_service() is not None:
                deleted_prompt_id = self.current_prompt_id
                success = await self._prompt_scope_service().delete_prompt(
                    mode=self._current_prompt_backend(),
                    prompt_identifier=self._source_prompt_identifier(self.current_prompt_id),
                )

                if success:
                    logger.info(f"Deleted prompt {deleted_prompt_id}")
                    self.window.post_message(PromptMessage.Deleted(deleted_prompt_id))
                    self.current_prompt_id = None
                    self.current_prompt_data = {}
                    self._clear_editor_fields()
                    try:
                        search_input = self.window.query_one("#ccp-prompt-search-input", Input)
                        await self.handle_search(search_input.value)
                    except Exception:
                        logger.debug("Prompt search refresh skipped after scoped delete", exc_info=True)
                else:
                    logger.error(f"Failed to delete prompt {deleted_prompt_id}")
                return

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

    async def handle_record_prompt_usage(self) -> None:
        """Record usage for the current prompt through the active prompt backend."""
        scope_service = self._prompt_scope_service()
        prompt_identifier = self._current_source_prompt_identifier()
        if scope_service is None or prompt_identifier is None:
            self._notify("Prompt usage recording is unavailable in this context.", severity="warning")
            return

        try:
            prompt_data = await scope_service.record_prompt_usage(
                mode=self._current_prompt_backend(),
                prompt_identifier=prompt_identifier,
            )
            self.current_prompt_id = prompt_data.get("id", self.current_prompt_id)
            self.current_prompt_data = prompt_data
            self.window.post_message(PromptMessage.Updated(self.current_prompt_id, prompt_data))
            self._update_prompt_usage_display(prompt_data)
            self._notify("Prompt usage recorded.", severity="success")
        except Exception as e:
            logger.error(f"Error recording prompt usage: {e}", exc_info=True)
            self._set_static_value("#ccp-editor-prompt-version-status", f"Usage update failed: {e}")
            self._notify(f"Prompt usage update failed: {e}", severity="error")

    async def handle_list_prompt_versions(self) -> None:
        """List server prompt versions for the current prompt."""
        scope_service = self._prompt_scope_service()
        prompt_identifier = self._current_source_prompt_identifier()
        if scope_service is None or prompt_identifier is None:
            self._set_static_value("#ccp-editor-prompt-version-status", "Prompt versions are unavailable.")
            return

        try:
            versions = await scope_service.list_prompt_versions(
                mode=self._current_prompt_backend(),
                prompt_identifier=prompt_identifier,
            )
            self._set_static_value("#ccp-editor-prompt-version-status", self._format_prompt_versions(versions))
        except Exception as e:
            logger.error(f"Error listing prompt versions: {e}", exc_info=True)
            self._set_static_value("#ccp-editor-prompt-version-status", f"Version history unavailable: {e}")
            self._notify(f"Prompt version history unavailable: {e}", severity="warning")

    async def handle_restore_prompt_version(self) -> None:
        """Restore a selected server prompt version into the current prompt working state."""
        scope_service = self._prompt_scope_service()
        prompt_identifier = self._current_source_prompt_identifier()
        if scope_service is None or prompt_identifier is None:
            self._set_static_value("#ccp-editor-prompt-version-status", "Prompt version restore is unavailable.")
            return

        try:
            version_text = self.window.query_one("#ccp-editor-prompt-version-input", Input).value.strip()
            version = int(version_text)
        except Exception:
            self._set_static_value("#ccp-editor-prompt-version-status", "Enter a numeric version to restore.")
            self._notify("Enter a numeric prompt version to restore.", severity="warning")
            return

        try:
            prompt_data = await scope_service.restore_prompt_version(
                mode=self._current_prompt_backend(),
                prompt_identifier=prompt_identifier,
                version=version,
            )
            self.current_prompt_id = prompt_data.get("id", self.current_prompt_id)
            self.current_prompt_data = prompt_data
            self.window.post_message(PromptMessage.Updated(self.current_prompt_id, prompt_data))
            self._display_prompt_in_editor()
            self._set_static_value("#ccp-editor-prompt-version-status", f"Prompt restored v{version}.")
            self._notify(f"Prompt restored to version {version}.", severity="success")
        except Exception as e:
            logger.error(f"Error restoring prompt version: {e}", exc_info=True)
            self._set_static_value("#ccp-editor-prompt-version-status", f"Version restore failed: {e}")
            self._notify(f"Prompt version restore failed: {e}", severity="error")
    
    async def handle_import(self) -> None:
        """Handle import request - prompts for file selection."""
        from ...Widgets.enhanced_file_picker import EnhancedFileOpen, Filters
        
        try:
            # Create filters for prompt files
            # Filters need callable testers; glob strings crash the picker.
            filters = Filters(
                ("Prompt Files", lambda p: p.suffix.lower() in (".json", ".yaml", ".yml", ".txt")),
                ("JSON Files", lambda p: p.suffix.lower() == ".json"),
                ("YAML Files", lambda p: p.suffix.lower() in (".yaml", ".yml")),
                ("Text Files", lambda p: p.suffix.lower() == ".txt"),
                ("All Files", lambda p: True),
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
