"""
MediaWindow v2 - Orchestrator for media browsing components.

This is a refactored version that uses the new component-based architecture.
"""

from typing import TYPE_CHECKING, List, Optional, Dict, Any
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.widgets import Button, Markdown
from loguru import logger

# Import media components
from ..Widgets.Media import (
    MediaNavigationPanel,
    MediaSearchPanel,
    MediaSearchEvent,
    MediaListPanel,
    MediaItemSelectedEvent,
    MediaViewerPanel
)
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE

# Import events
from ..Widgets.Media.media_navigation_panel import MediaTypeSelectedEvent
from ..Event_Handlers.media_events import (
    MediaMetadataUpdateEvent,
    MediaDeleteConfirmationEvent,
    MediaUndeleteEvent,
    MediaListCollapseEvent,
    SidebarCollapseEvent,
    MediaAnalysisRequestEvent,
    MediaAnalysisSaveEvent,
    MediaAnalysisOverwriteEvent
)

if TYPE_CHECKING:
    from ..app import TldwCli


class MediaWindow(Container):
    """
    Orchestrator for the Media Tab components.
    
    Manages communication between navigation, search, list, and viewer panels.
    """
    
    DEFAULT_CSS = """
    MediaWindow {
        layout: horizontal;
        height: 100%;
        width: 100%;
    }
    
    MediaWindow .main-content {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }
    
    MediaWindow .content-area {
        layout: horizontal;
        height: 1fr;
        width: 100%;
    }
    
    MediaWindow #media-list-panel.collapsed {
        display: none;
    }
    """
    
    # Reactive properties
    active_media_type: reactive[Optional[str]] = reactive(None)
    selected_media_id: reactive[Optional[int]] = reactive(None)
    media_active_view: reactive[Optional[str]] = reactive(None)
    sidebar_collapsed: reactive[bool] = reactive(False)
    list_collapsed: reactive[bool] = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the MediaWindow."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_types = self._get_media_types()
        
    def _get_media_types(self) -> List[str]:
        """Get media types from the app instance."""
        return getattr(self.app_instance, '_media_types_for_ui', [])
    
    def compose(self) -> ComposeResult:
        """Compose the MediaWindow UI."""
        # Navigation panel
        self.nav_panel = MediaNavigationPanel(
            self.app_instance,
            self.media_types,
            id="media-nav-panel"
        )
        yield self.nav_panel
        
        # Main content area
        with Container(classes="main-content"):
            # Search panel
            self.search_panel = MediaSearchPanel(
                self.app_instance,
                id="media-search-panel"
            )
            yield self.search_panel
            
            # Content area with list and viewer
            with Horizontal(classes="content-area"):
                # List panel
                self.list_panel = MediaListPanel(
                    self.app_instance,
                    id="media-list-panel"
                )
                yield self.list_panel
                
                # Viewer panel
                self.viewer_panel = MediaViewerPanel(
                    self.app_instance,
                    id="media-viewer-panel"
                )
                yield self.viewer_panel
    
    def on_mount(self) -> None:
        """Called when the MediaWindow is mounted."""
        logger.info("MediaWindow v2 mounted")
        
        # Don't activate initial view here - let activate_initial_view handle it
    
    def watch_sidebar_collapsed(self, collapsed: bool) -> None:
        """React to sidebar collapse changes."""
        self.nav_panel.collapsed = collapsed
    
    def watch_list_collapsed(self, collapsed: bool) -> None:
        """React to list collapse changes."""
        # Toggle display of list panel
        if collapsed:
            self.list_panel.add_class("collapsed")
        else:
            self.list_panel.remove_class("collapsed")
    
    @on(MediaTypeSelectedEvent)
    def handle_media_type_selected(self, event: MediaTypeSelectedEvent) -> None:
        """Handle media type selection from navigation panel."""
        logger.info(f"Media type selected: {event.type_slug}")
        self.activate_media_type(event.type_slug, event.display_name)
    
    @on(MediaSearchEvent)
    def handle_media_search(self, event: MediaSearchEvent) -> None:
        """Handle search event from search panel."""
        logger.info(f"Search triggered: term='{event.search_term}', keywords='{event.keyword_filter}'")
        
        if self.active_media_type:
            # Perform search
            self._perform_search(
                self.active_media_type,
                event.search_term,
                event.keyword_filter
            )
    
    @on(MediaItemSelectedEvent)
    def handle_media_item_selected(self, event: MediaItemSelectedEvent) -> None:
        """Handle media item selection from list panel."""
        logger.info(f"Media item selected: {event.media_id}")
        self.selected_media_id = event.media_id
        
        # Fetch full media data including content
        if self.app_instance.media_db:
            full_media_data = self.app_instance.media_db.get_media_by_id(event.media_id, include_trash=True)
            if full_media_data:
                # Also fetch the latest document version to get analysis
                try:
                    from ..DB.Client_Media_DB_v2 import get_document_version
                    doc_version = get_document_version(self.app_instance.media_db, event.media_id, include_content=False)
                    if doc_version and doc_version.get('analysis_content'):
                        full_media_data['analysis'] = doc_version['analysis_content']
                except Exception as e:
                    logger.debug(f"Could not fetch document version for analysis: {e}")
                
                self.viewer_panel.load_media(full_media_data)
            else:
                logger.error(f"Failed to fetch full data for media ID {event.media_id}")
                # Fall back to partial data
                self.viewer_panel.load_media(event.media_data)
        else:
            # No database available, use partial data
            self.viewer_panel.load_media(event.media_data)
    
    @on(MediaMetadataUpdateEvent)
    async def handle_metadata_update(self, event: MediaMetadataUpdateEvent) -> None:
        """Handle metadata update from viewer panel."""
        # Set the type slug
        event.type_slug = self.active_media_type or ""
        
        # Forward to existing handler
        from ..Event_Handlers import media_events
        await media_events.handle_media_metadata_update(self.app_instance, event)
        
        # Refresh the list
        if self.active_media_type:
            search_term = self.search_panel.search_term
            keyword_filter = self.search_panel.keyword_filter
            self._perform_search(
                self.active_media_type,
                search_term,
                keyword_filter
            )
    
    @on(MediaDeleteConfirmationEvent)
    async def handle_delete_confirmation(self, event: MediaDeleteConfirmationEvent) -> None:
        """Handle delete confirmation from viewer panel."""
        # Set the type slug
        event.type_slug = self.active_media_type or ""
        
        # Forward to existing handler
        from ..Event_Handlers import media_events
        await media_events.handle_media_delete_confirmation(self.app_instance, event)
    
    @on(MediaUndeleteEvent)
    async def handle_media_undelete(self, event: MediaUndeleteEvent) -> None:
        """Handle undelete event."""
        from ..Event_Handlers import media_events
        await media_events.handle_media_undelete(self.app_instance, event)
        
        # Refresh the list
        if self.active_media_type:
            search_term = self.search_panel.search_term
            keyword_filter = self.search_panel.keyword_filter
            self._perform_search(
                self.active_media_type,
                search_term,
                keyword_filter
            )
    
    @on(MediaListCollapseEvent)
    def handle_list_collapse(self) -> None:
        """Handle list collapse toggle."""
        self.list_collapsed = not self.list_collapsed
        # Update button text
        try:
            button = self.viewer_panel.query_one("#collapse-media-list", Button)
            button.label = "▶" if self.list_collapsed else "◀"
        except Exception:
            pass
    
    @on(SidebarCollapseEvent)
    def handle_sidebar_collapse(self) -> None:
        """Handle sidebar collapse toggle."""
        self.sidebar_collapsed = not self.sidebar_collapsed
        # Update button text
        try:
            button = self.search_panel.query_one("#media-sidebar-toggle", Button)
            from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
            # Toggle between collapsed and expanded state
            if self.sidebar_collapsed:
                button.label = "▶"  # Arrow pointing right when collapsed
            else:
                button.label = get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE)
        except Exception:
            pass
    
    @on(MediaAnalysisRequestEvent)
    def handle_analysis_request(self, event: MediaAnalysisRequestEvent) -> None:
        """Handle media analysis request."""
        # Set the type_slug
        event.type_slug = self.active_media_type or ""
        
        async def perform_analysis():
            try:
                # Get media content
                if not self.app_instance.media_db:
                    self.app_instance.notify("Media database not available", severity="error")
                    return
                
                media_data = self.app_instance.media_db.get_media_by_id(event.media_id)
                if not media_data:
                    self.app_instance.notify("Media item not found", severity="error")
                    return
                
                # Use the same chat_wrapper as the chat window
                logger.info(f"Calling {event.provider} with model {event.model} for analysis")
                
                # Prepare the chat history format expected by chat_wrapper
                chat_history = []
                
                # Get API key using the same method as chat
                api_key_for_call = None
                provider_settings_key = event.provider.lower().replace(" ", "_")
                provider_config_settings = self.app_instance.app_config.get("api_settings", {}).get(provider_settings_key, {})
                
                # First check direct api_key field
                if "api_key" in provider_config_settings:
                    config_api_key = provider_config_settings.get("api_key", "").strip()
                    if config_api_key and config_api_key != "<API_KEY_HERE>":
                        api_key_for_call = config_api_key
                        logger.debug(f"Using API key for '{event.provider}' from config file field.")
                
                # If not found, check environment variable
                if not api_key_for_call:
                    env_var_name = provider_config_settings.get("api_key_env_var", "").strip()
                    if env_var_name:
                        import os
                        env_api_key = os.environ.get(env_var_name, "").strip()
                        if env_api_key:
                            api_key_for_call = env_api_key
                            logger.debug(f"Using API key for '{event.provider}' from ENV var '{env_var_name}'.")
                
                # Check if key is required
                providers_requiring_key = ["OpenAI", "Anthropic", "Google", "MistralAI", "Groq", "Cohere", "OpenRouter", "HuggingFace", "DeepSeek"]
                if event.provider in providers_requiring_key and not api_key_for_call:
                    self.app_instance.notify(f"API Key for {event.provider} is missing.", severity="error")
                    return
                
                # Use chat_wrapper with the same parameters as the chat window
                def call_llm():
                    return self.app_instance.chat_wrapper(
                        message=event.user_prompt or "",
                        history=chat_history,
                        media_content={},  # No media for analysis
                        api_endpoint=event.provider,
                        api_key=api_key_for_call,
                        custom_prompt="",
                        temperature=event.temperature,
                        system_message=event.system_prompt or "",
                        streaming=False,  # No streaming for analysis
                        minp=event.min_p,
                        model=event.model,
                        topp=event.top_p,
                        topk=50,
                        llm_max_tokens=event.max_tokens,
                        llm_seed=None,
                        llm_stop=None,
                        llm_response_format=None,
                        llm_n=1,
                        llm_user_identifier=None,
                        llm_logprobs=False,
                        llm_top_logprobs=None,
                        llm_logit_bias=None,
                        llm_presence_penalty=0,
                        llm_frequency_penalty=0,
                        llm_tools=None,
                        llm_tool_choice=None,
                        llm_fixed_tokens_kobold=None,
                        current_image_input={},
                        selected_parts=[],
                        chatdict_entries=None,
                        max_tokens=500,
                        strategy="sorted_evenly",
                        strip_thinking_tags=False
                    )
                
                # Run in thread since it's a sync function
                import asyncio
                response = await asyncio.to_thread(call_llm)
                
                if response and isinstance(response, str):
                    # Update display directly
                    try:
                        analysis_display = self.viewer_panel.query_one("#analysis-display", Markdown)
                        await analysis_display.update(response)
                        # Update the viewer panel's current analysis state
                        self.viewer_panel.current_analysis = response
                        self.viewer_panel._update_analysis_button_states()
                    except Exception as e:
                        logger.error(f"Error updating analysis display: {e}")
                    
                    self.app_instance.notify("Analysis generated successfully", severity="information")
                else:
                    self.app_instance.notify("Failed to generate analysis", severity="error")
                    # Reset analysis display on failure
                    try:
                        analysis_display = self.viewer_panel.query_one("#analysis-display", Markdown)
                        await analysis_display.update("*Analysis generation failed*")
                    except:
                        pass
                    
            except Exception as e:
                logger.error(f"Error performing analysis: {e}", exc_info=True)
                self.app_instance.notify(f"Error: {str(e)[:100]}", severity="error")
        
        # Run the analysis in a worker
        self.run_worker(perform_analysis(), exclusive=True)
    
    @on(MediaAnalysisSaveEvent)
    def handle_analysis_save(self, event: MediaAnalysisSaveEvent) -> None:
        """Handle saving new analysis."""
        event.type_slug = self.active_media_type or ""
        
        try:
            if not self.app_instance.media_db:
                self.app_instance.notify("Media database not available", severity="error")
                return
            
            # Get the current content
            media_data = self.app_instance.media_db.get_media_by_id(event.media_id)
            if not media_data:
                self.app_instance.notify("Media item not found", severity="error")
                return
            
            # Create a new document version with the analysis
            version_info = self.app_instance.media_db.create_document_version(
                media_id=event.media_id,
                content=media_data.get('content', ''),
                analysis_content=event.analysis_content
            )
            
            if version_info:
                self.app_instance.notify("Analysis saved successfully", severity="information")
                # Update the viewer to show it's now saved
                self.viewer_panel.has_existing_analysis = True
                self.viewer_panel._update_analysis_button_states()
            else:
                self.app_instance.notify("Failed to save analysis", severity="error")
                
        except Exception as e:
            logger.error(f"Error saving analysis: {e}", exc_info=True)
            self.app_instance.notify(f"Error: {str(e)[:100]}", severity="error")
    
    @on(MediaAnalysisOverwriteEvent)
    def handle_analysis_overwrite(self, event: MediaAnalysisOverwriteEvent) -> None:
        """Handle overwriting existing analysis."""
        event.type_slug = self.active_media_type or ""
        
        try:
            if not self.app_instance.media_db:
                self.app_instance.notify("Media database not available", severity="error")
                return
            
            # Get the current content
            media_data = self.app_instance.media_db.get_media_by_id(event.media_id)
            if not media_data:
                self.app_instance.notify("Media item not found", severity="error")
                return
            
            # Create a new document version with the updated analysis
            # This will overwrite the analysis for this media item
            version_info = self.app_instance.media_db.create_document_version(
                media_id=event.media_id,
                content=media_data.get('content', ''),
                analysis_content=event.analysis_content
            )
            
            if version_info:
                self.app_instance.notify("Analysis overwritten successfully", severity="information")
                # Update the viewer state
                self.viewer_panel.has_existing_analysis = True
                self.viewer_panel.current_analysis = event.analysis_content
                self.viewer_panel._update_analysis_button_states()
            else:
                self.app_instance.notify("Failed to overwrite analysis", severity="error")
                
        except Exception as e:
            logger.error(f"Error overwriting analysis: {e}", exc_info=True)
            self.app_instance.notify(f"Error: {str(e)[:100]}", severity="error")
    
    def activate_media_type(self, type_slug: str, display_name: str) -> None:
        """Activate a media type and perform initial search."""
        logger.info(f"activate_media_type called: type_slug='{type_slug}', display_name='{display_name}'")
        self.active_media_type = type_slug
        
        # Update navigation panel
        self.nav_panel.selected_type = type_slug
        
        # Update search panel
        self.search_panel.set_type_filter(type_slug, display_name)
        
        # Clear viewer
        self.viewer_panel.clear_display()
        
        # Reset page to 1 when switching types
        self.list_panel.current_page = 1
        
        # Perform search
        logger.info(f"About to call _perform_search for type '{type_slug}'")
        self._perform_search(type_slug, "", "")
    
    def activate_initial_view(self) -> None:
        """Activate the initial view - called by app.py."""
        if not self.active_media_type and self.media_types:
            # Try "All Media" first
            if "All Media" in self.media_types:
                self.activate_media_type("all-media", "All Media")
            elif self.media_types:
                # Fall back to first available type
                from ..Utils.text import slugify
                first_type = self.media_types[0]
                self.activate_media_type(slugify(first_type), first_type)
    
    def update_search_results(self, results: List[Dict[str, Any]], page: int, total_pages: int) -> None:
        """Update search results in the list panel."""
        try:
            logger.info(f"Updating search results: {len(results)} items, page {page}/{total_pages}")
            self.list_panel.load_items(results, page, total_pages)
        except Exception as e:
            logger.error(f"Error updating search results: {e}", exc_info=True)
    
    def watch_media_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        """React to media_active_view changes from app.py button handlers."""
        if new_view and new_view.startswith("media-view-"):
            type_slug = new_view.replace("media-view-", "")
            # Find display name from media types
            display_name = type_slug.replace("-", " ").title()
            # Special case handling for known types
            if type_slug == "all-media":
                display_name = "All Media"
            elif type_slug == "analysis-review":
                display_name = "Analysis Review"
            elif type_slug == "collections-tags":
                display_name = "Collections/Tags"
            elif type_slug == "multi-item-review":
                display_name = "Multi-Item Review"
            
            self.activate_media_type(type_slug, display_name)
    
    def _perform_search(self, type_slug: str, search_term: str, keyword_filter: str) -> None:
        """Trigger media search in background."""
        logger.info(f"_perform_search called: type_slug='{type_slug}', search_term='{search_term}', keyword_filter='{keyword_filter}'")
        
        # Use run_worker with an async coroutine
        async def perform_search():
            logger.info(f"perform_search coroutine executing for type '{type_slug}'")
            try:
                if not self.app_instance.media_db:
                    logger.error("Media DB service not available")
                    return
                
                # Skip search for special windows
                if type_slug in ["collections-tags", "multi-item-review"]:
                    logger.info(f"Skipping search for special window: {type_slug}")
                    return
                
                # Set loading state
                self.list_panel.set_loading(True)
                
                # Prepare search parameters
                media_types_filter = None
                if type_slug not in ["all-media", "analysis-review"]:
                    db_media_type = type_slug.replace('-', '_')
                    media_types_filter = [db_media_type]
                
                # Parse keywords
                keywords_list = None
                if keyword_filter:
                    keywords_list = [k.strip() for k in keyword_filter.split(',') if k.strip()]

                # Search for media items using search_media_db method
                results, total_matches = self.app_instance.media_db.search_media_db(
                    search_query=search_term if search_term else None,
                    media_types=media_types_filter,
                    search_fields=['title', 'content', 'author', 'url', 'type', 'analysis_content'],
                    must_have_keywords=keywords_list,
                    sort_by="last_modified_desc",
                    page=self.list_panel.current_page,
                    results_per_page=self.list_panel.items_per_page,
                    include_trash=False,
                    include_deleted=self.search_panel.show_deleted
                )
                
                logger.info(f"Search returned {len(results)} results, total matches: {total_matches}")
                
                if results:
                    # Calculate total pages
                    total_pages = (total_matches + self.list_panel.items_per_page - 1) // self.list_panel.items_per_page
                    
                    # Update the list panel
                    self.update_search_results(
                        results,
                        self.list_panel.current_page,
                        total_pages
                    )
                    
                    logger.info(f"Found {len(results)} items for type '{type_slug}' (page {self.list_panel.current_page}/{total_pages})")
                else:
                    # No results
                    self.update_search_results([], 1, 1)
                    logger.info(f"No media items found for type '{type_slug}'")
                
                # Always set loading false when done
                self.list_panel.set_loading(False)
                    
            except Exception as e:
                logger.error(f"Error during media search: {e}", exc_info=True)
                self.list_panel.set_loading(False)
                # Notify user of error
                self.app_instance.notify(f"Error loading media: {str(e)[:100]}", severity="error")
        
        # Run the worker
        self.run_worker(perform_search(), exclusive=True)

#
# End of MediaWindow_v2.py
##############################A###############A###############A###############A###############A###############A
