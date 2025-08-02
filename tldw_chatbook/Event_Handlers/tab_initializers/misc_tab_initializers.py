"""
Miscellaneous Tab Initializers - Handles initialization for various tabs.

This module contains initializers for tabs that have simpler initialization
requirements, including CCP, Media, Search, Ingest, Tools Settings, LLM, and Evals.
"""

from typing import TYPE_CHECKING
from textual.css.query import QueryError

from .base_initializer import BaseTabInitializer

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class CCPTabInitializer(BaseTabInitializer):
    """Handles Conversations/Characters/Prompts tab initialization."""
    
    def get_tab_id(self) -> str:
        """Return the CCP tab ID."""
        from tldw_chatbook.Constants import TAB_CCP
        return TAB_CCP
    
    async def on_tab_shown(self) -> None:
        """Initialize the CCP tab when shown."""
        self.log_initialization("CCP tab shown, scheduling widget population...")
        
        def populate_ccp_widgets():
            """Populate CCP widgets after ensuring window is ready."""
            try:
                # Check if the window is actually initialized
                if not self.is_window_ready("conversations_characters_prompts-window"):
                    self.logger.warning("CCP window is still a placeholder, skipping widget population")
                    return
                
                # Import here to avoid circular imports
                from tldw_chatbook.Event_Handlers import conv_char_events as ccp_handlers
                
                # Now it's safe to populate widgets
                self.call_async_handler(ccp_handlers.populate_ccp_character_select, self.app)
                self.call_async_handler(ccp_handlers.populate_ccp_prompts_list_view, self.app)
                self.call_async_handler(ccp_handlers.populate_ccp_dictionary_select, self.app)
                self.call_async_handler(ccp_handlers.populate_ccp_worldbook_list, self.app)
                self.call_async_handler(ccp_handlers.perform_ccp_conversation_search, self.app)
                
                self.log_initialization("CCP widget population initiated")
                
            except QueryError:
                self.logger.error("CCP window not found during widget population")
        
        # Use a timer to ensure the window is fully initialized
        self.schedule_initialization(populate_ccp_widgets)


class MediaTabInitializer(BaseTabInitializer):
    """Handles Media tab initialization."""
    
    def get_tab_id(self) -> str:
        """Return the Media tab ID."""
        from tldw_chatbook.Constants import TAB_MEDIA
        return TAB_MEDIA
    
    async def on_tab_shown(self) -> None:
        """Initialize the Media tab when shown."""
        self.log_initialization("Media tab shown, activating initial view...")
        
        def activate_media_initial_view():
            """Activate the media window's initial view."""
            try:
                from tldw_chatbook.UI.MediaWindow import MediaWindow
                media_window = self.app.query_one(MediaWindow)
                media_window.activate_initial_view()
                self.log_initialization("Media initial view activated")
            except QueryError:
                self.logger.error("Could not find MediaWindow to activate its initial view")
        
        # Use a timer to ensure the window is fully initialized
        self.schedule_initialization(activate_media_initial_view)


class SearchTabInitializer(BaseTabInitializer):
    """Handles Search tab initialization."""
    
    def get_tab_id(self) -> str:
        """Return the Search tab ID."""
        from tldw_chatbook.Constants import TAB_SEARCH
        return TAB_SEARCH
    
    async def on_tab_shown(self) -> None:
        """Initialize the Search tab when shown."""
        self.log_initialization("Search tab shown, initializing sub-tab...")
        
        def initialize_search_tab():
            """Initialize the search tab after ensuring window is ready."""
            try:
                # Check if the window is actually initialized
                if not self.is_window_ready("search-window"):
                    self.logger.warning("Search window is still a placeholder, skipping sub-tab initialization")
                    return
                
                # Now it's safe to set the active sub-tab
                if not self.app.search_active_sub_tab:
                    self.app.search_active_sub_tab = self.app._initial_search_sub_tab_view
                    self.log_initialization(f"Search sub-tab set to: {self.app._initial_search_sub_tab_view}")
                    
            except QueryError:
                self.logger.error("Search window not found during initialization")
        
        # Use a timer to ensure the window is fully initialized
        self.schedule_initialization(initialize_search_tab)


class IngestTabInitializer(BaseTabInitializer):
    """Handles Ingest tab initialization."""
    
    def get_tab_id(self) -> str:
        """Return the Ingest tab ID."""
        from tldw_chatbook.Constants import TAB_INGEST
        return TAB_INGEST
    
    async def on_tab_shown(self) -> None:
        """Initialize the Ingest tab when shown."""
        if not self.app.ingest_active_view:
            self.log_initialization(f"Ingest tab shown, activating initial view: {self.app._initial_ingest_view}")
            # Use call_later to ensure the UI has settled after tab switch
            self.app.call_later(self.app._activate_initial_ingest_view)


class ToolsSettingsTabInitializer(BaseTabInitializer):
    """Handles Tools & Settings tab initialization."""
    
    def get_tab_id(self) -> str:
        """Return the Tools & Settings tab ID."""
        from tldw_chatbook.Constants import TAB_TOOLS_SETTINGS
        return TAB_TOOLS_SETTINGS
    
    async def on_tab_shown(self) -> None:
        """Initialize the Tools & Settings tab when shown."""
        self.log_initialization("Tools & Settings tab shown, initializing...")
        
        def initialize_tools_settings():
            """Initialize tools settings after ensuring window is ready."""
            try:
                # Check if the window is actually initialized
                if not self.is_window_ready("tools_settings-window"):
                    # Window isn't initialized yet, try again later
                    self.schedule_initialization(initialize_tools_settings)
                    return
                
                # Now it's safe to activate the initial view
                from tldw_chatbook.UI.Tools_Settings_Window import ToolsSettingsWindow
                tools_window = self.app.query_one("#tools_settings-window")
                
                if isinstance(tools_window, ToolsSettingsWindow):
                    tools_window.activate_initial_view()
                    if not self.app.tools_settings_active_view:
                        self.app.tools_settings_active_view = self.app._initial_tools_settings_view
                        self.log_initialization(f"Tools & Settings view set to: {self.app._initial_tools_settings_view}")
                        
            except QueryError:
                self.logger.error("Tools settings window not found during initialization")
        
        # Use a timer to ensure the window is ready
        self.schedule_initialization(initialize_tools_settings)


class LLMTabInitializer(BaseTabInitializer):
    """Handles LLM Management tab initialization."""
    
    def get_tab_id(self) -> str:
        """Return the LLM tab ID."""
        from tldw_chatbook.Constants import TAB_LLM
        return TAB_LLM
    
    async def on_tab_shown(self) -> None:
        """Initialize the LLM tab when shown."""
        if not self.app.llm_active_view:
            self.log_initialization(f"LLM tab shown, activating initial view: {self.app._initial_llm_view}")
            self.app.call_later(setattr, self.app, 'llm_active_view', self.app._initial_llm_view)
        
        # Import here to avoid circular imports
        from tldw_chatbook.Event_Handlers.LLM_Management_Events import llm_management_events
        
        # Populate LLM help texts
        self.call_async_handler(llm_management_events.populate_llm_help_texts, self.app)
        self.log_initialization("LLM help texts population initiated")


class EvalsTabInitializer(BaseTabInitializer):
    """Handles Evals tab initialization."""
    
    def get_tab_id(self) -> str:
        """Return the Evals tab ID."""
        from tldw_chatbook.Constants import TAB_EVALS
        return TAB_EVALS
    
    async def on_tab_shown(self) -> None:
        """Initialize the Evals tab when shown."""
        self.log_initialization("Evals tab shown, activating initial view...")
        
        try:
            from tldw_chatbook.UI.Evals_Window_v3 import EvalsWindow
            evals_window = self.app.query_one(EvalsWindow)
            evals_window.activate_initial_view()
            self.log_initialization("Evals initial view activated")
        except QueryError:
            self.logger.error("Could not find EvalsWindow to activate its initial view")