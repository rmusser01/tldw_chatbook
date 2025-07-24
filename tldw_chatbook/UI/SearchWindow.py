# tldw_chatbook/UI/SearchWindow.py
#
#
# Imports
from __future__ import annotations
from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.css.query import QueryError
from textual.widgets import Static, Button, Input, Markdown, Select, Checkbox
#
# Third-Party Libraries
from typing import TYPE_CHECKING, Optional, Tuple, List, Union, Dict, Any
import asyncio
from loguru import logger
from pathlib import Path

from ..Notes.Notes_Library import NotesInteropService

# Configure logger with context
logger = logger.bind(module="SearchWindow")
#
# Local Imports
from tldw_chatbook.config import get_cli_setting
from ..DB.ChaChaNotes_DB import CharactersRAGDB
from ..DB.Client_Media_DB_v2 import MediaDatabase, DatabaseError
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase, DatabaseError as MediaDatabaseError
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from ..Utils.pagination import paginated_fetch

if TYPE_CHECKING:
    from ..app import TldwCli

# Import dependency checking system
from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Set availability flags based on centralized dependency checking
WEB_SEARCH_AVAILABLE = DEPENDENCIES_AVAILABLE.get('websearch', False)

if WEB_SEARCH_AVAILABLE:
    try:
        from ..Web_Scraping.WebSearch_APIs import generate_and_search, analyze_and_aggregate
        logger.info("✅ Web Search dependencies found. Feature is enabled.")
    except (ImportError, ModuleNotFoundError) as e:
        WEB_SEARCH_AVAILABLE = False
        logger.warning(f"⚠️ Web Search dependencies not found, feature will be disabled. Reason: {e}")
        # Define placeholders so the rest of the file doesn't crash if they are referenced.
        generate_and_search = None
        analyze_and_aggregate = None
else:
    # Define placeholders so the rest of the file doesn't crash if they are referenced.
    generate_and_search = None
    analyze_and_aggregate = None
#
#######################################################################################################################
#
# Functions:


# Constants for clarity
SEARCH_VIEW_RAG_QA = "search-view-rag-qa"
SEARCH_VIEW_RAG_CHAT = "search-view-rag-chat"
SEARCH_VIEW_RAG_MANAGEMENT = "search-view-rag-management"
SEARCH_VIEW_WEB_SEARCH = "search-view-web-search"

SEARCH_NAV_RAG_QA = "search-nav-rag-qa"
SEARCH_NAV_RAG_CHAT = "search-nav-rag-chat"
SEARCH_NAV_RAG_MANAGEMENT = "search-nav-rag-management"
SEARCH_NAV_WEB_SEARCH = "search-nav-web-search"

# UI Constant for "Local Server" provider display name
LOCAL_SERVER_PROVIDER_DISPLAY_NAME = "Local OpenAI-Compliant Server"
LOCAL_SERVER_PROVIDER_INTERNAL_ID = "local_openai_compliant"  # Internal ID to distinguish


class SearchWindow(Container):
    """
    Container for the Search Tab's UI, featuring a vertical tab bar and content areas.
    """

    # Database display name mapping
    DB_DISPLAY_NAMES = {
        "media_db": "Media Items",
        "rag_chat_db": "Chat Items",
        "char_chat_db": "Note Items"
    }

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        """Called when the window is first mounted."""
        logger.info("SearchWindow.on_mount: Setting and initializing initial active sub-tab.")

        # First, ensure all views are properly initialized but hidden
        for view in self.query(".search-view-area"):
            # Hide the view
            view.display = False
            logger.debug(f"SearchWindow.on_mount: Hiding view {view.id}")
            
            # Special handling for SearchRAGWindow to ensure it's properly initialized
            if view.id == SEARCH_VIEW_RAG_QA:
                # Initialize children but don't hide them yet to allow proper initialization
                for child in view.children:
                    logger.debug(f"SearchWindow.on_mount: Initializing child {child.__class__.__name__} of {view.id}")

        # Default to RAG QA view unless web search is explicitly preferred
        default_view = SEARCH_VIEW_RAG_QA
        
        # Only default to web search if explicitly configured to do so
        if WEB_SEARCH_AVAILABLE and hasattr(self.app_instance, 'get_setting') and self.app_instance.get_setting("search", "default_to_web_search", False):
            default_view = SEARCH_VIEW_WEB_SEARCH
        
        initial_sub_tab = self.app_instance.search_active_sub_tab or default_view
        self.app_instance.search_active_sub_tab = initial_sub_tab  # Ensure it's set
        logger.debug(f"SearchWindow.on_mount: Initial active sub-tab set to {initial_sub_tab}")

        # Set active navigation button
        nav_button_id = initial_sub_tab.replace("-view-", "-nav-")
        try:
            # Clear active class from all nav buttons first
            for button in self.query(".search-nav-button"):
                button.remove_class("-active-search-sub-view")
                
            # Set active class on the selected button
            nav_button = self.query_one(f"#{nav_button_id}")
            nav_button.add_class("-active-search-sub-view")
            logger.debug(f"SearchWindow.on_mount: Added active class to nav button {nav_button_id}")
        except Exception as e:
            logger.warning(f"SearchWindow.on_mount: Could not set active class for nav button {nav_button_id}: {e}")

        # Show the active view
        try:
            # Hide all views first
            for view in self.query(".search-view-area"):
                view.display = False
                
            # Show the active view
            active_view = self.query_one(f"#{initial_sub_tab}")
            active_view.display = True
            logger.debug(f"SearchWindow.on_mount: Set display=True for active view {initial_sub_tab}")
            
            # If it's the RAG QA view, ensure its children are properly displayed
            if initial_sub_tab == SEARCH_VIEW_RAG_QA:
                for child in active_view.children:
                    # Ensure the child is visible
                    child.display = True
                    
                    # Force a refresh of the child to ensure it's properly rendered
                    if hasattr(child, "refresh"):
                        try:
                            await child.refresh()
                            logger.debug(f"SearchWindow.on_mount: Refreshed child {child.__class__.__name__} of {initial_sub_tab}")
                        except Exception as refresh_error:
                            logger.warning(f"SearchWindow.on_mount: Could not refresh child {child.__class__.__name__}: {refresh_error}")
                    
                    # If it's a SearchRAGWindow, try to focus the search input
                    if child.__class__.__name__ == "SearchRAGWindow":
                        try:
                            if hasattr(child, "search_input"):
                                child.search_input.focus()
                                logger.debug("SearchWindow.on_mount: Focused search input in SearchRAGWindow")
                        except Exception as focus_error:
                            logger.warning(f"SearchWindow.on_mount: Could not focus search input: {focus_error}")
                    
            logger.debug(f"SearchWindow.on_mount: Set active view {initial_sub_tab} to visible")
        except Exception as e:
            logger.error(f"SearchWindow.on_mount: Could not display initial active view {initial_sub_tab}: {e}")
            # Try to show a fallback view instead of returning
            try:
                fallback_view = self.query_one(f"#{SEARCH_VIEW_WEB_SEARCH}")
                fallback_view.display = True
                logger.warning(f"SearchWindow.on_mount: Showing fallback view {SEARCH_VIEW_WEB_SEARCH} due to error")
            except Exception as fallback_error:
                logger.error(f"SearchWindow.on_mount: Could not show fallback view either: {fallback_error}")
                return

        logger.info(f"SearchWindow.on_mount: Initialized view {initial_sub_tab}")

    def compose(self) -> ComposeResult:
        with Vertical(id="search-left-nav-pane", classes="search-nav-pane"):
            yield Button("RAG QA", id=SEARCH_NAV_RAG_QA, classes="search-nav-button")
            yield Button("RAG Chat", id=SEARCH_NAV_RAG_CHAT, classes="search-nav-button")
            yield Button("RAG Management", id=SEARCH_NAV_RAG_MANAGEMENT, classes="search-nav-button")
            if WEB_SEARCH_AVAILABLE:
                yield Button("Web Search", id=SEARCH_NAV_WEB_SEARCH, classes="search-nav-button")
            else:
                yield Button("Web Search", id="search-nav-web-search-disabled", classes="search-nav-button disabled")

        with Container(id="search-content-pane", classes="search-content-pane"):
            # Import and use the new SearchRAGWindow for RAG Q&A
            try:
                from .SearchRAGWindow import SearchRAGWindow
                with Container(id=SEARCH_VIEW_RAG_QA, classes="search-view-area"):
                    yield SearchRAGWindow(app_instance=self.app_instance)
            except ImportError as e:
                logger.warning(f"Could not import SearchRAGWindow: {e}")
                # Create a placeholder container with a message
                with Container(id=SEARCH_VIEW_RAG_QA, classes="search-view-area"):
                    yield Static(
                        "⚠️ RAG Search functionality is not available.\n\n"
                        "To enable RAG search, install the required dependencies:\n"
                        "pip install -e '.[embeddings_rag]'",
                        classes="rag-unavailable-message"
                    )
            yield Container(id=SEARCH_VIEW_RAG_CHAT, classes="search-view-area")
            
            # RAG Management view with chunking templates
            with Container(id=SEARCH_VIEW_RAG_MANAGEMENT, classes="search-view-area"):
                with VerticalScroll():
                    try:
                        from ..Widgets.chunking_templates_widget import ChunkingTemplatesWidget
                        yield ChunkingTemplatesWidget(app_instance=self.app_instance)
                    except ImportError as e:
                        logger.warning(f"Could not import ChunkingTemplatesWidget: {e}")
                        yield Static(
                            "⚠️ Chunking Templates widget is not available.\n\n"
                            f"Error: {str(e)}",
                            classes="rag-unavailable-message"
                        )

            if WEB_SEARCH_AVAILABLE:
                with Container(id=SEARCH_VIEW_WEB_SEARCH, classes="search-view-area"):
                    with VerticalScroll():
                        yield Input(placeholder="Enter search query...", id="web-search-input")
                        yield Button("Search", id="web-search-button", classes="search-action-button")
                        yield VerticalScroll(Markdown("", id="web-search-results"))
            else:
                with Container(id=SEARCH_VIEW_WEB_SEARCH, classes="search-view-area"):
                    with VerticalScroll():
                        yield Markdown("### Web Search/Scraping Is Not Currently Installed\n\n...")

    # --- HELPER METHODS ---


    def _get_db_path(self, db_type: str) -> str:
        base_path_str = get_cli_setting("database", "chachanotes_db_path")
        if not base_path_str:
            return "Path not configured"
        base_path = Path(base_path_str)

        if db_type == "media_db":
            return get_cli_setting("database", "media_db_path", "Media DB Path Not Set")
        elif db_type == "rag_chat_db":
            return str(base_path.parent / "rag_qa.db")  # Example
        elif db_type == "char_chat_db":
            return str(base_path)  # Main ChaChaNotes DB
        return "Unknown DB Type"

    # --- EVENT HANDLERS (New and Refactored) ---


    @on(Button.Pressed, ".search-nav-button")
    async def handle_search_nav_button_pressed(self, event: Button.Pressed) -> None:
        """Handles all navigation button presses within the search tab."""
        event.stop()
        button_id = event.button.id
        if not button_id or "-disabled" in button_id: 
            logger.debug(f"SearchWindow.handle_nav: Ignoring disabled button press: {button_id}")
            return

        logger.info(f"SearchWindow.handle_nav: Search nav button '{button_id}' pressed.")
        target_view_id = button_id.replace("-nav-", "-view-")
        self.app_instance.search_active_sub_tab = target_view_id

        # Update navigation buttons
        for button in self.query(".search-nav-button"):
            button.remove_class("-active-search-sub-view")
        event.button.add_class("-active-search-sub-view")

        # Hide all views first
        for view in self.query(".search-view-area"):
            view.display = False
            logger.debug(f"SearchWindow.handle_nav: Hiding view: {view.id}")

        # Show the target view
        try:
            target_view = self.query_one(f"#{target_view_id}")
            target_view.display = True
            logger.debug(f"SearchWindow.handle_nav: Showing view: {target_view_id}")
            
            # Special handling for SearchRAGWindow
            if target_view_id == SEARCH_VIEW_RAG_QA:
                for child in target_view.children:
                    # Make sure the child is visible
                    child.display = True
                    logger.debug(f"SearchWindow.handle_nav: Set child {child.__class__.__name__} display=True")
                    
                    # Force a refresh if the child supports it
                    if hasattr(child, "refresh"):
                        try:
                            await child.refresh()
                            logger.debug(f"SearchWindow.handle_nav: Refreshed child: {child.__class__.__name__}")
                        except Exception as refresh_error:
                            logger.warning(f"SearchWindow.handle_nav: Could not refresh child {child.__class__.__name__}: {refresh_error}")
                    
                    # If it's a SearchRAGWindow, try to focus the search input
                    if child.__class__.__name__ == "SearchRAGWindow":
                        try:
                            if hasattr(child, "search_input"):
                                child.search_input.focus()
                                logger.debug("SearchWindow.handle_nav: Focused search input in SearchRAGWindow")
                        except Exception as focus_error:
                            logger.warning(f"SearchWindow.handle_nav: Could not focus search input: {focus_error}")
            
            # Notify the app that the view has changed
            if hasattr(self.app_instance, 'notify'):
                view_name = target_view_id.replace('search-view-', '').replace('-', ' ').title()
                self.app_instance.notify(
                    f"Switched to {view_name}", 
                    severity="information", 
                    timeout=2
                )
                                    
        except Exception as e:
            logger.error(f"SearchWindow.handle_nav: Failed to display target view {target_view_id}: {e}")
            if hasattr(self.app_instance, 'notify'):
                self.app_instance.notify(f"Error displaying view: {target_view_id}", severity="error")
            
            # Try to show a fallback view
            try:
                fallback_view = self.query_one(f"#{SEARCH_VIEW_WEB_SEARCH}")
                fallback_view.display = True
                logger.warning(f"SearchWindow.handle_nav: Showing fallback view {SEARCH_VIEW_WEB_SEARCH} due to error")
            except Exception as fallback_error:
                logger.error(f"SearchWindow.handle_nav: Could not show fallback view either: {fallback_error}")
            return

        logger.debug(f"SearchWindow.handle_nav: Successfully switched to view '{target_view_id}'")






    # --- Event Handlers ---



#
# End of SearchWindow.py
########################################################################################################################
