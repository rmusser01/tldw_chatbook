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
from textual.widgets import Static, Button, Input, Markdown, Select, Checkbox, Label
#
# Third-Party Libraries
from typing import TYPE_CHECKING, Optional, Tuple, List, Union, Dict, Any
import asyncio
import inspect
from loguru import logger
from pathlib import Path

from ..Chat.chat_handoff_messages import USE_IN_CHAT_UNAVAILABLE_RECOVERY
from ..Notes.Notes_Library import NotesInteropService
from .Views.RAGSearch.search_handoff import build_search_chat_handoff_payload
from .Views.RAGSearch.search_result import SearchResult

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
SEARCH_VIEW_EMBEDDINGS_CREATE = "search-view-embeddings-create"
SEARCH_VIEW_EMBEDDINGS_MANAGE = "search-view-embeddings-manage"

SEARCH_NAV_RAG_QA = "search-nav-rag-qa"
SEARCH_NAV_RAG_CHAT = "search-nav-rag-chat"
SEARCH_NAV_RAG_MANAGEMENT = "search-nav-rag-management"
SEARCH_NAV_WEB_SEARCH = "search-nav-web-search"
SEARCH_NAV_EMBEDDINGS_CREATE = "search-nav-embeddings-create"
SEARCH_NAV_EMBEDDINGS_MANAGE = "search-nav-embeddings-manage"

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
        self.web_search_results: List[Dict[str, Any]] = []

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
            # Add Embeddings navigation buttons
            yield Button("Create Embeddings", id=SEARCH_NAV_EMBEDDINGS_CREATE, classes="search-nav-button")
            yield Button("Manage Embeddings", id=SEARCH_NAV_EMBEDDINGS_MANAGE, classes="search-nav-button")

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
                        yield VerticalScroll(id="web-search-results-list")
            else:
                with Container(id=SEARCH_VIEW_WEB_SEARCH, classes="search-view-area"):
                    with VerticalScroll():
                        yield Markdown("### Web Search/Scraping Is Not Currently Installed\n\n...")
                        
            # Embeddings views
            # Create Embeddings View - Now using new SearchEmbeddingsWindow
            with Container(id=SEARCH_VIEW_EMBEDDINGS_CREATE, classes="search-view-area"):
                try:
                    # Use the new SearchEmbeddingsWindow for improved UX
                    from .SearchEmbeddingsWindow import SearchEmbeddingsWindow
                    yield SearchEmbeddingsWindow(app_instance=self.app_instance)
                    logger.info("Using new SearchEmbeddingsWindow for embeddings creation")
                except ImportError as e:
                    logger.error(f"Failed to import SearchEmbeddingsWindow, falling back to wizard: {e}")
                    # Fallback to the original wizard if new window fails
                    try:
                        from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE, force_recheck_embeddings
                        embeddings_available = force_recheck_embeddings()
                        logger.info(f"Embeddings dependencies check in SearchWindow: {embeddings_available}")
                        
                        if embeddings_available:
                            from ..UI.Wizards.EmbeddingsWizard import SimpleEmbeddingsWizard
                            yield SimpleEmbeddingsWizard(app_instance=self.app_instance)
                        else:
                            yield Static(
                                "⚠️ Embeddings Creation functionality is not available.\n\n"
                                "Required dependencies are not properly detected.\n"
                                "Please restart the application.",
                                classes="embeddings-unavailable-message"
                            )
                    except ImportError as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                        yield Static(
                            "⚠️ Embeddings creation view is not available.\n\n"
                            f"Error: {str(e)}",
                            classes="embeddings-unavailable-message"
                        )
                    
            # Manage Embeddings View
            with Container(id=SEARCH_VIEW_EMBEDDINGS_MANAGE, classes="search-view-area"):
                try:
                    # Check dependencies again
                    from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE
                    if DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
                        from ..UI.Embeddings_Management_Window import EmbeddingsManagementWindow
                        yield EmbeddingsManagementWindow(app_instance=self.app_instance)
                    else:
                        yield Static(
                            "⚠️ Embeddings Management functionality is not available.\n\n"
                            "Required dependencies are not properly detected.\n"
                            "Please restart the application.",
                            classes="embeddings-unavailable-message"
                        )
                except ImportError as e:
                    logger.error(f"Could not import EmbeddingsManagementWindow: {e}")
                    yield Static(
                        "⚠️ Embeddings Management functionality is not available.\n\n"
                        f"Error: {str(e)}",
                        classes="embeddings-unavailable-message"
                    )

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

    def _authoritative_runtime_backend(self) -> str:
        get_source = getattr(self.app_instance, "get_authoritative_runtime_source", None)
        backend = get_source() if callable(get_source) else "local"
        backend = str(backend or "local").strip().lower()
        return backend if backend in {"local", "server"} else "local"

    def _build_search_chat_handoff_payload(self, result: Dict[str, Any]):
        return build_search_chat_handoff_payload(
            dict(result),
            runtime_backend=self._authoritative_runtime_backend(),
        )

    def _normalize_web_search_results(self, raw_results: Any, query: str) -> List[Dict[str, Any]]:
        if isinstance(raw_results, dict) and "web_search_results_dict" in raw_results:
            raw_results = raw_results.get("web_search_results_dict", {}).get("results", [])
        elif isinstance(raw_results, dict):
            raw_results = raw_results.get("results", [])

        normalized: List[Dict[str, Any]] = []
        for result in raw_results or []:
            if not isinstance(result, dict):
                continue
            url = result.get("url") or result.get("link") or result.get("href") or ""
            normalized.append(
                {
                    "title": result.get("title") or result.get("name") or url or "Web Result",
                    "content": result.get("snippet") or result.get("content") or result.get("description") or "",
                    "source": "web",
                    "score": result.get("score", 0.5),
                    "metadata": {
                        "url": url,
                        "displayUrl": result.get("displayUrl") or result.get("display_url") or url,
                        "query": query,
                    },
                }
            )
        return normalized

    async def _render_web_search_result_cards(self) -> None:
        results_list = self.query_one("#web-search-results-list")
        await results_list.remove_children()
        for index, result in enumerate(self.web_search_results):
            await results_list.mount(SearchResult(result, index))

    # --- EVENT HANDLERS (New and Refactored) ---


    @on(Button.Pressed, "#web-search-button")
    async def handle_web_search_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        query = self.query_one("#web-search-input", Input).value.strip()
        if not query:
            self.app_instance.notify("Enter a web search query.", severity="warning")
            return
        if generate_and_search is None:
            self.app_instance.notify("Web Search is not available.", severity="warning")
            return

        search_params = {
            "engine": get_cli_setting("search", "web_search_engine", "google"),
            "content_country": "US",
            "search_lang": "en",
            "output_lang": "en",
            "result_count": 10,
            "subquery_generation": False,
        }
        try:
            raw_results = generate_and_search(query, search_params)
            if inspect.isawaitable(raw_results):
                raw_results = await raw_results
            self.web_search_results = self._normalize_web_search_results(raw_results, query)
            await self._render_web_search_result_cards()
            self.app_instance.notify(
                f"Web Search completed: {len(self.web_search_results)} results found",
                severity="information",
            )
        except Exception as exc:
            logger.error(f"Web Search failed: {exc}", exc_info=True)
            self.app_instance.notify(f"Web Search failed: {exc}", severity="error")

    @on(SearchResult.UseInChatRequested)
    def handle_search_result_use_in_chat(self, event: SearchResult.UseInChatRequested) -> None:
        event.stop()
        payload = self._build_search_chat_handoff_payload(event.result)
        open_chat = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat):
            self.app_instance.notify(USE_IN_CHAT_UNAVAILABLE_RECOVERY, severity="warning")
            return
        open_chat(payload)


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
            
            # Special handling for embeddings create view to reactivate wizard
            elif target_view_id == SEARCH_VIEW_EMBEDDINGS_CREATE:
                for child in target_view.children:
                    # Make sure the child is visible
                    child.display = True
                    
                    # If it's a SimpleEmbeddingsWizard, reactivate the current step
                    if child.__class__.__name__ == "SimpleEmbeddingsWizard":
                        try:
                            # Find the actual wizard inside the SimpleEmbeddingsWizard
                            if hasattr(child, 'wizard') and child.wizard:
                                wizard = child.wizard
                                # Reactivate the current step
                                if hasattr(wizard, 'steps') and wizard.steps and hasattr(wizard, 'current_step'):
                                    current_step_index = wizard.current_step or 0
                                    if 0 <= current_step_index < len(wizard.steps):
                                        current_step = wizard.steps[current_step_index]
                                        current_step.add_class("active")
                                        current_step.remove_class("hidden")
                                        # Call on_show to fully reactivate the step
                                        if hasattr(current_step, 'on_show'):
                                            current_step.on_show()
                                        logger.debug(f"SearchWindow.handle_nav: Reactivated wizard step {current_step_index}")
                        except Exception as wizard_error:
                            logger.warning(f"SearchWindow.handle_nav: Could not reactivate wizard step: {wizard_error}")
            
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
