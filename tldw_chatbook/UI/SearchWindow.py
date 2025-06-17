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
from tldw_chatbook.Embeddings.Chroma_Lib import ChromaDBManager
from tldw_chatbook.config import get_cli_setting
from ..DB.ChaChaNotes_DB import CharactersRAGDB
from ..DB.Client_Media_DB_v2 import MediaDatabase, DatabaseError
from ..Embeddings.Embeddings_Lib import ModelCfg
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase, DatabaseError as MediaDatabaseError
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
if TYPE_CHECKING:
    from ..app import TldwCli
try:
    from ..Embeddings.Embeddings_Lib import *
    EMBEDDINGS_GENERATION_AVAILABLE = True
    logger.info("✅ Embeddings Generation dependencies found. Feature is enabled.")
except (ImportError, ModuleNotFoundError) as e:
    EMBEDDINGS_GENERATION_AVAILABLE = False
    logger.warning(f"Embeddings Generation depenedencies not found, features related will be disabled. Reason: {e}")
try:
    from ..Embeddings.Chroma_Lib import *
    VECTORDB_AVAILABLE = True
    logger.info("✅ Vector Database dependencies found. Feature is enabled.")
except (ImportError, ModuleNotFoundError) as e:
    VECTORDB_AVAILABLE = False
    logger.warning(f"Vector Database depenedencies not found, features related will be disabled. Reason: {e}")
try:
    from ..Web_Scraping.WebSearch_APIs import generate_and_search, analyze_and_aggregate

    WEB_SEARCH_AVAILABLE = True
    logger.info("✅ Web Search dependencies found. Feature is enabled.")
except (ImportError, ModuleNotFoundError) as e:
    WEB_SEARCH_AVAILABLE = False
    # Use a warning here, as this is an expected condition, not a critical error.
    logger.warning(f"⚠️ Web Search dependencies not found, feature will be disabled. Reason: {e}")
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
SEARCH_VIEW_EMBEDDINGS_CREATION = "search-view-embeddings-creation"
SEARCH_VIEW_RAG_MANAGEMENT = "search-view-rag-management"
SEARCH_VIEW_EMBEDDINGS_MANAGEMENT = "search-view-embeddings-management"
SEARCH_VIEW_WEB_SEARCH = "search-view-web-search"

SEARCH_NAV_RAG_QA = "search-nav-rag-qa"
SEARCH_NAV_RAG_CHAT = "search-nav-rag-chat"
SEARCH_NAV_EMBEDDINGS_CREATION = "search-nav-embeddings-creation"
SEARCH_NAV_RAG_MANAGEMENT = "search-nav-rag-management"
SEARCH_NAV_EMBEDDINGS_MANAGEMENT = "search-nav-embeddings-management"
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
        self._chroma_manager: Union["ChromaDBManager", None] = None
        # self._selected_embedding_id: Union[str, None] = None # For management view item ID from DB
        self._selected_embedding_collection_item_id: Union[
            str, None] = None  # For management view, ID of item in DB (e.g. media_1)
        self._selected_chroma_collection_name: Union[str, None] = None  # For management view

        # State to hold the mapping from display choice in dropdown to actual item ID
        self._mgmt_item_mapping: dict[str, str] = {}  # For management view items
        self._creation_item_mapping: dict[str, str] = {}  # For creation view items
        self._selected_item_display_name: Optional[str] = None  # For management view display name
        
        # Status history for embedding creation
        self._status_history: List[str] = []

    async def on_mount(self) -> None:
        """Called when the window is first mounted."""
        logger.info("SearchWindow.on_mount: Setting and initializing initial active sub-tab.")

        for view in self.query(".search-view-area"):
            view.display = False
            logger.debug(f"SearchWindow.on_mount: Setting view {view.id} display to False")

        initial_sub_tab = self.app_instance.search_active_sub_tab or SEARCH_VIEW_EMBEDDINGS_CREATION
        self.app_instance.search_active_sub_tab = initial_sub_tab  # Ensure it's set
        logger.debug(f"SearchWindow.on_mount: Initial active sub-tab set to {initial_sub_tab}")

        nav_button_id = initial_sub_tab.replace("-view-", "-nav-")
        try:
            nav_button = self.query_one(f"#{nav_button_id}")
            nav_button.add_class("-active-search-sub-view")
            logger.debug(f"SearchWindow.on_mount: Added active class to nav button {nav_button_id}")
        except Exception as e:
            logger.warning(f"SearchWindow.on_mount: Could not set active class for nav button {nav_button_id}: {e}")

        try:
            active_view = self.query_one(f"#{initial_sub_tab}")
            active_view.display = True
            logger.debug(f"SearchWindow.on_mount: Set display=True for active view {initial_sub_tab}")
        except Exception as e:
            logger.error(f"SearchWindow.on_mount: Could not display initial active view {initial_sub_tab}: {e}")
            return

        logger.info(f"SearchWindow.on_mount: Initializing view {initial_sub_tab}")
        if initial_sub_tab == SEARCH_VIEW_EMBEDDINGS_CREATION:
            await self._initialize_embeddings_creation_view()
        elif initial_sub_tab == SEARCH_VIEW_EMBEDDINGS_MANAGEMENT:
            await self._initialize_embeddings_management_view()

    def compose(self) -> ComposeResult:
        with Vertical(id="search-left-nav-pane", classes="search-nav-pane"):
            yield Button("RAG QA", id=SEARCH_NAV_RAG_QA, classes="search-nav-button")
            yield Button("RAG Chat", id=SEARCH_NAV_RAG_CHAT, classes="search-nav-button")
            yield Button("RAG Management", id=SEARCH_NAV_RAG_MANAGEMENT, classes="search-nav-button")
            if WEB_SEARCH_AVAILABLE:
                yield Button("Web Search", id=SEARCH_NAV_WEB_SEARCH, classes="search-nav-button")
            else:
                yield Button("Web Search", id="search-nav-web-search-disabled", classes="search-nav-button disabled")
            if EMBEDDINGS_GENERATION_AVAILABLE and VECTORDB_AVAILABLE:
                yield Button("Embeddings Creation", id=SEARCH_NAV_EMBEDDINGS_CREATION, classes="search-nav-button")
            else:
                yield Button("Embeddings Creation", id="search-nav-embeddings-creation-disabled",
                             classes="search-nav-button disabled")
            if VECTORDB_AVAILABLE:
                yield Button("Embeddings Management", id=SEARCH_NAV_EMBEDDINGS_MANAGEMENT, classes="search-nav-button")
            else:
                yield Button("Embeddings Management", id="search-nav-embeddings-management-disabled",
                             classes="search-nav-button disabled")

        with Container(id="search-content-pane", classes="search-content-pane"):
            yield Container(id=SEARCH_VIEW_RAG_QA, classes="search-view-area")
            yield Container(id=SEARCH_VIEW_RAG_CHAT, classes="search-view-area")
            yield Container(id=SEARCH_VIEW_RAG_MANAGEMENT, classes="search-view-area")

            if EMBEDDINGS_GENERATION_AVAILABLE and VECTORDB_AVAILABLE:
                with Container(id=SEARCH_VIEW_EMBEDDINGS_CREATION, classes="search-view-area"):
                    with VerticalScroll(classes="search-form-container"):
                        yield Static("Create Embeddings", classes="search-view-title")
                        yield Markdown(
                            "Select a database and configure embedding settings below. Embeddings allow semantic search and retrieval of your content.",
                            id="creation-help-text")

                        yield Static("Database Selection", classes="search-section-title")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Content Source:", classes="search-form-label")
                            yield Select(
                                [("Media DB", "media_db"), ("RAG Chat DB", "rag_chat_db"),
                                 ("Character Chat DB", "char_chat_db")],
                                id="creation-db-select",
                                prompt="Select Content Source...",
                                value="media_db"
                            )
                        with Horizontal(classes="search-form-row"):
                            yield Static("Database Path:", classes="search-form-label")
                            yield Input(id="creation-db-path-display", disabled=True, value="Path will appear here")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Collection Name:", classes="search-form-label")
                            yield Input(id="creation-collection-name-input", placeholder="ChromaDB collection name...")

                        yield Static("Content Selection", classes="search-section-title")
                        yield Markdown("Choose how you want to select content for embedding creation.",
                                       id="creation-selection-help-text")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Selection Mode:", classes="search-form-label")
                            yield Select(
                                [("All Items", "all"), ("Keyword Filter", "keyword"),
                                 ("Individual Selection", "individual")],
                                id="creation-selection-mode-select",
                                value="all"
                            )
                        with Horizontal(id="creation-keyword-filter-container", classes="search-form-row hidden"):
                            yield Static("Keyword:", classes="search-form-label")
                            yield Input(id="creation-keyword-input", placeholder="Enter keywords to filter items...")
                        with Container(id="creation-individual-selection-container", classes="hidden"):
                            with Horizontal(classes="search-form-row"):
                                yield Static("", classes="search-form-label")  # Spacer
                                yield Button("Refresh Item List", id="creation-refresh-list-button", variant="primary")
                            with Horizontal(classes="search-form-row"):
                                yield Static("Items:", classes="search-form-label")
                                yield Select([], id="creation-item-select", prompt="Select an item...")
                            yield Markdown("Select an item from the dropdown above.",
                                           id="creation-item-selection-help-text")

                        yield Static("Embedding Model", classes="search-section-title")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Provider:", classes="search-form-label")
                            yield Select([], id="creation-embedding-provider-select", prompt="Select Provider...")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Model:", classes="search-form-label")
                            yield Select([], id="creation-model-select", prompt="Select Model...")
                        with Horizontal(classes="search-form-row"):  # For "Custom" HuggingFace model path
                            yield Static("Custom HF Path:", classes="search-form-label")
                            yield Input(id="creation-custom-hf-model-path-input",
                                        placeholder="HuggingFace model name/path", classes="hidden")
                        with Horizontal(classes="search-form-row"):  # For "Local Server" API URL
                            yield Static("Local API URL:", classes="search-form-label")
                            yield Input(id="creation-local-api-url-input", placeholder="e.g., http://localhost:8080/v1",
                                        classes="hidden")

                        yield Static("Chunking Options", classes="search-section-title")
                        yield Markdown(
                            "Chunking divides your content into smaller pieces for better retrieval. Adjust these settings based on your content type.",
                            id="chunking-help-text")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Method:", classes="search-form-label")
                            yield Select([("By Words", "words"), ("By Sentences", "sentences")],
                                         id="creation-chunk-method-select", value="words")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Max Size:", classes="search-form-label")
                            yield Input(id="creation-chunk-size-input", value="400", type="integer")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Overlap:", classes="search-form-label")
                            yield Input(id="creation-chunk-overlap-input", value="200", type="integer")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Adaptive:", classes="search-form-label")
                            yield Checkbox("Use adaptive chunking (recommended)",
                                           id="creation-adaptive-chunking-checkbox", value=True)

                        yield Static("", classes="search-section-title")  # Spacer
                        yield Button("Create Embeddings", id="creation-create-embeddings-button",
                                     variant="success")  # Changed ID slightly
                        yield Markdown(
                            "Status: Ready to create embeddings. Click the button above to start the process.",
                            id="creation-status-output")
            else:  # Embeddings not available
                with Container(id=SEARCH_VIEW_EMBEDDINGS_CREATION, classes="search-view-area"):
                    with VerticalScroll():
                        yield Markdown(
                            "### Embeddings Creation Is Not Currently Available\n\nThe required dependencies for embeddings creation are not installed. Please install the necessary packages to use this feature.")

            # --- Embeddings Management View (Single Scrollable Pane) ---
            if VECTORDB_AVAILABLE:
                with Container(id=SEARCH_VIEW_EMBEDDINGS_MANAGEMENT, classes="search-view-area"):
                    with VerticalScroll(classes="search-form-container"):
                        yield Static("Manage Embeddings", classes="search-view-title")
                        yield Markdown(
                            "Select a database, collection, and item to view, or delete its embeddings.",
                            id="mgmt-help-text")

                        yield Static("Database & Collection", classes="search-section-title")
                        with Horizontal(classes="search-form-row"):
                            yield Static("DB Source:", classes="search-form-label")
                            yield Select(
                                [("Media DB", "media_db"), ("RAG Chat DB", "rag_chat_db"),
                                 ("Character Chat DB", "char_chat_db")],  # These are conceptual sources
                                id="mgmt-db-source-select",  # New ID for DB source
                                prompt="Select Content Source...",
                                value="media_db"
                            )
                        with Horizontal(classes="search-form-row"):
                            yield Static("Collection:", classes="search-form-label")
                            yield Select([], id="mgmt-collection-select",
                                         prompt="Select Collection...")  # For actual Chroma collections

                        yield Static("Item Selection", classes="search-section-title")
                        with Horizontal(classes="search-form-row"):
                            yield Static("", classes="search-form-label")  # Spacer
                            yield Button("Refresh Item List", id="mgmt-refresh-list-button", variant="primary")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Item:", classes="search-form-label")
                            yield Select([], id="mgmt-item-select",
                                         prompt="Select an item...")  # Items from the selected Chroma collection

                        yield Static("Embedding Status & Metadata", classes="search-section-title")
                        yield Markdown("Select an item above to see its embedding status and metadata.",
                                       id="mgmt-embedding-details-md")  # Combined status & metadata

                        yield Static("Actions", classes="search-section-title")
                        yield Markdown("Actions for the selected item or collection.",
                                       id="mgmt-actions-help-text")

                        yield Static("Selected Item Actions", classes="search-section-title")
                        # yield Button("Re-Embed Item (Future)", id="mgmt-reembed-item-button", variant="warning", disabled=True)
                        yield Button("Delete Item Embeddings", id="mgmt-delete-item-embeddings-button",
                                     variant="error")

                        yield Static("Selected Collection Actions", classes="search-section-title")
                        yield Button("Delete Entire Collection", id="mgmt-delete-collection-button",
                                     variant="error")

                        yield Markdown("Status: Select an item or collection to perform actions.",
                                       id="mgmt-status-output")
            else:  # VectorDB not available
                with Container(id=SEARCH_VIEW_EMBEDDINGS_MANAGEMENT, classes="search-view-area"):
                    with VerticalScroll():
                        yield Markdown(
                            "### Embeddings Management Is Not Currently Available\n\nThe required dependencies for vector database management are not installed. Please install the necessary packages to use this feature.")

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
    
    async def _append_status(self, message: str) -> None:
        """Append a message to the status output, keeping history."""
        status_output = self.query_one("#creation-status-output", Markdown)
        
        # Add message to history
        self._status_history.append(message)
        
        # Join all messages with double newlines for separation
        new_content = "\n\n".join(self._status_history)
        
        await status_output.update(new_content)

    async def _get_chroma_manager(self) -> "ChromaDBManager":
        """Get or create a ChromaDBManager instance using the app's configuration."""
        if not VECTORDB_AVAILABLE:
            raise RuntimeError("Vector-database functionality is disabled in this environment.")
        if self._chroma_manager is None:
            logger.info("ChromaDBManager instance not found, creating a new one.")
            try:
                # IMPORTANT: Ensure the app_config passed to ChromaDBManager has the
                # correctly structured "embedding_config" as expected by EmbeddingFactory.
                user_config = self.app_instance.app_config  # This should be the comprehensive config
                user_id = self.app_instance.notes_user_id
                self._chroma_manager = ChromaDBManager(user_id=user_id, user_embedding_config=user_config)
                logger.info(f"Successfully created ChromaDBManager for user '{user_id}'.")
            except Exception as e:
                logger.error(f"Failed to create ChromaDBManager: {e}", exc_info=True)
                self.app_instance.notify(f"Failed to initialize embedding system: {escape(str(e))}", severity="error",
                                         timeout=10)
                raise
        return self._chroma_manager

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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events delegated from the main app."""
        button_id = event.button.id
        if not button_id:
            return
            
        # Handle the Create Embeddings button specifically
        if button_id == "creation-create-embeddings-button":
            # Create an async task to handle the button press
            import asyncio
            asyncio.create_task(self.on_creation_create_button_pressed(event))
            event.stop()
            return
            
        # Let other events propagate to @on handlers
        # Event will be handled by the appropriate @on decorator if it matches

    @on(Button.Pressed, ".search-nav-button")
    async def handle_search_nav_button_pressed(self, event: Button.Pressed) -> None:
        """Handles all navigation button presses within the search tab."""
        event.stop()
        button_id = event.button.id
        if not button_id or "-disabled" in button_id: return

        logger.info(f"Search nav button '{button_id}' pressed.")
        target_view_id = button_id.replace("-nav-", "-view-")
        self.app_instance.search_active_sub_tab = target_view_id

        for button in self.query(".search-nav-button"):
            button.remove_class("-active-search-sub-view")
        event.button.add_class("-active-search-sub-view")

        for view in self.query(".search-view-area"):
            view.display = False

        try:
            target_view = self.query_one(f"#{target_view_id}")
            target_view.display = True
        except Exception as e:
            logger.error(f"Failed to display target view {target_view_id}: {e}")
            self.app_instance.notify(f"Error displaying view: {target_view_id}", severity="error")
            return

        try:
            if target_view_id == SEARCH_VIEW_EMBEDDINGS_CREATION:
                await self._initialize_embeddings_creation_view()
            elif target_view_id == SEARCH_VIEW_EMBEDDINGS_MANAGEMENT:
                await self._initialize_embeddings_management_view()
        except Exception as e:
            logger.error(f"Error initializing view '{target_view_id}': {e}", exc_info=True)
            self.app_instance.notify(f"Error loading view: {escape(str(e))}", severity="error")

    # --- Creation View Handlers ---
    @on(Select.Changed, "#creation-db-select")
    def on_creation_db_select_changed(self, event: Select.Changed) -> None:  # Renamed for clarity
        db_type = str(event.value)
        db_path_display = self.query_one("#creation-db-path-display", Input)
        db_path_display.value = self._get_db_path(db_type)

        collection_name_input = self.query_one("#creation-collection-name-input", Input)
        # Suggest a default collection name, e.g., user_embeddings_for_media_db
        # Allow user to override. This logic can be refined.
        sanitized_user_id = "".join(c if c.isalnum() else "_" for c in self.app_instance.notes_user_id)
        suggested_collection_name = f"embeddings_{sanitized_user_id}_{db_type}"
        if not collection_name_input.value or collection_name_input.value.startswith("embeddings_"):
            collection_name_input.value = suggested_collection_name

        # If individual selection mode is active, refresh items for the new DB
        selection_mode = str(self.query_one("#creation-selection-mode-select", Select).value)
        if selection_mode == "individual":
            asyncio.create_task(self._refresh_creation_item_list())

    @on(Select.Changed, "#creation-embedding-provider-select")
    def on_creation_provider_select_changed(self, event: Select.Changed) -> None:
        provider_internal_id = str(event.value)
        logger.debug(f"Creation provider select changed to: {provider_internal_id}")
        self._update_creation_model_select_for_provider(provider_internal_id)
        # Visibility of URL/Path inputs will be handled by the model select change triggered above

    @on(Select.Changed, "#creation-model-select")
    def on_creation_model_select_changed(self, event: Select.Changed) -> None:
        selected_model_key = str(event.value)  # This is the key from config or "custom_hf_path"
        provider_internal_id = str(self.query_one("#creation-embedding-provider-select", Select).value)
        logger.debug(f"Creation model select changed to: {selected_model_key} for provider: {provider_internal_id}")
        self._update_creation_provider_specific_inputs_visibility(provider_internal_id, selected_model_key)

    def _update_creation_provider_specific_inputs_visibility(self, provider_internal_id: str,
                                                             selected_model_key: Optional[str] = None):
        """Shows/hides custom HuggingFace path input and local API URL input (now read-only display)."""
        custom_hf_path_input = self.query_one("#creation-custom-hf-model-path-input", Input)
        local_api_url_input = self.query_one("#creation-local-api-url-input", Input)

        # Custom HF Path input
        custom_hf_path_input.display = (provider_internal_id == "huggingface" and
                                        selected_model_key == "custom_hf_path")

        # Local API URL input (now a display for the configured URL)
        if provider_internal_id == LOCAL_SERVER_PROVIDER_INTERNAL_ID and selected_model_key and selected_model_key != Select.BLANK:
            app_cfg = self.app_instance.app_config
            embedding_models_config: Dict[str, ModelCfg] = app_cfg.get("embedding_config", {}).get("models",
                                                                                                   {})  # type: ignore
            model_data = embedding_models_config.get(selected_model_key)

            if model_data and model_data.get("base_url"):
                local_api_url_input.value = str(model_data.get("base_url"))  # Display the configured base_url
                local_api_url_input.disabled = True  # Make it read-only
                local_api_url_input.display = True
            else:
                local_api_url_input.value = "Base URL not configured for this model."
                local_api_url_input.disabled = True
                local_api_url_input.display = True  # Still show it, but with a message
        else:
            local_api_url_input.display = False
            local_api_url_input.disabled = False  # Reset disabled state when hidden
            local_api_url_input.value = ""  # Clear it

        logger.debug(
            f"Provider specific inputs visibility updated. Custom HF: {custom_hf_path_input.display}, Local API URL: {local_api_url_input.display} (Value: '{local_api_url_input.value}', Disabled: {local_api_url_input.disabled})")

    @on(Select.Changed, "#creation-selection-mode-select")
    def on_creation_selection_mode_changed(self, event: Select.Changed) -> None:  # Renamed for clarity
        mode = str(event.value)
        keyword_container = self.query_one("#creation-keyword-filter-container")
        individual_container = self.query_one("#creation-individual-selection-container")

        keyword_container.display = mode == "keyword"
        individual_container.display = mode == "individual"

        if mode == "individual":
            item_select = self.query_one("#creation-item-select", Select)
            if item_select.value is Select.BLANK or not item_select._options:  # Check if options are empty
                asyncio.create_task(self._refresh_creation_item_list())

    @on(Button.Pressed, "#creation-create-embeddings-button")  # Matched new ID
    async def on_creation_create_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the Create Embeddings button press."""
        # Disable the button immediately to prevent multiple clicks
        create_button = self.query_one("#creation-create-embeddings-button", Button)
        create_button.disabled = True
        
        try:
            await self._handle_create_embeddings_process()
        finally:
            # Re-enable the button when done (whether successful or failed)
            create_button.disabled = False

    async def _handle_create_embeddings_process(self) -> None:
        """Handle the actual embedding creation process."""
        # Clear status history for a fresh start
        self._status_history = []
        
        db_type = str(self.query_one("#creation-db-select", Select).value)
        db_path = self._get_db_path(db_type)  # This remains a conceptual path for now

        collection_name_input = self.query_one("#creation-collection-name-input", Input)
        collection_name = collection_name_input.value.strip()
        if not collection_name:
            await self._append_status("❌ Collection name cannot be empty.")
            collection_name_input.focus()
            return

        selection_mode = str(self.query_one("#creation-selection-mode-select", Select).value)
        keyword: Optional[str] = None
        selected_item_db_ids: Optional[List[str]] = None # Actual DB IDs

        if selection_mode == "keyword":
            keyword = self.query_one("#creation-keyword-input", Input).value.strip()
            if not keyword:
                await self._append_status("❌ Keyword cannot be empty for keyword filter mode.")
                self.query_one("#creation-keyword-input", Input).focus()
                return
        elif selection_mode == "individual":
            selected_display_name = str(self.query_one("#creation-item-select", Select).value)
            if not selected_display_name or selected_display_name == Select.BLANK:
                await self._append_status("❌ Please select an item for individual mode.")
                self.query_one("#creation-item-select", Select).focus()
                return
            # Map display name back to actual item ID
            item_id = self._creation_item_mapping.get(selected_display_name)
            if not item_id:
                await self._append_status("❌ Error resolving selected item ID.")
                return
            selected_item_db_ids = [item_id]

        # --- Determine embedding_model_id_override for ChromaDBManager ---
        provider_internal_id = str(self.query_one("#creation-embedding-provider-select", Select).value)
        # `selected_model_config_key` is the key from app_config or "custom_hf_path"
        selected_model_config_key = str(self.query_one("#creation-model-select", Select).value)

        embedding_model_id_override: Optional[str] = None

        if provider_internal_id == "huggingface":
            if selected_model_config_key == "custom_hf_path":
                custom_hf_path = self.query_one("#creation-custom-hf-model-path-input", Input).value.strip()
                if not custom_hf_path:
                    await self.query_one("#creation-status-output", Markdown).update("❌ Custom HuggingFace model path cannot be empty.")
                    self.query_one("#creation-custom-hf-model-path-input", Input).focus()
                    return
                embedding_model_id_override = custom_hf_path # Factory will treat this as model_name_or_path
            else:
                # This IS the key from settings.toml (e.g., "e5-small-v2")
                embedding_model_id_override = selected_model_config_key

        elif provider_internal_id == "openai_official" or provider_internal_id == LOCAL_SERVER_PROVIDER_INTERNAL_ID:
            # For both official OpenAI and Local OpenAI-compliant, the `selected_model_config_key`
            # IS the key from settings.toml that `EmbeddingFactory` will use.
            # The factory will correctly pick up `base_url` from the config if it's a local model.
            embedding_model_id_override = selected_model_config_key

        else: # Should not happen if UI is populated correctly
            await self.query_one("#creation-status-output", Markdown).update(f"❌ Unknown provider selected: {provider_internal_id}")
            return


        if not embedding_model_id_override or embedding_model_id_override == Select.BLANK:
            await self.query_one("#creation-status-output", Markdown).update("❌ Could not determine embedding model ID. Please select a model.")
            self.query_one("#creation-model-select", Select).focus()
            return

        # Get chunking options
        chunk_method = str(self.query_one("#creation-chunk-method-select", Select).value)
        chunk_size_str = self.query_one("#creation-chunk-size-input", Input).value
        chunk_overlap_str = self.query_one("#creation-chunk-overlap-input", Input).value
        adaptive_chunking = self.query_one("#creation-adaptive-chunking-checkbox", Checkbox).value

        try:
            chunk_size = int(chunk_size_str)
            chunk_overlap = int(chunk_overlap_str)
            if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
                raise ValueError("Invalid chunk size or overlap.")
        except ValueError:
            await self.query_one("#creation-status-output", Markdown).update("❌ Invalid chunk size or overlap values.")
            return

        chunk_options = {
            "method": chunk_method,
            "size": chunk_size,
            "overlap": chunk_overlap,
            "adaptive": adaptive_chunking
        }

        status_output = self.query_one("#creation-status-output", Markdown)
        status_msg_prefix = ""
        if selection_mode == "all": status_msg_prefix = f"all items in {self.DB_DISPLAY_NAMES.get(db_type, db_type)}"
        elif selection_mode == "keyword": status_msg_prefix = f"items matching '{keyword}' in {self.DB_DISPLAY_NAMES.get(db_type, db_type)}"
        elif selected_item_db_ids: status_msg_prefix = f"{len(selected_item_db_ids)} selected item(s) in {self.DB_DISPLAY_NAMES.get(db_type, db_type)}"
        else: status_msg_prefix = "items (unexpected selection)"


        await self._append_status(f"⏳ Preparing to create embeddings for {status_msg_prefix} into collection '{collection_name}' using model ID '{embedding_model_id_override}'. This may take some time...")

        try:
            await self._create_embeddings( # Pass db_path as None or remove if not used by _load_items
                db_type=db_type,
                db_path="", # Conceptual, _load_items_for_embedding uses app_instance.db
                collection_name=collection_name,
                embedding_model_id_override=embedding_model_id_override,
                chunk_options=chunk_options,
                selection_mode=selection_mode,
                keyword=keyword,
                selected_item_ids=selected_item_db_ids # Pass actual DB item IDs
            )
            await self._append_status(f"✅ Successfully created embeddings for {status_msg_prefix} in collection '{collection_name}'.")
            if self.app_instance.search_active_sub_tab == SEARCH_VIEW_EMBEDDINGS_MANAGEMENT:
                 await self._refresh_mgmt_collections_list()

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}", exc_info=True)
            await self._append_status(f"❌ Error creating embeddings: {escape(str(e))}")
            self.app_instance.notify(f"Error creating embeddings: {escape(str(e))}", severity="error", timeout=10)

    # --- Management View Handlers ---
    @on(Select.Changed, "#mgmt-db-source-select")
    async def on_mgmt_db_source_select_changed(self, event: Select.Changed) -> None:
        # When conceptual DB source changes, refresh the list of actual Chroma collections.
        # The db_type from here might be used to *suggest* or *filter* collections if you adopt a naming convention.
        await self._refresh_mgmt_collections_list()

    @on(Select.Changed, "#mgmt-collection-select")
    async def on_mgmt_collection_select_changed(self, event: Select.Changed) -> None:
        # When a Chroma collection is selected, refresh the item list *from that collection*.
        self._selected_chroma_collection_name = str(event.value) if event.value != Select.BLANK else None
        await self._refresh_mgmt_item_list()  # This will now use the selected collection

    @on(Button.Pressed, "#mgmt-refresh-list-button")
    async def on_mgmt_refresh_item_list_button_pressed(self, event: Button.Pressed) -> None:  # Renamed for clarity
        # This button now specifically refreshes items within the selected collection.
        # Collection list refresh is triggered by DB source change.
        if not self._selected_chroma_collection_name:
            self.app_instance.notify("Please select a collection first to refresh its items.", severity="warning")
            return
        await self._refresh_mgmt_item_list()

    @on(Button.Pressed, "#creation-refresh-list-button")
    async def on_creation_refresh_item_list_button_pressed(self, event: Button.Pressed) -> None:  # Renamed for clarity
        await self._refresh_creation_item_list()

    @on(Select.Changed, "#mgmt-item-select")
    async def on_mgmt_item_select_changed(self, event: Select.Changed) -> None:  # Renamed for clarity
        if event.value is Select.BLANK:
            self._selected_item_display_name = None
            self._selected_embedding_collection_item_id = None  # Actual ID of item in Chroma (e.g., media_1_chunk_0)
            await self.query_one("#mgmt-embedding-details-md", Markdown).update("Select an item to see its details.")
        else:
            self._selected_item_display_name = str(event.value)  # This is "Title (media_id_chunk_idx)"
            # `_mgmt_item_mapping` should now map this display name to the actual Chroma ID
            self._selected_embedding_collection_item_id = self._mgmt_item_mapping.get(self._selected_item_display_name)
            await self._check_and_display_embedding_status()

    @on(Button.Pressed, "#mgmt-delete-item-embeddings-button")
    async def on_mgmt_delete_item_embeddings_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._selected_chroma_collection_name:
            self.app_instance.notify("No collection selected.", severity="error")
            return
        if not self._selected_embedding_collection_item_id:  # This is the Chroma ID like "media_1_chunk_0"
            self.app_instance.notify("No item selected from the collection to delete.", severity="warning")
            return

        status_output = self.query_one("#mgmt-status-output", Markdown)
        await status_output.update(
            f"⏳ Deleting item '{self._selected_embedding_collection_item_id}' from collection '{self._selected_chroma_collection_name}'...")
        try:
            chroma_manager = await self._get_chroma_manager()
            # We need to delete ALL chunks associated with the original media_id if that's the goal.
            # The current `_selected_embedding_collection_item_id` is likely one chunk.
            # For simplicity now, let's assume we delete the specific chunk ID shown.
            # A more robust delete would query all IDs with a common original_media_id.

            # Example: If _selected_embedding_collection_item_id is "originalMediaID_chunk_0"
            # You might want to delete all "originalMediaID_chunk_*"
            # For now, deleting the specific ID:
            chroma_manager.delete_from_collection(
                ids=[self._selected_embedding_collection_item_id],
                collection_name=self._selected_chroma_collection_name
            )
            await status_output.update(
                f"✅ Item '{self._selected_embedding_collection_item_id}' deleted from '{self._selected_chroma_collection_name}'.")
            logger.info(
                f"Deleted item '{self._selected_embedding_collection_item_id}' from collection '{self._selected_chroma_collection_name}'.")
            await self._refresh_mgmt_item_list()  # Refresh items
            await self.query_one("#mgmt-embedding-details-md", Markdown).update("Item deleted. Select another item.")
        except Exception as e:
            logger.error(f"Error deleting item embedding: {e}", exc_info=True)
            await status_output.update(f"❌ Error deleting item: {escape(str(e))}")

    @on(Button.Pressed, "#mgmt-delete-collection-button")
    async def on_mgmt_delete_collection_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._selected_chroma_collection_name:
            self.app_instance.notify("No collection selected to delete.", severity="error")
            return

        # Add a confirmation step here in a real app!
        status_output = self.query_one("#mgmt-status-output", Markdown)
        await status_output.update(f"⏳ Deleting collection '{self._selected_chroma_collection_name}'...")
        try:
            chroma_manager = await self._get_chroma_manager()
            chroma_manager.delete_collection(collection_name=self._selected_chroma_collection_name)
            await status_output.update(f"✅ Collection '{self._selected_chroma_collection_name}' deleted.")
            logger.info(f"Deleted collection '{self._selected_chroma_collection_name}'.")
            self._selected_chroma_collection_name = None
            self.query_one("#mgmt-item-select", Select).set_options([])  # Clear item list
            await self.query_one("#mgmt-embedding-details-md", Markdown).update(
                "Collection deleted. Select another collection.")
            await self._refresh_mgmt_collections_list()  # Refresh collection list
        except Exception as e:
            logger.error(f"Error deleting collection: {e}", exc_info=True)
            await status_output.update(f"❌ Error deleting collection: {escape(str(e))}")

    # --- INITIALIZATION HELPERS ---

    async def _initialize_embeddings_creation_view(self) -> None:
        """Populates dropdowns and sets defaults for the Creation view."""
        logger.info("Initializing Embeddings Creation view.")

        db_select = self.query_one("#creation-db-select", Select)
        # Trigger its changed event handler to set dependent fields
        self.on_creation_db_select_changed(Select.Changed(db_select, str(db_select.value)))  # type: ignore

        try:
            app_cfg = self.app_instance.app_config
            embedding_models_config = app_cfg.get("embedding_config", {}).get("models", {})

            config_providers = sorted(list(set(
                model_data.get("provider")
                for model_data in embedding_models_config.values()
                if model_data.get("provider")
            )))

            provider_options: List[Tuple[str, str]] = []
            if "huggingface" in config_providers:
                provider_options.append(("HuggingFace", "huggingface"))
            # "openai" provider can now serve both official OpenAI and local OpenAI-compliant servers
            # if configured with a base_url.
            if "openai" in config_providers:
                # Check if there are any "openai" provider models configured for local use (i.e., with a base_url)
                has_local_openai_models = any(
                    model_data.get("provider") == "openai" and model_data.get("base_url")
                    for model_data in embedding_models_config.values()
                )
                # Check if there are any "openai" provider models for official API (i.e., without a base_url)
                has_official_openai_models = any(
                    model_data.get("provider") == "openai" and not model_data.get("base_url")
                    for model_data in embedding_models_config.values()
                )

                if has_official_openai_models:
                    provider_options.append(("OpenAI API (Official)", "openai_official"))  # New internal ID
                if has_local_openai_models:
                    provider_options.append((LOCAL_SERVER_PROVIDER_DISPLAY_NAME, LOCAL_SERVER_PROVIDER_INTERNAL_ID))

            provider_select = self.query_one("#creation-embedding-provider-select", Select)
            if provider_options:
                provider_select.set_options(provider_options)
                default_provider_internal_id = provider_options[0][1]
                provider_select.value = default_provider_internal_id
                # This will now call the updated _update_creation_model_select_for_provider
                self.on_creation_provider_select_changed(
                    Select.Changed(provider_select, default_provider_internal_id))  # type: ignore
            else:
                provider_select.set_options([("No Providers Configured", Select.BLANK)])
                self.query_one("#creation-model-select", Select).set_options([])
                self.query_one("#creation-model-select", Select).prompt = "Configure Providers First"
                self._update_creation_provider_specific_inputs_visibility(Select.BLANK)  # type: ignore

        except Exception as e:
            logger.error(f"Error initializing embedding providers/models in SearchWindow (Creation): {e}",
                         exc_info=True)
            self.app_instance.notify("Error loading embedding model configs.", severity="error")
            self.query_one("#creation-embedding-provider-select", Select).set_options(
                [("Error Loading", Select.BLANK)])  # type: ignore
            self.query_one("#creation-model-select", Select).set_options([])
            self._update_creation_provider_specific_inputs_visibility(Select.BLANK)  # type: ignore

        status_output = self.query_one("#creation-status-output", Markdown)
        await status_output.update("Ready to create embeddings. Configure your settings and click the button above.")

    def _update_creation_model_select_for_provider(self, provider_internal_id: str) -> None:
        """Populates the model select dropdown based on the chosen provider for the Creation view."""
        model_select = self.query_one("#creation-model-select", Select)

        app_cfg = self.app_instance.app_config
        embedding_models_config: Dict[str, ModelCfg] = app_cfg.get("embedding_config", {}).get("models",
                                                                                               {})  # type: ignore

        model_options: List[Tuple[str, str]] = []

        if provider_internal_id == "huggingface":
            for model_key, model_data in embedding_models_config.items():
                if model_data.get("provider") == "huggingface":
                    display_name = f"{model_key} ({model_data.get('model_name_or_path', 'N/A')})"
                    model_options.append((display_name, model_key))  # Value is the config key
            model_options.append(("Custom HuggingFace Path", "custom_hf_path"))

        elif provider_internal_id == "openai_official":  # For official OpenAI API
            for model_key, model_data in embedding_models_config.items():
                # Official OpenAI models have provider "openai" and NO base_url
                if model_data.get("provider") == "openai" and not model_data.get("base_url"):
                    display_name = f"{model_key} ({model_data.get('model_name_or_path', 'N/A')})"
                    model_options.append((display_name, model_key))  # Value is the config key

        elif provider_internal_id == LOCAL_SERVER_PROVIDER_INTERNAL_ID:
            for model_key, model_data in embedding_models_config.items():
                if model_data.get("provider") == "openai" and model_data.get("base_url"):
                    # Use model_name_or_path for display if available, else model_key
                    name_part = model_data.get('model_name_or_path', model_key)
                    url_part = str(model_data.get('base_url'))
                    display_name = f"{name_part} (Local @ {url_part})"  # More informative
                    model_options.append((display_name, model_key))

        if model_options:
            model_select.set_options(model_options)
            model_select.value = model_options[0][1]
            model_select.prompt = "Select Model..."
            # Trigger model changed to update visibility of API URL field
            self.on_creation_model_select_changed(Select.Changed(model_select, model_select.value))  # type: ignore
        else:
            model_select.set_options([])
            model_select.value = Select.BLANK  # type: ignore
            if provider_internal_id and provider_internal_id != Select.BLANK:
                model_select.prompt = f"No '{provider_internal_id}' models in config"
            else:
                model_select.prompt = "Select Provider First"
            # Ensure URL field is hidden if no models
            self._update_creation_provider_specific_inputs_visibility(provider_internal_id, None)

    async def _initialize_embeddings_management_view(self) -> None:
        """Populates dropdowns and sets defaults for the Management view."""
        logger.info("Initializing Embeddings Management view.")
        # DB Source Select is static.
        # Collection Select needs to be populated based on ChromaDB.
        await self._refresh_mgmt_collections_list()
        # Item select will be populated when a collection is chosen.
        self.query_one("#mgmt-item-select", Select).set_options([])
        self.query_one("#mgmt-item-select", Select).prompt = "Select Collection First"
        await self.query_one("#mgmt-embedding-details-md", Markdown).update("Select a database source and collection.")
        await self.query_one("#mgmt-status-output", Markdown).update("Ready for management tasks.")

    async def _refresh_collections_list(self) -> None:
        """Alias for _refresh_mgmt_collections_list for compatibility with main app."""
        await self._refresh_mgmt_collections_list()

    async def _refresh_mgmt_collections_list(self) -> None:
        """Refreshes the list of ChromaDB collections in the management view."""
        collection_select = self.query_one("#mgmt-collection-select", Select)
        status_output = self.query_one("#mgmt-status-output", Markdown)

        await status_output.update("⏳ Loading collections from ChromaDB...")
        try:
            chroma_manager = await self._get_chroma_manager()
            collections = chroma_manager.list_collections()  # Returns list of Collection objects

            collection_options: List[Tuple[str, str]] = []
            if collections:
                for coll in collections:
                    # Display name, value. Here, name is fine for both.
                    collection_options.append((coll.name, coll.name))

            if collection_options:
                collection_select.set_options(collection_options)
                collection_select.prompt = "Select Collection..."
                # Optionally select the first one, or leave it blank
                # collection_select.value = collection_options[0][1]
                # self.on_mgmt_collection_select_changed(Select.Changed(collection_select, collection_select.value))
                await status_output.update(f"✅ Found {len(collections)} collections. Select one.")
            else:
                collection_select.set_options([])
                collection_select.prompt = "No collections found"
                await status_output.update("ℹ️ No collections found in ChromaDB for this user.")

            # Clear dependent item list and details
            self.query_one("#mgmt-item-select", Select).set_options([])
            self.query_one("#mgmt-item-select", Select).prompt = "Select Collection First"
            await self.query_one("#mgmt-embedding-details-md", Markdown).update("Select a collection to see its items.")

        except Exception as e:
            logger.error(f"Error refreshing ChromaDB collections list: {e}", exc_info=True)
            collection_select.set_options([])
            collection_select.prompt = "Error loading collections"
            await status_output.update(f"❌ Error loading collections: {escape(str(e))}")

    async def _refresh_creation_item_list(self) -> None:
        """Fetches items for the selected DB and populates the creation item selection dropdown."""
        item_select = self.query_one("#creation-item-select", Select)
        db_type = str(self.query_one("#creation-db-select", Select).value)
        status_output = self.query_one("#creation-status-output", Markdown)
        db_display_name = self.DB_DISPLAY_NAMES.get(db_type, db_type)

        await status_output.update(f"⏳ Loading items from {db_display_name} for selection...")
        logger.info(f"Refreshing creation item list for DB type: {db_type}")

        loaded_db_items: List[Dict[str, Any]] = []
        load_error_occurred = False

        if db_type == "media_db":
            if not self.app_instance.media_db:
                logger.error("MediaDatabase instance not available in app for creation list.")
                await status_output.update("❌ Media DB not initialized.")
                load_error_occurred = True
            else:
                media_db: MediaDatabase = self.app_instance.media_db
                try:
                    raw_items = await asyncio.to_thread(
                        media_db.get_all_active_media_for_selection_dropdown,
                        limit=200  # Adjust limit as needed
                    )
                    loaded_db_items = [{"id": str(item['id']), "title": item.get("title", f"Media {item['id']}")} for item in raw_items]
                except MediaDatabaseError as e:
                    logger.error(f"MediaDB error refreshing creation item list: {e}", exc_info=True)
                    await status_output.update(f"❌ MediaDB error: {escape(str(e))}")
                    load_error_occurred = True
                except Exception as e:
                    logger.error(f"Unexpected error refreshing media creation item list: {e}", exc_info=True)
                    await status_output.update(f"❌ Error loading media items: {escape(str(e))}")
                    load_error_occurred = True

        elif db_type == "char_chat_db":  # Notes
            if not self.app_instance.notes_service:
                logger.error("NotesService instance not available for creation list.")
                await status_output.update("❌ Notes service not initialized.")
                load_error_occurred = True
            else:
                notes_service = self.app_instance.notes_service
                user_id = self.app_instance.notes_user_id
                try:
                    # Fetch recent notes for the dropdown
                    raw_items = await asyncio.to_thread(notes_service.list_notes, user_id=user_id, limit=200)
                    # Notes already have 'id' (UUID string) and 'title'
                    loaded_db_items = [{"id": item['id'], "title": item.get("title", f"Note {item['id'][:8]}")} for item in raw_items]
                except CharactersRAGDBError as e: # Catch specific DB error
                    logger.error(f"ChaChaNotesDB error refreshing notes creation item list: {e}", exc_info=True)
                    await status_output.update(f"❌ Notes DB error: {escape(str(e))}")
                    load_error_occurred = True
                except Exception as e:
                    logger.error(f"Unexpected error refreshing notes creation item list: {e}", exc_info=True)
                    await status_output.update(f"❌ Error loading notes: {escape(str(e))}")
                    load_error_occurred = True

        elif db_type == "rag_chat_db":  # Conversations
            if not self.app_instance.chachanotes_db:
                logger.error("ChaChaNotes_DB instance not available for creation list.")
                await status_output.update("❌ Chat DB not initialized.")
                load_error_occurred = True
            else:
                chat_db: CharactersRAGDB = self.app_instance.chachanotes_db
                try:
                    # Use the new method to list conversations
                    raw_items = await asyncio.to_thread(chat_db.list_all_active_conversations, limit=200)
                    # Conversations have 'id' (UUID string) and 'title'
                    loaded_db_items = [{"id": item['id'], "title": item.get("title", f"Conversation {item['id'][:8]}")} for item in raw_items]
                except CharactersRAGDBError as e: # Catch specific DB error
                    logger.error(f"ChaChaNotesDB error refreshing convos creation item list: {e}", exc_info=True)
                    await status_output.update(f"❌ Chat DB error: {escape(str(e))}")
                    load_error_occurred = True
                except Exception as e:
                    logger.error(f"Unexpected error refreshing convos creation item list: {e}", exc_info=True)
                    await status_output.update(f"❌ Error loading conversations: {escape(str(e))}")
                    load_error_occurred = True
        else:
            await status_output.update(f"⚠️ Unknown database type '{db_type}' for item listing.")
            item_select.set_options([])
            item_select.prompt = "Select DB type"
            self._creation_item_mapping = {}
            return

        if load_error_occurred:
            item_select.set_options([])
            item_select.prompt = "Error loading items"
            self._creation_item_mapping = {}
            return

        new_mapping: Dict[str, str] = {} # Maps display_name to actual_db_id (string)
        choices: List[Tuple[str, str]] = [] # List of (display_name, select_value)

        if loaded_db_items:
            for item in loaded_db_items:
                item_id_str = str(item.get('id', '')) # Ensure ID is string for consistency
                item_title = str(item.get('title', 'Untitled Item'))
                if not item_id_str:
                    logger.warning(f"Item from {db_type} missing ID: {item}")
                    continue

                display_name = f"{item_title} (ID: {item_id_str[:8]}...)"

                # Ensure display_name is unique for the dropdown choices if titles could be identical
                original_display_name = display_name
                count = 1
                while display_name in new_mapping: # Check against keys in new_mapping
                    display_name = f"{original_display_name} ({count})"
                    count += 1

                choices.append((display_name, display_name))
                new_mapping[display_name] = item_id_str

            self._creation_item_mapping = new_mapping
            item_select.set_options(choices)
            item_select.prompt = "Select an item..."
            await status_output.update(f"✅ Found {len(loaded_db_items)} items in {db_display_name}. Select one for individual embedding.")
        else:
            self._creation_item_mapping = {}
            item_select.set_options([])
            item_select.prompt = f"No items in {db_display_name}"
            await status_output.update(f"ℹ️ No items found in {db_display_name} to list for individual selection.")

    async def _refresh_mgmt_item_list(self) -> None:
        """Fetches items for the selected ChromaDB collection and populates the management dropdown."""
        item_select = self.query_one("#mgmt-item-select", Select)
        status_md = self.query_one("#mgmt-embedding-details-md", Markdown)  # Combined details display
        mgmt_status_output = self.query_one("#mgmt-status-output", Markdown)

        if not self._selected_chroma_collection_name:
            item_select.set_options([])
            item_select.prompt = "Select Collection First"
            await status_md.update("Select a collection to see its items.")
            return

        await status_md.update(
            f"⏳ Refreshing items from collection '{self._selected_chroma_collection_name}', please wait...")
        await mgmt_status_output.update(f"⏳ Loading items from '{self._selected_chroma_collection_name}'...")

        try:
            chroma_manager = await self._get_chroma_manager()
            # Fetch ALL items from the collection to list them.
            # Chroma's get() with no IDs and include=["metadatas"] can fetch all.
            # This might be very large for big collections. Consider pagination or filtering if needed.
            collection = chroma_manager.client.get_collection(name=self._selected_chroma_collection_name)
            # Fetch a limited number of items, e.g., first 1000, for display purposes.
            # The `get` method fetches by IDs. To list content, `peek` or a limited `get` with all known IDs (if feasible)
            # or a query with no filter returning all metadatas might be needed.
            # A simple way for a small/medium collection:
            results = collection.get(limit=1000, include=["metadatas"])  # Get up to 1000 items

            choices: List[Tuple[str, str]] = []
            new_mapping: Dict[str, str] = {}  # Maps display name to Chroma ID (e.g. media_1_chunk_0)

            if results and results['ids']:
                for i, chroma_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else {}
                    # Try to create a user-friendly display name
                    # Example: "My Video Title (Chunk 0 of original_media_id)"
                    file_name = metadata.get("file_name", "Unknown File")
                    media_id = metadata.get("media_id", "UnknownMedia")  # Original media ID
                    chunk_idx = metadata.get("chunk_index", "N/A")

                    # Display name could be just the chroma_id if metadata is sparse
                    # display_name = f"{file_name} - Chunk {chunk_idx} (ID: {chroma_id})"
                    # More simply, use a reference from metadata if available, else chroma_id
                    title_ref = metadata.get("original_chunk_text_ref", chroma_id)
                    title_ref = title_ref[:50] + "..." if len(title_ref) > 50 else title_ref
                    display_name = f"{title_ref} (media: {media_id}, chunk: {chunk_idx})"

                    choices.append((display_name, display_name))  # Use display_name as value for Select
                    new_mapping[display_name] = chroma_id  # Map display_name to Chroma ID

            self._mgmt_item_mapping = new_mapping
            if choices:
                item_select.set_options(choices)
                item_select.prompt = "Select an item..."
                await status_md.update(
                    f"✅ Found {len(choices)} items in '{self._selected_chroma_collection_name}'. Select one.")
                await mgmt_status_output.update(
                    f"Ready. {len(choices)} items loaded from '{self._selected_chroma_collection_name}'.")
            else:
                item_select.set_options([])
                item_select.prompt = "No items in collection"
                await status_md.update(f"ℹ️ No items found in collection '{self._selected_chroma_collection_name}'.")
                await mgmt_status_output.update(f"No items found in '{self._selected_chroma_collection_name}'.")

        except Exception as e:
            logger.error(f"Error refreshing item list from Chroma collection: {e}", exc_info=True)
            item_select.set_options([])
            item_select.prompt = "Error loading items"
            await status_md.update(f"❌ Error loading items: {escape(str(e))}")
            await mgmt_status_output.update(
                f"Error: Failed to load items from '{self._selected_chroma_collection_name}'. See logs.")

    async def _load_items_for_embedding(self,
                                        db_type: str,
                                        selection_mode: str,
                                        keyword: Optional[str],
                                        selected_item_db_ids: Optional[List[str]]
                                        ) -> List[Dict[str, Any]]:
        """
        Loads items from the specified database based on selection criteria.
        Returns a list of dicts, each with 'id' (original DB item ID as string),
        'content', and 'filename'.
        """
        items_to_embed: List[Dict[str, Any]] = []
        logger.info(
            f"Loading items for embedding. DB Type: '{db_type}', Mode: '{selection_mode}', Keyword: '{keyword}', Selected IDs: {selected_item_db_ids}"
        )

        try:
            status_output = self.query_one("#creation-status-output", Markdown)
        except QueryError:
            logger.error("Could not find #creation-status-output widget. Status updates will be missing.")
            # Non-critical, proceed with logging, but notify the user if possible.
            self.app_instance.notify("Error: Status display widget not found.", severity="error")
            status_output = None  # So we can check for it before updating

        async def update_status(message: str):
            if status_output:
                await status_output.update(message)
            logger.info(message)

        if db_type == "media_db":
            if not self.app_instance.media_db:
                logger.error("MediaDatabase instance (self.app_instance.media_db) is not available.")
                await update_status("❌ Error: Media DB not initialized. Cannot load media items.")
                return []

            media_db_instance: MediaDatabase = self.app_instance.media_db
            raw_media_items: List[Dict[str, Any]] = []

            try:
                if selection_mode == "all":
                    await update_status(
                        f"⏳ Loading ALL items from Media DB ({self.DB_DISPLAY_NAMES.get(db_type, db_type)})...")
                    # Use get_all_active_media_for_embedding to get items with content
                    raw_media_items = await asyncio.to_thread(
                        media_db_instance.get_all_active_media_for_embedding,
                        limit=10000  # Large number to fetch "all"
                    )


                elif selection_mode == "individual" and selected_item_db_ids:
                    await update_status(f"⏳ Loading {len(selected_item_db_ids)} selected item(s) from Media DB...")
                    valid_ids_to_fetch: List[int] = []
                    for item_id_str in selected_item_db_ids:
                        try:
                            valid_ids_to_fetch.append(int(item_id_str))
                        except ValueError:
                            logger.warning(f"Invalid media item ID format: '{item_id_str}'. Skipping.")

                    if valid_ids_to_fetch:
                        # Assuming a method like 'get_media_by_ids_for_embedding' exists.
                        # It should take a list of integer IDs and return a list of media item dicts.
                        # For now, fetching one by one if a batch method isn't available.
                        for mid in valid_ids_to_fetch:
                            item = await asyncio.to_thread(media_db_instance.get_media_by_id, mid)
                            if item:
                                raw_media_items.append(item)
                    else:
                        logger.warning("No valid numeric IDs provided for individual media item selection.")

                elif selection_mode == "keyword" and keyword:
                    await update_status(f"⏳ Searching Media DB for items matching keyword '{keyword}'...")
                    # Assuming 'search_media_by_keyword_for_embedding' exists, or adapting search_media_db.
                    # This method should return media items whose content or title matches the keyword.
                    raw_media_items_tuple = await asyncio.to_thread(
                        media_db_instance.search_media_db,
                        search_query=keyword,
                        media_types=None,  # Search all types
                        search_fields=['title', 'content'],  # Fields to search within
                        sort_by="last_modified_desc",
                        page=1,
                        results_per_page=1000,  # Limit results for keyword search
                        include_trash=False,
                        include_deleted=False
                    )
                    raw_media_items = raw_media_items_tuple[0]

                for item in raw_media_items:
                    item_id = item.get("id")
                    item_content = item.get("content")
                    if item_id is not None and item_content:  # Ensure both ID and content are present
                        items_to_embed.append({
                            "id": str(item_id),  # Ensure ID is string
                            "text": item_content,  # Changed from "content" to "text" for ChromaDBManager
                            "filename": item.get("title", f"media_item_{item_id}")
                        })
                    else:
                        logger.warning(
                            f"Skipping media item due to missing ID or content: {item.get('id', 'Unknown ID')}")

            except MediaDatabaseError as e:
                logger.error(f"MediaDatabaseError while loading media items: {e}", exc_info=True)
                await update_status(f"❌ Database Error: Failed to load media items. {escape(str(e))}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error loading media items: {e}", exc_info=True)
                await update_status(f"❌ Error: Failed to load media items. {escape(str(e))}")
                return []

        elif db_type == "char_chat_db":  # Notes from ChaChaNotes_DB via NotesInteropService
            if not self.app_instance.notes_service:
                logger.error("NotesService (self.app_instance.notes_service) is not available.")
                await update_status("❌ Error: Notes service not initialized. Cannot load notes.")
                return []

            notes_service: NotesInteropService = self.app_instance.notes_service
            user_id = self.app_instance.notes_user_id
            raw_notes_data: List[Dict[str, Any]] = []

            try:
                if selection_mode == "all":
                    await update_status(
                        f"⏳ Loading ALL notes for user '{user_id}' ({self.DB_DISPLAY_NAMES.get(db_type, db_type)})...")
                    raw_notes_data = await asyncio.to_thread(notes_service.list_notes, user_id=user_id,
                                                             limit=10000)  # High limit for "all"

                elif selection_mode == "individual" and selected_item_db_ids:
                    await update_status(f"⏳ Loading {len(selected_item_db_ids)} selected note(s)...")
                    for note_id_str in selected_item_db_ids:  # note_id_str is the UUID from ChaChaNotes
                        note = await asyncio.to_thread(notes_service.get_note_by_id, user_id=user_id,
                                                       note_id=note_id_str)
                        if note:
                            raw_notes_data.append(note)
                        else:
                            logger.warning(f"Note with ID '{note_id_str}' not found for user '{user_id}'.")

                elif selection_mode == "keyword" and keyword:
                    await update_status(f"⏳ Searching notes with keyword '{keyword}' for user '{user_id}'...")
                    raw_notes_data = await asyncio.to_thread(notes_service.search_notes, user_id=user_id,
                                                             search_term=keyword,
                                                             limit=1000)  # Limit for keyword search

                for note_item in raw_notes_data:
                    note_id = note_item.get("id")
                    note_content = note_item.get("content")
                    if note_id and note_content:  # Ensure both ID (already string) and content
                        items_to_embed.append({
                            "id": note_id,
                            "text": note_content,  # Changed from "content" to "text" for ChromaDBManager
                            "filename": note_item.get("title", f"note_{note_id}")
                        })
                    else:
                        logger.warning(
                            f"Skipping note due to missing ID or content: {note_item.get('id', 'Unknown ID')}")

            except CharactersRAGDBError as e:  # Assuming NotesService uses this error type
                logger.error(f"CharactersRAGDBError while loading notes: {e}", exc_info=True)
                await update_status(f"❌ Database Error: Failed to load notes. {escape(str(e))}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error loading notes: {e}", exc_info=True)
                await update_status(f"❌ Error: Failed to load notes. {escape(str(e))}")
                return []

        elif db_type == "rag_chat_db":  # Conversations from ChaChaNotes_DB via CharactersRAGDB
            if not self.app_instance.chachanotes_db:
                logger.error("CharactersRAGDB instance (self.app_instance.chachanotes_db) is not available.")
                await update_status("❌ Error: Chat DB not initialized. Cannot load conversations.")
                return []

            chat_db: CharactersRAGDB = self.app_instance.chachanotes_db
            raw_conv_data: List[Dict[str, Any]] = []

            try:
                if selection_mode == "all":
                    await update_status(
                        f"⏳ Loading ALL conversations from Chat DB ({self.DB_DISPLAY_NAMES.get(db_type, db_type)})...")
                    raw_conv_data = await asyncio.to_thread(chat_db.list_all_active_conversations, limit=10000)

                elif selection_mode == "individual" and selected_item_db_ids:
                    await update_status(f"⏳ Loading {len(selected_item_db_ids)} selected conversation(s)...")
                    for conv_id_str in selected_item_db_ids:  # conv_id_str is UUID
                        conv = await asyncio.to_thread(chat_db.get_conversation_by_id, conversation_id=conv_id_str)
                        if conv:
                            raw_conv_data.append(conv)
                        else:
                            logger.warning(f"Conversation with ID '{conv_id_str}' not found.")

                elif selection_mode == "keyword" and keyword:
                    await update_status(f"⏳ Searching conversations by title matching '{keyword}'...")
                    raw_conv_data = await asyncio.to_thread(chat_db.search_conversations_by_title, title_query=keyword,
                                                            limit=1000)

                for conv_item in raw_conv_data:
                    conv_id = conv_item.get("id")
                    if not conv_id:
                        logger.warning(
                            f"Skipping conversation due to missing ID: {conv_item.get('title', 'Untitled Conversation')}")
                        continue

                    messages = await asyncio.to_thread(chat_db.get_messages_for_conversation, conversation_id=conv_id,
                                                       limit=1000,
                                                       order_by_timestamp="ASC")  # Fetch a good number of messages

                    full_conv_content_parts = []
                    for msg in messages:
                        sender = msg.get("sender", "Unknown")
                        content_text = msg.get("content", "")
                        if content_text:  # Only include messages with actual text content
                            full_conv_content_parts.append(f"{sender}: {content_text}")

                    if full_conv_content_parts:
                        concatenated_content = "\n\n".join(full_conv_content_parts)  # Separate messages clearly
                        items_to_embed.append({
                            "id": conv_id,
                            "text": concatenated_content,  # Changed from "content" to "text" for ChromaDBManager
                            "filename": conv_item.get("title", f"conversation_{conv_id}")
                        })
                    else:
                        logger.info(
                            f"Conversation ID '{conv_id}' (Title: {conv_item.get('title', 'N/A')}) has no textual messages to embed.")

            except CharactersRAGDBError as e:
                logger.error(f"CharactersRAGDBError while loading conversations: {e}", exc_info=True)
                await update_status(f"❌ Database Error: Failed to load conversations. {escape(str(e))}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error loading conversations: {e}", exc_info=True)
                await update_status(f"❌ Error: Failed to load conversations. {escape(str(e))}")
                return []
        else:
            logger.error(f"Unknown db_type '{db_type}' encountered in _load_items_for_embedding.")
            await update_status(f"❌ Error: Unsupported database source '{db_type}'.")
            return []

        if items_to_embed:
            logger.info(
                f"Successfully loaded {len(items_to_embed)} items for embedding from {self.DB_DISPLAY_NAMES.get(db_type, db_type)}.")
            await update_status(
                f"ℹ️ Loaded {len(items_to_embed)} items from {self.DB_DISPLAY_NAMES.get(db_type, db_type)} for processing.")
        else:
            logger.info(
                f"No items loaded for embedding from {self.DB_DISPLAY_NAMES.get(db_type, db_type)} with current criteria.")
            await update_status(
                f"ℹ️ No items found in {self.DB_DISPLAY_NAMES.get(db_type, db_type)} matching your selection.")

        return items_to_embed


    async def _create_embeddings(self, db_type: str, db_path: str, # db_path is conceptual for now
                               collection_name: str,
                               embedding_model_id_override: str, # This is the key for EmbeddingFactory config
                               chunk_options: Dict[str, Any],
                               selection_mode: str = "all",
                               keyword: Optional[str] = None,
                               selected_item_ids: Optional[List[str]] = None) -> None:
        """
        Loads data based on db_type and selection, then creates and stores embeddings.
        """
        if not EMBEDDINGS_GENERATION_AVAILABLE:
            await self.query_one("#creation-status-output", Markdown).update(
                "⚠️ Embedding generation is disabled because dependencies are missing.")
            return
        logger.info(f"_create_embeddings: Starting embedding creation for {db_type}, collection '{collection_name}', model '{embedding_model_id_override}', mode '{selection_mode}'")
        logger.debug(f"_create_embeddings: Parameters - db_type={db_type}, db_path={db_path}, collection_name={collection_name}, " +
                    f"embedding_model_id_override={embedding_model_id_override}, selection_mode={selection_mode}, " +
                    f"keyword={keyword}, selected_item_ids={selected_item_ids}, chunk_options={chunk_options}")

        status_output = self.query_one("#creation-status-output", Markdown)
        logger.debug("_create_embeddings: Retrieved status_output widget")

        logger.debug(f"_create_embeddings: Loading items for embedding with selection_mode={selection_mode}")
        items_to_embed = await self._load_items_for_embedding(
            db_type, selection_mode, keyword, selected_item_ids
        )

        if not items_to_embed:
            await self._append_status(f"ℹ️ No items found in {self.DB_DISPLAY_NAMES.get(db_type, db_type)} matching selection criteria. Nothing to embed.")
            logger.warning(f"_create_embeddings: No items to embed for {db_type} with mode {selection_mode}")
            return

        logger.info(f"_create_embeddings: Found {len(items_to_embed)} items to embed")
        await self._append_status(f"⏳ Found {len(items_to_embed)} items. Processing embeddings with model '{embedding_model_id_override}'...")

        # ---- Step 2: Process each item with ChromaDBManager ----
        logger.debug("_create_embeddings: Getting ChromaDB manager")
        chroma_manager = await self._get_chroma_manager()
        if not chroma_manager:
            logger.error("_create_embeddings: Failed to get ChromaDB manager")
            await status_output.update("❌ Error: ChromaDB manager not available")
            return

        successful_embeds = 0
        failed_embeds = 0

        logger.info(f"_create_embeddings: Beginning to process {len(items_to_embed)} items for embedding")
        for idx, item_data in enumerate(items_to_embed):
            item_id = item_data.get("id") # This is the original item's ID (e.g., Media.id, Note.id)
            item_content = item_data.get("text")  # Changed from "content" to "text" to match the data structure
            item_filename = item_data.get("filename", str(item_id))

            if not item_id or not item_content:
                logger.warning(f"_create_embeddings: Skipping item {idx+1}/{len(items_to_embed)} due to missing ID or content: {item_data}")
                failed_embeds += 1
                continue

            try:
                # Update UI before blocking call
                current_status_msg = f"⏳ Embedding '{item_filename}' (ID: {item_id}). Processed: {successful_embeds+failed_embeds}/{len(items_to_embed)}"
                await self._append_status(current_status_msg)
                logger.info(f"_create_embeddings: {current_status_msg}")
                logger.debug(f"_create_embeddings: Processing item {idx+1}/{len(items_to_embed)}, ID={item_id}, filename={item_filename}")

                # process_and_store_content is synchronous, run in thread for TUI
                # Pass self.app_instance.app_config as user_embedding_config if ChromaDBManager needs it for situate_context's LLM call
                logger.debug(f"_create_embeddings: Calling process_and_store_content for item {item_id}")
                await asyncio.to_thread(
                    chroma_manager.process_and_store_content,
                    content=item_content,
                    media_id=item_id,
                    file_name=item_filename,
                    collection_name=collection_name,
                    embedding_model_id_override=embedding_model_id_override,
                    create_embeddings=True,
                    create_contextualized=False, # TODO: Add UI option for this
                    # llm_model_for_context=self.app_instance.app_config.get("embedding_config",{}).get("default_llm_for_contextualization"), # If contextualized
                    chunk_options=chunk_options
                )
                successful_embeds += 1
                logger.info(f"_create_embeddings: Successfully processed and stored embeddings for item ID '{item_id}' into collection '{collection_name}'")
            except Exception as e_process:
                failed_embeds += 1
                logger.error(f"_create_embeddings: Failed to process item ID '{item_id}': {e_process}", exc_info=True)
                await self._append_status(f"⚠️ Error embedding '{item_filename}': {escape(str(e_process))[:100]}...")
                await asyncio.sleep(0.2) # Brief pause for UI update

        if failed_embeds > 0:
            final_status_message = f"⚠️ Embedding Creation was partially successful. Please see log for more details. Collection: '{collection_name}'. Successful: {successful_embeds}, Failed: {failed_embeds}."
        else:
            final_status_message = f"✅ Embedding process complete for collection '{collection_name}'. Successful: {successful_embeds}, Failed: {failed_embeds}."
        
        logger.info(f"_create_embeddings: {final_status_message}")
        await self._append_status(final_status_message)


    async def _check_and_display_embedding_status(self) -> None:
        """Fetches and displays the status of the currently selected embedding from a Chroma collection."""
        details_md = self.query_one("#mgmt-embedding-details-md", Markdown)
        mgmt_status_output = self.query_one("#mgmt-status-output", Markdown)

        if not self._selected_chroma_collection_name:
            await details_md.update("### No Collection Selected\n\nPlease select a collection first.")
            return
        if not self._selected_embedding_collection_item_id:  # This is the Chroma ID (e.g., media_1_chunk_0)
            await details_md.update("### No Item Selected\n\nPlease select an item from the collection.")
            return

        item_display_name = self._selected_item_display_name or "Selected Item"  # Fallback display name
        await details_md.update(
            f"### ⏳ Checking Status\n\nRetrieving embedding information for: `{item_display_name}` from collection `{self._selected_chroma_collection_name}`...")
        await mgmt_status_output.update(f"⏳ Checking embedding status for {item_display_name}...")

        try:
            chroma_manager = await self._get_chroma_manager()
            collection = chroma_manager.client.get_collection(name=self._selected_chroma_collection_name)

            # Fetch the specific item by its Chroma ID
            results = collection.get(
                ids=[self._selected_embedding_collection_item_id],
                include=["metadatas", "documents", "embeddings"]  # Include document for context
            )

            if not results or not results['ids']:
                await details_md.update(
                    f"### ❌ No Embedding Found\n\nThe item `{item_display_name}` (ID: `{self._selected_embedding_collection_item_id}`) was not found in collection `{self._selected_chroma_collection_name}`.")
                await mgmt_status_output.update(f"Item '{item_display_name}' not found in collection.")
                return

            # Item exists, display its info
            metadata = results['metadatas'][0] if results.get('metadatas') else {}
            document_content = results['documents'][0] if results.get('documents') else "N/A"
            embedding_vector = results['embeddings'][0] if results.get('embeddings') else []

            embedding_preview = str(embedding_vector[:5]) + "..." if embedding_vector else "N/A"
            embedding_dimensions = len(embedding_vector) if embedding_vector else "Unknown"

            md_content = f"### Embedding Information for `{item_display_name}`\n\n"
            md_content += f"- **Chroma ID:** `{self._selected_embedding_collection_item_id}`\n"
            md_content += f"- **Collection:** `{self._selected_chroma_collection_name}`\n"
            md_content += f"- **Dimensions:** `{embedding_dimensions}`\n"
            md_content += f"- **Vector Preview:** `{escape(embedding_preview)}`\n\n"

            md_content += f"#### Original Text Chunk:\n```\n{escape(document_content[:300])}"
            if len(document_content) > 300:
                md_content += "...\n```\n\n"
            else:
                md_content += "\n```\n\n"

            if metadata:
                md_content += f"#### Metadata:\n"
                for key, val in metadata.items():
                    md_content += f"- **{escape(str(key))}:** `{escape(str(val))}`\n"
            else:
                md_content += "No additional metadata available for this chunk."

            await details_md.update(md_content)
            await mgmt_status_output.update(f"Details loaded for: {item_display_name}")

        except Exception as e:
            logger.error(f"Error checking embedding status: {e}", exc_info=True)
            await details_md.update(
                f"### ❌ Error Checking Status\n\nFailed to retrieve embedding information.\n\n```\n{escape(str(e))}\n```")
            await mgmt_status_output.update(f"Error: Failed to check status for {item_display_name}. See logs.")

#
# End of SearchWindow.py
########################################################################################################################
