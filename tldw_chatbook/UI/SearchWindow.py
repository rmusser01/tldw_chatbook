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

# Conditional imports for embeddings functionality
if DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
    try:
        from tldw_chatbook.Embeddings.Chroma_Lib import ChromaDBManager
        from ..Embeddings.Embeddings_Lib import ModelCfg
    except ImportError:
        ChromaDBManager = None
        ModelCfg = None
else:
    ChromaDBManager = None
    ModelCfg = None

# Set availability flags based on centralized dependency checking
EMBEDDINGS_GENERATION_AVAILABLE = DEPENDENCIES_AVAILABLE.get('embeddings_rag', False)
VECTORDB_AVAILABLE = DEPENDENCIES_AVAILABLE.get('embeddings_rag', False)
WEB_SEARCH_AVAILABLE = DEPENDENCIES_AVAILABLE.get('websearch', False)

# Import modules conditionally
if EMBEDDINGS_GENERATION_AVAILABLE:
    try:
        from ..Embeddings.Embeddings_Lib import *
        logger.info("✅ Embeddings Generation dependencies found. Feature is enabled.")
    except (ImportError, ModuleNotFoundError) as e:
        EMBEDDINGS_GENERATION_AVAILABLE = False
        logger.warning(f"Embeddings Generation dependencies not found, features related will be disabled. Reason: {e}")

if VECTORDB_AVAILABLE:
    try:
        from ..Embeddings.Chroma_Lib import *
        logger.info("✅ Vector Database dependencies found. Feature is enabled.")
    except (ImportError, ModuleNotFoundError) as e:
        VECTORDB_AVAILABLE = False
        logger.warning(f"Vector Database dependencies not found, features related will be disabled. Reason: {e}")

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
        self._selected_item_display_name: Optional[str] = None  # For management view display name

    async def on_mount(self) -> None:
        """Called when the window is first mounted."""
        logger.info("SearchWindow.on_mount: Setting and initializing initial active sub-tab.")

        for view in self.query(".search-view-area"):
            view.display = False
            logger.debug(f"SearchWindow.on_mount: Setting view {view.id} display to False")

        # Default to a view based on available dependencies
        default_view = SEARCH_VIEW_RAG_QA
        if EMBEDDINGS_GENERATION_AVAILABLE and VECTORDB_AVAILABLE:
            default_view = SEARCH_VIEW_EMBEDDINGS_CREATION
        elif WEB_SEARCH_AVAILABLE:
            default_view = SEARCH_VIEW_WEB_SEARCH
        
        initial_sub_tab = self.app_instance.search_active_sub_tab or default_view
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
            yield Container(id=SEARCH_VIEW_RAG_MANAGEMENT, classes="search-view-area")

            if EMBEDDINGS_GENERATION_AVAILABLE and VECTORDB_AVAILABLE:
                try:
                    from .Embeddings_Creation_Window import EmbeddingsCreationWindow
                    yield EmbeddingsCreationWindow(app_instance=self.app_instance, id=SEARCH_VIEW_EMBEDDINGS_CREATION, classes="search-view-area")
                except ImportError as e:
                    logger.warning(f"Could not import EmbeddingsCreationWindow: {e}")
                    with Container(id=SEARCH_VIEW_EMBEDDINGS_CREATION, classes="search-view-area"):
                        with VerticalScroll():
                            yield Markdown(
                                "### Embeddings Creation Is Not Currently Available\n\nThe required dependencies for embeddings creation are not installed. Please install the necessary packages to use this feature.")
            else:  # Embeddings not available
                with Container(id=SEARCH_VIEW_EMBEDDINGS_CREATION, classes="search-view-area"):
                    with VerticalScroll():
                        yield Markdown(
                            "### Embeddings Creation Is Not Currently Available\n\nThe required dependencies for embeddings creation are not installed. Please install the necessary packages to use this feature.")

            # --- Embeddings Management View (Single Scrollable Pane) ---
            if VECTORDB_AVAILABLE:
                try:
                    from .Embeddings_Management_Window import EmbeddingsManagementWindow
                    yield EmbeddingsManagementWindow(app_instance=self.app_instance, id=SEARCH_VIEW_EMBEDDINGS_MANAGEMENT, classes="search-view-area")
                except ImportError as e:
                    logger.warning(f"Could not import EmbeddingsManagementWindow: {e}")
                    with Container(id=SEARCH_VIEW_EMBEDDINGS_MANAGEMENT, classes="search-view-area"):
                        with VerticalScroll():
                            yield Markdown(
                                "### Embeddings Management Is Not Currently Available\n\nThe required dependencies for vector database management are not installed. Please install the necessary packages to use this feature.")
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

    async def _get_chroma_manager(self) -> "ChromaDBManager":
        """Get or create a ChromaDBManager instance using the app's configuration."""
        if not VECTORDB_AVAILABLE or ChromaDBManager is None:
            raise RuntimeError("Vector-database functionality is disabled due to missing dependencies. Install with: pip install tldw_chatbook[embeddings_rag]")
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

        # No need to initialize windows anymore - they handle their own initialization
        logger.debug(f"Switched to view '{target_view_id}'")






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


    async def _refresh_collections_list(self) -> None:
        """Alias for _refresh_mgmt_collections_list for compatibility with main app."""
        await self._refresh_mgmt_collections_list()

    async def _refresh_mgmt_collections_list(self) -> None:
        """Refreshes the list of ChromaDB collections in the management view."""
        # Check if vector database dependencies are available
        if not VECTORDB_AVAILABLE:
            logger.warning("Vector database dependencies not available, skipping collection refresh")
            return
            
        try:
            collection_select = self.query_one("#mgmt-collection-select", Select)
            status_output = self.query_one("#mgmt-status-output", Markdown)
        except QueryError:
            logger.warning("Management view elements not found, skipping collection refresh")
            return

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
