# tldw_cli - Textual CLI for LLMs
# Description: This file contains the main application logic for the tldw_cli, a Textual-based CLI for interacting with various LLM APIs.
#
# Imports
import functools
import inspect
import logging
import logging.handlers
import subprocess
import traceback
from typing import Union, Optional, Any, Dict, List, Callable
#
# 3rd-Party Libraries
from PIL import Image
from loguru import logger as loguru_logger, logger
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import (
    Static, Button, Input, Header, RichLog, TextArea, Select, ListView, Checkbox, Collapsible, ListItem, Label
)

from textual.containers import Container
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.binding import Binding
from textual.dom import DOMNode  # For type hinting if needed
from textual.timer import Timer
from textual.css.query import QueryError
from pathlib import Path

from tldw_chatbook.Utils.text import slugify
from tldw_chatbook.css.Themes.themes import ALL_THEMES
#
# --- Local API library Imports ---
from .Event_Handlers.LLM_Management_Events import (llm_management_events, llm_management_events_mlx_lm,
    llm_management_events_ollama, llm_management_events_onnx, llm_management_events_transformers,
                                                   llm_management_events_vllm)
from tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events import handle_streaming_chunk, handle_stream_done
from tldw_chatbook.Event_Handlers.worker_events import StreamingChunk, StreamDone
from .Widgets.AppFooterStatus import AppFooterStatus
from .Utils import Utils
from .config import (
    get_media_db_path,
)
from .Logging_Config import configure_application_logging
from tldw_chatbook.Constants import ALL_TABS, TAB_CCP, TAB_CHAT, TAB_LOGS, TAB_NOTES, TAB_STATS, TAB_TOOLS_SETTINGS, \
    TAB_INGEST, TAB_LLM, TAB_MEDIA, TAB_SEARCH, TAB_EVALS, LLAMA_CPP_SERVER_ARGS_HELP_TEXT, \
    LLAMAFILE_SERVER_ARGS_HELP_TEXT, TAB_CODING
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.config import CLI_APP_CLIENT_ID
from tldw_chatbook.Logging_Config import RichLogHandler
from tldw_chatbook.Prompt_Management import Prompts_Interop as prompts_interop
from tldw_chatbook.Utils.Emoji_Handling import get_char, EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN, EMOJI_TITLE_NOTE, \
    FALLBACK_TITLE_NOTE, EMOJI_TITLE_SEARCH, FALLBACK_TITLE_SEARCH, supports_emoji
from .config import (
    CONFIG_TOML_CONTENT,
    DEFAULT_CONFIG_PATH,
    load_settings,
    get_cli_setting,
    get_cli_providers_and_models,
    API_MODELS_BY_PROVIDER,
    LOCAL_PROVIDERS, )
from .Event_Handlers import (
    conv_char_events as ccp_handlers,
    notes_events as notes_handlers,
    worker_events as worker_handlers, worker_events, ingest_events,
    llm_nav_events, media_events, notes_events, app_lifecycle, tab_events,
)
from .Event_Handlers.Chat_Events import chat_events as chat_handlers, chat_events_sidebar
from tldw_chatbook.Event_Handlers.Chat_Events import chat_events
from .Notes.Notes_Library import NotesInteropService
from .DB.ChaChaNotes_DB import CharactersRAGDBError, ConflictError
from .Widgets.chat_message import ChatMessage
from .Widgets.notes_sidebar_left import NotesSidebarLeft
from .Widgets.notes_sidebar_right import NotesSidebarRight
from .Widgets.titlebar import TitleBar
from .LLM_Calls.LLM_API_Calls import (
        chat_with_openai, chat_with_anthropic, chat_with_cohere,
        chat_with_groq, chat_with_openrouter, chat_with_huggingface,
        chat_with_deepseek, chat_with_mistral, chat_with_google,
)
from .LLM_Calls.LLM_API_Calls_Local import (
    chat_with_llama, chat_with_kobold, chat_with_oobabooga,
    chat_with_vllm, chat_with_tabbyapi, chat_with_aphrodite,
    chat_with_ollama, chat_with_custom_openai, chat_with_custom_openai_2, chat_with_local_llm
)
from tldw_chatbook.config import get_chachanotes_db_path, settings, chachanotes_db as global_db_instance
from .UI.Chat_Window import ChatWindow
from .UI.Conv_Char_Window import CCPWindow
from .UI.Notes_Window import NotesWindow
from .UI.Logs_Window import LogsWindow
from .UI.Stats_Window import StatsWindow
from .UI.Ingest_Window import IngestWindow, INGEST_NAV_BUTTON_IDS, MEDIA_TYPES
from .UI.Tools_Settings_Window import ToolsSettingsWindow
from .UI.LLM_Management_Window import LLMManagementWindow
from .UI.Evals_Window import EvalsWindow # Added EvalsWindow
from .UI.Coding_Window import CodingWindow
from .UI.Tab_Bar import TabBar
from .UI.MediaWindow import MediaWindow
from .UI.SearchWindow import SearchWindow
from .UI.SearchWindow import ( # Import new constants from SearchWindow.py
    SEARCH_VIEW_RAG_QA
)
API_IMPORTS_SUCCESSFUL = True
#
#######################################################################################################################
#
# Statics

if API_IMPORTS_SUCCESSFUL:
    API_FUNCTION_MAP = {
        "OpenAI": chat_with_openai,
        "Anthropic": chat_with_anthropic,
        "Cohere": chat_with_cohere,
        "HuggingFace": chat_with_huggingface,
        "DeepSeek": chat_with_deepseek,
        "Google": chat_with_google, # Key from config
        "Groq": chat_with_groq,
        "koboldcpp": chat_with_kobold,  # Key from config
        "llama_cpp": chat_with_llama,  # Key from config
        "MistralAI": chat_with_mistral,  # Key from config
        "Oobabooga": chat_with_oobabooga,  # Key from config
        "OpenRouter": chat_with_openrouter,
        "vllm": chat_with_vllm,  # Key from config
        "TabbyAPI": chat_with_tabbyapi,  # Key from config
        "Aphrodite": chat_with_aphrodite,  # Key from config
        "Ollama": chat_with_ollama,  # Key from config
        "Custom": chat_with_custom_openai,  # Key from config
        "Custom_2": chat_with_custom_openai_2,  # Key from config
        "local-llm": chat_with_local_llm
    }
    logging.info(f"API_FUNCTION_MAP populated with {len(API_FUNCTION_MAP)} entries.")
else:
    API_FUNCTION_MAP = {}
    logging.error("API_FUNCTION_MAP is empty due to import failures.")

ALL_API_MODELS = {**API_MODELS_BY_PROVIDER, **LOCAL_PROVIDERS} # If needed for sidebar defaults
AVAILABLE_PROVIDERS = list(ALL_API_MODELS.keys()) # If needed
#
#
#####################################################################################################################
#
# Functions:

# --- Global variable for config ---
APP_CONFIG = load_settings()

# Configure root logger based on config BEFORE app starts fully
_initial_log_level_str = APP_CONFIG.get("general", {}).get("log_level", "INFO").upper()
_initial_log_level = getattr(logging, _initial_log_level_str, logging.INFO)
# Define a basic initial format
_initial_log_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
# Remove existing handlers before basicConfig to avoid duplicates if script is re-run
logging.basicConfig(level=_initial_log_level, format=_initial_log_format,
                    force=True)  # force=True might help override defaults
logging.info("Initial basic logging configured.")


# --- Main App ---
class TldwCli(App[None]):  # Specify return type for run() if needed, None is common
    """A Textual app for interacting with LLMs."""
    #TITLE = "🧠📝🔍  tldw CLI"
    TITLE = f"{get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}{get_char(EMOJI_TITLE_NOTE, FALLBACK_TITLE_NOTE)}{get_char(EMOJI_TITLE_SEARCH, FALLBACK_TITLE_SEARCH)}  tldw CLI"
    # Use forward slashes for paths, works cross-platform
    CSS_PATH = str(Path(__file__).parent / "css/tldw_cli.tcss")
    BINDINGS = [Binding("ctrl+q", "quit", "Quit App", show=True)]

    ALL_INGEST_VIEW_IDS = [
        "ingest-view-prompts", "ingest-view-characters",
        "ingest-view-media", "ingest-view-notes",
        *[f"ingest-view-tldw-api-{mt}" for mt in MEDIA_TYPES]
    ]
    ALL_MAIN_WINDOW_IDS = [ # Assuming these are your main content window IDs
        "chat-window", "conversations_characters_prompts-window",
        "ingest-window", "tools_settings-window", "llm_management-window",
        "media-window", "search-window", "logs-window", "evals-window", "coding-window"
    ]

    # Define reactive at class level with a placeholder default and type hint
    current_tab: reactive[str] = reactive("")
    ccp_active_view: reactive[str] = reactive("conversation_details_view")

    # Add state to hold the currently streaming AI message widget
    current_ai_message_widget: Optional[ChatMessage] = None
    current_chat_worker: Optional[Worker] = None
    current_chat_is_streaming: bool = False

    # --- REACTIVES FOR PROVIDER SELECTS ---
    # Initialize with a dummy value or fetch default from config here
    # Ensure the initial value matches what's set in compose/settings_sidebar
    # Fetching default provider from config:
    _default_chat_provider = APP_CONFIG.get("chat_defaults", {}).get("provider", "Ollama")
    _default_ccp_provider = APP_CONFIG.get("character_defaults", {}).get("provider", "Anthropic") # Changed from character_defaults

    chat_api_provider_value: reactive[Optional[str]] = reactive(_default_chat_provider)
    # Renamed character_api_provider_value to ccp_api_provider_value for clarity with TAB_CCP
    ccp_api_provider_value: reactive[Optional[str]] = reactive(_default_ccp_provider)

    # --- Reactives for CCP Character EDITOR (Center Pane) ---
    current_editing_character_id: reactive[Optional[str]] = reactive(None)
    current_editing_character_data: reactive[Optional[Dict[str, Any]]] = reactive(None)

    # DB Size checker - now using AppFooterStatus
    _db_size_status_widget: Optional[AppFooterStatus] = None
    _db_size_update_timer: Optional[Timer] = None

    # Reactives for sidebar
    chat_sidebar_collapsed: reactive[bool] = reactive(False)
    chat_right_sidebar_collapsed: reactive[bool] = reactive(False)  # For character sidebar
    notes_sidebar_left_collapsed: reactive[bool] = reactive(False)
    notes_sidebar_right_collapsed: reactive[bool] = reactive(False)
    conv_char_sidebar_left_collapsed: reactive[bool] = reactive(False)
    conv_char_sidebar_right_collapsed: reactive[bool] = reactive(False)
    evals_sidebar_collapsed: reactive[bool] = reactive(False) # Added for Evals tab

    # Reactive variables for selected note details
    current_selected_note_id: reactive[Optional[str]] = reactive(None)
    current_selected_note_version: reactive[Optional[int]] = reactive(None)
    current_selected_note_title: reactive[Optional[str]] = reactive(None)
    current_selected_note_content: reactive[Optional[str]] = reactive("")

    # --- Reactives for chat sidebar prompt display ---
    chat_sidebar_selected_prompt_id: reactive[Optional[int]] = reactive(None)
    chat_sidebar_selected_prompt_system: reactive[Optional[str]] = reactive(None)
    chat_sidebar_selected_prompt_user: reactive[Optional[str]] = reactive(None)

    # Chats
    current_chat_is_ephemeral: reactive[bool] = reactive(True)  # Start new chats as ephemeral
    # Reactive variable for current chat conversation ID
    current_chat_conversation_id: reactive[Optional[str]] = reactive(None)
    # Reactive variable for current conversation loaded in the Conversations, Characters & Prompts tab
    current_conv_char_tab_conversation_id: reactive[Optional[str]] = reactive(None)
    current_chat_active_character_data: reactive[Optional[Dict[str, Any]]] = reactive(None)
    current_ccp_character_details: reactive[Optional[Dict[str, Any]]] = reactive(None)
    current_ccp_character_image: Optional[Image.Image] = None

    # For Chat Sidebar Prompts section
    chat_sidebar_loaded_prompt_id: reactive[Optional[Union[int, str]]] = reactive(None)
    chat_sidebar_loaded_prompt_title_text: reactive[str] = reactive("")
    chat_sidebar_loaded_prompt_system_text: reactive[str] = reactive("")
    chat_sidebar_loaded_prompt_user_text: reactive[str] = reactive("")
    chat_sidebar_loaded_prompt_keywords_text: reactive[str] = reactive("")
    chat_sidebar_prompt_display_visible: reactive[bool] = reactive(False, layout=True)

    # Prompts
    current_prompt_id: reactive[Optional[int]] = reactive(None)
    current_prompt_uuid: reactive[Optional[str]] = reactive(None)
    current_prompt_name: reactive[Optional[str]] = reactive(None)
    current_prompt_author: reactive[Optional[str]] = reactive(None)
    current_prompt_details: reactive[Optional[str]] = reactive(None)
    current_prompt_system: reactive[Optional[str]] = reactive(None)
    current_prompt_user: reactive[Optional[str]] = reactive(None)
    current_prompt_keywords_str: reactive[Optional[str]] = reactive("") # Store as comma-sep string for UI
    current_prompt_version: reactive[Optional[int]] = reactive(None) # If DB provides it and you need it
    # is_new_prompt can be inferred from current_prompt_id being None

    # Media Tab
    _media_types_for_ui: List[str] = []
    _initial_media_view_slug: Optional[str] = reactive(slugify("All Media"))  # Default to "All Media" slug

    current_media_type_filter_slug: reactive[Optional[str]] = reactive(slugify("All Media"))  # Slug for filtering
    current_media_type_filter_display_name: reactive[Optional[str]] = reactive("All Media")  # Display name
    media_current_page: reactive[int] = reactive(1) # Search results pagination

    # current_media_search_term: reactive[str] = reactive("") # Handled by inputs directly
    current_loaded_media_item: reactive[Optional[Dict[str, Any]]] = reactive(None)
    _media_search_timers: Dict[str, Timer] = {}  # For debouncing per media type
    _media_sidebar_search_timer: Optional[Timer] = None # For chat sidebar media search debouncing

    # Add media_types_for_ui to store fetched types
    media_types_for_ui: List[str] = []
    _initial_media_view: Optional[str] = "media-view-video-audio"  # Default to the first sub-tab
    media_db: Optional[MediaDatabase] = None
    current_sidebar_media_item: Optional[Dict[str, Any]] = None # For chat sidebar media review

    # Search Tab's active sub-view reactives
    search_active_sub_tab: reactive[Optional[str]] = reactive(None)
    _initial_search_sub_tab_view: Optional[str] = SEARCH_VIEW_RAG_QA

    # Ingest Tab
    ingest_active_view: reactive[Optional[str]] = reactive(None)
    _initial_ingest_view: Optional[str] = "ingest-view-prompts"
    selected_prompt_files_for_import: List[Path] = []
    parsed_prompts_for_preview: List[Dict[str, Any]] = []
    last_prompt_import_dir: Optional[Path] = None
    selected_note_files_for_import: List[Path] = []
    parsed_notes_for_preview: List[Dict[str, Any]] = []
    last_note_import_dir: Optional[Path] = None
    # Add attributes to hold the handlers (optional, but can be useful)
    prompt_import_success_handler: Optional[Callable] = None
    prompt_import_failure_handler: Optional[Callable] = None
    character_import_success_handler: Optional[Callable] = None
    character_import_failure_handler: Optional[Callable] = None
    note_import_success_handler: Optional[Callable] = None
    note_import_failure_handler: Optional[Callable] = None

    # Tools Tab
    tools_settings_active_view: reactive[Optional[str]] = reactive(None)  # Or a default view ID
    _initial_tools_settings_view: Optional[str] = "view_general_settings"

    _prompt_search_timer: Optional[Timer] = None

    # LLM Inference Tab
    llm_active_view: reactive[Optional[str]] = reactive(None)
    _initial_llm_view: Optional[str] = "llm-view-llama-cpp"
    llamacpp_server_process: Optional[subprocess.Popen] = None
    llamafile_server_process: Optional[subprocess.Popen] = None
    vllm_server_process: Optional[subprocess.Popen] = None
    ollama_server_process: Optional[subprocess.Popen] = None
    mlx_server_process: Optional[subprocess.Popen] = None
    onnx_server_process: Optional[subprocess.Popen] = None

    # De-Bouncers
    _conv_char_search_timer: Optional[Timer] = None
    _conversation_search_timer: Optional[Timer] = None
    _notes_search_timer: Optional[Timer] = None
    _chat_sidebar_prompt_search_timer: Optional[Timer] = None # New timer

    # Make API_IMPORTS_SUCCESSFUL accessible if needed by old methods or directly
    API_IMPORTS_SUCCESSFUL = API_IMPORTS_SUCCESSFUL

    # User ID for notes, will be initialized in __init__
    current_user_id: str = "default_user" # Will be overridden by self.notes_user_id

    # For Chat Tab's Notes section
    current_chat_note_id: Optional[str] = None
    current_chat_note_version: Optional[int] = None

    # Shared state for tldw API requests
    _last_tldw_api_request_context: Dict[str, Any] = {}

    def __init__(self):
        super().__init__()
        self.MediaDatabase = MediaDatabase
        self.app_config = load_settings()
        self.loguru_logger = loguru_logger # Make loguru_logger an instance variable for handlers
        self.prompts_client_id = "tldw_tui_client_v1" # Store client ID for prompts service

        self.parsed_prompts_for_preview = [] # <<< INITIALIZATION for prompts
        self.last_prompt_import_dir = None

        self.selected_character_files_for_import = []
        self.parsed_characters_for_preview = [] # <<< INITIALIZATION for characters
        self.last_character_import_dir = None
        # Initialize Ingest Tab related attributes
        self.selected_prompt_files_for_import = []
        self.parsed_prompts_for_preview = []
        self.last_prompt_import_dir = Path.home()  # Or Path(".")
        self.selected_notes_files_for_import = []
        self.parsed_notes_for_preview = [] # <<< INITIALIZATION for notes
        self.last_notes_import_dir = None
        # Llama.cpp server process
        self.llamacpp_server_process = None
        # LlamaFile server process
        self.llamafile_server_process = None
        # vLLM server process
        self.vllm_server_process = None
        self.ollama_server_process = None
        self.mlx_server_process = None
        self.onnx_server_process = None
        self.media_current_page = 1
        self.media_search_current_page = 1
        self.media_search_total_pages = 1

        # 1. Get the user name from the loaded settings
        # The fallback here should match what you expect if settings doesn't have it,
        # or what's defined as the ultimate default in config.py.
        user_name_for_notes = settings.get("USERS_NAME", "default_tui_user")
        self.notes_user_id = user_name_for_notes  # This ID will be passed to service methods

        # 2. Get the full path to the unified ChaChaNotes DB FILE
        chachanotes_db_file_path = get_chachanotes_db_path()  # This comes from config.py
        logger.info(f"Unified ChaChaNotes DB file path: {chachanotes_db_file_path}")

        # 3. Determine the PARENT DIRECTORY for NotesInteropService's 'base_db_directory'
        #    This is what NotesInteropService's __init__ expects for its mkdir check.
        actual_base_directory_for_service = chachanotes_db_file_path.parent
        unified_db_file_path = get_chachanotes_db_path()
        base_directory_for_notes_service = unified_db_file_path.parent
        logger.info(f"Notes for user '{self.notes_user_id}' will use the unified DB: {chachanotes_db_file_path}")
        logger.info(f"Base directory to be passed to NotesInteropService: {actual_base_directory_for_service}")

        try:
            self.notes_service = NotesInteropService(
                base_db_directory=actual_base_directory_for_service,
                api_client_id="tldw_tui_client_v1",  # Consistent client ID
                global_db_to_use=global_db_instance  # Pass the actual DB object
            )
            # The logger inside NotesInteropService.__init__ will confirm its setup.
            logger.info(f"NotesInteropService successfully initialized for user '{self.notes_user_id}'.")

        except CharactersRAGDBError as e:
            logger.error(f"Failed to initialize NotesInteropService: {e}", exc_info=True)
            self.notes_service = None
        except Exception as e_notes_init:  # Catch any other unexpected error
            logger.error(f"Unexpected error during NotesInteropService initialization: {e_notes_init}", exc_info=True)
            self.notes_service = None

        # --- Providers & Models ---
        logging.debug("__INIT__: Attempting to get providers and models...")
        try:
            # Call the function from the config module
            self.providers_models = get_cli_providers_and_models()
            logging.info(
                f"__INIT__: Successfully retrieved providers_models. Count: {len(self.providers_models)}. Keys: {list(self.providers_models.keys())}")
        except Exception as e_providers:
            logging.error(f"__INIT__: Failed to get providers and models: {e_providers}", exc_info=True)
            self.providers_models = {}

        # --- Initial Tab ---
        initial_tab_from_config = get_cli_setting("general", "default_tab", TAB_CHAT)
        self._initial_tab_value = initial_tab_from_config if initial_tab_from_config in ALL_TABS else TAB_CHAT
        if self._initial_tab_value != initial_tab_from_config: # Log if fallback occurred
            logging.warning(f"Default tab '{initial_tab_from_config}' from config not valid. Falling back to '{self._initial_tab_value}'.")
        logging.info(f"App __init__: Determined initial tab value: {self._initial_tab_value}")
        # current_tab reactive will be set in on_mount after UI is composed

        self._rich_log_handler: Optional[RichLogHandler] = None # For the RichLog widget in Logs tab

        # --- PromptsInteropService Initialization ---
        self.prompts_service_initialized = False
        try:
            prompts_db_path_str = get_cli_setting("database", "prompts_db_path", str(Path.home() / ".local/share/tldw_cli/tldw_cli_prompts_v2.db"))
            prompts_db_path = Path(prompts_db_path_str).expanduser().resolve()
            prompts_db_path.parent.mkdir(parents=True, exist_ok=True)
            prompts_interop.initialize_interop(db_path=prompts_db_path, client_id=self.prompts_client_id)
            self.prompts_service_initialized = True
            logging.info(f"Prompts Interop Service initialized with DB: {prompts_db_path}")
        except Exception as e_prompts:
            self.prompts_service_initialized = False
            logging.error(f"Failed to initialize Prompts Interop Service: {e_prompts}", exc_info=True)

        self._prompt_search_timer = None  # Initialize here

        try:
            media_db_path = get_media_db_path()  # From your config.py
            self.media_db = MediaDatabase(db_path=media_db_path,
                                          client_id=CLI_APP_CLIENT_ID)  # Use constant for client_id
            self.loguru_logger.info(
                f"Media_DB_v2 initialized successfully for client '{CLI_APP_CLIENT_ID}' at {media_db_path}")
            self.loguru_logger.debug(f"ULTRA EARLY APP INIT: self.media_db instance: {self.media_db}")
            if self.media_db:
                self.loguru_logger.debug(f"ULTRA EARLY APP INIT: self.media_db.db_path_str: {self.media_db.db_path_str}")
            else:
                self.loguru_logger.debug("ULTRA EARLY APP INIT: self.media_db is None immediately after successful initialization block (should not happen).")
        except Exception as e_media_init:
            self.loguru_logger.debug(f"ULTRA EARLY APP INIT: CRITICAL ERROR initializing self.media_db: {e_media_init}", exc_info=True)
            self.media_db = None
            self.loguru_logger.critical("ULTRA EARLY APP INIT: self.media_db is None due to exception during initialization.")

        # --- Pre-fetch media types for UI ---
        try:
            if self.media_db:
                db_types = self.media_db.get_distinct_media_types(include_deleted=False, include_trash=False)
                # Now, construct the final list for the UI, adding "All Media"
                self._media_types_for_ui = ["All Media"] + sorted(list(set(db_types)))
                self.loguru_logger.info(f"App __init__: Pre-fetched {len(self._media_types_for_ui)} media types for UI.")
            else:
                self.loguru_logger.error("App __init__: self.media_db is None, cannot pre-fetch media types.")
                self._media_types_for_ui = ["Error: Media DB not loaded"]
        except Exception as e_media_types_fetch:
            self.loguru_logger.critical(f"ULTRA EARLY APP INIT: CRITICAL ERROR fetching _media_types_for_ui: {e_media_types_fetch}", exc_info=True)
            self._media_types_for_ui = ["Error: Exception fetching media types"]

        self.loguru_logger.debug(f"ULTRA EARLY APP INIT: self._media_types_for_ui VALUE: {self._media_types_for_ui}")
        self.loguru_logger.debug(f"ULTRA EARLY APP INIT: self._media_types_for_ui TYPE: {type(self._media_types_for_ui)}")

        # --- Setup Default view for CCP tab ---
        # Initialize self.ccp_active_view based on initial tab or default state if needed
        if self._initial_tab_value == TAB_CCP:
            self.ccp_active_view = "conversation_details_view"  # Default view for CCP tab
        # else: it will default to "conversation_details_view" anyway
        self._ui_ready = False  # Track if UI is fully composed

        # --- Assign DB instances for event handlers ---
        if self.prompts_service_initialized:
            # Get the database instance using the get_db_instance() function
            try:
                self.prompts_db = prompts_interop.get_db_instance()
                logging.info("Assigned prompts_interop.get_db_instance() to self.prompts_db")
            except RuntimeError as e:
                logging.error(f"Error getting prompts_db instance: {e}")
                self.prompts_db = None # Explicitly set to None
        else:
            self.prompts_db = None # Ensure it's None if service failed
            logging.warning("Prompts service not initialized, self.prompts_db set to None.")

        if self.notes_service and hasattr(self.notes_service, 'db') and self.notes_service.db:
            self.chachanotes_db = self.notes_service.db # ChaChaNotesDB is used by NotesInteropService
            logging.info("Assigned self.notes_service.db to self.chachanotes_db")
        elif global_db_instance: # Fallback to global if notes_service didn't set it up as expected on itself
            self.chachanotes_db = global_db_instance
            logging.info("Assigned global_db_instance to self.chachanotes_db as fallback.")
        else:
            logging.error("ChaChaNotesDB (CharactersRAGDB) instance not found/assigned in app.__init__.")
            self.chachanotes_db = None # Explicitly set to None

        # --- Create the master handler map ---
        # This one-time setup makes the dispatcher clean and fast.
        self.button_handler_map = self._build_handler_map()

    def _build_handler_map(self) -> dict:
        """Constructs the master button handler map from all event modules."""

        # --- Generic, Awaitable Helper Handlers ---
        async def _handle_nav(app: 'TldwCli', event: Button.Pressed, *, prefix: str, reactive_attr: str) -> None:
            """Generic handler for switching views within a tab."""
            view_to_activate = event.button.id.replace(f"{prefix}-nav-", f"{prefix}-view-")
            app.loguru_logger.debug(f"Nav button '{event.button.id}' pressed. Activating view '{view_to_activate}'.")
            setattr(app, reactive_attr, view_to_activate)

        async def _handle_sidebar_toggle(app: 'TldwCli', event: Button.Pressed, *, reactive_attr: str) -> None:
            """Generic handler for toggling a sidebar's collapsed state."""
            setattr(app, reactive_attr, not getattr(app, reactive_attr))

        # --- LLM Management Handlers ---
        llm_handlers_map = {
            **llm_management_events.LLM_MANAGEMENT_BUTTON_HANDLERS,
            **llm_nav_events.LLM_NAV_BUTTON_HANDLERS,
            **llm_management_events_mlx_lm.MLX_LM_BUTTON_HANDLERS,
            #**llm_management_events_ollama.OLLAMA_BUTTON_HANDLERS,
            **llm_management_events_onnx.ONNX_BUTTON_HANDLERS,
            **llm_management_events_transformers.TRANSFORMERS_BUTTON_HANDLERS,
            **llm_management_events_vllm.VLLM_BUTTON_HANDLERS,
        }

        # --- Chat Handlers ---

        chat_handlers_map = {
            **chat_events.CHAT_BUTTON_HANDLERS,
            **chat_events_sidebar.CHAT_SIDEBAR_BUTTON_HANDLERS,
            "toggle-chat-left-sidebar": functools.partial(_handle_sidebar_toggle, reactive_attr="chat_sidebar_collapsed"),
            "toggle-chat-right-sidebar": functools.partial(_handle_sidebar_toggle, reactive_attr="chat_right_sidebar_collapsed"),
        }

        # --- Media Tab Handlers (NEW DYNAMIC WAY) ---
        media_handlers_map = {}
        for media_type_name in self._media_types_for_ui:
            slug = slugify(media_type_name)
            media_handlers_map[f"media-nav-{slug}"] = media_events.handle_media_nav_button_pressed
            media_handlers_map[f"media-load-selected-button-{slug}"] = media_events.handle_media_load_selected_button_pressed
            media_handlers_map[f"media-prev-page-button-{slug}"] = media_events.handle_media_page_change_button_pressed
            media_handlers_map[f"media-next-page-button-{slug}"] = media_events.handle_media_page_change_button_pressed

        # # --- Search Handlers ---
        # search_handlers = {
        #     SEARCH_NAV_RAG_QA: functools.partial(_handle_nav, prefix="search", reactive_attr="search_active_sub_tab"),
        #     SEARCH_NAV_RAG_CHAT: functools.partial(_handle_nav, prefix="search", reactive_attr="search_active_sub_tab"),
        #     SEARCH_NAV_EMBEDDINGS_CREATION: functools.partial(_handle_nav, prefix="search",
        #                                                       reactive_attr="search_active_sub_tab"),
        #     SEARCH_NAV_RAG_MANAGEMENT: functools.partial(_handle_nav, prefix="search",
        #                                                  reactive_attr="search_active_sub_tab"),
        #     SEARCH_NAV_EMBEDDINGS_MANAGEMENT: functools.partial(_handle_nav, prefix="search",
        #                                                         reactive_attr="search_active_sub_tab"),
        # }

        # --- Ingest Handlers ---
        ingest_handlers_map = {
            **ingest_events.INGEST_BUTTON_HANDLERS,
            # Add nav handlers using the helper
            **{button_id: functools.partial(_handle_nav, prefix="ingest", reactive_attr="ingest_active_view")
               for button_id in INGEST_NAV_BUTTON_IDS}
        }

        # --- Tools & Settings Handlers ---
        tools_settings_handlers = {
            "ts-nav-general-settings": functools.partial(_handle_nav, prefix="ts-view",
                                                         reactive_attr="tools_settings_active_view"),
            "ts-nav-config-file-settings": functools.partial(_handle_nav, prefix="ts-view",
                                                             reactive_attr="tools_settings_active_view"),
            "ts-nav-db-tools": functools.partial(_handle_nav, prefix="ts-view",
                                                 reactive_attr="tools_settings_active_view"),
            "ts-nav-appearance": functools.partial(_handle_nav, prefix="ts-view",
                                                   reactive_attr="tools_settings_active_view"),
        }

        # --- Evals Handler ---
        evals_handlers = {
            "toggle-evals-sidebar": functools.partial(_handle_sidebar_toggle, reactive_attr="evals_sidebar_collapsed"),
        }

        # Master map organized by tab
        return {
            TAB_CHAT: chat_handlers_map,
            TAB_CCP: {
                **ccp_handlers.CCP_BUTTON_HANDLERS,
                "toggle-conv-char-left-sidebar": functools.partial(_handle_sidebar_toggle,
                                                                   reactive_attr="conv_char_sidebar_left_collapsed"),
                "toggle-conv-char-right-sidebar": functools.partial(_handle_sidebar_toggle,
                                                                    reactive_attr="conv_char_sidebar_right_collapsed"),
            },
            TAB_NOTES: {
                **notes_events.NOTES_BUTTON_HANDLERS,
                "toggle-notes-sidebar-left": functools.partial(_handle_sidebar_toggle,
                                                               reactive_attr="notes_sidebar_left_collapsed"),
                "toggle-notes-sidebar-right": functools.partial(_handle_sidebar_toggle,
                                                                reactive_attr="notes_sidebar_right_collapsed"),
            },
            TAB_MEDIA: {
                **media_events.MEDIA_BUTTON_HANDLERS,
                **{f"media-nav-{slugify(media_type)}": functools.partial(_handle_nav, prefix="media",
                                                                               reactive_attr="media_active_view")
                   for media_type in self._media_types_for_ui},
                "media-nav-all-media": functools.partial(_handle_nav, prefix="media",
                                                         reactive_attr="media_active_view"),
            },
            TAB_INGEST: ingest_handlers_map,
            TAB_LLM: llm_handlers_map,
            TAB_LOGS: app_lifecycle.APP_LIFECYCLE_BUTTON_HANDLERS,
            TAB_TOOLS_SETTINGS: tools_settings_handlers,
            #TAB_SEARCH: search_handlers,
            TAB_EVALS: evals_handlers,
        }

    def _setup_logging(self):
        """Set up logging for the application.

        If early logging was already initialized, this will just set up the RichLogHandler
        for the UI log display widget.
        """
        # Check if we're running as a module (via entry point) or as a script
        if hasattr(self, '_early_logging_initialized') and self._early_logging_initialized:
            # Early logging was already initialized, just set up the RichLogHandler
            logging.info("Logging already initialized early, setting up UI log handlers only")
            try:
                log_display_widget = self.query_one("#app-log-display", RichLog)
                if not self._rich_log_handler:
                    self._rich_log_handler = RichLogHandler(log_display_widget)
                    rich_log_handler_level_str = self.app_config.get("logging", {}).get("rich_log_level", "DEBUG").upper()
                    rich_log_handler_level = getattr(logging, rich_log_handler_level_str, logging.DEBUG)
                    self._rich_log_handler.setLevel(rich_log_handler_level)
                    logging.getLogger().addHandler(self._rich_log_handler)
                    logging.info(f"Added RichLogHandler to existing logging setup (Level: {logging.getLevelName(self._rich_log_handler.level)}).")
            except QueryError:
                logging.error("!!! ERROR: Failed to find #app-log-display widget for RichLogHandler setup.")
            except Exception as e:
                logging.error(f"!!! ERROR setting up RichLogHandler: {e}", exc_info=True)
        else:
            # No early logging, do full initialization
            configure_application_logging(self)

    def compose(self) -> ComposeResult:
        logging.debug("App composing UI...")
        yield Header()
        # Set up the main title bar with a static title
        yield TitleBar()

        # Use new TabBar widget
        yield TabBar(tab_ids=ALL_TABS, initial_active_tab=self._initial_tab_value)

        yield from self.compose_content_area() # Call refactored content area composer

        # Yield the new AppFooterStatus widget instead of the old Footer
        yield AppFooterStatus(id="app-footer-status")
        logging.debug("App compose finished.")

    ############################################################
    #
    # Code that builds the content area of the app aka the main UI.
    #
    ###########################################################
    def compose_content_area(self) -> ComposeResult:
        """Yields the main window component for each tab."""
        with Container(id="content"):
            yield ChatWindow(self, id="chat-window", classes="window")
            yield CCPWindow(self, id="conversations_characters_prompts-window", classes="window")
            yield NotesWindow(self, id="notes-window", classes="window")
            yield MediaWindow(self, id="media-window", classes="window")
            yield SearchWindow(self, id="search-window", classes="window")
            yield IngestWindow(self, id="ingest-window", classes="window")
            yield ToolsSettingsWindow(self, id="tools_settings-window", classes="window")
            yield LLMManagementWindow(self, id="llm_management-window", classes="window")
            yield LogsWindow(self, id="logs-window", classes="window")
            yield StatsWindow(self, id="stats-window", classes="window")
            yield EvalsWindow(self, id="evals-window", classes="window")
            yield CodingWindow(self, id="coding-window", classes="window")

    @on(ChatMessage.Action)
    async def handle_chat_message_action(self, event: ChatMessage.Action) -> None:
        """Handles actions (edit, copy, etc.) from within a ChatMessage widget."""
        button_classes = " ".join(event.button.classes) # Get class string for logging
        self.loguru_logger.debug(
            f"ChatMessage.Action received for button "
            f"(Classes: {button_classes}, Label: '{event.button.label}') "
            f"on message role: {event.message_widget.role}"
        )
        # The event directly gives us the context we need.
        # Now we call the existing handler function with the correct arguments.
        await chat_events.handle_chat_action_button_pressed(
            self, event.button, event.message_widget
        )

    # --- Watcher for CCP Active View ---
    def watch_ccp_active_view(self, old_view: Optional[str], new_view: str) -> None:
        loguru_logger.debug(f"CCP active view changing from '{old_view}' to: '{new_view}'")
        if not self._ui_ready:
            loguru_logger.debug("watch_ccp_active_view: UI not ready, returning.")
            return
        try:
            conversation_messages_view = self.query_one("#ccp-conversation-messages-view")
            prompt_editor_view = self.query_one("#ccp-prompt-editor-view")
            character_card_view = self.query_one("#ccp-character-card-view")
            character_editor_view = self.query_one("#ccp-character-editor-view")

            # Default all to hidden, then enable the correct one
            conversation_messages_view.display = False
            prompt_editor_view.display = False
            character_card_view.display = False
            character_editor_view.display = False

            # REMOVE or COMMENT OUT the query for llm_settings_container_right:
            # llm_settings_container_right = self.query_one("#ccp-right-pane-llm-settings-container")
            # conv_details_collapsible_right = self.query_one("#ccp-conversation-details-collapsible", Collapsible) # Keep if you manipulate its collapsed state

            if new_view == "prompt_editor_view":
                # Center Pane: Show Prompt Editor
                prompt_editor_view.display = True
                # LLM settings container is gone, no need to hide it.
                # llm_settings_container_right.display = False

                # Optionally, manage collapsed state of other sidebars
                self.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = False

                # Focus an element in prompt editor
                try:
                    self.query_one("#ccp-editor-prompt-name-input", Input).focus()
                except QueryError:
                    loguru_logger.warning("Could not focus prompt name input in editor view.")

            elif new_view == "character_editor_view":
                # Center Pane: Show Character Editor
                character_editor_view.display = True
                # Optionally manage right-pane collapsibles
                self.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True
                loguru_logger.info("Character editor view activated. Focus pending specific input fields.")

            elif new_view == "character_card_view":
                # Center Pane: Show Character Card Display
                character_card_view.display = True
                character_editor_view.display = False
                # Optionally manage right-pane collapsibles
                self.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True
                loguru_logger.info("Character card display view activated.")

                if self.current_ccp_character_details:
                    details = self.current_ccp_character_details
                    loguru_logger.info(f"Populating character card with details for: {details.get('name', 'Unknown')}")
                    try:
                        self.query_one("#ccp-card-name-display", Static).update(details.get("name", "N/A"))
                        self.query_one("#ccp-card-description-display", TextArea).text = details.get("description", "")
                        self.query_one("#ccp-card-personality-display", TextArea).text = details.get("personality", "")
                        self.query_one("#ccp-card-scenario-display", TextArea).text = details.get("scenario", "")
                        self.query_one("#ccp-card-first-message-display", TextArea).text = details.get("first_message", "")

                        image_placeholder = self.query_one("#ccp-card-image-placeholder", Static)
                        if self.current_ccp_character_image:
                            image_placeholder.update("Character image loaded (display not implemented)")
                        else:
                            image_placeholder.update("No image available")
                        loguru_logger.debug("Character card widgets populated.")
                    except QueryError as qe:
                        loguru_logger.error(f"QueryError populating character card: {qe}", exc_info=True)
                else:
                    loguru_logger.info("No character details available to populate card view.")
                    try:
                        self.query_one("#ccp-card-name-display", Static).update("N/A")
                        self.query_one("#ccp-card-description-display", TextArea).text = ""
                        self.query_one("#ccp-card-personality-display", TextArea).text = ""
                        self.query_one("#ccp-card-scenario-display", TextArea).text = ""
                        self.query_one("#ccp-card-first-message-display", TextArea).text = ""
                        self.query_one("#ccp-card-image-placeholder", Static).update("No character loaded")
                        loguru_logger.debug("Character card widgets cleared.")
                    except QueryError as qe:
                        loguru_logger.error(f"QueryError clearing character card: {qe}", exc_info=True)

            elif new_view == "conversation_details_view":
                # Center Pane: Show Conversation Messages
                conversation_messages_view.display = True
                # LLM settings container is gone, no need to show it.
                # llm_settings_container_right.display = True
                self.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = False
                self.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True

                try:
                    # If a conversation is loaded, maybe focus its title in right pane
                    if self.current_conv_char_tab_conversation_id:
                        self.query_one("#conv-char-title-input", Input).focus()
                    else:  # Otherwise, maybe focus the search in left pane
                        self.query_one("#conv-char-search-input", Input).focus()
                except QueryError:
                    loguru_logger.warning("Could not focus default element in conversation details view.")
            else:  # Default or unknown view (treat as conversation_details_view)
                # Center Pane: Show Conversation Messages (default)
                conversation_messages_view.display = True
                loguru_logger.warning(
                    f"Unknown ccp_active_view: {new_view}, defaulting to conversation_details_view.")

        except QueryError as e:
            loguru_logger.exception(f"UI component not found during CCP view switch: {e}")
        except Exception as e_watch:
            loguru_logger.exception(f"Unexpected error in watch_ccp_active_view: {e_watch}")

    # --- Watcher for Right Sidebar in CCP Tab ---
    def watch_conv_char_sidebar_right_collapsed(self, collapsed: bool) -> None:
        """Hide or show the Conversations, Characters & Prompts right sidebar pane."""
        if not self._ui_ready:
            loguru_logger.debug("watch_conv_char_sidebar_right_collapsed: UI not ready.")
            return
        try:
            sidebar_pane = self.query_one("#conv-char-right-pane")
            sidebar_pane.set_class(collapsed, "collapsed")  # Add if true, remove if false
            loguru_logger.debug(f"CCP right pane collapsed state: {collapsed}, class set.")
        except QueryError:
            loguru_logger.error("CCP right pane (#conv-char-right-pane) not found for collapse toggle.")
        except Exception as e:
            loguru_logger.error(f"Error toggling CCP right pane: {e}", exc_info=True)

    # ###################################################################
    # --- Helper methods for Local LLM Inference logging ---
    # ###################################################################
    def _update_llamacpp_log(self, message: str) -> None:
        """Helper to write messages to the Llama.cpp log widget."""
        try:
            log_widget = self.query_one("#llamacpp-log-output", RichLog)
            log_widget.write(message)
        except QueryError:
            self.loguru_logger.error("Failed to query #llamacpp-log-output to write message.")
        except Exception as e:  # pylint: disable=broad-except
            self.loguru_logger.error(f"Error writing to Llama.cpp log: {e}", exc_info=True)

    def _update_transformers_log(self, message: str) -> None:
        """Helper to write messages to the Transformers log widget."""
        try:
            # Assuming the Transformers view is active when this is called,
            # or the log widget is always part of the composed layout.
            log_widget = self.query_one("#transformers-log-output", RichLog)
            log_widget.write(message)
        except QueryError:
            self.loguru_logger.error("Failed to query #transformers-log-output to write message.")
        except Exception as e: # pylint: disable=broad-except
            self.loguru_logger.error(f"Error writing to Transformers log: {e}", exc_info=True)

    def _update_llamafile_log(self, message: str) -> None:
        """Helper to write messages to the Llamafile log widget."""
        try:
            log_widget = self.query_one("#llamafile-log-output", RichLog)
            log_widget.write(message)
        except QueryError:
            self.loguru_logger.error("Failed to query #llamafile-log-output to write message.")
        except Exception as e:
            self.loguru_logger.error(f"Error writing to Llamafile log: {e}", exc_info=True)

    def _update_vllm_log(self, message: str) -> None:
        try:
            log_widget = self.query_one("#vllm-log-output", RichLog)
            log_widget.write(message)
        except QueryError:
            self.loguru_logger.error("Failed to query #vllm-log-output to write message.")
        except Exception as e:
            self.loguru_logger.error(f"Error writing to vLLM log: {e}", exc_info=True)
    # ###################################################################
    # --- End of Helper methods for Local LLM Inference logging ---
    # ###################################################################

    # --- Modify _clear_prompt_fields and _load_prompt_for_editing ---
    def _clear_prompt_fields(self) -> None:
        """Clears prompt input fields in the CENTER PANE editor."""
        try:
            self.query_one("#ccp-editor-prompt-name-input", Input).value = ""
            self.query_one("#ccp-editor-prompt-author-input", Input).value = ""
            self.query_one("#ccp-editor-prompt-description-textarea", TextArea).text = ""
            self.query_one("#ccp-editor-prompt-system-textarea", TextArea).text = ""
            self.query_one("#ccp-editor-prompt-user-textarea", TextArea).text = ""
            self.query_one("#ccp-editor-prompt-keywords-textarea", TextArea).text = ""
            loguru_logger.debug("Cleared prompt editor fields in center pane.")
        except QueryError as e:
            loguru_logger.error(f"Error clearing prompt editor fields in center pane: {e}")

    async def _load_prompt_for_editing(self, prompt_id: Optional[int], prompt_uuid: Optional[str] = None) -> None:
        if not self.prompts_service_initialized:
            self.notify("Prompts service not available.", severity="error")
            return

        # Switch to prompt editor view
        self.ccp_active_view = "prompt_editor_view"  # This will trigger the watcher

        identifier_to_fetch = prompt_id if prompt_id is not None else prompt_uuid
        if identifier_to_fetch is None:
            self._clear_prompt_fields()
            self.current_prompt_id = None  # Reset all reactive prompt states
            self.current_prompt_uuid = None
            self.current_prompt_name = None
            # ... etc. for other prompt reactives
            loguru_logger.warning("_load_prompt_for_editing called with no ID/UUID after view switch.")
            return

        try:
            prompt_details = prompts_interop.fetch_prompt_details(identifier_to_fetch)

            if prompt_details:
                self.current_prompt_id = prompt_details.get('id')
                self.current_prompt_uuid = prompt_details.get('uuid')
                self.current_prompt_name = prompt_details.get('name', '')
                self.current_prompt_author = prompt_details.get('author', '')
                self.current_prompt_details = prompt_details.get('details', '')
                self.current_prompt_system = prompt_details.get('system_prompt', '')
                self.current_prompt_user = prompt_details.get('user_prompt', '')
                self.current_prompt_keywords_str = ", ".join(prompt_details.get('keywords', []))
                self.current_prompt_version = prompt_details.get('version')

                # Populate UI in the CENTER PANE editor
                self.query_one("#ccp-editor-prompt-name-input", Input).value = self.current_prompt_name
                self.query_one("#ccp-editor-prompt-author-input", Input).value = self.current_prompt_author
                self.query_one("#ccp-editor-prompt-description-textarea",
                               TextArea).text = self.current_prompt_details
                self.query_one("#ccp-editor-prompt-system-textarea", TextArea).text = self.current_prompt_system
                self.query_one("#ccp-editor-prompt-user-textarea", TextArea).text = self.current_prompt_user
                self.query_one("#ccp-editor-prompt-keywords-textarea",
                               TextArea).text = self.current_prompt_keywords_str

                self.query_one("#ccp-editor-prompt-name-input", Input).focus()  # Focus after loading
                self.notify(f"Prompt '{self.current_prompt_name}' loaded for editing.", severity="information")
            else:
                self.notify(f"Failed to load prompt (ID/UUID: {identifier_to_fetch}).", severity="error")
                self._clear_prompt_fields()  # Clear editor if load fails
                self.current_prompt_id = None  # Reset reactives
        except Exception as e:
            loguru_logger.error(f"Error loading prompt for editing: {e}", exc_info=True)
            self.notify(f"Error loading prompt: {type(e).__name__}", severity="error")
            self._clear_prompt_fields()
            self.current_prompt_id = None  # Reset reactives

    async def refresh_notes_tab_after_ingest(self) -> None:
        """Called after notes are ingested from the Ingest tab to refresh the Notes tab."""
        self.loguru_logger.info("Refreshing Notes tab data after ingestion.")
        if hasattr(notes_handlers, 'load_and_display_notes_handler'):
            await notes_handlers.load_and_display_notes_handler(self)
        else:
            self.loguru_logger.error("notes_handlers.load_and_display_notes_handler not found for refresh.")

    # ##################################################
    # --- Watcher for Search Tab Active Sub-View ---
    # ##################################################
    def watch_search_active_sub_tab(self, old_sub_tab: Optional[str], new_sub_tab: Optional[str]) -> None:
        """Shows the correct sub-tab view in the Search tab and hides others."""
        if not self._ui_ready or not new_sub_tab:
            return

        self.loguru_logger.debug(f"Search sub-tab watcher: Changing from '{old_sub_tab}' to '{new_sub_tab}'")
        try:
            # First, find the parent content pane
            content_pane = self.query_one("#search-content-pane")

            # Iterate through all direct children that are view areas
            for view in content_pane.query(".search-view-area"):
                # Show the view if its ID matches the new sub-tab, otherwise hide it.
                view.styles.display = "block" if view.id == new_sub_tab else "none"

            # Also update the active button style
            nav_pane = self.query_one("#search-left-nav-pane")
            for button in nav_pane.query(".search-nav-button"):
                button_id_as_view = button.id.replace("-nav-", "-view-")
                button.set_class(button_id_as_view == new_sub_tab, "-active-search-sub-view")

            self.loguru_logger.info(f"Switched search sub-tab view to: {new_sub_tab}")

        except QueryError as e:
            self.loguru_logger.error(f"UI component not found during Search sub-tab switch: {e}", exc_info=True)
        except Exception as e_watch:
            self.loguru_logger.error(f"Unexpected error in watch_search_active_sub_tab: {e_watch}", exc_info=True)

        # ############################################
        # --- Media Loaded Item Watcher ---
        # ############################################
        async def watch_current_loaded_media_item(self, media_data: Optional[Dict[str, Any]]) -> None:
            """Watcher to display details when a media item is loaded."""
            if not self._ui_ready:
                self.loguru_logger.debug("watch_current_loaded_media_item: UI not ready, returning.")
                return

            type_slug = self.current_media_type_filter_slug
            if not type_slug:
                self.loguru_logger.warning(
                    "watch_current_loaded_media_item: type_slug is not set, cannot update details display.")
                return

            details_display_widget_id = f"media-details-display-{type_slug}"
            try:
                # Target Markdown widget
                details_display = self.query_one(f"#{details_display_widget_id}", Markdown)

                if media_data:
                    # Special formatting for "analysis-review"
                    if type_slug == "analysis-review":
                        title = media_data.get('title', 'Untitled')
                        url = media_data.get('url', 'No URL')
                        analysis_content = media_data.get('analysis_content', '')
                        if not analysis_content:
                            analysis_content = "No analysis available for this item."
                        markdown_details_string = f"## {title}\n\n**URL:** {url}\n\n### Analysis\n{analysis_content}"
                    else:
                        # Use the existing format_media_details_as_markdown function from media_events
                        markdown_details_string = media_events.format_media_details_as_markdown(self, media_data)

                    await details_display.update(markdown_details_string)  # Use await and update()
                    # self.notify(f"Details for '{media_data.get('title', 'N/A')}' displayed via watcher.") # Optional notification
                else:
                    await details_display.update("### No media item loaded or item cleared.")  # Use await and update()

            except QueryError:
                self.loguru_logger.warning(
                    f"watch_current_loaded_media_item: Could not find Markdown details display '#{details_display_widget_id}' for slug '{type_slug}' to update."
                )
            except Exception as e:
                self.loguru_logger.error(f"Error in watch_current_loaded_media_item: {e}", exc_info=True)

    # ############################################
    # --- Ingest Tab Watcher ---
    # ############################################
    def watch_ingest_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        self.loguru_logger.info(f"watch_ingest_active_view called. Old view: '{old_view}', New view: '{new_view}'")
        if not hasattr(self, "app") or not self.app:
            self.loguru_logger.debug("watch_ingest_active_view: App not fully ready.")
            return
        if not self._ui_ready:
            self.loguru_logger.debug("watch_ingest_active_view: UI not ready.")
            return
        self.loguru_logger.debug(f"Ingest active view changing from '{old_view}' to: '{new_view}'")

        # Get the content pane for the Ingest tab
        try:
            content_pane = self.query_one("#ingest-content-pane")
        except QueryError:
            self.loguru_logger.error("#ingest-content-pane not found. Cannot switch Ingest views.")
            return

        found_new_view = False
        # Iterate over all child elements of #ingest-content-pane that have the class .ingest-view-area
        for child_view_container in content_pane.query(".ingest-view-area"):
            child_id = child_view_container.id # Assuming child_view_container is a DOMNode with an 'id' attribute
            if child_id == new_view:
                child_view_container.styles.display = "block" # Or "flex" if that's the original display style
                self.loguru_logger.info(f"Displaying Ingest view: {child_id}")
                found_new_view = True
            else:
                child_view_container.styles.display = "none"
                self.loguru_logger.debug(f"Hiding Ingest view: {child_id}")

        if new_view and not found_new_view:
            self.loguru_logger.error(f"Target Ingest view '{new_view}' was not found among .ingest-view-area children to display.")
        elif not new_view: # This case occurs if ingest_active_view is set to None
            self.loguru_logger.debug("Ingest active view is None, all ingest sub-views are now hidden (handled by loop).")

    def watch_tools_settings_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        self.loguru_logger.debug(f"Tools & Settings active view changing from '{old_view}' to: '{new_view}'")
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        if not new_view:  # If new_view is None, hide all
            try:
                for view_area in self.query(".ts-view-area"):  # Query all potential view areas
                    view_area.styles.display = "none"
            except QueryError:
                self.loguru_logger.warning(
                    "No .ts-view-area found to hide on tools_settings_active_view change to None.")
            return

        try:
            content_pane = self.query_one("#tools-settings-content-pane")
            # Hide all views first
            for child in content_pane.children:
                if child.id and child.id.startswith("ts-view-"):  # Check if it's one of our view containers
                    child.styles.display = "none"

            # Show the selected view
            if new_view:  # new_view here is the ID of the view container, e.g., "ts-view-general-settings"
                target_view_id_selector = f"#{new_view}"  # Construct selector from the new_view string
                view_to_show = content_pane.query_one(target_view_id_selector, Container)
                view_to_show.styles.display = "block"
                self.loguru_logger.info(f"Switched Tools & Settings view to: {new_view}")

                # Optional: Focus an element within the newly shown view
                # try:
                # view_to_show.query(Input, Button)[0].focus() # Example: focus first Input or Button
                # except IndexError:
                #     pass # No focusable element
            else:  # Should be caught by the `if not new_view:` at the start
                self.loguru_logger.debug("Tools & Settings active view is None, all views hidden.")


        except QueryError as e:
            self.loguru_logger.error(f"UI component not found during Tools & Settings view switch: {e}", exc_info=True)
        except Exception as e_watch:
            self.loguru_logger.error(f"Unexpected error in watch_tools_settings_active_view: {e_watch}", exc_info=True)

    # --- LLM Tab Watcher ---
    def watch_llm_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        self.loguru_logger.debug(f"LLM Management active view changing from '{old_view}' to: '{new_view}'")

        try:
            content_pane = self.query_one("#llm-content-pane")
        except QueryError:
            self.loguru_logger.error("#llm-content-pane not found. Cannot switch LLM views.")
            return

        for child in content_pane.query(".llm-view-area"):  # Query by common class
            child.styles.display = "none"

        if new_view:
            try:
                target_view_id_selector = f"#{new_view}"
                view_to_show = content_pane.query_one(target_view_id_selector, Container)
                view_to_show.styles.display = "block"
                self.loguru_logger.info(f"Switched LLM Management view to: {new_view}")
                # Populate help text when view becomes active
                if new_view == "llm-view-llama-cpp":
                    try:
                        help_widget = view_to_show.query_one("#llamacpp-args-help-display", RichLog)
                        # Check if help_widget has any lines. RichLog.lines is a list of segments.
                        # A simple check is if it has any children (lines are added as children internally).
                        # Or, more robustly, we can set a flag or check if the first line matches our help text.
                        # For simplicity, let's assume if it has children, it's been populated.
                        # A more direct way: RichLog stores its lines in a deque called 'lines'.
                        if not help_widget.lines: # Check if the internal lines deque is empty
                            self.loguru_logger.debug(f"Populating Llama.cpp help text in {new_view} as it's empty.")
                            help_widget.clear() # Ensure it's clear before writing
                            help_widget.write(LLAMA_CPP_SERVER_ARGS_HELP_TEXT)
                        else:
                            self.loguru_logger.debug(f"Llama.cpp help text in {new_view} already populated or not empty.")
                    except QueryError:
                        self.loguru_logger.warning(f"Help display widget #llamacpp-args-help-display not found in {new_view} during view switch.")
                    except Exception as e_help_populate:
                        self.loguru_logger.error(f"Error ensuring Llama.cpp help text in {new_view}: {e_help_populate}", exc_info=True)
                elif new_view == "llm-view-llamafile":
                    try:
                        help_widget = view_to_show.query_one("#llamafile-args-help-display", RichLog)
                        help_widget.clear()  # Clear and rewrite when tab becomes active
                        help_widget.write(LLAMAFILE_SERVER_ARGS_HELP_TEXT)
                        self.loguru_logger.debug(f"Ensured Llamafile help text in {new_view}.")
                    except QueryError:
                        self.loguru_logger.warning(
                            f"Help display widget for Llamafile not found in {new_view} during view switch.")
                # Add similar for other views like llamafile, vllm if they have help sections
                # elif new_view == "llm-view-llamafile":
                #     try:
                #         help_widget = view_to_show.query_one("#llamafile-args-help-display", RichLog)
                #         if not help_widget.document.strip():
                #             help_widget.write(LLAMAFILE_ARGS_HELP_TEXT)
                #     except QueryError: pass
            except QueryError as e:
                self.loguru_logger.error(f"UI component '{new_view}' not found in #llm-content-pane: {e}",
                                         exc_info=True)
        else:
            self.loguru_logger.debug("LLM Management active view is None, all LLM views hidden.")

    def watch_current_chat_is_ephemeral(self, is_ephemeral: bool) -> None:
        self.loguru_logger.debug(f"Chat ephemeral state changed to: {is_ephemeral}")
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        try:
            # --- Controls for EPHEMERAL chat actions ---
            save_current_chat_button = self.query_one("#chat-save-current-chat-button", Button)
            save_current_chat_button.disabled = not is_ephemeral  # Enable if ephemeral

            # --- Controls for PERSISTENT chat metadata ---
            title_input = self.query_one("#chat-conversation-title-input", Input)
            keywords_input = self.query_one("#chat-conversation-keywords-input", TextArea)
            save_details_button = self.query_one("#chat-save-conversation-details-button", Button)
            uuid_display = self.query_one("#chat-conversation-uuid-display", Input)

            title_input.disabled = is_ephemeral  # Disable if ephemeral
            keywords_input.disabled = is_ephemeral  # Disable if ephemeral
            save_details_button.disabled = is_ephemeral  # Disable if ephemeral (cannot save details for non-existent chat)

            if is_ephemeral:
                # Clear details and set UUID display when switching TO ephemeral
                title_input.value = ""
                keywords_input.text = ""
                # Ensure UUID display is also handled
                try:
                    uuid_display = self.query_one("#chat-conversation-uuid-display", Input)
                    uuid_display.value = "Ephemeral Chat"
                except QueryError:
                    loguru_logger.warning(
                        "Could not find #chat-conversation-uuid-display to update for ephemeral state.")
            # ELSE: If switching TO persistent (is_ephemeral is False),
            # the calling function (e.g., load chat, save ephemeral chat button handler)
            # is responsible for POPULATING the title/keywords fields.
            # This watcher correctly enables them here.

        except QueryError as e:
            self.loguru_logger.warning(f"UI component not found while watching ephemeral state: {e}. Tab might not be fully composed or active.")
        except Exception as e_watch:
            self.loguru_logger.error(f"Unexpected error in watch_current_chat_is_ephemeral: {e_watch}", exc_info=True)

    # --- Add explicit methods to update reactives from Select changes ---
    def update_chat_provider_reactive(self, new_value: Optional[str]) -> None:
        self.chat_api_provider_value = new_value # Watcher will call _update_model_select

    def update_ccp_provider_reactive(self, new_value: Optional[str]) -> None: # Renamed
        self.ccp_api_provider_value = new_value # Watcher will call _update_model_select

    def on_mount(self) -> None:
        """Configure logging and schedule post-mount setup."""
        self._setup_logging()
        if self._rich_log_handler:
            self.loguru_logger.debug("Starting RichLogHandler processor task...")
            self._rich_log_handler.start_processor(self)

        # Schedule setup to run after initial rendering
        self.call_after_refresh(self._post_mount_setup)
        self.call_after_refresh(self.hide_inactive_windows)

        # Load up dem themes
        for theme_name in ALL_THEMES:
            self.register_theme(theme_name)

    def hide_inactive_windows(self) -> None:
        """Hides all windows that are not the current active tab."""
        initial_tab = self._initial_tab_value
        self.loguru_logger.debug(f"Hiding inactive windows, keeping '{initial_tab}-window' visible.")
        for window in self.query(".window"):
            is_active = window.id == f"{initial_tab}-window"
            window.display = is_active

    async def _set_initial_tab(self) -> None:  # New method for deferred tab setting
        self.loguru_logger.info("Setting initial tab via call_later.")
        self.current_tab = self._initial_tab_value
        self.loguru_logger.info(f"Initial tab set to: {self.current_tab}")

    async def _post_mount_setup(self) -> None:
        """Operations to perform after the main UI is expected to be fully mounted."""
        self.loguru_logger.info("App _post_mount_setup: Binding Select widgets and populating dynamic content...")
        # Populate LLM help texts
        self.call_later(llm_management_events.populate_llm_help_texts, self)

        try:
            chat_select = self.query_one(f"#{TAB_CHAT}-api-provider", Select)
            self.watch(chat_select, "value", self.update_chat_provider_reactive, init=False)
            self.loguru_logger.debug(f"Bound chat provider Select ({chat_select.id})")
        except QueryError:
            self.loguru_logger.error(
                f"_post_mount_setup: Failed to find chat provider select: #{TAB_CHAT}-api-provider")
        except Exception as e:
            self.loguru_logger.error(f"_post_mount_setup: Error binding chat provider select: {e}", exc_info=True)

        # try:
        #     ccp_select = self.query_one(f"#{TAB_CCP}-api-provider", Select)
        #     #self.watch(ccp_select, "value", self.update_ccp_provider_reactive, init=False)
        #     #self.loguru_logger.debug(f"Bound CCP provider Select ({ccp_select.id})")
        # except QueryError:
        #     self.loguru_logger.error(f"_post_mount_setup: Failed to find CCP provider select: #{TAB_CCP}-api-provider")
        # except Exception as e:
        #     self.loguru_logger.error(f"_post_mount_setup: Error binding CCP provider select: {e}", exc_info=True)

        # Set initial tab now that other bindings might be ready
        # self.current_tab = self._initial_tab_value # This triggers watchers

        # Populate dynamic selects and lists
        # These also might rely on the main tab windows being fully composed.
        self.call_later(chat_handlers.populate_chat_conversation_character_filter_select, self)
        self.call_later(ccp_handlers.populate_ccp_character_select, self)
        self.call_later(ccp_handlers.populate_ccp_prompts_list_view, self)

        # Crucially, set the initial tab *after* bindings and other setup that might depend on queries.
        # The _set_initial_tab will trigger watchers.
        self.call_later(self._set_initial_tab)

        # If initial tab is CCP, trigger its initial search.
        # This should happen *after* current_tab is set.
        # We can put this logic at the end of _set_initial_tab or make watch_current_tab handle it robustly.
        # For now, let's assume watch_current_tab will handle it.
        # if self._initial_tab_value == TAB_CCP: # Check against the initial value
        #    self.call_later(ccp_handlers.perform_ccp_conversation_search, self)
        self.current_tab = self._initial_tab_value
        self.loguru_logger.info(f"Initial tab set to: {self.current_tab}")

        # --- DB Size Indicator Setup ---
        try:
            # Query for the AppFooterStatus widget instance
            self._db_size_status_widget = self.query_one(AppFooterStatus)
            # Or use ID: self._db_size_status_widget = self.query_one("#app-footer-status", AppFooterStatus)
            self.loguru_logger.info("AppFooterStatus widget instance acquired.")

            await self.update_db_sizes()  # Initial population
            self._db_size_update_timer = self.set_interval(60, self.update_db_sizes) # Periodic updates
            self.loguru_logger.info("DB size update timer started for AppFooterStatus.")
        except QueryError:
            self.loguru_logger.error("Failed to find AppFooterStatus widget for DB size display.")
        except Exception as e_db_size:
            self.loguru_logger.error(f"Error setting up DB size indicator with AppFooterStatus: {e_db_size}", exc_info=True)
        # --- End DB Size Indicator Setup ---

        # CRITICAL: Set UI ready state after all bindings and initializations
        self._ui_ready = True

        self.loguru_logger.info("App _post_mount_setup: Post-mount setup completed.")


    async def update_db_sizes(self) -> None:
        """Updates the database size information in the AppFooterStatus widget."""
        self.loguru_logger.debug("Attempting to update DB sizes in AppFooterStatus.")
        if not self._db_size_status_widget:
            self.loguru_logger.warning("_db_size_status_widget (AppFooterStatus) is None, cannot update DB sizes.")
            return

        try:
            # Prompts DB
            prompts_db_path_str = get_cli_setting("database", "prompts_db_path", str(Path.home() / ".local/share/tldw_cli/tldw_cli_prompts_v2.db"))
            prompts_db_file = Path(prompts_db_path_str).expanduser().resolve()
            prompts_size_str = Utils.get_formatted_file_size(prompts_db_file)
            if prompts_size_str is None:
                prompts_size_str = "N/A"

            # ChaChaNotes DB
            chachanotes_db_file = get_chachanotes_db_path()
            chachanotes_size_str = Utils.get_formatted_file_size(chachanotes_db_file)
            if chachanotes_size_str is None:
                chachanotes_size_str = "N/A"

            # Media DB
            media_db_file = get_media_db_path()
            media_size_str = Utils.get_formatted_file_size(media_db_file)
            if media_size_str is None:
                media_size_str = "N/A"

            status_string = f"Prompts DB: {prompts_size_str}  |  Chats/Notes DB: {chachanotes_size_str}  |  Media DB: {media_size_str}"
            self.loguru_logger.debug(f"DB size status string to display in AppFooterStatus: '{status_string}'")
            # Call the custom update method on the AppFooterStatus widget
            self._db_size_status_widget.update_db_sizes_display(status_string)
            self.loguru_logger.info(f"Successfully updated DB sizes in AppFooterStatus: {status_string}")
        except Exception as e:
            self.loguru_logger.error(f"Error updating DB sizes in AppFooterStatus: {e}", exc_info=True)
            if self._db_size_status_widget: # Check again in case it became None somehow
                self._db_size_status_widget.update_db_sizes_display("Error loading DB sizes")


    async def on_shutdown_request(self) -> None:  # Use the imported ShutdownRequest
        logging.info("--- App Shutdown Requested ---")
        if self._rich_log_handler:
            await self._rich_log_handler.stop_processor()
            logging.info("RichLogHandler processor stopped.")

        # --- Stop DB Size Update Timer ---
        if self._db_size_update_timer:
            self._db_size_update_timer.stop()
            self.loguru_logger.info("DB size update timer stopped.")
        # --- End Stop DB Size Update Timer ---

    async def on_unmount(self) -> None:
        """Clean up logging resources on application exit."""
        logging.info("--- App Unmounting ---")
        self._ui_ready = False
        if self._rich_log_handler: # Ensure it's removed if it exists
            logging.getLogger().removeHandler(self._rich_log_handler)
            logging.info("RichLogHandler removed.")

        # Stop DB size update timer on unmount as well, if not already handled by shutdown_request
        if self._db_size_update_timer: # Removed .is_running check
            self._db_size_update_timer.stop()
            self.loguru_logger.info("DB size update timer stopped during unmount.")

        # Find and remove file handler (more robustly)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                try:
                    handler.close()
                    root_logger.removeHandler(handler)
                    logging.info("RotatingFileHandler removed and closed.")
                except Exception as e_fh_close:
                    logging.error(f"Error removing/closing file handler: {e_fh_close}")
        logging.shutdown()
        self.loguru_logger.info("--- App Unmounted (Loguru) ---")

    ########################################################################
    #
    # WATCHER - Handles UI changes when current_tab's VALUE changes
    #
    # ######################################################################
    def watch_current_tab(self, old_tab: Optional[str], new_tab: str) -> None:
        """Shows/hides the relevant content window when the tab changes."""
        if not new_tab:  # Skip if empty
            return
        if not self._ui_ready:
            return
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        # (Your existing watcher code is likely fine, just ensure the QueryErrors aren't hiding a problem)
        loguru_logger.debug(f"\n>>> DEBUG: watch_current_tab triggered! Old: '{old_tab}', New: '{new_tab}'")
        if not isinstance(new_tab, str) or not new_tab:
            print(f">>> DEBUG: watch_current_tab: Invalid new_tab '{new_tab!r}', aborting.")
            logging.error(f"Watcher received invalid new_tab value: {new_tab!r}. Aborting tab switch.")
            return
        if old_tab and not isinstance(old_tab, str):
            print(f">>> DEBUG: watch_current_tab: Invalid old_tab '{old_tab!r}', setting to None.")
            logging.warning(f"Watcher received invalid old_tab value: {old_tab!r}.")
            old_tab = None

        logging.debug(f"Watcher: Switching tab from '{old_tab}' to '{new_tab}'")

        # --- Hide Old Tab ---
        if old_tab and old_tab != new_tab:
            try: self.query_one(f"#tab-{old_tab}", Button).remove_class("-active")
            except QueryError: logging.warning(f"Watcher: Could not find old button #tab-{old_tab}")
            try: self.query_one(f"#{old_tab}-window").display = False
            except QueryError: logging.warning(f"Watcher: Could not find old window #{old_tab}-window")

        # Show New Tab UI
        try:
            self.query_one(f"#tab-{new_tab}", Button).add_class("-active")
            new_window = self.query_one(f"#{new_tab}-window")
            new_window.display = True

            # Focus input logic (as in original, adjust if needed)
            if new_tab not in [TAB_LOGS, TAB_STATS]: # Don't focus input on these tabs
                input_to_focus: Optional[Union[TextArea, Input]] = None
                try: input_to_focus = new_window.query_one(TextArea)
                except QueryError:
                    try: input_to_focus = new_window.query_one(Input) # Check for Input if TextArea not found
                    except QueryError: pass # No primary input found

                if input_to_focus:
                    self.set_timer(0.1, input_to_focus.focus) # Slight delay for focus
                    logging.debug(f"Watcher: Scheduled focus for input in '{new_tab}'")
                else:
                    logging.debug(f"Watcher: No primary input (TextArea or Input) found to focus in '{new_tab}'")
        except QueryError:
            logging.error(f"Watcher: Could not find new button or window for #tab-{new_tab} / #{new_tab}-window")
        except Exception as e_show_new:
            logging.error(f"Watcher: Error showing new tab '{new_tab}': {e_show_new}", exc_info=True)

        loguru_logger.debug(">>> DEBUG: watch_current_tab finished.")

        # Tab-specific actions on switch
        if new_tab == TAB_CHAT:
            # If chat tab becomes active, maybe re-focus chat input
            try: self.query_one("#chat-input", TextArea).focus()
            except QueryError: pass
            # Add this line to populate prompts when chat tab is opened:
            # Use call_after_refresh for async functions to ensure proper execution
            self.call_after_refresh(chat_handlers.handle_chat_sidebar_prompt_search_changed, self, "") # Call with empty search term
            self.call_after_refresh(chat_handlers._populate_chat_character_search_list, self) # Populate character list
        elif new_tab == TAB_CCP:
            # Initial population for CCP tab when switched to
            # Use call_after_refresh for async functions to ensure proper execution
            self.call_after_refresh(ccp_handlers.populate_ccp_character_select, self)
            self.call_after_refresh(ccp_handlers.populate_ccp_prompts_list_view, self)
            self.call_after_refresh(ccp_handlers.perform_ccp_conversation_search, self) # Initial search/list for conversations
        elif new_tab == TAB_NOTES:
            # Use call_after_refresh for async function
            self.call_after_refresh(notes_handlers.load_and_display_notes_handler, self)
        elif new_tab == TAB_MEDIA:
            try:
                media_window = self.query_one(MediaWindow)
                media_window.activate_initial_view()
            except QueryError:
                self.loguru_logger.error("Could not find MediaWindow to activate its initial view.")
        elif new_tab == TAB_SEARCH:
            if not self.search_active_sub_tab: # If no sub-tab is active yet for Search tab
                self.loguru_logger.debug(f"Switched to Search tab, activating initial sub-tab view: {self._initial_search_sub_tab_view}")
                # Use call_later to ensure the UI for SearchWindow is fully composed and ready
                self.call_later(setattr, self, 'search_active_sub_tab', self._initial_search_sub_tab_view)

            # Explicitly initialize the SearchWindow views
            try:
                search_window = self.query_one(SearchWindow)
                self.call_after_refresh(search_window._initialize_embeddings_creation_view)
                self.call_after_refresh(search_window._refresh_collections_list)
                self.loguru_logger.debug("Initialized SearchWindow views after tab activation")
            except Exception as e:
                self.loguru_logger.error(f"Failed to initialize SearchWindow views: {e}", exc_info=True)
        elif new_tab == TAB_INGEST:
            if not self.ingest_active_view:
                self.loguru_logger.debug(
                    f"Switched to Ingest tab, activating initial view: {self._initial_ingest_view}") # Reverted to original debug log
                # Use call_later to ensure the UI has settled after tab switch before changing sub-view
                self.call_later(self._activate_initial_ingest_view)
        elif new_tab == TAB_TOOLS_SETTINGS:
            if not self.tools_settings_active_view:
                self.loguru_logger.debug(
                    f"Switched to Tools & Settings tab, activating initial view: {self._initial_tools_settings_view}")
                self.call_later(setattr, self, 'tools_settings_active_view', self._initial_tools_settings_view)
        elif new_tab == TAB_LLM:  # New elif block for LLM tab
            if not self.llm_active_view:  # If no view is active yet
                self.loguru_logger.debug(
                    f"Switched to LLM Management tab, activating initial view: {self._initial_llm_view}")
                self.call_later(setattr, self, 'llm_active_view', self._initial_llm_view)
        elif new_tab == TAB_EVALS: # Added for Evals tab
            # Placeholder for any specific actions when Evals tab is selected
            # For example, if EvalsWindow has sub-views or needs initial data loading:
            # if not self.evals_active_view: # Assuming an 'evals_active_view' reactive
            #     self.loguru_logger.debug(f"Switched to Evals tab, activating initial view...")
            #     self.call_later(setattr, self, 'evals_active_view', self._initial_evals_view) # Example
            self.loguru_logger.debug(f"Switched to Evals tab. Initial sidebar state: collapsed={self.evals_sidebar_collapsed}")


    async def _activate_initial_ingest_view(self) -> None:
        self.loguru_logger.info("Attempting to activate initial ingest view via _activate_initial_ingest_view.")
        if not self.ingest_active_view: # Check if it hasn't been set by some other means already
            self.loguru_logger.debug(f"Setting ingest_active_view to initial: {self._initial_ingest_view}")
            self.ingest_active_view = self._initial_ingest_view
        else:
            self.loguru_logger.debug(f"Ingest active view already set to '{self.ingest_active_view}'. No change made by _activate_initial_ingest_view.")


    # Watchers for sidebar collapsed states (keep as is)
    def watch_chat_sidebar_collapsed(self, collapsed: bool) -> None:
        if not self._ui_ready: # Keep the UI ready guard
            self.loguru_logger.debug("watch_chat_sidebar_collapsed: UI not ready.")
            return
        try:
            # Query for the new ID
            sidebar = self.query_one("#chat-left-sidebar") # <<< CHANGE THIS LINE
            sidebar.display = not collapsed # True = visible, False = hidden
            self.loguru_logger.debug(f"Chat left sidebar (#chat-left-sidebar) display set to {not collapsed}")
        except QueryError:
            # Update the error message to reflect the new ID
            self.loguru_logger.error("Chat left sidebar (#chat-left-sidebar) not found by watcher.") # <<< UPDATE ERROR MSG
        except Exception as e:
            self.loguru_logger.error(f"Error toggling chat left sidebar: {e}", exc_info=True)

    def watch_chat_right_sidebar_collapsed(self, collapsed: bool) -> None:
        """Hide or show the character settings sidebar."""
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        try:
            sidebar = self.query_one("#chat-right-sidebar")  # ID from create_chat_right_sidebar
            sidebar.display = not collapsed
        except QueryError:
            logging.error("Character sidebar widget (#chat-right-sidebar) not found.")

    def watch_notes_sidebar_left_collapsed(self, collapsed: bool) -> None:
        """Hide or show the notes left sidebar."""
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        try:
            sidebar = self.query_one("#notes-sidebar-left", NotesSidebarLeft)
            sidebar.display = not collapsed
            # Optional: adjust layout of notes-main-content if needed
        except QueryError:
            logging.error("Notes left sidebar widget (#notes-sidebar-left) not found.")

    def watch_notes_sidebar_right_collapsed(self, collapsed: bool) -> None:
        """Hide or show the notes right sidebar."""
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        try:
            sidebar = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            sidebar.display = not collapsed
            # Optional: adjust layout of notes-main-content if needed
        except QueryError:
            logging.error("Notes right sidebar widget (#notes-sidebar-right) not found.")

    def watch_conv_char_sidebar_left_collapsed(self, collapsed: bool) -> None:
        """Hide or show the Conversations, Characters & Prompts left sidebar pane."""
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        try:
            sidebar_pane = self.query_one("#conv-char-left-pane") # The ID of the VerticalScroll
            sidebar_pane.display = not collapsed # True means visible, False means hidden
            logging.debug(f"Conversations, Characters & Prompts left pane display set to {not collapsed}")
        except QueryError:
            logging.error("Conversations, Characters & Prompts left sidebar pane (#conv-char-left-pane) not found.")
        except Exception as e:
            logging.error(f"Error toggling Conversations, Characters & Prompts left sidebar pane: {e}", exc_info=True)

    def watch_evals_sidebar_collapsed(self, collapsed: bool) -> None:
        """Hide or show the Evals sidebar."""
        if not self._ui_ready:
            self.loguru_logger.debug("watch_evals_sidebar_collapsed: UI not ready.")
            return
        try:
            sidebar = self.query_one("#evals-nav-pane") # Updated ID from new EvalsWindow implementation
            toggle_button = self.query_one("#evals-sidebar-toggle-button")
            sidebar.set_class(collapsed, "collapsed")
            toggle_button.set_class(collapsed, "collapsed")
            self.loguru_logger.debug(f"Evals sidebar collapsed state: {collapsed}, class set/removed.")
        except QueryError:
            self.loguru_logger.error("Evals sidebar (#evals-nav-pane) not found for collapse toggle.")
        except Exception as e:
            self.loguru_logger.error(f"Error toggling Evals sidebar: {e}", exc_info=True)

    def show_ingest_view(self, view_id_to_show: Optional[str]):
        """
        Shows the specified ingest view within the ingest-content-pane and hides others.
        If view_id_to_show is None, hides all ingest views.
        """
        self.log.debug(f"Attempting to show ingest view: {view_id_to_show}")
        try:
            ingest_content_pane = self.query_one("#ingest-content-pane")
            if view_id_to_show:
                ingest_content_pane.display = True
        except QueryError:
            self.log.error("#ingest-content-pane not found. Cannot manage ingest views.")
            return

        for view_id in self.ALL_INGEST_VIEW_IDS:
            try:
                view_container = self.query_one(f"#{view_id}")
                is_target = (view_id == view_id_to_show)
                view_container.display = is_target
                if is_target:
                    self.log.info(f"Displaying ingest view: #{view_id}")
            except QueryError:
                self.log.warning(f"Ingest view container '#{view_id}' not found during show_ingest_view.")

    async def save_current_note(self) -> bool:
        """Saves the currently selected note's title and content to the database."""
        if not self.notes_service or not self.current_selected_note_id or self.current_selected_note_version is None:
            logging.warning("No note selected or service unavailable. Cannot save.")
            # Optionally: self.notify("No note selected to save.", severity="warning")
            return False

        try:
            editor = self.query_one("#notes-editor-area", TextArea)
            title_input = self.query_one("#notes-title-input", Input)
            current_content = editor.text
            current_title = title_input.value

            # Check if title or content actually changed to avoid unnecessary saves.
            # This requires storing the original loaded title/content if not already done by reactives.
            # For now, we save unconditionally if a note is selected.
            # A more advanced check could compare with self.current_selected_note_title and self.current_selected_note_content

            logging.info(
                f"Attempting to save note ID: {self.current_selected_note_id}, Version: {self.current_selected_note_version}")
            success = self.notes_service.update_note(
                user_id=self.notes_user_id,
                note_id=self.current_selected_note_id,
                update_data={'title': current_title, 'content': current_content},
                expected_version=self.current_selected_note_version
            )
            if success:
                logging.info(f"Note {self.current_selected_note_id} saved successfully.")
                # Update version and potentially title/content reactive vars if update_note returns new state
                # For now, we re-fetch to get the new version.
                updated_note_details = self.notes_service.get_note_by_id(
                    user_id=self.notes_user_id,
                    note_id=self.current_selected_note_id
                )
                if updated_note_details:
                    self.current_selected_note_version = updated_note_details.get('version')
                    self.current_selected_note_title = updated_note_details.get('title')  # Update reactive
                    # self.current_selected_note_content = updated_note_details.get('content') # Update reactive

                    # Refresh the list in the left sidebar to reflect title changes and update item version
                    await notes_handlers.load_and_display_notes_handler(self)
                    # self.notify("Note saved!", severity="information") # If notifications are setup
                else:
                    # Note might have been deleted by another client after our successful update, though unlikely.
                    logging.warning(f"Note {self.current_selected_note_id} not found after presumably successful save.")
                    # self.notify("Note saved, but failed to refresh details.", severity="warning")

                return True
            else:
                # This path should ideally not be reached if update_note raises exceptions on failure.
                logging.warning(
                    f"notes_service.update_note for {self.current_selected_note_id} returned False without raising error.")
                # self.notify("Failed to save note (unknown reason).", severity="error")
                return False

        except ConflictError as e:
            logging.error(f"Conflict saving note {self.current_selected_note_id}: {e}", exc_info=True)
            # self.notify(f"Save conflict: {e}. Please reload the note.", severity="error")
            # Optionally, offer to reload the note or overwrite. For now, just log.
            # await self.handle_save_conflict() # A new method to manage this
            return False
        except CharactersRAGDBError as e:
            logging.error(f"Database error saving note {self.current_selected_note_id}: {e}", exc_info=True)
            # self.notify("Error saving note to database.", severity="error")
            return False
        except QueryError as e:
            logging.error(f"UI component not found while saving note: {e}", exc_info=True)
            # self.notify("UI error while saving note.", severity="error")
            return False
        except Exception as e:
            logging.error(f"Unexpected error saving note {self.current_selected_note_id}: {e}", exc_info=True)
            # self.notify("Unexpected error saving note.", severity="error")
            return False


    #######################################################################
    # --- Notes UI Event Handlers (Chat Tab Sidebar) ---
    #######################################################################
    @on(Button.Pressed, "#chat-notes-create-new-button")
    async def handle_chat_notes_create_new(self, event: Button.Pressed) -> None:
        """Handles the 'Create New Note' button press in the chat sidebar's notes section."""
        self.loguru_logger.info(f"Attempting to create new note for user: {self.notes_user_id}")
        default_title = "New Note"
        default_content = ""

        if not self.notes_service:
            self.notify("Notes service is not available.", severity="error")
            self.loguru_logger.error("Notes service not available in handle_chat_notes_create_new.")
            return

        try:
            # 1. Call self.notes_service.add_note
            new_note_id = self.notes_service.add_note(
                user_id=self.notes_user_id,
                title=default_title,
                content=default_content,
                # keywords, parent_id, etc., can be added if needed
            )

            if new_note_id:
                # 2. Store Note ID and Version
                self.current_chat_note_id = new_note_id
                self.current_chat_note_version = 1  # Assuming version starts at 1
                self.loguru_logger.info(f"New note created with ID: {new_note_id}, Version: {self.current_chat_note_version}")

                # 3. Update UI Input Fields
                title_input = self.query_one("#chat-notes-title-input", Input)
                content_textarea = self.query_one("#chat-notes-content-textarea", TextArea)
                title_input.value = default_title
                content_textarea.text = default_content

                # 4. Add to ListView
                notes_list_view = self.query_one("#chat-notes-listview", ListView)
                new_list_item = ListItem(Label(default_title))
                new_list_item.id = f"note-item-{new_note_id}" # Ensure unique DOM ID for the ListItem itself
                # We'll need a way to store the actual note_id on the ListItem for retrieval,
                # Textual's ListItem doesn't have a direct `data` attribute.
                # A common pattern is to use a custom ListItem subclass or manage a mapping.
                # For now, we can set the DOM ID and parse it, or use a custom attribute if we make one.
                # setattr(new_list_item, "_note_id", new_note_id) # Example of custom attribute
                # Or, more simply for now, we can rely on on_chat_notes_collapsible_toggle to refresh the whole list
                # which will then pick up the new note from the DB.
                # For immediate feedback without full list refresh:
                await notes_list_view.prepend(new_list_item) # Prepend to show at top

                # 5. Set Focus
                title_input.focus()

                self.notify("New note created in sidebar.", severity="information")
            else:
                self.notify("Failed to create new note (no ID returned).", severity="error")
                self.loguru_logger.error("notes_service.add_note did not return a new_note_id.")

        except CharactersRAGDBError as e: # Specific DB error
            self.loguru_logger.error(f"Database error creating new note: {e}", exc_info=True)
            self.notify(f"DB error creating note: {e}", severity="error")
        except Exception as e: # Catch-all for other unexpected errors
            self.loguru_logger.error(f"Unexpected error creating new note: {e}", exc_info=True)
            self.notify(f"Error creating note: {type(e).__name__}", severity="error")

    @on(Button.Pressed, "#chat-notes-search-button")
    async def handle_chat_notes_search(self, event: Button.Pressed) -> None:
        """Handles the 'Search' button press in the chat sidebar's notes section."""
        self.loguru_logger.info(f"Search Notes button pressed. User ID: {self.notes_user_id}")

        if not self.notes_service:
            self.notify("Notes service is not available.", severity="error")
            self.loguru_logger.error("Notes service not available in handle_chat_notes_search.")
            return

        try:
            search_input = self.query_one("#chat-notes-search-input", Input)
            search_term = search_input.value.strip()

            notes_list_view = self.query_one("#chat-notes-listview", ListView)
            await notes_list_view.clear()

            listed_notes: List[Dict[str, Any]] = []
            limit = 50

            if not search_term:
                self.loguru_logger.info("Empty search term, listing all notes.")
                listed_notes = self.notes_service.list_notes(user_id=self.notes_user_id, limit=limit)
            else:
                self.loguru_logger.info(f"Searching notes with term: '{search_term}'")
                listed_notes = self.notes_service.search_notes(user_id=self.notes_user_id, search_term=search_term, limit=limit)

            if listed_notes:
                for note in listed_notes:
                    note_title = note.get('title', 'Untitled Note')
                    note_id = note.get('id')
                    if not note_id:
                        self.loguru_logger.warning(f"Note found without an ID during search: {note_title}. Skipping.")
                        continue

                    list_item_label = Label(note_title)
                    new_list_item = ListItem(list_item_label)
                    new_list_item.id = f"note-item-{note_id}"
                    # setattr(new_list_item, "_note_data", note) # If needed later for load
                    await notes_list_view.append(new_list_item)

                self.notify(f"{len(listed_notes)} notes found.", severity="information")
                self.loguru_logger.info(f"Populated notes list with {len(listed_notes)} search results.")
                self.loguru_logger.debug(f"ListView child count after search population: {notes_list_view.child_count}") # Added log
            else:
                msg = "No notes match your search." if search_term else "No notes found."
                self.notify(msg, severity="information")
                self.loguru_logger.info(msg)

        except CharactersRAGDBError as e:
            self.loguru_logger.error(f"Database error searching notes: {e}", exc_info=True)
            self.notify(f"DB error searching notes: {e}", severity="error")
        except QueryError as e_query:
            self.loguru_logger.error(f"UI element not found during notes search: {e_query}", exc_info=True)
            self.notify("UI error during notes search.", severity="error")
        except Exception as e:
            self.loguru_logger.error(f"Unexpected error searching notes: {e}", exc_info=True)
            self.notify(f"Error searching notes: {type(e).__name__}", severity="error")

    @on(Button.Pressed, "#chat-notes-load-button")
    async def handle_chat_notes_load(self, event: Button.Pressed) -> None:
        """Handles the 'Load Note' button press in the chat sidebar's notes section."""
        self.loguru_logger.info(f"Load Note button pressed. User ID: {self.notes_user_id}")

        if not self.notes_service:
            self.notify("Notes service is not available.", severity="error")
            self.loguru_logger.error("Notes service not available in handle_chat_notes_load.")
            return

        try:
            notes_list_view = self.query_one("#chat-notes-listview", ListView)
            selected_list_item = notes_list_view.highlighted_child

            if selected_list_item is None or not selected_list_item.id:
                self.notify("Please select a note to load.", severity="warning")
                return

            # Extract actual_note_id from the ListItem's DOM ID
            dom_id_parts = selected_list_item.id.split("note-item-")
            if len(dom_id_parts) < 2 or not dom_id_parts[1]:
                self.notify("Selected item has an invalid ID format.", severity="error")
                self.loguru_logger.error(f"Invalid ListItem ID format: {selected_list_item.id}")
                return

            actual_note_id = dom_id_parts[1]
            self.loguru_logger.info(f"Attempting to load note with ID: {actual_note_id}")

            note_data = self.notes_service.get_note_by_id(user_id=self.notes_user_id, note_id=actual_note_id)

            if note_data:
                title_input = self.query_one("#chat-notes-title-input", Input)
                content_textarea = self.query_one("#chat-notes-content-textarea", TextArea)

                loaded_title = note_data.get('title', '')
                loaded_content = note_data.get('content', '')
                loaded_version = note_data.get('version')
                loaded_id = note_data.get('id')

                title_input.value = loaded_title
                content_textarea.text = loaded_content

                self.current_chat_note_id = loaded_id
                self.current_chat_note_version = loaded_version

                self.notify(f"Note '{loaded_title}' loaded.", severity="information")
                self.loguru_logger.info(f"Note '{loaded_title}' (ID: {loaded_id}, Version: {loaded_version}) loaded into UI.")
            else:
                self.notify(f"Could not load note (ID: {actual_note_id}). It might have been deleted.", severity="warning")
                self.loguru_logger.warning(f"Note with ID {actual_note_id} not found by service.")
                # Clear fields if note not found
                self.query_one("#chat-notes-title-input", Input).value = ""
                self.query_one("#chat-notes-content-textarea", TextArea).text = ""
                self.current_chat_note_id = None
                self.current_chat_note_version = None

        except CharactersRAGDBError as e_db:
            self.loguru_logger.error(f"Database error loading note: {e_db}", exc_info=True)
            self.notify(f"DB error loading note: {e_db}", severity="error")
        except QueryError as e_query:
            self.loguru_logger.error(f"UI element not found during note load: {e_query}", exc_info=True)
            self.notify("UI error during note load.", severity="error")
        except Exception as e:
            self.loguru_logger.error(f"Unexpected error loading note: {e}", exc_info=True)
            self.notify(f"Error loading note: {type(e).__name__}", severity="error")

    @on(Button.Pressed, "#chat-notes-save-button")
    async def handle_chat_notes_save(self, event: Button.Pressed) -> None:
        """Handles the 'Save Note' button press in the chat sidebar's notes section."""
        self.loguru_logger.info(f"Save Note button pressed. User ID: {self.notes_user_id}")

        if not self.notes_service:
            self.notify("Notes service is not available.", severity="error")
            self.loguru_logger.error("Notes service not available in handle_chat_notes_save.")
            return

        if not self.current_chat_note_id or self.current_chat_note_version is None:
            self.notify("No active note to save. Load or create a note first.", severity="warning")
            self.loguru_logger.warning("handle_chat_notes_save called without an active note_id or version.")
            return

        try:
            title_input = self.query_one("#chat-notes-title-input", Input)
            content_textarea = self.query_one("#chat-notes-content-textarea", TextArea)

            title = title_input.value
            content = content_textarea.text

            update_data = {"title": title, "content": content}

            self.loguru_logger.info(f"Attempting to save note ID: {self.current_chat_note_id}, Version: {self.current_chat_note_version}")

            success = self.notes_service.update_note(
                user_id=self.notes_user_id,
                note_id=self.current_chat_note_id,
                update_data=update_data,
                expected_version=self.current_chat_note_version
            )

            if success: # Should be true if no exception was raised by DB layer for non-Conflict errors
                self.current_chat_note_version += 1
                self.notify("Note saved successfully.", severity="information")
                self.loguru_logger.info(f"Note {self.current_chat_note_id} saved. New version: {self.current_chat_note_version}")

                # Update ListView item
                try:
                    notes_list_view = self.query_one("#chat-notes-listview", ListView)
                    # Find the specific ListItem to update its Label
                    # This requires iterating or querying if the ListItem's DOM ID is known
                    item_dom_id = f"note-item-{self.current_chat_note_id}"
                    for item in notes_list_view.children:
                        if isinstance(item, ListItem) and item.id == item_dom_id:
                            # Assuming the first child of ListItem is the Label we want to update
                            label_to_update = item.query_one(Label)
                            label_to_update.update(title) # Update with the new title
                            self.loguru_logger.debug(f"Updated title in ListView for note ID {self.current_chat_note_id} to '{title}'")
                            break
                        else:
                            self.loguru_logger.debug(f"ListItem with ID {item_dom_id} not found for title update after save (iterated item ID: {item.id}).")
                except QueryError as e_lv_update:
                    self.loguru_logger.error(f"Error querying Label within ListView item to update title: {e_lv_update}")
                except Exception as e_item_update: # Catch other errors during list item update
                    self.loguru_logger.error(f"Unexpected error updating list item title: {e_item_update}", exc_info=True)
            else:
                # This case might not be hit if service raises exceptions for all failures
                self.notify("Failed to save note. Reason unknown.", severity="error")
                self.loguru_logger.error(f"notes_service.update_note returned False for note {self.current_chat_note_id}")

        except ConflictError:
            self.loguru_logger.warning(f"Save conflict for note {self.current_chat_note_id}. Expected version: {self.current_chat_note_version}")
            self.notify("Save conflict: Note was modified elsewhere. Please reload and reapply changes.", severity="error", timeout=10)
        except CharactersRAGDBError as e_db:
            self.loguru_logger.error(f"Database error saving note {self.current_chat_note_id}: {e_db}", exc_info=True)
            self.notify(f"DB error saving note: {e_db}", severity="error")
        except QueryError as e_query:
            self.loguru_logger.error(f"UI element not found during note save: {e_query}", exc_info=True)
            self.notify("UI error during note save.", severity="error")
        except Exception as e:
            self.loguru_logger.error(f"Unexpected error saving note {self.current_chat_note_id}: {e}", exc_info=True)
            self.notify(f"Error saving note: {type(e).__name__}", severity="error")

    @on(Collapsible.Toggled, "#chat-notes-collapsible")
    async def on_chat_notes_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Handles the expansion/collapse of the Notes collapsible section in the chat sidebar."""
        if not event.collapsible.collapsed:  # If the collapsible was just expanded
            self.loguru_logger.info(f"Notes collapsible opened in chat sidebar. User ID: {self.notes_user_id}. Refreshing list.")

            if not self.notes_service:
                self.notify("Notes service is not available.", severity="error")
                self.loguru_logger.error("Notes service not available in on_chat_notes_collapsible_toggle.")
                return

    @on(Collapsible.Toggled, "#chat-active-character-info-collapsible")
    async def on_chat_active_character_info_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Handles the expansion/collapse of the Active Character Info collapsible section in the chat sidebar."""
        if not event.collapsible.collapsed:  # If the collapsible was just expanded
            self.loguru_logger.info("Active Character Info collapsible opened in chat sidebar. Refreshing character list.")

            # Call the function to populate the character list
            from tldw_chatbook.Event_Handlers.Chat_Events import chat_events
            await chat_events._populate_chat_character_search_list(self)

            try:
                # 1. Clear ListView
                notes_list_view = self.query_one("#chat-notes-listview", ListView)
                await notes_list_view.clear()

                # 2. Call self.notes_service.list_notes
                # Limit to a reasonable number, e.g., 50, most recent first if service supports sorting
                listed_notes = self.notes_service.list_notes(user_id=self.notes_user_id, limit=50)

                # 3. Populate ListView
                if listed_notes:
                    for note in listed_notes:
                        note_title = note.get('title', 'Untitled Note')
                        note_id = note.get('id')
                        if not note_id:
                            self.loguru_logger.warning(f"Note found without an ID: {note_title}. Skipping.")
                            continue

                        list_item_label = Label(note_title)
                        new_list_item = ListItem(list_item_label)
                        # Store the actual note_id on the ListItem for retrieval.
                        # Using a unique DOM ID for the ListItem itself.
                        new_list_item.id = f"note-item-{note_id}"
                        # A custom attribute to store data:
                        # setattr(new_list_item, "_note_data", note) # Store whole note or just id/version

                        await notes_list_view.append(new_list_item)
                    self.notify("Notes list refreshed.", severity="information")
                    self.loguru_logger.info(f"Populated notes list with {len(listed_notes)} items.")
                else:
                    self.notify("No notes found.", severity="information")
                    self.loguru_logger.info("No notes found for user after refresh.")

            except CharactersRAGDBError as e: # Specific DB error
                self.loguru_logger.error(f"Database error listing notes: {e}", exc_info=True)
                self.notify(f"DB error listing notes: {e}", severity="error")
            except QueryError as e_query: # If UI elements are not found
                 self.loguru_logger.error(f"UI element not found in notes toggle: {e_query}", exc_info=True)
                 self.notify("UI error while refreshing notes.", severity="error")
            except Exception as e: # Catch-all for other unexpected errors
                self.loguru_logger.error(f"Unexpected error listing notes: {e}", exc_info=True)
                self.notify(f"Error listing notes: {type(e).__name__}", severity="error")
        else:
            self.loguru_logger.info("Notes collapsible closed in chat sidebar.")

    @on(Collapsible.Toggled, "#chat-active-character-info-collapsible")
    async def on_chat_active_character_info_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Handles the expansion/collapse of the Active Character Info collapsible section in the chat sidebar."""
        if not event.collapsible.collapsed:  # If the collapsible was just expanded
            self.loguru_logger.info("Active Character Info collapsible opened in chat sidebar. Refreshing character list.")

            # Call the function to populate the character list
            from tldw_chatbook.Event_Handlers.Chat_Events import chat_events
            await chat_events._populate_chat_character_search_list(self)
        else:
            self.loguru_logger.info("Active Character Info collapsible closed in chat sidebar.")

    ########################################################################
    #
    # --- EVENT DISPATCHERS ---
    #
    ########################################################################
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatches button presses to the appropriate event handler using a map."""
        button_id = event.button.id
        if not button_id:
            return

        self.loguru_logger.debug(f"Button pressed: ID='{button_id}' on Tab='{self.current_tab}'")

        # 1. Handle global tab switching first
        if button_id.startswith("tab-"):
            await tab_events.handle_tab_button_pressed(self, event)
            return

        # 2. Try to delegate to the appropriate window component
        try:
            # Determine which window component should handle this button press based on current tab
            window_id_map = {
                TAB_CHAT: "chat-window",
                TAB_CCP: "conversations_characters_prompts-window",
                TAB_NOTES: "notes-window",
                TAB_MEDIA: "media-window",
                TAB_SEARCH: "search-window",
                TAB_INGEST: "ingest-window",
                TAB_TOOLS_SETTINGS: "tools_settings-window",
                TAB_LLM: "llm_management-window",
                TAB_LOGS: "logs-window",
                TAB_STATS: "stats-window",
                TAB_EVALS: "evals-window",
                TAB_CODING: "coding-window"
            }

            window_id = window_id_map.get(self.current_tab)
            if window_id:
                window = self.query_one(f"#{window_id}")
                # Check if the window has an on_button_pressed method
                if hasattr(window, "on_button_pressed") and callable(window.on_button_pressed):
                    window.on_button_pressed(event)
                    # Check if event has been stopped (some event types don't have is_stopped)
                    if hasattr(event, 'is_stopped') and event.is_stopped:
                        return

        except QueryError:
            self.loguru_logger.error(f"Could not find window component for tab '{self.current_tab}'")
        except Exception as e:
            self.loguru_logger.error(f"Error delegating button press to window component: {e}", exc_info=True)

        # 3. Use the handler map for buttons not handled by window components
        current_tab_handlers = self.button_handler_map.get(self.current_tab, {})
        handler = current_tab_handlers.get(button_id)

        if handler:
            if callable(handler):
                # Call the handler, which is expected to return a coroutine (an awaitable object).
                result = handler(self, event)

                # Check if the result is indeed awaitable before awaiting it.
                # This makes the code more robust and satisfies static type checkers.
                if inspect.isawaitable(result):
                    await result
                else:
                    self.loguru_logger.warning(
                        f"Handler for button '{button_id}' did not return an awaitable object."
                    )
            else:
                self.loguru_logger.error(f"Handler for button '{button_id}' is not callable: {handler}")
            return  # The button press was handled (or an error occurred).

        # 4. Fallback for unmapped buttons
        self.loguru_logger.warning(f"Unhandled button press for ID '{button_id}' on tab '{self.current_tab}'.")

    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handles text area changes, e.g., for live updates to character data."""
        control_id = event.control.id
        current_active_tab = self.current_tab

        if current_active_tab == TAB_CHAT and control_id and control_id.startswith("chat-character-"):
            # Ensure it's one of the actual attribute TextAreas, not something else
            if control_id in [
                "chat-character-description-edit",
                "chat-character-personality-edit",
                "chat-character-scenario-edit",
                "chat-character-system-prompt-edit",
                "chat-character-first-message-edit"
            ]:
                await chat_handlers.handle_chat_character_attribute_changed(self, event)

    def _update_model_download_log(self, message: str) -> None:
        """Helper to write messages to the model download log widget."""
        try:
            # This ID must match the RichLog widget in your "Download Models" view
            log_widget = self.query_one("#model-download-log-output", RichLog)
            log_widget.write(message)
        except QueryError:
            self.loguru_logger.error("Failed to query #model-download-log-output to write message.")
        except Exception as e:
            self.loguru_logger.error(f"Error writing to model download log: {e}", exc_info=True)

    def _update_mlx_log(self, message: str) -> None:
        """Helper to write messages to the MLX-LM log widget."""
        try:
            log_widget = self.query_one("#mlx-log-output", RichLog)
            log_widget.write(message)
        except QueryError:
            self.loguru_logger.error("Failed to query #mlx-log-output to write message.")
        except Exception as e:
            self.loguru_logger.error(f"Error writing to MLX-LM log: {e}", exc_info=True)

    async def on_input_changed(self, event: Input.Changed) -> None:
        input_id = event.input.id
        current_active_tab = self.current_tab
        # --- Notes Search ---
        if input_id == "notes-search-input" and current_active_tab == TAB_NOTES: # Changed from elif to if
            await notes_handlers.handle_notes_search_input_changed(self, event.value)
        # --- Chat Sidebar Conversation Search ---
        elif input_id == "chat-conversation-search-bar" and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_conversation_search_bar_changed(self, event.value)
        elif input_id == "conv-char-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_conversation_search_input_changed(self, event)
        elif input_id == "ccp-prompt-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_prompt_search_input_changed(self, event)
        elif input_id == "chat-prompt-search-input" and current_active_tab == TAB_CHAT: # New condition
            if self._chat_sidebar_prompt_search_timer: # Use the new timer variable
                self._chat_sidebar_prompt_search_timer.stop()
            self._chat_sidebar_prompt_search_timer = self.set_timer(
                0.5,
                lambda: chat_handlers.handle_chat_sidebar_prompt_search_changed(self, event.value.strip())
            )
        elif input_id == "chat-character-search-input" and current_active_tab == TAB_CHAT:
            # No debouncer here, direct call as per existing handler
            await chat_handlers.handle_chat_character_search_input_changed(self, event)
        elif input_id == "chat-character-name-edit" and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_character_attribute_changed(self, event)
        elif input_id == "chat-template-search-input" and current_active_tab == TAB_CHAT:
            # No debouncer here, direct call for template search
            await chat_handlers.handle_chat_template_search_input_changed(self, event.value)
        # --- Chat Tab Media Search Input ---
        # elif input_id == "chat-media-search-input" and current_active_tab == TAB_CHAT:
        #     await handle_chat_media_search_input_changed(self, event.input)
        # Add more specific input handlers if needed, e.g., for title inputs if they need live validation/reaction

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        list_view_id = event.list_view.id
        current_active_tab = self.current_tab
        item_details = f"Item prompt_id: {getattr(event.item, 'prompt_id', 'N/A')}, Item prompt_uuid: {getattr(event.item, 'prompt_uuid', 'N/A')}"
        self.loguru_logger.info(
            f"ListView.Selected: list_view_id='{list_view_id}', current_tab='{current_active_tab}', {item_details}"
        )

        if list_view_id and list_view_id.startswith("media-list-view-") and current_active_tab == TAB_MEDIA:
            self.loguru_logger.debug("Dispatching to media_events.handle_media_list_item_selected")
            await media_events.handle_media_list_item_selected(self, event)

        elif list_view_id == "notes-list-view" and current_active_tab == TAB_NOTES:
            self.loguru_logger.debug("Dispatching to notes_handlers.handle_notes_list_view_selected")
            await notes_handlers.handle_notes_list_view_selected(self, list_view_id, event.item)

        elif list_view_id == "ccp-prompts-listview" and current_active_tab == TAB_CCP:
            self.loguru_logger.debug("Dispatching to ccp_handlers.handle_ccp_prompts_list_view_selected")
            await ccp_handlers.handle_ccp_prompts_list_view_selected(self, list_view_id, event.item)

        elif list_view_id == "chat-sidebar-prompts-listview" and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug("Dispatching to chat_handlers.handle_chat_sidebar_prompts_list_view_selected")
            await ccp_handlers.handle_ccp_prompts_list_view_selected(self, list_view_id, event.item)

        elif list_view_id == "chat-media-search-results-listview" and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug("Dispatching to chat_events_sidebar.handle_media_item_selected")
            await chat_events_sidebar.handle_media_item_selected(self, event.item)

        # Note: chat-conversation-search-results-list and conv-char-search-results-list selections
        # are typically handled by their respective "Load Selected" buttons rather than direct on_list_view_selected.
        else:
            self.loguru_logger.warning(
            f"No specific handler for ListView.Selected from list_view_id='{list_view_id}' on tab='{current_active_tab}'")

    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        checkbox_id = event.checkbox.id
        current_active_tab = self.current_tab

        if checkbox_id.startswith("chat-conversation-search-") and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_search_checkbox_changed(self, checkbox_id, event.value)
        # Add handlers for checkboxes in other tabs if any

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handles changes in Select widgets if specific actions are needed beyond watchers."""
        select_id = event.select.id
        current_active_tab = self.current_tab

        if select_id == "conv-char-character-select" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_character_select_changed(self, event.value)

        current_active_tab = self.current_tab

        if select_id == "conv-char-character-select" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_character_select_changed(self, event.value)
        elif select_id == "tldw-api-auth-method" and current_active_tab == TAB_INGEST:
            await ingest_events.handle_tldw_api_auth_method_changed(self, str(event.value))
        elif select_id == "tldw-api-media-type" and current_active_tab == TAB_INGEST:
            await ingest_events.handle_tldw_api_media_type_changed(self, str(event.value))

    ##################################################################
    # --- Event Handlers for Streaming and Worker State Changes ---
    ##################################################################
    @on(StreamingChunk)
    async def on_streaming_chunk(self, event: StreamingChunk) -> None:
        await handle_streaming_chunk(self, event)

    @on(StreamDone)
    async def on_stream_done(self, event: StreamDone) -> None:
        await handle_stream_done(self, event)

    @on(Checkbox.Changed, "#chat-strip-thinking-tags-checkbox")
    async def handle_strip_thinking_tags_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handles changes to the 'Strip Thinking Tags' checkbox."""
        new_value = event.value
        self.loguru_logger.info(f"'Strip Thinking Tags' checkbox changed to: {new_value}")

        if "chat_defaults" not in self.app_config:
            self.app_config["chat_defaults"] = {}
        self.app_config["chat_defaults"]["strip_thinking_tags"] = new_value

        # Persist the change
        try:
            # save_setting_to_cli_config is not defined, assuming this is a placeholder
            # You would need to implement this function to write to your config file.
            # Example implementation:
            # from .config import save_setting_to_cli_config
            # save_setting_to_cli_config("chat_defaults", "strip_thinking_tags", new_value)
            self.notify(f"Thinking tag stripping {'enabled' if new_value else 'disabled'}.", timeout=2)
        except Exception as e:
            self.loguru_logger.error(f"Failed to save 'strip_thinking_tags' setting: {e}", exc_info=True)
            self.notify("Error saving thinking tag setting.", severity="error", timeout=4)
    #####################################################################
    # --- End of Chat Event Handlers for Streaming & thinking tags ---
    #####################################################################


    #####################################################################
    # --- Event Handlers for Worker State Changes ---
    #####################################################################
    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        worker_name_attr = event.worker.name
        worker_group = event.worker.group
        worker_state = event.state
        worker_description = event.worker.description  # For more informative logging

        self.loguru_logger.debug(
            f"on_worker_state_changed: Worker NameAttr='{worker_name_attr}' (Type: {type(worker_name_attr)}), "
            f"Group='{worker_group}', State='{worker_state}', Desc='{worker_description}'"
        )

        #######################################################################
        # --- Handle Chat-related API Calls ---
        #######################################################################
        if isinstance(worker_name_attr, str) and \
                (worker_name_attr.startswith("API_Call_chat") or
                 worker_name_attr.startswith("API_Call_ccp") or
                 worker_name_attr == "respond_for_me_worker"):

            self.loguru_logger.debug(f"Chat-related worker '{worker_name_attr}' detected. State: {worker_state}.")
            stop_button_id_selector = "#stop-chat-generation"  # Correct ID selector

            if worker_state == WorkerState.RUNNING:
                self.loguru_logger.info(f"Chat-related worker '{worker_name_attr}' is RUNNING.")
                try:
                    # Enable the stop button
                    stop_button_widget = self.query_one(stop_button_id_selector, Button)
                    stop_button_widget.disabled = False
                    self.loguru_logger.info(f"Button '{stop_button_id_selector}' ENABLED.")
                except QueryError:
                    self.loguru_logger.error(f"Could not find button '{stop_button_id_selector}' to enable it.")
                # Note: The original code delegated SUCCESS/ERROR states.
                # RUNNING state for chat workers was not explicitly handled here for the stop button.

            elif worker_state in [WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED]:
                self.loguru_logger.info(f"Chat-related worker '{worker_name_attr}' finished with state {worker_state}.")
                try:
                    # Disable the stop button
                    stop_button_widget = self.query_one(stop_button_id_selector, Button)
                    stop_button_widget.disabled = True
                    self.loguru_logger.info(f"Button '{stop_button_id_selector}' DISABLED.")
                except QueryError:
                    self.loguru_logger.error(f"Could not find button '{stop_button_id_selector}' to disable it.")

                # Existing delegation for SUCCESS/ERROR, which might update UI based on worker result.
                # The worker_handlers.handle_api_call_worker_state_changed should focus on
                # processing the worker's result/error, not managing the stop button's disabled state,
                # as we are now handling it directly here.
                if worker_state == WorkerState.SUCCESS or worker_state == WorkerState.ERROR:
                    self.loguru_logger.debug(
                        f"Delegating state {worker_state} for chat-related worker '{worker_name_attr}' to worker_handlers for result processing."
                    )
                    # This handler is responsible for updating the ChatMessage widget with the final response or error.
                    await worker_handlers.handle_api_call_worker_state_changed(self, event)

                elif worker_state == WorkerState.CANCELLED:
                    self.loguru_logger.info(f"Worker '{worker_name_attr}' was cancelled.")
                    # The StreamDone event (if streaming) or the handle_stop_chat_generation_pressed
                    # (if non-streaming) should handle updating the AI message widget UI.
                    # We've already disabled the stop button above.
                    # If the StreamDone event doesn't appropriately update the current_ai_message_widget display
                    # for cancellations, some finalization logic might be needed here too.
                    # For now, assuming StreamDone or the stop handler manage the message UI.
                    if self.current_ai_message_widget and not self.current_ai_message_widget.generation_complete:
                        self.loguru_logger.debug("Finalizing AI message widget UI due to worker CANCELLED state.")
                        # Attempt to update the message UI to reflect cancellation if not already handled by StreamDone
                        try:
                            static_text_widget = self.current_ai_message_widget.query_one(".message-text", Static)
                            # Check if already updated by handle_stop_chat_generation_pressed for non-streaming
                            if "[italic]Chat generation cancelled by user.[/]" not in self.current_ai_message_widget.message_text:
                                self.current_ai_message_widget.message_text += "\n[italic](Stream Cancelled)[/]"
                                static_text_widget.update(Text.from_markup(self.current_ai_message_widget.message_text))

                            self.current_ai_message_widget.mark_generation_complete()
                            self.current_ai_message_widget = None  # Clear ref
                        except QueryError as qe_cancel_ui:
                            self.loguru_logger.error(f"Error updating AI message UI on CANCELLED state: {qe_cancel_ui}")
            else:
                self.loguru_logger.debug(f"Chat-related worker '{worker_name_attr}' in other state: {worker_state}")

        #######################################################################
        # --- Handle tldw server API Calls Worker (tldw API Ingestion) ---
        #######################################################################
        elif worker_group == "api_calls":
            self.loguru_logger.info(f"TLDW API worker '{event.worker.name}' finished with state {event.state}.")
            if worker_state == WorkerState.SUCCESS:
                await ingest_events.handle_tldw_api_worker_success(self, event)
            elif worker_state == WorkerState.ERROR:
                await ingest_events.handle_tldw_api_worker_failure(self, event)

        #######################################################################
        # --- Handle Ollama API Worker ---
        #######################################################################
        elif worker_group == "ollama_api":
            self.loguru_logger.info(f"Ollama API worker '{event.worker.name}' finished with state {event.state}.")
            await llm_management_events_ollama.handle_ollama_worker_completion(self, event)


        #######################################################################
        # --- Handle Llama.cpp Server Worker (identified by group) ---
        #######################################################################
        # This handles the case where worker_name_attr was a list.
        elif worker_group == "llamacpp_server":
            self.loguru_logger.info(
                f"Llama.cpp server worker (Group: '{worker_group}') state changed to {worker_state}."
            )
            actual_start_button_id = "#llamacpp-start-server-button"
            actual_stop_button_id = "#llamacpp-stop-server-button"

            if worker_state == WorkerState.PENDING:
                self.loguru_logger.debug("Llama.cpp server worker is PENDING.")
                # You might disable the start button here as well, or show a "starting..." status
                try:
                    self.query_one(actual_start_button_id, Button).disabled = True
                    self.query_one(actual_stop_button_id, Button).disabled = True  # Disable stop until fully running
                except QueryError:
                    self.loguru_logger.warning("Could not find Llama.cpp server buttons to update for PENDING state.")

            elif worker_state == WorkerState.RUNNING:
                self.loguru_logger.info("Llama.cpp server worker is RUNNING (subprocess launched).")
                try:
                    self.query_one(actual_start_button_id, Button).disabled = True
                    self.query_one(actual_stop_button_id, Button).disabled = False  # Enable stop when running
                    self.notify("Llama.cpp server process starting...", title="Server Status")
                except QueryError:
                    self.loguru_logger.warning("Could not find Llama.cpp server buttons to update for RUNNING state.")

            elif worker_state == WorkerState.SUCCESS:
                self.loguru_logger.info(f"Llama.cpp server worker finished successfully (Textual worker perspective).")
                # Define result_message and is_actual_server_error *inside* this block
                result_message = str(
                    event.worker.result).strip() if event.worker.result else "Worker completed with no specific result message."
                self.loguru_logger.info(f"Llama.cpp worker result message: '{result_message}'")

                is_actual_server_error = "exited quickly with error code" in result_message.lower() or \
                                         "exited with non-zero code" in result_message.lower() or \
                                         "error:" in result_message.lower()  # Broader check

                if is_actual_server_error:
                    self.notify(f"Llama.cpp server process reported an error. Check logs.", title="Server Status",
                                severity="error", timeout=10)
                elif "exited quickly with code: 0" in result_message.lower():
                    self.notify(
                        f"Llama.cpp server exited quickly (but successfully). Check logs if this was unexpected.",
                        title="Server Status", severity="warning", timeout=10)
                else:
                    self.notify("Llama.cpp server process finished.", title="Server Status")

                # Worker has finished, so the process it was managing should be gone or is now orphaned.
                # The worker's 'finally' block should have cleared self.llamacpp_server_process.
                # We double-check here.
                if self.llamacpp_server_process is not None:
                    self.loguru_logger.warning(f"Llama.cpp worker SUCCEEDED, but app.llamacpp_server_process was not None. Clearing it now.")
                    self.llamacpp_server_process = None

                try:
                    self.query_one(actual_start_button_id, Button).disabled = False
                    self.query_one(actual_stop_button_id, Button).disabled = True
                except QueryError:
                    self.loguru_logger.warning("Could not find Llama.cpp server buttons to update for SUCCESS state.")

            elif worker_state == WorkerState.ERROR:
                self.loguru_logger.error(f"Llama.cpp server worker itself failed with an exception: {event.worker.error}")
                self.notify(f"Llama.cpp worker error: {str(event.worker.error)[:100]}", title="Server Worker Error", severity="error")

                # Worker failed, so the process it was managing might be in an indeterminate state or already gone.
                # The worker's 'finally' block should attempt cleanup and clear self.llamacpp_server_process.
                if self.llamacpp_server_process is not None:
                    self.loguru_logger.warning(f"Llama.cpp worker ERRORED, but app.llamacpp_server_process was not None. Clearing it now.")
                    self.llamacpp_server_process = None

                try:
                    self.query_one(actual_start_button_id, Button).disabled = False
                    self.query_one(actual_stop_button_id, Button).disabled = True
                except QueryError:
                    self.loguru_logger.warning("Could not find Llama.cpp server buttons to update for ERROR state.")


        #######################################################################
        # --- Handle Llamafile Server Worker (identified by group) ---
        #######################################################################
        elif worker_group == "llamafile_server":  # Add this new elif block
            self.loguru_logger.info(
                f"Llamafile server worker (Group: '{worker_group}') state changed to {worker_state}."
            )
            actual_start_button_id = "#llamafile-start-server-button"
            actual_stop_button_id = "#llamafile-stop-server-button"  # Assuming you have one

            if worker_state == WorkerState.PENDING:
                self.loguru_logger.debug("Llamafile server worker is PENDING.")
                try:
                    self.query_one(actual_start_button_id, Button).disabled = True
                    if self.query_one(actual_stop_button_id, Button):  # Check if stop button exists
                        self.query_one(actual_stop_button_id, Button).disabled = True
                except QueryError:
                    self.loguru_logger.warning("Could not find Llamafile server buttons for PENDING state.")

            elif worker_state == WorkerState.RUNNING:
                self.loguru_logger.info("Llamafile server worker is RUNNING.")
                try:
                    self.query_one(actual_start_button_id, Button).disabled = True
                    if self.query_one(actual_stop_button_id, Button):
                        self.query_one(actual_stop_button_id, Button).disabled = False
                    self.notify("Llamafile server process is running.", title="Server Status")
                except QueryError:
                    self.loguru_logger.warning("Could not find Llamafile server buttons for RUNNING state.")

            elif worker_state == WorkerState.SUCCESS:
                self.loguru_logger.info(f"Llamafile server worker finished successfully.")
                result_message = str(event.worker.result).strip() if event.worker.result else "Worker completed."
                self.loguru_logger.info(f"Llamafile worker result: '{result_message}'")

                is_server_error = "non-zero code" in result_message.lower() or "error" in result_message.lower()
                if is_server_error:
                    self.notify(f"Llamafile server process ended with an issue. Check logs.", title="Server Status",
                                severity="error")
                else:
                    self.notify("Llamafile server process finished.", title="Server Status")

                # if self.llamafile_server_process is not None: # If managing process
                #     self.llamafile_server_process = None

                try:
                    self.query_one(actual_start_button_id, Button).disabled = False
                    if self.query_one(actual_stop_button_id, Button):
                        self.query_one(actual_stop_button_id, Button).disabled = True
                except QueryError:
                    self.loguru_logger.warning("Could not find Llamafile server buttons for SUCCESS state.")

            elif worker_state == WorkerState.ERROR:
                self.loguru_logger.error(f"Llamafile server worker failed: {event.worker.error}")
                self.notify(f"Llamafile worker error: {str(event.worker.error)[:100]}", title="Server Worker Error",
                            severity="error")
                # if self.llamafile_server_process is not None: # If managing process
                #     self.llamafile_server_process = None
                try:
                    self.query_one(actual_start_button_id, Button).disabled = False
                    if self.query_one(actual_stop_button_id, Button):
                        self.query_one(actual_stop_button_id, Button).disabled = True
                except QueryError:
                    self.loguru_logger.warning("Could not find Llamafile server buttons for ERROR state.")


        #######################################################################
        # --- Handle vLLM Server Worker (identified by group) ---
        #######################################################################
        elif worker_group == "vllm_server":
            self.loguru_logger.info(
                f"vLLM server worker (Group: '{worker_group}', NameAttr: '{worker_name_attr}') state changed to {worker_state}."
            )
            if worker_state == WorkerState.RUNNING:
                self.loguru_logger.info("vLLM server is RUNNING.")
                try:
                    self.query_one("#vllm-start-server-button", Button).disabled = True
                    self.query_one("#vllm-stop-server-button", Button).disabled = False
                    self.vllm_server_process = event.worker._worker_thread._target_args[
                        1]  # Accessing underlying process if needed, BE CAREFUL
                    self.notify("vLLM server started.", title="Server Status")
                except QueryError:
                    self.loguru_logger.warning("Could not find vLLM server buttons to update state for RUNNING.")
                except (AttributeError, IndexError):
                    self.loguru_logger.warning("Could not retrieve vLLM process object from worker.")

            elif worker_state == WorkerState.SUCCESS or worker_state == WorkerState.ERROR:
                status_message = "successfully" if worker_state == WorkerState.SUCCESS else "with an error"
                self.loguru_logger.info(
                    f"vLLM server process finished {status_message}. Final output handled by 'done' callback.")
                try:
                    self.query_one("#vllm-start-server-button", Button).disabled = False
                    self.query_one("#vllm-stop-server-button", Button).disabled = True
                    self.vllm_server_process = None  # Clear process reference
                    self.notify(f"vLLM server stopped {status_message}.", title="Server Status",
                                severity="information" if worker_state == WorkerState.SUCCESS else "error")
                except QueryError:
                    self.loguru_logger.warning("Could not find vLLM server buttons to update state for STOPPED/ERROR.")

            #######################################################################
            # --- Handle MLX-LM Server Worker ---
            #######################################################################
            elif worker_group == "mlx_lm_server":
                self.loguru_logger.info(f"MLX-LM server worker state changed to {worker_state}.")
                start_button_id = "#mlx-start-server-button"
                stop_button_id = "#mlx-stop-server-button"

                if worker_state == WorkerState.RUNNING:
                    self.loguru_logger.info("MLX-LM server worker is RUNNING.")
                    try:
                        self.query_one(start_button_id, Button).disabled = True
                        self.query_one(stop_button_id, Button).disabled = False
                        self.notify("MLX-LM server process is running.", title="Server Status")
                    except QueryError:
                        self.loguru_logger.warning("Could not find MLX-LM server buttons for RUNNING state.")

                elif worker_state in [WorkerState.SUCCESS, WorkerState.ERROR]:
                    result_message = str(event.worker.result).strip() if event.worker.result else "Worker finished."
                    self.loguru_logger.info(f"MLX-LM worker finished. Result: '{result_message}'")

                    severity = "error" if "error" in result_message.lower() or "non-zero code" in result_message.lower() else "information"
                    self.notify(f"MLX-LM server process finished.", title="Server Status", severity=severity)

                    try:
                        self.query_one(start_button_id, Button).disabled = False
                        self.query_one(stop_button_id, Button).disabled = True
                    except QueryError:
                        self.loguru_logger.warning("Could not find MLX-LM server buttons to reset state.")

                    if self.mlx_server_process is not None:
                        self.mlx_server_process = None

                #######################################################################
                # --- Handle ONNX Server Worker ---
                #######################################################################
            elif worker_group == "onnx_server":
                self.loguru_logger.info(f"ONNX server worker state changed to {worker_state}.")
                start_button_id = "#onnx-start-server-button"
                stop_button_id = "#onnx-stop-server-button"

                if worker_state == WorkerState.RUNNING:
                    self.loguru_logger.info("ONNX server worker is RUNNING.")
                    try:
                        self.query_one(start_button_id, Button).disabled = True
                        self.query_one(stop_button_id, Button).disabled = False
                        self.notify("ONNX server process is running.", title="Server Status")
                    except QueryError:
                        self.loguru_logger.warning("Could not find ONNX server buttons for RUNNING state.")

                elif worker_state in [WorkerState.SUCCESS, WorkerState.ERROR]:
                    result_message = str(event.worker.result).strip() if event.worker.result else "Worker finished."
                    severity = "error" if "error" in result_message.lower() or "non-zero code" in result_message.lower() else "information"
                    self.notify(f"ONNX server process finished.", title="Server Status", severity=severity)

                    try:
                        self.query_one(start_button_id, Button).disabled = False
                        self.query_one(stop_button_id, Button).disabled = True
                    except QueryError:
                        self.loguru_logger.warning("Could not find ONNX server buttons to reset state.")

        #######################################################################
        # --- Handle Transformers Server Worker (identified by group) ---
        #######################################################################
        elif worker_group == "transformers_download":
            self.loguru_logger.info(
                f"Transformers Download worker (Group: '{worker_group}') state changed to {worker_state}."
            )
            download_button_id = "#transformers-download-model-button"

            if worker_state == WorkerState.RUNNING:
                self.loguru_logger.info("Transformers model download worker is RUNNING.")
                try:
                    self.query_one(download_button_id, Button).disabled = True
                except QueryError:
                    self.loguru_logger.warning(
                        f"Could not find button {download_button_id} to disable for RUNNING state.")

            elif worker_state == WorkerState.SUCCESS:
                result_message = str(event.worker.result).strip() if event.worker.result else "Download completed."
                self.loguru_logger.info(f"Transformers Download worker SUCCESS. Result: {result_message}")
                if "failed" in result_message.lower() or "error" in result_message.lower() or "non-zero code" in result_message.lower():
                    self.notify(f"Model Download: {result_message}", title="Download Issue", severity="error",
                                timeout=10)
                else:
                    self.notify(f"Model Download: {result_message}", title="Download Complete", severity="information",
                                timeout=7)
                try:
                    self.query_one(download_button_id, Button).disabled = False
                except QueryError:
                    self.loguru_logger.warning(
                        f"Could not find button {download_button_id} to enable for SUCCESS state.")

            elif worker_state == WorkerState.ERROR:
                error_details = str(event.worker.error) if event.worker.error else "Unknown worker error."
                self.loguru_logger.error(f"Transformers Download worker FAILED. Error: {error_details}")
                self.notify(f"Model Download Failed: {error_details[:100]}...", title="Download Error",
                            severity="error", timeout=10)
                try:
                    self.query_one(download_button_id, Button).disabled = False
                except QueryError:
                    self.loguru_logger.warning(f"Could not find button {download_button_id} to enable for ERROR state.")



        #######################################################################
        # --- Handle Llamafile Server Worker (identified by group) ---
        #######################################################################
        elif worker_group == "llamafile_server":
            self.loguru_logger.info(
                f"Llamafile server worker (Group: '{worker_group}', NameAttr: '{worker_name_attr}') state changed to {worker_state}."
            )
            if worker_state == WorkerState.RUNNING:
                self.loguru_logger.info("Llamafile server is RUNNING.")
                try:
                    self.query_one("#llamafile-start-server-button", Button).disabled = True
                    # Assuming you have a stop button:
                    # self.query_one("#llamafile-stop-server-button", Button).disabled = False
                    self.notify("Llamafile server started.", title="Server Status")
                except QueryError:
                    self.loguru_logger.warning("Could not find Llamafile server buttons to update state for RUNNING.")

            elif worker_state == WorkerState.SUCCESS or worker_state == WorkerState.ERROR:
                status_message = "successfully" if worker_state == WorkerState.SUCCESS else "with an error"
                self.loguru_logger.info(
                    f"Llamafile server process finished {status_message}. Final output handled by 'done' callback.")
                try:
                    self.query_one("#llamafile-start-server-button", Button).disabled = False
                    # self.query_one("#llamafile-stop-server-button", Button).disabled = True
                    self.notify(f"Llamafile server stopped {status_message}.", title="Server Status",
                                severity="information" if worker_state == WorkerState.SUCCESS else "error")
                except QueryError:
                    self.loguru_logger.warning(
                        "Could not find Llamafile server buttons to update state for STOPPED/ERROR.")


        #######################################################################
        # --- Handle Model Download Worker (identified by group) ---
        #######################################################################
        elif worker_group == "model_download":
            self.loguru_logger.info(
                f"Model Download worker (Group: '{worker_group}', NameAttr: '{worker_name_attr}') state changed to {worker_state}."
            )
            # The 'done' callback (stream_worker_output_to_log) handles the detailed log output.
            if worker_state == WorkerState.SUCCESS:
                self.notify("Model download completed successfully.", title="Download Status")
            elif worker_state == WorkerState.ERROR:
                self.notify("Model download failed. Check logs for details.", title="Download Status", severity="error")

            # Re-enable the download button regardless of success or failure
            try:
                # Ensure this ID matches your actual "Start Download" button in LLM_Management_Window.py
                # The provided LLM_Management_Window.py doesn't show a model download section yet.
                # If it's added, ensure the button ID is correct here.
                # For now, this is a placeholder ID.
                download_button = self.query_one("#model-download-start-button", Button)  # Placeholder ID
                download_button.disabled = False
            except QueryError:
                self.loguru_logger.warning(
                    "Could not find model download button to re-enable (ID might be incorrect or view not present).")


        #######################################################################
        # --- Fallback for any other workers not explicitly handled above ---
        #######################################################################
        else:
            # This branch handles workers that are not chat-related and not one of the explicitly grouped servers.
            # It also catches the case where worker_name_attr was a list but not for a known group.
            name_for_log = worker_name_attr if isinstance(worker_name_attr,
                                                          str) else f"[List Name for Group: {worker_group}]"
            self.loguru_logger.debug(
                f"Unhandled worker type or state in on_worker_state_changed: "
                f"Name='{name_for_log}', Group='{worker_group}', State='{worker_state}', Desc='{worker_description}'"
            )

            # Generic handling for workers that might have output but no specific 'done' callback
            # or if their 'done' callback isn't comprehensive for all states.
            if worker_state == WorkerState.SUCCESS or worker_state == WorkerState.ERROR:
                final_message_parts = []
                try:
                    # Check if worker.output is available and is an async iterable
                    if hasattr(event.worker, 'output') and event.worker.output is not None:
                        async for item in event.worker.output:  # type: ignore
                            final_message_parts.append(str(item))

                    final_message = "".join(final_message_parts)
                    if final_message.strip():
                        self.loguru_logger.info(
                            f"Final output from unhandled worker '{name_for_log}' (Group: {worker_group}): {final_message.strip()}")
                        # self.notify(f"Worker '{name_for_log}' finished: {final_message.strip()[:70]}...", title="Worker Update") # Optional notification
                    elif worker_state == WorkerState.SUCCESS:
                        self.loguru_logger.info(
                            f"Unhandled worker '{name_for_log}' (Group: {worker_group}) completed successfully with no final message.")
                        # self.notify(f"Worker '{name_for_log}' completed.", title="Worker Update") # Optional
                    elif worker_state == WorkerState.ERROR:
                        self.loguru_logger.error(
                            f"Unhandled worker '{name_for_log}' (Group: {worker_group}) failed with no specific final message. Error: {event.worker.error}")
                        # self.notify(f"Worker '{name_for_log}' failed. Error: {str(event.worker.error)[:70]}...", title="Worker Error", severity="error") # Optional

                except Exception as e_output:
                    self.loguru_logger.error(
                        f"Error reading output from unhandled worker '{name_for_log}' (Group: {worker_group}): {e_output}",
                        exc_info=True)
    ######################################################
    # --- End of Worker State Change Handlers ---
    ######################################################


    ######################################################
    # --- Watchers for chat sidebar prompt display ---
    ######################################################
    def watch_chat_sidebar_selected_prompt_system(self, new_system_prompt: Optional[str]) -> None:
        try:
            self.query_one("#chat-prompt-system-display", TextArea).load_text(new_system_prompt or "")
        except QueryError:
            self.loguru_logger.error("Chat sidebar system prompt display area (#chat-prompt-system-display) not found.")

    def watch_chat_sidebar_selected_prompt_user(self, new_user_prompt: Optional[str]) -> None:
        try:
            self.query_one("#chat-prompt-user-display", TextArea).load_text(new_user_prompt or "")
        except QueryError:
            self.loguru_logger.error("Chat sidebar user prompt display area (#chat-prompt-user-display) not found.")

    def _clear_chat_sidebar_prompt_display(self) -> None:
        """Clears the prompt display TextAreas in the chat sidebar."""
        self.loguru_logger.debug("Clearing chat sidebar prompt display areas.")
        self.chat_sidebar_selected_prompt_id = None
        self.chat_sidebar_selected_prompt_system = None # Triggers watcher to clear TextArea
        self.chat_sidebar_selected_prompt_user = None   # Triggers watcher to clear TextArea
        # Also clear the list and search inputs if desired when clearing display
        try:
            self.query_one("#chat-sidebar-prompts-listview", ListView).clear()
        except QueryError:
            pass # If not found, it's fine

    def watch_chat_api_provider_value(self, new_value: Optional[str]) -> None:
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        self.loguru_logger.debug(f"Watcher: chat_api_provider_value changed to {new_value}")
        if new_value is None or new_value == Select.BLANK:
            self._update_model_select(TAB_CHAT, [])
            return
        models = self.providers_models.get(new_value, [])
        self._update_model_select(TAB_CHAT, models)

    def watch_ccp_api_provider_value(self, new_value: Optional[str]) -> None: # Renamed from watch_character_...
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        self.loguru_logger.debug(f"Watcher: ccp_api_provider_value changed to {new_value}")
        if new_value is None or new_value == Select.BLANK:
            self._update_model_select(TAB_CCP, [])
            return
        models = self.providers_models.get(new_value, [])
        self._update_model_select(TAB_CCP, models)


    def _update_model_select(self, id_prefix: str, models: list[str]) -> None:
        if not self._ui_ready:  # Add guard
            return
        model_select_id = f"#{id_prefix}-api-model"
        try:
            model_select = self.query_one(model_select_id, Select)
            new_model_options = [(model, model) for model in models]
            # Store current value if it's valid for new options, otherwise try default
            current_value = model_select.value
            model_select.set_options(new_model_options) # This might clear the value

            if current_value in models:
                model_select.value = current_value
            elif models:
                model_select.value = models[0] # Default to first if old value invalid
            else:
                model_select.value = Select.BLANK # No models, ensure blank

            model_select.prompt = "Select Model..." if models else "No models available"
        except QueryError:
            logging.error(f"Helper ERROR: Cannot find model select '{model_select_id}'")
        except Exception as e_set_options:
             logging.error(f"Helper ERROR setting options/value for {model_select_id}: {e_set_options}")

    def chat_wrapper(self, strip_thinking_tags: bool = True, **kwargs: Any) -> Any:
        """
        Delegates to the actual worker target function in worker_events.py.
        This method is called by app.run_worker.
        """
        # All necessary parameters (message, history, api_endpoint, model, etc.)
        # are passed via kwargs from the calling event handler (e.g., handle_chat_send_button_pressed).
        return worker_events.chat_wrapper_function(self, strip_thinking_tags=strip_thinking_tags, **kwargs) # Pass self as 'app_instance'

    ########################################################
    # --- End of Watchers and Helper Methods ---
    # ######################################################

# Initialize logging at the earliest possible point
def initialize_early_logging():
    """Initialize logging as early as possible to capture all logs from startup."""
    from .Logging_Config import configure_application_logging
    # Create a temporary app-like object with just enough attributes for configure_application_logging
    class EarlyLoggingApp:
        def __init__(self):
            self.app_config = load_settings()
            self._rich_log_handler = None

        def query_one(self, *args, **kwargs):
            # This will fail in configure_application_logging, but that's expected
            # for early logging - we just want to set up file and console logging
            raise QueryError("Early logging setup - UI not available yet")

    # Configure logging with our minimal app-like object
    early_app = EarlyLoggingApp()
    configure_application_logging(early_app)
    logging.info("Early logging initialization complete")
    loguru_logger.info("Early logging initialization complete (loguru)")
    return early_app

# --- Main execution block ---
if __name__ == "__main__":
    # Initialize logging first
    early_logging_app = initialize_early_logging()

    # Ensure config file exists (create default if missing)
    try:
        if not DEFAULT_CONFIG_PATH.exists():
            logging.info(f"Config file not found at {DEFAULT_CONFIG_PATH}, creating default.")
            DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_CONFIG_PATH, "w", encoding='utf-8') as f:
                f.write(CONFIG_TOML_CONTENT)
    except Exception as e_cfg_main:
        logging.error(f"Could not ensure creation of default config file: {e_cfg_main}", exc_info=True)

    # --- Emoji Check ---
    emoji_is_supported = supports_emoji() # Call it once
    loguru_logger.info(f"Terminal emoji support detected: {emoji_is_supported}")
    loguru_logger.info(f"Using brain: {get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}")
    loguru_logger.info("-" * 30)

    # --- CSS File Handling ---
    try:
        from .Constants import css_content
        css_dir = Path(__file__).parent / "css"
        css_dir.mkdir(exist_ok=True)
        css_file_path = css_dir / "tldw_cli.tcss"
        if not css_file_path.exists():
            with open(css_file_path, "w", encoding='utf-8') as f:
                f.write(css_content)
            logging.info(f"Created default CSS file: {css_file_path}")
    except Exception as e_css_main:
        logging.error(f"Error handling CSS file: {e_css_main}", exc_info=True)

    # Create instance with early logging flag
    app_instance = TldwCli()
    # Set the early logging flag so _setup_logging knows logging was already initialized
    app_instance._early_logging_initialized = True
    try:
        app_instance.run()
    except Exception as e:
        loguru_logger.exception("--- CRITICAL ERROR DURING app.run() ---")
        traceback.print_exc()  # Make sure traceback prints
    finally:
        # This might run even if app exits early internally in run()
        loguru_logger.info("--- FINALLY block after app.run() ---")

    loguru_logger.info("--- AFTER app.run() call (if not crashed hard) ---")

# Entry point for the tldw-cli command
def main_cli_runner():
    """Entry point for the tldw-cli command.

    This function is referenced in pyproject.toml as the entry point for the tldw-cli command.
    It initializes logging early and then runs the TldwCli app.
    """
    # Initialize logging first
    early_logging_app = initialize_early_logging()

    # Ensure config file exists (create default if missing)
    try:
        if not DEFAULT_CONFIG_PATH.exists():
            logging.info(f"Config file not found at {DEFAULT_CONFIG_PATH}, creating default.")
            DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_CONFIG_PATH, "w", encoding='utf-8') as f:
                f.write(CONFIG_TOML_CONTENT)
    except Exception as e_cfg_main:
        logging.error(f"Could not ensure creation of default config file: {e_cfg_main}", exc_info=True)

    # --- Emoji Check ---
    emoji_is_supported = supports_emoji() # Call it once
    loguru_logger.info(f"Terminal emoji support detected: {emoji_is_supported}")
    loguru_logger.info(f"Using brain: {get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}")
    loguru_logger.info("-" * 30)

    # --- CSS File Handling ---
    try:
        from .Constants import css_content
        css_dir = Path(__file__).parent / "css"
        css_dir.mkdir(exist_ok=True)
        css_file_path = css_dir / "tldw_cli.tcss"
        if not css_file_path.exists():
            with open(css_file_path, "w", encoding='utf-8') as f:
                f.write(css_content)
            logging.info(f"Created default CSS file: {css_file_path}")
    except Exception as e_css_main:
        logging.error(f"Error handling CSS file: {e_css_main}", exc_info=True)

    # Create instance with early logging flag
    app_instance = TldwCli()
    # Set the early logging flag so _setup_logging knows logging was already initialized
    app_instance._early_logging_initialized = True
    try:
        app_instance.run()
    except Exception as e:
        loguru_logger.exception("--- CRITICAL ERROR DURING app.run() ---")
        traceback.print_exc()  # Make sure traceback prints
    finally:
        # This might run even if app exits early internally in run()
        loguru_logger.info("--- FINALLY block after app.run() ---")

    loguru_logger.info("--- AFTER app.run() call (if not crashed hard) ---")

#
# End of app.py
#######################################################################################################################
