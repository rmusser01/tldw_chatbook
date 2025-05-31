# tldw_cli - Textual CLI for LLMs
# Description: This file contains the main application logic for the tldw_cli, a Textual-based CLI for interacting with various LLM APIs.
#
# Imports
import logging
import logging.handlers
import subprocess
import sys
from pathlib import Path
import traceback
from typing import Union, Optional, Any, Dict, List, Callable
#
# 3rd-Party Libraries
from PIL import Image
# --- Textual Imports ---
from loguru import logger as loguru_logger, logger  # Keep if app.py uses it directly, or pass app.loguru_logger
from rich.text import Text
from textual import on
# --- Textual Imports ---
from textual.app import App, ComposeResult
from textual.logging import TextualHandler
from textual.widgets import (
    Static, Button, Input, Header, Footer, RichLog, TextArea, Select, ListView, Checkbox, Collapsible, ListItem, Label
)
from textual.containers import Horizontal, Container, HorizontalScroll, VerticalScroll
# Import for escape_markup
from rich.markup import escape as escape_markup
from textual.reactive import reactive
from textual.worker import Worker
from textual.binding import Binding
from textual.dom import DOMNode  # For type hinting if needed
from textual.timer import Timer
from textual.css.query import QueryError  # For specific error handling

from tldw_chatbook.Event_Handlers.worker_events import StreamingChunk, StreamDone
#
# --- Local API library Imports ---
from .UI.MediaWindow import MediaWindow, slugify as media_slugify
from tldw_chatbook.Constants import ALL_TABS, TAB_CCP, TAB_CHAT, TAB_LOGS, TAB_NOTES, TAB_STATS, TAB_TOOLS_SETTINGS, \
    TAB_INGEST, TAB_LLM, TAB_MEDIA, TAB_SEARCH, TAB_EVALS
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.config import chachanotes_db as global_chachanotes_db_instance, get_media_db_path, CLI_APP_CLIENT_ID
from tldw_chatbook.Logging_Config import RichLogHandler
from tldw_chatbook.Prompt_Management import Prompts_Interop as prompts_interop
from tldw_chatbook.Utils.Emoji_Handling import get_char, EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN, EMOJI_TITLE_NOTE, \
    FALLBACK_TITLE_NOTE, EMOJI_TITLE_SEARCH, FALLBACK_TITLE_SEARCH, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE, \
    EMOJI_SEND, FALLBACK_SEND, EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON, supports_emoji
from tldw_chatbook.Utils.Utils import safe_float, safe_int
from .config import (
    CONFIG_TOML_CONTENT,
    DEFAULT_CONFIG_PATH,
    load_settings,
    get_cli_setting,
    get_cli_log_file_path,
    get_cli_providers_and_models,
    API_MODELS_BY_PROVIDER,
    LOCAL_PROVIDERS, get_chachanotes_db_path, settings
)
from .Screens.Stats_screen import StatsScreen
from .Event_Handlers import (
    app_lifecycle as app_lifecycle_handlers,
    tab_events as tab_handlers,
    sidebar_events as sidebar_handlers,
    chat_events as chat_handlers,
    conv_char_events as ccp_handlers,
    media_events as media_handlers,
    notes_events as notes_handlers,
    worker_events as worker_handlers, worker_events, ingest_events,
    llm_nav_events as llm_handlers,
)
from .Character_Chat import Character_Chat_Lib as ccl
from .Notes.Notes_Library import NotesInteropService
from .DB.ChaChaNotes_DB import CharactersRAGDBError, ConflictError, InputError
from .Widgets.chat_message import ChatMessage
from .Widgets.settings_sidebar import create_settings_sidebar
from .Widgets.notes_sidebar_left import NotesSidebarLeft
from .Widgets.notes_sidebar_right import NotesSidebarRight
from .Widgets.titlebar import TitleBar
from .LLM_Calls.LLM_API_Calls import (
        chat_with_openai, chat_with_anthropic, chat_with_cohere,
        chat_with_groq, chat_with_openrouter, chat_with_huggingface,
        chat_with_deepseek, chat_with_mistral, chat_with_google,
)
from .LLM_Calls.LLM_API_Calls_Local import (
    # Add local API functions if they are in the same file
    chat_with_llama, chat_with_kobold, chat_with_oobabooga,
    chat_with_vllm, chat_with_tabbyapi, chat_with_aphrodite,
    chat_with_ollama, chat_with_custom_openai, chat_with_custom_openai_2, chat_with_local_llm
)
from tldw_chatbook.config import get_chachanotes_db_path, settings, chachanotes_db as global_db_instance
# Import new UI window classes
from .UI.Chat_Window import ChatWindow
from .UI.Conv_Char_Window import CCPWindow
from .UI.Notes_Window import NotesWindow
from .UI.Logs_Window import LogsWindow
from .UI.Stats_Window import StatsWindow
from .UI.Ingest_Window import IngestWindow
from .UI.Tools_Settings_Window import ToolsSettingsWindow
from .UI.LLM_Management_Window import LLMManagementWindow
from .UI.Evals_Window import EvalsWindow # Added EvalsWindow
from .UI.Tab_Bar import TabBar
from .UI.MediaWindow import MediaWindow
from .UI.SearchWindow import SearchWindow
from .UI.SearchWindow import ( # Import new constants from SearchWindow.py
    SEARCH_VIEW_RAG_QA, SEARCH_VIEW_RAG_CHAT, SEARCH_VIEW_EMBEDDINGS_CREATION,
    SEARCH_VIEW_RAG_MANAGEMENT, SEARCH_VIEW_EMBEDDINGS_MANAGEMENT,
    SEARCH_NAV_RAG_QA, SEARCH_NAV_RAG_CHAT, SEARCH_NAV_EMBEDDINGS_CREATION,
    SEARCH_NAV_RAG_MANAGEMENT, SEARCH_NAV_EMBEDDINGS_MANAGEMENT
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
        "ingest-view-media", "ingest-view-notes", "ingest-view-tldw-api"
    ]
    ALL_MAIN_WINDOW_IDS = [ # Assuming these are your main content window IDs
        "chat-window", "conversations_characters_prompts-window",
        "ingest-window", "tools_settings-window", "llm_management-window",
        "media-window", "search-window", "logs-window", "evals-window" # Added "evals-window"
    ]

    # Define reactive at class level with a placeholder default and type hint
    current_tab: reactive[str] = reactive("")
    ccp_active_view: reactive[str] = reactive("conversation_details_view")

    # Add state to hold the currently streaming AI message widget
    current_ai_message_widget: Optional[ChatMessage] = None

    # --- REACTIVES FOR PROVIDER SELECTS ---
    # Initialize with a dummy value or fetch default from config here
    # Ensure the initial value matches what's set in compose/settings_sidebar
    # Fetching default provider from config:
    _default_chat_provider = APP_CONFIG.get("chat_defaults", {}).get("provider", "Ollama")
    _default_ccp_provider = APP_CONFIG.get("character_defaults", {}).get("provider", "Anthropic") # Changed from character_defaults

    chat_api_provider_value: reactive[Optional[str]] = reactive(_default_chat_provider)
    # Renamed character_api_provider_value to ccp_api_provider_value for clarity with TAB_CCP
    ccp_api_provider_value: reactive[Optional[str]] = reactive(_default_ccp_provider)

    # DB Size checker
    _db_size_status_widget: Optional[Static] = None
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
    media_active_view: reactive[Optional[str]] = reactive(None)
    _media_types_for_ui: List[str] = []
    _initial_media_view_slug: Optional[str] = reactive(media_slugify("All Media"))  # Default to "All Media" slug

    current_media_type_filter_slug: reactive[Optional[str]] = reactive(media_slugify("All Media"))  # Slug for filtering
    current_media_type_filter_display_name: reactive[Optional[str]] = reactive("All Media")  # Display name

    # current_media_search_term: reactive[str] = reactive("") # Handled by inputs directly
    current_loaded_media_item: reactive[Optional[Dict[str, Any]]] = reactive(None)
    _media_search_timers: Dict[str, Timer] = {}  # For debouncing per media type

    # Add media_types_for_ui to store fetched types
    media_types_for_ui: List[str] = []
    _initial_media_view: Optional[str] = "media-view-video-audio"  # Default to the first sub-tab
    media_db: Optional[MediaDatabase] = None

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
    vllm_server_process: Optional[subprocess.Popen] = None

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
        except Exception as e:
            self.loguru_logger.error(f"Failed to initialize Media_DB_v2: {e}", exc_info=True)
            self.media_db = None

        # --- Setup Default view for CCP tab ---
        # Initialize self.ccp_active_view based on initial tab or default state if needed
        if self._initial_tab_value == TAB_CCP:
            self.ccp_active_view = "conversation_details_view"  # Default view for CCP tab
        # else: it will default to "conversation_details_view" anyway
        self._ui_ready = False  # Track if UI is fully composed

        # --- Assign DB instances for event handlers ---
        if self.prompts_service_initialized:
            # Assuming prompts_interop holds the db instance after initialization
            # This might need adjustment based on how PromptsDatabase is exposed by prompts_interop
            if hasattr(prompts_interop, 'db_instance') and prompts_interop.db_instance:
                self.prompts_db = prompts_interop.db_instance
                logging.info("Assigned prompts_interop.db_instance to self.prompts_db")
            elif hasattr(prompts_interop, 'db') and prompts_interop.db: # Alternative common name
                self.prompts_db = prompts_interop.db
                logging.info("Assigned prompts_interop.db to self.prompts_db")
            else:
                logging.error("prompts_interop initialized, but prompts_db instance (db_instance or db) not found/assigned in app.__init__.")
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


    def _setup_logging(self):
        """Sets up all logging handlers, including Loguru integration."""
        # FIXME - LOGGING MAY BRING BACK BLINKING
        temp_handler = logging.StreamHandler(sys.stdout)
        temp_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(temp_handler)
        # This first logging.info will go to the stderr handler from the initial basicConfig
        logging.info("--- _setup_logging START ---")
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # --- BEGIN LOGURU MANAGEMENT (Your existing code is mostly fine here) ---
        try:
            loguru_logger.remove()  # Good: removes Loguru's default stderr sink
            logging.info("Loguru: All pre-existing sinks removed.")

            def sink_to_standard_logging(message):
                # ... (your existing sink_to_standard_logging function)
                record = message.record
                level_mapping = {
                    "TRACE": logging.DEBUG, "DEBUG": logging.DEBUG, "INFO": logging.INFO,
                    "SUCCESS": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR,
                    "CRITICAL": logging.CRITICAL,
                }
                std_level = level_mapping.get(record["level"].name, logging.INFO)
                std_logger = logging.getLogger(record["name"])
                if record["exception"]:
                    std_logger.log(std_level, record["message"], exc_info=record["exception"])
                else:
                    std_logger.log(std_level, record["message"])

            loguru_logger.add(
                sink_to_standard_logging,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                level="TRACE"
            )
            # This log message will also currently go to the initial basicConfig stderr handler
            logging.info("Loguru: Configured to forward its messages to standard Python logging system.")
        except Exception as e:
            # This log message will also currently go to the initial basicConfig stderr handler
            logging.error(f"Loguru: Error during Loguru reconfiguration: {e}", exc_info=True)
        # --- END LOGURU MANAGEMENT ---

        # --- CONFIGURE STANDARD PYTHON LOGGING ROOT LOGGER ---
        root_logger = logging.getLogger()

        # !!! IMPORTANT FIX: Remove all existing handlers from the root logger !!!
        # This will get rid of the StreamHandler (to stderr) added by the initial
        # global logging.basicConfig() call.
        initial_handlers_removed_count = 0
        for handler in root_logger.handlers[:]:  # Iterate over a copy
            root_logger.removeHandler(handler)
            if hasattr(handler, 'close') and callable(handler.close):
                try:
                    handler.close()
                except Exception:
                    pass  # Ignore errors during close of old handlers
            initial_handlers_removed_count += 1

        # Log this removal using Loguru, as standard logging has no handlers yet.
        # This message will go to Loguru's sink (which forwards to std logging,
        # but std logging has no handlers yet, so it might hit Python's "last resort" stderr).
        # Or, better, print to stderr just for this one-off setup message if needed, then rely on proper handlers.
        if initial_handlers_removed_count > 0:
            # Using print here because logging state is actively being changed.
            # This should be one of the last messages to hit raw stderr if setup is correct.
            print(
                f"INFO: _setup_logging: Removed {initial_handlers_removed_count} pre-existing handler(s) from root logger.",
                file=sys.stderr)

        # Now that root_logger is clean, set its overall level.
        # This level acts as a filter before messages reach any of its handlers.
        initial_log_level_str = self.app_config.get("general", {}).get("log_level", "INFO").upper()
        initial_log_level = getattr(logging, initial_log_level_str, logging.INFO)
        root_logger.setLevel(initial_log_level)
        # (A temporary print to confirm, as logging to root_logger now might go to "last resort" until a handler is added)
        print(f"INFO: _setup_logging: Root logger level set to {logging.getLevelName(root_logger.level)}",
              file=sys.stderr)

        # --- Add TextualHandler (to standard logging) ---
        # (Your existing TextualHandler setup code is fine)
        # Ensure it's added AFTER clearing old handlers and setting root level.
        # ...
        has_textual_handler = any(isinstance(h, TextualHandler) for h in root_logger.handlers)
        if not has_textual_handler:
            textual_console_handler = TextualHandler()
            textual_console_handler.setLevel(initial_log_level)  # Respects app_config
            console_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            textual_console_handler.setFormatter(console_formatter)
            root_logger.addHandler(textual_console_handler)
            # Now, logging.info should go to Textual's dev console (and other handlers added below)
            logging.info(
                f"Standard Logging: Added TextualHandler (Level: {logging.getLevelName(textual_console_handler.level)}).")
        else:
            logging.info("Standard Logging: TextualHandler already exists.")

        # Test Loguru message again. It should now go to TextualHandler (and others).
        loguru_logger.info(
            "Loguru Test: This message from Loguru should now appear in Textual dev console (and other configured handlers).")

        # --- Setup RichLog Handler (to standard logging) ---
        # (Your existing RichLogHandler setup code is fine, ensure it's added AFTER clearing)
        # ...
        try:
            log_display_widget = self.query_one("#app-log-display", RichLog)
            # Check if it's already added by a previous call (should not happen if _setup_logging is called once)
            if not any(isinstance(h, RichLogHandler) and h.rich_log_widget is log_display_widget for h in
                       root_logger.handlers):
                if not self._rich_log_handler:  # Create if it doesn't exist
                    self._rich_log_handler = RichLogHandler(log_display_widget)
                # Configure and add
                rich_log_handler_level_str = self.app_config.get("logging", {}).get("rich_log_level", "DEBUG").upper()
                rich_log_handler_level = getattr(logging, rich_log_handler_level_str, logging.DEBUG)
                self._rich_log_handler.setLevel(rich_log_handler_level)
                root_logger.addHandler(self._rich_log_handler)
                logging.info(
                    f"Standard Logging: Added RichLogHandler (Level: {logging.getLevelName(self._rich_log_handler.level)}).")
            else:
                logging.info("Standard Logging: RichLogHandler already exists and is added.")
        except QueryError:
            logging.error("!!! ERROR: Failed to find #app-log-display widget for RichLogHandler setup.")
            self._rich_log_handler = None
        except Exception as e:
            logging.error(f"!!! ERROR setting up RichLogHandler: {e}", exc_info=True)
            self._rich_log_handler = None

        # --- Setup File Logging (to standard logging) ---
        # (Your existing FileHandler setup code is fine, ensure it's added AFTER clearing)
        # ... (your existing code to add file_handler to root_logger) ...
        try:
            log_file_path = get_cli_log_file_path()
            log_dir = log_file_path.parent
            log_dir.mkdir(parents=True, exist_ok=True)

            has_file_handler = any(
                isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == str(log_file_path) for h in
                root_logger.handlers)

            if not has_file_handler:
                max_bytes_default = 10485760
                backup_count_default = 5
                file_log_level_default_str = "INFO"
                max_bytes = int(get_cli_setting("logging", "log_max_bytes", max_bytes_default))
                backup_count = int(get_cli_setting("logging", "log_backup_count", backup_count_default))
                file_log_level_str = get_cli_setting("logging", "file_log_level", file_log_level_default_str).upper()
                file_log_level = getattr(logging, file_log_level_str, logging.INFO)

                file_handler = logging.handlers.RotatingFileHandler(
                    log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
                )
                file_handler.setLevel(file_log_level)
                file_formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
                logging.info(
                    f"Standard Logging: Added RotatingFileHandler (File: '{log_file_path}', Level: {logging.getLevelName(file_log_level)}).")
            else:
                logging.info("Standard Logging: RotatingFileHandler already exists for this file path.")
        except Exception as e:
            logging.warning(f"!!! ERROR setting up file logging: {e}", exc_info=True)

        # Re-evaluate lowest level for standard logging root logger
        # (Your existing logic for this is fine)
        all_std_handlers = root_logger.handlers
        if all_std_handlers:
            handler_levels = [h.level for h in all_std_handlers if h.level > 0]
            if handler_levels:
                lowest_effective_level = min(handler_levels)
                current_root_level = root_logger.level
                # Only adjust root logger level if it's currently *less* verbose (higher numeric value)
                # than the most verbose handler.
                if current_root_level > lowest_effective_level:
                    logging.info(
                        f"Standard Logging: Adjusting root logger level from {logging.getLevelName(current_root_level)} to {logging.getLevelName(lowest_effective_level)} to match most verbose handler.")
                    root_logger.setLevel(lowest_effective_level)
            logging.info(f"Standard Logging: Final Root logger level is: {logging.getLevelName(root_logger.level)}")
        else:
            logging.warning("Standard Logging: No handlers found on root logger after setup!")

        logging.info("Logging setup complete.")
        logging.info("--- _setup_logging END ---")

    def compose(self) -> ComposeResult:
        logging.debug("App composing UI...")
        yield Header()
        # Set up the main title bar with a static title
        yield TitleBar()

        # Use new TabBar widget
        yield TabBar(tab_ids=ALL_TABS, initial_active_tab=self._initial_tab_value)

        # FIXME - OLD
        #yield from self.compose_tabs()

        yield from self.compose_content_area() # Call refactored content area composer

        with Footer():
            yield Static(id="db-size-indicator", markup=False) # markup=False to display text literally
        logging.debug("App compose finished.")

    def compose_tabs(self) -> ComposeResult:
        # The outer container still docks and defines the overall tab bar height
        with Horizontal(id="tabs-outer-container"):  # This container takes the 'dock: top' and 'height: 3'
            # The HorizontalScroll allows the buttons inside to scroll if they overflow
            with HorizontalScroll(id="tabs"):  # This is where the buttons will go
                for tab_id_loop in ALL_TABS:
                    label_text = "CCP" if tab_id_loop == TAB_CCP else \
                        "Tools & Settings" if tab_id_loop == TAB_TOOLS_SETTINGS else \
                        "Ingest Content" if tab_id_loop == TAB_INGEST else \
                        "LLM Management" if tab_id_loop == TAB_LLM else \
                        tab_id_loop.replace('_', ' ').capitalize()
                    yield Button(
                        label_text,
                        id=f"tab-{tab_id_loop}",
                        classes="-active" if tab_id_loop == self._initial_tab_value else ""
                    )

    ############################################################
    #
    # Code that builds the content area of the app aka the main UI.
    #
    ###########################################################
    def compose_content_area(self) -> ComposeResult:
        self.loguru_logger.info(f"--- ENTERING COMPOSE CONTENT AREA (Direct Yield Pattern with explicit Media/Search) ---")
        self.loguru_logger.info(f"Initial _initial_tab_value: {self._initial_tab_value}")
        self.loguru_logger.info(f"Constants.ALL_TABS content: {ALL_TABS}")

        # This parent container is crucial
        with Container(id="content"):
            composed_window_ids = set()

            def _yield_and_track(window_instance, tab_constant_val, actual_window_id_val):
                nonlocal composed_window_ids
                if self._initial_tab_value != tab_constant_val:
                    window_instance.styles.display = "none"
                else:
                    window_instance.styles.display = "block"
                yield window_instance
                composed_window_ids.add(actual_window_id_val)
                self.loguru_logger.debug(f"Yielded {window_instance.__class__.__name__}, ID: {actual_window_id_val}, Display: {window_instance.styles.display}")

            self.loguru_logger.debug("Instantiating and yielding concrete tab windows...")

            yield from _yield_and_track(ChatWindow(self, id="chat-window", classes="window"), TAB_CHAT, "chat-window")
            yield from _yield_and_track(CCPWindow(self, id="conversations_characters_prompts-window", classes="window"), TAB_CCP, "conversations_characters_prompts-window")
            yield from _yield_and_track(NotesWindow(self, id="notes-window", classes="window"), TAB_NOTES, "notes-window")
            yield from _yield_and_track(IngestWindow(self, id="ingest-window", classes="window"), TAB_INGEST, "ingest-window")
            yield from _yield_and_track(ToolsSettingsWindow(self, id="tools_settings-window", classes="window"), TAB_TOOLS_SETTINGS, "tools_settings-window")
            yield from _yield_and_track(LLMManagementWindow(self, id="llm_management-window", classes="window"), TAB_LLM, "llm_management-window")
            yield from _yield_and_track(LogsWindow(self, id="logs-window", classes="window"), TAB_LOGS, "logs-window")
            yield from _yield_and_track(StatsWindow(self, id="stats-window", classes="window"), TAB_STATS, "stats-window")

            # --- Pass fetched media types to MediaWindow ---
            media_window_instance = MediaWindow(self, id="media-window", classes="window")
            media_window_instance.media_types_from_db = self._media_types_for_ui # Set before compose
            yield from _yield_and_track(media_window_instance, TAB_MEDIA, "media-window")
            # --- End MediaWindow with passed types ---

            yield from _yield_and_track(SearchWindow(self, id="search-window", classes="window"), TAB_SEARCH, "search-window")
            yield from _yield_and_track(EvalsWindow(self, id="evals-window", classes="window"), TAB_EVALS, "evals-window") # Added EvalsWindow

            self.loguru_logger.info(f"Finished yielding concrete windows. Composed IDs: {composed_window_ids}")

            self.loguru_logger.info(f"Starting placeholder loop. ALL_TABS: {ALL_TABS}")
            unique_tab_constants = set(ALL_TABS)
            self.loguru_logger.info(f"Unique tab constants for placeholder loop: {unique_tab_constants}")
            self.loguru_logger.info(f"Current composed_window_ids: {composed_window_ids}")

            for tab_constant_for_placeholder in unique_tab_constants:
                target_window_id = "llm_management-window" if tab_constant_for_placeholder == TAB_LLM else f"{tab_constant_for_placeholder}-window"
                self.loguru_logger.debug(f"Placeholder Loop: tab_const='{tab_constant_for_placeholder}', target_id='{target_window_id}'")
                if target_window_id not in composed_window_ids:
                    self.loguru_logger.info(f"  --> CREATING placeholder for '{tab_constant_for_placeholder}' (ID '{target_window_id}') as it's not in composed_window_ids.")
                    placeholder_container = Container(id=target_window_id, classes="window placeholder-window")
                    if self._initial_tab_value != tab_constant_for_placeholder:
                        placeholder_container.styles.display = "none"
                    else:
                        placeholder_container.styles.display = "block"
                    with placeholder_container:
                        yield Static(f"{tab_constant_for_placeholder.replace('_', ' ').capitalize()} Window Placeholder")
                        yield Button("Coming Soon...", id=f"ph-btn-{tab_constant_for_placeholder}", disabled=True)
                    yield placeholder_container
                    composed_window_ids.add(target_window_id)
                    self.loguru_logger.debug(f"  --> Yielded and added placeholder '{target_window_id}'. composed_window_ids: {composed_window_ids}")
                else:
                    self.loguru_logger.debug(f"  --> SKIPPING placeholder for '{target_window_id}', already in composed_window_ids.")
            self._ui_ready = True
            self.loguru_logger.info(f"--- FINISHED COMPOSE CONTENT AREA --- Final composed IDs: {composed_window_ids}")
            self.loguru_logger.info("UI composition completed - watchers enabled")

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
                        self.query_one("#ccp-card-first-message-display", TextArea).text = details.get("first_mes", "")

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
        if not self._ui_ready:
            self.loguru_logger.debug(
                f"watch_search_active_sub_tab: UI not ready. Old: {old_sub_tab}, New: {new_sub_tab}.")
            return
        if not new_sub_tab:  # If new_sub_tab is None (e.g. on initial load before set)
            self.loguru_logger.debug(f"watch_search_active_sub_tab: new_sub_tab is None. Old: {old_sub_tab}.")
            # Optionally hide all if new_sub_tab is explicitly set to None to clear view
            if old_sub_tab:  # If there was an old tab, ensure it's hidden and button deactivated
                try:
                    self.query_one(f"#{old_sub_tab.replace('-view-', '-nav-')}", Button).remove_class(
                        "-active-search-sub-view")
                    self.query_one(f"#{old_sub_tab}", Container).styles.display = "none"
                except QueryError:
                    pass  # It might already be gone or not exist
            return

        self.loguru_logger.debug(f"Search active sub-tab changing from '{old_sub_tab}' to: '{new_sub_tab}'")

        try:
            search_content_pane = self.query_one("#search-content-pane")
            search_nav_pane = self.query_one("#search-left-nav-pane")

            # Hide all search view areas first
            for view_area in search_content_pane.query(".search-view-area"):
                view_area.styles.display = "none"

            # Deactivate all nav buttons in search tab
            for nav_button in search_nav_pane.query(".search-nav-button"):
                nav_button.remove_class("-active-search-sub-view")

            # Show the selected view and activate its button
            target_view_id_selector = f"#{new_sub_tab}"
            view_to_show = self.query_one(target_view_id_selector, Container)
            view_to_show.styles.display = "block"  # Or "flex" etc. if needed by content

            # Activate corresponding button
            # Button ID is like "search-nav-..." and view ID is "search-view-..."
            button_id_to_activate = new_sub_tab.replace("-view-", "-nav-")
            try:
                active_button = self.query_one(f"#{button_id_to_activate}", Button)
                active_button.add_class("-active-search-sub-view")
            except QueryError:
                self.loguru_logger.warning(
                    f"Could not find button '{button_id_to_activate}' to activate for sub-tab '{new_sub_tab}'.")

            self.loguru_logger.info(f"Switched Search sub-tab view to: {new_sub_tab}")

            # Optional: Focus an element within the newly shown view
            # try:
            #     first_focusable = view_to_show.query(Input, TextArea, Button)[0]
            #     self.set_timer(0.1, first_focusable.focus)
            # except IndexError:
            #     pass # No focusable element
            # except QueryError: # If view_to_show doesn't exist (should not happen if previous query_one succeeded)
            #     self.loguru_logger.error(f"Cannot focus in {new_sub_tab}, view not found after successful query.")

        except QueryError as e:
            self.loguru_logger.error(f"UI component not found during Search sub-tab view switch: {e}",
                                     exc_info=True)
        except Exception as e_watch:
            self.loguru_logger.error(f"Unexpected error in watch_search_active_sub_tab: {e_watch}", exc_info=True)

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
                # Optional: Focus
                # try:
                #     view_to_show.query(Input, Button)[0].focus()
                # except IndexError: pass
            except QueryError as e:
                self.loguru_logger.error(f"UI component '{new_view}' not found in #llm-content-pane: {e}",
                                         exc_info=True)
        else:
            self.loguru_logger.debug("LLM Management active view is None, all LLM views hidden.")

    # --- Media Tab Watcher ---
    def watch_media_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        if not hasattr(self, "app") or not self.app:
            return
        if not self._ui_ready:
            return
        self.loguru_logger.debug(f"Media active view changing from '{old_view}' to: '{new_view}'")

        try:
            content_pane = self.query_one("#media-content-pane")
        except QueryError:
            self.loguru_logger.error("#media-content-pane not found. Cannot switch Media views.")
            return

        # Hide all media view areas first
        for child in content_pane.query(".media-view-area"):
            child.styles.display = "none"

        if new_view: # new_view is the ID of the view container, e.g., "media-view-video-audio"
            try:
                target_view_id_selector = f"#{new_view}"
                view_to_show = content_pane.query_one(target_view_id_selector, Container)
                view_to_show.styles.display = "block" # Or "flex", etc.
                self.loguru_logger.info(f"Switched Media view to: {new_view}")
            except QueryError as e:
                self.loguru_logger.error(f"UI component '{new_view}' not found in #media-content-pane: {e}", exc_info=True)
        else:
            self.loguru_logger.debug("Media active view is None, all media views hidden.")



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

    async def _set_initial_tab(self) -> None:  # New method for deferred tab setting
        self.loguru_logger.info("Setting initial tab via call_later.")
        self.current_tab = self._initial_tab_value
        self.loguru_logger.info(f"Initial tab set to: {self.current_tab}")

    async def _post_mount_setup(self) -> None:
        """Operations to perform after the main UI is expected to be fully mounted."""
        self.loguru_logger.info("App _post_mount_setup: Binding Select widgets and populating dynamic content...")

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
        self.call_later(self._populate_chat_conversation_character_filter_select)
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
        self.loguru_logger.info("App _post_mount_setup: Post-mount setup completed.")


    async def on_shutdown_request(self) -> None:  # Use the imported ShutdownRequest
        logging.info("--- App Shutdown Requested ---")
        if self._rich_log_handler:
            await self._rich_log_handler.stop_processor()
            logging.info("RichLogHandler processor stopped.")

    async def on_unmount(self) -> None:
        """Clean up logging resources on application exit."""
        logging.info("--- App Unmounting ---")
        self._ui_ready = False
        if self._rich_log_handler: # Ensure it's removed if it exists
            logging.getLogger().removeHandler(self._rich_log_handler)
            logging.info("RichLogHandler removed.")
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
            self.call_later(chat_handlers.handle_chat_sidebar_prompt_search_changed, self, "") # Call with empty search term
            self.call_later(chat_handlers._populate_chat_character_search_list, self) # Populate character list
        elif new_tab == TAB_CCP:
            # Initial population for CCP tab when switched to
            self.call_later(ccp_handlers.populate_ccp_character_select, self)
            self.call_later(ccp_handlers.populate_ccp_prompts_list_view, self)
            self.call_later(ccp_handlers.perform_ccp_conversation_search, self) # Initial search/list for conversations
        elif new_tab == TAB_NOTES:
            self.call_later(notes_handlers.load_and_display_notes_handler, self)
        elif new_tab == TAB_SEARCH:
            if not self.search_active_sub_tab: # If no sub-tab is active yet for Search tab
                self.loguru_logger.debug(f"Switched to Search tab, activating initial sub-tab view: {self._initial_search_sub_tab_view}")
                # Use call_later to ensure the UI for SearchWindow is fully composed and ready
                self.call_later(setattr, self, 'search_active_sub_tab', self._initial_search_sub_tab_view)
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
            sidebar = self.query_one("#evals-sidebar") # ID from EvalsWindow.py
            sidebar.set_class(collapsed, "collapsed") # Assumes "collapsed" class handles display: none
            # Alternatively: sidebar.display = not collapsed
            self.loguru_logger.debug(f"Evals sidebar collapsed state: {collapsed}, class set/removed.")
        except QueryError:
            self.loguru_logger.error("Evals sidebar (#evals-sidebar) not found for collapse toggle.")
        except Exception as e:
            self.loguru_logger.error(f"Error toggling Evals sidebar: {e}", exc_info=True)

    # --- Method DEFINITION for show_ingest_view ---
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

    # --- Notes UI Event Handlers (Chat Tab Sidebar) ---
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

    ########################################################################
    #
    # --- EVENT DISPATCHERS ---
    #
    ########################################################################
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for tabs, sending messages, and message actions."""
        button = event.button
        button_id = button.id
        self.loguru_logger.debug(f"Button pressed: ID: {button_id}, Classes: {button.classes}, Label: '{button.label}' on tab {self.current_tab}") # More info

        # Tab Switching
        if button_id and button_id.startswith("tab-"):
            await tab_handlers.handle_tab_button_pressed(self, button_id)
            return # Handled

        current_active_tab = self.current_tab

        # Sidebar Toggles (now tab-specific)
        if button_id and button_id.startswith("toggle-"):
            if current_active_tab == TAB_CHAT:
                if button_id in ["toggle-chat-left-sidebar", "toggle-chat-right-sidebar"]:
                    await chat_handlers.handle_chat_tab_sidebar_toggle(self, button_id)
                    return
            elif current_active_tab == TAB_CCP:
                if button_id in ["toggle-conv-char-left-sidebar", "toggle-conv-char-right-sidebar"]:
                    await ccp_handlers.handle_ccp_tab_sidebar_toggle(self, button_id)
                    return
            elif current_active_tab == TAB_NOTES:
                if button_id in ["toggle-notes-sidebar-left", "toggle-notes-sidebar-right"]:
                    await notes_handlers.handle_notes_tab_sidebar_toggle(self, button_id)
                    return
            elif current_active_tab == TAB_SEARCH:
                self.loguru_logger.info(f"Button '{button_id}' on active Search tab.")
                search_button_id_to_view_map = {
                    SEARCH_NAV_RAG_QA: SEARCH_VIEW_RAG_QA,
                    SEARCH_NAV_RAG_CHAT: SEARCH_VIEW_RAG_CHAT,
                    SEARCH_NAV_EMBEDDINGS_CREATION: SEARCH_VIEW_EMBEDDINGS_CREATION,
                    SEARCH_NAV_RAG_MANAGEMENT: SEARCH_VIEW_RAG_MANAGEMENT,
                    SEARCH_NAV_EMBEDDINGS_MANAGEMENT: SEARCH_VIEW_EMBEDDINGS_MANAGEMENT,
                }
                if button_id in search_button_id_to_view_map:
                    view_to_activate = search_button_id_to_view_map[button_id]
                    self.loguru_logger.debug(
                        f"Search nav button '{button_id}' pressed. Activating view '{view_to_activate}'.")
                    self.search_active_sub_tab = view_to_activate
                    return  # Search sub-navigation button handled.
                else:
                    self.loguru_logger.warning(
                        f"Unhandled button on SEARCH tab: ID:{button_id}, Label:'{event.button.label}'")
                return  # Explicitly return after handling (or logging unhandled) buttons on Search tab.
                # --- Ingest Nav Pane Buttons (within Ingest Tab) ---
            elif self.current_tab == "ingest":  # Only process these if ingest tab is active
                if button_id == "ingest-nav-prompts":
                    if button_id and button_id.startswith("ingest-nav-"):
                        view_to_activate = button_id.replace("ingest-nav-", "-view-")  # Corrected replacement
                        self.loguru_logger.info(
                            f"Ingest nav button '{button_id}' pressed. Activating view '{view_to_activate}'.")
                        self.ingest_active_view = view_to_activate
                        # Simplified and more careful initialization for the prompts view
                        if view_to_activate == "ingest-view-prompts":
                            try:
                                selected_list_view = self.query_one("#ingest-prompts-selected-files-list", ListView)
                                if not selected_list_view.children:  # Only if it's truly empty
                                    await selected_list_view.clear()  # Clear just in case, then append
                                    await selected_list_view.append(ListItem(Label("No files selected.")))

                                preview_area = self.query_one("#ingest-prompts-preview-area", VerticalScroll)
                                # Only add placeholder if preview is empty
                                if not preview_area.children:
                                   #await preview_area.clear()  # Clear just in case, then mount
                                    await preview_area.mount(
                                        Static("Select files to see a preview.",
                                               id="ingest-prompts-preview-placeholder"))
                            except QueryError:
                                self.loguru_logger.warning(
                                    "Failed to initialize prompts list/preview for ingest-view-prompts on nav click.")
                        return  # Nav button handled
                    self.show_ingest_view("ingest-view-prompts")
                    try:
                        selected_list_view = self.query_one("#ingest-prompts-selected-files-list", ListView)
                        if not selected_list_view._nodes:
                            await selected_list_view.append(ListItem(Label("No files selected.")))
                        preview_area = self.query_one("#ingest-prompts-preview-area", VerticalScroll)
                        await preview_area.remove_children()  # Clear old preview
                        await preview_area.mount(
                            Static("Select files to see a preview.", id="ingest-prompts-preview-placeholder"))
                    except QueryError:
                        self.log.warning("Failed to initialize prompts list view elements on nav click.")
                elif button_id == "ingest-nav-characters":
                    self.show_ingest_view("ingest-view-characters")
                elif button_id == "ingest-nav-media":
                    self.show_ingest_view("ingest-view-media")
                elif button_id == "ingest-nav-notes":
                    self.show_ingest_view("ingest-view-notes")
                elif button_id == "ingest-nav-tldw-api":
                    self.show_ingest_view("ingest-view-tldw-api")

                # --- Buttons within ingest-view-prompts ---
                # Ensure these handlers are only called if the ingest-view-prompts is active
                # or simply rely on the button_id being unique enough.
                elif button_id == "ingest-prompts-select-file-button":
                    # This is where ingest_events.handle_... is CALLED
                    await ingest_events.handle_ingest_prompts_select_file_button_pressed(self)
                elif button_id == "ingest-prompts-clear-files-button":
                    await ingest_events.handle_ingest_prompts_clear_files_button_pressed(self)
                elif button_id == "ingest-prompts-import-now-button":
                    await ingest_events.handle_ingest_prompts_import_now_button_pressed(self)


        # --- Tab-Specific Button Actions ---
        if current_active_tab == TAB_CHAT:
            action_widget = self._get_chat_message_widget_from_button(button)
            if action_widget:
                self.loguru_logger.debug(
                    f"Button (ID: {button_id}, Label: '{button.label}') identified as part of ChatMessage. Delegating to chat_actions.")
                await chat_handlers.handle_chat_action_button_pressed(self, button, action_widget)
                return
            if button_id == "send-chat": await chat_handlers.handle_chat_send_button_pressed(self, TAB_CHAT)
            elif button_id == "chat-new-conversation-button": await chat_handlers.handle_chat_new_conversation_button_pressed(self)
            elif button_id == "chat-new-temp-chat-button": await chat_handlers.handle_chat_new_conversation_button_pressed(self) # Reuses existing handler
            elif button_id == "chat-save-current-chat-button": await chat_handlers.handle_chat_save_current_chat_button_pressed(self)
            elif button_id == "chat-save-conversation-details-button": await chat_handlers.handle_chat_save_details_button_pressed(self)
            elif button_id == "chat-conversation-load-selected-button": await chat_handlers.handle_chat_load_selected_button_pressed(self)
            elif button_id == "chat-prompt-load-selected-button": await chat_handlers.handle_chat_view_selected_prompt_button_pressed(self)
            elif button_id == "chat-prompt-copy-system-button": await chat_handlers.handle_chat_copy_system_prompt_button_pressed(self)
            elif button_id == "chat-prompt-copy-user-button": await chat_handlers.handle_chat_copy_user_prompt_button_pressed(self)
            elif button_id == "chat-load-character-button": await chat_handlers.handle_chat_load_character_button_pressed(self, event) # Pass event if needed by handler
            elif button_id == "chat-clear-active-character-button": # Ensure this exact line exists
                await chat_handlers.handle_chat_clear_active_character_button_pressed(self)
            # --- Chat Tab Notes Sidebar Buttons ---
            # These are handled by @on decorators, so we just acknowledge them here to prevent "unhandled" warnings.
            elif button_id == "chat-notes-create-new-button": self.loguru_logger.debug(f"Button {button_id} handled by @on decorator.")
            elif button_id == "chat-notes-search-button": self.loguru_logger.debug(f"Button {button_id} handled by @on decorator.")
            elif button_id == "chat-notes-load-button": self.loguru_logger.debug(f"Button {button_id} handled by @on decorator.")
            elif button_id == "chat-notes-save-button": self.loguru_logger.debug(f"Button {button_id} handled by @on decorator.")
            else: self.loguru_logger.warning(f"Unhandled button on CHAT tab -> ID: {button_id}, Label: '{button.label}'")

        elif current_active_tab == TAB_CCP:
            # ---- First, try to identify if it's an action button within a ChatMessage (if CCP tab uses them) ----
            action_widget_ccp = self._get_chat_message_widget_from_button(button)
            if action_widget_ccp:
                self.loguru_logger.debug(
                    f"Button (ID: {button_id}, Label: '{button.label}') identified as part of ChatMessage on CCP tab. Delegating.")
                await chat_handlers.handle_chat_action_button_pressed(self, button, action_widget_ccp) # Assuming generic actions
                return
            if button_id == "conv-char-conversation-search-button": await ccp_handlers.handle_ccp_conversation_search_button_pressed(self)
            elif button_id == "ccp-import-character-button": await ccp_handlers.handle_ccp_import_character_button_pressed(self)
            elif button_id == "conv-char-load-button": await ccp_handlers.handle_ccp_load_conversation_button_pressed(self)
            elif button_id == "conv-char-save-details-button": await ccp_handlers.handle_ccp_save_conversation_details_button_pressed(self)
            elif button_id == "ccp-import-prompt-button": await ccp_handlers.handle_ccp_import_prompt_button_pressed(self)
            elif button_id == "ccp-prompt-create-new-button": await ccp_handlers.handle_ccp_prompt_create_new_button_pressed(self)
            elif button_id == "ccp-prompt-load-selected-button": await ccp_handlers.handle_ccp_prompt_load_selected_button_pressed(self)
            elif button_id == "ccp-prompt-save-button": await ccp_handlers.handle_ccp_prompt_save_button_pressed(self) # For RIGHT pane editor
            elif button_id == "ccp-prompt-clone-button": await ccp_handlers.handle_ccp_prompt_clone_button_pressed(self) # For RIGHT pane editor
            elif button_id == "ccp-prompt-delete-button": await ccp_handlers.handle_ccp_prompt_delete_button_pressed(self) # For RIGHT pane editor
            # Buttons for CENTER PANE editor
            elif button_id == "ccp-editor-prompt-save-button": await ccp_handlers.handle_ccp_editor_prompt_save_button_pressed(self)
            elif button_id == "ccp-editor-prompt-clone-button": await ccp_handlers.handle_ccp_editor_prompt_clone_button_pressed(self)
            elif button_id == "ccp-editor-prompt-delete-button": await ccp_handlers.handle_ccp_editor_prompt_delete_button_pressed(self)
            elif button_id == "ccp-import-conversation-button": await ccp_handlers.handle_ccp_import_conversation_button_pressed(self)
            elif button_id == "ccp-right-pane-load-character-button":
                self.loguru_logger.info(f"CCP Right Pane Load Character button pressed: {button_id}")
                await ccp_handlers.handle_ccp_left_load_character_button_pressed(self)
            else: self.loguru_logger.warning(f"Unhandled button on CCP tab -> ID: {button_id}, Label: '{button.label}'")

        # --- Notes Tab ---
        elif current_active_tab == TAB_NOTES:
            if button_id == "notes-create-new-button": await notes_handlers.handle_notes_create_new_button_pressed(self)
            elif button_id == "notes-edit-selected-button": await notes_handlers.handle_notes_edit_selected_button_pressed(self)
            elif button_id == "notes-import-button": await notes_handlers.handle_notes_import_button_pressed(self)
            elif button_id == "notes-search-button": await notes_handlers.handle_notes_search_button_pressed(self)
            elif button_id == "notes-load-selected-button": await notes_handlers.handle_notes_load_selected_button_pressed(self)
            elif button_id == "notes-save-current-button": await notes_handlers.handle_notes_save_current_button_pressed(self)
            elif button_id == "notes-save-button": await notes_handlers.handle_notes_main_save_button_pressed(self)
            elif button_id == "notes-delete-button": await notes_handlers.handle_notes_delete_button_pressed(self)
            elif button_id == "notes-save-keywords-button": await notes_handlers.handle_notes_save_keywords_button_pressed(self)
            else: self.loguru_logger.warning(f"Unhandled button on NOTES tab: {button_id}")

        # --- Media Tab ---
        elif current_active_tab == TAB_MEDIA:
            if button_id and button_id.startswith("media-nav-"):
                # e.g., "media-nav-video-audio" -> "media-view-video-audio"
                view_to_activate = button_id.replace("media-nav-", "media-view-")
                self.loguru_logger.debug(f"Media nav button '{button_id}' pressed. Activating view '{view_to_activate}'.")
                self.media_active_view = view_to_activate # Triggers watcher
                await media_handlers.handle_media_nav_button_pressed(self, button_id)
                return
            elif button_id and button_id.startswith("media-search-button-"):
                await media_handlers.handle_media_search_button_pressed(self, button_id)
                return
            elif button_id and button_id.startswith("media-load-selected-button-"):
                await media_handlers.handle_media_load_selected_button_pressed(self, button_id)
                return
            else:
                self.loguru_logger.warning(f"Unhandled button on MEDIA tab: ID:{button_id}, Label:'{button.label}'")

        # --- Ingestion Tab ---
        elif current_active_tab == TAB_INGEST:
            # Navigation buttons within the Ingest tab's left pane
            if button_id and button_id.startswith("ingest-nav-"):
                view_to_activate_nav = button_id.replace("ingest-nav-", "ingest-view-")
                self.loguru_logger.info(
                    f"Ingest nav button '{button_id}' pressed. Activating view '{view_to_activate_nav}'.")
                self.ingest_active_view = view_to_activate_nav

                if view_to_activate_nav == "ingest-view-prompts":
                    try:
                        selected_list_view = self.query_one("#ingest-prompts-selected-files-list", ListView)
                        if not selected_list_view.children:
                            await selected_list_view.clear()
                            await selected_list_view.append(ListItem(Label("No files selected.")))
                        preview_area = self.query_one("#ingest-prompts-preview-area", VerticalScroll)
                        if not preview_area.children:
                            await preview_area.mount(
                                Static("Select files to see a preview.", id="ingest-prompts-preview-placeholder"))
                    except QueryError:
                        self.loguru_logger.warning(
                            "Failed to initialize prompts list/preview elements on nav click to prompts.")
                elif view_to_activate_nav == "ingest-view-characters":
                    try:
                        selected_list_view = self.query_one("#ingest-characters-selected-files-list", ListView)
                        if not selected_list_view.children:
                            await selected_list_view.clear()
                            await selected_list_view.append(ListItem(Label("No files selected.")))
                        preview_area = self.query_one("#ingest-characters-preview-area", VerticalScroll)
                        if not preview_area.children:
                            await preview_area.mount(
                                Static("Select files to see a preview.", id="ingest-characters-preview-placeholder"))
                    except QueryError:
                        self.loguru_logger.warning(
                            "Failed to initialize characters list/preview for ingest-view-characters on nav click.")
                    return  # Nav button handled
                return

            # ELSE, if not a nav button, it must be a button within an active sub-view
            else:
                active_ingest_sub_view = self.ingest_active_view

                if active_ingest_sub_view == "ingest-view-prompts":
                    if button_id == "ingest-prompts-select-file-button":
                        await ingest_events.handle_ingest_prompts_select_file_button_pressed(self);
                        return
                    elif button_id == "ingest-prompts-clear-files-button":
                        await ingest_events.handle_ingest_prompts_clear_files_button_pressed(self);
                        return
                    elif button_id == "ingest-prompts-import-now-button":
                        await ingest_events.handle_ingest_prompts_import_now_button_pressed(self);
                        return

                elif active_ingest_sub_view == "ingest-view-characters":
                    if button_id == "ingest-characters-select-file-button":
                        await ingest_events.handle_ingest_characters_select_file_button_pressed(self);
                        return
                    elif button_id == "ingest-characters-clear-files-button":
                        await ingest_events.handle_ingest_characters_clear_files_button_pressed(self);
                        return
                    elif button_id == "ingest-characters-import-now-button":
                        await ingest_events.handle_ingest_characters_import_now_button_pressed(self);
                        return

                # Add other sub-views like ingest-view-notes here
                # elif active_ingest_sub_view == "ingest-view-notes":
                #     # ... handle buttons for notes ingest ...
                #     pass # Remember to return if handled

                # If no sub-view button matched after checking the active sub-view:
                self.loguru_logger.warning(
                    f"Unhandled button on INGEST tab: ID:{button_id}, Label:'{event.button.label}' (Active Ingest View: {active_ingest_sub_view})")
                return  # Return after logging unhandled Ingest tab button

        # --- Tools & Settings Tab ---
        elif current_active_tab == TAB_TOOLS_SETTINGS:
            if button_id and button_id.startswith("ts-nav-"):
                # Extract the view name from the button ID
                # e.g., "ts-nav-general-settings" -> "ts-view-general-settings"
                view_to_activate = button_id.replace("ts-nav-", "ts-view-")
                self.loguru_logger.debug(
                    f"Tools & Settings nav button '{button_id}' pressed. Activating view '{view_to_activate}'.")
                self.tools_settings_active_view = view_to_activate  # This will trigger the watcher
            else:
                self.loguru_logger.warning(
                    f"Unhandled button on TOOLS & SETTINGS tab: ID:{button_id}, Label:'{button.label}'")

        # --- LLM Inference Tab ---
        elif current_active_tab == TAB_LLM:
            if button_id and button_id.startswith("llm-nav-"):
                await llm_handlers.handle_llm_nav_button_pressed(self, button_id)
            else:
                self.loguru_logger.warning(
                    f"Unhandled button on LLM MANAGEMENT tab: ID:{button_id}, Label:'{button.label}'")

        # --- Logging Tab ---
        elif current_active_tab == TAB_LOGS:
            if button_id == "copy-logs-button": await app_lifecycle_handlers.handle_copy_logs_button_pressed(self)
            else: self.loguru_logger.warning(f"Unhandled button on LOGS tab: {button_id}")

        # --- Evals Tab ---
        elif current_active_tab == TAB_EVALS:
            if button_id == "toggle-evals-sidebar":
                self.evals_sidebar_collapsed = not self.evals_sidebar_collapsed
                self.loguru_logger.debug(f"Evals sidebar toggle button pressed. New state: {self.evals_sidebar_collapsed}")
                return # Handled
            else:
                self.loguru_logger.warning(f"Unhandled button on EVALS tab: ID:{button_id}, Label:'{button.label}'")
            return # Explicitly return after handling buttons on Evals tab.

        else:
            self.loguru_logger.warning(f"Button '{button_id}' pressed on unhandled/unknown tab '{current_active_tab}' or unhandled button ID.")

    def _get_chat_message_widget_from_button(self, button: Button) -> Optional[ChatMessage]:
        """Helper to find the parent ChatMessage widget from an action button within it."""
        self.loguru_logger.debug(f"_get_chat_message_widget_from_button searching for parent of button ID: {button.id}, Classes: {button.classes}")
        node: Optional[DOMNode] = button.parent
        depth = 0
        max_depth = 5 # Safety break
        while node is not None and depth < max_depth:
            self.loguru_logger.debug(f"  Traversal depth {depth}: current node is {type(node)}, id: {getattr(node, 'id', 'N/A')}, classes: {getattr(node, 'classes', '')}")
            if isinstance(node, ChatMessage):
                self.loguru_logger.debug(f"  Found ChatMessage parent!")
                return node
            node = node.parent
            depth += 1
        if depth >= max_depth:
            self.loguru_logger.warning(f"  _get_chat_message_widget_from_button reached max depth for button: {button.id}")
        else:
            self.loguru_logger.warning(f"  _get_chat_message_widget_from_button could not find parent ChatMessage for button: {button.id}")
        return None

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

    async def on_input_changed(self, event: Input.Changed) -> None:
        input_id = event.input.id
        current_active_tab = self.current_tab
        # --- Chat Sidebar Prompt Search ---
        if input_id == "chat-prompt-search-input" and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_sidebar_prompt_search_input_changed(self, event.value)
        # --- Notes Search ---
        elif input_id == "notes-search-input" and current_active_tab == TAB_NOTES:
            await notes_handlers.handle_notes_search_input_changed(self, event.value)
        # --- Chat Sidebar Conversation Search ---
        elif input_id == "chat-conversation-search-bar" and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_conversation_search_bar_changed(self, event.value)
        elif input_id == "conv-char-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_conversation_search_input_changed(self, event.value)
        elif input_id == "ccp-prompt-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_prompt_search_input_changed(self, event.value)
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
        # Add more specific input handlers if needed, e.g., for title inputs if they need live validation/reaction

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        list_view_id = event.list_view.id
        current_active_tab = self.current_tab
        item_details = f"Item prompt_id: {getattr(event.item, 'prompt_id', 'N/A')}, Item prompt_uuid: {getattr(event.item, 'prompt_uuid', 'N/A')}"
        self.loguru_logger.info(
            f"ListView.Selected: list_view_id='{list_view_id}', current_tab='{current_active_tab}', {item_details}"
        )

        if list_view_id == "notes-list-view" and current_active_tab == TAB_NOTES:
            self.loguru_logger.debug("Dispatching to notes_handlers.handle_notes_list_view_selected")
            await notes_handlers.handle_notes_list_view_selected(self, list_view_id, event.item)
        elif list_view_id == "ccp-prompts-listview" and current_active_tab == TAB_CCP:
            self.loguru_logger.debug("Dispatching to ccp_handlers.handle_ccp_prompts_list_view_selected")
            await ccp_handlers.handle_ccp_prompts_list_view_selected(self, list_view_id, event.item)
        elif list_view_id == "chat-sidebar-prompts-listview" and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug("Dispatching to chat_handlers.handle_chat_sidebar_prompts_list_view_selected")
            await ccp_handlers.handle_ccp_prompts_list_view_selected(self, list_view_id, event.item)
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

    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        # Delegate all worker state changes to the central handler
        await worker_handlers.handle_api_call_worker_state_changed(self, event)

    async def on_streaming_chunk(self, event: StreamingChunk) -> None:
        """Handles incoming chunks of text during streaming."""
        logger = getattr(self, 'loguru_logger', logging)
        if self.current_ai_message_widget and self.current_ai_message_widget.is_mounted:
            try:
                # The thinking placeholder should have been cleared when the worker started.
                # The role and header should also have been set at the start of the AI turn.
                static_text_widget = self.current_ai_message_widget.query_one(".message-text", Static)

                # Append the clean text chunk
                self.current_ai_message_widget.message_text += event.text_chunk

                # Update the display with the accumulated, escaped text
                static_text_widget.update(escape_markup(self.current_ai_message_widget.message_text))

                # Scroll the chat log to the end
                # Determine the correct chat log container based on the active tab
                chat_log_id_to_query = ""
                if self.current_tab == TAB_CHAT:
                    chat_log_id_to_query = "#chat-log"
                elif self.current_tab == TAB_CCP:
                    chat_log_id_to_query = "#ccp-conversation-log" # Assuming this is the ID in CCPWindow

                if chat_log_id_to_query:
                    chat_log_container = self.query_one(chat_log_id_to_query, VerticalScroll)
                    chat_log_container.scroll_end(animate=False, duration=0.05) # Fast scroll
                else:
                    logger.warning(f"on_streaming_chunk: Could not determine chat log container for tab {self.current_tab}")

            except QueryError as e:
                logger.error(f"Error accessing UI components during streaming chunk update: {e}", exc_info=True)
            except Exception as e_chunk: # Catch any other unexpected error
                logger.error(f"Unexpected error processing streaming chunk: {e_chunk}", exc_info=True)
        else:
            logger.warning("Received StreamingChunk but no current_ai_message_widget is active/mounted or tab is not Chat/CCP.")

    async def on_stream_done(self, event: StreamDone) -> None:
        """Handles the end of a stream, including errors and successful completion."""
        logger = getattr(self, 'loguru_logger', logging)
        logger.info(f"StreamDone received. Final text length: {len(event.full_text)}. Error: '{event.error}'")

        ai_widget = self.current_ai_message_widget # Use a local variable for clarity

        if not ai_widget or not ai_widget.is_mounted:
            logger.warning("Received StreamDone but current_ai_message_widget is missing or not mounted.")
            if event.error: # If there was an error, at least notify the user
                self.notify(f"Stream error (display widget missing): {event.error}", severity="error", timeout=10)
            # Ensure current_ai_message_widget is None even if it was already None or unmounted
            self.current_ai_message_widget = None
            # Attempt to focus input if possible as a fallback
            try:
                if self.current_tab == TAB_CHAT:
                    self.query_one("#chat-input", TextArea).focus()
                elif self.current_tab == TAB_CCP: # Assuming similar input ID convention
                    self.query_one("#ccp-chat-input", TextArea).focus() # Adjust if ID is different
            except QueryError: pass # Ignore if input not found
            return

        try:
            static_text_widget = ai_widget.query_one(".message-text", Static)

            if event.error:
                logger.error(f"Stream completed with error: {event.error}")
                # If full_text has content, it means some chunks were received before the error.
                # Display partial text along with the error.
                error_message_content = event.full_text + f"\n\n[bold red]Stream Error:[/]\n{escape_markup(event.error)}"

                ai_widget.message_text = event.full_text + f"\nStream Error: {event.error}" # Update internal raw text
                static_text_widget.update(Text.from_markup(error_message_content))
                ai_widget.role = "System" # Change role to "System" or "Error"
                try:
                    header_label = ai_widget.query_one(".message-header", Label)
                    header_label.update("System Error") # Update header
                except QueryError:
                    logger.warning("Could not update AI message header for stream error display.")
                # Do NOT save to database if there was an error.
            else: # No error, stream completed successfully
                logger.info("Stream completed successfully.")
                ai_widget.message_text = event.full_text # Ensure internal state has the final, complete text
                static_text_widget.update(escape_markup(event.full_text)) # Update display with final, escaped text

                # Determine sender name for DB (already set on widget by handle_api_call_worker_state_changed)
                # This is just to ensure the correct name is used for DB saving if needed.
                ai_sender_name_for_db = ai_widget.role # Role should be correctly set by now

                # Save to DB if applicable (not ephemeral, not empty, and DB available)
                if self.chachanotes_db and self.current_chat_conversation_id and \
                   not self.current_chat_is_ephemeral and event.full_text.strip():
                    try:
                        logger.debug(f"Attempting to save streamed AI message to DB. ConvID: {self.current_chat_conversation_id}, Sender: {ai_sender_name_for_db}")
                        ai_msg_db_id = ccl.add_message_to_conversation(
                            self.chachanotes_db,
                            self.current_chat_conversation_id,
                            ai_sender_name_for_db,
                            event.full_text # Save the clean, full text
                        )
                        if ai_msg_db_id:
                            saved_ai_msg_details = self.chachanotes_db.get_message_by_id(ai_msg_db_id)
                            if saved_ai_msg_details:
                                ai_widget.message_id_internal = saved_ai_msg_details.get('id')
                                ai_widget.message_version_internal = saved_ai_msg_details.get('version')
                                logger.info(f"Streamed AI message saved to DB. ConvID: {self.current_chat_conversation_id}, MsgID: {saved_ai_msg_details.get('id')}")
                            else:
                                logger.error(f"Failed to retrieve saved streamed AI message details (ID: {ai_msg_db_id}) from DB.")
                        else:
                            logger.error("Failed to save streamed AI message to DB (no ID returned).")
                    except (CharactersRAGDBError, InputError) as e_save_ai_stream:
                        logger.error(f"DB Error saving streamed AI message: {e_save_ai_stream}", exc_info=True)
                        self.notify(f"DB error saving message: {e_save_ai_stream}", severity="error")
                    except Exception as e_save_unexp:
                        logger.error(f"Unexpected error saving streamed AI message: {e_save_unexp}", exc_info=True)
                        self.notify("Unexpected error saving message.", severity="error")
                elif not event.full_text.strip() and not event.error:
                    logger.info("Stream finished with no error but content was empty/whitespace. Not saving to DB.")


            ai_widget.mark_generation_complete() # Mark as complete in both error/success cases if widget exists

        except QueryError as e:
            logger.error(f"QueryError during StreamDone UI update (event.error='{event.error}'): {e}", exc_info=True)
            if event.error: # If there was an underlying stream error, make sure user sees it
                 self.notify(f"Stream Error (UI issue): {event.error}", severity="error", timeout=10)
            else: # If stream was fine, but UI update failed
                 self.notify("Error finalizing AI message display.", severity="error")
        except Exception as e_done_unexp: # Catch any other unexpected error during the try block
            logger.error(f"Unexpected error in on_stream_done (event.error='{event.error}'): {e_done_unexp}", exc_info=True)
            self.notify("Internal error finalizing stream.", severity="error")
        finally:
            # This block executes regardless of exceptions in the try block above.
            # Crucial for resetting state and UI.
            self.current_ai_message_widget = None # Clear the reference to the AI message widget
            logger.debug("Cleared current_ai_message_widget in on_stream_done's finally block.")

            # Focus the appropriate input based on the current tab
            input_id_to_focus = None
            if self.current_tab == TAB_CHAT:
                input_id_to_focus = "#chat-input"
            elif self.current_tab == TAB_CCP:
                input_id_to_focus = "#ccp-chat-input" # Adjust if ID is different for CCP tab's input

            if input_id_to_focus:
                try:
                    input_widget = self.query_one(input_id_to_focus, TextArea)
                    input_widget.focus()
                    logger.debug(f"Focused input '{input_id_to_focus}' in on_stream_done.")
                except QueryError:
                    logger.warning(f"Could not focus input '{input_id_to_focus}' in on_stream_done (widget not found).")
                except Exception as e_focus_final:
                    logger.error(f"Error focusing input '{input_id_to_focus}' in on_stream_done: {e_focus_final}", exc_info=True)
            else:
                logger.debug(f"No specific input to focus for tab {self.current_tab} in on_stream_done.")

    # --- Helper methods that remain in app.py (mostly for UI orchestration or complex state) ---
    def _safe_float(self, value: str, default: float, name: str) -> float:
        return safe_float(value, default, name) # Delegate to imported helper

    def _safe_int(self, value: str, default: Optional[int], name: str) -> Optional[int]:
        return safe_int(value, default, name) # Delegate to imported helper

    def _get_api_name(self, provider: str, endpoints: dict) -> Optional[str]:
        if not self._ui_ready:
            return None
        provider_key_map = { "llama_cpp": "llama_cpp", "Ollama": "Ollama", "Oobabooga": "Oobabooga", "koboldcpp": "koboldcpp", "vllm": "vllm", "Custom": "Custom", "Custom-2": "Custom_2", }
        endpoint_key = provider_key_map.get(provider)
        if endpoint_key:
            url = endpoints.get(endpoint_key)
            if url: return url
            else: logging.warning(f"URL key '{endpoint_key}' for provider '{provider}' missing in config [api_endpoints].")
        return None

    # --- Watchers for chat sidebar prompt display ---
    def watch_chat_sidebar_selected_prompt_system(self, new_system_prompt: Optional[str]) -> None:
        try:
            self.query_one("#chat-sidebar-prompt-system-display", TextArea).load_text(new_system_prompt or "")
        except QueryError:
            self.loguru_logger.error("Chat sidebar system prompt display area not found.")

    def watch_chat_sidebar_selected_prompt_user(self, new_user_prompt: Optional[str]) -> None:
        try:
            self.query_one("#chat-sidebar-prompt-user-display", TextArea).load_text(new_user_prompt or "")
        except QueryError:
            self.loguru_logger.error("Chat sidebar user prompt display area not found.")

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


    async def _populate_chat_conversation_character_filter_select(self) -> None:
        """Populates the character filter select in the Chat tab's conversation search."""
        # ... (Keep original implementation as is) ...
        logging.info("Attempting to populate #chat-conversation-search-character-filter-select.")
        if not self.notes_service:
            logging.error("Notes service not available for char filter select (Chat Tab).")
            # Optionally update the select to show an error state
            try:
                char_filter_select_err = self.query_one("#chat-conversation-search-character-filter-select", Select)
                char_filter_select_err.set_options([("Service Offline", Select.BLANK)])
            except QueryError: pass
            return
        try:
            db = self.notes_service._get_db(self.notes_user_id)
            character_cards = db.list_character_cards(limit=1000)
            options = [(char['name'], char['id']) for char in character_cards if char.get('name') and char.get('id')]

            char_filter_select = self.query_one("#chat-conversation-search-character-filter-select", Select)
            char_filter_select.set_options(options if options else [("No characters", Select.BLANK)])
            # Default to BLANK, user must explicitly choose or use "All Characters" checkbox
            char_filter_select.value = Select.BLANK
            logging.info(f"Populated #chat-conversation-search-character-filter-select with {len(options)} chars.")
        except QueryError as e_q:
            logging.error(f"Failed to find #chat-conversation-search-character-filter-select: {e_q}", exc_info=True)
        except CharactersRAGDBError as e_db: # Catch specific DB error
            logging.error(f"DB error populating char filter select (Chat Tab): {e_db}", exc_info=True)
        except Exception as e_unexp:
            logging.error(f"Unexpected error populating char filter select (Chat Tab): {e_unexp}", exc_info=True)

    def chat_wrapper(self, **kwargs: Any) -> Any:
        """
        Delegates to the actual worker target function in worker_events.py.
        This method is called by app.run_worker.
        """
        # All necessary parameters (message, history, api_endpoint, model, etc.)
        # are passed via kwargs from the calling event handler (e.g., handle_chat_send_button_pressed).
        return worker_events.chat_wrapper_function(self, **kwargs) # Pass self as 'app_instance'

# --- Main execution block ---
if __name__ == "__main__":
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

    app_instance = TldwCli() # Create instance
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
