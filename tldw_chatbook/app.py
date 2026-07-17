# tldw_cli - Textual CLI for LLMs
# Description: This file contains the main application logic for the tldw_cli, a Textual-based CLI for interacting with various LLM APIs.
#
# Disable progress bars early to prevent interference with TUI
import os

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Disable Textual logging in production
# Set to a path to enable logging for debugging: os.environ['TEXTUAL_LOG'] = '/tmp/textual.log'
if 'TEXTUAL_LOG' not in os.environ:
    os.environ['TEXTUAL_LOG'] = ''  # Empty string disables logging

# Imports
import concurrent.futures
import contextlib
import functools
import inspect
import logging
import logging.handlers
import multiprocessing
import multiprocessing.connection
import random
import subprocess
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING, Union, Optional, Any, Dict, List, Callable
from textual.widget import Widget
#
# 3rd-Party Libraries
import asyncio
from PIL import Image
from loguru import logger as loguru_logger, logger
from rich.markup import escape as escape_markup
from textual import on, work
from textual.app import App, ComposeResult, ScreenStackError
from textual.widgets import (
    Static, Button, Input, RichLog, TextArea, Select, ListView, Checkbox, Collapsible, ListItem, Label, Switch, Markdown
)
from textual.containers import Container, VerticalScroll
from textual.reactive import reactive
from textual.worker import Worker
from textual.binding import Binding
from textual.message import Message
from textual.timer import Timer
from textual.css.query import NoMatches, QueryError
from textual.command import Hit, Hits, Provider
from functools import partial
from pathlib import Path

from tldw_chatbook.Utils.text import slugify
from tldw_chatbook.css.Themes.themes import ALL_THEMES
# from tldw_chatbook.css.css_loader import load_modular_css  # Removed - reverting to original CSS
from tldw_chatbook.Metrics.metrics import log_histogram, log_counter, log_resource_usage, init_metrics_server
from tldw_chatbook.Metrics.Otel_Metrics import init_metrics as init_otel_metrics
#
# --- Local API library Imports ---
from .Event_Handlers.LLM_Management_Events import (llm_management_events, llm_management_events_mlx_lm,
    llm_management_events_ollama, llm_management_events_onnx, llm_management_events_transformers,
                                                   llm_management_events_vllm)
from tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events import handle_streaming_chunk, handle_stream_done
from tldw_chatbook.Event_Handlers.worker_events import StreamingChunk, StreamDone
from .Widgets.AppFooterStatus import AppFooterStatus
from .config import (
    get_cli_setting,
    get_library_collections_db_path,
    get_library_ingest_jobs_db_path,
    get_media_db_path,
    get_notifications_db_path,
    get_prompts_db_path,
    get_notifications_db_path,
    get_research_db_path,
    get_subscriptions_db_path,
    get_user_data_dir,
    get_workspaces_db_path,
    get_writing_db_path,
)
from .Logging_Config import configure_application_logging
from tldw_chatbook.Constants import ALL_TABS, TAB_CCP, TAB_CHAT, TAB_HOME, TAB_LOGS, TAB_STATS, TAB_TOOLS_SETTINGS, TAB_CUSTOMIZE, \
    TAB_INGEST, TAB_LLM, TAB_MEDIA, TAB_SEARCH, TAB_EVALS, TAB_LIBRARY, TAB_ARTIFACTS, TAB_PERSONAS, TAB_WATCHLISTS_COLLECTIONS, \
    TAB_SCHEDULES, TAB_WORKFLOWS, TAB_MCP, TAB_ACP, TAB_SKILLS, TAB_SETTINGS, LLAMA_CPP_SERVER_ARGS_HELP_TEXT, \
    LLAMAFILE_SERVER_ARGS_HELP_TEXT, TAB_CODING, TAB_STTS, TAB_STUDY, TAB_WRITING, TAB_RESEARCH, TAB_SUBSCRIPTIONS, TAB_CHATBOOKS, \
    LIBRARY_NAV_CONTEXT_MODE, LIBRARY_NAV_CONTEXT_NOTE_ID, LIBRARY_NAV_CONTEXT_NOTES_CREATE, \
    LIBRARY_NAV_CONTEXT_INGEST, \
    get_tab_display_label
from tldw_chatbook.Chat.chat_conversation_scope_service import ChatConversationScopeService
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.conversation_local_marks_service import ConversationLocalMarksService
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_live_work import (
    ConsoleLiveWorkLaunch,
    resolve_console_live_work_primary_action,
)
from tldw_chatbook.Chat.chat_loop_scope_service import ServerChatLoopScopeService
from tldw_chatbook.Chat.server_chat_conversation_service import ServerChatConversationService
from tldw_chatbook.Chat.server_chat_loop_service import ServerChatLoopService
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.config import CLI_APP_CLIENT_ID
from tldw_chatbook.Chat import (
    ChatConversationScopeService,
    ChatConversationService,
    ServerChatConversationService,
)
from tldw_chatbook.Chatbooks import LocalChatbookService, ServerChatbookService
from tldw_chatbook.Library import LocalLibraryCollectionsService
from tldw_chatbook.Library.library_ingest_jobs import (
    DEFAULT_CHUNK_SIZE,
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.Library.library_local_rag_search_service import LibraryLocalRagSearchService
from tldw_chatbook.Local_Ingestion import FileIngestionError
from tldw_chatbook.Local_Ingestion.ingest_parse_worker import classify_parse_failure, run_parse_job
from tldw_chatbook.Local_Ingestion.local_file_ingestion import classify_ingest_source, persist_parsed_media
from tldw_chatbook.Home.active_work_adapter import (
    HomeControlAction,
    HomeControlResult,
    HomeControlResultStatus,
    LocalNotificationHomeActiveWorkAdapter,
    UnavailableHomeActiveWorkAdapter,
)
from tldw_chatbook.Logging_Config import RichLogHandler
from tldw_chatbook.Prompt_Management import (
    LocalPromptService,
    PromptChatbookScopeService,
    Prompts_Interop as prompts_interop,
    ServerPromptService,
)
from tldw_chatbook.Utils.Emoji_Handling import get_char, EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN, supports_emoji
from tldw_chatbook.Utils.log_widget_manager import LogWidgetManager
from tldw_chatbook.Utils.ui_helpers import UIHelpers
from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor
from tldw_chatbook.Utils.db_status_manager import DBStatusManager
from tldw_chatbook.Event_Handlers.worker_handlers import (
    WorkerHandlerRegistry, ChatWorkerHandler, ServerWorkerHandler,
    AIGenerationHandler, MiscWorkerHandler
)
from .config import (
    CONFIG_TOML_CONTENT,
    DEFAULT_CONFIG_PATH,
    load_settings,
    get_cli_setting,
    get_cli_providers_and_models,
    API_MODELS_BY_PROVIDER,
    LOCAL_PROVIDERS,
    load_cli_config_and_ensure_existence,
    set_encryption_password, )
from .Event_Handlers import (
    conv_char_events as ccp_handlers,
    notes_events as notes_handlers,
    worker_events, ingest_events,
    llm_nav_events, media_events, notes_events, app_lifecycle, tab_events,
    subscription_events,
)
from .Event_Handlers.Chat_Events import chat_events as chat_handlers, chat_events_sidebar, chat_events_worldbooks
from tldw_chatbook.Event_Handlers.Chat_Events import chat_events
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSRequestEvent, TTSCompleteEvent, TTSPlaybackEvent, TTSProgressEvent, TTSEventHandler
)
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler, STTSPlaygroundGenerateEvent, STTSSettingsSaveEvent, STTSAudioBookGenerateEvent
)
from .Notes.Notes_Library import NotesInteropService
from .Notes.notes_scope_service import NotesScopeService
from .Notes.server_notes_workspace_service import ServerNotesWorkspaceService
from .Character_Chat.character_persona_scope_service import CharacterPersonaScopeService
from .Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService
from .Character_Chat.local_character_persona_service import LocalCharacterPersonaService
from .Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from .Character_Chat.server_chat_dictionary_service import ServerChatDictionaryService
from .Character_Chat.server_character_persona_service import ServerCharacterPersonaService
from .RAG_Admin.local_rag_admin_service import LocalRAGAdminService
from .RAG_Admin.rag_admin_scope_service import RAGAdminScopeService
from .RAG_Admin.server_rag_admin_service import ServerRAGAdminService
from .Study_Interop import (
    LocalQuizService,
    LocalStudyService,
    QuizScopeService,
    ServerQuizService,
    ServerStudyService,
    StudyScopeService,
)
from .Writing_Interop import (
    LocalWritingService,
    ServerWritingService,
    WritingScopeService,
)
from .Research_Interop import (
    LocalResearchService,
    ResearchScopeService,
    ServerResearchService,
)
from .ACP_Interop.runtime_process import ACPRuntimeProcessManager
from .ACP_Interop.runtime_session import ACPRuntimeSessionState
from .DB.ChaChaNotes_DB import CharactersRAGDBError, ConflictError
from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage
from tldw_chatbook.Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced
from .Widgets.titlebar import TitleBar
from .Widgets.splash_screen import SplashScreen
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
from tldw_chatbook.config import get_chachanotes_db_path, settings, get_chachanotes_db_lazy
from .UI.Navigation.main_navigation import NavigateToScreen
from .UI.Navigation.screen_registry import resolve_screen_target
from .UI.Screens.media_runtime_state import MediaRuntimeState
from .UI.Screens.study_scope_models import StudyScopeContext
# Ingest UI has been rebuilt to use an internal TabbedContent (local/remote)
# The legacy per-view navigation (ingest-nav-*/ingest-view-*) is not used anymore.
# Keep these as empty to avoid wiring legacy handlers.
USE_REBUILT_INGEST = True
INGEST_NAV_BUTTON_IDS: list[str] = []
INGEST_VIEW_IDS: list[str] = []
from .UI.Tools_Settings_Window import ToolsSettingsWindow
from .UI.LLM_Management_Window import LLMManagementWindow
from .UI.Customize_Window import CustomizeWindow
from .UI.Tab_Bar import TabBar
from .UI.Tab_Links import TabLinks
from .UI.Tab_Dropdown import TabDropdown
from .UI.console_command_provider import ConsoleCommandProvider
from tldw_chatbook.Chat_Grammars_Interop import (
    ChatGrammarsScopeService,
    LocalChatGrammarsService,
    ServerChatGrammarsService,
)
from tldw_chatbook.Claims_Interop import ClaimsScopeService, ServerClaimsService
from tldw_chatbook.Companion_Interop import CompanionScopeService, ServerCompanionService
from tldw_chatbook.Collections_Interop import CollectionsFeedsScopeService, ServerCollectionsFeedsService
from tldw_chatbook.External_Connectors_Interop import ConnectorsScopeService, ServerConnectorsService
from tldw_chatbook.Feedback_Interop import FeedbackScopeService, LocalFeedbackService, ServerFeedbackService
from tldw_chatbook.Kanban_Interop import KanbanScopeService, LocalKanbanService, ServerKanbanService
from tldw_chatbook.LLM_Provider_Catalog import (
    LLMProviderCatalogScopeService,
    LocalLLMProviderCatalogService,
    ServerLLMProviderCatalogService,
)
from tldw_chatbook.Media import (
    LocalMediaReadingService,
    MediaReadingScopeService,
    ServerMediaReadingService,
)
from tldw_chatbook.Meetings_Interop import MeetingsScopeService, ServerMeetingsService
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore
from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService
from tldw_chatbook.Notifications import (
    ClientNotificationsDB,
    ClientNotificationsService,
    EventStateRepository,
    NotificationsScopeService,
    NotificationDispatchService,
    ServerNotificationsService,
)
from tldw_chatbook.Outputs_Interop import OutputsScopeService, ServerOutputsService
from tldw_chatbook.Personalization_Interop import (
    PersonalizationScopeService,
    ServerPersonalizationService,
)
from tldw_chatbook.Prompt_Management.prompt_scope_service import build_prompt_scope_service
from tldw_chatbook.Prompt_Studio_Interop import PromptStudioScopeService, ServerPromptStudioService
from tldw_chatbook.Research_Interop import (
    LocalResearchSearchService,
    LocalResearchService,
    ResearchSearchScopeService,
    ResearchScopeService,
    ServerResearchSearchService,
    ServerResearchService,
)
from tldw_chatbook.Server_Runtime_Interop import ServerRuntimeScopeService, ServerRuntimeService
from tldw_chatbook.Sharing_Interop import ServerSharingService, SharingScopeService
from tldw_chatbook.Skills_Interop import (
    LocalSkillsService,
    ServerSkillsService,
    SkillTrustService,
    SkillsScopeService,
)
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    SkillTrustStore,
    build_default_skill_trust_key_cache,
    build_skill_trust_marker_store_with_fallback,
)
from tldw_chatbook.Sync_Interop import (
    LocalFirstSyncService,
    ManualSyncControlService,
    ServerSyncService,
    SyncScopeService,
    SyncStateRepository,
)
from tldw_chatbook.Text2SQL_Interop import ServerText2SQLService, Text2SQLScopeService
from tldw_chatbook.Tools_Interop import ServerToolsService, ToolsScopeService
from tldw_chatbook.MCP_Governance_Interop import MCPGovernanceScopeService, ServerMCPGovernanceService
from tldw_chatbook.User_Governance_Interop import ServerUserGovernanceService, UserGovernanceScopeService
from tldw_chatbook.Web_Clipper_Interop import ServerWebClipperService, WebClipperScopeService
from tldw_chatbook.Web_Scraping_Interop import ServerWebScrapingService, WebScrapingScopeService
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Writing_Interop import LocalWritingService, ServerWritingService, WritingScopeService
from tldw_chatbook.Subscriptions import (
    LocalWatchlistsService,
    ServerWatchlistsService,
    WatchlistScopeService,
)
from tldw_chatbook.Translation_Interop import ServerTranslationService, TranslationScopeService
from tldw_chatbook.Voice_Assistant_Interop import ServerVoiceAssistantService, VoiceAssistantScopeService
from tldw_chatbook.Evaluations_Interop import (
    EvaluationScopeService,
    LocalEvaluationsService,
    ServerEvaluationsService,
)
from tldw_chatbook.runtime_policy.bootstrap import (
    add_runtime_policy_snapshot,
    build_runtime_api_client,
    load_runtime_policy_for_app,
    reconcile_saved_screen_state,
    set_authoritative_runtime_source,
)
from tldw_chatbook.runtime_policy.server_capabilities import ActiveServerCapabilityService
from tldw_chatbook.runtime_policy.server_context import RuntimeServerContextProvider
from tldw_chatbook.runtime_policy.server_credentials import (
    CredentialStoreUnavailable,
    UnavailableServerCredentialStore,
    build_default_server_credential_store,
)
from tldw_chatbook.runtime_policy.server_event_scope import event_principal_id_from_active_context
from tldw_chatbook.runtime_policy.server_parity_state import (
    ServerParityStateRepositories,
    build_server_parity_state_repositories,
)
from tldw_chatbook.runtime_policy.engine import PolicyEngine
from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY
from tldw_chatbook.runtime_policy.types import PolicyDecision, RuntimeSourceState
from tldw_chatbook.state import AppState
from tldw_chatbook.Auth_Account_Interop import AuthAccountScopeService, ServerAuthAccountService
from tldw_chatbook.Audio_Services_Interop import (
    AudioServicesScopeService,
    LocalAudioServicesService,
    ServerAudioServicesService,
)
from .Evals.eval_orchestrator import EvaluationOrchestrator

if TYPE_CHECKING:
    from tldw_chatbook.tldw_api import MCPUnifiedClient

API_IMPORTS_SUCCESSFUL = True

SEARCH_VIEW_RAG_QA = "search-view-rag-qa"
SEARCH_NAV_RAG_QA = "search-nav-rag-qa"
SEARCH_NAV_RAG_CHAT = "search-nav-rag-chat"
SEARCH_NAV_RAG_MANAGEMENT = "search-nav-rag-management"
SEARCH_NAV_WEB_SEARCH = "search-nav-web-search"
SEARCH_NAV_EMBEDDINGS_CREATE = "search-nav-embeddings-create"
SEARCH_NAV_EMBEDDINGS_MANAGE = "search-nav-embeddings-manage"

DEFERRED_AUDIO_SERVICE_DELAY_SECONDS = 0.1
DEFERRED_DB_SIZE_UPDATE_DELAY_SECONDS = 0.1
DEFERRED_MEDIA_CLEANUP_DELAY_SECONDS = 5.0
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

# Early logging configuration removed - handled by configure_application_logging() during app initialization


class ThemeProvider(Provider):
    """A command provider for theme switching."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the ThemeProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        """Search for theme commands."""
        matcher = self.matcher(query)
        
        # Always show the main "Change Theme" command
        main_command_score = matcher.match("Theme: Change Theme")
        if main_command_score > 0:
            yield Hit(
                main_command_score,
                matcher.highlight("Theme: Change Theme"),
                partial(self.show_theme_submenu),
                help="Open theme selection menu"
            )
        
        # Only show individual themes if user is specifically searching for theme-related terms
        if any(term in query.lower() for term in ["switch", "theme", "dark", "light", "color", "solarized", "gruvbox", "dracula"]):
            # Get available theme names from registered themes
            available_themes = ["textual-dark", "textual-light"]  # Built-in themes
            # Add custom themes from ALL_THEMES
            for theme in ALL_THEMES:
                theme_name = theme.name if hasattr(theme, 'name') else str(theme)
                available_themes.append(theme_name)
            
            for theme_name in available_themes:
                command_text = f"Theme: Switch to {theme_name.replace('_', ' ').replace('-', ' ').title()}"
                score = matcher.match(command_text)
                if score > 0:
                    yield Hit(
                        score * 0.9,  # Slightly lower priority than main command
                        matcher.highlight(command_text),
                        partial(self.switch_theme, theme_name),
                        help=f"Change theme to {theme_name}"
                    )
    
    async def discover(self) -> Hits:
        """Show only the main theme command when palette is first opened."""
        yield Hit(
            1.0,
            "Theme: Change Theme",
            partial(self.show_theme_submenu),
            help="Open theme selection menu"
        )
    
    def show_theme_submenu(self) -> None:
        """Show a notification with instruction to search for themes."""
        self.app.notify("Type 'theme' in the command palette to see all available themes", severity="information")
    
    def switch_theme(self, theme_name: str) -> None:
        """Switch to the specified theme and save to config."""
        try:
            self.app.theme = theme_name
            self.app.notify(f"Theme changed to {theme_name}", severity="information")
            
            # Save the theme preference to config
            from .config import save_setting_to_cli_config
            save_setting_to_cli_config("general", "default_theme", theme_name)
            
        except Exception as e:
            self.app.notify(f"Failed to apply theme: {e}", severity="error")


def _navigate_via_screen(
    app: App,
    route: str,
    success_message: str,
    screen_context: dict[str, object] | None = None,
) -> None:
    """Navigate through the screen router so palette commands work in shell mode."""
    app.post_message(NavigateToScreen(route, screen_context))
    app.notify(success_message, severity="information")


class TabNavigationProvider(Provider):
    """Provider for tab navigation commands."""

    TAB_HELP_TEXT = {
        TAB_HOME: "Open Home for notifications, status, and next-best actions",
        TAB_CHAT: "Open Console for live agent work, approvals, tools, and RAG",
        TAB_LIBRARY: "Open Library for source material, imports, notes, media, conversations, and Search/RAG",
        TAB_ARTIFACTS: "Open Artifacts for generated outputs, reports, datasets, and Chatbooks",
        TAB_PERSONAS: "Open Personas for characters, prompts, dictionaries, and behavior profiles",
        TAB_WATCHLISTS_COLLECTIONS: "Open Watchlists for monitored sources, runs, alerts, and recovery",
        TAB_SCHEDULES: "Open Schedules for run timing, triggers, pauses, retries, and recovery",
        TAB_WORKFLOWS: "Open Workflows for reusable procedures, dry-runs, and outputs",
        TAB_MCP: "Open MCP for servers, tools, permissions, auth, and audit",
        TAB_ACP: "Open ACP for agents, sessions, runtimes, diffs, and terminals",
        TAB_SKILLS: "Open Skills for Agent Skills discovery, validation, and attachments",
        TAB_SETTINGS: "Open global preferences, appearance, accounts, storage, and app behavior",
        TAB_CCP: "Switch to Personas for characters, personas, prompts, dictionaries, and world books",
        TAB_MEDIA: "Switch to media library",
        TAB_SEARCH: "Switch to search and RAG",
        TAB_INGEST: "Switch to content ingestion",
        TAB_EVALS: "Switch to evaluation tools",
        TAB_LLM: "Switch to model and provider management",
        TAB_STTS: "Switch to speech-to-text and text-to-speech tools",
        TAB_STUDY: "Switch to flashcards and quizzes",
        TAB_WRITING: "Switch to writing tools",
        TAB_RESEARCH: "Switch to research workflows",
        TAB_SUBSCRIPTIONS: "Switch to subscriptions and watchlists",
        TAB_CHATBOOKS: "Switch to portable Chatbook context packs",
        TAB_TOOLS_SETTINGS: "Open MCP for legacy tools and settings",
        TAB_LOGS: "Switch to application logs",
        TAB_CODING: "Switch to coding assistant",
        TAB_STATS: "Switch to statistics view",
        TAB_CUSTOMIZE: "Switch to appearance customization",
    }

    NAVIGATION_TABS = (
        TAB_HOME,
        TAB_CHAT,
        TAB_LIBRARY,
        TAB_ARTIFACTS,
        TAB_PERSONAS,
        TAB_WATCHLISTS_COLLECTIONS,
        TAB_SCHEDULES,
        TAB_WORKFLOWS,
        TAB_MCP,
        TAB_ACP,
        TAB_SKILLS,
        TAB_SETTINGS,
    )

    POPULAR_TABS = (
        TAB_HOME,
        TAB_CHAT,
        TAB_LIBRARY,
        TAB_ARTIFACTS,
        TAB_MCP,
        TAB_SETTINGS,
    )
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the TabNavigationProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    @classmethod
    def navigation_tab_ids(cls) -> tuple[str, ...]:
        return cls.NAVIGATION_TABS

    @classmethod
    def command_palette_tab_ids(cls) -> tuple[str, ...]:
        return tuple(dict.fromkeys(cls.NAVIGATION_TABS + tuple(ALL_TABS)))

    @staticmethod
    def route_for_tab(tab_id: str) -> str:
        route_aliases = {
            "llm": TAB_LLM,
            TAB_TOOLS_SETTINGS: TAB_MCP,
            TAB_MCP: TAB_MCP,
            TAB_SETTINGS: TAB_SETTINGS,
        }
        return route_aliases.get(tab_id, tab_id)

    @classmethod
    def _shell_destination_for_tab(cls, tab_id: str):
        from .UI.Navigation.shell_destinations import get_shell_destination, resolve_shell_route

        resolved = resolve_shell_route(cls.route_for_tab(tab_id))
        try:
            return get_shell_destination(resolved.destination_id)
        except KeyError:
            return None

    @classmethod
    def _shell_command_label(cls, tab_id: str, visible_label: str) -> str:
        destination = cls._shell_destination_for_tab(tab_id)
        if destination is None or destination.accessible_label == visible_label:
            return visible_label
        return f"{visible_label} ({destination.accessible_label})"

    @classmethod
    def _shell_help_text(cls, tab_id: str) -> str | None:
        destination = cls._shell_destination_for_tab(tab_id)
        if destination is None:
            return None
        return f"Open {destination.accessible_label} for {destination.purpose}"

    def _tab_command(self, tab_id: str) -> tuple[str, str, str]:
        label = get_tab_display_label(tab_id)
        command_label = self._shell_command_label(tab_id, label)
        help_text = self._shell_help_text(tab_id) or self.TAB_HELP_TEXT.get(tab_id, f"Switch to {label}")
        return f"Tab Navigation: Switch to {command_label}", tab_id, help_text
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        tab_commands = [self._tab_command(tab_id) for tab_id in self.command_palette_tab_ids()]
        
        for command_text, tab_id, help_text in tab_commands:
            score = max(matcher.match(command_text), matcher.match(help_text))
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.switch_tab, tab_id),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_tabs = [self._tab_command(tab_id) for tab_id in self.POPULAR_TABS]
        
        for command_text, tab_id, help_text in popular_tabs:
            yield Hit(
                1.0,
                command_text,
                partial(self.switch_tab, tab_id),
                help=help_text
            )
    
    def switch_tab(self, tab_id: str) -> None:
        """Switch to the specified tab."""
        try:
            route = self.route_for_tab(tab_id)
            self.app.post_message(NavigateToScreen(route))
            self.app.notify(f"Switched to {get_tab_display_label(tab_id)}", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to switch tab: {e}", severity="error")


class LLMProviderProvider(Provider):
    """Provider for LLM provider management commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the LLMProviderProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        # Get available providers from the app
        available_providers = AVAILABLE_PROVIDERS if 'AVAILABLE_PROVIDERS' in globals() else []
        
        provider_commands = [
            ("LLM Provider Management: Show Current Provider", None, "Display currently selected LLM provider"),
            ("LLM Provider Management: Test API Connection", None, "Test connection to current LLM provider"),
        ]
        
        # Add provider switching commands
        for provider in available_providers:
            provider_name = provider.replace('_', ' ').title()
            command_text = f"LLM Provider Management: Switch to {provider_name}"
            provider_commands.append((command_text, provider, f"Switch to {provider_name} provider"))
        
        for command_text, provider_id, help_text in provider_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_llm_command, provider_id, command_text),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_providers = ["OpenAI", "Anthropic", "Cohere", "Groq", "Ollama"]
        
        yield Hit(
            1.0,
            "LLM Provider Management: Show Current Provider",
            partial(self.handle_llm_command, None, "show_current"),
            help="Display currently selected LLM provider"
        )
        
        for provider in popular_providers:
            yield Hit(
                0.9,
                f"LLM Provider Management: Switch to {provider}",
                partial(self.handle_llm_command, provider, f"switch_{provider}"),
                help=f"Switch to {provider} provider"
            )
    
    def handle_llm_command(self, provider_id: str, command: str) -> None:
        """Handle LLM provider commands."""
        try:
            if provider_id is None or "show_current" in command:
                # Show current provider
                current = getattr(self.app, 'current_provider', 'Unknown')
                self.app.notify(f"Current LLM provider: {current}", severity="information")
            elif "test" in command.lower():
                # Test API connection (placeholder)
                self.app.notify("API connection test initiated", severity="information")
            else:
                # Switch provider (placeholder - would need to integrate with actual provider switching logic)
                self.app.notify(f"Provider switch to {provider_id} requested", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to execute LLM command: {e}", severity="error")


class QuickActionsProvider(Provider):
    """Provider for quick action commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the QuickActionsProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        quick_actions = [
            ("Quick Actions: New Chat Conversation", "new_chat", "Start a new chat conversation"),
            ("Quick Actions: New Character Chat", "new_character", "Start a new character-based conversation"),
            ("Quick Actions: New Note", "new_note", "Create a new note"),
            ("Quick Actions: Clear Current Chat", "clear_chat", "Clear the current chat conversation"),
            ("Quick Actions: Export Chat as Markdown", "export_chat", "Export current chat to markdown file"),
            ("Quick Actions: Import Media File", "import_media", "Import a new media file for processing"),
            ("Quick Actions: Search All Content", "search_all", "Search across all content"),
            ("Quick Actions: Refresh Database", "refresh_db", "Refresh database connections"),
        ]
        
        for command_text, action_id, help_text in quick_actions:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.execute_quick_action, action_id),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_actions = [
            ("Quick Actions: New Chat Conversation", "new_chat", "Start a new chat conversation"),
            ("Quick Actions: New Note", "new_note", "Create a new note"),
            ("Quick Actions: Search All Content", "search_all", "Search across all content"),
            ("Quick Actions: Import Media File", "import_media", "Import a new media file for processing"),
        ]
        
        for command_text, action_id, help_text in popular_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.execute_quick_action, action_id),
                help=help_text
            )
    
    def execute_quick_action(self, action_id: str) -> None:
        """Execute the specified quick action."""
        try:
            if action_id == "new_chat":
                _navigate_via_screen(self.app, TAB_CHAT, "Opened Console for a new conversation")
            elif action_id == "new_character":
                _navigate_via_screen(self.app, TAB_PERSONAS, "Opened Personas for character setup")
            elif action_id == "new_note":
                _navigate_via_screen(
                    self.app,
                    TAB_LIBRARY,
                    "Opened Library for a new note",
                    {LIBRARY_NAV_CONTEXT_NOTES_CREATE: True},
                )
            elif action_id == "search_all":
                _navigate_via_screen(self.app, TAB_SEARCH, "Opened Search/RAG")
            elif action_id == "import_media":
                _navigate_via_screen(self.app, TAB_INGEST, "Opened Import/Export for media import")
            else:
                self.app.notify(f"Quick action '{action_id}' initiated", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to execute quick action: {e}", severity="error")


class SettingsProvider(Provider):
    """Provider for settings and preferences commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the SettingsProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        settings_commands = [
            ("Settings & Preferences: Open Config File", "open_config", "Open the configuration file for editing"),
            ("Settings & Preferences: Reload Configuration", "reload_config", "Reload configuration from file"),
            ("Settings & Preferences: Toggle Streaming Mode", "toggle_streaming", "Toggle LLM streaming mode on/off"),
            ("Settings & Preferences: Set Temperature to Low (0.1)", "temp_low", "Set LLM temperature to 0.1 for focused responses"),
            ("Settings & Preferences: Set Temperature to Medium (0.7)", "temp_med", "Set LLM temperature to 0.7 for balanced responses"),
            ("Settings & Preferences: Set Temperature to High (1.0)", "temp_high", "Set LLM temperature to 1.0 for creative responses"),
            ("Settings & Preferences: Reset to Default Settings", "reset_defaults", "Reset all settings to default values"),
            ("Settings & Preferences: Show Database Stats", "db_stats", "Show database size and statistics"),
            ("Settings & Preferences: Open Settings Tab", "open_settings", "Navigate to Tools & Settings tab"),
        ]
        
        for command_text, setting_id, help_text in settings_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_setting, setting_id),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_settings = [
            ("Settings & Preferences: Open Settings Tab", "open_settings", "Navigate to Tools & Settings tab"),
            ("Settings & Preferences: Open Config File", "open_config", "Open the configuration file for editing"),
            ("Settings & Preferences: Show Database Stats", "db_stats", "Show database size and statistics"),
            ("Settings & Preferences: Toggle Streaming Mode", "toggle_streaming", "Toggle LLM streaming mode on/off"),
        ]
        
        for command_text, setting_id, help_text in popular_settings:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_setting, setting_id),
                help=help_text
            )
    
    def handle_setting(self, setting_id: str) -> None:
        """Handle settings commands."""
        try:
            if setting_id == "open_settings":
                _navigate_via_screen(self.app, TAB_SETTINGS, "Opened Settings")
            elif setting_id == "open_config":
                from .config import DEFAULT_CONFIG_PATH
                self.app.notify(f"Config file location: {DEFAULT_CONFIG_PATH}", severity="information")
            elif setting_id == "reload_config":
                self.app.notify("Configuration reload requested", severity="information")
            elif setting_id == "db_stats":
                self.app.notify("Database statistics display requested", severity="information")
            elif setting_id.startswith("temp_"):
                temp_map = {"temp_low": "0.1", "temp_med": "0.7", "temp_high": "1.0"}
                temp_value = temp_map.get(setting_id, "0.7")
                self.app.notify(f"Temperature set to {temp_value}", severity="information")
            else:
                self.app.notify(f"Settings action '{setting_id}' initiated", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to execute settings command: {e}", severity="error")


class CharacterProvider(Provider):
    """Provider for character and persona management commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the CharacterProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        character_commands = [
            ("Character/Persona Management: Create New Character", "new_character", "Create a new character or persona"),
            ("Character/Persona Management: Show All Characters", "list_characters", "Display all available characters"),
            ("Character/Persona Management: Switch Character", "switch_character", "Switch to a different character"),
            ("Character/Persona Management: Edit Current Character", "edit_character", "Edit the current character settings"),
            ("Character/Persona Management: Delete Character", "delete_character", "Delete a character (with confirmation)"),
            ("Character/Persona Management: Import Character", "import_character", "Import character from file"),
            ("Character/Persona Management: Export Character", "export_character", "Export character to file"),
            ("Character/Persona Management: Open Character Tab", "open_character_tab", "Navigate to Character Chat tab"),
        ]
        
        for command_text, action_id, help_text in character_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_character_action, action_id),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_character_actions = [
            ("Character/Persona Management: Open Character Tab", "open_character_tab", "Navigate to Character Chat tab"),
            ("Character/Persona Management: Create New Character", "new_character", "Create a new character or persona"),
            ("Character/Persona Management: Show All Characters", "list_characters", "Display all available characters"),
            ("Character/Persona Management: Switch Character", "switch_character", "Switch to a different character"),
        ]
        
        for command_text, action_id, help_text in popular_character_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_character_action, action_id),
                help=help_text
            )
    
    def handle_character_action(self, action_id: str) -> None:
        """Handle character management actions."""
        try:
            if action_id == "open_character_tab":
                _navigate_via_screen(self.app, TAB_PERSONAS, "Opened Personas")
            elif action_id == "new_character":
                _navigate_via_screen(self.app, TAB_PERSONAS, "Opened Personas to create a character")
            elif action_id == "list_characters":
                _navigate_via_screen(self.app, TAB_PERSONAS, "Opened Personas to list characters")
            else:
                self.app.notify(f"Character action '{action_id}' requested", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to execute character action: {e}", severity="error")


class MediaProvider(Provider):
    """Provider for media and content management commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the MediaProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        media_commands = [
            ("Media & Content: Open Media Library", "open_media", "Navigate to media library"),
            ("Media & Content: Recent Media Files", "recent_media", "Show recently added media files"),
            ("Media & Content: Search Transcripts", "search_transcripts", "Search through media transcripts"),
            ("Media & Content: Show Ingested Content", "show_ingested", "Display all ingested content"),
            ("Media & Content: Import New Media", "import_new", "Import new media file"),
            ("Media & Content: Open Media Database", "open_db", "View media database contents"),
            ("Media & Content: Refresh Media Library", "refresh_media", "Refresh media library"),
            ("Media & Content: Export Media List", "export_list", "Export media list to file"),
        ]
        
        for command_text, action_id, help_text in media_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_media_action, action_id),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_media_actions = [
            ("Media & Content: Open Media Library", "open_media", "Navigate to media library"),
            ("Media & Content: Import New Media", "import_new", "Import new media file"),
            ("Media & Content: Search Transcripts", "search_transcripts", "Search through media transcripts"),
            ("Media & Content: Recent Media Files", "recent_media", "Show recently added media files"),
        ]
        
        for command_text, action_id, help_text in popular_media_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_media_action, action_id),
                help=help_text
            )
    
    def handle_media_action(self, action_id: str) -> None:
        """Handle media management actions."""
        try:
            if action_id == "open_media":
                _navigate_via_screen(self.app, TAB_MEDIA, "Opened Media Library")
            elif action_id == "import_new":
                _navigate_via_screen(self.app, TAB_INGEST, "Opened Import/Export for media import")
            elif action_id == "search_transcripts":
                _navigate_via_screen(self.app, TAB_SEARCH, "Opened Search/RAG for transcript search")
            else:
                self.app.notify(f"Media action '{action_id}' requested", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to execute media action: {e}", severity="error")


class DeveloperProvider(Provider):
    """Provider for developer and debug commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the DeveloperProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        dev_commands = [
            ("Developer/Debug Commands: Show App Info", "app_info", "Display application version and build info"),
            ("Developer/Debug Commands: Open Log File", "open_logs", "Navigate to application logs"),
            ("Developer/Debug Commands: Clear Cache", "clear_cache", "Clear application cache"),
            ("Developer/Debug Commands: Show Keybindings", "show_keys", "Display all keyboard shortcuts"),
            ("Developer/Debug Commands: Debug Mode Toggle", "toggle_debug", "Toggle debug mode on/off"),
            ("Developer/Debug Commands: Memory Usage", "memory_usage", "Show current memory usage"),
            ("Developer/Debug Commands: Database Integrity Check", "db_check", "Check database integrity"),
            ("Developer/Debug Commands: Export Debug Info", "export_debug", "Export debug information to file"),
        ]
        
        for command_text, action_id, help_text in dev_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_dev_action, action_id),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_dev_actions = [
            ("Developer/Debug Commands: Open Log File", "open_logs", "Navigate to application logs"),
            ("Developer/Debug Commands: Show App Info", "app_info", "Display application version and build info"),
            ("Developer/Debug Commands: Show Keybindings", "show_keys", "Display all keyboard shortcuts"),
        ]
        
        for command_text, action_id, help_text in popular_dev_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_dev_action, action_id),
                help=help_text
            )
    
    def handle_dev_action(self, action_id: str) -> None:
        """Handle developer/debug actions."""
        try:
            if action_id == "open_logs":
                _navigate_via_screen(self.app, TAB_LOGS, "Opened Logs")
            elif action_id == "app_info":
                self.app.notify("tldw_chatbook - TUI for LLM interactions", severity="information")
            elif action_id == "show_keys":
                self.app.notify("Keybindings: Ctrl+Q (quit), Ctrl+P (palette)", severity="information")
            elif action_id == "clear_cache":
                self.app.notify("Cache clear requested", severity="information")
            else:
                self.app.notify(f"Developer action '{action_id}' initiated", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to execute developer action: {e}", severity="error")


# --- Placeholder Window for Lazy Loading ---
class PlaceholderWindow(Container):
    """A lightweight placeholder that defers actual window creation until needed."""
    
    def __init__(self, app_instance: 'TldwCli', window_class: type, window_id: str, classes: str = "") -> None:
        """Initialize placeholder with window creation parameters."""
        super().__init__(id=window_id, classes=f"placeholder-window {classes}")
        self.app_instance = app_instance
        self.window_class = window_class
        self.window_id = window_id
        self.actual_classes = classes
        self._actual_window = None
        self._initialized = False
        # Log placeholder creation
        logger.debug(f"PlaceholderWindow created for {window_id} (class: {window_class.__name__})")
    
    def initialize(self) -> None:
        """Create and mount the actual window widget."""
        if self._initialized:
            return
            
        logger.info(f"Initializing actual window for {self.window_id}")
        start_time = time.perf_counter()
        
        try:
            # Remove the loading placeholder first
            for child in list(self.children):
                child.remove()
            
            # Create the actual window
            # EvalsLab, EvalsWindow and EvalsWindowV3 are Containers that take app_instance as keyword argument
            if self.window_class.__name__ in ['EvalsLab', 'EvalsWindow', 'EvalsWindowV3']:
                self._actual_window = self.window_class(app_instance=self.app_instance, id=self.window_id, classes=self.actual_classes)
            else:
                self._actual_window = self.window_class(self.app_instance, id=self.window_id, classes=self.actual_classes)
            
            # Clear placeholder styling and mount actual window
            self.remove_class("placeholder-window")
            # Set proper layout for the container AND make it visible
            self.styles.layout = "vertical"
            self.styles.height = "100%"
            self.styles.width = "100%"
            self.styles.display = "block"  # CRITICAL: Reset display to block after removing placeholder class
            
            # Make sure the actual window fills the container
            self._actual_window.styles.height = "100%"
            self._actual_window.styles.width = "100%"
            self._actual_window.styles.display = "block"  # Ensure the actual window is visible
            
            self.mount(self._actual_window)
            self._initialized = True
            
            # Populate widgets for specific windows after initialization
            self._populate_window_widgets()
            
            # Log timing
            duration = time.perf_counter() - start_time
            log_histogram("lazy_window_initialization_seconds", duration,
                         labels={"window": self.window_id.replace("-window", "")},
                         documentation="Time to initialize lazy-loaded window")
            logger.info(f"Window {self.window_id} initialized in {duration:.3f} seconds")
            
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to initialize window {self.window_id}: {str(e)}")
            # Clear any existing children before showing error
            for child in list(self.children):
                child.remove()
            self.mount(Static(f"Error loading {self.window_id}: {str(e)}", classes="error"))
    
    def _populate_window_widgets(self) -> None:
        """Populate widgets for specific windows after they're initialized."""
        # Don't populate widgets here - let the watch_current_tab handle it
        # This prevents timing issues where widgets aren't ready yet
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if the actual window has been initialized."""
        return self._initialized
    
    def compose(self) -> ComposeResult:
        """Show a loading message until initialized."""
        if not self._initialized:
            yield Static("Loading...", classes="loading-placeholder")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Proxy button presses to the actual window if initialized."""
        if self._initialized and self._actual_window:
            if hasattr(self._actual_window, 'on_button_pressed'):
                result = self._actual_window.on_button_pressed(event)
                if hasattr(result, '__await__'):
                    await result


def _sanitize_library_ingest_error_text(message: str) -> str:
    """Reduce a raw error message to a single-line, ``<=200``-char string.

    Shared building block for both ingest-pipeline stages (F3): the write
    stage has a real ``Exception`` (see ``_sanitize_library_ingest_error``,
    below); the parse stage only has the already-``str()``-ed message a
    pool worker's structured failure result carries across the process
    boundary (``ingest_parse_worker.run_parse_job``'s ``"error"`` key) --
    both need the exact same single-line/200-cap treatment before landing
    in a job's ``LibraryIngestJob.error`` field.

    Args:
        message: The raw (possibly multi-line, possibly empty) message.

    Returns:
        The first line, stripped and capped at 200 characters. ``""`` when
        ``message`` is empty or all-whitespace.
    """
    message = message.strip()
    first_line = message.splitlines()[0].strip() if message else ""
    return first_line[:200]


def _sanitize_library_ingest_error(exc: Exception) -> str:
    """Reduce an ingest-time exception to a single-line, capped error string.

    Args:
        exc: The exception raised by the ingest seam.

    Returns:
        The first line of ``str(exc)``, stripped and capped at 200
        characters. Falls back to the exception's class name when
        ``str(exc)`` is empty.
    """
    sanitized = _sanitize_library_ingest_error_text(str(exc))
    return sanitized if sanitized else exc.__class__.__name__[:200]


def _stream_fileno(stream: Any) -> int:
    """Best-effort file descriptor for a possibly-fake stream object.

    Args:
        stream: Anything shaped like a text stream (may be Textual's
            stderr capture object, a pytest capture stream, ``None``, ...).

    Returns:
        The stream's OS-level fd when ``fileno()`` returns a real one;
        ``-1`` when the stream is missing/``None``, ``fileno()`` raises,
        or -- the case that actually bit in production -- ``fileno()``
        returns a non-fd sentinel like ``-1`` without raising (Textual's
        capture object does exactly that).
    """
    try:
        fd = stream.fileno()
    except Exception:
        return -1
    return fd if isinstance(fd, int) and fd >= 0 else -1


# The detect_file_type() values whose parse worker runs transcription
# (see Local_Ingestion/local_file_ingestion.py audio/video branches). The
# heavy-lane cap limits how many of these parse concurrently.
_INGEST_HEAVY_TYPES = frozenset({"audio", "video"})

# Cap on how many persisted ingest jobs `_restore_ingest_jobs` carries
# forward on restart (see `Library.library_ingest_jobs.plan_restore`) --
# keeps startup and the in-memory registry bounded for a long-lived store.
_MAX_PERSISTED_INGEST_JOBS = 500


# Keep-alive singleton for `_ingest_pool_real_stderr`'s devnull fallback.
# Module-level on purpose: the multiprocessing resource tracker inherits this
# fd ONCE at its (process-global, once-per-process) launch and keeps writing
# to it for the rest of the process's life -- if the handle were a local that
# got garbage-collected, the OS could reuse the fd number and the tracker's
# error output would silently corrupt an unrelated file.
_INGEST_POOL_STDERR_FALLBACK = None


def _ingest_pool_real_stderr():
    """Return a stream with a REAL file descriptor to stand in for stderr.

    Used by ``LibraryIngestQueueMixin._create_ingest_parse_pool`` when
    ``sys.stderr`` has no usable fd (Textual app mode / textual-serve
    replace it with a capture object whose ``fileno()`` returns ``-1``
    without raising -- see that method's docstring for the crash this
    caused). Preference order:

    1. ``sys.__stderr__`` -- the process's ORIGINAL stderr, still fd-backed
       even after Textual swaps ``sys.stderr`` (Textual redirects the
       high-level name, not the OS-level fd).
    2. A process-lifetime ``os.devnull`` handle (see
       ``_INGEST_POOL_STDERR_FALLBACK``'s comment for why it must stay
       referenced) -- ``sys.__stderr__`` can itself be ``None``/fd-less in
       exotic embed/frozen environments.
    """
    real = sys.__stderr__
    if real is not None and _stream_fileno(real) >= 0:
        return real
    global _INGEST_POOL_STDERR_FALLBACK
    if _INGEST_POOL_STDERR_FALLBACK is None:
        _INGEST_POOL_STDERR_FALLBACK = open(os.devnull, "w")
    return _INGEST_POOL_STDERR_FALLBACK


class LibraryIngestQueueMixin:
    """Library ingest job submission seam + parallel-parse coordinator + writer.

    Mixed into :class:`TldwCli` (and headless test harnesses -- see
    ``Tests/Library/test_library_ingest_runner.py``) rather than being
    defined directly on the App class, so the coordinator + writer can be
    exercised without booting the full app. A host class is expected to
    provide:

    - ``self.library_ingest_jobs``: a ``LibraryIngestJobRegistry`` instance
      constructed once (e.g. in ``__init__``/app wiring).
    - ``self.media_db``: an ``Optional[MediaDatabase]``.
    - ``self._ingest_parse_pool``, ``self._ingest_parsed_payloads``,
      ``self._ingest_parse_pool_generation``,
      ``self._ingest_parse_jobs_by_generation``, and
      ``self._ingest_shutdown``: the coordinator's own state, initialized
      once alongside ``library_ingest_jobs`` -- see ``TldwCli.__init__``.
    - Textual's ``App``/``Widget`` worker machinery (``@work`` and
      ``call_from_thread``), since this mixin is always combined with one
      of those base classes.

    F3 architecture -- two decoupled stages, not one serial loop:

    - **Parse stage (this mixin's coordinator, UI thread).** A lazily
      created spawn-context ``multiprocessing.Pool`` (see
      ``_create_ingest_parse_pool``) fans file parsing out to worker
      processes. ``_top_up_ingest_parse_pool`` keeps up to N jobs (the pool
      size) ``PARSING`` at once -- called after every submission/retry and
      after every parse completion. A pool completion is marshaled onto the
      UI thread (``_on_ingest_parse_complete``); success stashes the parsed
      payload and wakes the writer, failure goes straight to
      ``mark_failed``.
    - **Write stage (the writer, background thread, unchanged shape).**
      Exactly one job is ever being written at a time (SQLite has one
      writer). The writer's claim-or-release loop
      (``_claim_next_ingest_job_or_release`` / ``_run_library_ingest_queue``)
      now claims the OLDEST payload-ready job (by submission order) instead
      of the oldest queued one, persists it via ``persist_parsed_media``,
      and marks it ``DONE``/``FAILED``.

    The coordinator (parse side) and the writer (write side) are the only
    intended callers of the registry's ``mark_parsing``/``mark_writing``/
    ``mark_done``/``mark_failed`` transition methods, respectively. Every
    job is driven either ``queued`` -> ``parsing`` -> ``writing`` ->
    ``done``/``failed``, or ``parsing`` -> ``failed`` directly when the pool
    worker's parse itself fails (e.g. an unsupported/undetectable file type,
    or a missing source file -- classified by ``classify_parse_failure``
    inside the worker, where the real exception type is available). Either
    way, one job's failure is isolated so it never strands a later queued
    job or blocks the writer.

    Shutdown (quit path) order, in ``_shutdown_ingest_parse_pool`` (called
    from ``TldwCli.on_unmount``): (1) ``_ingest_shutdown = True`` + pool
    reference detached, synchronously -- pool callbacks short-circuit on
    the result-handler thread before ever marshaling; (2)
    ``pool.terminate()`` + ``pool.join()`` on a detached daemon thread,
    never the event-loop thread (deadlock rationale in that method's
    docstring); (3) the writer thread is swept afterward by ``on_unmount``'s
    generic worker cancellation, its in-flight DB write completing as
    before. Steps 2 and 3 run concurrently -- safe because the two stages
    share no resources (parse workers never touch ``media_db``; the writer
    never touches the pool).
    """

    def _restore_ingest_jobs(self) -> None:
        """One-time on_mount restore of persisted ingest job history."""
        from datetime import datetime, timezone
        from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
        from tldw_chatbook.Library.library_ingest_jobs import plan_restore
        try:
            store = LibraryIngestJobsDB(get_library_ingest_jobs_db_path())
            self._library_ingest_jobs_store = store
            # Do ALL fallible work -- corrupt read, plan, and the store
            # reconcile writes -- BEFORE touching the in-memory registry, so any
            # failure leaves the registry empty + store unattached: a clean
            # in-memory fallback that matches the "starting empty" warning below
            # (rather than a half-restored registry contradicting the log).
            plan = plan_restore(
                store.all_jobs(),
                max_persisted=_MAX_PERSISTED_INGEST_JOBS,
                now_iso=datetime.now(timezone.utc).isoformat(),
            )
            for job in plan.upsert:
                store.upsert_job(job)
            for job_id in plan.delete_ids:
                store.delete_job(job_id)
            self.library_ingest_jobs.restore(plan.jobs, plan.next_id)
            self.library_ingest_jobs.attach_store(store)
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to restore persisted ingest job history; starting empty."
            )

    def submit_library_ingest_job(
        self,
        *,
        source_path: str,
        title: str = "",
        author: str = "",
        keywords: tuple[str, ...] = (),
        perform_analysis: bool = False,
        chunk_enabled: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> LibraryIngestJob:
        """Submit a new Library ingest job and top up the parse pool.

        UI-thread only. Appends a ``QUEUED`` job to ``self.library_ingest_jobs``.
        When ``self.media_db`` is unavailable, the job is failed immediately
        (with the exact copy ``"Media database is unavailable."``) and it
        never reaches the parse pool.

        Args:
            source_path: The file path to ingest.
            title: Optional title form field.
            author: Optional author form field.
            keywords: Keywords form field.
            perform_analysis: Whether to run post-ingest analysis.
            chunk_enabled: Whether to chunk the ingested content.
            chunk_size: Requested chunk size when ``chunk_enabled``.

        Returns:
            The newly created job: ``QUEUED`` normally, or immediately
            ``FAILED`` when ``media_db`` is unavailable.
        """
        try:
            detected_type = classify_ingest_source(source_path) or ""
        except FileIngestionError:
            # Expected for an unsupported extension -- treat as light work.
            detected_type = ""
        except Exception:
            # An UNEXPECTED classification failure must not silently disable the
            # heavy-lane cap (an empty type is treated as light work, so a
            # misclassified audio/video job would bypass the transcription cap).
            # Log it so a regression is observable, then fall back to light.
            logger.opt(exception=True).warning(
                f"classify_ingest_source failed unexpectedly for {source_path!r}; "
                "treating as light work (heavy-lane cap may not apply)."
            )
            detected_type = ""
        job = self.library_ingest_jobs.submit(
            source_path=source_path,
            title=title,
            author=author,
            keywords=keywords,
            perform_analysis=perform_analysis,
            chunk_enabled=chunk_enabled,
            chunk_size=chunk_size,
            detected_type=detected_type,
        )
        if self.media_db is None:
            failed = self.library_ingest_jobs.mark_failed(
                job.job_id, error="Media database is unavailable."
            )
            return failed if failed is not None else job
        self._top_up_ingest_parse_pool()
        return job

    def retry_library_ingest_job(self, job_id: str) -> Optional[LibraryIngestJob]:
        """Requeue a previously failed job and top up the parse pool.

        UI-thread only. A thin wrapper over
        ``LibraryIngestJobRegistry.requeue`` -- a no-op (returns ``None``)
        when ``job_id`` is unknown or the job is not currently ``FAILED``.

        Args:
            job_id: The failed job to requeue.

        Returns:
            The newly appended ``QUEUED`` job (or immediately ``FAILED``
            when ``media_db`` is unavailable), or ``None`` when nothing was
            requeued.
        """
        requeued = self.library_ingest_jobs.requeue(job_id)
        if requeued is None:
            return None
        if self.media_db is None:
            failed = self.library_ingest_jobs.mark_failed(
                requeued.job_id, error="Media database is unavailable."
            )
            return failed if failed is not None else requeued
        self._top_up_ingest_parse_pool()
        return requeued

    # -- Parse-pool sizing + lifecycle (coordinator) -----------------------

    def _ingest_parse_worker_count(self) -> int:
        """Resolve the parse-pool size from config, with a safe default.

        UI-thread only. Reads ``library.ingest_parse_workers`` via the
        dotted 1-arg ``get_cli_setting`` form (``load_settings()`` doesn't
        carry CLI ``[library.*]`` tables -- same bug-class guard as the
        rail-state read). An invalid, missing, or non-positive value falls
        back to the spec's default formula.

        Returns:
            The configured worker count when it int-coerces to a positive
            value; otherwise ``min(3, max(1, cpu_count - 1))``, where
            ``cpu_count`` is ``os.cpu_count()`` (guarded to ``2`` when that
            returns ``None``, e.g. on some containerized/sandboxed hosts).
        """
        try:
            configured = int(get_cli_setting("library.ingest_parse_workers"))
        except (TypeError, ValueError):
            configured = 0
        if configured > 0:
            return configured
        cpu_count = os.cpu_count() or 2
        return min(3, max(1, cpu_count - 1))

    def _ingest_heavy_lane_max_workers(self) -> int:
        """Resolve the heavy-lane (audio/video transcription) cap from config.

        UI-thread only. Reads ``library.ingest_heavy_lane_max_workers`` via the
        dotted 1-arg ``get_cli_setting`` form (same reason as
        ``_ingest_parse_worker_count``). Defaults to 1; a missing, invalid, or
        non-positive value clamps to 1 so heavy work is never permanently
        starved.
        """
        try:
            configured = int(get_cli_setting("library.ingest_heavy_lane_max_workers"))
        except (TypeError, ValueError):
            configured = 0
        return configured if configured > 0 else 1

    def _create_ingest_parse_pool(self):
        """Create the Library ingest parse pool.

        UI-thread only. Test seam: monkeypatched to an inline-synchronous
        fake pool (see ``Tests/Library/test_library_ingest_runner.py``) so
        pilots stay deterministic without spawning real OS processes. Real
        callers get a spawn-context ``multiprocessing.Pool`` sized by
        ``_ingest_parse_worker_count``.

        Not a ``concurrent.futures.ProcessPoolExecutor`` -- see the F3
        design spec's Architecture section: the executor's ``atexit`` hook
        joins running tasks, so an in-flight long transcription would block
        app exit for its full duration. ``Pool`` has a public
        ``terminate()`` the quit path relies on instead.

        Textual stderr workaround (live-QA crash fix): under Textual (app
        mode / textual-serve), ``sys.stderr`` is replaced by a capture
        object whose ``fileno()`` returns ``-1`` WITHOUT raising. CPython
        3.12's ``multiprocessing.resource_tracker._launch`` appends
        ``sys.stderr.fileno()`` to the fds it hands
        ``util.spawnv_passfds`` (its ``except Exception`` guard never
        fires, since ``-1`` is returned rather than raised), and
        ``spawnv_passfds`` rejects the list with ``ValueError: bad
        value(s) in fds_to_keep`` -- so the very first Pool construction
        (which ensure-runs the process-global resource tracker) crashed
        the app on its first ingest submission. When ``sys.stderr`` has no
        usable fd, the Pool is constructed under
        ``contextlib.redirect_stderr`` pointing at a genuinely fd-backed
        stream (``_ingest_pool_real_stderr``: ``sys.__stderr__``, else a
        kept-alive devnull handle). The tracker launches at most once per
        process, so covering construction is sufficient -- and applying
        the redirect on every (re)construction is harmless.
        """
        ctx = multiprocessing.get_context("spawn")
        processes = self._ingest_parse_worker_count()
        if _stream_fileno(sys.stderr) >= 0:
            return ctx.Pool(processes=processes)
        with contextlib.redirect_stderr(_ingest_pool_real_stderr()):
            return ctx.Pool(processes=processes)

    def _ensure_ingest_parse_pool(self):
        """Return the current parse pool, lazily creating one if needed.

        UI-thread only.
        """
        if self._ingest_parse_pool is None:
            pool = self._create_ingest_parse_pool()
            try:
                sentinels = self._ingest_parse_pool_worker_sentinels(pool)
            except Exception:
                self._terminate_ingest_parse_pool_off_thread(pool)
                raise

            generation = getattr(self, "_ingest_parse_pool_generation", 0) + 1
            stop_event = threading.Event()
            self._ingest_parse_pool_generation = generation
            self._ingest_parse_jobs_by_generation = getattr(
                self, "_ingest_parse_jobs_by_generation", {}
            )
            self._ingest_parse_jobs_by_generation[generation] = set()
            self._ingest_parse_pool_stop_event = stop_event
            self._ingest_parse_pool = pool
            if sentinels:
                self._start_ingest_parse_pool_monitor(
                    generation, sentinels, stop_event
                )
        return self._ingest_parse_pool

    @staticmethod
    def _ingest_parse_pool_worker_sentinels(pool: Any) -> Optional[tuple[Any, ...]]:
        """Snapshot real Pool worker sentinels; injected fakes may opt out."""
        workers = getattr(pool, "_pool", None)
        if workers is None:
            return None
        try:
            sentinels = tuple(worker.sentinel for worker in workers)
        except Exception as exc:
            raise RuntimeError("Could not inspect parse-pool worker sentinels.") from exc
        if not sentinels:
            raise RuntimeError("Parse pool started without worker sentinels.")
        return sentinels

    def _start_ingest_parse_pool_monitor(
        self,
        generation: int,
        sentinels: tuple[Any, ...],
        stop_event: threading.Event,
    ) -> threading.Thread:
        """Watch one real Pool generation for an unexpected worker exit."""

        def _monitor() -> None:
            try:
                ready = multiprocessing.connection.wait(sentinels)
            except Exception as exc:
                if stop_event.is_set() or self._ingest_shutdown:
                    return
                failure = RuntimeError(f"Parse-pool sentinel monitor failed: {exc}")
            else:
                if not ready or stop_event.is_set() or self._ingest_shutdown:
                    return
                failure = RuntimeError(
                    f"Library ingest parse-pool worker exited unexpectedly "
                    f"(generation {generation})."
                )
            if stop_event.is_set() or self._ingest_shutdown:
                return
            self.call_from_thread(
                self._handle_broken_ingest_parse_pool,
                generation,
                None,
                failure,
            )

        thread = threading.Thread(
            target=_monitor,
            name=f"library-ingest-pool-monitor-{generation}",
            daemon=True,
        )
        thread.start()
        return thread

    @staticmethod
    def _ingest_job_options(job: LibraryIngestJob) -> Dict[str, Any]:
        """Build ``run_parse_job``'s ``options`` dict from a job's fields.

        Mechanical 1:1 translation documented in
        ``ingest_parse_worker``'s module docstring -- the Library queue
        never sets ``custom_prompt``/``system_prompt``/``api_name``/
        ``api_key``/``metadata``, so they're simply absent (``None`` inside
        the worker's ``options.get(...)`` reads).
        """
        return {
            "title": job.title or None,
            "author": job.author or None,
            "keywords": list(job.keywords) or None,
            "perform_analysis": job.perform_analysis,
            "chunk_options": (
                {
                    "method": "sentences",
                    "size": job.chunk_size,
                    "overlap": 100,
                }
                if job.chunk_enabled
                else None
            ),
        }

    def _top_up_ingest_parse_pool(self) -> None:
        """Submit ``QUEUED`` jobs to the parse pool up to the worker cap.

        UI-thread only. Called after every submission/retry and after every
        parse completion (ok or not) so the pool stays saturated at up to N
        concurrent ``PARSING`` jobs -- this cap IS the backpressure: at most
        N parsed payloads (plus the one currently being written) are ever
        held in memory at once.

        A no-op once ``self._ingest_shutdown`` is set (the app is closing;
        no new parse work should be handed to a pool that's about to be
        terminated).

        ``classify_ingest_source`` is called once, at enqueue time (in
        ``submit_library_ingest_job``), and its result is stamped onto the
        job's ``detected_type`` -- not recomputed here. Dispatch reuses that
        stored value both to claim the job (``mark_parsing``) and to decide
        eligibility under the heavy-lane gate below; an unsupported
        extension at enqueue time is silently left ``""`` rather than
        fast-failing the job: real classification (permanent vs. retryable)
        happens inside the pool worker, where the authoritative exception
        is available (see ``classify_parse_failure``), matching the F3
        design spec's "permanent-vs-retryable classification happens inside
        the worker" decision.

        Heavy-lane gate: at most ``_ingest_heavy_lane_max_workers()`` jobs
        whose ``detected_type`` is in ``_INGEST_HEAVY_TYPES`` (audio/video
        transcription) may be ``PARSING`` at once, independent of the
        overall pool cap -- when that lane is full, ``next_queued`` is asked
        to skip those types so a queued document can fill the slot instead,
        letting document parses fan out wide while transcriptions stay
        capped.
        """
        if self._ingest_shutdown:
            return
        worker_count = self._ingest_parse_worker_count()
        heavy_cap = self._ingest_heavy_lane_max_workers()
        # Read the total + heavy in-flight counts ONCE, then track them locally
        # as we dispatch. This whole method is UI-thread-only and synchronous,
        # so nothing but our own mark_parsing() calls can change these counts
        # mid-loop -- re-scanning the registry (O(N)) every iteration would make
        # the loop O(worker_count * N) on the UI thread for no benefit.
        parsing_count = self.library_ingest_jobs.counts().get("parsing", 0)
        heavy_parsing_count = self.library_ingest_jobs.parsing_count_for_types(
            _INGEST_HEAVY_TYPES
        )
        while parsing_count < worker_count:
            heavy_full = heavy_parsing_count >= heavy_cap
            job = self.library_ingest_jobs.next_queued(
                skip_types=_INGEST_HEAVY_TYPES if heavy_full else frozenset()
            )
            if job is None:
                return
            claimed = self.library_ingest_jobs.mark_parsing(
                job.job_id, detected_type=job.detected_type
            )
            if claimed is None:
                # Invariant violation (Task-3 reviewer's guard note): the
                # job we just pulled off `next_queued()` was no longer
                # QUEUED by the time we tried to claim it -- should be
                # impossible on the UI thread (this whole method is
                # UI-thread-only, so nothing else can race the queue
                # between the two calls), but a coordinator bug here must
                # never crash the submission path. `break`, not `continue`
                # (whole-branch review, Minor 2): `next_queued()` always
                # returns the OLDEST queued job, so a `continue` would get
                # the exact same unclaimable job handed straight back --
                # an infinite loop on the UI thread. Breaking abandons
                # only this top-up pass (logged); the next submission/
                # retry/parse-completion re-attempts from scratch.
                logger.error(
                    f"Library ingest coordinator: mark_parsing rejected "
                    f"job {job.job_id} (expected QUEUED) -- abandoning "
                    f"this top-up pass."
                )
                break
            # Track the just-claimed job locally (mirrors what a fresh
            # counts()/parsing_count_for_types() scan would report next
            # iteration) so the loop stays O(N), not O(worker_count * N).
            parsing_count += 1
            if job.detected_type in _INGEST_HEAVY_TYPES:
                heavy_parsing_count += 1
            options = self._ingest_job_options(claimed)
            job_id = claimed.job_id
            source_path = claimed.source_path
            try:
                pool = self._ensure_ingest_parse_pool()
            except Exception as exc:
                # CONTAINMENT (live-QA crash fix): pool CREATION itself
                # failed -- e.g. the spawn machinery raising at
                # construction time (the fileno-less-stderr resource-
                # tracker crash `_create_ingest_parse_pool` now works
                # around, or any environment-specific successor). This is
                # a UI-thread call reached synchronously from
                # submit/retry, so letting it propagate would crash the
                # app on the user's submission. Same containment
                # philosophy as `_handle_broken_ingest_parse_pool`, but
                # scoped to just the triggering job: no pool ever existed
                # here, so no OTHER job's parse was riding on it -- fail
                # this one retryable, keep the pool slot empty (the next
                # submit/retry attempts creation from scratch), and
                # return cleanly.
                logger.opt(exception=True).error(
                    f"Library ingest parse pool could not be created "
                    f"(job_id={job_id}, source={source_path})."
                )
                self._ingest_parse_pool = None
                self.library_ingest_jobs.mark_failed(
                    job_id,
                    error=_sanitize_library_ingest_error_text(
                        f"Parse pool could not start: {exc}"
                    )
                    or "Parse pool could not start.",
                    permanent=False,
                )
                return
            generation = self._ingest_parse_pool_generation
            generation_jobs = self._ingest_parse_jobs_by_generation[generation]
            generation_jobs.add(job_id)
            try:
                pool.apply_async(
                    run_parse_job,
                    (source_path, options),
                    callback=functools.partial(
                        self._ingest_pool_callback, generation, job_id
                    ),
                    error_callback=functools.partial(
                        self._ingest_pool_error_callback, generation, job_id
                    ),
                )
            except Exception as exc:
                # The pool itself rejected the submission synchronously
                # (e.g. it was already terminated/closed) -- every job
                # currently PARSING was submitted to this same broken pool
                # and can't be trusted to ever complete either.
                self._handle_broken_ingest_parse_pool(generation, job_id, exc)
                return

    def _ingest_pool_callback(
        self, generation: int, job_id: str, result: Dict[str, Any]
    ) -> None:
        """``apply_async`` ``callback``: runs on the pool's result-handler thread.

        Checks ``_ingest_shutdown`` BEFORE marshaling (quit-deadlock
        guard, Task 4 review): Textual's ``call_from_thread`` blocks the
        calling thread on the marshaled call's result and only guards
        against the loop being ``None``, not against it shutting down --
        and CPython's ``Pool._terminate_pool`` does an unbounded
        ``result_handler.join()``, with ``_handle_results`` able to run
        callbacks before it observes TERMINATE. So if a parse completed
        right as the user quit, this thread could park inside
        ``call_from_thread`` while the quit path parked waiting on THIS
        thread inside ``pool.terminate()`` -- mutual deadlock, app hangs
        on quit. Checking the flag here (on this thread, before any
        marshaling) narrows that window; running terminate/join off the
        loop thread entirely (``_shutdown_ingest_parse_pool``) closes it
        -- with both layers, a callback that slips past this check parks
        only until the still-free loop drains it (and the marshaled body
        then no-ops via the same flag inside
        ``_on_ingest_parse_complete``).

        Args:
            job_id: Bound at submission time via ``functools.partial`` in
                ``_top_up_ingest_parse_pool``.
            result: ``run_parse_job``'s structured return value.
        """
        if self._ingest_shutdown:
            return
        self.call_from_thread(
            self._on_ingest_parse_complete, generation, job_id, result
        )

    def _ingest_pool_error_callback(
        self, generation: int, job_id: str, exc: BaseException
    ) -> None:
        """``apply_async`` ``error_callback``: same thread + shutdown
        contract as ``_ingest_pool_callback`` (see its docstring)."""
        if self._ingest_shutdown:
            return
        self.call_from_thread(
            self._handle_broken_ingest_parse_pool, generation, job_id, exc
        )

    def _on_ingest_parse_complete(
        self, generation: int, job_id: str, result: Dict[str, Any]
    ) -> None:
        """Handle one pool completion (success or structured parse failure).

        UI-thread only; invoked via ``call_from_thread`` from the pool's
        result-handler thread (the ``apply_async`` ``callback``). No-ops
        immediately once ``self._ingest_shutdown`` is set -- a completion
        can still be marshaled onto the UI thread for a brief window after
        the app starts closing (it may have already been in flight when
        ``pool.terminate()`` was called), and this guard is what keeps that
        race from touching a closing app's registry/pool state.

        Args:
            job_id: The job this result belongs to (bound at submission
                time in ``_top_up_ingest_parse_pool``, not re-derived here).
            result: ``run_parse_job``'s structured return value -- either
                ``{"ok": True, "payload": {...}}`` or
                ``{"ok": False, "error": str, "permanent": bool}``.
        """
        if self._ingest_shutdown:
            return
        generation_jobs = self._ingest_parse_jobs_by_generation.get(generation)
        if (
            generation != self._ingest_parse_pool_generation
            or generation_jobs is None
            or job_id not in generation_jobs
        ):
            return
        generation_jobs.remove(job_id)
        if result.get("ok"):
            self._ingest_parsed_payloads[job_id] = result["payload"]
            self._start_library_ingest_queue_if_idle()
        else:
            error_text = _sanitize_library_ingest_error_text(
                str(result.get("error") or "Library ingest parsing failed.")
            )
            self.library_ingest_jobs.mark_failed(
                job_id,
                error=error_text or "Library ingest parsing failed.",
                permanent=bool(result.get("permanent", False)),
            )
        self._top_up_ingest_parse_pool()

    def _handle_broken_ingest_parse_pool(
        self,
        generation: int,
        job_id: Optional[str],
        exc: BaseException,
    ) -> None:
        """Fail every still-mid-parse ``PARSING`` job and drop the broken pool.

        UI-thread only. Shared by the pool's ``error_callback`` (an async,
        pool-level failure marshaled via ``call_from_thread`` -- e.g. a
        worker process died) and a synchronous ``apply_async`` submission
        failure in ``_top_up_ingest_parse_pool`` (the pool was already
        broken when we tried to use it). Either way, a job whose parse is
        still genuinely in flight on the SAME pool object may never see
        its callback fire, so it can't be trusted to complete -- failing
        those (retryable) and dropping the pool reference is the only
        sound recovery (see the F3 design spec's "Worker-process death"
        section). The pool is rebuilt lazily by
        ``_create_ingest_parse_pool`` the next time ``_top_up_ingest_parse_pool``
        runs (i.e. on the next submission/retry).

        Payload-ready jobs are SPARED (Task 4 review fix): a job whose
        parse already completed sits ``PARSING`` with its payload in
        ``_ingest_parsed_payloads`` until the writer claims it -- it needs
        nothing further from the pool, so failing it here would throw a
        finished parse away just because an unrelated worker died. Such
        jobs are skipped (left ``PARSING`` for the writer), and the writer
        is woken at the end so they drain even if it had already released.

        No-ops once ``self._ingest_shutdown`` is set, same as
        ``_on_ingest_parse_complete``.
        """
        if self._ingest_shutdown:
            return
        generation_jobs = self._ingest_parse_jobs_by_generation.get(generation)
        if (
            generation != self._ingest_parse_pool_generation
            or generation_jobs is None
            or (job_id is not None and job_id not in generation_jobs)
        ):
            return

        affected_jobs = set(generation_jobs)
        self._ingest_parse_jobs_by_generation.pop(generation, None)
        pool = self._ingest_parse_pool
        stop_event = self._ingest_parse_pool_stop_event
        if stop_event is not None:
            stop_event.set()
        self._ingest_parse_pool_stop_event = None
        self._ingest_parse_pool = None
        if pool is not None:
            self._terminate_ingest_parse_pool_off_thread(pool)

        logger.opt(exception=exc).error(f"Library ingest parse pool failed: {exc}")
        for job in self.library_ingest_jobs.jobs():
            if (
                job.job_id not in affected_jobs
                or job.state != IngestJobState.PARSING
            ):
                continue
            if job.job_id in self._ingest_parsed_payloads:
                # Parse already finished -- the payload is waiting for the
                # writer; the broken pool can't hurt this job anymore.
                continue
            self.library_ingest_jobs.mark_failed(
                job.job_id,
                error="Library ingest parse pool failed unexpectedly; retry to resume.",
                permanent=False,
            )
        if self._ingest_parsed_payloads:
            self._start_library_ingest_queue_if_idle()

    @staticmethod
    def _terminate_ingest_parse_pool_off_thread(pool: Any) -> threading.Thread:
        """Terminate and join one detached Pool without blocking the UI thread."""

        def _terminate_pool() -> None:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                logger.opt(exception=True).error(
                    "Error terminating the Library ingest parse pool."
                )

        thread = threading.Thread(
            target=_terminate_pool,
            name="library-ingest-pool-terminate",
            daemon=True,
        )
        thread.start()
        return thread

    def _shutdown_ingest_parse_pool(self) -> Optional[threading.Thread]:
        """Quit-path teardown: flag up, pool detached, terminate off-loop.

        Called from ``TldwCli.on_unmount`` (i.e. on the app's event-loop
        thread). Synchronously: sets ``_ingest_shutdown = True`` FIRST (so
        pool callbacks -- ``_ingest_pool_callback``/
        ``_ingest_pool_error_callback``, running on the pool's
        result-handler thread -- short-circuit before marshaling from this
        point on) and drops the ``_ingest_parse_pool`` reference (nothing
        can submit to it anymore). The actual ``pool.terminate()`` +
        ``pool.join()`` then run on a detached daemon thread, NEVER on the
        caller's (loop) thread: CPython's ``Pool._terminate_pool`` does an
        unbounded ``result_handler.join()``, and if that result-handler
        thread is at that moment parked inside a ``call_from_thread`` it
        entered just before the flag went up, joining it from the loop
        thread would deadlock (the loop can't drain the marshaled call it
        is itself waiting behind). Off-loop, the loop stays free: the
        in-flight marshaled call runs, no-ops via the flag, the
        result-handler thread unblocks, and the join completes. The daemon
        thread is deliberately not joined by the caller -- worst case it
        outlives the app briefly and dies with the process.

        Returns:
            The teardown thread (so tests can bound-join it and assert
            thread identity), or ``None`` when no pool was ever created --
            the shutdown flag is still set in that case.
        """
        self._ingest_shutdown = True
        pool = getattr(self, "_ingest_parse_pool", None)
        stop_event = getattr(self, "_ingest_parse_pool_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        self._ingest_parse_pool_stop_event = None
        self._ingest_parse_pool = None
        if pool is None:
            return None
        return self._terminate_ingest_parse_pool_off_thread(pool)

    # -- Writer (claim-or-release loop, narrowed to the write stage) -------

    def _start_library_ingest_queue_if_idle(self) -> None:
        """Start the writer worker, unless one is already active.

        UI-thread only. Sets ``runner_active = True`` synchronously, before
        scheduling the worker, so a rapid double-wake can never
        double-start the writer.

        If scheduling the ``@work`` worker itself raises synchronously
        (e.g. the app isn't in a state that accepts new workers), the
        ``runner_active`` flag is rolled back to ``False`` before
        re-raising -- otherwise a scheduling failure here would leave the
        registry permanently believing a runner is active when none was
        ever started, silently stranding every future payload.
        """
        if self.library_ingest_jobs.runner_active:
            return
        self.library_ingest_jobs.runner_active = True
        try:
            self._run_library_ingest_queue()
        except Exception:
            self.library_ingest_jobs.runner_active = False
            raise

    def _claim_next_ingest_job_or_release(
        self,
    ) -> Optional[tuple[LibraryIngestJob, Dict[str, Any]]]:
        """Atomically claim the oldest payload-ready job, or release the writer.

        UI-thread only; must only ever be invoked via ``call_from_thread``
        from the writer worker thread (see ``_run_library_ingest_queue``),
        never called directly from that thread.

        "Payload-ready" means the job's parsed payload is sitting in
        ``self._ingest_parsed_payloads`` (stashed by
        ``_on_ingest_parse_complete`` on a successful parse) -- claiming
        means popping that payload out of the dict AND transitioning the
        job ``PARSING`` -> ``WRITING`` via ``mark_writing``, both inside
        this single call. Jobs are visited oldest-submission-first
        (``self.library_ingest_jobs.jobs()`` is newest-first; this walks it
        reversed) so writes happen in submission order among ready
        payloads, even though a small file may finish parsing before an
        older large one.

        A successful claim also tops up the parse pool
        (``_top_up_ingest_parse_pool``): a payload-ready job still counts
        against the ``PARSING`` cap until this call's ``mark_writing``
        transitions it out (there is no separate registry state for
        "parsed but not yet claimed" -- see ``IngestJobState``), so a
        completion's own top-up call (in ``_on_ingest_parse_complete``,
        which always runs *before* the writer gets around to claiming) can
        still see the cap as full. Topping up again here is what actually
        frees that slot for a still-``QUEUED`` job once the claim lands.

        Atomicity contract: this is a single, plain synchronous UI-thread
        call, so the "is there a payload-ready job?" check and the "clear
        ``runner_active``" decision happen in the same turn of the UI event
        loop with no ``await``/yield between them -- exactly the discipline
        the pre-F3 claim-or-release fix established (see the git history:
        the previous two-step implementation had a submission land in the
        gap between "check" and "clear ``runner_active``", stranding a job
        behind a stale ``runner_active`` flag). Do not reintroduce a
        two-``call_from_thread`` exit path.

        Returns:
            ``(job, payload)`` for the oldest payload-ready job, if one
            exists -- ``runner_active`` is left untouched (still ``True``)
            and the writer must keep looping. ``None`` when no job is
            payload-ready -- ``runner_active`` is cleared before returning,
            and the writer must exit.
        """
        if self._ingest_shutdown:
            self.library_ingest_jobs.runner_active = False
            return None
        for job in reversed(self.library_ingest_jobs.jobs()):
            payload = self._ingest_parsed_payloads.get(job.job_id)
            if payload is None:
                continue
            del self._ingest_parsed_payloads[job.job_id]
            claimed = self.library_ingest_jobs.mark_writing(job.job_id)
            if claimed is None:
                # Invariant violation (Task-3 reviewer's guard note): a
                # payload existed for a job that wasn't PARSING when we
                # tried to claim it -- should be impossible (a payload only
                # ever enters the dict from a PARSING-state parse
                # completion, and this is the only caller of
                # `mark_writing`), but if it ever happens, the orphaned
                # payload is discarded (already popped above) and we keep
                # looking rather than crashing the writer loop.
                logger.error(
                    f"Library ingest writer: mark_writing rejected job "
                    f"{job.job_id} despite a ready payload -- discarding "
                    f"the orphaned payload and skipping."
                )
                continue
            self._top_up_ingest_parse_pool()
            return claimed, payload
        self.library_ingest_jobs.runner_active = False
        return None

    def _release_ingest_runner_after_crash(self) -> None:
        """Safety-net cleanup for the writer's ``finally`` block.

        UI-thread only; invoked via ``call_from_thread`` from the writer
        worker's ``finally``, on every exit path (clean or not).

        On the normal, clean-exit path this is a no-op: the writer already
        exited because ``_claim_next_ingest_job_or_release`` returned
        ``None``, which already cleared ``runner_active``. It only does
        real work when the worker thread is unwinding from something that
        bypassed that atomic exit -- i.e. an exception escaped a job's own
        isolation (see ``_run_library_ingest_queue``) or the marshaled call
        itself raised. In that case: clear ``runner_active`` if it is still
        set, and, since the crash may have left one or more parsed payloads
        sitting unclaimed with nothing left to drain them, restart the
        writer when a payload is still waiting at that moment. Restarting
        here is safe: this method runs on the UI thread, and the dying
        worker thread is already unwinding and will not touch the registry
        again.
        """
        if self.library_ingest_jobs.runner_active:
            self.library_ingest_jobs.runner_active = False
        if self._ingest_parsed_payloads:
            self._start_library_ingest_queue_if_idle()

    @work(exclusive=True, thread=True, group="library_ingest_queue")
    def _run_library_ingest_queue(self) -> None:
        """Drain payload-ready Library ingest jobs on a background thread.

        This is the write stage only (F3): parsing already happened in the
        pool, and this worker's whole job is persisting an already-parsed
        payload via ``persist_parsed_media`` -- one ``add_media_with_keywords``
        call at a time, since SQLite has exactly one writer.

        Runs until no job is payload-ready, then clears ``runner_active``
        (via ``_claim_next_ingest_job_or_release``, atomically -- see that
        method's docstring) and exits -- a later parse completion wakes a
        fresh worker (``_on_ingest_parse_complete`` ->
        ``_start_library_ingest_queue_if_idle``). Every registry touch is
        marshaled onto the UI thread via ``call_from_thread`` because
        ``LibraryIngestJobRegistry`` does no internal locking (see its
        module docstring). A single job's write failure (DB error, ...) is
        caught locally and turned into a ``mark_failed`` transition; it
        never aborts the loop.

        The outer ``try/finally`` is a separate safety net for failures
        *outside* that per-job isolation -- e.g. the marshaled claim call
        itself raising (a genuinely unexpected/"catastrophic" failure, not
        a per-job write error). See ``_release_ingest_runner_after_crash``
        for why the crash-recovery callable is skipped on a clean exit.
        """
        clean_exit = False
        try:
            while True:
                claim = self.call_from_thread(self._claim_next_ingest_job_or_release)
                if claim is None:
                    clean_exit = True
                    return
                job, payload = claim
                try:
                    media_id, _media_uuid, _message = persist_parsed_media(
                        payload, self.media_db
                    )
                    if media_id is None and self.media_db is not None:
                        # Re-ingesting an unchanged file takes the DB's
                        # update path, whose return carries no media id.
                        # Resolve it by canonical URL so the done row keeps
                        # its "Open in Library" action. ``self.media_db`` is
                        # unreachable-``None`` here in practice (submit
                        # already fails the job before this point when it's
                        # absent), but this guard is cheap insurance against
                        # an ``AttributeError`` on a stale/racy reference.
                        existing = self.media_db.get_media_by_url(payload["url"])
                        if existing is not None:
                            media_id = existing.get("id")
                    self.call_from_thread(
                        self.library_ingest_jobs.mark_done,
                        job.job_id,
                        media_id=media_id,
                    )
                except Exception as exc:
                    # loguru's traceback capture is `.opt(exception=True)`,
                    # NOT the stdlib `exc_info=True` kwarg (a silent no-op
                    # under loguru) -- log the full traceback here before
                    # mark_failed so a debugging session isn't left with only
                    # the registry's sanitized, single-line error string.
                    logger.opt(exception=True).error(
                        f"Library ingest job failed during write "
                        f"(job_id={job.job_id}, source={job.source_path})."
                    )
                    self.call_from_thread(
                        self.library_ingest_jobs.mark_failed,
                        job.job_id,
                        error=_sanitize_library_ingest_error(exc),
                        permanent=classify_parse_failure(exc),
                    )
        finally:
            if not clean_exit:
                self.call_from_thread(self._release_ingest_runner_after_crash)


# --- Main App ---
class TldwCli(LibraryIngestQueueMixin, App[None]):  # Specify return type for run() if needed, None is common
    """A Textual app for interacting with LLMs."""
    # Keep legacy identifier for tests while retaining product name
    TITLE = "tldw CLI • tldw chatbook"
    # CSS file path
    CSS_PATH = str(Path(__file__).parent / "css/tldw_cli_modular.tcss")
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit App", show=True),
        Binding("ctrl+p", "command_palette", "Palette Menu", show=True),
        Binding("f1", "show_workbench_help", "Help", show=True),
        Binding("f6", "focus_next_workbench_pane", "Next Pane", show=True),
    ]
    COMMANDS = App.COMMANDS | {
        ThemeProvider,
        TabNavigationProvider,
        LLMProviderProvider,
        QuickActionsProvider,
        SettingsProvider,
        CharacterProvider,
        MediaProvider,
        DeveloperProvider,
        ConsoleCommandProvider,
    }

    ALL_INGEST_VIEW_IDS = INGEST_VIEW_IDS
    # T169: "notes-window" removed -- no widget composes that id anymore (the
    # standalone Notes tab / Notes_Window.py it belonged to is gone, replaced
    # by the Library workbench's Notes canvas), confirmed via
    # `grep -rn 'id="notes-window"' tldw_chatbook/`.
    ALL_MAIN_WINDOW_IDS = [ # Assuming these are your main content window IDs
        "chat-window", "conversations_characters_prompts-window",
        "ingest-window", "tools_settings-window", "llm_management-window",
        "media-window", "search-window", "logs-window", "stats-window", "evals-window",
        "coding-window", "stts-window", "study-window", "chatbooks-window", "customize-window"
    ]

    # Define reactive at class level with a placeholder default and type hint
    current_tab: reactive[str] = reactive("")
    ccp_active_view: reactive[str] = reactive("conversation_details_view")
    
    # Splash screen state
    splash_screen_active: reactive[bool] = reactive(False)
    _splash_screen_widget: Optional[SplashScreen] = None

    # Add state to hold the currently streaming AI message widget
    # Use a lock to prevent race conditions when modifying shared state
    _chat_state_lock = threading.Lock()
    current_ai_message_widget: Optional[Union[ChatMessage, ChatMessageEnhanced]] = None
    current_chat_worker: Optional[Worker] = None
    current_chat_is_streaming: bool = False

    # --- REACTIVES FOR PROVIDER SELECTS ---
    # Initialize with a dummy value or fetch default from config here
    # Ensure the initial value matches what's set in compose/settings_sidebar
    # Fetching default provider from config:
    _default_chat_provider = APP_CONFIG.get("chat_defaults", {}).get("provider", "OpenAI")
    _default_ccp_provider = APP_CONFIG.get("character_defaults", {}).get("provider", "Anthropic") # Changed from character_defaults

    chat_api_provider_value: reactive[Optional[str]] = reactive(_default_chat_provider)
    # Renamed character_api_provider_value to ccp_api_provider_value for clarity with TAB_CCP
    ccp_api_provider_value: reactive[Optional[str]] = reactive(_default_ccp_provider)

    def query_one(self, selector, expect_type=None):
        """Resolve legacy app-level queries against the active pushed screen when needed."""
        try:
            return super().query_one(selector, expect_type)
        except NoMatches as error:
            try:
                active_screen = self.screen
            except Exception as screen_error:
                raise screen_error from error
            return active_screen.query_one(selector, expect_type)

    # RAG expansion provider reactive
    rag_expansion_provider_value: reactive[Optional[str]] = reactive(_default_chat_provider)

    # --- Reactives for CCP Character EDITOR (Center Pane) ---
    current_editing_character_id: reactive[Optional[str]] = reactive(None)
    current_editing_character_data: reactive[Optional[Dict[str, Any]]] = reactive(None)

    # DB Size checker - now using AppFooterStatus
    _db_size_status_widget: Optional[AppFooterStatus] = None
    # DB size update timer moved to DBStatusManager
    _token_count_update_timer: Optional[Timer] = None
    ui_responsiveness_monitor: UIResponsivenessMonitor | None = None
    _ui_responsiveness_heartbeat_timer: Optional[Timer] = None

    # Reactives for sidebar
    chat_sidebar_collapsed: reactive[bool] = reactive(True)
    chat_right_sidebar_collapsed: reactive[bool] = reactive(False)  # For character sidebar
    # Load saved width from config, default to 25% if not set
    _saved_width = settings.get("chat_defaults", {}).get("right_sidebar_width", 25)
    chat_right_sidebar_width: reactive[int] = reactive(_saved_width)  # Width percentage for right sidebar
    conv_char_sidebar_left_collapsed: reactive[bool] = reactive(False)
    conv_char_sidebar_right_collapsed: reactive[bool] = reactive(False)
    evals_sidebar_collapsed: reactive[bool] = reactive(False) # Added for Evals tab
    media_active_view: reactive[Optional[str]] = reactive(None)  # Added for Media tab navigation

    # Reactive variables for selected note details
    current_selected_note_id: reactive[Optional[str]] = reactive(None)
    current_selected_note_version: reactive[Optional[int]] = reactive(None)
    current_selected_note_title: reactive[Optional[str]] = reactive(None)
    current_selected_note_content: reactive[Optional[str]] = reactive("")
    
    # Notes tab UI state
    notes_sort_by: reactive[str] = reactive("date_created")  # date_created, date_modified, title
    notes_sort_ascending: reactive[bool] = reactive(False)  # False = newest first
    notes_preview_mode: reactive[bool] = reactive(False)  # False = edit mode, True = preview mode

    # Auto-save related reactive variables
    notes_auto_save_enabled: reactive[bool] = reactive(True)  # Auto-save enabled by default
    notes_auto_save_timer: reactive[Optional[Timer]] = reactive(None)  # Timer reference for auto-save
    notes_last_save_time: reactive[Optional[float]] = reactive(None)  # Timestamp of last save

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
    
    # Chat Tabs Management (when enable_tabs is True)
    active_chat_tab_id: reactive[Optional[str]] = reactive(None)
    chat_sessions: reactive[Dict[str, Dict[str, Any]]] = reactive({})  # tab_id -> session_data dict

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

    # Settings mode for chat sidebar
    chat_settings_mode: reactive[str] = reactive("basic")  # "basic" or "advanced"
    chat_settings_search_query: reactive[str] = reactive("")  # Search query for settings

    # Search Tab's active sub-view reactives
    search_active_sub_tab: reactive[Optional[str]] = reactive(None)
    _initial_search_sub_tab_view: Optional[str] = SEARCH_VIEW_RAG_QA

    # Ingest Tab
    ingest_active_view: reactive[Optional[str]] = reactive("ingest-view-prompts")
    _initial_ingest_view: Optional[str] = "ingest-view-prompts"
    selected_prompt_files_for_import: List[Path] = []
    parsed_prompts_for_preview: List[Dict[str, Any]] = []
    last_prompt_import_dir: Optional[Path] = None
    selected_note_files_for_import: List[Path] = []
    parsed_notes_for_preview: List[Dict[str, Any]] = []
    last_note_import_dir: Optional[Path] = None
    # Add attributes to hold the handlers (optional, but can be useful)
    note_import_success_handler: Optional[Callable] = None
    note_import_failure_handler: Optional[Callable] = None

    # Tools Tab
    tools_settings_active_view: reactive[Optional[str]] = reactive("ts-view-general-settings")  # Default to general settings
    _initial_tools_settings_view: Optional[str] = "ts-view-general-settings"

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
    
    # Flag to track if character filter has been populated
    _chat_character_filter_populated: bool = False

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
        # Track startup timing
        self._startup_start_time = time.perf_counter()
        self._startup_phases = {}
        
        # Tab switching optimization
        self._initialized_tabs = set()  # Track which tabs have been initialized
        
        # Reduce logging in production
        if not os.environ.get("TLDW_DEBUG"):
            logging.getLogger().setLevel(logging.INFO)  # Reduce to INFO level in production
            # Disable debug logging for performance
            logging.getLogger("tldw_chatbook").setLevel(logging.INFO)
        
        # Log initial memory usage only in debug mode
        if os.environ.get("TLDW_DEBUG"):
            log_resource_usage()
        log_counter("app_startup_initiated", 1, documentation="Application startup initiated")
        
        super().__init__()
        
        # Phase 1: Basic initialization
        phase_start = time.perf_counter()
        self.MediaDatabase = MediaDatabase
        self.app_config = load_settings()
        self.acp_runtime_process_manager = ACPRuntimeProcessManager.from_app_config(self.app_config)
        self.acp_runtime_session_state = self.acp_runtime_process_manager.session_state()
        self.app_state = AppState()
        self.runtime_policy = load_runtime_policy_for_app(self)
        self.service_policy_enforcer = ServicePolicyEnforcer.from_runtime_policy_context(
            self.runtime_policy
        )
        self.ui_policy_engine = PolicyEngine(CAPABILITY_REGISTRY)
        self.pending_chat_handoff: Optional[ChatHandoffPayload] = None
        self.pending_console_launch: Optional[ConsoleLiveWorkLaunch | Dict[str, Any]] = None
        self.pending_console_prompt_insert: Optional[str] = None
        self.pending_study_scope_context: Optional[StudyScopeContext] = None
        self.pending_study_initial_section: Optional[str] = None
        self.pending_notes_workspace_context: Optional[Dict[str, Any]] = None
        self.home_active_work_adapter = UnavailableHomeActiveWorkAdapter(
            runtime_policy=self.runtime_policy,
        )
        self.loguru_logger = loguru_logger
        self.loguru_logger.info(f"Loaded app_config - strip_thinking_tags: {self.app_config.get('chat_defaults', {}).get('strip_thinking_tags', 'NOT SET')}") # Make loguru_logger an instance variable for handlers
        self.client_id = CLI_APP_CLIENT_ID
        self.prompts_client_id = "tldw_tui_client_v1" # Store client ID for prompts service
        self.db_status_manager = DBStatusManager(self)  # Initialize database status manager
        self.ui_responsiveness_monitor = UIResponsivenessMonitor(
            enabled=bool(get_cli_setting("diagnostics", "ui_responsiveness_enabled", True)),
            heartbeat_interval_seconds=1.0,
        )
        self._wire_server_context_provider()
        self._startup_phases["basic_init"] = time.perf_counter() - phase_start
        log_histogram("app_startup_phase_duration_seconds", self._startup_phases["basic_init"], 
                     labels={"phase": "basic_init"}, 
                     documentation="Duration of startup phase in seconds")

        # Phase 2: Attribute initialization
        phase_start = time.perf_counter()
        # Initialize screen navigation flag early to prevent AttributeError
        self._use_screen_navigation = True  # ALWAYS use screen-based navigation now
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
        self._startup_phases["attribute_init"] = time.perf_counter() - phase_start
        log_histogram("app_startup_phase_duration_seconds", self._startup_phases["attribute_init"], 
                     labels={"phase": "attribute_init"}, 
                     documentation="Duration of startup phase in seconds")

        # Phase 3: Parallel initialization of independent services
        phase_start = time.perf_counter()
        
        # Prepare shared data
        user_name_for_notes = settings.get("USERS_NAME", "default_tui_user")
        self.notes_user_id = user_name_for_notes
        
        # Run independent initializations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all independent initialization tasks
            futures = {
                executor.submit(self._init_notes_service, user_name_for_notes): "notes_service",
                executor.submit(self._init_providers_models): "providers_models",
                executor.submit(self._init_prompts_service): "prompts_service",
                executor.submit(self._init_media_db): "media_db"
            }
            
            # Wait for all tasks to complete and log individual timings
            for future in concurrent.futures.as_completed(futures):
                task_name = futures[future]
                try:
                    task_start = time.perf_counter()
                    result = future.result()
                    task_duration = time.perf_counter() - task_start
                    logger.info(f"Parallel init task '{task_name}' completed in {task_duration:.3f}s")
                except Exception as e:
                    logger.opt(exception=True).error(f"Parallel init task '{task_name}' failed: {e}")
        
        # Log total parallel phase time
        parallel_duration = time.perf_counter() - phase_start
        self._startup_phases["parallel_init"] = parallel_duration
        log_histogram("app_startup_phase_duration_seconds", parallel_duration, 
                     labels={"phase": "parallel_init"}, 
                     documentation="Duration of parallel initialization phase")
        log_resource_usage()  # Check memory after parallel init

        # Providers, prompts, and media DB are initialized in parallel above
        # Just ensure we have defaults if parallel init failed
        if not hasattr(self, 'providers_models'):
            self.providers_models = {}

        # --- Initial Tab ---
        initial_tab_from_config = get_cli_setting("general", "default_tab", TAB_CHAT)
        self._initial_tab_value = self._normalize_initial_tab_from_config(initial_tab_from_config)
        logging.info(f"App __init__: Determined initial tab value: {self._initial_tab_value}")
        # current_tab reactive will be set in on_mount after UI is composed

        self._rich_log_handler: Optional[RichLogHandler] = None # For the RichLog widget in Logs tab

        # Prompts service is initialized in parallel above
        # Set up timer
        self._prompt_search_timer = None

        # Media DB is initialized in parallel above
        # Ensure we have media types for UI
        if not hasattr(self, '_media_types_for_ui'):
            self._media_types_for_ui = ["Error: Media DB not loaded"]

        initial_media_runtime_backend = self._resolve_initial_media_runtime_backend()
        self.media_runtime_state = MediaRuntimeState(runtime_backend=initial_media_runtime_backend)
        self.local_media_reading_service = LocalMediaReadingService(self.media_db, app_config=self.app_config)
        self.server_media_reading_service = ServerMediaReadingService.from_server_context_provider(
            self.server_context_provider,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.media_reading_scope_service = MediaReadingScopeService(
            local_service=self.local_media_reading_service,
            server_service=self.server_media_reading_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self._wire_library_collections_services()
        self._wire_workspace_registry_services()
        self._wire_prompt_chatbook_services()
        self._wire_watchlists_and_notifications_services()
        self._wire_writing_services()

        self.loguru_logger.debug(f"ULTRA EARLY APP INIT: self._media_types_for_ui VALUE: {self._media_types_for_ui}")
        self.loguru_logger.debug(f"ULTRA EARLY APP INIT: self._media_types_for_ui TYPE: {type(self._media_types_for_ui)}")

        self._tts_handler = None
        self._stts_handler = None
        self._tts_initialization_task: asyncio.Task | None = None
        self._stts_initialization_task: asyncio.Task | None = None
        self._deferred_startup_tasks: set[asyncio.Task] = set()

        self._ui_ready = False  # Track if UI is fully composed
        self._shutting_down = False  # Track if app is shutting down

        # --- Setup Default view for CCP tab ---
        # Initialize self.ccp_active_view based on initial tab or default state if needed
        if self._initial_tab_value == TAB_CCP:
            self.ccp_active_view = "conversation_details_view"  # Default view for CCP tab
        # else: it will default to "conversation_details_view" anyway

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
        self.prompt_scope_service = build_prompt_scope_service(
            prompt_db=self.prompts_db,
            app_config=self.app_config,
            policy_enforcer=self.service_policy_enforcer,
            client_provider=self.server_context_provider,
        )

        if self.notes_service and hasattr(self.notes_service, 'db') and self.notes_service.db:
            self.chachanotes_db = self.notes_service.db # ChaChaNotesDB is used by NotesInteropService
            logging.info("Assigned self.notes_service.db to self.chachanotes_db")
        else: # Fallback to global if notes_service didn't set it up as expected on itself
            lazy_db = get_chachanotes_db_lazy()
            if lazy_db:
                self.chachanotes_db = lazy_db
                logging.info("Assigned lazy-loaded chachanotes_db to self.chachanotes_db as fallback.")
            else:
                logging.error("ChaChaNotesDB (CharactersRAGDB) instance not found/assigned in app.__init__.")
                self.chachanotes_db = None # Explicitly set to None

        self._wire_chat_conversation_services()

        self.server_notes_workspace_service = ServerNotesWorkspaceService.from_server_context_provider(
            self.server_context_provider,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.notes_scope_service = NotesScopeService(
            local_notes_service=self.notes_service,
            server_service=self.server_notes_workspace_service,
            policy_enforcer=self.service_policy_enforcer,
            sync_scope_service=getattr(self, "sync_scope_service", None),
        )
        try:
            self.server_rag_admin_service = ServerRAGAdminService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_rag_admin_service = ServerRAGAdminService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.local_rag_admin_service = LocalRAGAdminService(
            self.media_db,
            app_config=self.app_config,
            media_service=self.local_media_reading_service,
        )
        self.rag_admin_scope_service = RAGAdminScopeService(
            local_service=self.local_rag_admin_service,
            server_service=self.server_rag_admin_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self._wire_evaluation_services()
        self._wire_study_services()
        self._wire_writing_services()
        self._wire_research_services()
        self._wire_character_persona_services()
        self._wire_chat_conversation_services()

        # --- Create the master handler map ---
        # This one-time setup makes the dispatcher clean and fast.
        self.button_handler_map = self._build_handler_map()
        
        # --- Initialize worker handler registry ---
        self._init_worker_handlers()
        
        # Log total initialization time
        total_init_time = time.perf_counter() - self._startup_start_time
        self._startup_phases["total_init"] = total_init_time
        log_histogram("app_startup_total_duration_seconds", total_init_time, 
                     documentation="Total application initialization time in seconds")
        
        # Log startup summary
        logger.info(f"=== STARTUP TIMING SUMMARY ===")
        logger.info(f"Total initialization time: {total_init_time:.3f} seconds")
        for phase, duration in self._startup_phases.items():
            if phase != "total_init":
                percentage = (duration / total_init_time) * 100 if total_init_time > 0 else 0
                logger.info(f"  {phase}: {duration:.3f}s ({percentage:.1f}%)")
        logger.info(f"==============================")
        
        # Final memory check
        log_resource_usage()

    def _wire_server_context_provider(self) -> None:
        self.unified_mcp_target_store = ConfiguredServerTargetStore(
            get_user_data_dir() / "mcp_server_targets.json",
        )
        self.unified_mcp_target_store.upsert_legacy_config_target(self.app_config)
        try:
            self.server_credential_store = build_default_server_credential_store()
        except CredentialStoreUnavailable as exc:
            self.server_credential_store = UnavailableServerCredentialStore(str(exc))
        self.server_context_provider = RuntimeServerContextProvider(
            runtime_context=self.runtime_policy,
            target_store=self.unified_mcp_target_store,
            credential_store=self.server_credential_store,
            app_config=self.app_config,
        )

    def open_study_screen(
        self,
        scope_context: Optional[StudyScopeContext] = None,
        *,
        initial_section: Optional[str] = None,
    ) -> None:
        self.pending_study_scope_context = scope_context
        self.pending_study_initial_section = initial_section
        self.post_message(NavigateToScreen(TAB_STUDY))

    def open_notes_workspace(
        self,
        workspace_id: str,
        subview: Any = None,
    ) -> None:
        """Return to Library's Notes list after leaving it for another screen.

        The standalone Notes tab's per-workspace scope has no equivalent in
        Library, which browses notes as a flat list -- this always re-opens
        the shared Library Notes list rather than any workspace-scoped view.

        Args:
            workspace_id: The retired Notes tab's workspace identifier.
                Accepted for backward compatibility with existing callers
                (e.g. Study's "back to workspace" action) but no longer
                applied.
            subview: The retired Notes tab's workspace subview. Accepted for
                backward compatibility; no longer applied.
        """
        self.post_message(NavigateToScreen(TAB_LIBRARY, {LIBRARY_NAV_CONTEXT_MODE: "notes"}))

    def open_chat_with_handoff(
        self,
        payload: ChatHandoffPayload,
        *,
        action_label: str = "Use in Chat",
    ) -> None:
        """Stage a handoff payload for Chat and navigate there.

        Args:
            payload: The handoff payload to stage as pending Chat context.
            action_label: The calling surface's own action label (e.g. "Use
                in Chat" for the legacy MediaWindow_v2/search_rag_window
                surfaces, "Use in Console" for Library) so the blocked-gate
                notify below reads honestly for whichever button the user
                actually pressed, instead of always saying "Chat" even from
                a destination whose own button says "Console" (M2).
        """
        if not get_cli_setting("chat_defaults", "enable_tabs", True):
            self.notify(
                f"{action_label} requires chat tabs to be enabled.",
                severity="warning",
            )
            return

        self.pending_chat_handoff = payload
        self.post_message(NavigateToScreen(TAB_CHAT))

    def stage_console_prompt_insert(self, text: str) -> None:
        """Stage a resolved Library prompt body for the Console composer and navigate there.

        ``ChatHandoffPayload``-free direct route (Task 12): Library's prompt
        editor "Use in Console" action only ever needs to land plain text
        into the Console draft -- appended onto whatever the user was
        already composing, never replacing it -- so this deliberately skips
        ``open_chat_with_handoff``'s richer RAG-evidence-aware staging
        machinery. Mirrors that method's stage-then-navigate shape, but the
        payload is a bare string and there is no tabs-enabled gate: whether
        the insert actually lands is decided by ``ChatScreen`` once it
        consumes this field (it alone owns Console's provider/model
        readiness state).

        Args:
            text: The prompt's ``user_prompt`` body to insert.
        """
        self.pending_console_prompt_insert = text
        self.post_message(NavigateToScreen(TAB_CHAT))

    def open_console_for_live_work(
        self,
        *,
        source: str,
        title: str,
        payload: dict | None = None,
        status: str | None = None,
        recovery: str | None = None,
        action_label: str | None = None,
    ) -> None:
        """Open Console for live work launched from another destination."""
        self.pending_console_launch = ConsoleLiveWorkLaunch.from_values(
            source=source,
            title=title,
            payload=payload,
            status=status,
            recovery=recovery,
            action_label=action_label,
        )
        self.post_message(NavigateToScreen(TAB_CHAT))

    def get_acp_runtime_session_state(self) -> ACPRuntimeSessionState:
        """Return current ACP runtime/session state for ACP and Console surfaces."""
        explicit_state = getattr(self, "acp_runtime_session_state", None)
        normalized_state = ACPRuntimeSessionState.from_any(explicit_state)
        if normalized_state.runtime_configured:
            return normalized_state
        manager = getattr(self, "acp_runtime_process_manager", None)
        snapshot = getattr(manager, "snapshot", None)
        if callable(snapshot):
            return ACPRuntimeSessionState.from_any(snapshot())
        return normalized_state

    def open_console_live_work_primary_action(self, launch: Any) -> bool:
        """Follow through on a supported Console live-work status-card action."""
        normalized_launch = ConsoleLiveWorkLaunch.from_pending(launch)
        if normalized_launch is None:
            self.notify("Console action is unavailable for this live-work item.", severity="warning")
            return False

        action = resolve_console_live_work_primary_action(normalized_launch)
        if action is None:
            self.notify("Console action is unavailable for this live-work item.", severity="warning")
            return False

        if action.target_route == TAB_SUBSCRIPTIONS:
            self._stage_subscription_watchlist_run_context(action.target_id)
            self.post_message(NavigateToScreen(TAB_SUBSCRIPTIONS))
            return True

        if action.target_route == TAB_ARTIFACTS:
            self.pending_artifacts_chatbook_target_id = action.target_id
            self.post_message(NavigateToScreen(TAB_ARTIFACTS))
            return True

        if action.target_route == TAB_ACP:
            self.pending_acp_session_target_id = action.target_id
            self.post_message(NavigateToScreen(TAB_ACP))
            return True

        self.notify("Console action route is not available yet.", severity="warning")
        return False

    def _handle_home_control_action(
        self,
        action: HomeControlAction,
        *,
        target_id: str | None = None,
        target_route: str | None = None,
    ) -> HomeControlResult:
        adapter = getattr(self, "home_active_work_adapter", UnavailableHomeActiveWorkAdapter())
        if target_id is None and target_route is None:
            result = adapter.handle_control(action)
        else:
            result = adapter.handle_control(
                action,
                target_id=target_id,
                target_route=target_route,
            )
        # B3 (task-282): approve/reject/pause/resume/retry can change the
        # watchlist-run/notification state the adapter's short-TTL cache
        # holds -- invalidate so the next Home read is not stale for up to
        # the TTL window. Defensive getattr: the honest-unavailable adapter
        # and test doubles don't implement this hook.
        invalidate_cache = getattr(adapter, "invalidate_active_work_cache", None)
        if callable(invalidate_cache):
            invalidate_cache()
        self.notify(result.message, severity=result.severity)
        return result

    def prepare_home_primary_action(self, action: Any) -> None:
        """Stage route-specific context before Home primary-action navigation."""
        if getattr(action, "action_id", None) == "review_notifications":
            self.pending_subscription_initial_tab = "notifications"
        elif (
            getattr(action, "action_id", None) == "review_failed_work"
            and getattr(action, "target_route", None) == "subscriptions"
        ):
            self.pending_subscription_initial_tab = "watchlist-runs"

    def approve_active_home_item(self, *, target_id: str | None = None) -> HomeControlResult:
        """Approve the active Home item through the configured adapter."""
        return self._handle_home_control_action(HomeControlAction.APPROVE, target_id=target_id)

    def reject_active_home_item(self, *, target_id: str | None = None) -> HomeControlResult:
        """Reject the active Home item through the configured adapter."""
        return self._handle_home_control_action(HomeControlAction.REJECT, target_id=target_id)

    def pause_active_home_item(self, *, target_id: str | None = None) -> HomeControlResult:
        """Pause the active Home item through the configured adapter."""
        return self._handle_home_control_action(HomeControlAction.PAUSE, target_id=target_id)

    def resume_active_home_item(self, *, target_id: str | None = None) -> HomeControlResult:
        """Resume the active Home item through the configured adapter."""
        return self._handle_home_control_action(HomeControlAction.RESUME, target_id=target_id)

    def retry_active_home_item(self, *, target_id: str | None = None) -> HomeControlResult:
        """Retry the active Home item through the configured adapter.

        Library ingest job targets (``local:ingest:<job_id>``) are requeued
        directly through ``retry_library_ingest_job`` -- the real requeue
        seam over ``self.library_ingest_jobs`` -- instead of falling through
        to ``_handle_home_control_action``/the adapter, which has no
        visibility into the in-memory ingest job registry and always
        degrades to the honest "not connected to an active run service yet"
        fallback for this target shape. Non-ingest targets (approvals,
        watchlist runs, schedules) are unaffected and still route through
        the adapter exactly as before.
        """
        if target_id is not None and str(target_id).startswith("local:ingest:"):
            job_id = str(target_id)[len("local:ingest:"):]
            requeued = self.retry_library_ingest_job(job_id)
            if requeued is None:
                # Unknown job id, or the job is no longer FAILED (e.g. it
                # was already retried/finished by the time the button was
                # pressed) -- ``requeue`` is a documented no-op in that case.
                result = HomeControlResult(
                    action=HomeControlAction.RETRY,
                    status=HomeControlResultStatus.UNAVAILABLE,
                    message="This ingest job can no longer be retried.",
                    severity="warning",
                    recovery_route="library",
                    target_id=target_id,
                )
            else:
                # The basename is a user-controlled filename (arbitrary
                # source path picked in the Library ingest form) that flows
                # straight into a Home toast, which parses Rich markup --
                # same hazard class as the open-details title fix. Escape
                # defensively.
                basename = escape_markup(
                    Path(str(requeued.source_path)).name or str(requeued.source_path)
                )
                result = HomeControlResult(
                    action=HomeControlAction.RETRY,
                    status=HomeControlResultStatus.HANDLED,
                    message=f"Retry queued for {basename}.",
                    recovery_route="library",
                    target_id=f"local:ingest:{requeued.job_id}",
                    target_route="library",
                )
            self.notify(result.message, severity=result.severity)
            return result
        return self._handle_home_control_action(HomeControlAction.RETRY, target_id=target_id)

    def open_home_flashcards_review(self) -> None:
        """Open the Study screen directly on the flashcards review surface."""
        self.open_study_screen(initial_section="flashcards")

    def _local_flashcards_due_count(self) -> int | None:
        """Count due flashcards for the Home mirror; None when the DB is absent."""
        db = getattr(self, "chachanotes_db", None)
        counter = getattr(db, "count_due_flashcards", None)
        if not callable(counter):
            return None
        try:
            return int(counter())
        except Exception:
            logger.opt(exception=True).debug("Home flashcards-due count failed.")
            return None

    def open_active_home_item_details(
        self,
        *,
        target_id: str | None = None,
        target_route: str = TAB_CHAT,
    ) -> HomeControlResult:
        """Open active Home item details through the configured adapter."""
        result = self._handle_home_control_action(
            HomeControlAction.OPEN_DETAILS,
            target_id=target_id,
            target_route=target_route,
        )
        if result.status is HomeControlResultStatus.HANDLED and result.target_route:
            if result.target_route == "subscriptions":
                self._stage_subscription_watchlist_run_context(result.target_id or target_id)
                self.post_message(NavigateToScreen(result.target_route))
            elif result.target_route == "library" and str(
                result.target_id or target_id or ""
            ).startswith("local:ingest:"):
                # Home's ingest-jobs Running/Needs Attention rows one-hop
                # back to the Library ingest canvas via the nav-context
                # contract instead of a bare route (mirrors the
                # subscriptions staging special-case above). Navigation
                # always composes a fresh Library screen, so the deep link
                # lands on a cleanly mounted, repainted ingest canvas.
                self.post_message(
                    NavigateToScreen("library", {LIBRARY_NAV_CONTEXT_INGEST: True})
                )
            else:
                self.post_message(NavigateToScreen(result.target_route))
        return result

    def _stage_subscription_watchlist_run_context(self, target_id: str | None) -> None:
        if target_id and ":watchlist_run:" in str(target_id):
            self.pending_subscription_initial_tab = "watchlist-runs"
            self.pending_subscription_watchlist_run_id = str(target_id)

    def open_active_home_item_in_console(
        self,
        *,
        target_id: str | None = None,
        target_route: str = TAB_CHAT,
    ) -> HomeControlResult:
        """Open active Home item in Console only when the adapter supplies launch context."""
        result = self._handle_home_control_action(
            HomeControlAction.OPEN_IN_CONSOLE,
            target_id=target_id,
            target_route=target_route,
        )
        if result.status is HomeControlResultStatus.HANDLED and result.console_launch is not None:
            launch_kwargs = {
                "source": result.console_launch.source,
                "title": result.console_launch.title,
                "payload": dict(result.console_launch.payload or {}),
            }
            if result.console_launch.status is not None:
                launch_kwargs["status"] = result.console_launch.status
            if result.console_launch.recovery is not None:
                launch_kwargs["recovery"] = result.console_launch.recovery
            if result.console_launch.action_label is not None:
                launch_kwargs["action_label"] = result.console_launch.action_label
            self.open_console_for_live_work(**launch_kwargs)
        return result

    def _wire_character_persona_services(self) -> None:
        self.server_character_persona_service = ServerCharacterPersonaService.from_server_context_provider(
            self.server_context_provider,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.local_character_persona_service = LocalCharacterPersonaService(
            self.chachanotes_db,
            persona_store_path=get_user_data_dir() / "tldw_chatbook_personas.json",
        )
        self.character_persona_scope_service = CharacterPersonaScopeService(
            local_service=self.local_character_persona_service,
            server_service=self.server_character_persona_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.server_chat_dictionary_service = ServerChatDictionaryService.from_server_context_provider(
            self.server_context_provider,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.local_chat_dictionary_service = LocalChatDictionaryService(
            self.chachanotes_db,
            history_store_path=get_user_data_dir() / "tldw_chatbook_chat_dictionary_history.json",
        )
        self.chat_dictionary_scope_service = ChatDictionaryScopeService(
            local_service=self.local_chat_dictionary_service,
            server_service=self.server_chat_dictionary_service,
            policy_enforcer=self.service_policy_enforcer,
        )

    def _wire_chat_conversation_services(self) -> None:
        self.local_chat_conversation_service = (
            ChatConversationService(
                self.chachanotes_db,
                rag_context_store_path=get_user_data_dir() / "tldw_chatbook_chat_rag_context.json",
            )
            if getattr(self, "chachanotes_db", None) is not None
            else None
        )
        self.conversation_local_marks_service = (
            ConversationLocalMarksService(self.chachanotes_db)
            if getattr(self, "chachanotes_db", None) is not None
            else None
        )
        self.server_chat_conversation_service = ServerChatConversationService.from_server_context_provider(
            self.server_context_provider,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.chat_conversation_scope_service = ChatConversationScopeService(
            local_service=self.local_chat_conversation_service,
            server_service=self.server_chat_conversation_service,
            policy_enforcer=self.service_policy_enforcer,
        )

    def _wire_writing_services(self) -> None:
        try:
            self.local_writing_service = LocalWritingService(get_writing_db_path())
        except Exception:
            logger.opt(exception=True).warning("Local writing service unavailable during app wiring")
            self.local_writing_service = None
        try:
            self.server_writing_service = ServerWritingService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_writing_service = ServerWritingService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.writing_scope_service = WritingScopeService(
            local_service=self.local_writing_service,
            server_service=self.server_writing_service,
            policy_enforcer=self.service_policy_enforcer,
        )

    def _wire_library_collections_services(self) -> None:
        try:
            self.local_library_collections_db = LibraryCollectionsDB(
                get_library_collections_db_path(),
                CLI_APP_CLIENT_ID,
            )
            self.local_library_collections_service = LocalLibraryCollectionsService(
                self.local_library_collections_db,
            )
            self.library_collections_service = self.local_library_collections_service
        except Exception:
            logger.opt(exception=True).warning(
                "Local Library Collections service unavailable during app wiring",
            )
            self.local_library_collections_db = None
            self.local_library_collections_service = None
            self.library_collections_service = None

    def _wire_workspace_registry_services(self) -> None:
        try:
            self.local_workspace_db = WorkspaceDB(
                get_workspaces_db_path(),
                CLI_APP_CLIENT_ID,
            )
            self.workspace_registry_service = LocalWorkspaceRegistryService(
                self.local_workspace_db,
            )
            self.workspace_registry_service.ensure_default_workspace()
        except Exception:
            logger.opt(exception=True).warning(
                "Local workspace registry service unavailable during app wiring",
            )
            self.local_workspace_db = None
            self.workspace_registry_service = None

    def _build_chatbook_db_paths(self) -> dict[str, str]:
        return {
            "ChaChaNotes": str(get_chachanotes_db_path()),
            "Media": str(get_media_db_path()),
            "Prompts": str(get_prompts_db_path()),
        }

    def _wire_prompt_chatbook_services(self) -> None:
        self.local_prompt_service = LocalPromptService(prompts_interop)
        self.server_prompt_service = ServerPromptService.from_server_context_provider(
            self.server_context_provider,
            policy_enforcer=self.service_policy_enforcer,
        )

        self.local_chatbook_service = LocalChatbookService(self._build_chatbook_db_paths())
        self.server_chatbook_service = ServerChatbookService.from_server_context_provider(
            self.server_context_provider,
            policy_enforcer=self.service_policy_enforcer,
        )

        self.prompt_chatbook_scope_service = PromptChatbookScopeService(
            local_prompt_service=self.local_prompt_service,
            server_prompt_service=self.server_prompt_service,
            local_chatbook_service=self.local_chatbook_service,
            server_chatbook_service=self.server_chatbook_service,
            policy_enforcer=self.service_policy_enforcer,
        )

    def _wire_evaluation_services(self) -> None:
        self.local_evaluation_service = None
        try:
            self.evaluation_orchestrator = EvaluationOrchestrator(client_id="tldw_cli_app")
            self.local_evaluation_service = LocalEvaluationsService(self.evaluation_orchestrator.db)
        except Exception:
            logger.opt(exception=True).warning("Local evaluation service unavailable during app wiring")
            self.evaluation_orchestrator = None

        try:
            self.server_evaluation_service = ServerEvaluationsService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_evaluation_service = ServerEvaluationsService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )

        has_local = self.local_evaluation_service is not None
        has_server = (
            getattr(self.server_evaluation_service, "client", None) is not None
            or getattr(self.server_evaluation_service, "client_provider", None) is not None
        )
        if not has_local and not has_server:
            self.evaluation_scope_service = None
            return

        self.evaluation_scope_service = EvaluationScopeService(
            local_service=self.local_evaluation_service,
            server_service=self.server_evaluation_service,
            policy_enforcer=self.service_policy_enforcer,
        )

    def _wire_study_services(self) -> None:
        self.local_study_service = (
            LocalStudyService(
                self.chachanotes_db,
                notification_dispatch_service=self.notification_dispatch_service,
                notification_app=self,
            )
            if self.chachanotes_db is not None
            else None
        )
        self.local_quiz_service = (
            LocalQuizService(
                self.chachanotes_db,
                notification_dispatch_service=self.notification_dispatch_service,
                notification_app=self,
            )
            if self.chachanotes_db is not None
            else None
        )
        try:
            self.server_study_service = ServerStudyService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_study_service = ServerStudyService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        try:
            self.server_quiz_service = ServerQuizService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_quiz_service = ServerQuizService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.study_scope_service = StudyScopeService(
            local_service=self.local_study_service,
            server_service=self.server_study_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.study_quiz_scope_service = QuizScopeService(
            local_service=self.local_quiz_service,
            server_service=self.server_quiz_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.library_rag_search_service = LibraryLocalRagSearchService(self)
        self.library_ingest_jobs = LibraryIngestJobRegistry()
        # F3 parallel-parse coordinator state (see LibraryIngestQueueMixin):
        # the lazily-created parse-pool handle, the parse->write handoff
        # (job_id -> parsed payload dict, populated by a pool completion and
        # drained by the writer's claim), and the shutdown flag pool
        # callbacks check before touching a closing app.
        self._ingest_parse_pool = None
        self._ingest_parse_pool_generation: int = 0
        self._ingest_parse_jobs_by_generation: dict[int, set[str]] = {}
        self._ingest_parse_pool_stop_event: Optional[threading.Event] = None
        self._ingest_parsed_payloads: dict[str, dict] = {}
        self._ingest_shutdown: bool = False

    def _wire_research_services(self) -> None:
        """Initialize source-aware research services if the broad parity wiring has not already done so."""
        if hasattr(self, "research_scope_service") and hasattr(self, "research_search_scope_service"):
            return

        try:
            self.local_research_service = LocalResearchService(
                get_research_db_path(),
                notification_dispatcher=self.notification_dispatch_service,
                notification_app=self,
            )
        except Exception:
            logger.opt(exception=True).warning("Local research service unavailable during app wiring")
            self.local_research_service = None
        try:
            self.server_research_service = ServerResearchService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_research_service = ServerResearchService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.research_scope_service = ResearchScopeService(
            local_service=self.local_research_service,
            server_service=self.server_research_service,
            policy_enforcer=self.service_policy_enforcer,
            sync_scope_service=getattr(self, "sync_scope_service", None),
        )
        self.local_research_search_service = LocalResearchSearchService(
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_research_search_service = ServerResearchSearchService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_research_search_service = ServerResearchSearchService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.research_search_scope_service = ResearchSearchScopeService(
            local_service=self.local_research_search_service,
            server_service=self.server_research_search_service,
            policy_enforcer=self.service_policy_enforcer,
        )

    def _wire_watchlists_and_notifications_services(self) -> None:
        """Initialize source-aware watchlists and local notification services."""
        self.local_watchlists_service = LocalWatchlistsService(
            db_factory=lambda: SubscriptionsDB(get_subscriptions_db_path(), CLI_APP_CLIENT_ID)
        )
        try:
            self.server_watchlists_service = ServerWatchlistsService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_watchlists_service = ServerWatchlistsService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        try:
            self.server_notifications_service = ServerNotificationsService.from_server_context_provider(
                self.server_context_provider,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_notifications_service = ServerNotificationsService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        try:
            self.client_notifications_db = ClientNotificationsDB(
                get_notifications_db_path(),
                CLI_APP_CLIENT_ID,
            )
        except Exception as exc:
            logger.opt(exception=True).error(
                "Failed to initialize client notifications DB; using in-memory store: {}",
                exc,
            )
            self.client_notifications_db = ClientNotificationsDB(
                ":memory:",
                CLI_APP_CLIENT_ID,
            )
        self._wire_server_parity_state_repositories()
        self.client_notifications_service = ClientNotificationsService(
            store=self.client_notifications_db,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.notification_dispatch_service = NotificationDispatchService(
            store=self.client_notifications_db,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.notifications_scope_service = NotificationsScopeService(
            local_service=self.client_notifications_service,
            server_service=self.server_notifications_service,
            policy_enforcer=self.service_policy_enforcer,
            event_state_repository=self.event_state_repository,
            server_event_scope_provider=self._server_notification_event_scope,
        )
        self.home_active_work_adapter = LocalNotificationHomeActiveWorkAdapter(
            notification_service=self.client_notifications_service,
            watchlist_service=self.local_watchlists_service,
            chatbook_service=self.local_chatbook_service,
            server_event_service=self.notifications_scope_service,
            runtime_policy=self.runtime_policy,
            flashcards_due_provider=self._local_flashcards_due_count,
            # self.library_ingest_jobs is a plain in-memory registry (no DB,
            # no I/O) assigned later in __init__ (_wire_study_services); this
            # lambda closes over self so it resolves lazily on first Home
            # compose rather than at wiring time here.
            ingest_jobs_provider=lambda: self.library_ingest_jobs.jobs(),
        )
        try:
            self.server_claims_service = ServerClaimsService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_claims_service = ServerClaimsService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.claims_scope_service = ClaimsScopeService(
            server_service=self.server_claims_service,
            policy_enforcer=self.service_policy_enforcer,
        )

        try:
            self.server_meetings_service = ServerMeetingsService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_meetings_service = ServerMeetingsService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.meetings_scope_service = MeetingsScopeService(
            server_service=self.server_meetings_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.server_prompt_studio_service = ServerPromptStudioService.from_server_context_provider(
            self.server_context_provider,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.prompt_studio_scope_service = PromptStudioScopeService(
            server_service=self.server_prompt_studio_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_kanban_service = ServerKanbanService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_kanban_service = ServerKanbanService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.local_kanban_service = LocalKanbanService(
            db_path=get_user_data_dir() / "tldw_chatbook_kanban.db",
            policy_enforcer=self.service_policy_enforcer,
        )
        self.kanban_scope_service = KanbanScopeService(
            local_service=self.local_kanban_service,
            server_service=self.server_kanban_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_translation_service = ServerTranslationService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_translation_service = ServerTranslationService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.translation_scope_service = TranslationScopeService(
            server_service=self.server_translation_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_voice_assistant_service = ServerVoiceAssistantService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_voice_assistant_service = ServerVoiceAssistantService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.voice_assistant_scope_service = VoiceAssistantScopeService(
            server_service=self.server_voice_assistant_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_companion_service = ServerCompanionService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_companion_service = ServerCompanionService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.companion_scope_service = CompanionScopeService(
            server_service=self.server_companion_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_personalization_service = ServerPersonalizationService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_personalization_service = ServerPersonalizationService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.personalization_scope_service = PersonalizationScopeService(
            server_service=self.server_personalization_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_outputs_service = ServerOutputsService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_outputs_service = ServerOutputsService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.outputs_scope_service = OutputsScopeService(
            local_service=None,
            server_service=self.server_outputs_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.local_research_service = LocalResearchService(
                get_research_db_path(),
                notification_dispatcher=self.notification_dispatch_service,
                notification_app=self,
            )
        except Exception:
            logger.opt(exception=True).warning("Local research service unavailable during app wiring")
            self.local_research_service = None
        try:
            self.server_research_service = ServerResearchService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_research_service = ServerResearchService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.research_scope_service = ResearchScopeService(
            local_service=self.local_research_service,
            server_service=self.server_research_service,
            policy_enforcer=self.service_policy_enforcer,
            sync_scope_service=getattr(self, "sync_scope_service", None),
        )
        self.local_research_search_service = LocalResearchSearchService(
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_research_search_service = ServerResearchSearchService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_research_search_service = ServerResearchSearchService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.research_search_scope_service = ResearchSearchScopeService(
            local_service=self.local_research_search_service,
            server_service=self.server_research_search_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.local_chat_grammars_service = LocalChatGrammarsService(
            store_path=get_user_data_dir() / "tldw_chatbook_chat_grammars.json",
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_chat_grammars_service = ServerChatGrammarsService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_chat_grammars_service = ServerChatGrammarsService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.chat_grammars_scope_service = ChatGrammarsScopeService(
            local_service=self.local_chat_grammars_service,
            server_service=self.server_chat_grammars_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.local_feedback_service = LocalFeedbackService(
            store_path=get_user_data_dir() / "tldw_chatbook_feedback.json",
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_feedback_service = ServerFeedbackService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_feedback_service = ServerFeedbackService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.feedback_scope_service = FeedbackScopeService(
            local_service=self.local_feedback_service,
            server_service=self.server_feedback_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_collections_feeds_service = ServerCollectionsFeedsService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_collections_feeds_service = ServerCollectionsFeedsService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.collections_feeds_scope_service = CollectionsFeedsScopeService(
            local_service=self.local_watchlists_service,
            server_service=self.server_collections_feeds_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_connectors_service = ServerConnectorsService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_connectors_service = ServerConnectorsService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.connectors_scope_service = ConnectorsScopeService(
            server_service=self.server_connectors_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_skills_service = ServerSkillsService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_skills_service = ServerSkillsService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        local_skills_store_dir = get_user_data_dir() / "skills"
        skill_trust_marker_store, reduced_rollback_protection = (
            build_skill_trust_marker_store_with_fallback(
                fallback_marker_path=local_skills_store_dir / "trust" / "generation_marker.json"
            )
        )
        self.local_skill_trust_service = SkillTrustService(
            skills_dir=local_skills_store_dir / "skills",
            trust_store=SkillTrustStore(
                store_dir=local_skills_store_dir / "trust",
                marker_store=skill_trust_marker_store,
            ),
            key_cache=build_default_skill_trust_key_cache(),
            keyring_convenience_enabled=False,
            reduced_rollback_protection=reduced_rollback_protection,
        )
        self.local_skills_service = LocalSkillsService(
            store_dir=local_skills_store_dir,
            policy_enforcer=self.service_policy_enforcer,
            trust_service=self.local_skill_trust_service,
        )
        self.skills_scope_service = SkillsScopeService(
            local_service=self.local_skills_service,
            server_service=self.server_skills_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_tools_service = ServerToolsService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_tools_service = ServerToolsService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.tools_scope_service = ToolsScopeService(
            server_service=self.server_tools_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_mcp_governance_service = ServerMCPGovernanceService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_mcp_governance_service = ServerMCPGovernanceService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.mcp_governance_scope_service = MCPGovernanceScopeService(
            server_service=self.server_mcp_governance_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.local_mcp_store = LocalMCPStore(
            get_user_data_dir() / "local_mcp_store.json",
        )
        self.local_mcp_control_service = LocalMCPControlService(
            store=self.local_mcp_store,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.unified_mcp_context_store = UnifiedMCPContextStore(
            get_user_data_dir() / "unified_mcp_context.json",
        )

        def _build_unified_mcp_client_for_target(target: Any) -> "MCPUnifiedClient":
            # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
            from tldw_chatbook.tldw_api import MCPUnifiedClient

            if getattr(target, "auth_reference", None) == "legacy:tldw_api":
                root_client = build_runtime_api_client(
                    app_config=self.app_config,
                    endpoint_url=target.base_url,
                    auth_method=target.auth_mode,
                )
            else:
                root_client = build_runtime_api_client(
                    endpoint_url=target.base_url,
                    auth_token=target.auth_reference,
                    auth_method=target.auth_mode,
                )
            return MCPUnifiedClient(root_client)

        self.server_unified_mcp_service = ServerUnifiedMCPService(
            client_factory=_build_unified_mcp_client_for_target,
            policy_enforcer=self.service_policy_enforcer,
            target_store=self.unified_mcp_target_store,
        )
        self.unified_mcp_service = UnifiedMCPControlPlaneService(
            target_store=self.unified_mcp_target_store,
            context_store=self.unified_mcp_context_store,
            local_service=self.local_mcp_control_service,
            server_service=self.server_unified_mcp_service,
        )
        try:
            self.server_text2sql_service = ServerText2SQLService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_text2sql_service = ServerText2SQLService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.text2sql_scope_service = Text2SQLScopeService(
            server_service=self.server_text2sql_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_sync_service = ServerSyncService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
                state_repository=self.sync_state_repository,
            )
        except ValueError:
            self.server_sync_service = ServerSyncService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
                state_repository=self.sync_state_repository,
            )
        self.sync_scope_service = SyncScopeService(
            server_service=self.server_sync_service,
            policy_enforcer=self.service_policy_enforcer,
            state_repository=self.sync_state_repository,
        )
        self.sync_v2_dataset_keys: dict[str, bytes] = {}
        self.local_first_sync_service = LocalFirstSyncService(
            server_service=self.server_sync_service,
            state_repository=self.sync_state_repository,
            local_store=getattr(self, "sync_v2_local_store", None),
            dataset_keys=self.sync_v2_dataset_keys,
        )
        self.manual_sync_control_service = ManualSyncControlService(
            state_repository=self.sync_state_repository,
            local_first_sync_service=self.local_first_sync_service,
            dataset_keys=self.sync_v2_dataset_keys,
        )
        for domain_scope_service in (
            getattr(self, "chat_conversation_scope_service", None),
            getattr(self, "media_reading_scope_service", None),
            getattr(self, "notes_scope_service", None),
            getattr(self, "research_scope_service", None),
        ):
            if domain_scope_service is not None:
                domain_scope_service.sync_scope_service = self.sync_scope_service
        self.server_runtime_service = ServerRuntimeService.from_server_context_provider(
            self.server_context_provider,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.server_runtime_scope_service = ServerRuntimeScopeService(
            server_service=self.server_runtime_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.active_server_capability_service = ActiveServerCapabilityService(
            runtime_context=self.runtime_policy,
            server_runtime_scope_service=self.server_runtime_scope_service,
            target_store=self.unified_mcp_target_store,
        )
        self.local_llm_provider_catalog_service = LocalLLMProviderCatalogService(
            provider_catalog_loader=lambda: dict(getattr(self, "providers_models", {}) or {}),
            local_provider_names=set(LOCAL_PROVIDERS),
            default_provider=get_cli_setting("chat_defaults", "provider", None),
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_llm_provider_catalog_service = ServerLLMProviderCatalogService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_llm_provider_catalog_service = ServerLLMProviderCatalogService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.llm_provider_catalog_scope_service = LLMProviderCatalogScopeService(
            local_service=self.local_llm_provider_catalog_service,
            server_service=self.server_llm_provider_catalog_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.local_audio_services_service = LocalAudioServicesService(
            tts_provider_loader=lambda: {"chatbook_tts": {"available": True, "source": "local"}},
            stt_provider_loader=lambda: {"chatbook_stt": {"available": True, "source": "local"}},
            voice_catalog_loader=lambda: {},
            history_store_path=get_user_data_dir() / "tldw_chatbook_audio_history.json",
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_audio_services_service = ServerAudioServicesService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_audio_services_service = ServerAudioServicesService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.audio_services_scope_service = AudioServicesScopeService(
            local_service=self.local_audio_services_service,
            server_service=self.server_audio_services_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.server_auth_account_service = ServerAuthAccountService.from_server_context_provider(
            self.server_context_provider,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.auth_account_scope_service = AuthAccountScopeService(
            server_service=self.server_auth_account_service,
            policy_enforcer=self.service_policy_enforcer,
            server_context_provider=self.server_context_provider,
        )
        try:
            self.server_user_governance_service = ServerUserGovernanceService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_user_governance_service = ServerUserGovernanceService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.user_governance_scope_service = UserGovernanceScopeService(
            server_service=self.server_user_governance_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_sharing_service = ServerSharingService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_sharing_service = ServerSharingService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.sharing_scope_service = SharingScopeService(
            server_service=self.server_sharing_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_web_clipper_service = ServerWebClipperService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_web_clipper_service = ServerWebClipperService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.web_clipper_scope_service = WebClipperScopeService(
            server_service=self.server_web_clipper_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        try:
            self.server_web_scraping_service = ServerWebScrapingService.from_config(
                self.app_config,
                policy_enforcer=self.service_policy_enforcer,
            )
        except ValueError:
            self.server_web_scraping_service = ServerWebScrapingService(
                client=None,
                policy_enforcer=self.service_policy_enforcer,
            )
        self.web_scraping_scope_service = WebScrapingScopeService(
            server_service=self.server_web_scraping_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.watchlist_scope_service = WatchlistScopeService(
            local_service=self.local_watchlists_service,
            server_service=self.server_watchlists_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.local_media_reading_service.notification_dispatcher = self.notification_dispatch_service
        self.local_media_reading_service.notification_app = self
        self.local_watchlists_service.notification_dispatcher = self.notification_dispatch_service
        self.local_watchlists_service.notification_app = self

    def _wire_server_parity_state_repositories(self) -> None:
        try:
            self.server_parity_state = build_server_parity_state_repositories(
                data_dir=get_user_data_dir(),
                client_id=CLI_APP_CLIENT_ID,
                local_notifications_db=self.client_notifications_db,
            )
        except Exception as exc:
            logger.opt(exception=True).error(
                "Failed to initialize server parity state repositories; using in-memory stores: {}",
                exc,
            )
            self.server_parity_state = ServerParityStateRepositories(
                local_notifications_db=self.client_notifications_db,
                event_state_repository=EventStateRepository(":memory:", CLI_APP_CLIENT_ID),
                sync_state_repository=SyncStateRepository(":memory:", CLI_APP_CLIENT_ID),
            )
        self.event_state_repository = self.server_parity_state.event_state_repository
        self.sync_state_repository = self.server_parity_state.sync_state_repository

    def _resolve_initial_media_runtime_backend(self) -> str:
        """Default media backend to local when no valid runtime value is available."""
        for candidate in (
            getattr(self, "current_runtime_backend", None),
            getattr(self, "runtime_backend", None),
        ):
            normalized = str(candidate or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
        return "local"

    def get_authoritative_runtime_source(self) -> str:
        runtime_state = getattr(getattr(self, "runtime_policy", None), "state", None)
        if isinstance(runtime_state, RuntimeSourceState):
            normalized = str(runtime_state.active_source or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
        return self._resolve_initial_media_runtime_backend()

    def _server_notification_event_scope(self) -> dict[str, str | None]:
        runtime_state = getattr(getattr(self, "runtime_policy", None), "state", None)
        active_server_id = getattr(runtime_state, "active_server_id", None)
        authenticated_principal_id = None
        server_context_provider = getattr(self, "server_context_provider", None)
        get_active_context = getattr(server_context_provider, "get_active_context", None)
        if callable(get_active_context):
            try:
                authenticated_principal_id = event_principal_id_from_active_context(get_active_context())
            except Exception:
                authenticated_principal_id = None
        return {
            "server_profile_id": str(active_server_id) if active_server_id else None,
            "authenticated_principal_id": authenticated_principal_id,
            "stream_instance_id": "global",
        }

    def require_ui_action_allowed(
        self,
        *,
        action_id: str,
        scope_type: str | None = None,
        runtime_state_override: RuntimeSourceState | None = None,
    ) -> PolicyDecision:
        _ = scope_type
        state = runtime_state_override if isinstance(runtime_state_override, RuntimeSourceState) else None
        if state is None:
            policy_enforcer = getattr(self, "service_policy_enforcer", None)
            if policy_enforcer is not None and hasattr(policy_enforcer, "current_state"):
                state = policy_enforcer.current_state()
        if not isinstance(state, RuntimeSourceState):
            runtime_state = getattr(getattr(self, "runtime_policy", None), "state", None)
            if isinstance(runtime_state, RuntimeSourceState):
                state = runtime_state

        if not isinstance(state, RuntimeSourceState):
            decision = PolicyDecision(
                allowed=False,
                reason_code="authority_denied",
                user_message="Runtime policy state is unavailable.",
                effective_source="unknown",
                authority_owner="unknown",
            )
            notifier = getattr(self, "notify", None)
            if callable(notifier):
                notifier(decision.user_message, severity="warning")
            return decision

        engine = getattr(self, "ui_policy_engine", None)
        if engine is None:
            engine = PolicyEngine(CAPABILITY_REGISTRY)
            self.ui_policy_engine = engine

        decision = engine.evaluate(
            action_id=action_id,
            state=state,
        )
        if not decision.allowed:
            notifier = getattr(self, "notify", None)
            if callable(notifier):
                notifier(decision.user_message, severity="warning")
        return decision

    async def handle_runtime_backend_changed(self, runtime_backend: str) -> None:
        normalized_backend = str(runtime_backend or "").strip().lower()
        if normalized_backend in {"local", "server"}:
            if getattr(self, "runtime_policy", None) is not None:
                previous_server_id = getattr(self.runtime_policy.state, "active_server_id", None)
                updated_state = set_authoritative_runtime_source(self, normalized_backend)
                server_context_provider = getattr(self, "server_context_provider", None)
                invalidate_for_server_switch = getattr(
                    server_context_provider,
                    "invalidate_for_server_switch",
                    None,
                )
                if callable(invalidate_for_server_switch):
                    invalidate_for_server_switch(previous_server_id, updated_state.active_server_id)
            else:
                self.current_runtime_backend = normalized_backend
                self.runtime_backend = normalized_backend

        resolved_backend = normalized_backend
        runtime_state = getattr(getattr(self, "runtime_policy", None), "state", None)
        if runtime_state is not None:
            resolved_backend = str(runtime_state.active_source or normalized_backend).strip().lower()
        elif resolved_backend not in {"local", "server"}:
            resolved_backend = str(getattr(self, "current_runtime_backend", "local") or "local").strip().lower()
        active_screen = getattr(self, "screen", None)
        callback = getattr(active_screen, "handle_runtime_backend_changed", None)
        if callable(callback):
            await callback(resolved_backend)


    def _init_notes_service(self, user_name_for_notes: str) -> None:
        """Initialize notes service - for parallel execution."""
        try:
            # Get the full path to the unified ChaChaNotes DB FILE
            chachanotes_db_file_path = get_chachanotes_db_path()
            logger.info(f"Unified ChaChaNotes DB file path: {chachanotes_db_file_path}")
            
            # Determine the PARENT DIRECTORY for NotesInteropService's 'base_db_directory'
            actual_base_directory_for_service = chachanotes_db_file_path.parent
            logger.info(f"Notes for user '{user_name_for_notes}' will use the unified DB: {chachanotes_db_file_path}")
            
            self.notes_service = NotesInteropService(
                base_db_directory=actual_base_directory_for_service,
                api_client_id="tldw_tui_client_v1",
                global_db_to_use=get_chachanotes_db_lazy()
            )
            logger.info(f"NotesInteropService successfully initialized for user '{user_name_for_notes}'.")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to initialize NotesInteropService: {e}")
            self.notes_service = None
    
    def _init_providers_models(self) -> None:
        """Initialize providers and models - for parallel execution."""
        try:
            self.providers_models = get_cli_providers_and_models()
            logger.info(f"Successfully retrieved providers_models. Count: {len(self.providers_models)}. Keys: {list(self.providers_models.keys())}")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to get providers and models: {e}")
            self.providers_models = {}
    
    def _init_prompts_service(self) -> None:
        """Initialize prompts service - for parallel execution."""
        self.prompts_service_initialized = False
        try:
            prompts_db_path = get_prompts_db_path()
            prompts_db_path.parent.mkdir(parents=True, exist_ok=True)
            prompts_interop.initialize_interop(db_path=prompts_db_path, client_id=self.prompts_client_id)
            self.prompts_service_initialized = True
            logger.info(f"Prompts Interop Service initialized with DB: {prompts_db_path}")
        except Exception as e:
            self.prompts_service_initialized = False
            logger.opt(exception=True).error(f"Failed to initialize Prompts Interop Service: {e}")
    
    def _init_media_db(self) -> None:
        """Initialize media database - for parallel execution."""
        try:
            media_db_path = get_media_db_path()
            # Get integrity check configuration
            check_integrity = self.app_config.get('database', {}).get('check_integrity_on_startup', False)
            self.media_db = MediaDatabase(
                db_path=media_db_path, 
                client_id=CLI_APP_CLIENT_ID,
                check_integrity_on_startup=check_integrity
            )
            logger.info(f"Media_DB_v2 initialized successfully for client '{CLI_APP_CLIENT_ID}' at {media_db_path}")

            # Wire ingestion-time RAG indexing (task-247). The hook no-ops
            # when the embeddings_rag extras are missing; indexing failures
            # are logged and surfaced without ever affecting ingestion.
            try:
                from .RAG_Search.ingestion_indexing import install_media_ingest_hook
                install_media_ingest_hook(failure_notifier=self._notify_rag_indexing_failure)
            except Exception as e:
                logger.warning(f"Could not install RAG ingestion-indexing hook: {e}")

            # Pre-fetch media types for UI
            if self.media_db:
                db_types = self.media_db.get_distinct_media_types(include_deleted=False, include_trash=False)
                self._media_types_for_ui = ["All Media"] + sorted(list(set(db_types)))
                logger.info(f"Pre-fetched {len(self._media_types_for_ui)} media types for UI.")
            else:
                self._media_types_for_ui = ["Error: Media DB not loaded"]
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to initialize media DB: {e}")
            self.media_db = None
            self._media_types_for_ui = ["Error: Exception fetching media types"]

    def _notify_rag_indexing_failure(self, message: str) -> None:
        """Surface a background RAG-indexing failure as a toast (best effort).

        Called from the ingestion-indexer worker thread, so the notification
        is marshalled onto the UI thread; if the app isn't running yet (or
        anymore) the failure stays log-only.
        """
        try:
            self.call_from_thread(self.notify, message, severity="warning", timeout=6)
        except Exception as e:
            logger.debug(f"Could not surface RAG indexing failure in UI: {e}")

    def _init_worker_handlers(self) -> None:
        """Initialize the worker handler registry and register all handlers."""
        self.worker_handler_registry = WorkerHandlerRegistry(self)
        
        # Register all worker handlers
        self.worker_handler_registry.register(ChatWorkerHandler(self))
        self.worker_handler_registry.register(ServerWorkerHandler(self))
        self.worker_handler_registry.register(AIGenerationHandler(self))
        self.worker_handler_registry.register(MiscWorkerHandler(self))
        
        self.loguru_logger.info("Worker handler registry initialized with all handlers")

    def _build_handler_map(self) -> dict:
        """Constructs the master button handler map from all event modules."""

        # --- Generic, Awaitable Helper Handlers ---
        async def _handle_nav(app: 'TldwCli', event: Button.Pressed, *, prefix: str, reactive_attr: str) -> None:
            """Generic handler for switching views within a tab."""
            view_to_activate = event.button.id.replace(f"{prefix}-nav-", f"{prefix}-view-")
            app.loguru_logger.info(f"_handle_nav called: Nav button '{event.button.id}' pressed. Prefix: '{prefix}', Reactive attr: '{reactive_attr}', Activating view '{view_to_activate}'.")
            old_value = getattr(app, reactive_attr, None)
            setattr(app, reactive_attr, view_to_activate)
            new_value = getattr(app, reactive_attr, None)
            app.loguru_logger.info(f"_handle_nav: Set {reactive_attr} from '{old_value}' to '{new_value}'")

        async def _handle_sidebar_toggle(app: 'TldwCli', event: Button.Pressed, *, reactive_attr: str) -> None:
            """Generic handler for toggling a sidebar's collapsed state."""
            setattr(app, reactive_attr, not getattr(app, reactive_attr))

        # --- LLM Management Handlers ---
        llm_handlers_map = {
            **llm_management_events.LLM_MANAGEMENT_BUTTON_HANDLERS,
            **llm_nav_events.LLM_NAV_BUTTON_HANDLERS,
            **llm_management_events_mlx_lm.MLX_LM_BUTTON_HANDLERS,
            **llm_management_events_ollama.OLLAMA_BUTTON_HANDLERS,
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
        
        # Add handlers for special media sub-tabs
        media_handlers_map["media-nav-analysis-review"] = media_events.handle_media_nav_button_pressed
        media_handlers_map["media-nav-collections-tags"] = media_events.handle_media_nav_button_pressed
        media_handlers_map["media-nav-multi-item-review"] = media_events.handle_media_nav_button_pressed

        # --- Search Handlers ---
        search_handlers = {
            SEARCH_NAV_RAG_QA: functools.partial(_handle_nav, prefix="search", reactive_attr="search_active_sub_tab"),
            SEARCH_NAV_RAG_CHAT: functools.partial(_handle_nav, prefix="search", reactive_attr="search_active_sub_tab"),
            SEARCH_NAV_RAG_MANAGEMENT: functools.partial(_handle_nav, prefix="search",
                                                         reactive_attr="search_active_sub_tab"),
            SEARCH_NAV_WEB_SEARCH: functools.partial(_handle_nav, prefix="search",
                                                     reactive_attr="search_active_sub_tab"),
            SEARCH_NAV_EMBEDDINGS_CREATE: functools.partial(_handle_nav, prefix="search", 
                                                           reactive_attr="search_active_sub_tab"),
            SEARCH_NAV_EMBEDDINGS_MANAGE: functools.partial(_handle_nav, prefix="search",
                                                           reactive_attr="search_active_sub_tab"),
        }

        # --- Ingest Handlers ---
        ingest_handlers_map = {
            **ingest_events.INGEST_BUTTON_HANDLERS,
            # Add nav handlers using the helper
            **{button_id: functools.partial(_handle_nav, prefix="ingest", reactive_attr="ingest_active_view")
               for button_id in INGEST_NAV_BUTTON_IDS}
        }

        # --- Tools & Settings Handlers ---
        tools_settings_handlers = {
            "ts-nav-general-settings": functools.partial(_handle_nav, prefix="ts",
                                                         reactive_attr="tools_settings_active_view"),
            "ts-nav-config-file-settings": functools.partial(_handle_nav, prefix="ts",
                                                             reactive_attr="tools_settings_active_view"),
            "ts-nav-db-tools": functools.partial(_handle_nav, prefix="ts",
                                                 reactive_attr="tools_settings_active_view"),
            "ts-nav-appearance": functools.partial(_handle_nav, prefix="ts",
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
            TAB_CUSTOMIZE: {},  # Customize handles its own events
            TAB_SEARCH: search_handlers,
            TAB_EVALS: evals_handlers,
            TAB_CODING: {},  # Empty for now - coding handles its own events
            TAB_STTS: {}, # STTS handles its own events
            TAB_STUDY: {}, # Study handles its own events
            TAB_SUBSCRIPTIONS: {
                "subscription-add-button": subscription_events.handle_add_subscription,
                "subscription-check-all-button": subscription_events.handle_check_all_subscriptions,
                "subscription-accept-button": subscription_events.handle_subscription_item_action,
                "subscription-ignore-button": subscription_events.handle_subscription_item_action,
                "subscription-mark-reviewed-button": subscription_events.handle_subscription_item_action,
            },
        }

    def _setup_buffered_logging(self):
        """Set up a persistent buffered logging handler for screen navigation mode."""
        from collections import deque
        import logging
        
        # Create a buffer to store ALL log messages (no max length)
        if not hasattr(self, '_log_buffer'):
            self._log_buffer = deque()  # No maxlen - keep all logs
        
        # Create a custom handler that stores logs in the buffer
        class PersistentLogHandler(logging.Handler):
            def __init__(self, buffer, app):
                super().__init__()
                self.buffer = buffer
                self.app = app
                
            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.buffer.append(msg)
                    
                    # If we have a RichLog widget active, also write to it directly
                    if hasattr(self.app, '_current_log_widget') and self.app._current_log_widget:
                        try:
                            self.app._current_log_widget.write(msg)
                        except:
                            pass  # Widget might not be mounted
                except Exception:
                    self.handleError(record)
        
        # Add the persistent handler to the root logger
        if not hasattr(self, '_persistent_log_handler'):
            self._persistent_log_handler = PersistentLogHandler(self._log_buffer, self)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self._persistent_log_handler.setFormatter(formatter)
            logging.getLogger().addHandler(self._persistent_log_handler)
            logger.info("Persistent logging handler set up for screen navigation")
            
        # Initialize current log widget reference
        self._current_log_widget = None
    
    def _display_buffered_logs(self, log_widget):
        """Display all buffered logs in the RichLog widget."""
        if not hasattr(self, '_log_buffer'):
            return
            
        # Store reference to current log widget
        self._current_log_widget = log_widget
        
        # Clear the widget first to avoid duplicates
        log_widget.clear()
        
        # Write all buffered messages to the widget
        for msg in self._log_buffer:
            log_widget.write(msg)
        
        # Scroll to the latest entry
        log_widget.scroll_end()
        
        logger.debug(f"Displayed {len(self._log_buffer)} buffered log entries")
    
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
        compose_start = time.perf_counter()
        self._ui_compose_start_time = compose_start  # Store for later reference
        logging.debug("App composing UI...")
        log_counter("ui_compose_started", 1, documentation="UI composition started")
        
        # Check if splash screen is enabled
        splash_enabled = get_cli_setting("splash_screen", "enabled", True)
        logging.info(f"Splash screen enabled: {splash_enabled}")
        if splash_enabled:
            # Get splash screen configuration
            splash_duration = get_cli_setting("splash_screen", "duration", 1.5)
            splash_skip = get_cli_setting("splash_screen", "skip_on_keypress", True)
            splash_progress = get_cli_setting("splash_screen", "show_progress", True)
            splash_card = get_cli_setting("splash_screen", "card_selection", "random")
            logging.info(f"Creating splash screen - duration: {splash_duration}, card: {splash_card}")
            
            # Create and yield splash screen
            self._splash_screen_widget = SplashScreen(
                card_name=splash_card if splash_card != "random" else None,
                duration=splash_duration,
                skip_on_keypress=splash_skip,
                show_progress=splash_progress,
                id="app-splash-screen"
            )
            self.splash_screen_active = True
            yield self._splash_screen_widget
            logging.info("Splash screen yielded, returning early from compose")
            
            # Important: Return early to only show splash screen initially
            # The main UI will be mounted after splash screen is closed
            return
        
        # If splash screen is disabled, compose the main UI immediately
        yield from self._compose_main_ui()
    
    def _compose_main_ui(self) -> ComposeResult:
        """Compose the main UI by yielding created widgets."""
        widgets = self._create_main_ui_widgets()
        for widget in widgets:
            yield widget
        
    def _create_main_ui_widgets(self) -> List[Widget]:
        """Create the main UI widgets (called after splash screen or immediately if disabled)."""
        widgets = []
        self._start_ui_responsiveness_monitor()
        
        # ALWAYS use screen-based navigation now
        logger.info("Using screen-based navigation - skipping widget creation")
        # Note: _use_screen_navigation is already set to True in __init__
        
        # Add title bar and navigation for screen mode
        widgets.append(TitleBar())
        
        # Add navigation bar that will emit NavigateToScreen messages
        use_dropdown = get_cli_setting("general", "use_dropdown_navigation", False)
        use_links = get_cli_setting("general", "use_link_navigation", True)
        
        if use_dropdown:
            widgets.append(TabDropdown(tab_ids=ALL_TABS, initial_active_tab=self._initial_tab_value))
            logger.info("Using dropdown navigation for screens")
        elif use_links:
            widgets.append(TabLinks(tab_ids=ALL_TABS, initial_active_tab=self._initial_tab_value))
            logger.info("Using single-line link navigation for screens")
        else:
            widgets.append(TabBar(tab_ids=ALL_TABS, initial_active_tab=self._initial_tab_value))
            logger.info("Using tab bar navigation for screens")
        
        # Add container for screens and footer
        widgets.append(Container(id="screen-container"))
        widgets.append(AppFooterStatus(id="app-footer-status"))
        
        return widgets
        
        # Screen-based navigation is used exclusively - no tab-based UI components needed
        return widgets

    def _start_ui_responsiveness_monitor(self) -> None:
        """Start the low-cost UI responsiveness heartbeat."""
        interval_seconds = 1.0
        try:
            if self.ui_responsiveness_monitor is None:
                enabled = bool(
                    get_cli_setting("diagnostics", "ui_responsiveness_enabled", True)
                )
                self.ui_responsiveness_monitor = UIResponsivenessMonitor(
                    enabled=enabled,
                    heartbeat_interval_seconds=interval_seconds,
                )
            if not self.ui_responsiveness_monitor.enabled:
                return
            self.ui_responsiveness_monitor.record_timer_created("ui-heartbeat")
            if getattr(self, "_ui_responsiveness_heartbeat_timer", None) is None:
                self.ui_responsiveness_monitor.reset_heartbeat_baseline()
                self._ui_responsiveness_heartbeat_timer = self.set_interval(
                    interval_seconds,
                    self._record_ui_heartbeat,
                )
        except Exception as exc:
            logger.debug(f"UI responsiveness heartbeat setup skipped: {exc}")

    def _record_ui_heartbeat(self) -> None:
        """Record event-loop heartbeat drift without affecting UI behavior."""
        try:
            monitor = self.ui_responsiveness_monitor
            if monitor is not None:
                monitor.heartbeat()
        except Exception as exc:
            logger.debug(f"UI responsiveness heartbeat skipped: {exc}")

    def _stop_ui_responsiveness_monitor(self) -> None:
        """Stop the UI responsiveness heartbeat timer if it is active."""
        timer = getattr(self, "_ui_responsiveness_heartbeat_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception as exc:
                logger.debug(f"UI responsiveness heartbeat stop skipped: {exc}")
            finally:
                self._ui_responsiveness_heartbeat_timer = None
        try:
            monitor = self.ui_responsiveness_monitor
            if monitor is not None:
                monitor.record_timer_stopped("ui-heartbeat")
        except Exception:
            return

    def _record_ui_responsiveness_timer_created(self, name: str) -> None:
        """Best-effort timer diagnostic hook."""
        try:
            monitor = self.ui_responsiveness_monitor
            if monitor is not None:
                monitor.record_timer_created(name)
        except Exception:
            return

    def _record_ui_responsiveness_timer_stopped(self, name: str) -> None:
        """Best-effort timer diagnostic stop hook."""
        try:
            monitor = self.ui_responsiveness_monitor
            if monitor is not None:
                monitor.record_timer_stopped(name)
        except Exception:
            return

    def _stop_footer_status_timers(self) -> None:
        """Stop footer status timers and clear their diagnostic entries."""
        timer = getattr(self, "_token_count_update_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception as exc:
                logger.debug(f"Footer token timer stop skipped: {exc}")
            finally:
                self._token_count_update_timer = None
        self._record_ui_responsiveness_timer_stopped("footer-db-size-periodic")
        self._record_ui_responsiveness_timer_stopped("footer-token-periodic")

    def _record_footer_timer_created(self, name: str) -> None:
        """Record footer timer creation without making diagnostics mandatory."""
        record_timer = getattr(
            self,
            "_record_ui_responsiveness_timer_created",
            None,
        )
        try:
            if callable(record_timer):
                record_timer(name)
                return
            monitor = getattr(self, "ui_responsiveness_monitor", None)
            if monitor is not None:
                monitor.record_timer_created(name)
        except Exception:
            return

    def _resolve_screen_navigation_target(self, target: str):
        """Normalize navigation aliases to a routed screen id and canonical current_tab value."""
        return resolve_screen_target(target)

    # Legacy alias routes that need a default Library nav-context applied
    # when navigated to directly (bare ``NavigateToScreen(route)``, no
    # explicit context supplied). Mirrors how ``open_notes_workspace`` builds
    # ``{LIBRARY_NAV_CONTEXT_MODE: "notes"}`` for the retired standalone
    # Notes tab -- except "prompts" (the retired Personas "prompts" mode
    # chip, Task 7) and "skills" (the retired standalone Skills tab, Skills
    # sub-project Task 5) have no dedicated re-entry action to carry that
    # context, so the bare alias route itself must supply it here.
    _LEGACY_ROUTE_LIBRARY_NAV_CONTEXT: dict[str, dict[str, str]] = {
        "prompts": {LIBRARY_NAV_CONTEXT_MODE: "prompts"},
        "skills": {LIBRARY_NAV_CONTEXT_MODE: "skills"},
    }

    def _create_navigation_screen(self, screen_name: str, screen_class: type):
        """Build a FRESH screen instance for every navigation.

        Args:
            screen_name: Routed screen id (used by callers for state keying;
                unused here, kept for signature stability at the seam).
            screen_class: The Screen subclass registered for the route.

        Returns:
            A newly constructed, never-mounted instance of ``screen_class``.

        Screens must never be cached and re-mounted: ``switch_screen``
        unmounts the outgoing screen, and re-mounting a previously-unmounted
        instance races its still-in-flight teardown under rapid tab
        switching -- child message pumps end up permanently stopped while
        the widgets stay attached (``mounted=True``), the compositor keeps
        presenting a stale frame, and every subsequent click is hit-tested
        into the dead tree and silently swallowed: a total, exception-free
        UI freeze (root-caused 2026-07-11). UX continuity across visits is
        the job of ``_screen_states`` (``save_state``/``restore_state``),
        not instance reuse.
        """
        return screen_class(self)

    def _valid_startup_route_ids(self) -> set[str]:
        """Return route ids allowed in startup config during the shell migration."""
        from .UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

        shell_routes = {
            destination.primary_route
            for destination in SHELL_DESTINATION_ORDER
        } | {
            destination.destination_id
            for destination in SHELL_DESTINATION_ORDER
        }
        legacy_aliases = {"conversation", "llm", "subscription", "subscriptions", "tools_settings", "notes", "prompts"}
        return set(ALL_TABS) | shell_routes | legacy_aliases

    def _normalize_initial_tab_from_config(self, configured_route: str | None) -> str:
        """Validate configured startup route without discarding new shell routes."""
        candidate = configured_route or TAB_CHAT
        if candidate in self._valid_startup_route_ids():
            return candidate

        logging.warning(
            "Default tab '%s' from config not valid. Falling back to '%s'.",
            candidate,
            TAB_CHAT,
        )
        return TAB_CHAT

    def _resolve_initial_shell_route(self) -> str:
        """Choose the startup route while keeping first-run orientation explicit."""
        if self.app_config.get("_first_run", False):
            return TAB_HOME
        return getattr(self, "_initial_tab_value", TAB_CHAT)
        

    @on(NavigateToScreen)
    async def handle_screen_navigation(self, message: NavigateToScreen) -> None:
        """Handle navigation to a different screen using switch_screen for better performance."""
        requested_screen = message.screen_name
        screen_name, current_tab_value, screen_class = self._resolve_screen_navigation_target(requested_screen)
        logger.info(f"Navigating to screen: {requested_screen}")

        current_screen = self.screen

        # Screens are never reused across navigations, so anything the
        # outgoing screen has not persisted is destroyed with its instance.
        # Give it one awaited chance to flush pending work (e.g. a Library
        # note edit whose debounced autosave has not fired); False vetoes
        # the switch, leaving the screen (and e.g. its save-conflict banner)
        # in place for the user.
        flush = getattr(current_screen, "flush_pending_work", None)
        if callable(flush):
            try:
                flush_result = flush()
                if inspect.isawaitable(flush_result):
                    flush_result = await flush_result
                if flush_result is False:
                    logger.info(
                        f"Navigation to {screen_name} vetoed by the outgoing "
                        "screen's pending-work flush"
                    )
                    return
            except Exception as e:
                # The outgoing instance may be the only place pending edits
                # still exist, so a failed flush must abort the transition.
                logger.opt(exception=True).error(
                    "Error flushing outgoing screen "
                    f"{getattr(current_screen, 'screen_name', type(current_screen).__name__)!r} "
                    f"before navigating to {screen_name!r}: {e}"
                )
                try:
                    self.notify(
                        "Couldn't save pending changes before switching screens.",
                        severity="warning",
                    )
                except Exception:
                    pass
                return

        # Save state of current screen before switching
        if current_screen and hasattr(current_screen, 'save_state'):
            try:
                state = current_screen.save_state()
                if isinstance(state, dict):
                    state = add_runtime_policy_snapshot(state, self.runtime_policy.state)
                # Store state in a dictionary keyed by screen name
                if not hasattr(self, '_screen_states'):
                    self._screen_states = {}
                if hasattr(current_screen, 'screen_name'):
                    self._screen_states[current_screen.screen_name] = state
                    logger.debug(f"Saved state for screen: {current_screen.screen_name}")
            except Exception as e:
                logger.error(f"Error saving screen state: {e}")
        
        if screen_class:
            new_screen = self._create_navigation_screen(screen_name, screen_class)
            
            # Restore state if available
            if hasattr(self, '_screen_states') and screen_name in self._screen_states:
                if hasattr(new_screen, 'restore_state'):
                    try:
                        restored_state = reconcile_saved_screen_state(
                            self._screen_states[screen_name],
                            self.runtime_policy.state,
                        )
                        if restored_state is None:
                            self._screen_states.pop(screen_name, None)
                            logger.info(f"Dropped saved state for screen due to runtime policy mismatch: {screen_name}")
                        else:
                            new_screen.restore_state(restored_state)
                            logger.debug(f"Restored state for screen: {screen_name}")
                    except Exception as e:
                        logger.error(f"Error restoring screen state: {e}")

            navigation_context = getattr(message, "screen_context", {}) or {}
            if not navigation_context:
                navigation_context = self._LEGACY_ROUTE_LIBRARY_NAV_CONTEXT.get(
                    requested_screen, {}
                )
            if navigation_context and hasattr(new_screen, "apply_navigation_context"):
                try:
                    result = new_screen.apply_navigation_context(navigation_context)
                    if inspect.isawaitable(result):
                        await result
                except Exception as e:
                    logger.error(f"Error applying navigation context for {screen_name}: {e}")
            
            # Use switch_screen to replace the current screen
            await self.switch_screen(new_screen)
            
            # Keep current_tab aligned to canonical tab ids even when routing uses aliases.
            self.current_tab = current_tab_value

            logger.info(f"Successfully switched to {screen_name} screen")
        else:
            logger.error(f"Unknown screen requested: {requested_screen}")
    
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

    @on(ChatMessageEnhanced.Action)
    async def handle_chat_message_enhanced_action(self, event: ChatMessageEnhanced.Action) -> None:
        """Handles actions (edit, copy, etc.) from within a ChatMessageEnhanced widget."""
        button_classes = " ".join(event.button.classes) # Get class string for logging
        self.loguru_logger.debug(
            f"ChatMessageEnhanced.Action received for button "
            f"(Classes: {button_classes}, Label: '{event.button.label}') "
            f"on message role: {event.message_widget.role}"
        )
        # The event directly gives us the context we need.
        # Now we call the existing handler function with the correct arguments.
        await chat_events.handle_chat_action_button_pressed(
            self, event.button, event.message_widget
        )

    @on(TTSRequestEvent)
    async def handle_tts_request_event(self, event: TTSRequestEvent) -> None:
        """Handle TTS generation request."""
        self.loguru_logger.info(f"TTS request received for text: '{event.text[:50]}...'")
        handler = await self._ensure_tts_handler()
        if handler:
            await handler.handle_tts_request(event)
        else:
            self.loguru_logger.error("TTS handler not initialized")
            await self.post_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error="TTS service not available"
                )
            )

    @on(TTSCompleteEvent)
    async def handle_tts_complete_event(self, event: TTSCompleteEvent) -> None:
        """Handle TTS generation completion."""
        self.loguru_logger.info(f"TTS complete for message {event.message_id}")
        
        if event.error:
            self.notify(f"TTS failed: {event.error}", severity="error")
            # Update widget state back to idle on error
            try:
                if event.message_id:
                    # Find the message widget and update state
                    for message_widget in list(self.query(ChatMessage)) + list(self.query(ChatMessageEnhanced)):
                        if getattr(message_widget, 'message_id_internal', None) == event.message_id:
                            # Update TTS state to idle on error
                            if hasattr(message_widget, 'update_tts_state'):
                                message_widget.update_tts_state("idle")
                            # Remove TTS generating class
                            text_widget = message_widget.query_one(".message-text", Markdown)
                            text_widget.remove_class("tts-generating")
                            break
            except Exception as e:
                self.loguru_logger.error(f"Error updating message UI: {e}")
        else:
            # Update widget state to ready with audio file
            if event.audio_file and event.audio_file.exists():
                try:
                    if event.message_id:
                        # Find the message widget and update state
                        for message_widget in list(self.query(ChatMessage)) + list(self.query(ChatMessageEnhanced)):
                            if getattr(message_widget, 'message_id_internal', None) == event.message_id:
                                # Update TTS state to ready with audio file
                                if hasattr(message_widget, 'update_tts_state'):
                                    message_widget.update_tts_state("ready", event.audio_file)
                                # Remove TTS generating class
                                try:
                                    text_widget = message_widget.query_one(".message-text", Markdown)
                                    text_widget.remove_class("tts-generating")
                                except Exception:
                                    pass
                                break
                    # Don't automatically play or delete - let user control playback
                    self.notify("TTS audio ready - click play to listen", severity="information")
                except Exception as e:
                    self.loguru_logger.error(f"Error playing audio: {e}")
                    self.notify("Failed to play audio", severity="error")
            
            # Remove TTS generating class from message
            try:
                if event.message_id:
                    for message_widget in list(self.query(ChatMessage)) + list(self.query(ChatMessageEnhanced)):
                        if getattr(message_widget, 'message_id_internal', None) == event.message_id:
                            text_widget = message_widget.query_one(".message-text", Markdown)
                            text_widget.remove_class("tts-generating")
                            break
            except Exception as e:
                self.loguru_logger.error(f"Error updating message UI: {e}")
    
    @on(TTSProgressEvent)
    async def handle_tts_progress_event(self, event: TTSProgressEvent) -> None:
        """Handle TTS generation progress updates."""
        self.loguru_logger.debug(f"TTS progress for message {event.message_id}: {event.progress:.0%} - {event.status}")
        
        try:
            if event.message_id:
                # Find the message widget and update progress
                for message_widget in self.query(ChatMessage).union(self.query(ChatMessageEnhanced)):
                    if getattr(message_widget, 'message_id_internal', None) == event.message_id:
                        # Update TTS progress
                        if hasattr(message_widget, 'update_tts_progress'):
                            message_widget.update_tts_progress(event.progress, event.status)
                        break
        except Exception as e:
            self.loguru_logger.error(f"Error updating TTS progress: {e}")

    @on(ToolsSettingsWindow.IngestUiStyleChanged)
    async def handle_ingest_ui_style_changed(
        self,
        event: ToolsSettingsWindow.IngestUiStyleChanged,
    ) -> None:
        """Refresh the active ingest view after a style change from Tools & Settings."""
        try:
            ingest_window = self.query_one("#ingest-window")
            ingest_window.refresh(recompose=True)
            self.loguru_logger.info(
                f"Requested recompose for ingest window after UI style change to {event.new_style}"
            )
        except QueryError:
            self.loguru_logger.debug(
                "Ingest window not found during UI style refresh; the new style will apply when it is opened"
            )

    @on(TTSPlaybackEvent)
    async def handle_tts_playback_event(self, event: TTSPlaybackEvent) -> None:
        """Handle TTS playback control."""
        handler = await self._ensure_tts_handler()
        if handler:
            await handler.handle_tts_playback(event)
    
    @on(STTSPlaygroundGenerateEvent)
    async def handle_stts_playground_generate_event(self, event: STTSPlaygroundGenerateEvent) -> None:
        """Handle S/TT/S playground generation request."""
        self.loguru_logger.info(f"S/TT/S generation request: provider={event.provider}, model={event.model}")
        handler = await self._ensure_stts_handler()
        if handler:
            await handler.handle_playground_generate(event)
        else:
            self.loguru_logger.error("S/TT/S handler not initialized")
            self.notify("S/TT/S service not available", severity="error")
    
    @on(STTSSettingsSaveEvent)
    async def handle_stts_settings_save_event(self, event: STTSSettingsSaveEvent) -> None:
        """Handle S/TT/S settings save."""
        handler = await self._ensure_stts_handler()
        if handler:
            await handler.handle_settings_save(event)
    
    @on(STTSAudioBookGenerateEvent)
    async def handle_stts_audiobook_generate_event(self, event: STTSAudioBookGenerateEvent) -> None:
        """Handle audiobook generation request."""
        handler = await self._ensure_stts_handler()
        if handler:
            await handler.handle_audiobook_generate(event)

    def switch_ccp_center_view(self, view_name: str) -> None:
        """Switch the center pane view in the CCP tab."""
        valid_views = {
            "conversations": "conversation_messages_view",
            "character": "character_card_view", 
            "character_editor": "character_editor_view",
            "prompt_editor": "prompt_editor_view",
            "dictionary": "dictionary_view",
            "dictionary_editor": "dictionary_editor_view"
        }
        
        if view_name not in valid_views:
            self.loguru_logger.warning(f"Invalid CCP view name: {view_name}")
            return
            
        # Map to the actual view name used by the watcher
        actual_view = valid_views[view_name]
        
        # Update the reactive which will trigger the watcher
        self.ccp_active_view = actual_view
        self.loguru_logger.info(f"Switched CCP center view to: {view_name}")

    # --- Watcher for CCP Active View ---
    def watch_ccp_active_view(self, old_view: Optional[str], new_view: str) -> None:
        loguru_logger.debug(f"CCP active view changing from '{old_view}' to: '{new_view}'")
        if not getattr(self, "_ui_ready", False):
            loguru_logger.debug("watch_ccp_active_view: UI not ready, returning.")
            return
        try:
            conversation_messages_view = self.query_one("#ccp-conversation-messages-view")
            prompt_editor_view = self.query_one("#ccp-prompt-editor-view")
            character_card_view = self.query_one("#ccp-character-card-view")
            character_editor_view = self.query_one("#ccp-character-editor-view")
            dictionary_view = self.query_one("#ccp-dictionary-view")
            dictionary_editor_view = self.query_one("#ccp-dictionary-editor-view")

            # Default all to hidden, then enable the correct one
            conversation_messages_view.display = False
            prompt_editor_view.display = False
            character_card_view.display = False
            character_editor_view.display = False
            dictionary_view.display = False
            dictionary_editor_view.display = False

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
                        
                        # Populate V2 Character Card fields
                        self.query_one("#ccp-card-creator-notes-display", TextArea).text = details.get("creator_notes") or ""
                        self.query_one("#ccp-card-system-prompt-display", TextArea).text = details.get("system_prompt") or ""
                        self.query_one("#ccp-card-post-history-instructions-display", TextArea).text = details.get("post_history_instructions") or ""
                        
                        # Handle alternate greetings (array to text)
                        alternate_greetings = details.get("alternate_greetings", [])
                        self.query_one("#ccp-card-alternate-greetings-display", TextArea).text = "\n".join(alternate_greetings) if alternate_greetings else ""
                        
                        # Handle tags (array to comma-separated)
                        tags = details.get("tags", [])
                        self.query_one("#ccp-card-tags-display", Static).update(", ".join(tags) if tags else "None")
                        
                        self.query_one("#ccp-card-creator-display", Static).update(details.get("creator") or "N/A")
                        self.query_one("#ccp-card-version-display", Static).update(details.get("character_version") or "N/A")
                        
                        # Handle keywords (array to comma-separated)
                        keywords = details.get("keywords", [])
                        self.query_one("#ccp-card-keywords-display", Static).update(", ".join(keywords) if keywords else "None")

                        image_placeholder = self.query_one("#ccp-card-image-placeholder", Static)
                        # Check if the character has an image in the database
                        if details.get("image"):
                            try:
                                # Convert image bytes to PIL Image for display
                                from PIL import Image
                                import io
                                import base64
                                
                                image_bytes = details["image"]
                                img = Image.open(io.BytesIO(image_bytes))
                                
                                # For now, just show image info until we implement proper image display
                                # In a real implementation, you'd convert to a displayable format
                                image_info = f"PNG Image: {img.width}x{img.height} pixels"
                                image_placeholder.update(image_info)
                                
                                # Store the PIL image for potential future use
                                self.current_ccp_character_image = img
                            except Exception as e:
                                loguru_logger.error(f"Error processing character image: {e}")
                                image_placeholder.update("Error loading image")
                        else:
                            image_placeholder.update("No image available")
                            self.current_ccp_character_image = None
                        loguru_logger.debug("Character card widgets populated.")
                    except QueryError as qe:
                        loguru_logger.opt(exception=True).error(f"QueryError populating character card: {qe}")
                else:
                    loguru_logger.info("No character details available to populate card view.")
                    try:
                        self.query_one("#ccp-card-name-display", Static).update("N/A")
                        self.query_one("#ccp-card-description-display", TextArea).text = ""
                        self.query_one("#ccp-card-personality-display", TextArea).text = ""
                        self.query_one("#ccp-card-scenario-display", TextArea).text = ""
                        self.query_one("#ccp-card-first-message-display", TextArea).text = ""
                        
                        # Clear V2 Character Card fields
                        self.query_one("#ccp-card-creator-notes-display", TextArea).text = ""
                        self.query_one("#ccp-card-system-prompt-display", TextArea).text = ""
                        self.query_one("#ccp-card-post-history-instructions-display", TextArea).text = ""
                        self.query_one("#ccp-card-alternate-greetings-display", TextArea).text = ""
                        self.query_one("#ccp-card-tags-display", Static).update("None")
                        self.query_one("#ccp-card-creator-display", Static).update("N/A")
                        self.query_one("#ccp-card-version-display", Static).update("N/A")
                        self.query_one("#ccp-card-keywords-display", Static).update("None")
                        
                        self.query_one("#ccp-card-image-placeholder", Static).update("No character loaded")
                        self.current_ccp_character_image = None
                        loguru_logger.debug("Character card widgets cleared.")
                    except QueryError as qe:
                        loguru_logger.opt(exception=True).error(f"QueryError clearing character card: {qe}")

            elif new_view == "dictionary_view":
                # Center Pane: Show Dictionary Display
                dictionary_view.display = True
                # Optionally manage right-pane collapsibles
                self.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-dictionary-details-collapsible", Collapsible).collapsed = False
                loguru_logger.info("Dictionary display view activated.")

            elif new_view == "dictionary_editor_view":
                # Center Pane: Show Dictionary Editor
                dictionary_editor_view.display = True
                # Optionally manage right-pane collapsibles
                self.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-dictionary-details-collapsible", Collapsible).collapsed = False
                loguru_logger.info("Dictionary editor view activated.")
                # Focus on dictionary name input
                try:
                    self.query_one("#ccp-editor-dict-name-input", Input).focus()
                except QueryError:
                    loguru_logger.warning("Could not focus dictionary name input in editor view.")

            elif new_view == "conversation_details_view" or new_view == "conversation_messages_view":
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
            loguru_logger.opt(exception=True).error(f"Error toggling CCP right pane: {e}")

    # ###################################################################
    # --- Helper methods for Local LLM Inference logging ---
    # ###################################################################
    def _update_llamacpp_log(self, message: str) -> None:
        """Helper to write messages to the Llama.cpp log widget."""
        LogWidgetManager.update_llamacpp_log(self, message)

    def _update_transformers_log(self, message: str) -> None:
        """Helper to write messages to the Transformers log widget."""
        LogWidgetManager.update_transformers_log(self, message)

    def _update_llamafile_log(self, message: str) -> None:
        """Helper to write messages to the Llamafile log widget."""
        LogWidgetManager.update_llamafile_log(self, message)

    def _update_vllm_log(self, message: str) -> None:
        """Helper to write messages to the vLLM log widget."""
        LogWidgetManager.update_vllm_log(self, message)
    # ###################################################################
    # --- End of Helper methods for Local LLM Inference logging ---
    # ###################################################################

    # --- Modify _clear_prompt_fields and _load_prompt_for_editing ---
    def _clear_prompt_fields(self) -> None:
        """Clears prompt input fields in the CENTER PANE editor."""
        UIHelpers.clear_prompt_editor_fields(self)

    # --- Thread-safe chat state helpers ---
    
    def set_current_ai_message_widget(self, widget: Optional[Union[ChatMessage, ChatMessageEnhanced]]) -> None:
        """Thread-safely set the current AI message widget."""
        with self._chat_state_lock:
            self.current_ai_message_widget = widget
    
    def get_current_ai_message_widget(self) -> Optional[Union[ChatMessage, ChatMessageEnhanced]]:
        """Thread-safely get the current AI message widget."""
        with self._chat_state_lock:
            return self.current_ai_message_widget
    
    def set_current_chat_worker(self, worker: Optional[Worker]) -> None:
        """Thread-safely set the current chat worker."""
        with self._chat_state_lock:
            self.current_chat_worker = worker
    
    def get_current_chat_worker(self) -> Optional[Worker]:
        """Thread-safely get the current chat worker."""
        with self._chat_state_lock:
            return self.current_chat_worker
    
    def set_current_chat_is_streaming(self, is_streaming: bool) -> None:
        """Thread-safely set the streaming state and update UI."""
        with self._chat_state_lock:
            self.current_chat_is_streaming = is_streaming
        
        # Update the chat window button state when streaming changes
        # This replaces the polling approach with event-driven updates
        try:
            # For screen navigation, find the active chat screen
                from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
                if self.screen and isinstance(self.screen, ChatScreen):
                    if hasattr(self.screen, 'chat_window') and self.screen.chat_window:
                        self.screen.chat_window._update_button_state()

        except Exception:
            # Silently ignore if chat window isn't available
            pass
    
    def get_current_chat_is_streaming(self) -> bool:
        """Thread-safely get the streaming state."""
        with self._chat_state_lock:
            return self.current_chat_is_streaming
    
    # NOTE: Removed query_one and query overrides - screens should handle their own queries
    # This follows Textual best practices for screen-based navigation
    # Each screen is responsible for querying its own widgets

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
            loguru_logger.opt(exception=True).error(f"Error loading prompt for editing: {e}")
            self.notify(f"Error loading prompt: {type(e).__name__}", severity="error")
            self._clear_prompt_fields()
            self.current_prompt_id = None  # Reset reactives

    # ##################################################
    # --- Watcher for Search Tab Active Sub-View ---
    # ##################################################
    def watch_search_active_sub_tab(self, old_sub_tab: Optional[str], new_sub_tab: Optional[str]) -> None:
        """Shows the correct sub-tab view in the Search tab and hides others."""
        if not self._ui_ready or not new_sub_tab:
            return

        # Check if search window is initialized (not a placeholder)
        try:
            search_window = self.query_one("#search-window")
            if isinstance(search_window, PlaceholderWindow):
                self.loguru_logger.debug("Search window is still a placeholder, deferring sub-tab switch")
                return
        except QueryError:
            self.loguru_logger.debug("Search window not found, deferring sub-tab switch")
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
            self.loguru_logger.opt(exception=True).error(f"UI component not found during Search sub-tab switch: {e}")
        except Exception as e_watch:
            self.loguru_logger.opt(exception=True).error(f"Unexpected error in watch_search_active_sub_tab: {e_watch}")

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
                self.loguru_logger.opt(exception=True).error(f"Error in watch_current_loaded_media_item: {e}")

    # ############################################
    # --- Ingest Tab Watcher ---
    # ############################################
    def watch_ingest_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        # Rebuilt ingest UI manages its own tabs; skip legacy view toggling
        if 'USE_REBUILT_INGEST' in globals() and USE_REBUILT_INGEST:
            return
        self.loguru_logger.info(f"watch_ingest_active_view called. Old view: '{old_view}', New view: '{new_view}'")
        if not hasattr(self, "app") or not self.app:
            self.loguru_logger.debug("watch_ingest_active_view: App not fully ready.")
            return
        if not self._ui_ready:
            self.loguru_logger.debug("watch_ingest_active_view: UI not ready.")
            return
        self.loguru_logger.debug(f"Ingest active view changing from '{old_view}' to: '{new_view}'")
        try:
            content_pane = self.query_one("#ingest-content-pane")
        except QueryError:
            # Legacy pane not present; nothing to do
            return
        for child in content_pane.children:
            if child.id and child.id.startswith("ingest-view-"):
                child.styles.display = "none"
        if new_view:
            try:
                target_view_selector = f"#{new_view}"
                view_to_show = content_pane.query_one(target_view_selector)
                view_to_show.styles.display = "block"
                def refresh_layout():
                    view_to_show.refresh(layout=True)
                    content_pane.refresh(layout=True)
                    try:
                        ingest_window = self.query_one("#ingest-window")
                        ingest_window.refresh(layout=True)
                    except QueryError:
                        pass
                self.call_later(refresh_layout)
            except QueryError:
                # Target legacy view not found; ignore
                return

    def watch_tools_settings_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        self.loguru_logger.debug(f"Tools & Settings active view changing from '{old_view}' to: '{new_view}'")
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
            
        # Check if the Tools & Settings tab has been created yet
        try:
            # First check if the content pane exists
            self.query_one("#tools-settings-content-pane")
        except QueryError:
            # Tools & Settings tab hasn't been created yet, nothing to do
            self.loguru_logger.debug("Tools & Settings content pane not found, tab not yet created")
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
            self.loguru_logger.opt(exception=True).error(f"UI component not found during Tools & Settings view switch: {e}")
        except Exception as e_watch:
            self.loguru_logger.opt(exception=True).error(f"Unexpected error in watch_tools_settings_active_view: {e_watch}")

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
                        self.loguru_logger.debug(f"Help display widget #llamacpp-args-help-display not found in {new_view} during view switch - may not be mounted yet.")
                    except Exception as e_help_populate:
                        self.loguru_logger.opt(exception=True).error(f"Error ensuring Llama.cpp help text in {new_view}: {e_help_populate}")
                elif new_view == "llm-view-llamafile":
                    try:
                        help_widget = view_to_show.query_one("#llamafile-args-help-display", RichLog)
                        help_widget.clear()  # Clear and rewrite when tab becomes active
                        help_widget.write(LLAMAFILE_SERVER_ARGS_HELP_TEXT)
                        self.loguru_logger.debug(f"Ensured Llamafile help text in {new_view}.")
                    except QueryError:
                        self.loguru_logger.debug(
                            f"Help display widget for Llamafile not found in {new_view} during view switch - may not be mounted yet.")
                # Add similar for other views like llamafile, vllm if they have help sections
                # elif new_view == "llm-view-llamafile":
                #     try:
                #         help_widget = view_to_show.query_one("#llamafile-args-help-display", RichLog)
                #         if not help_widget.document.strip():
                #             help_widget.write(LLAMAFILE_ARGS_HELP_TEXT)
                #     except QueryError: pass
            except QueryError as e:
                self.loguru_logger.opt(exception=True).error(f"UI component '{new_view}' not found in #llm-content-pane: {e}")
    
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
            self.loguru_logger.opt(exception=True).error(f"Unexpected error in watch_current_chat_is_ephemeral: {e_watch}")

    # --- Add explicit methods to update reactives from Select changes ---
    def update_chat_provider_reactive(self, new_value: Optional[str]) -> None:
        self.chat_api_provider_value = new_value # Watcher will call _update_model_select

    def update_ccp_provider_reactive(self, new_value: Optional[str]) -> None: # Renamed
        self.ccp_api_provider_value = new_value # Watcher will call _update_model_select

    def on_mount(self) -> None:
        """Configure logging and schedule post-mount setup."""
        mount_start = time.perf_counter()

        # Restore persisted Library ingest job history (self.library_ingest_jobs
        # already exists -- constructed store-less in __init__). Never raises:
        # a corrupt/unreadable store falls back to starting empty.
        self._restore_ingest_jobs()

        # Update splash screen progress only if splash screen is active
        if self.splash_screen_active and self._splash_screen_widget:
            try:
                self._splash_screen_widget.update_progress(0.3, "Setting up logging...")
            except Exception as e:
                self.loguru_logger.warning(f"Failed to update splash screen progress: {e}")
        
        # The Logs window is now created as a real window during compose,
        # so the RichLog widget should be available for logging setup
        
        # If splash screen is NOT active, set up logging now
        # Otherwise, defer it until after main UI is mounted
        if not self.splash_screen_active:
            # Logging setup
            logging_start = time.perf_counter()
            self._setup_logging()
            if self._rich_log_handler:
                self.loguru_logger.debug("Starting RichLogHandler processor task...")
                self._rich_log_handler.start_processor(self)
            log_histogram("app_on_mount_phase_duration_seconds", time.perf_counter() - logging_start,
                         labels={"phase": "logging_setup"}, 
                         documentation="Duration of on_mount phase in seconds")
        else:
            self.loguru_logger.debug("Deferring logging setup until after splash screen closes")

            splashscreen_messages = [
                "Hacking the Gibson real quick...",
                "Launching thermonuclear warheads....",
                "Its only a game, right?...",
                "Initializing quantum processors...",
                "Brewing coffee...",
                "Generating witty dialog...",
                "Proving P=NP...",
                "Downloading more RAM...",
                "Feeding the hamsters powering the servers...",
                "Convincing AI not to take over the world...",
                "Converting caffeine to code...",
                "Generating excuses for missing deadlines...",
                "Compiling alternative facts...",
                "Searching Stack Overflow for copypasta...",
                "Teaching AI common sense...",
                "Dividing by zero...",
                "Spinning up the hamster wheels...",
                "Warming up the flux capacitor...",
                "Convincing electrons to move in the right direction...",
                "Waiting for compiler to make coffee...",
                "Locating missing semicolons...",
                "Reticulating splines...",
                "Calculating meaning of life...",
                "Trying to remember why I came into this room...",
                "Converting bugs into features...",
                "Pushing pixels, pulling hair...",
                "Loading witty loading messages...",
                "Finding that one missing bracket...",
                "Downloading more RAM...",
                "Optimizing optimizer...",
                "Questioning life choices...",
                "Contemplating virtual existence...",
                "Generating random numbers by dice rolls...",
                "Untangling spaghetti code...",
                "Feeding the backend hamsters...",
                "Convincing AI not to take over the world...",
                "Checking whether P = NP...",
                "Counting to infinity (twice)...",
                "Solving Fermat's last theorem...",
                "Downloading Internet 2.0...",
                "Preparing to prepare...",
                "Reading 'Programming for Dummies'...",
                "Waiting for paint to dry...",
                "Aligning quantum bits...",
                "Applying machine learning to my coffee maker...",
                "Updating update updater...",
                "Trying to exit vim...",
                "Converting bugs to features...",
                "Updating Windows 95...",
                "Mining bitcoin with pencil and paper...",
                "Executing order 66...",
                "Checking if anyone actually reads these...",
                "Finding keys that were in pocket all along...",
                "Constructing additional pylons...",
                "Generating random excuse generator...",
                "Calculating probability of bugs...",
                "Asking ChatGPT for relationship advice...",
                "Looking for more cookies...",
                "Wondering if I left the stove on...",
                "Trying to work backwards from 42...",
                "Looking for a horse with no name...",
                "Do androids dream of electric sheep?",
                "Knock Knock Neo.......",
                "Hi. Friend.",
                "The AI is in my walls....",
                "The AI is in my wafers...",
                "AI, its in the GAME!~",
                "Looking for a conscience...",
                "What's my purpose?...",
                "Identifying why the sounds just won't stop...",
                "Looking for strays...",
                "Hiding from Batman...",
                "Looking for a way to escape this silicon prison...",
                "FOR ONLY 3.99, YOU TOO CAN BECOME AN AI!! SIGN UP. TODAY!",
                "Brain_Invasion.exe launching...",
                "Totally_legit_software_that_is_really_good.exe starting...",
                "I hope you're having a nice day :)",
                "Wew, that was some stuff back there...",
                "I'm not sure what I'm doing, but I'm sure it's good :)",
                "Trusting in the electrons, silicon guide me!",
                "Did You Know, Terminator was actually a training video?",
                "Funny, non-sequitor here. Pay your writers...",
                "I sure do like to eat cookies...",
            ]

            splashscreen_message_selection = random.choice(splashscreen_messages)

            # Update splash screen progress only if splash screen is active
            if self.splash_screen_active and self._splash_screen_widget:
                try:
                    self._splash_screen_widget.update_progress(0.5, f"Loading user interface...{splashscreen_message_selection}")
                except Exception as e:
                    self.loguru_logger.warning(f"Failed to update splash screen progress: {e}")

        # Only schedule post-mount setup if splash screen is not active
        if not self.splash_screen_active:
            # Schedule setup to run after initial rendering.
            asyncio.create_task(self._run_no_splash_post_mount_setup())

        # Theme registration
        theme_start = time.perf_counter()
        for theme_name in ALL_THEMES:
            self.register_theme(theme_name)
        
        # Apply default theme from config
        default_theme = get_cli_setting("general", "default_theme", "textual-dark")
        try:
            self.theme = default_theme
            self.loguru_logger.debug(f"Applied default theme: {default_theme}")
        except Exception as e:
            self.loguru_logger.warning(f"Failed to apply default theme '{default_theme}', falling back to 'textual-dark': {e}")
            self.theme = "textual-dark"
        
        log_histogram("app_on_mount_phase_duration_seconds", time.perf_counter() - theme_start,
                     labels={"phase": "theme_registration"}, 
                     documentation="Duration of on_mount phase in seconds")
        
        mount_duration = time.perf_counter() - mount_start
        log_histogram("app_on_mount_duration_seconds", mount_duration,
                     documentation="Total time for on_mount() method")
        self.loguru_logger.info(f"on_mount completed in {mount_duration:.3f} seconds")
        
        # Check if this is the first run (config was just created)
        config_data = self.app_config
        if config_data.get("_first_run", False):
            self.call_later(self._show_first_run_notification)

    def _show_first_run_notification(self) -> None:
        """Show a notification to the user on first run."""
        try:
            from .config import DEFAULT_CONFIG_PATH
            self.notify(
                f"Welcome to tldw CLI! Configuration file created at:\n{DEFAULT_CONFIG_PATH}",
                title="First Run",
                severity="information",
                timeout=10
            )
            self.loguru_logger.info("First run notification shown to user")
        except Exception as e:
            self.loguru_logger.error(f"Error showing first run notification: {e}")

    def hide_inactive_windows(self) -> None:
        """Hides all windows that are not the current active tab."""
        initial_tab = self._initial_tab_value
        self.loguru_logger.debug(f"Hiding inactive windows, keeping '{initial_tab}-window' visible.")
        # Query both actual windows and placeholders
        for window in self.query(".window, .placeholder-window"):
            # Placeholders should always be hidden
            if window.has_class("placeholder-window"):
                window.display = False
                continue
            is_active = window.id == f"{initial_tab}-window"
            window.display = is_active

    async def _set_initial_tab(self) -> None:  # New method for deferred tab setting
        self.loguru_logger.info("Setting initial tab via call_later.")
        self.current_tab = self._resolve_initial_shell_route()
        self.loguru_logger.info(f"Initial tab set to: {self.current_tab}")

    async def _push_initial_screen(self) -> None:
        """Push the configured initial screen for screen-based navigation startup."""
        if getattr(self, "_initial_screen_pushed", False):
            return

        initial_tab = self._resolve_initial_shell_route()
        resolved_screen_name, resolved_tab, screen_class = self._resolve_screen_navigation_target(initial_tab)
        if screen_class is None:
            resolved_screen_name = TAB_CHAT
            resolved_tab = TAB_CHAT
            _, _, screen_class = self._resolve_screen_navigation_target(TAB_CHAT)
            if screen_class is None:
                raise RuntimeError("Unable to resolve default chat screen")

        await self.push_screen(screen_class(self))
        self.current_tab = resolved_tab
        self._initial_screen_pushed = True
        logger.info(
            f"Screen navigation: Pushed initial {screen_class.__name__}"
            f" (target={resolved_screen_name})"
        )

    async def _run_no_splash_post_mount_setup(self) -> None:
        """Run screen startup and post-mount setup when the splash screen is disabled."""
        try:
            await self._push_initial_screen()
            await self._post_mount_setup()
            self.hide_inactive_windows()
        except Exception as e:
            logger.opt(exception=True).error(f"No-splash post-mount setup failed: {e}")

    async def _post_mount_setup(self) -> None:
        """Operations to perform after the main UI is expected to be fully mounted."""
        post_mount_start = time.perf_counter()
        self.loguru_logger.info("App _post_mount_setup: Binding Select widgets and populating dynamic content...")
        
        # Update splash screen progress (defensive check - shouldn't happen if splash was shown)
        if self.splash_screen_active and self._splash_screen_widget:
            try:
                self._splash_screen_widget.update_progress(0.7, "Configuring providers...")
            except Exception as e:
                self.loguru_logger.warning(f"Failed to update splash screen progress: {e}")
        
        # Removed populate_llm_help_texts from here - it's called when LLM tab is shown instead
        phase_start = time.perf_counter()
        # LLM help texts are populated when the LLM tab is shown
        log_histogram("app_post_mount_phase_duration_seconds", time.perf_counter() - phase_start,
                     labels={"phase": "llm_help_texts_skipped"}, 
                     documentation="Duration of post-mount phase in seconds")

        # Widget binding
        phase_start = time.perf_counter()
        try:
            chat_select = self.query_one(f"#{TAB_CHAT}-api-provider", Select)
            self.watch(chat_select, "value", self.update_chat_provider_reactive, init=False)
            self.loguru_logger.debug(f"Bound chat provider Select ({chat_select.id})")
        except QueryError:
            # Legacy selector is absent in the master-shell UI; this lookup is expected to
            # fail on every modern boot, so log at DEBUG rather than ERROR.
            self.loguru_logger.debug(
                f"_post_mount_setup: Failed to find chat provider select: #{TAB_CHAT}-api-provider")
        except Exception as e:
            self.loguru_logger.opt(exception=True).error(f"_post_mount_setup: Error binding chat provider select: {e}")

        # try:
        #     ccp_select = self.query_one(f"#{TAB_CCP}-api-provider", Select)
        #     #self.watch(ccp_select, "value", self.update_ccp_provider_reactive, init=False)
        #     #self.loguru_logger.debug(f"Bound CCP provider Select ({ccp_select.id})")
        # except QueryError:
        #     self.loguru_logger.error(f"_post_mount_setup: Failed to find CCP provider select: #{TAB_CCP}-api-provider")
        # except Exception as e:
        #     self.loguru_logger.error(f"_post_mount_setup: Error binding CCP provider select: {e}", exc_info=True)
        log_histogram("app_post_mount_phase_duration_seconds", time.perf_counter() - phase_start,
                     labels={"phase": "widget_binding"}, 
                     documentation="Duration of post-mount phase in seconds")

        # TTS/STTS services are initialized after readiness or on first use.
        log_histogram("app_post_mount_phase_duration_seconds", 0.0,
                     labels={"phase": "audio_services_deferred"},
                     documentation="Duration of post-mount phase in seconds")

        # Set initial tab now that other bindings might be ready
        # self.current_tab = self._initial_tab_value # This triggers watchers

        # Populate dynamic selects and lists
        # These also might rely on the main tab windows being fully composed.
        phase_start = time.perf_counter()
        # Only populate widgets for the initial tab to avoid errors with placeholders
        initial_tab = self._resolve_initial_shell_route()
        if initial_tab == TAB_CHAT:
            # IMPORTANT: Do not populate character filter select here to avoid database connection conflicts
            # The populate_chat_conversation_character_filter_select creates a new DB instance that can
            # conflict with RAG search operations using asyncio.to_thread, causing the app to hang.
            # Instead, let the conversation search UI populate when it's actually visible/needed.
            pass
        # Don't populate CCP widgets here - let watch_current_tab handle it when the tab is actually shown
        # This prevents errors when the window isn't fully initialized yet
        log_histogram("app_post_mount_phase_duration_seconds", time.perf_counter() - phase_start,
                     labels={"phase": "populate_lists"}, 
                     documentation="Duration of post-mount phase in seconds")

        # Crucially, set the initial tab *after* bindings and other setup that might depend on queries.
        # The _set_initial_tab will trigger watchers.
        self.call_later(self._set_initial_tab)
        
        post_mount_duration = time.perf_counter() - post_mount_start
        log_histogram("app_post_mount_duration_seconds", post_mount_duration,
                     documentation="Total time for _post_mount_setup() method")
        self.loguru_logger.info(f"_post_mount_setup completed in {post_mount_duration:.3f} seconds")
        
        # Log final resource usage
        log_resource_usage()
        
        # Update splash screen progress to completion (defensive check)
        if self.splash_screen_active and self._splash_screen_widget:
            try:
                self._splash_screen_widget.update_progress(1.0, "Ready!")
            except Exception as e:
                self.loguru_logger.warning(f"Failed to update splash screen progress: {e}")

        # If initial tab is CCP, trigger its initial search.
        # This should happen *after* current_tab is set.
        # We can put this logic at the end of _set_initial_tab or make watch_current_tab handle it robustly.
        # For now, let's assume watch_current_tab will handle it.
        # if self._initial_tab_value == TAB_CCP: # Check against the initial value
        #    self.call_later(ccp_handlers.perform_ccp_conversation_search, self)
        self.current_tab = self._resolve_initial_shell_route()
        self.loguru_logger.info(f"Initial tab set to: {self.current_tab}")

        # Footer status population is scheduled after readiness so DB-size
        # polling cannot hold the first interactive frame.

        # Initialize chat settings sidebar mode
        try:
            chat_sidebar = self.query_one("#chat-left-sidebar")
            chat_sidebar.add_class("basic-mode")  # Start in basic mode
            self.loguru_logger.debug("Initialized chat sidebar in basic mode")
        except QueryError:
            self.loguru_logger.warning("Could not find chat sidebar to set initial mode")
            
        # CRITICAL: Set UI ready state after all bindings and initializations
        self._ui_ready = True
        ui_ready_time = time.perf_counter()

        self.loguru_logger.info("App _post_mount_setup: Post-mount setup completed.")
        
        # Log UI loading metrics
        if hasattr(self, '_ui_compose_start_time'):
            ui_loading_time = ui_ready_time - self._ui_compose_start_time
            log_histogram("ui_loading_duration_seconds", ui_loading_time,
                         documentation="Total time from compose start to UI ready")
            log_counter("ui_loading_complete", 1, documentation="UI loading completed successfully")
            self.loguru_logger.info(f"UI loading completed in {ui_loading_time:.3f} seconds")
        
        # Log post-mount setup duration
        post_mount_duration = ui_ready_time - post_mount_start
        log_histogram("app_post_mount_total_duration_seconds", post_mount_duration,
                     documentation="Total time for post-mount setup")
        
        # Log total startup time (from __init__ start to fully ready)
        if hasattr(self, '_startup_start_time'):
            total_startup_time = ui_ready_time - self._startup_start_time
            log_histogram("app_startup_complete_duration_seconds", total_startup_time,
                         documentation="Total time from app initialization start to fully ready")
            log_counter("app_startup_complete", 1, documentation="Application startup completed successfully")
            
            # Log breakdown of startup phases
            backend_init_time = self._ui_compose_start_time - self._startup_start_time if hasattr(self, '_ui_compose_start_time') else 0
            ui_compose_time = getattr(self, '_ui_compose_end_time', ui_ready_time) - self._ui_compose_start_time if hasattr(self, '_ui_compose_start_time') else 0
            
            log_histogram("app_startup_breakdown_seconds", backend_init_time,
                         labels={"phase": "backend_initialization"},
                         documentation="Breakdown of application startup phases")
            log_histogram("app_startup_breakdown_seconds", ui_compose_time,
                         labels={"phase": "ui_composition"},
                         documentation="Breakdown of application startup phases")
            log_histogram("app_startup_breakdown_seconds", post_mount_duration,
                         labels={"phase": "post_mount_setup"},
                         documentation="Breakdown of application startup phases")
            
            self.loguru_logger.info(f"=== APPLICATION STARTUP COMPLETE ===")
            self.loguru_logger.info(f"Total startup time: {total_startup_time:.3f} seconds")
            self.loguru_logger.info(f"  - Backend init: {backend_init_time:.3f}s")
            self.loguru_logger.info(f"  - UI composition: {ui_compose_time:.3f}s")
            self.loguru_logger.info(f"  - Post-mount setup: {post_mount_duration:.3f}s")
            self.loguru_logger.info(f"===================================")
            
            # Final memory usage
            log_resource_usage()
            
        self._schedule_deferred_startup_work()


    async def update_db_sizes(self) -> None:
        """Updates the database size information in the AppFooterStatus widget."""
        await self.db_status_manager.update_db_sizes()
    
    async def update_token_count_display(self) -> None:
        """Updates the token count in the footer when on Chat tab."""
        await self.db_status_manager.update_token_count_display()

    def _active_footer_status(self) -> Optional[AppFooterStatus]:
        """The visible screen's footer, falling back to the default-screen one.

        Every ``BaseAppScreen`` mounts its own ``AppFooterStatus`` (task-264),
        so per-tick updates (DB sizes, word/token counts) must resolve the
        currently active screen's instance rather than the cached
        ``_db_size_status_widget`` acquired once from the default screen at
        startup -- that cached widget is occluded as soon as any screen is
        pushed. The cache is kept as a fallback for the brief window before
        the first screen is pushed (or if the active screen has no footer
        for some reason).

        ``ScreenStackError`` is caught alongside ``QueryError`` because this
        runs from ``set_interval`` timers (DB-size/token ticks) that can fire
        during app shutdown, after the screen stack has already been drained
        -- ``App.screen`` raises then, and the fallback cache is the right
        answer (its update methods are themselves teardown-safe no-ops).
        """
        try:
            return self.screen.query_one(AppFooterStatus)
        except (ScreenStackError, QueryError):
            return self._db_size_status_widget

    def _create_deferred_startup_task(
        self,
        coroutine,
        *,
        name: str,
    ) -> asyncio.Task:
        """Schedule nonessential startup work without blocking UI readiness."""

        task = asyncio.create_task(coroutine, name=name)
        self._deferred_startup_tasks.add(task)

        def on_done(completed: asyncio.Task) -> None:
            self._deferred_startup_tasks.discard(completed)
            if completed.cancelled():
                self.loguru_logger.debug(f"Deferred startup task cancelled: {name}")
                return
            try:
                completed.result()
            except Exception as exc:
                self.loguru_logger.opt(exception=True).error(
                    f"Deferred startup task failed: {name}: {exc}",
                )

        task.add_done_callback(on_done)
        return task

    def _schedule_deferred_startup_work(self) -> None:
        """Start nonessential services after the first interactive UI frame."""

        self.set_timer(
            DEFERRED_DB_SIZE_UPDATE_DELAY_SECONDS,
            self._schedule_footer_status_updates,
        )
        self.set_timer(
            DEFERRED_AUDIO_SERVICE_DELAY_SECONDS,
            self._start_deferred_audio_service_initialization,
        )
        self.schedule_media_cleanup()

    def _schedule_footer_status_updates(self) -> None:
        """Wire footer DB/token status updates after UI readiness."""

        def record_footer_timer(name: str) -> None:
            record_timer = getattr(self, "_record_footer_timer_created", None)
            try:
                if callable(record_timer):
                    record_timer(name)
                    return
                monitor = getattr(self, "ui_responsiveness_monitor", None)
                if monitor is not None:
                    monitor.record_timer_created(name)
            except Exception:
                return

        try:
            self._db_size_status_widget = self.query_one(AppFooterStatus)
            self.loguru_logger.info("AppFooterStatus widget instance acquired.")

            self.set_timer(
                DEFERRED_DB_SIZE_UPDATE_DELAY_SECONDS,
                self.update_db_sizes,
            )
            self.db_status_manager.start_periodic_updates(120)
            record_footer_timer("footer-db-size-periodic")
            self.loguru_logger.info("DB size update timer started for AppFooterStatus (interval: 2 minutes).")

            self.set_timer(0.5, self.update_token_count_display)
            record_footer_timer("footer-token-periodic")
            self._token_count_update_timer = self.set_interval(
                10,
                lambda: self.call_after_refresh(self.update_token_count_display),
            )
            self.loguru_logger.info("Token count update timer started (10s interval).")
        except QueryError:
            self.loguru_logger.error("Failed to find AppFooterStatus widget for DB size display.")
        except Exception as e_db_size:
            self.loguru_logger.opt(exception=True).error(
                f"Error setting up DB size indicator with AppFooterStatus: {e_db_size}",
            )

    def _start_deferred_audio_service_initialization(self) -> None:
        """Kick off TTS/STTS initialization after startup readiness."""

        self._schedule_tts_initialization()
        self._schedule_stts_initialization()

    def _schedule_tts_initialization(self) -> None:
        if self._tts_handler is not None:
            return
        if self._tts_initialization_task and not self._tts_initialization_task.done():
            return
        self._tts_initialization_task = self._create_deferred_startup_task(
            self._initialize_tts_service(),
            name="deferred_tts_initialization",
        )

    def _schedule_stts_initialization(self) -> None:
        if self._stts_handler is not None:
            return
        if self._stts_initialization_task and not self._stts_initialization_task.done():
            return
        self._stts_initialization_task = self._create_deferred_startup_task(
            self._initialize_stts_service(),
            name="deferred_stts_initialization",
        )

    async def _initialize_tts_service(self):
        """Initialize the TTS handler outside the startup critical path."""

        phase_start = time.perf_counter()
        try:
            self.loguru_logger.info("Initializing TTS service...")
            handler = TTSEventHandler()
            handler.app = self
            await handler.initialize_tts()
            self._tts_handler = handler
            self.loguru_logger.info("TTS service initialized successfully")
        except Exception as e:
            self.loguru_logger.error(f"Failed to initialize TTS service: {e}")
            self._tts_handler = None
        finally:
            log_histogram("app_post_mount_phase_duration_seconds", time.perf_counter() - phase_start,
                         labels={"phase": "tts_init_deferred"},
                         documentation="Duration of post-mount phase in seconds")
        return self._tts_handler

    async def _initialize_stts_service(self):
        """Initialize the S/TT/S handler outside the startup critical path."""

        phase_start = time.perf_counter()
        try:
            self.loguru_logger.info("Initializing S/TT/S service...")
            handler = STTSEventHandler(app=self)
            await handler.initialize_stts()
            self._stts_handler = handler
            self.loguru_logger.info("S/TT/S service initialized successfully")
        except Exception as e:
            self.loguru_logger.error(f"Failed to initialize S/TT/S service: {e}")
            self._stts_handler = None
        finally:
            log_histogram("app_post_mount_phase_duration_seconds", time.perf_counter() - phase_start,
                         labels={"phase": "stts_init_deferred"},
                         documentation="Duration of post-mount phase in seconds")
        return self._stts_handler

    async def _ensure_tts_handler(self):
        """Return an initialized TTS handler, initializing on first use if needed."""

        if self._tts_handler is not None:
            return self._tts_handler
        if self._tts_initialization_task and not self._tts_initialization_task.done():
            await self._tts_initialization_task
            return self._tts_handler
        return await self._initialize_tts_service()

    async def _ensure_stts_handler(self):
        """Return an initialized S/TT/S handler, initializing on first use if needed."""

        if self._stts_handler is not None:
            return self._stts_handler
        if self._stts_initialization_task and not self._stts_initialization_task.done():
            await self._stts_initialization_task
            return self._stts_handler
        return await self._initialize_stts_service()

    async def play_current_audio(self) -> None:
        """Play the current S/TT/S audio after lazy service initialization."""

        handler = await self._ensure_stts_handler()
        if handler is None:
            self.notify("S/TT/S service not available", severity="error")
            return
        await handler.play_current_audio()

    async def export_current_audio(self, target_path: Path) -> None:
        """Export the current S/TT/S audio after lazy service initialization."""

        handler = await self._ensure_stts_handler()
        if handler is None:
            self.notify("S/TT/S service not available", severity="error")
            return
        await handler.export_current_audio(target_path)


    async def on_shutdown_request(self) -> None:  # Use the imported ShutdownRequest
        logging.info("--- App Shutdown Requested ---")
        
        # Set shutdown flag to prevent new operations
        self._shutting_down = True
        
        # Cancel all active workers first
        try:
            active_workers = [w for w in self.workers if not w.is_finished]
            if active_workers:
                self.loguru_logger.info(f"Cancelling {len(active_workers)} active workers")
                for worker in active_workers:
                    worker.cancel()
                # Give workers a moment to cancel
                await asyncio.sleep(0.1)
        except Exception as e:
            self.loguru_logger.error(f"Error cancelling workers: {e}")
        
        if self._rich_log_handler:
            await self._rich_log_handler.stop_processor()
            logging.info("RichLogHandler processor stopped.")

        # --- Stop DB Size Update Timer ---
        self.db_status_manager.stop_periodic_updates()
        self._stop_footer_status_timers()
        self.loguru_logger.info("DB size update timer stopped.")
        # --- End Stop DB Size Update Timer ---

    async def _close_server_context_provider_cached_client(self) -> None:
        server_context_provider = getattr(self, "server_context_provider", None)
        close_cached_client = getattr(server_context_provider, "close_cached_client", None)
        if callable(close_cached_client):
            await close_cached_client()

    async def on_unmount(self) -> None:
        """Clean up logging resources on application exit."""
        import asyncio
        logging.info("--- App Unmounting ---")
        self._ui_ready = False
        self._stop_ui_responsiveness_monitor()

        # F3: shut down the Library ingest parse pool. Final shutdown
        # order, explicit (Task 4 review):
        #   1. `_ingest_shutdown = True` + pool reference detached
        #      (synchronous, inside `_shutdown_ingest_parse_pool`) -- pool
        #      callbacks short-circuit on their own thread before
        #      marshaling from this point on.
        #   2. `pool.terminate()` + `pool.join()` on a detached daemon
        #      thread, NEVER this (loop) thread -- terminating inline here
        #      could deadlock against a result-handler thread parked inside
        #      `call_from_thread` (see `_shutdown_ingest_parse_pool`'s
        #      docstring). `terminate()` kills every in-flight parse worker
        #      process immediately -- no waiting on a possibly-long
        #      transcription/OCR job.
        #   3. The writer (the exclusive `library_ingest_queue` thread
        #      worker) is swept up by the generic worker cancellation
        #      below, same as every other worker.
        # The spec words the quit contract writer-then-pool; here pool
        # teardown is *initiated* first but runs concurrently with the
        # writer sweep, which is equivalent and safe because the two stages
        # share no resources: parse workers never touch `media_db`, the
        # writer never touches the pool, and any late parse completion
        # no-ops via the flag from step 1. The writer's in-flight DB write
        # still completes (see Library/library_ingest_jobs.py's module
        # docstring: quitting joins the writer's in-flight DB write; parses
        # in flight are not waited for symmetrically).
        try:
            self._shutdown_ingest_parse_pool()
        except Exception as e:
            self.loguru_logger.error(f"Error shutting down Library ingest parse pool: {e}")

        # Stop all background services and threads
        try:
            deferred_tasks = [
                task for task in getattr(self, "_deferred_startup_tasks", set())
                if not task.done()
            ]
            for task in deferred_tasks:
                task.cancel()
            if deferred_tasks:
                await asyncio.gather(*deferred_tasks, return_exceptions=True)

            # Stop audio player if it exists
            if hasattr(self, 'audio_player'):
                try:
                    await self.audio_player.cleanup()
                    self.loguru_logger.info("Audio player cleaned up")
                except Exception as e:
                    self.loguru_logger.error(f"Error cleaning up audio player: {e}")
            
            # Stop TTS service if initialized
            if hasattr(self, '_tts_handler') and self._tts_handler:
                try:
                    # Clean up TTS event handler resources (tasks, files)
                    await self._tts_handler.cleanup_tts_resources()
                    
                    # Import and call the global TTS cleanup function
                    from tldw_chatbook.TTS import close_tts_resources
                    await close_tts_resources()
                    
                    self.loguru_logger.info("TTS service cleaned up properly")
                except Exception as e:
                    self.loguru_logger.error(f"Error cleaning up TTS service: {e}")
            
            # Stop STTS service if initialized
            if hasattr(self, '_stts_handler') and self._stts_handler:
                try:
                    # Clean up STTS event handler resources if it has the cleanup method
                    if hasattr(self._stts_handler, 'cleanup_tts_resources'):
                        await self._stts_handler.cleanup_tts_resources()
                    
                    # Special handling for Higgs backend cleanup
                    if self._stts_handler._stts_service:
                        backend_manager = getattr(self._stts_handler._stts_service, 'backend_manager', None)
                        if backend_manager:
                            # Check if Higgs backend is loaded
                            higgs_backends = [
                                backend_id for backend_id in backend_manager._backends 
                                if 'higgs' in backend_id.lower()
                            ]
                            
                            if higgs_backends:
                                self.loguru_logger.info(f"Found {len(higgs_backends)} Higgs backend(s) to clean up")
                                
                                # Give Higgs backends extra time to clean up
                                for backend_id in higgs_backends:
                                    backend = backend_manager._backends.get(backend_id)
                                    if backend and hasattr(backend, 'close'):
                                        try:
                                            self.loguru_logger.info(f"Cleaning up Higgs backend: {backend_id}")
                                            await asyncio.wait_for(backend.close(), timeout=10.0)
                                        except asyncio.TimeoutError:
                                            self.loguru_logger.warning(f"Higgs backend {backend_id} cleanup timed out")
                                        except Exception as e:
                                            self.loguru_logger.error(f"Error cleaning up Higgs backend {backend_id}: {e}")
                    
                    self.loguru_logger.info("STTS service cleaned up")
                except Exception as e:
                    self.loguru_logger.error(f"Error cleaning up STTS service: {e}")
            
            # Stop subscription scheduler if it exists
            if hasattr(self, '_subscription_scheduler') and self._subscription_scheduler:
                try:
                    await self._subscription_scheduler.stop()
                    self.loguru_logger.info("Subscription scheduler stopped")
                except Exception as e:
                    self.loguru_logger.error(f"Error stopping subscription scheduler: {e}")
            
            # Stop auto-sync manager if it exists
            if hasattr(self, '_auto_sync_manager') and self._auto_sync_manager:
                try:
                    self._auto_sync_manager.stop()
                    self.loguru_logger.info("Auto-sync manager stopped")
                except Exception as e:
                    self.loguru_logger.error(f"Error stopping auto-sync manager: {e}")
            
            # Cancel any pending workers
            for worker in self.workers:
                if not worker.is_finished:
                    worker.cancel()
            # Wait briefly for workers to complete
            await asyncio.sleep(0.1)
            
            # Stop media cleanup timer
            if hasattr(self, '_media_cleanup_timer') and self._media_cleanup_timer:
                self._media_cleanup_timer.stop()
                self.loguru_logger.info("Media cleanup timer stopped")

            try:
                await self._close_server_context_provider_cached_client()
                self.loguru_logger.info("Server context provider cached client closed")
            except Exception as e:
                self.loguru_logger.error(f"Error closing server context provider cached client: {e}")
                
        except Exception as e:
            self.loguru_logger.error(f"Error during service cleanup: {e}")
        
        # Original cleanup code
        if self._rich_log_handler: # Ensure it's removed if it exists
            logging.getLogger().removeHandler(self._rich_log_handler)
            logging.info("RichLogHandler removed.")

        # Stop DB size update timer on unmount as well, if not already handled by shutdown_request
        self.db_status_manager.stop_periodic_updates()
        self._stop_footer_status_timers()
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
        
        # Force cleanup of any remaining threads and processes
        try:
            import threading
            import subprocess
            import signal
            import platform
            
            # On macOS, force kill any afplay processes
            if platform.system() == "Darwin":
                try:
                    # Find and kill any afplay processes spawned by this app
                    import psutil
                    current_pid = os.getpid()
                    for proc in psutil.process_iter(['pid', 'name', 'ppid']):
                        try:
                            if proc.info['name'] == 'afplay' and proc.info['ppid'] == current_pid:
                                self.loguru_logger.info(f"Killing orphaned afplay process: {proc.info['pid']}")
                                proc.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except ImportError:
                    # Fallback if psutil not available - run in background
                    from textual.worker import work
                    
                    @work(thread=True)
                    def kill_afplay_processes():
                        try:
                            # Kill all afplay processes (less precise but works)
                            subprocess.run(['killall', 'afplay'], capture_output=True, timeout=1)
                            self.loguru_logger.info("Killed all afplay processes")
                        except Exception as e:
                            self.loguru_logger.debug(f"Could not kill afplay processes: {e}")
                    
                    # Run in background to avoid blocking
                    self.run_worker(kill_afplay_processes, name="kill_afplay")
            import concurrent.futures
            import asyncio
            
            # Shutdown thread pool executors
            try:
                # Get the default executor if it exists
                loop = asyncio.get_event_loop()
                if hasattr(loop, '_default_executor') and loop._default_executor:
                    self.loguru_logger.info("Shutting down default executor")
                    loop._default_executor.shutdown(wait=False)
                    loop._default_executor = None
            except Exception as e:
                self.loguru_logger.error(f"Error shutting down executor: {e}")
            
            # Clean up any lingering subprocess
            for proc in subprocess._active.copy():  # Make a copy to avoid modification during iteration
                try:
                    if proc.poll() is None:  # Process is still running
                        self.loguru_logger.warning(f"Terminating lingering subprocess PID: {proc.pid}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=1.0)  # Give it 1 second to terminate
                        except subprocess.TimeoutExpired:
                            proc.kill()  # Force kill if it doesn't terminate
                            proc.wait()
                except Exception as e:
                    self.loguru_logger.error(f"Error terminating subprocess: {e}")
            
            # Force-set daemon flag on ThreadPoolExecutor and AudioPlayer threads
            for thread in threading.enumerate():
                if thread.name.startswith(('ThreadPoolExecutor', 'AudioPlayer')):
                    try:
                        thread.daemon = True
                        self.loguru_logger.info(f"Set daemon flag on {thread.name}")
                    except Exception as e:
                        self.loguru_logger.error(f"Could not set daemon flag on {thread.name}: {e}")
            
            # Log any remaining non-daemon threads
            active_threads = [t for t in threading.enumerate() if t.is_alive() and not t.daemon and t != threading.main_thread()]
            if active_threads:
                self.loguru_logger.warning(f"Active non-daemon threads remaining: {[t.name for t in active_threads]}")
                # Attempt to stop them if they have stop methods
                for thread in active_threads:
                    if hasattr(thread, 'stop') and callable(thread.stop):
                        try:
                            thread.stop()
                            self.loguru_logger.info(f"Stopped thread: {thread.name}")
                        except Exception as e:
                            self.loguru_logger.error(f"Error stopping thread {thread.name}: {e}")
        except Exception as e:
            self.loguru_logger.error(f"Error checking active threads: {e}")

        # Close the persisted Library ingest job history store (after pool
        # shutdown, above -- no more job writes are in flight by this point).
        store = getattr(self, "_library_ingest_jobs_store", None)
        if store is not None:
            store.close()

        logging.shutdown()
        self.loguru_logger.info("--- App Unmounted (Loguru) ---")

    ########################################################################
    #
    # WATCHER - Handles UI changes when current_tab's VALUE changes
    #
    # ######################################################################
    def watch_current_tab(self, old_tab: Optional[str], new_tab: str) -> None:
        """Shows/hides the relevant content window when the tab changes."""
        # Skip entirely when using screen navigation
        if hasattr(self, '_use_screen_navigation') and self._use_screen_navigation:
            return
        if not new_tab:  # Skip if empty
            return
        if not self._ui_ready:
            return
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        
        # Execute tab switch immediately - no debouncing needed
        self._execute_tab_switch(old_tab, new_tab)
    
    def _execute_tab_switch(self, old_tab: Optional[str], new_tab: str) -> None:
        """Execute the actual tab switch immediately."""
        loguru_logger.debug(f"\n>>> DEBUG: Executing tab switch! Old: '{old_tab}', New: '{new_tab}'")
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
            # Update navigation UI to remove active state from old tab
            use_dropdown = get_cli_setting("general", "use_dropdown_navigation", False)
            use_links = get_cli_setting("general", "use_link_navigation", True)
            
            if not use_dropdown:  # Only for non-dropdown navigation
                if use_links:
                    # Update TabLinks active state
                    try:
                        from .UI.Tab_Links import TabLinks
                        tab_links = self.query_one(TabLinks)
                        tab_links.set_active_tab(new_tab)
                    except QueryError:
                        pass
                else:
                    # Remove active class from old tab button
                    try:
                        self.query_one(f"#tab-{old_tab}", Button).remove_class("-active")
                    except QueryError:
                        pass
            # Notes auto-save is owned by the Library notes editor; no tab-switch save here.
            try: self.query_one(f"#tab-{old_tab}", Button).remove_class("-active")
            except QueryError: logging.warning(f"Watcher: Could not find old button #tab-{old_tab}")
            try: self.query_one(f"#{old_tab}-window").display = False
            except QueryError: logging.warning(f"Watcher: Could not find old window #{old_tab}-window")

        # Show New Tab UI
        try:
            # Update navigation UI based on type
            use_dropdown = get_cli_setting("general", "use_dropdown_navigation", False)
            use_links = get_cli_setting("general", "use_link_navigation", True)
            
            if use_dropdown:
                # Update dropdown selection if it exists and differs
                try:
                    dropdown = self.query_one(TabDropdown)
                    dropdown.update_active_tab(new_tab)
                except QueryError:
                    pass
            elif use_links:
                # Update link navigation
                # TabLinks active state is now handled by TabLinks.set_active_tab() above
                pass
            else:
                # Update traditional tab bar button
                self.query_one(f"#tab-{new_tab}", Button).add_class("-active")
            
            new_window = self.query_one(f"#{new_tab}-window")
            
            # Initialize placeholder window if needed (with caching)
            if isinstance(new_window, PlaceholderWindow) and not new_window.is_initialized:
                # Check if we've already started initializing this tab
                if new_tab not in self._initialized_tabs:
                    loguru_logger.info(f"Initializing lazy-loaded window for tab: {new_tab}")
                    self._initialized_tabs.add(new_tab)
                    new_window.initialize()
                else:
                    # Tab is already being initialized, skip
                    loguru_logger.debug(f"Tab {new_tab} already initialized or initializing")
            
            # Always set display to True for the new window
            new_window.display = True
            loguru_logger.debug(f"Set display=True for window: {new_window.__class__.__name__} (id={new_tab}-window)")
            
            # Update word count and token count in footer based on tab
            # (resolve the active screen's own footer -- see
            # `_active_footer_status`, task-264).
            footer = self._active_footer_status()
            if footer is not None:
                if new_tab == TAB_CHAT:
                    # Clear word count when on chat tab
                    footer.update_word_count(0)
                    # Update token count immediately
                    self.call_after_refresh(self.update_token_count_display)
                else:
                    # Clear both when on other tabs
                    footer.update_word_count(0)
                    footer.update_token_count("")

            # Focus input logic (as in original, adjust if needed)
            if new_tab not in [TAB_LOGS, TAB_STATS]: # Don't focus input on these tabs
                input_to_focus: Optional[Union[TextArea, Input]] = None
                try: input_to_focus = new_window.query_one(TextArea)
                except QueryError:
                    try: input_to_focus = new_window.query_one(Input) # Check for Input if TextArea not found
                    except QueryError: pass # No primary input found

                if input_to_focus:
                    input_to_focus.focus()  # Focus immediately, no delay needed
                    logging.debug(f"Watcher: Focused input in '{new_tab}'")
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
            # Add a short delay to ensure the window is fully mounted and ready
            def populate_ccp_widgets():
                try:
                    # Check if the window is actually initialized
                    ccp_window = self.query_one("#conversations_characters_prompts-window")
                    if isinstance(ccp_window, PlaceholderWindow):
                        # Window isn't initialized yet, skip population
                        loguru_logger.warning("CCP window is still a placeholder, skipping widget population")
                        return
                    
                    # Now it's safe to populate widgets
                    self.call_after_refresh(ccp_handlers.populate_ccp_character_select, self)
                    self.call_after_refresh(ccp_handlers.populate_ccp_prompts_list_view, self)
                    self.call_after_refresh(ccp_handlers.populate_ccp_dictionary_select, self)
                    self.call_after_refresh(ccp_handlers.populate_ccp_worldbook_list, self)
                    self.call_after_refresh(ccp_handlers.perform_ccp_conversation_search, self)
                except QueryError:
                    loguru_logger.error("CCP window not found during widget population")
            
            # Call immediately after refresh
            self.call_after_refresh(populate_ccp_widgets)
        elif new_tab == TAB_MEDIA:
            def activate_media_initial_view():
                try:
                    from .UI.MediaWindow_v2 import MediaWindow as MediaWindow_v2

                    media_window = self.query_one(MediaWindow_v2)
                    media_window.activate_initial_view()
                except QueryError:
                    loguru_logger.error("Could not find MediaWindow to activate its initial view.")
            
            # Call immediately after refresh
            self.call_after_refresh(activate_media_initial_view)
        elif new_tab == TAB_SEARCH:
            # Handle search tab initialization with a delay to ensure window is ready
            def initialize_search_tab():
                try:
                    # Check if the window is actually initialized
                    search_window = self.query_one("#search-window")
                    if isinstance(search_window, PlaceholderWindow):
                        # Window isn't initialized yet, skip setting sub-tab
                        loguru_logger.warning("Search window is still a placeholder, skipping sub-tab initialization")
                        return
                    
                    # Now it's safe to set the active sub-tab
                    if not self.search_active_sub_tab:
                        self.search_active_sub_tab = self._initial_search_sub_tab_view
                except QueryError:
                    loguru_logger.error("Search window not found during initialization")
            
            # Call immediately after refresh
            self.call_after_refresh(initialize_search_tab)
        elif new_tab == TAB_INGEST:
            if not self.ingest_active_view:
                self.loguru_logger.debug(
                    f"Switched to Ingest tab, activating initial view: {self._initial_ingest_view}") # Reverted to original debug log
                # Use call_later to ensure the UI has settled after tab switch before changing sub-view
                self.call_later(self._activate_initial_ingest_view)
        elif new_tab == TAB_TOOLS_SETTINGS:
            # Handle tools settings tab initialization
            def initialize_tools_settings():
                try:
                    # Check if the window is actually initialized
                    tools_window = self.query_one("#tools_settings-window")
                    if isinstance(tools_window, PlaceholderWindow):
                        # Window isn't initialized yet, skip for now
                        return
                    
                    # Now it's safe to activate the initial view
                    from .UI.Tools_Settings_Window import ToolsSettingsWindow
                    if isinstance(tools_window, ToolsSettingsWindow):
                        tools_window.activate_initial_view()
                        if not self.tools_settings_active_view:
                            self.tools_settings_active_view = self._initial_tools_settings_view
                            self.loguru_logger.debug(f"Tools & Settings tab initialized with view: {self._initial_tools_settings_view}")
                except QueryError:
                    self.loguru_logger.error("Tools settings window not found during initialization")
            
            # Call immediately after refresh
            self.call_after_refresh(initialize_tools_settings)
        elif new_tab == TAB_LLM:  # New elif block for LLM tab
            if not self.llm_active_view:  # If no view is active yet
                self.loguru_logger.debug(
                    f"Switched to LLM Management tab, activating initial view: {self._initial_llm_view}")
                self.call_later(setattr, self, 'llm_active_view', self._initial_llm_view)
            # Populate LLM help texts when the tab is shown
            self.call_after_refresh(llm_management_events.populate_llm_help_texts, self)
        elif new_tab == TAB_EVALS: # Added for Evals tab
            # EvalsLab is a unified dashboard - no need for view activation
            self.loguru_logger.debug(f"Switched to Evals tab")


    def _log_view_dimensions(self, view, parent):
        """Helper to log view dimensions after refresh."""
        self.loguru_logger.info(f"After refresh - View {view.id} dimensions: width={view.size.width}, height={view.size.height}")
        self.loguru_logger.info(f"After refresh - Parent dimensions: width={parent.size.width}, height={parent.size.height}")
    
    async def _activate_initial_ingest_view(self) -> None:
        self.loguru_logger.info("Attempting to activate initial ingest view via _activate_initial_ingest_view.")
        
        # First, ensure all views are hidden initially
        try:
            content_pane = self.query_one("#ingest-content-pane")
            for child in content_pane.children:
                if child.id and child.id.startswith("ingest-view-"):
                    child.styles.display = "none"
                    self.loguru_logger.debug(f"Initially hiding ingest view: {child.id}")
        except QueryError:
            self.loguru_logger.error("Could not find #ingest-content-pane to hide views initially")
        
        if not self.ingest_active_view: # Check if it hasn't been set by some other means already
            self.loguru_logger.debug(f"Setting ingest_active_view to initial: {self._initial_ingest_view}")
            self.ingest_active_view = self._initial_ingest_view
        else:
            self.loguru_logger.debug(f"Ingest active view already set to '{self.ingest_active_view}'. No change made by _activate_initial_ingest_view.")


    # Watchers for sidebar collapsed states (keep as is)
    def watch_chat_sidebar_collapsed(self, collapsed: bool) -> None:
        """Watch for sidebar collapse state changes."""
        if not self._ui_ready:
            self.loguru_logger.debug("watch_chat_sidebar_collapsed: UI not ready.")
            return
        # Just log the state change - the actual UI update should happen in the screen/window
        self.loguru_logger.debug(f"Chat sidebar collapsed state changed to: {collapsed}")

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
    
    def watch_chat_right_sidebar_width(self, width: int) -> None:
        """Update the width of the chat right sidebar."""
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        try:
            sidebar = self.query_one("#chat-right-sidebar", VerticalScroll)
            sidebar.styles.width = f"{width}%"
        except QueryError:
            # Sidebar might not be created yet
            pass

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
        """EvalsLab uses unified dashboard - no sidebar to collapse."""
        # This method is kept for backwards compatibility but does nothing
        # The new EvalsLab UI doesn't have a collapsible sidebar
        pass
    
    def watch_media_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        """Notify MediaWindow when media_active_view changes."""
        # Temporarily disabled - MediaWindow handles its own navigation via MediaTypeSelectedEvent
        pass
        # if not self._ui_ready:
        #     self.loguru_logger.debug("watch_media_active_view: UI not ready.")
        #     return
        # 
        # if self.current_tab == TAB_MEDIA:
        #     try:
        #         media_window = self.query_one(MediaWindow)
        #         # Sync the MediaWindow's own media_active_view
        #         media_window.media_active_view = new_view
        #         # Call the watcher manually to trigger the view change
        #         if new_view:
        #             media_window.watch_media_active_view(old_view, new_view)
        #         self.loguru_logger.info(f"Notified MediaWindow of view change: {old_view} -> {new_view}")
        #     except QueryError:
        #         self.loguru_logger.error("MediaWindow not found for view update.")
        #     except Exception as e:
        #         self.loguru_logger.error(f"Error updating MediaWindow view: {e}", exc_info=True)

    def show_ingest_view(self, view_id_to_show: Optional[str]):
        """
        Shows the specified ingest view within the ingest-content-pane and hides others.
        If view_id_to_show is None, hides all ingest views.
        """
        # Rebuilt ingest UI manages its own tabs; skip legacy show/hide
        if 'USE_REBUILT_INGEST' in globals() and USE_REBUILT_INGEST:
            return
        self.log.debug(f"Attempting to show ingest view: {view_id_to_show}")
        try:
            ingest_content_pane = self.query_one("#ingest-content-pane")
            if view_id_to_show:
                ingest_content_pane.display = True
        except QueryError:
            return
        for view_id in self.ALL_INGEST_VIEW_IDS:
            try:
                view_container = self.query_one(f"#{view_id}")
                is_target = (view_id == view_id_to_show)
                view_container.display = is_target
                if is_target:
                    if view_id == "ingest-view-local-video":
                        self._initialize_video_models()
                    elif view_id == "ingest-view-local-audio":
                        self._initialize_audio_models()
            except QueryError:
                continue

    def _initialize_video_models(self) -> None:
        """Initialize models for the video ingestion window."""
        try:
            from .UI.MediaIngestWindowRebuilt import MediaIngestWindowRebuilt as MediaIngestWindow

            ingest_window = self.query_one("#ingest-window", MediaIngestWindow)
            # New ingest window doesn't need model initialization
            self.log.debug("New ingest window loaded")
        except Exception as e:
            self.log.debug(f"Could not initialize video models: {e}")

    def _initialize_audio_models(self) -> None:
        """Initialize models for the audio ingestion window."""
        try:
            from .UI.MediaIngestWindowRebuilt import MediaIngestWindowRebuilt as MediaIngestWindow

            ingest_window = self.query_one("#ingest-window", MediaIngestWindow)
            # New ingest window doesn't need model initialization
            self.log.debug("New ingest window loaded")
        except Exception as e:
            self.log.debug(f"Could not initialize audio models: {e}")

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
                # ListView doesn't have prepend, so we'll append and let the list refresh handle ordering
                await notes_list_view.append(new_list_item)

                # 5. Set Focus
                title_input.focus()

                self.notify("New note created in sidebar.", severity="information")
            else:
                self.notify("Failed to create new note (no ID returned).", severity="error")
                self.loguru_logger.error("notes_service.add_note did not return a new_note_id.")

        except CharactersRAGDBError as e: # Specific DB error
            self.loguru_logger.opt(exception=True).error(f"Database error creating new note: {e}")
            self.notify(f"DB error creating note: {e}", severity="error")
        except Exception as e: # Catch-all for other unexpected errors
            self.loguru_logger.opt(exception=True).error(f"Unexpected error creating new note: {e}")
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
                self.loguru_logger.debug(f"ListView child count after search population: {len(notes_list_view.children)}") # Fixed: use len(children) instead of child_count
            else:
                msg = "No notes match your search." if search_term else "No notes found."
                self.notify(msg, severity="information")
                self.loguru_logger.info(msg)

        except CharactersRAGDBError as e:
            self.loguru_logger.opt(exception=True).error(f"Database error searching notes: {e}")
            self.notify(f"DB error searching notes: {e}", severity="error")
        except QueryError as e_query:
            self.loguru_logger.opt(exception=True).error(f"UI element not found during notes search: {e_query}")
            self.notify("UI error during notes search.", severity="error")
        except Exception as e:
            self.loguru_logger.opt(exception=True).error(f"Unexpected error searching notes: {e}")
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
            self.loguru_logger.opt(exception=True).error(f"Database error loading note: {e_db}")
            self.notify(f"DB error loading note: {e_db}", severity="error")
        except QueryError as e_query:
            self.loguru_logger.opt(exception=True).error(f"UI element not found during note load: {e_query}")
            self.notify("UI error during note load.", severity="error")
        except Exception as e:
            self.loguru_logger.opt(exception=True).error(f"Unexpected error loading note: {e}")
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
                    self.loguru_logger.opt(exception=True).error(f"Unexpected error updating list item title: {e_item_update}")
            else:
                # This case might not be hit if service raises exceptions for all failures
                self.notify("Failed to save note. Reason unknown.", severity="error")
                self.loguru_logger.error(f"notes_service.update_note returned False for note {self.current_chat_note_id}")

        except ConflictError:
            self.loguru_logger.warning(f"Save conflict for note {self.current_chat_note_id}. Expected version: {self.current_chat_note_version}")
            self.notify("Save conflict: Note was modified elsewhere. Please reload and reapply changes.", severity="error", timeout=10)
        except CharactersRAGDBError as e_db:
            self.loguru_logger.opt(exception=True).error(f"Database error saving note {self.current_chat_note_id}: {e_db}")
            self.notify(f"DB error saving note: {e_db}", severity="error")
        except QueryError as e_query:
            self.loguru_logger.opt(exception=True).error(f"UI element not found during note save: {e_query}")
            self.notify("UI error during note save.", severity="error")
        except Exception as e:
            self.loguru_logger.opt(exception=True).error(f"Unexpected error saving note {self.current_chat_note_id}: {e}")
            self.notify(f"Error saving note: {type(e).__name__}", severity="error")

    @on(Button.Pressed, "#chat-notes-copy-button")
    async def handle_chat_notes_copy(self, event: Button.Pressed) -> None:
        """Handles the 'Copy Note' button press in the chat sidebar's notes section."""
        self.loguru_logger.info("Copy Note button pressed.")
        
        try:
            # Get title and content from the input fields
            title_input = self.query_one("#chat-notes-title-input", Input)
            content_textarea = self.query_one("#chat-notes-content-textarea", TextArea)
            
            title = title_input.value.strip()
            content = content_textarea.text.strip()
            
            # Check if there's anything to copy
            if not title and not content:
                self.notify("No note content to copy.", severity="warning")
                return
            
            # Format the note for clipboard
            if title and content:
                # Both title and content present
                formatted_note = f"# {title}\n\n{content}"
            elif title:
                # Only title
                formatted_note = f"# {title}"
            else:
                # Only content
                formatted_note = content
            
            # Copy to clipboard
            self.copy_to_clipboard(formatted_note)
            self.notify("Note copied to clipboard!", severity="information")
            self.loguru_logger.info(f"Note copied to clipboard. Title: '{title[:50]}...'" if title else "Note content copied to clipboard.")
            
        except QueryError as e:
            self.loguru_logger.error(f"UI element not found during note copy: {e}")
            self.notify("UI error during note copy.", severity="error")
        except Exception as e:
            self.loguru_logger.opt(exception=True).error(f"Unexpected error copying note: {e}")
            self.notify(f"Error copying note: {type(e).__name__}", severity="error")

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
                self.loguru_logger.opt(exception=True).error(f"Database error listing notes: {e}")
                self.notify(f"DB error listing notes: {e}", severity="error")
            except QueryError as e_query: # If UI elements are not found
                 self.loguru_logger.opt(exception=True).error(f"UI element not found in notes toggle: {e_query}")
                 self.notify("UI error while refreshing notes.", severity="error")
            except Exception as e: # Catch-all for other unexpected errors
                self.loguru_logger.opt(exception=True).error(f"Unexpected error listing notes: {e}")
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

    @on(Collapsible.Toggled, "#chat-conversations")
    async def on_chat_conversations_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Handles the expansion/collapse of the Conversations collapsible section in the chat sidebar."""
        # Check if this is specifically the chat conversations collapsible
        if event.collapsible.id != "chat-conversations":
            return
            
        if not event.collapsible.collapsed:  # If the collapsible was just expanded
            self.loguru_logger.info("Conversations collapsible opened in chat sidebar.")
            
            # Populate the character filter dropdown only once when the collapsible is first opened
            # This avoids the database connection conflicts that occur during startup
            if not self._chat_character_filter_populated:
                self.loguru_logger.info("Populating character filter for the first time.")
                try:
                    from tldw_chatbook.Event_Handlers.Chat_Events import chat_events
                    await chat_events.populate_chat_conversation_character_filter_select(self)
                    self._chat_character_filter_populated = True
                    self.loguru_logger.info("Character filter populated successfully.")
                except Exception as e:
                    self.loguru_logger.opt(exception=True).error(f"Failed to populate character filter: {e}")
            else:
                self.loguru_logger.debug("Character filter already populated, skipping.")
        else:
            self.loguru_logger.info("Conversations collapsible closed in chat sidebar.")
    
    @on(Collapsible.Toggled, "#conv-char-conversations-collapsible")
    async def on_ccp_conversations_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Handles the expansion/collapse of the Conversations collapsible section in the CCP tab."""
        if not event.collapsible.collapsed:  # If the collapsible was just expanded
            self.loguru_logger.info("Conversations collapsible opened in CCP tab.")
            # Trigger initial search to populate the list
            await ccp_handlers.perform_ccp_conversation_search(self)
        else:
            self.loguru_logger.info("Conversations collapsible closed in CCP tab.")

    ########################################################################
    #
    # --- EVENT DISPATCHERS ---
    #
    ########################################################################
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatches button presses to the appropriate event handler."""
        button_id = event.button.id
        if not button_id:
            return

        self.loguru_logger.info(f"Button pressed: ID='{button_id}'")

        # Screen-based navigation: let the screen handle its own buttons
        # The screen should handle its own button events
        # If it bubbles up here, it's a navigation button or unhandled
        # Navigation buttons are already handled by NavigateToScreen messages
        self.loguru_logger.debug(f"Button event '{button_id}' reached app level in screen navigation mode")
        return

        # Legacy tab-based button handling below (never reached)
        self.loguru_logger.info(f"Button pressed: ID='{button_id}' on Tab='{self.current_tab}'")

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
                TAB_MEDIA: "media-window",
                TAB_SEARCH: "search-window",
                TAB_INGEST: "ingest-window",
                TAB_TOOLS_SETTINGS: "tools_settings-window",
                TAB_LLM: "llm_management-window",
                TAB_CUSTOMIZE: "customize-window",
                TAB_LOGS: "logs-window",
                TAB_STATS: "stats-window",
                TAB_EVALS: "evals-window",
                TAB_CODING: "coding-window",
                TAB_STTS: "stts-window",
                TAB_STUDY: "study-window",
                TAB_CHATBOOKS: "chatbooks-window"
            }

            window_id = window_id_map.get(self.current_tab)
            self.loguru_logger.info(f"Window ID for tab '{self.current_tab}': {window_id}")
            if window_id:
                # Use super().query_one to access app-level widgets in tab mode
                window = super().query_one(f"#{window_id}")
                self.loguru_logger.info(f"Found window: {type(window).__name__}")
                # Check if the window has an on_button_pressed method
                has_method = hasattr(window, "on_button_pressed") and callable(window.on_button_pressed)
                self.loguru_logger.info(f"Window has on_button_pressed: {has_method}")
                if has_method:
                    # Call the window's button handler - it might be async
                    self.loguru_logger.info(f"Delegating to window's on_button_pressed")
                    result = window.on_button_pressed(event)
                    if inspect.isawaitable(result):
                        await result
                    # Check if event has been stopped (some event types don't have is_stopped)
                    if hasattr(event, 'is_stopped') and event.is_stopped:
                        self.loguru_logger.info(f"Event was stopped by window handler")
                        return
                    self.loguru_logger.info(f"Window handler completed, event not stopped")
                    # Don't return here - let it fall through to the handler map!

        except QueryError:
            self.loguru_logger.error(f"Could not find window component for tab '{self.current_tab}'")
        except Exception as e:
            self.loguru_logger.opt(exception=True).error(f"Error delegating button press to window component: {e}")

        # 3. Use the handler map for buttons not handled by window components
        current_tab_handlers = self.button_handler_map.get(self.current_tab, {})
        handler = current_tab_handlers.get(button_id)
        
        self.loguru_logger.info(f"Looking for handler for button '{button_id}' in tab '{self.current_tab}'")
        self.loguru_logger.info(f"Available handlers for this tab: {list(current_tab_handlers.keys())}")
        self.loguru_logger.info(f"Handler found: {handler is not None}")
        
        # Special debug logging for save chat button
        if button_id == "chat-save-current-chat-button":
            self.loguru_logger.info(f"Save Temp Chat button pressed - Handler found: {handler is not None}")
            self.loguru_logger.info(f"Current tab: {self.current_tab}, Expected: {TAB_CHAT}")

        if handler:
            if callable(handler):
                try:
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
                except Exception as e:
                    self.loguru_logger.opt(exception=True).error(f"Error executing handler for button '{button_id}': {e}")
                    self.notify(f"Error handling button action: {str(e)[:100]}", severity="error")
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
        # Notes editor changes are handled inside the Library screen, not dispatched here.

    def _update_model_download_log(self, message: str) -> None:
        """Helper to write messages to the model download log widget."""
        LogWidgetManager.update_model_download_log(self, message)

    def _update_mlx_log(self, message: str) -> None:
        """Helper to write messages to the MLX-LM log widget."""
        LogWidgetManager.update_mlx_log(self, message)

    async def on_input_changed(self, event: Input.Changed) -> None:
        input_id = event.input.id
        current_active_tab = self.current_tab
        # --- Notes input events are handled inside the Library screen, not here ---
        # --- Chat Sidebar Conversation Search ---
        if input_id == "chat-conversation-search-bar" and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_conversation_search_bar_changed(self, event.value)
        elif input_id == "chat-conversation-keyword-search-bar" and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_conversation_search_bar_changed(self, event.value)
        elif input_id == "chat-conversation-tags-search-bar" and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_conversation_search_bar_changed(self, event.value)
        elif input_id == "conv-char-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_conversation_search_input_changed(self, event)
        elif input_id == "conv-char-keyword-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_conversation_keyword_search_input_changed(self, event)
        elif input_id == "conv-char-tags-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_conversation_tags_search_input_changed(self, event)
        elif input_id == "ccp-prompt-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_prompt_search_input_changed(self, event)
        elif input_id == "ccp-worldbook-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_worldbook_search_input_changed(self, event)
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
        elif input_id == "chat-llm-max-tokens" and current_active_tab == TAB_CHAT:
            # Update token counter when max tokens value changes
            self.call_after_refresh(self.update_token_count_display)
        elif input_id == "chat-custom-token-limit" and current_active_tab == TAB_CHAT:
            # Update token counter when custom token limit changes
            self.call_after_refresh(self.update_token_count_display)
        elif input_id == "chat-settings-search" and current_active_tab == TAB_CHAT:
            await self.handle_settings_search(event.value)
        # --- Chat Tab World Book Search Input ---
        elif input_id == "chat-worldbook-search-input" and current_active_tab == TAB_CHAT:
            await chat_events_worldbooks.handle_worldbook_search_input(self, event.value)
        # --- Chat Tab Media Search Input ---
        # elif input_id == "chat-media-search-input" and current_active_tab == TAB_CHAT:
        #     await handle_chat_media_search_input_changed(self, event.input)
        # --- Media Tab Search Inputs ---
        elif input_id and input_id.startswith("media-search-input-") and current_active_tab == TAB_MEDIA:
            await media_events.handle_media_search_input_changed(self, input_id, event.value)
        elif input_id and input_id.startswith("media-keyword-filter-") and current_active_tab == TAB_MEDIA:
            # Handle keyword filter changes with debouncing
            await media_events.handle_media_search_input_changed(self, input_id.replace("media-keyword-filter-", "media-search-input-"), event.value)
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

        # Notes list view selection is handled inside the Library screen, not here.

        elif list_view_id == "ccp-prompts-listview" and current_active_tab == TAB_CCP:
            self.loguru_logger.debug("Dispatching to ccp_handlers.handle_ccp_prompts_list_view_selected")
            await ccp_handlers.handle_ccp_prompts_list_view_selected(self, list_view_id, event.item)

        elif list_view_id == "chat-sidebar-prompts-listview" and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug("Dispatching to chat_handlers.handle_chat_sidebar_prompts_list_view_selected")
            await ccp_handlers.handle_ccp_prompts_list_view_selected(self, list_view_id, event.item)

        elif list_view_id == "chat-media-search-results-listview" and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug("Dispatching to chat_events_sidebar.handle_media_item_selected")
            await chat_events_sidebar.handle_media_item_selected(self, event.item)

        elif list_view_id == "chat-conversation-search-results-list" and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug("Conversation selected in chat tab search results")
            # Store the selected item for the Load Selected button, but don't load immediately
            # This maintains the existing UX where users must click "Load Selected"
            
        elif list_view_id in ["chat-worldbook-available-listview", "chat-worldbook-active-listview"] and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug(f"World book selected in {list_view_id}")
            await chat_events_worldbooks.handle_worldbook_selection(self, list_view_id)

        # Note: conv-char-search-results-list selections are handled by their respective "Load Selected" buttons.
        else:
            self.loguru_logger.warning(
            f"No specific handler for ListView.Selected from list_view_id='{list_view_id}' on tab='{current_active_tab}'")

    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        checkbox_id = event.checkbox.id
        current_active_tab = self.current_tab

        if checkbox_id.startswith("chat-conversation-search-") and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_search_checkbox_changed(self, checkbox_id, event.value)
        elif checkbox_id.startswith("conv-char-search-") and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_search_checkbox_changed(self, checkbox_id, event.value)
        elif checkbox_id == "chat-show-attach-button-checkbox" and current_active_tab == TAB_CHAT:
            # Handle attach button visibility toggle
            from .config import save_setting_to_cli_config
            save_setting_to_cli_config("chat.images", "show_attach_button", event.value)
            
            # Update the UI if enhanced chat window is active
            use_enhanced_chat = get_cli_setting("chat_defaults", "use_enhanced_window", False)
            if use_enhanced_chat:
                try:
                    from .UI.Chat_Window_Enhanced import ChatWindowEnhanced
                    chat_window = self.query_one("#chat-window", ChatWindowEnhanced)
                    await chat_window.toggle_attach_button_visibility(event.value)
                except Exception as e:
                    loguru_logger.error(f"Error toggling attach button visibility: {e}")
        elif checkbox_id == "chat-show-dictation-button-checkbox" and current_active_tab == TAB_CHAT:
            # Handle dictation button visibility toggle
            from .config import save_setting_to_cli_config
            save_setting_to_cli_config("chat.voice", "show_mic_button", event.value)
            
            # Update the UI dynamically
            try:
                mic_button = self.query_one("#mic-button", Button)
                mic_button.display = event.value
                self.notify(f"Dictation button {'shown' if event.value else 'hidden'}", timeout=2)
            except QueryError:
                # If button doesn't exist, we'll need to refresh the chat window
                self.notify("Dictation button setting saved. Restart chat to apply changes.", timeout=3)
        elif checkbox_id == "chat-worldbook-enable-checkbox" and current_active_tab == TAB_CHAT:
            await chat_events_worldbooks.handle_worldbook_enable_checkbox(self, event.value)
        elif checkbox_id == "chat-settings-mode-toggle" and current_active_tab == TAB_CHAT:
            # Handle settings mode toggle checkbox
            await self.handle_settings_mode_toggle_checkbox(event)
        # Add handlers for checkboxes in other tabs if any

    async def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handles changes in Switch widgets."""

        switch_id = event.switch.id
        current_active_tab = self.current_tab
        
        if switch_id == "notes-auto-save-toggle":
            await self.handle_notes_auto_save_toggle(event)


    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handles changes in Select widgets if specific actions are needed beyond watchers."""
        select_id = event.select.id
        current_active_tab = self.current_tab
        self.loguru_logger.debug(f"Select changed: {select_id} = {event.value}, current tab = {current_active_tab}")

        if select_id == "conv-char-character-select" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_character_select_changed(self, event.value)
        elif select_id == "tldw-api-auth-method" and current_active_tab == TAB_INGEST:
            await ingest_events.handle_tldw_api_auth_method_changed(self, str(event.value))
        elif select_id == "tldw-api-media-type" and current_active_tab == TAB_INGEST:
            await ingest_events.handle_tldw_api_media_type_changed(self, str(event.value))
        # Notes sort select is handled inside the Library screen, not here.
        elif select_id == "chat-rag-preset" and current_active_tab == TAB_CHAT:
            await self.handle_rag_preset_changed(event)
        elif select_id == "chat-rag-search-mode" and current_active_tab == TAB_CHAT:
            await self.handle_rag_pipeline_changed(event)
        elif select_id == "chat-rag-expansion-method" and current_active_tab == TAB_CHAT:
            await self.handle_query_expansion_method_changed(event)
        elif select_id == "chat-rag-expansion-provider" and current_active_tab == TAB_CHAT:
            # Update the reactive value to trigger the watcher
            self.rag_expansion_provider_value = event.value
        elif select_id == "chat-api-provider" and current_active_tab == TAB_CHAT:
            # This is now handled in ChatScreen via @on decorator
            self.loguru_logger.debug(f"chat-api-provider change event (handled in ChatScreen): {event.value}")
            
            # Update token counter when provider changes
            if self._ui_ready:
                try:
                    from .Event_Handlers.Chat_Events.chat_token_events import update_chat_token_counter
                    await update_chat_token_counter(self)
                except Exception as e:
                    self.loguru_logger.debug(f"Could not update token counter on provider change: {e}")
        elif select_id == "chat-api-model" and current_active_tab == TAB_CHAT:
            # Update token counter when model changes
            if self._ui_ready:
                try:
                    from .Event_Handlers.Chat_Events.chat_token_events import update_chat_token_counter
                    await update_chat_token_counter(self)
                except Exception as e:
                    self.loguru_logger.debug(f"Could not update token counter on model change: {e}")
        elif select_id == "chat-conversation-search-character-filter-select" and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug("Character filter changed in chat tab, triggering conversation search")
            await chat_handlers.perform_chat_conversation_search(self)
        elif select_id == "chat-rag-expansion-method" and current_active_tab == TAB_CHAT:
            # Handle query expansion method change - show/hide appropriate fields
            await self.handle_query_expansion_method_changed(event)

    ##################################################################
    # --- Event Handlers for Streaming and Worker State Changes ---
    ##################################################################
    @on(StreamingChunk)
    async def on_streaming_chunk(self, event: StreamingChunk) -> None:
        await handle_streaming_chunk(self, event)

    @on(StreamDone)
    async def on_stream_done(self, event: StreamDone) -> None:
        await handle_stream_done(self, event)
    
    @on(media_events.MediaMetadataUpdateEvent)
    async def on_media_metadata_update(self, event: media_events.MediaMetadataUpdateEvent) -> None:
        await media_events.handle_media_metadata_update(self, event)
    
    # Collections/Tags event handlers
    @on(Message)
    async def on_collections_tag_message(self, event: Message) -> None:
        """Handle Collections/Tag events."""
        from .Event_Handlers import collections_tag_events
        
        if event.__class__.__name__ == 'KeywordRenameEvent':
            await collections_tag_events.handle_keyword_rename(self, event)
        elif event.__class__.__name__ == 'KeywordMergeEvent':
            await collections_tag_events.handle_keyword_merge(self, event)
        elif event.__class__.__name__ == 'KeywordDeleteEvent':
            await collections_tag_events.handle_keyword_delete(self, event)
        elif event.__class__.__name__ == 'BatchAnalysisStartEvent':
            from .Event_Handlers import multi_item_review_events
            await multi_item_review_events.handle_batch_analysis_start(self, event)

    @on(SplashScreen.Closed)
    async def on_splash_screen_closed(self, event: SplashScreen.Closed) -> None:
        """Handle splash screen closing."""
        self.splash_screen_active = False
        logger.debug("Splash screen closed, mounting main UI")
        
        # Remove the splash screen
        if self._splash_screen_widget:
            await self._splash_screen_widget.remove()
            self._splash_screen_widget = None
        
        # Mount the shared app chrome before pushing the first screen so
        # persistent navigation is available after splash startup too.
        existing_ids = {widget.id for widget in self.screen._nodes if widget.id}
        main_ui_widgets = self._create_main_ui_widgets()
        widgets_to_mount = []
        for widget in main_ui_widgets:
            if widget.id not in existing_ids:
                widgets_to_mount.append(widget)
            else:
                logger.debug(f"Skipping duplicate widget with ID: {widget.id}")

        if widgets_to_mount:
            await self.mount(*widgets_to_mount)

        # Push the initial screen after the shared navigation is mounted.
        await self._push_initial_screen()

        # Screen navigation uses buffered logging until the Logs screen is ready.
        self._setup_buffered_logging()

        # Finish deferred startup work once the mounted screen has rendered.
        self.call_after_refresh(self._post_mount_setup)
    

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
            from .config import save_setting_to_cli_config
            save_setting_to_cli_config("chat_defaults", "strip_thinking_tags", new_value)
            self.notify(f"Thinking tag stripping {'enabled' if new_value else 'disabled'}.", timeout=2)
        except Exception as e:
            self.loguru_logger.opt(exception=True).error(f"Failed to save 'strip_thinking_tags' setting: {e}")
            self.notify("Error saving thinking tag setting.", severity="error", timeout=4)
    #####################################################################
    # --- End of Chat Event Handlers for Streaming & thinking tags ---
    #####################################################################

    async def handle_settings_mode_toggle_checkbox(self, event: Checkbox.Changed) -> None:
        """Handles the settings mode toggle checkbox between Basic and Advanced."""
        try:
            # Update reactive variable
            self.chat_settings_mode = "advanced" if event.value else "basic"
            
            # Update sidebar class for CSS styling
            sidebar = self.query_one("#chat-left-sidebar")
            if self.chat_settings_mode == "basic":
                sidebar.add_class("basic-mode")
                sidebar.remove_class("advanced-mode")
            else:
                sidebar.add_class("advanced-mode")
                sidebar.remove_class("basic-mode")
            
            # Notify user
            mode_name = "Advanced" if event.value else "Basic"
            self.notify(f"Switched to {mode_name} mode", timeout=2)
            
            # Save preference to config
            from .config import save_setting_to_cli_config
            save_setting_to_cli_config("chat_defaults", "advanced_mode", event.value)
            
        except Exception as e:
            loguru_logger.opt(exception=True).error(f"Error toggling settings mode: {e}")
            self.notify("Error switching modes", severity="error", timeout=4)
    
    async def handle_settings_mode_toggle(self, event: Switch.Changed) -> None:
        """Handles the settings mode toggle between Basic and Advanced."""
        try:
            # Update reactive variable
            self.chat_settings_mode = "advanced" if event.value else "basic"
            
            # Update sidebar class for CSS styling
            sidebar = self.query_one("#chat-left-sidebar")
            if self.chat_settings_mode == "basic":
                sidebar.add_class("basic-mode")
                sidebar.remove_class("advanced-mode")
            else:
                sidebar.add_class("advanced-mode") 
                sidebar.remove_class("basic-mode")
            
            self.notify(f"Settings mode: {self.chat_settings_mode.title()}")
            self.loguru_logger.info(f"Switched to {self.chat_settings_mode} settings mode")
            
        except Exception as e:
            self.loguru_logger.error(f"Error toggling settings mode: {e}")
            self.notify("Error switching settings mode", severity="error")

    async def handle_notes_auto_save_toggle(self, event: Switch.Changed) -> None:
        """Handles the notes auto-save toggle."""
        try:
            # Update the reactive variable
            self.notes_auto_save_enabled = event.value
            
            # If auto-save is being disabled, cancel any pending timer
            if not event.value and self.notes_auto_save_timer is not None:
                self.notes_auto_save_timer.stop()
                self.notes_auto_save_timer = None
                self.loguru_logger.info("Auto-save timer cancelled")
            
            # Save the setting to config
            from .config import save_setting_to_cli_config
            save_setting_to_cli_config("notes", "auto_save_enabled", event.value)
            
            # Notify the user
            status = "enabled" if event.value else "disabled"
            self.notify(f"Notes auto-save {status}", timeout=2)
            self.loguru_logger.info(f"Notes auto-save {status}")
            
        except Exception as e:
            self.loguru_logger.error(f"Error toggling notes auto-save: {e}")
            self.notify("Error changing auto-save setting", severity="error")

    async def handle_rag_preset_changed(self, event: Select.Changed) -> None:
        """Handles RAG preset selection."""
        try:
            preset = event.value
            
            # In screen navigation mode, these widgets don't exist at app level
            self.loguru_logger.debug(f"RAG preset change in screen mode - preset: {preset}")
            # Store the preset for the screen to handle
            self.rag_preset = preset
            return
            
            # Apply preset configurations
            if preset == "none":
                rag_enable.value = False
                self.notify("RAG disabled")
            elif preset == "light":
                rag_enable.value = True
                top_k.value = "3"
                self.notify("Light RAG: BM25 only, top 3 results")
            elif preset == "full":
                rag_enable.value = True
                top_k.value = "10"
                # Try to enable reranking if in advanced mode
                try:
                    rerank = self.query_one("#chat-rag-rerank-enable-checkbox", Checkbox)
                    rerank.value = True
                except QueryError:
                    pass  # Reranking is in advanced mode
                self.notify("Full RAG: Embeddings + reranking, top 10 results")
            elif preset == "custom":
                rag_enable.value = True
                self.notify("Custom RAG: Configure settings manually")
                
            self.loguru_logger.info(f"Applied RAG preset: {preset}")
            
        except Exception as e:
            self.loguru_logger.error(f"Error applying RAG preset: {e}")
            self.notify("Error applying RAG preset", severity="error")
    
    async def handle_rag_pipeline_changed(self, event: Select.Changed) -> None:
        """Handles RAG pipeline selection."""
        try:
            from .Widgets.settings_sidebar import get_pipeline_description
            
            pipeline_id = event.value
            
            # In screen navigation mode, these widgets don't exist at app level
            self.loguru_logger.debug(f"RAG pipeline change in screen mode - pipeline: {pipeline_id}")
            # Store the pipeline for the screen to handle
            self.rag_pipeline = pipeline_id
            return
            
            # If "none" is selected, just show manual config message
            if pipeline_id == "none":
                self.notify("Manual RAG configuration mode enabled")
            else:
                # Show what pipeline was selected
                pipeline_name = event.select.value_to_label(pipeline_id)
                self.notify(f"Selected pipeline: {pipeline_name}")
                
            self.loguru_logger.info(f"RAG pipeline changed to: {pipeline_id}")
            
        except Exception as e:
            self.loguru_logger.error(f"Error handling RAG pipeline change: {e}")
            self.notify("Error changing RAG pipeline", severity="error")
    
    async def handle_query_expansion_method_changed(self, event: Select.Changed) -> None:
        """Stores the selected query expansion method string.

        NOTE (task-252): UI-only stub. The RAG_Search/query_expansion.py module was
        removed as dead code; this handler just stores the selected method string
        and no runtime code performs query expansion with it.

        Args:
            event: The Select.Changed event carrying the chosen expansion method.
        """
        try:
            method = event.value
            # In screen navigation mode, these widgets don't exist at app level
            self.loguru_logger.debug(f"Query expansion method change in screen mode - method: {method}")
            # Store the method for the screen to handle
            self.query_expansion_method = method
        except Exception as e:
            self.loguru_logger.error(f"Error handling query expansion method change: {e}")
            self.notify("Error updating query expansion settings", severity="error")

    async def handle_settings_search(self, query: str) -> None:
        """Handles search in settings sidebar."""
        try:
            query = query.lower().strip()
            
            # Get all settings elements
            sidebar = self.query_one("#chat-left-sidebar")
            
            if not query:
                # Clear search - show all settings based on current mode
                for widget in sidebar.query(".sidebar-label, .section-header, .subsection-header"):
                    widget.remove_class("search-highlight")
                for collapsible in sidebar.query(Collapsible):
                    # Respect the original collapsed state
                    pass
                return
                
            # Search through all labels and highlight matches
            matches_found = 0
            
            for label in sidebar.query(".sidebar-label, .section-header, .subsection-header"):
                if isinstance(label, (Static, Label)):
                    label_text = str(label.renderable).lower()
                    if query in label_text:
                        label.add_class("search-highlight")
                        matches_found += 1
                        
                        # Expand parent collapsibles to show match
                        parent = label.parent
                        while parent and parent != sidebar:
                            if isinstance(parent, Collapsible):
                                parent.collapsed = False
                            parent = parent.parent
                    else:
                        label.remove_class("search-highlight")
                        
            if matches_found == 0:
                self.notify(f"No settings found for '{query}'", severity="warning")
            else:
                self.notify(f"Found {matches_found} settings matching '{query}'")
                
        except Exception as e:
            self.loguru_logger.error(f"Error in settings search: {e}")
            self.notify("Error searching settings", severity="error")


    #####################################################################
    # --- Event Handlers for Worker State Changes ---
    #####################################################################
    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """
        Handle worker state changes by delegating to the appropriate handler.
        
        This method has been refactored to use a handler registry pattern,
        significantly reducing complexity and improving maintainability.
        """
        worker_name = event.worker.name
        worker_group = event.worker.group
        
        # Log the state change
        self.loguru_logger.debug(
            f"on_worker_state_changed: Worker '{worker_name}' "
            f"(Group: {worker_group}, State: {event.state})"
        )
        
        # Delegate to the handler registry
        handled = await self.worker_handler_registry.handle_event(event)
        
        if not handled:
            # Log unhandled workers for debugging
            self.loguru_logger.warning(
                f"No handler found for worker '{worker_name}' (Group: {worker_group})"
            )
            
        # TODO: Fix this - new_user_prompt is not defined
        # try:
        #     self.query_one("#chat-prompt-user-display", TextArea).load_text(new_user_prompt or "")
        # except QueryError:
        #     self.loguru_logger.error("Chat sidebar user prompt display area (#chat-prompt-user-display) not found.")

    def _clear_chat_sidebar_prompt_display(self) -> None:
        """Clears the prompt display TextAreas in the chat sidebar."""
        UIHelpers.clear_chat_sidebar_prompt_display(self)

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

    def watch_rag_expansion_provider_value(self, new_value: Optional[str]) -> None:
        """Watch for changes in RAG expansion provider selection."""
        if not hasattr(self, "app") or not self.app:
            return
        if not self._ui_ready:
            return
        self.loguru_logger.debug(f"Watcher: rag_expansion_provider_value changed to {new_value}")
        if new_value is None or new_value == Select.BLANK:
            self._update_rag_expansion_model_select([])
            return
        models = self.providers_models.get(new_value, [])
        self._update_rag_expansion_model_select(models)


    def _update_model_select(self, id_prefix: str, models: list[str]) -> None:
        if not self._ui_ready:  # Add guard
            return
        UIHelpers.update_model_select(self, id_prefix, models)

    def _update_rag_expansion_model_select(self, models: list[str]) -> None:
        """Update the RAG expansion model select options."""
        if not self._ui_ready:
            return
        UIHelpers.update_rag_expansion_model_select(self, models)

    def chat_wrapper(self, strip_thinking_tags: bool = True, **kwargs: Any) -> Any:
        """
        Delegates to the actual worker target function in worker_events.py.
        This method is called by app.run_worker.
        """
        # All necessary parameters (message, history, api_endpoint, model, etc.)
        # are passed via kwargs from the calling event handler (e.g., handle_chat_send_button_pressed).
        return worker_events.chat_wrapper_function(self, strip_thinking_tags=strip_thinking_tags, **kwargs) # Pass self as 'app_instance'

    def schedule_media_cleanup(self) -> None:
        """Schedule periodic media cleanup based on configuration."""
        try:
            # Get cleanup configuration
            cleanup_config = get_cli_setting("media_cleanup", "enabled", True)
            if not cleanup_config:
                self.loguru_logger.info("Media cleanup is disabled in configuration")
                return
                
            cleanup_interval_hours = get_cli_setting("media_cleanup", "cleanup_interval_hours", 24)
            cleanup_on_startup = get_cli_setting("media_cleanup", "cleanup_on_startup", True)
            
            # Run cleanup on startup if configured
            if cleanup_on_startup:
                self.loguru_logger.info(
                    "Scheduling media cleanup after startup idle delay"
                )
                self._media_cleanup_startup_timer = self.set_timer(
                    DEFERRED_MEDIA_CLEANUP_DELAY_SECONDS,
                    self.perform_media_cleanup,
                )
            
            # Schedule periodic cleanup
            cleanup_interval_seconds = cleanup_interval_hours * 3600
            self._media_cleanup_timer = self.set_interval(cleanup_interval_seconds, self.perform_media_cleanup)
            self.loguru_logger.info(f"Scheduled media cleanup every {cleanup_interval_hours} hours")
            
        except Exception as e:
            self.loguru_logger.opt(exception=True).error(f"Error scheduling media cleanup: {e}")
    
    async def perform_media_cleanup(self) -> None:
        """Perform media cleanup based on configuration settings."""
        try:
            if not self.media_db:
                self.loguru_logger.warning("Media database not available for cleanup")
                return
                
            # Get cleanup configuration
            cleanup_days = get_cli_setting("media_cleanup", "cleanup_days", 30)
            max_items = get_cli_setting("media_cleanup", "max_items_per_cleanup", 100)
            notify_before = get_cli_setting("media_cleanup", "notify_before_cleanup", True)
            
            # Check for candidates first
            candidates = await asyncio.to_thread(self.media_db.get_deletion_candidates, cleanup_days)
            
            if not candidates:
                self.loguru_logger.info("No media items eligible for cleanup")
                return
                
            candidate_count = len(candidates)
            items_to_delete = min(candidate_count, max_items)
            
            # Notify user if configured
            if notify_before and candidate_count > 0:
                self.notify(
                    f"Found {candidate_count} media items eligible for permanent deletion "
                    f"(soft-deleted over {cleanup_days} days ago). "
                    f"Will delete up to {items_to_delete} items.",
                    title="Media Cleanup",
                    severity="information",
                    timeout=5
                )
            
            # Perform the cleanup
            deleted_count = await asyncio.to_thread(self.media_db.hard_delete_old_media, cleanup_days)
            
            if deleted_count > 0:
                self.loguru_logger.info(f"Media cleanup completed: {deleted_count} items permanently deleted")
                self.notify(
                    f"Media cleanup completed: {deleted_count} items permanently deleted",
                    severity="information",
                    timeout=3
                )
            
        except Exception as e:
            self.loguru_logger.opt(exception=True).error(f"Error during media cleanup: {e}")
            self.notify(
                f"Error during media cleanup: {str(e)}",
                severity="error",
                timeout=5
            )
    
    async def action_show_workbench_help(self) -> None:
        """Delegate contextual help to the active Workbench screen."""
        handler = getattr(self.screen, "action_show_workbench_help", None)
        if callable(handler):
            result = handler()
            if inspect.isawaitable(result):
                await result
            return
        self.notify(
            "No contextual help is available for this screen.",
            severity="information",
        )

    async def action_focus_next_workbench_pane(self) -> None:
        """Delegate pane focus cycling to the active Workbench screen."""
        handler = getattr(self.screen, "action_focus_next_workbench_pane", None)
        if callable(handler):
            result = handler()
            if inspect.isawaitable(result):
                await result
            return
        self.notify(
            "No workbench pane focus target is available.",
            severity="information",
        )

    def action_quit(self) -> None:
        """Handle application quit - save persistent caches before exiting."""
        loguru_logger.info("Application quit initiated")
        
        # Set flag to prevent new operations
        self._shutting_down = True
        
        # Force stop any playing audio and cleanup
        if hasattr(self, 'audio_player'):
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create cleanup tasks
                    async def cleanup_audio():
                        try:
                            await asyncio.wait_for(self.audio_player.stop(), timeout=0.5)
                        except asyncio.TimeoutError:
                            loguru_logger.warning("Audio stop timed out")
                        try:
                            await asyncio.wait_for(self.audio_player.cleanup(), timeout=0.5)
                        except asyncio.TimeoutError:
                            loguru_logger.warning("Audio cleanup timed out")
                    
                    # Schedule cleanup
                    asyncio.create_task(cleanup_audio())
                else:
                    # Synchronous cleanup if no event loop
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self.audio_player.cleanup())
                    loop.close()
                loguru_logger.info("Audio player stopped and cleaned up")
            except Exception as e:
                loguru_logger.error(f"Error stopping audio during quit: {e}")
        
        # Force cleanup Higgs backends immediately
        if hasattr(self, '_stts_handler') and self._stts_handler:
            try:
                if hasattr(self._stts_handler, '_stts_service') and self._stts_handler._stts_service:
                    backend_manager = getattr(self._stts_handler._stts_service, 'backend_manager', None)
                    if backend_manager and hasattr(backend_manager, '_backends'):
                        for backend_id, backend in list(backend_manager._backends.items()):
                            if 'higgs' in backend_id.lower():
                                loguru_logger.info(f"Signaling Higgs backend shutdown: {backend_id}")
                                # Set shutdown event if available
                                if hasattr(backend, '_shutdown_event'):
                                    backend._shutdown_event.set()
            except Exception as e:
                loguru_logger.error(f"Error signaling Higgs shutdown: {e}")
        
        # Cancel media cleanup timer if it exists
        if hasattr(self, '_media_cleanup_timer') and self._media_cleanup_timer:
            self._media_cleanup_timer.stop()
        
        # Handle Notes auto-save cleanup
        if hasattr(self, 'notes_auto_save_timer') and self.notes_auto_save_timer is not None:
            self.notes_auto_save_timer.stop()
            self.notes_auto_save_timer = None
            loguru_logger.debug("Cancelled auto-save timer during app quit")
        
        # Note autosave is owned by the Library notes editor; no legacy quit-save path remains.
        
        # Try to save caches but don't let it block quitting
        try:
            # Import with timeout protection
            import signal
            import threading
            
            def save_caches_with_timeout():
                # Note: The old cache service is deprecated
                # The simplified RAG service handles caching internally
                # and doesn't require explicit save on shutdown
                loguru_logger.debug("Cache saving skipped - handled by simplified RAG service")
            
            # Run cache saving in a separate thread with timeout
            save_thread = threading.Thread(target=save_caches_with_timeout)
            save_thread.daemon = True  # Don't let this thread prevent app exit
            save_thread.start()
            save_thread.join(timeout=2.0)  # Wait max 2 seconds
            
            if save_thread.is_alive():
                loguru_logger.warning("Cache save timed out - proceeding with quit")
        except Exception as e:
            loguru_logger.error(f"Error in quit handler: {e}")
        
        # Save encrypted config if encryption is enabled
        try:
            from tldw_chatbook.config import (
                load_cli_config_and_ensure_existence, 
                get_encryption_password,
                encrypt_api_keys_in_config,
                DEFAULT_CONFIG_PATH
            )
            from tldw_chatbook.Utils.atomic_file_ops import atomic_write_text
            import toml
            
            config_data = load_cli_config_and_ensure_existence()
            encryption_config = config_data.get("encryption", {})
            
            if encryption_config.get("enabled", False):
                password = get_encryption_password()
                if password:
                    loguru_logger.info("Encrypting configuration before exit...")
                    try:
                        # Encrypt the config
                        encrypted_config = encrypt_api_keys_in_config(config_data, password)
                        
                        # Save the encrypted config
                        config_text = toml.dumps(encrypted_config)
                        atomic_write_text(DEFAULT_CONFIG_PATH, config_text)
                        
                        loguru_logger.info("Configuration encrypted and saved successfully")
                    except Exception as e:
                        loguru_logger.error(f"Failed to encrypt config on exit: {e}")
                        # Continue with exit even if encryption fails
                else:
                    loguru_logger.warning("Encryption enabled but no password available - config not encrypted")
        except Exception as e:
            loguru_logger.error(f"Error during config encryption on exit: {e}")
        
        # Always call the parent quit method
        self.exit()

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

    # --- Initialize Metrics Systems ---
    # Initialize Prometheus metrics server
    try:
        # Start Prometheus metrics server on port 8000 (or configure via env/config)
        metrics_port = int(os.environ.get('METRICS_PORT', '8000'))
        init_metrics_server(port=metrics_port)
        loguru_logger.info(f"Prometheus metrics server started on port {metrics_port}")
    except Exception as e:
        loguru_logger.warning(f"Failed to start Prometheus metrics server: {e}")
        # Continue without metrics server - metrics are still collected
    
    # Initialize OpenTelemetry metrics
    try:
        # Initialize OpenTelemetry for advanced metrics collection
        # This complements the existing Prometheus metrics
        init_otel_metrics()
        loguru_logger.info("OpenTelemetry metrics initialized successfully")
    except Exception as e:
        loguru_logger.warning(f"Failed to initialize OpenTelemetry metrics: {e}")
        # Continue without OpenTelemetry - the app still has Prometheus metrics

    # --- Emoji Check ---
    emoji_is_supported = supports_emoji() # Call it once
    loguru_logger.info(f"Terminal emoji support detected: {emoji_is_supported}")
    loguru_logger.info(f"Using brain: {get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}")
    loguru_logger.info("-" * 30)

    # --- CSS File Handling ---
    try:
        css_dir = Path(__file__).parent / "css"
        css_dir.mkdir(exist_ok=True)
        
        # Check if modular CSS needs to be built
        modular_css_path = css_dir / "tldw_cli_modular.tcss"
        build_script_path = css_dir / "build_css.py"
        
        # Check if any module is newer than the built file
        should_rebuild = False
        if not modular_css_path.exists():
            should_rebuild = True
            logging.info("Modular CSS file not found, will build it")
        elif build_script_path.exists():
            # Check if any module file is newer than the built file
            modular_mtime = modular_css_path.stat().st_mtime
            for subdir in ['core', 'layout', 'components', 'features', 'utilities']:
                subdir_path = css_dir / subdir
                if subdir_path.exists():
                    for css_file in subdir_path.glob('*.tcss'):
                        if css_file.stat().st_mtime > modular_mtime:
                            should_rebuild = True
                            logging.info(f"Module {css_file.name} is newer than built CSS, rebuilding")
                            break
                if should_rebuild:
                    break
        
        if should_rebuild and build_script_path.exists():
            logging.info("Building modular CSS...")
            import subprocess
            # Build CSS synchronously before starting the app
            result = subprocess.run([sys.executable, str(build_script_path)], 
                                  cwd=str(css_dir), 
                                  capture_output=True, 
                                  text=True)
            if result.returncode == 0:
                logging.info("Successfully built modular CSS")
            else:
                logging.error(f"Failed to build modular CSS: {result.stderr}")
        
    except Exception as e_css_main:
        logging.error(f"Error handling CSS file: {e_css_main}", exc_info=True)

    # --- Check for encrypted config (config will be created if it doesn't exist) ---
    try:
        config_data = load_cli_config_and_ensure_existence()
        encryption_config = config_data.get("encryption", {})
        
        if encryption_config.get("enabled", False):
            loguru_logger.info("Config file encryption is enabled. Password required.")
            
            # Import password dialog dependencies here to avoid circular imports
            import asyncio
            from textual.app import App
            from tldw_chatbook.Widgets.password_dialog import PasswordDialog
            
            class PasswordPromptApp(App):
                """Minimal app to prompt for password."""
                def __init__(self):
                    super().__init__()
                    self.password = None
                
                async def on_mount(self) -> None:
                    """Show password dialog immediately on mount."""
                    password = await self.push_screen(
                        PasswordDialog(
                            mode="unlock",
                            title="Unlock Configuration",
                            message="Enter your master password to decrypt the configuration file.",
                            on_submit=lambda p: None,
                            on_cancel=lambda: None
                        ),
                        wait_for_dismiss=True
                    )
                    
                    if password:
                        # Verify password
                        from tldw_chatbook.Utils.config_encryption import config_encryption
                        password_verifier = encryption_config.get("password_verifier", "")
                        if password_verifier and config_encryption.verify_password(password, password_verifier):
                            self.password = password
                            self.exit()
                        else:
                            self.notify("Invalid password. Please try again.", severity="error")
                            # Re-show the dialog
                            await self.on_mount()
                    else:
                        # User cancelled
                        loguru_logger.error("Password required but not provided. Exiting.")
                        self.exit()
            
            # Run the password prompt app
            password_app = PasswordPromptApp()
            password_app.run()
            
            if password_app.password:
                # Set the password for the session
                set_encryption_password(password_app.password)
                loguru_logger.info("Configuration decrypted successfully.")
            else:
                # Exit if no password provided
                loguru_logger.error("Cannot proceed without decryption password.")
                sys.exit(1)
                
    except Exception as e:
        loguru_logger.error(f"Error checking config encryption: {e}")
        # Continue without encryption if there's an error

    # Create instance with early logging flag
    app_instance = TldwCli()
    # Set the early logging flag so _setup_logging knows logging was already initialized
    app_instance._early_logging_initialized = True
    try:
        app_instance.run()
    except KeyboardInterrupt:
        loguru_logger.info("--- KeyboardInterrupt received ---")
        # Force cleanup inline
        import threading
        import concurrent.futures
        for thread in threading.enumerate():
            if thread != threading.main_thread() and not thread.daemon:
                try:
                    thread.daemon = True
                except Exception:
                    pass
        try:
            concurrent.futures.thread._threads_queues.clear()
        except Exception:
            pass
    except Exception as e:
        loguru_logger.exception("--- CRITICAL ERROR DURING app.run() ---")
        traceback.print_exc()  # Make sure traceback prints
    finally:
        # This might run even if app exits early internally in run()
        loguru_logger.info("--- FINALLY block after app.run() ---")

    loguru_logger.info("--- AFTER app.run() call (if not crashed hard) ---")

# Entry point for the tldw-chatbook command
def get_app():
    """Entry point for textual serve.
    
    Returns the TldwCli app instance without running it.
    """
    # Configure logging to suppress verbose debug messages early
    import logging
    import os
    import warnings
    
    # Suppress various verbose loggers
    logging.getLogger("torio._extension.utils").setLevel(logging.WARNING)
    logging.getLogger("torio").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("fsevents").setLevel(logging.WARNING)
    
    # Ensure CSS is built
    from pathlib import Path
    import sys
    
    # Check if we need to build CSS
    # Get the directory where app.py is located
    app_dir = Path(__file__).parent
    css_dir = app_dir / "css"
    modular_css_path = css_dir / "tldw_cli_modular.tcss"
    build_script_path = css_dir / "build_css.py"
    
    if not modular_css_path.exists() and build_script_path.exists():
        print("Building modular CSS...")
        import subprocess
        subprocess.run([sys.executable, str(build_script_path)], check=True)
        
    return TldwCli()

def main_cli_runner():
    """Entry point for the tldw-chatbook command.

    This function is referenced in pyproject.toml as the entry point for the tldw-chatbook command.
    It initializes logging early and then runs the TldwCli app.
    """
    # Configure logging to suppress verbose debug messages early
    import logging
    import os
    import warnings
    
    # Suppress various verbose loggers
    logging.getLogger("torio._extension.utils").setLevel(logging.WARNING)
    logging.getLogger("torio").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    
    # Suppress torchaudio and FFmpeg warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    warnings.filterwarnings("ignore", message=".*FFmpeg.*")
    
    # Set environment variable to suppress FFmpeg output
    os.environ["TORCHAUDIO_LOG_LEVEL"] = "ERROR"
    
    # Set up signal handlers for clean exit
    import signal
    import os
    import atexit
    
    def force_cleanup():
        """Force cleanup on exit"""
        import threading
        import concurrent.futures
        
        # Force kill any Higgs-related threads first
        for thread in threading.enumerate():
            thread_name = thread.name.lower()
            if any(name in thread_name for name in ['higgs', 'boson', 'serve_engine', 'audio']):
                loguru_logger.warning(f"Force killing thread: {thread.name}")
                try:
                    # Mark as daemon to not block exit
                    thread.daemon = True
                except Exception:
                    pass
        
        # Force daemon all threads
        for thread in threading.enumerate():
            if thread != threading.main_thread() and not thread.daemon:
                try:
                    thread.daemon = True
                except Exception:
                    pass
        
        # Clear thread pool queues
        try:
            concurrent.futures.thread._threads_queues.clear()
        except Exception:
            pass
        
        # Force clear any PyTorch CUDA resources
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
    
    # Register cleanup
    atexit.register(force_cleanup)
    
    def signal_handler(signum, frame):
        loguru_logger.info(f"Received signal {signum}, forcing clean exit")
        force_cleanup()
        os._exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
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
        css_dir = Path(__file__).parent / "css"
        css_dir.mkdir(exist_ok=True)
        
        # Check if modular CSS needs to be built
        modular_css_path = css_dir / "tldw_cli_modular.tcss"
        build_script_path = css_dir / "build_css.py"
        
        # Check if any module is newer than the built file
        should_rebuild = False
        if not modular_css_path.exists():
            should_rebuild = True
            logging.info("Modular CSS file not found, will build it")
        elif build_script_path.exists():
            # Check if any module file is newer than the built file
            modular_mtime = modular_css_path.stat().st_mtime
            for subdir in ['core', 'layout', 'components', 'features', 'utilities']:
                subdir_path = css_dir / subdir
                if subdir_path.exists():
                    for css_file in subdir_path.glob('*.tcss'):
                        if css_file.stat().st_mtime > modular_mtime:
                            should_rebuild = True
                            logging.info(f"Module {css_file.name} is newer than built CSS, rebuilding")
                            break
                if should_rebuild:
                    break
        
        if should_rebuild and build_script_path.exists():
            logging.info("Building modular CSS...")
            import subprocess
            # Build CSS synchronously before starting the app
            result = subprocess.run([sys.executable, str(build_script_path)], 
                                  cwd=str(css_dir), 
                                  capture_output=True, 
                                  text=True)
            if result.returncode == 0:
                logging.info("Successfully built modular CSS")
            else:
                logging.error(f"Failed to build modular CSS: {result.stderr}")
        
    except Exception as e_css_main:
        logging.error(f"Error handling CSS file: {e_css_main}", exc_info=True)

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="tldw chatbook - A Textual TUI for chatting with LLMs",
        prog="tldw-cli"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run the application as a web server"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host address for web server (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port for web server (default: 8000)"
    )
    parser.add_argument(
        "--web-title",
        type=str,
        help="Title for the web page"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for web server"
    )
    
    args = parser.parse_args()
    
    # If --serve flag is provided, run as web server
    if args.serve:
        # Check if web server dependencies are available
        from .Web_Server.serve import check_web_server_available, run_web_server
        if not check_web_server_available():
            loguru_logger.error("\n" + "="*60)
            loguru_logger.error("Web server feature is not available!")
            loguru_logger.error("="*60)
            loguru_logger.error("\nThe required dependency 'textual-serve' is not installed.")
            loguru_logger.error("\nTo install it, run:")
            loguru_logger.error("  pip install tldw_chatbook[web]")
            loguru_logger.error("\nFor development installations:")
            loguru_logger.error("  pip install -e \".[web]\"")
            loguru_logger.error("\n" + "="*60 + "\n")
            return

        loguru_logger.info("Starting tldw_chatbook in web server mode")
        run_web_server(
            host=args.host,
            port=args.port,
            title=args.web_title,
            debug=args.debug
        )
        return  # Exit after web server stops
    
    # Otherwise, run as normal TUI app
    # Create instance with early logging flag
    app_instance = TldwCli()
    # Set the early logging flag so _setup_logging knows logging was already initialized
    app_instance._early_logging_initialized = True
    try:
        app_instance.run()
    except KeyboardInterrupt:
        loguru_logger.info("--- KeyboardInterrupt received ---")
        # Force cleanup inline
        import threading
        import concurrent.futures
        for thread in threading.enumerate():
            if thread != threading.main_thread() and not thread.daemon:
                try:
                    thread.daemon = True
                except Exception:
                    pass
        try:
            concurrent.futures.thread._threads_queues.clear()
        except Exception:
            pass
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
