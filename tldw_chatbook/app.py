# tldw_cli - Textual CLI for LLMs
# Description: This file contains the main application logic for the tldw_cli, a Textual-based CLI for interacting with various LLM APIs.
#
# Disable progress bars early to prevent interference with TUI
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable Textual logging in production
# Set to a path to enable logging for debugging: os.environ['TEXTUAL_LOG'] = '/tmp/textual.log'
if "TEXTUAL_LOG" not in os.environ:
    os.environ["TEXTUAL_LOG"] = ""  # Empty string disables logging

# (task-2016) Spawn-pool workers re-import this module as ``__mp_main__``
# with an inherited REAL-TTY stderr (see ``_create_ingest_parse_pool``'s
# Textual-stderr workaround), so import-time noise from the chain below --
# loguru's default stderr sink ("python-frontmatter not installed…"),
# ``RequestsDependencyWarning`` -- painted raw text over the parent's TUI
# on every first submit. This guard MUST run before the heavy imports:
# the noise is emitted while they import. The pool ``initializer``
# (``silence_ingest_worker_import_noise``) still runs afterwards as a
# belt for post-import noise.
import multiprocessing as _early_multiprocessing

# ``__mp_main__`` is the name spawn gives this module while re-importing it
# in a child; ``parent_process()`` alone is NOT yet populated at that point
# (live-verified: the flood survived a parent_process()-only guard).
if __name__ == "__mp_main__" or _early_multiprocessing.parent_process() is not None:
    import logging as _early_logging
    import warnings as _early_warnings

    _early_warnings.simplefilter("ignore")
    # (task-2041) A bare ``logging.warning()`` on a handler-less root
    # logger auto-basicConfigs a stderr StreamHandler
    # ("WARNING:root:OpenTelemetry not installed…" painted over the TUI).
    # A NullHandler makes root non-empty, so neither auto-basicConfig nor
    # lastResort fires.
    _early_logging.getLogger().addHandler(_early_logging.NullHandler())
    try:
        from loguru import logger as _early_worker_logger

        _early_worker_logger.remove()
    except Exception:
        pass

# Imports
import argparse
import concurrent.futures
import contextlib
import functools
import hashlib
import inspect
import logging
import logging.handlers
import multiprocessing
import multiprocessing.connection
import queue
import random
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import traceback
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Any, Dict, List, Callable, Iterable, Mapping
from textual.widget import Widget

#
# 3rd-Party Libraries
import asyncio
from loguru import logger as loguru_logger, logger
from rich.markup import escape as escape_markup
from textual import on, work
from textual.app import App, ComposeResult, ScreenStackError
from textual.widgets import RichLog, Markdown
from textual.containers import Container
from textual.reactive import reactive
from textual.worker import Worker, WorkerCancelled, WorkerState
from textual.binding import Binding
from textual.timer import Timer
from textual.css.query import NoMatches, QueryError
from textual.command import Hit, Hits, Provider
from functools import partial
from pathlib import Path, PurePath

from tldw_chatbook.css import build_css, widget_css
from tldw_chatbook.css.tie_aware_stylesheet import TieAwareStylesheet
from tldw_chatbook.css.Themes.themes import ALL_THEMES

# from tldw_chatbook.css.css_loader import load_modular_css  # Removed - reverting to original CSS
from tldw_chatbook.Metrics.metrics import (
    log_histogram,
    log_counter,
    log_resource_usage,
    init_metrics_server,
)
from tldw_chatbook.Metrics.Otel_Metrics import init_metrics as init_otel_metrics

#
# --- Local API library Imports ---
from .config import (
    get_cli_setting,
    first_profile_created_this_session,
    get_library_collections_db_path,
    get_library_ingest_jobs_db_path,
    get_media_db_path,
    get_prompts_db_path,
    get_notifications_db_path,
    get_notes_sync_state_db_path,
    get_notes_sync_recovery_capacity_bytes,
    get_notes_sync_watcher_intervals,
    get_research_db_path,
    get_scheduled_tasks_db_path,
    get_subscriptions_db_path,
    get_tts_profiles_db_path,
    get_user_data_dir,
    get_workspaces_db_path,
    get_writing_db_path,
)
from .Logging_Config import configure_application_logging
from tldw_chatbook.Utils.instance_lock import (
    InstanceLockStatus,
    acquire_profile_instance_lock,
)
from tldw_chatbook.Constants import (
    ALL_TABS,
    TAB_CCP,
    TAB_CHAT,
    TAB_HOME,
    TAB_LOGS,
    TAB_STATS,
    TAB_TOOLS_SETTINGS,
    TAB_INGEST,
    TAB_LLM,
    TAB_MEDIA,
    TAB_SEARCH,
    TAB_EVALS,
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
    TAB_STTS,
    TAB_STUDY,
    TAB_WRITING,
    TAB_RESEARCH,
    TAB_CHATBOOKS,
    LIBRARY_NAV_CONTEXT_MODE,
    LIBRARY_NAV_CONTEXT_NOTES_CREATE,
    LIBRARY_NAV_CONTEXT_INGEST,
    WATCHLISTS_NAV_CONTEXT_BACKEND,
    WATCHLISTS_NAV_CONTEXT_RUN_ID,
    WATCHLISTS_NAV_CONTEXT_SECTION,
    WATCHLISTS_SECTION_RUNS,
    get_tab_display_label,
)
from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.citation_artifact_ownership import (
    CitationArtifactOwnershipCoordinator,
)
from tldw_chatbook.Chat.citation_service_factory import (
    build_local_citation_conversation_service,
)
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_live_work import (
    ConsoleLiveWorkLaunch,
    resolve_console_live_work_primary_action,
)
from tldw_chatbook.Chat.console_image_edit_operations import (
    ImageEditOperationRegistry,
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime, dispose_console_runtime
from tldw_chatbook.Chat.server_chat_conversation_service import (
    ServerChatConversationService,
)
from tldw_chatbook.DB.Client_Media_DB_v2 import (
    DatabaseError as MediaDatabaseError,
    InputError as MediaInputError,
    MediaDatabase,
)
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.config import CLI_APP_CLIENT_ID
from tldw_chatbook.Chatbooks import LocalChatbookService, ServerChatbookService
from tldw_chatbook.Library import LocalLibraryCollectionsService
from tldw_chatbook.Library.ingest_analysis import resolve_ingest_analysis_provider
from tldw_chatbook.Library.ingest_capabilities import (
    field_gate_open,
    generic_option_default,
    get_type_group,
)
from tldw_chatbook.Library.ingest_preflight import collect_directory_files
from tldw_chatbook.Library.server_ingest_reconcile import (
    pending_remote_batches,
    reconcile_remote_ingest_jobs,
)
from tldw_chatbook.Library.server_ingest_request import (
    ServerIngestUnsupported,
    build_server_ingest_kwargs,
)
from tldw_chatbook.Library.web_clip_request import (
    NotAWebClipSource,
    build_web_clip_kwargs,
    clip_failure_reason,
    is_web_clip_source,
)
from tldw_chatbook.Library.library_ingest_jobs import (
    ActiveIngestConsentScope,
    ActiveIngestJobRef,
    ActiveIngestSubmissionRefused,
    DEFAULT_CHUNK_SIZE,
    INGEST_DUPLICATE_PROGRESS_PREFIX,
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
    build_active_ingest_consent_scope,
    normalize_active_ingest_source,
)
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)
from tldw_chatbook.Local_Ingestion import FileIngestionError
from tldw_chatbook.Local_Ingestion.ingest_parse_worker import (
    classify_parse_failure,
    initialize_ingest_parse_worker,
    run_parse_job,
)
from tldw_chatbook.Local_Ingestion.ingest_parse_progress import (
    INGEST_PARSE_PROGRESS_FLUSH_SECONDS,
    INGEST_PARSE_PROGRESS_QUEUE_MAXSIZE,
    ParseProgressCoalescer,
    ParseProgressEvent,
    make_parse_progress_event,
)
from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    classify_ingest_source,
    persist_parsed_media,
)
from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
    BatchSTTRoutingError,
    PARAKEET_V2_MODEL,
    resolve_batch_stt_route,
)
from tldw_chatbook.STT.contracts import (
    TRANSCRIPTION_FAILURE_CONTRACT,
    ExecutionDevice,
    FileAudioSource,
    TranscriptionFailureCode,
)
from tldw_chatbook.STT.executor import (
    ExecutorBusyError,
    ExecutorEvent,
    ExecutorFailure,
    ExecutorResult,
    ExecutorUnavailableError,
    LocalSTTExecutor,
    ModelIdentity,
    WorkerPhase,
    snapshot_local_source,
)
from tldw_chatbook.STT.dispatch_coordinator import LocalSTTDispatchCoordinator
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
from tldw_chatbook.Utils.Emoji_Handling import (
    get_char,
    EMOJI_TITLE_BRAIN,
    FALLBACK_TITLE_BRAIN,
    supports_emoji,
)
from tldw_chatbook.Utils.app_shutdown import (
    arm_exit_watchdog,
    install_termination_handlers,
    register_running_app,
    unregister_running_app,
)
from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor
from tldw_chatbook.Utils.db_status_manager import DBStatusManager
from tldw_chatbook.Utils.persistent_diagnostics import persist_event
from tldw_chatbook.Utils.text_selection_crash_guard import TextSelectionCrashGuard
from tldw_chatbook.TTS import TTSProfileService
from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
    AudioCppArtifactLeaseCoordinator,
    AudioCppArtifactRemovalEvidence,
    AudioCppModelLibraryObservationSnapshot,
    AudioCppManagedConsumerIdentity,
    project_audio_cpp_artifact_removal_evidence,
)
from tldw_chatbook.TTS.audio_cpp_guided_config import (
    AudioCppSettingsConfig,
    project_audio_cpp_settings_config,
)
from tldw_chatbook.TTS.adapter_bootstrap import build_default_tts_service
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.TTS.profile_types import ProfileRepositoryState
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.voice_bundle_service import (
    TTSVoiceBundlePortabilityService,
)
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
    SpeechTTSPanelDraftSnapshot,
)
from tldw_chatbook.TTS._async_lifecycle import join_retained_task
from tldw_chatbook.Model_Artifacts.store import managed_service
from tldw_chatbook.TTS.TTS_Generation import (
    bind_tts_service,
    close_tts_resources,
)
from tldw_chatbook.Event_Handlers.worker_handlers import (
    WorkerHandlerRegistry,
    MiscWorkerHandler,
)
from .config import (
    get_cli_config_path,
    load_settings,
    get_cli_providers_and_models,
    API_MODELS_BY_PROVIDER,
    LOCAL_PROVIDERS,
    load_cli_config_and_ensure_existence,
    persist_cli_config_for_shutdown,
    set_encryption_password,
    get_config_load_failure,
)
from .Event_Handlers import worker_events
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSGlobalOverrideDecisionEvent,
    TTSMessageSpeechRequestEvent,
    TTSRequestEvent,
    TTSCompleteEvent,
    TTSPlaybackEvent,
    TTSProgressEvent,
    TTSEventHandler,
)
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSPlaygroundGenerateEvent,
    STTSProviderConfigurationChanged,
    STTSSettingsSaveEvent,
    STTSAudioBookGenerateEvent,
)
from .Notes.Notes_Library import NotesInteropService
from .Notes.file_notes_git_service import build_file_notes_session_owner
from .Notes.note_folder_repository import LocalNoteFolderRepository
from .Notes.notes_scope_service import NotesScopeService, ScopeType
from .Notes.notes_sync_legacy import legacy_sync_directory_configured
from .Notes.notes_sync_runtime import (
    build_notes_sync_legacy_migrator,
    build_notes_sync_runtime_owner,
)
from .Notes.server_notes_workspace_service import ServerNotesWorkspaceService
from .Character_Chat.character_persona_scope_service import CharacterPersonaScopeService
from .Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService
from .Character_Chat.local_character_persona_service import LocalCharacterPersonaService
from .Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from .Character_Chat.server_chat_dictionary_service import ServerChatDictionaryService
from .Character_Chat.server_character_persona_service import (
    ServerCharacterPersonaService,
)
from .Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from .Actor_Packs.creation import ActorPackCreationService
from .Actor_Packs.controller import ActorPackExportController
from .Actor_Packs.export import ActorPackExportService
from .Actor_Packs.repository import ActorPackRepository
# Persona_Buddy is deliberately NOT imported at module scope (TASK-21103):
# its controller drags Persona_Visual and PIL (1.28 s cold) onto the boot
# path. See the lazy persona_buddy_controller property.
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
from .Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from .Scheduling.constants import (
    HANDLER_TIMEOUT_SECONDS,
    MISSED_FIRE_GRACE_SECONDS,
    SCHEDULER_POLL_INTERVAL_SECONDS,
)
from .Scheduling.services.scheduling_service import SchedulingService
from .Scheduling.services.server_client import SchedulingServerClient
from .Scheduling.scheduler.loop import SchedulerLoop, Handler
from .Scheduling.scheduler.handlers.reminder_handler import ReminderHandler
from .Scheduling.scheduler.handlers.watchlist_check_handler import WatchlistCheckHandler
from .Scheduling.scheduler.handlers.briefing_handler import BriefingJobHandler
from .Scheduling.services.watchlist_projection import WatchlistProjection
from .Scheduling.services.briefing_projection import BriefingProjection
from .ACP_Interop.runtime_process import ACPRuntimeProcessManager
from .ACP_Interop.runtime_session import ACPRuntimeSessionState
from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage

# chat_message_enhanced is deliberately NOT imported at module scope
# (TASK-21103): it pulls PIL and the textual_image package at import time.
# The two TTS event handlers that query it import it function-locally.
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.glyph_fallback import set_ascii_glyph_mode
from .Widgets.AppFooterStatus import AppFooterStatus
from .Widgets.splash_screen import SplashScreen
from .LLM_Calls.LLM_API_Calls import (
    chat_with_openai,
    chat_with_anthropic,
    chat_with_cohere,
    chat_with_groq,
    chat_with_openrouter,
    chat_with_huggingface,
    chat_with_deepseek,
    chat_with_mistral,
    chat_with_google,
)
from .LLM_Calls.LLM_API_Calls_Local import (
    chat_with_llama,
    chat_with_kobold,
    chat_with_oobabooga,
    chat_with_vllm,
    chat_with_tabbyapi,
    chat_with_aphrodite,
    chat_with_ollama,
    chat_with_custom_openai,
    chat_with_custom_openai_2,
    chat_with_local_llm,
)
from tldw_chatbook.config import (
    get_chachanotes_db_path,
    settings,
    get_chachanotes_db_lazy,
    seed_builtin_content,
)
from .UI.Navigation.main_navigation import MainNavigationBar, NavigateToScreen
from .UI.Navigation.audio_cpp_model_handoff import AudioCppModelInstallOwner
from .UI.Navigation.pending_handoff_store import (
    ConsoleProviderIntent,
    HandoffChannel,
    HandoffValueError,
    PendingHandoffStore,
)
from .UI.Navigation.screen_state_store import (
    ConsolePromptTargetProjection,
    RuntimeIdentity,
    ScreenStateStore,
)
from .UI.Navigation.screen_registry import (
    ScreenRoute,
    registered_screen_aliases,
    registered_screen_routes,
    resolve_screen_target,
    screen_load_error,
)
from .UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER
from .UI.Workbench.help import WorkbenchHelpPanel, WorkbenchHelpState
from .UI.Screens.study_scope_models import StudyScopeContext
from .UI.stable_command_palette import StableCommandPalette
from .Prompt_Management.prompt_variables import PromptVariableApplication

from .UI.Tools_Settings_Window import ToolsSettingsWindow  # noqa: E402
from .UI.console_command_provider import ConsoleCommandProvider  # noqa: E402
from .UI.image_gen_command_provider import ImageGenCommandProvider  # noqa: E402
from tldw_chatbook.Chat_Grammars_Interop import (  # noqa: E402
    ChatGrammarsScopeService,
    LocalChatGrammarsService,
    ServerChatGrammarsService,
)
from tldw_chatbook.Claims_Interop import ClaimsScopeService, ServerClaimsService  # noqa: E402
from tldw_chatbook.Companion_Interop import (  # noqa: E402
    CompanionScopeService,
    ServerCompanionService,
)
from tldw_chatbook.Collections_Interop import (  # noqa: E402
    CollectionsFeedsScopeService,
    ServerCollectionsFeedsService,
)
from tldw_chatbook.External_Connectors_Interop import (  # noqa: E402
    ConnectorsScopeService,
    ServerConnectorsService,
)
from tldw_chatbook.Feedback_Interop import (  # noqa: E402
    FeedbackScopeService,
    LocalFeedbackService,
    ServerFeedbackService,
)
from tldw_chatbook.Kanban_Interop import (  # noqa: E402
    KanbanScopeService,
    LocalKanbanService,
    ServerKanbanService,
)
from tldw_chatbook.LLM_Provider_Catalog import (  # noqa: E402
    LLMProviderCatalogScopeService,
    LocalLLMProviderCatalogService,
    ServerLLMProviderCatalogService,
)
from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import ModelCatalogRefreshed  # noqa: E402
from tldw_chatbook.Media import (  # noqa: E402
    LocalMediaReadingService,
    MediaReadingScopeService,
    ServerMediaReadingService,
)
from tldw_chatbook.Meetings_Interop import MeetingsScopeService, ServerMeetingsService  # noqa: E402
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService  # noqa: E402
from tldw_chatbook.MCP.local_store import LocalMCPStore  # noqa: E402
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore  # noqa: E402
from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService  # noqa: E402
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore  # noqa: E402
from tldw_chatbook.MCP.unified_control_plane_service import (  # noqa: E402
    UnifiedMCPControlPlaneService,
)
from tldw_chatbook.Notifications import (  # noqa: E402
    ClientNotificationsDB,
    ClientNotificationsService,
    EventStateRepository,
    NotificationsScopeService,
    NotificationDispatchService,
    ServerNotificationsService,
)
from tldw_chatbook.Outputs_Interop import OutputsScopeService, ServerOutputsService  # noqa: E402
from tldw_chatbook.Personalization_Interop import (  # noqa: E402
    PersonalizationScopeService,
    ServerPersonalizationService,
)
from tldw_chatbook.Prompt_Management.prompt_scope_service import (  # noqa: E402
    build_prompt_scope_service,
)
from tldw_chatbook.Prompt_Studio_Interop import (  # noqa: E402
    PromptStudioScopeService,
    ServerPromptStudioService,
)
from tldw_chatbook.Research_Interop import (  # noqa: E402
    LocalResearchSearchService,
    ResearchSearchScopeService,
    ServerResearchSearchService,
)
from tldw_chatbook.Server_Runtime_Interop import (  # noqa: E402
    ServerRuntimeScopeService,
    ServerRuntimeService,
)
from tldw_chatbook.Sharing_Interop import ServerSharingService, SharingScopeService  # noqa: E402
from tldw_chatbook.Skills_Interop import (  # noqa: E402
    LocalSkillsService,
    ServerSkillsService,
    SkillTrustService,
    SkillsScopeService,
    default_local_skills_store_dir,
)
from tldw_chatbook.Skills_Interop.skill_trust_store import (  # noqa: E402
    MARKER_FILENAME as _SKILL_TRUST_MARKER_FILENAME,
    SkillTrustStore,
    build_default_skill_trust_key_cache,
    build_skill_trust_marker_store_with_fallback,
    default_trust_store_dir,
    skill_trust_account_scope,
)
from tldw_chatbook.Sync_Interop import (  # noqa: E402
    LocalFirstSyncService,
    ManualSyncControlService,
    ServerSyncService,
    SyncScopeService,
    SyncStateRepository,
)
from tldw_chatbook.Text2SQL_Interop import ServerText2SQLService, Text2SQLScopeService  # noqa: E402
from tldw_chatbook.Tools_Interop import ServerToolsService, ToolsScopeService  # noqa: E402
from tldw_chatbook.MCP_Governance_Interop import (  # noqa: E402
    MCPGovernanceScopeService,
    ServerMCPGovernanceService,
)
from tldw_chatbook.User_Governance_Interop import (  # noqa: E402
    ServerUserGovernanceService,
    UserGovernanceScopeService,
)
from tldw_chatbook.Web_Clipper_Interop import (  # noqa: E402
    ServerWebClipperService,
    WebClipperScopeService,
)
from tldw_chatbook.Web_Scraping_Interop import (  # noqa: E402
    ServerWebScrapingService,
    WebScrapingScopeService,
)
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService  # noqa: E402
from tldw_chatbook.Subscriptions import (  # noqa: E402
    LocalWatchlistsService,
    ServerWatchlistsService,
    WatchlistScopeService,
)
from tldw_chatbook.Subscriptions.fts_backfill import (  # noqa: E402
    FTSBackfillError,
    backfill_subscription_items_fts,
)
from tldw_chatbook.Subscriptions.watchlist_bundle_service import (  # noqa: E402
    WatchlistBundleService,
)
from tldw_chatbook.Translation_Interop import (  # noqa: E402
    ServerTranslationService,
    TranslationScopeService,
)
from tldw_chatbook.Voice_Assistant_Interop import (  # noqa: E402
    ServerVoiceAssistantService,
    VoiceAssistantScopeService,
)
from tldw_chatbook.Evaluations_Interop import (  # noqa: E402
    EvaluationScopeService,
    LocalEvaluationsService,
    ServerEvaluationsService,
)
from tldw_chatbook.runtime_policy.bootstrap import (  # noqa: E402
    build_runtime_api_client,
    load_runtime_policy_for_app,
    set_authoritative_runtime_source,
)
from tldw_chatbook.runtime_policy.server_capabilities import (  # noqa: E402
    ActiveServerCapabilityService,
)
from tldw_chatbook.runtime_policy.server_context import RuntimeServerContextProvider  # noqa: E402
from tldw_chatbook.runtime_policy.server_credentials import (  # noqa: E402
    CredentialStoreUnavailable,
    UnavailableServerCredentialStore,
    build_default_server_credential_store,
)
from tldw_chatbook.runtime_policy.server_event_scope import (  # noqa: E402
    event_principal_id_from_active_context,
)
from tldw_chatbook.runtime_policy.server_parity_state import (  # noqa: E402
    ServerParityStateRepositories,
    build_server_parity_state_repositories,
)
from tldw_chatbook.runtime_policy.engine import PolicyEngine  # noqa: E402
from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer  # noqa: E402
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY  # noqa: E402
from tldw_chatbook.runtime_policy.types import PolicyDecision, RuntimeSourceState  # noqa: E402
from tldw_chatbook.Auth_Account_Interop import (  # noqa: E402
    AuthAccountScopeService,
    ServerAuthAccountService,
)
from tldw_chatbook.Audio_Services_Interop import (  # noqa: E402
    AudioServicesScopeService,
    LocalAudioServicesService,
    ServerAudioServicesService,
)
from .Evals.eval_orchestrator import EvaluationOrchestrator  # noqa: E402

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
        ModelCatalogDiskStore,
    )
    from tldw_chatbook.tldw_api import MCPUnifiedClient

API_IMPORTS_SUCCESSFUL = True

DEFERRED_AUDIO_SERVICE_DELAY_SECONDS = 0.1
DEFERRED_DB_SIZE_UPDATE_DELAY_SECONDS = 0.1
DEFERRED_MEDIA_CLEANUP_DELAY_SECONDS = 5.0

# task-19561: how long a cancelled worker gets to settle before shutdown
# stops waiting on it and says so. Replaces a flat `asyncio.sleep(0.1)` that
# waited on nothing in particular. Sized to be invisible on a quiet exit
# (the wait ends the moment the last worker finishes) while still bounding a
# thread worker that will not notice cancellation at all.
WORKER_CANCELLATION_GRACE_SECONDS = 3.0

# task-15472: after first paint, warm the lazy screen-module import cache from
# a background thread so the FIRST click to each tab doesn't pay for a
# synchronous, UI-thread `import_module` inside the FIFO-locked navigation
# worker (`UI/Navigation/screen_registry.py`'s `load_screen_class`) --
# chat_screen.py is ~20k lines, library_screen.py ~26k, settings_screen.py
# ~19k (Docs/Design/2026-08-11-input-latency-audit.md). Scheduled slightly
# after the other 0.1s deferred-startup timers (footer status, audio
# services) so it is strictly the lowest-priority background task: nothing
# depends on it finishing, it only warms a cache.
DEFERRED_SCREEN_PREIMPORT_DELAY_SECONDS = 0.2

# Chat/Library/Settings are the three screens the audit measured as
# multi-thousand-line modules -- import them first so a thread that gets cut
# short (app quit shortly after startup) still banked the highest-value work
# before spending time on the rest of the registry.
SCREEN_PREIMPORT_PRIORITY_ROUTE_IDS: tuple[str, ...] = ("chat", "library", "settings")

# TASK-1240. The `component` this module passes to `persist_event`. It is a
# bounded metadata token (`persist_event` raises `ValueError` otherwise) and is
# used raw to build the diagnostics logger name, so the four emit sites in this
# file must agree on one spelling. Private to `app.py`: every event emitted
# here belongs to the application lifecycle.
_DIAGNOSTICS_COMPONENT_APP = "app"
# Task-4 review round 2: `_offer_tts_global_override`'s confirmation dialog
# must name which configured-voice domain actually failed -- a per-character
# assignment, or the app-wide default voice profile (slice 3, task 4) --
# since the user is consenting to hear a different voice than the one they
# configured. Keyed by `CharacterTTSResolutionError.domain` /
# `TTSEventHandler.peek_global_override_voice_domain`'s bounded return
# value; `None` (unknown/expired token, or no handler bound) falls back to
# the domain-neutral entry below, which stays accurate for both without
# being vaguer than either domain's own precise copy.
_TTS_GLOBAL_OVERRIDE_PROMPT_COPY: dict[str | None, str] = {
    "character": (
        "The assigned character voice could not be resolved. "
        "Use the current global TTS voice for this message?"
    ),
    "default_profile": (
        "Your default voice profile could not be used. "
        "Use the current global TTS voice for this message?"
    ),
    None: (
        "Your configured voice could not be used for this message. "
        "Use the current global TTS voice instead?"
    ),
}
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
        "Google": chat_with_google,  # Key from config
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
        "local-llm": chat_with_local_llm,
    }
    logging.info(f"API_FUNCTION_MAP populated with {len(API_FUNCTION_MAP)} entries.")
else:
    API_FUNCTION_MAP = {}
    logging.error("API_FUNCTION_MAP is empty due to import failures.")

ALL_API_MODELS = {
    **API_MODELS_BY_PROVIDER,
    **LOCAL_PROVIDERS,
}  # If needed for sidebar defaults
AVAILABLE_PROVIDERS = list(ALL_API_MODELS.keys())  # If needed
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
                help="Open theme selection menu",
            )

        # Only show individual themes if user is specifically searching for theme-related terms
        if any(
            term in query.lower()
            for term in [
                "switch",
                "theme",
                "dark",
                "light",
                "color",
                "solarized",
                "gruvbox",
                "dracula",
            ]
        ):
            # Get available theme names from registered themes
            available_themes = ["textual-dark", "textual-light"]  # Built-in themes
            # Add custom themes from ALL_THEMES
            for theme in ALL_THEMES:
                theme_name = theme.name if hasattr(theme, "name") else str(theme)
                available_themes.append(theme_name)

            for theme_name in available_themes:
                command_text = f"Theme: Switch to {theme_name.replace('_', ' ').replace('-', ' ').title()}"
                score = matcher.match(command_text)
                if score > 0:
                    yield Hit(
                        score * 0.9,  # Slightly lower priority than main command
                        matcher.highlight(command_text),
                        partial(self.switch_theme, theme_name),
                        help=f"Change theme to {theme_name}",
                    )

    async def discover(self) -> Hits:
        """Show only the main theme command when palette is first opened."""
        yield Hit(
            1.0,
            "Theme: Change Theme",
            partial(self.show_theme_submenu),
            help="Open theme selection menu",
        )

    def show_theme_submenu(self) -> None:
        """Show a notification with instruction to search for themes."""
        self.app.notify(
            "Type 'theme' in the command palette to see all available themes",
            severity="information",
        )

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


def _bindings_to_shortcuts(bindings: Any) -> tuple[tuple[str, str], ...]:
    """Flatten BINDINGS entries into (key, description) pairs for help display.

    Accepts both Binding objects and the legacy tuple form so any screen's
    BINDINGS can be rendered as truthful shortcut help.
    """
    pairs: list[tuple[str, str]] = []
    for entry in bindings or ():
        if isinstance(entry, Binding):
            pairs.append((entry.key, entry.description))
        elif isinstance(entry, (tuple, list)) and entry:
            key = str(entry[0])
            description = str(entry[2]) if len(entry) > 2 else ""
            pairs.append((key, description))
    return tuple(pairs)


class TabNavigationProvider(Provider):
    """Provider for tab navigation commands."""

    TAB_HELP_TEXT = {
        TAB_HOME: "Open Home for notifications, status, and next-best actions",
        TAB_CHAT: "Open Console for live agent work, approvals, tools, and RAG",
        TAB_LIBRARY: "Open Library for source material, imports, notes, media, conversations, and Search/RAG",
        TAB_ARTIFACTS: "Open Artifacts for generated outputs, reports, datasets, and Chatbooks",
        TAB_PERSONAS: "Open Roleplay for characters, personas, dictionaries, and behavior profiles",
        TAB_WATCHLISTS_COLLECTIONS: "Open Watchlists for monitored sources, runs, alerts, and recovery",
        TAB_SCHEDULES: "Open Schedules for run timing, triggers, pauses, retries, and recovery",
        TAB_WORKFLOWS: "Open Workflows for reusable procedures, dry-runs, and outputs",
        TAB_MCP: "Open MCP for servers, tools, permissions, auth, and audit",
        TAB_ACP: "Open ACP for agents, sessions, runtimes, diffs, and terminals",
        TAB_SKILLS: "Open Skills for Agent Skills discovery, validation, and attachments",
        TAB_SETTINGS: "Open global preferences, appearance, storage, and app behavior",
        TAB_CCP: "Switch to Roleplay for characters, personas, dictionaries, and world books",
        TAB_MEDIA: "Switch to media library",
        TAB_SEARCH: "Switch to Library search and RAG",
        TAB_INGEST: "Switch to content ingestion",
        TAB_EVALS: "Switch to evaluation tools",
        TAB_LLM: "Switch to model and provider management",
        TAB_STTS: "Switch to speech-to-text and text-to-speech tools",
        TAB_STUDY: "Switch to flashcards and quizzes",
        TAB_WRITING: "Switch to writing tools",
        TAB_RESEARCH: "Switch to research workflows",
        TAB_CHATBOOKS: "Switch to portable Chatbook context packs",
        TAB_TOOLS_SETTINGS: "Open MCP for legacy tools and settings",
        TAB_LOGS: "Switch to application logs",
        TAB_STATS: "Switch to statistics view",
    }

    NAVIGATION_TABS = tuple(
        destination.primary_route for destination in SHELL_DESTINATION_ORDER
    )

    POPULAR_TABS = (
        TAB_HOME,
        TAB_CHAT,
        TAB_LIBRARY,
        TAB_ARTIFACTS,
        TAB_MCP,
        TAB_SETTINGS,
    )

    # task-423: labeled deep-link commands into Library-folded content
    # types that would otherwise only fuzzy-match the generic Library
    # command (which lands on generic Library, not the content row). Each
    # entry is (legacy route, command text, help text); the route rides
    # ``_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT`` to land on its rail row.
    LIBRARY_SUBROUTE_COMMANDS: tuple[tuple[str, str, str], ...] = (
        (
            "skills",
            "Tab Navigation: Library — Skills",
            "Open Library's Skills row for Agent Skills packs, validation, and trust",
        ),
    )

    def __init__(self, screen, *args, **kwargs):
        """Initialize the TabNavigationProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    @classmethod
    def navigation_tab_ids(cls) -> tuple[str, ...]:
        return cls.NAVIGATION_TABS

    @classmethod
    def command_palette_tab_ids(cls) -> tuple[str, ...]:
        # One palette entry per shell destination. Legacy route ids (media,
        # search, ccp, tools_settings, llm_management, stts, evals, coding,
        # logs, stats, writing, research, ...) are no longer separate labeled
        # commands; they are alias terms on their owning destination's single
        # command (see search()).
        return cls.NAVIGATION_TABS

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
        from .UI.Navigation.shell_destinations import (
            get_shell_destination,
            resolve_shell_route,
        )

        resolved = resolve_shell_route(cls.route_for_tab(tab_id))
        try:
            return get_shell_destination(resolved.destination_id)
        except KeyError:
            return None

    @classmethod
    def _destination_alias_terms(cls, destination) -> tuple[str, ...]:
        """Searchable legacy route names that resolve to ``destination``."""
        terms = {
            destination.destination_id,
            destination.label,
            destination.primary_route,
        }
        if destination.full_label:
            terms.add(destination.full_label)
        for legacy_route in destination.legacy_routes:
            terms.add(legacy_route)
            terms.add(get_tab_display_label(legacy_route))
        return tuple(sorted(term for term in terms if term))

    @classmethod
    def _shell_help_text(cls, tab_id: str) -> str | None:
        destination = cls._shell_destination_for_tab(tab_id)
        if destination is None:
            return None
        return f"Open {destination.accessible_label} for {destination.purpose}"

    def _tab_command(self, tab_id: str) -> tuple[str, str, str]:
        destination = self._shell_destination_for_tab(tab_id)
        label = (
            destination.accessible_label
            if destination is not None
            else get_tab_display_label(tab_id)
        )
        help_text = self._shell_help_text(tab_id) or self.TAB_HELP_TEXT.get(
            tab_id, f"Switch to {label}"
        )
        return f"Tab Navigation: Switch to {label}", tab_id, help_text

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        tab_commands = [
            self._tab_command(tab_id) for tab_id in self.command_palette_tab_ids()
        ]

        for command_text, tab_id, help_text in tab_commands:
            destination = self._shell_destination_for_tab(tab_id)
            alias_terms = (
                self._destination_alias_terms(destination)
                if destination is not None
                else ()
            )
            score = max(
                matcher.match(command_text),
                matcher.match(help_text),
                *(matcher.match(term) for term in alias_terms),
            )
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.switch_tab, tab_id),
                    help=help_text,
                )

        # task-423: Library sub-route deep links (e.g. "skills").
        for route, command_text, help_text in self.LIBRARY_SUBROUTE_COMMANDS:
            score = max(
                matcher.match(command_text),
                matcher.match(help_text),
                matcher.match(route),
            )
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.switch_tab, route),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_tabs = [self._tab_command(tab_id) for tab_id in self.POPULAR_TABS]

        for command_text, tab_id, help_text in popular_tabs:
            yield Hit(
                1.0, command_text, partial(self.switch_tab, tab_id), help=help_text
            )

    def switch_tab(self, tab_id: str) -> None:
        """Switch to the specified tab."""
        try:
            route = self.route_for_tab(tab_id)
            self.app.post_message(NavigateToScreen(route))
            destination = self._shell_destination_for_tab(tab_id)
            label = (
                destination.accessible_label
                if destination is not None
                else get_tab_display_label(tab_id)
            )
            self.app.notify(f"Switched to {label}", severity="information")
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
        available_providers = (
            AVAILABLE_PROVIDERS if "AVAILABLE_PROVIDERS" in globals() else []
        )

        provider_commands = [
            (
                "LLM Provider Management: Show Current Provider",
                None,
                "Display currently selected LLM provider",
            ),
        ]

        # Add provider switching commands
        for provider in available_providers:
            provider_name = provider.replace("_", " ").title()
            command_text = f"LLM Provider Management: Switch to {provider_name}"
            provider_commands.append(
                (command_text, provider, f"Switch to {provider_name} provider")
            )

        for command_text, provider_id, help_text in provider_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_llm_command, provider_id, command_text),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_providers = ["OpenAI", "Anthropic", "Cohere", "Groq", "Ollama"]

        yield Hit(
            1.0,
            "LLM Provider Management: Show Current Provider",
            partial(self.handle_llm_command, None, "show_current"),
            help="Display currently selected LLM provider",
        )

        for provider in popular_providers:
            yield Hit(
                0.9,
                f"LLM Provider Management: Switch to {provider}",
                partial(self.handle_llm_command, provider, f"switch_{provider}"),
                help=f"Switch to {provider} provider",
            )

    def handle_llm_command(self, provider_id: str | None, command: str) -> None:
        """Handle LLM provider commands."""
        try:
            if provider_id is None or "show_current" in command:
                current = self._current_provider()
                self.app.notify(
                    f"Current LLM provider: {current}", severity="information"
                )
            else:
                self.app.pending_handoffs.stage(
                    HandoffChannel.CONSOLE_PROVIDER,
                    ConsoleProviderIntent(provider=provider_id),
                )
                chat_screen = self._mounted_chat_screen()
                if chat_screen is not None:
                    chat_screen.consume_pending_console_provider_intent()
                else:
                    self.app.notify(
                        "Provider selection queued for the next Console entry.",
                        severity="information",
                    )
        except Exception as e:
            self.app.notify(
                f"Failed to execute LLM command ({type(e).__name__}).",
                severity="error",
            )

    def _mounted_chat_screen(self):
        """Return the active production Console screen beneath any modal."""
        if getattr(self.app, "current_tab", None) != TAB_CHAT:
            return None
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

        for screen in reversed(tuple(getattr(self.app, "screen_stack", ()))):
            if isinstance(screen, ChatScreen):
                return screen
        return None

    def _current_provider(self) -> str:
        """Resolve current provider from its lifetime owner."""
        chat_screen = self._mounted_chat_screen()
        if chat_screen is not None:
            provider = chat_screen.current_console_provider_for_command()
            if provider:
                return provider
        config = getattr(self.app, "app_config", {})
        defaults = config.get("chat_defaults", {}) if isinstance(config, dict) else {}
        if isinstance(defaults, dict):
            provider = str(defaults.get("provider") or "").strip()
            if provider:
                return provider
        return "Unknown"


#: task-18812 / ADR-071: the command-palette entry for the Console focus
#: toggle -- one tuple reused by both QuickActionsProvider lists so the
#: command text, action id, and help string cannot drift apart.
FOCUS_TOGGLE_PALETTE_ENTRY = (
    "Quick Actions: Toggle Focus Mode",
    "toggle_focus_mode",
    "Hide or restore the Console's nav bar and header (Ctrl+Shift+F)",
)


class QuickActionsProvider(Provider):
    """Provider for quick action commands."""

    def __init__(self, screen, *args, **kwargs):
        """Initialize the QuickActionsProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        quick_actions = [
            (
                "Quick Actions: New Chat Conversation",
                "new_chat",
                "Start a new chat conversation",
            ),
            (
                "Quick Actions: New Character Chat",
                "new_character",
                "Start a new character-based conversation",
            ),
            ("Quick Actions: New Note", "new_note", "Create a new note"),
            (
                "Quick Actions: Import Media File",
                "import_media",
                "Import a new media file for processing",
            ),
            (
                "Quick Actions: Search All Content",
                "search_all",
                "Search across all content",
            ),
            FOCUS_TOGGLE_PALETTE_ENTRY,
        ]

        for command_text, action_id, help_text in quick_actions:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.execute_quick_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_actions = [
            (
                "Quick Actions: New Chat Conversation",
                "new_chat",
                "Start a new chat conversation",
            ),
            ("Quick Actions: New Note", "new_note", "Create a new note"),
            (
                "Quick Actions: Search All Content",
                "search_all",
                "Search across all content",
            ),
            (
                "Quick Actions: Import Media File",
                "import_media",
                "Import a new media file for processing",
            ),
            FOCUS_TOGGLE_PALETTE_ENTRY,
        ]

        for command_text, action_id, help_text in popular_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.execute_quick_action, action_id),
                help=help_text,
            )

    def execute_quick_action(self, action_id: str) -> None:
        """Execute the specified quick action."""
        try:
            if action_id == "new_chat":
                _navigate_via_screen(
                    self.app, TAB_CHAT, "Opened Console for a new conversation"
                )
            elif action_id == "new_character":
                _navigate_via_screen(
                    self.app,
                    TAB_PERSONAS,
                    "Opened Roleplay for character setup",
                )
            elif action_id == "new_note":
                _navigate_via_screen(
                    self.app,
                    TAB_LIBRARY,
                    "Opened Library for a new note",
                    {LIBRARY_NAV_CONTEXT_NOTES_CREATE: True},
                )
            elif action_id == "search_all":
                _navigate_via_screen(self.app, TAB_SEARCH, "Opened Library Search/RAG")
            elif action_id == "import_media":
                _navigate_via_screen(
                    self.app, TAB_INGEST, "Opened Import/Export for media import"
                )
            elif action_id == FOCUS_TOGGLE_PALETTE_ENTRY[1]:
                self.app.action_toggle_focus_mode()
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
            (
                "Settings & Preferences: Open Config File",
                "open_config",
                "Open the configuration file for editing",
            ),
            (
                "Settings & Preferences: Show Database Stats",
                "db_stats",
                "Show database size and statistics",
            ),
            (
                "Settings & Preferences: Open Settings Tab",
                "open_settings",
                "Navigate to the Settings tab",
            ),
        ]

        for command_text, setting_id, help_text in settings_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_setting, setting_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_settings = [
            (
                "Settings & Preferences: Open Settings Tab",
                "open_settings",
                "Navigate to the Settings tab",
            ),
            (
                "Settings & Preferences: Open Config File",
                "open_config",
                "Open the configuration file for editing",
            ),
            (
                "Settings & Preferences: Show Database Stats",
                "db_stats",
                "Show database size and statistics",
            ),
        ]

        for command_text, setting_id, help_text in popular_settings:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_setting, setting_id),
                help=help_text,
            )

    def handle_setting(self, setting_id: str) -> None:
        """Handle settings commands."""
        try:
            if setting_id == "open_settings":
                _navigate_via_screen(self.app, TAB_SETTINGS, "Opened Settings")
            elif setting_id == "open_config":
                self.app.notify(
                    f"Config file location: {get_cli_config_path()}",
                    severity="information",
                )
            elif setting_id == "db_stats":
                _navigate_via_screen(self.app, TAB_STATS, "Opened Statistics")
        except Exception as e:
            self.app.notify(
                f"Failed to execute settings command: {e}", severity="error"
            )


class CharacterProvider(Provider):
    """Provider for character and persona management commands."""

    def __init__(self, screen, *args, **kwargs):
        """Initialize the CharacterProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        character_commands = [
            (
                "Character/Persona Management: Create New Character",
                "new_character",
                "Create a new character or persona",
            ),
            (
                "Character/Persona Management: Show All Characters",
                "list_characters",
                "Display all available characters",
            ),
            (
                "Character/Persona Management: Open Character Tab",
                "open_character_tab",
                "Navigate to Character Chat tab",
            ),
        ]

        for command_text, action_id, help_text in character_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_character_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_character_actions = [
            (
                "Character/Persona Management: Open Character Tab",
                "open_character_tab",
                "Navigate to Character Chat tab",
            ),
            (
                "Character/Persona Management: Create New Character",
                "new_character",
                "Create a new character or persona",
            ),
            (
                "Character/Persona Management: Show All Characters",
                "list_characters",
                "Display all available characters",
            ),
        ]

        for command_text, action_id, help_text in popular_character_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_character_action, action_id),
                help=help_text,
            )

    def handle_character_action(self, action_id: str) -> None:
        """Handle character management actions."""
        try:
            if action_id == "open_character_tab":
                _navigate_via_screen(
                    self.app,
                    TAB_PERSONAS,
                    "Opened Roleplay",
                )
            elif action_id == "new_character":
                _navigate_via_screen(
                    self.app,
                    TAB_PERSONAS,
                    "Opened Roleplay to create a character",
                )
            elif action_id == "list_characters":
                _navigate_via_screen(
                    self.app,
                    TAB_PERSONAS,
                    "Opened Roleplay to list characters",
                )
        except Exception as e:
            self.app.notify(
                f"Failed to execute character action: {e}", severity="error"
            )


class MediaProvider(Provider):
    """Provider for media and content management commands."""

    def __init__(self, screen, *args, **kwargs):
        """Initialize the MediaProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        media_commands = [
            (
                "Media & Content: Open Media Library",
                "open_media",
                "Navigate to media library",
            ),
            (
                "Media & Content: Search Transcripts",
                "search_transcripts",
                "Search through media transcripts",
            ),
            (
                "Media & Content: Import New Media",
                "import_new",
                "Import new media file",
            ),
        ]

        for command_text, action_id, help_text in media_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_media_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_media_actions = [
            (
                "Media & Content: Open Media Library",
                "open_media",
                "Navigate to media library",
            ),
            (
                "Media & Content: Import New Media",
                "import_new",
                "Import new media file",
            ),
            (
                "Media & Content: Search Transcripts",
                "search_transcripts",
                "Search through media transcripts",
            ),
        ]

        for command_text, action_id, help_text in popular_media_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_media_action, action_id),
                help=help_text,
            )

    def handle_media_action(self, action_id: str) -> None:
        """Handle media management actions."""
        try:
            if action_id == "open_media":
                # task-2851: "media" now aliases to Library's own Media row
                # (screen_registry._SCREEN_ALIASES) instead of the retired
                # standalone MediaScreen -- the toast says so, matching the
                # "Opened Library X" wording the search_transcripts branch
                # below already uses for its own Library-folded route.
                _navigate_via_screen(self.app, TAB_MEDIA, "Opened Library Media")
            elif action_id == "import_new":
                _navigate_via_screen(
                    self.app, TAB_INGEST, "Opened Import/Export for media import"
                )
            elif action_id == "search_transcripts":
                _navigate_via_screen(
                    self.app,
                    TAB_SEARCH,
                    "Opened Library Search/RAG for transcript search",
                )
        except Exception as e:
            self.app.notify(f"Failed to execute media action: {e}", severity="error")


class LibraryIngestProvider(Provider):
    """Provider for the Library ingest deep-link command."""

    COMMANDS = (
        (
            "Library: Import…",
            "open_library_ingest",
            "Open Library and import content",
        ),
    )

    def __init__(self, screen, *args, **kwargs):
        """Initialize the LibraryIngestProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        for command_text, action_id, help_text in self.COMMANDS:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_library_ingest_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        for command_text, action_id, help_text in self.COMMANDS:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_library_ingest_action, action_id),
                help=help_text,
            )

    def handle_library_ingest_action(self, action_id: str) -> None:
        """Handle Library ingest actions."""
        try:
            if action_id == "open_library_ingest":
                _navigate_via_screen(
                    self.app,
                    TAB_LIBRARY,
                    "Opened Library to import content",
                    {LIBRARY_NAV_CONTEXT_INGEST: True},
                )
        except Exception as e:
            self.app.notify(f"Failed to open Library import: {e}", severity="error")


class SetupWizardProvider(Provider):
    """Provider for re-running the first-run setup wizard."""

    COMMANDS = (
        (
            "Setup: Run setup wizard…",
            "run_setup_wizard",
            "Walk through providers, models, and app configuration",
        ),
    )

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for command_text, action_id, help_text in self.COMMANDS:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_setup_wizard_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        for command_text, action_id, help_text in self.COMMANDS:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_setup_wizard_action, action_id),
                help=help_text,
            )

    def handle_setup_wizard_action(self, action_id: str) -> None:
        try:
            if action_id == "run_setup_wizard":
                from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
                    FirstRunSetupWizard,
                )

                self.app.push_screen(
                    FirstRunSetupWizard(self.app, rerun=True),
                    self.app.handle_first_run_wizard_result,
                )
        except Exception as e:
            self.app.notify(f"Failed to open setup wizard: {e}", severity="error")


class DeveloperProvider(Provider):
    """Provider for developer and debug commands."""

    def __init__(self, screen, *args, **kwargs):
        """Initialize the DeveloperProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        dev_commands = [
            (
                "Developer/Debug Commands: Show App Info",
                "app_info",
                "Display application version and build info",
            ),
            (
                "Developer/Debug Commands: Open Log File",
                "open_logs",
                "Navigate to application logs",
            ),
            (
                "Developer/Debug Commands: Show Keybindings",
                "show_keys",
                "Display all keyboard shortcuts",
            ),
        ]

        for command_text, action_id, help_text in dev_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_dev_action, action_id),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        popular_dev_actions = [
            (
                "Developer/Debug Commands: Open Log File",
                "open_logs",
                "Navigate to application logs",
            ),
            (
                "Developer/Debug Commands: Show App Info",
                "app_info",
                "Display application version and build info",
            ),
            (
                "Developer/Debug Commands: Show Keybindings",
                "show_keys",
                "Display all keyboard shortcuts",
            ),
        ]

        for command_text, action_id, help_text in popular_dev_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_dev_action, action_id),
                help=help_text,
            )

    def handle_dev_action(self, action_id: str) -> None:
        """Handle developer/debug actions."""
        try:
            if action_id == "open_logs":
                _navigate_via_screen(self.app, TAB_LOGS, "Opened Logs")
            elif action_id == "app_info":
                self.app.notify(
                    "tldw_chatbook - TUI for LLM interactions", severity="information"
                )
            elif action_id == "show_keys":
                self.show_keybindings()
        except Exception as e:
            self.app.notify(
                f"Failed to execute developer action: {e}", severity="error"
            )

    def show_keybindings(self) -> None:
        """Show a generated keybindings panel built from the app's BINDINGS."""
        try:
            state = WorkbenchHelpState(
                route_id="keybindings",
                title="App Keybindings",
                shortcuts=_bindings_to_shortcuts(getattr(self.app, "BINDINGS", ())),
            )
            self.app.push_screen(WorkbenchHelpPanel(state))
        except Exception as e:
            self.app.notify(f"Failed to show keybindings: {e}", severity="error")


class TabDropdown(Widget):
    """Placeholder for dropdown navigation (not yet implemented)."""

    def update_active_tab(self, tab_id: str) -> None:
        """No-op until the dropdown is implemented."""


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


def _library_ingest_write_failure_category(exc: BaseException) -> str:
    """Classify an exception escaping the ingest WRITE stage.

    (task-14821) The stage covers two different things: refusing an empty
    extraction, which happens BEFORE any write, and a genuine database
    write failure. Exceptions that know which they are declare it on
    ``ingest_error_category``.

    (xhigh review round) Everything else used to default to
    ``"write_error"`` -- the ONE category ``ingest_retry_advice`` still
    answers with "a retry can succeed if the write failure was temporary
    — the file itself parsed fine". So the optimistic branch task-14821
    was filed to remove stayed reachable for every unclassified cause,
    through the default. Only a failure of the database write itself
    earns that name now; an unknown cause is unnamed, and an unnamed
    category is silent (task-14821 AC#2) rather than encouraging.

    Args:
        exc: The exception raised while persisting a parsed payload.

    Returns:
        The ``error_detail`` category token, or ``""`` when the cause is
        not known to be a write failure.
    """
    declared = str(getattr(exc, "ingest_error_category", "") or "").strip()
    if declared:
        return declared
    if isinstance(exc, (MediaDatabaseError, MediaInputError, sqlite3.Error)):
        return "write_error"
    return ""


def _resolve_ingest_cookies_file(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Validate the audio/video panel's ``Cookies file for gated URLs`` value.

    (task-3306 xhigh review round) The value used to be forwarded verbatim
    as ``options["cookies"]``. ``download_video`` treats a string that is
    not an existing file as cookie JSON, so a typo'd or moved path became a
    ``json.JSONDecodeError`` caught into a single "Invalid cookie format"
    warning -- the download then ran un-authenticated and failed later for
    a reason that named neither cookies nor the path. Validating here, at
    the option boundary, is the earliest point this module owns.

    NOTE: the canonical home for per-field validation is the shared
    ``validate_ingest_option_value`` seam in ``library_ingest_state``, which
    is where the sibling text fields (``start_time``/``end_time``) are
    format-gated. This check lives here instead because existence is not a
    format question -- a path can be well-formed at typing time and gone by
    the time the job is claimed, which is exactly when this runs.

    Args:
        raw: The stripped field value; ``""`` means the option is unset.

    Returns:
        ``(cookies_path, problem)``. Exactly one is non-``None`` for a
        non-empty input; both are ``None`` when no cookies were requested.
    """
    if not raw:
        return None, None

    from tldw_chatbook.Utils.path_validation import validate_path_simple

    try:
        # Repo security rule: user-supplied file paths go through
        # path_validation before they become a subprocess/library argument.
        validate_path_simple(os.path.expanduser(raw))
    except ValueError as exc:
        return None, f"Unsafe cookies file path: {_sanitize_library_ingest_error(exc)}"

    candidate = Path(os.path.expanduser(raw))
    if not candidate.is_file():
        return None, f"Cookies file not found: {raw}"
    return str(candidate), None


def _library_ingest_done_progress(
    source_path: str, *, was_duplicate: bool, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a done job's ``progress`` dict from its persisted payload.

    (task-3301) Pure and module-level so the analysis-skipped annotation is
    unit-testable without a writer thread. When the parse payload carries
    ``analysis_skipped_reason`` (analysis was requested but no callable
    provider was configured at dispatch time), the done row says so --
    "analysis skipped: ..." on the row's progress sub-line -- instead of
    the analysis being silently absent. Duplicate-match outcomes keep their
    exact ``INGEST_DUPLICATE_PROGRESS_PREFIX`` message untouched: nothing
    new was imported, so there was nothing to analyze.

    Args:
        source_path: The job's source path (basename feeds the message).
        was_duplicate: Whether the write resolved to an existing item.
        payload: The parse payload that was just persisted.

    Returns:
        The ``progress`` dict for ``LibraryIngestJobRegistry.mark_done``.
    """
    if was_duplicate:
        return {
            "message": (
                f"{INGEST_DUPLICATE_PROGRESS_PREFIX} — "
                "matched an existing item; nothing new was "
                "imported."
            )
        }
    # (task-2016) The basename, not the absolute path: the row line already
    # identifies the file and the details surface carries the full path.
    source_name = Path(source_path).name or source_path
    progress: Dict[str, Any] = {"message": f"Imported {source_name}"}
    skip_reason = str(payload.get("analysis_skipped_reason") or "").strip()
    if skip_reason:
        progress["message"] += f" — analysis skipped: {skip_reason}"
        progress["analysis_skipped"] = skip_reason
    # (task-3301 xhigh review round, F4) An analysis that RAN and failed
    # (provider exception or an in-band "Error: ..." result) annotates the
    # done row the same way a skipped one does -- the import succeeded,
    # the analysis did not, and the user must be able to see which.
    failed_reason = str(payload.get("analysis_failed_reason") or "").strip()
    if failed_reason:
        progress["message"] += f" — analysis failed: {failed_reason}"
        progress["analysis_failed"] = failed_reason
    # (task-3306 xhigh review round) A cookies path the option boundary
    # refused to forward. The import itself is fine -- a public URL never
    # needed cookies -- so this is an annotation, not a failure; without it
    # a gated import that silently ran un-authenticated looks identical to
    # one that worked.
    cookies_problem = str(payload.get("cookies_problem") or "").strip()
    if cookies_problem:
        progress["message"] += f" — cookies ignored: {cookies_problem}"
        progress["cookies_problem"] = cookies_problem
    return progress


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

# (task 10, spec §9.1 AC 37/AC-24b) The named template errors the ingest
# dispatch fails an item on: an unresolvable choice (deleted/renamed) and a
# stored-invalid body refused by the validator.
#
# (task-21102) Resolved at except-time by the sole consumer (the ingest job
# dispatch loop) rather than imported at module scope: these two imports were
# one of the six entry points that executed the full Chunking package
# (~15k LOC shim + vendored engine) during ``import tldw_chatbook.app``.
# The lazily imported classes are the SAME objects the raising code
# (``_ingest_job_options`` -> ``Chunking.template_runtime`` /
# ``chunking_interop_library``) raises, so the except clause catches exactly
# what it always caught.
#
# (task-21102 review round) Because ``except _template_resolution_errors()``
# evaluates this for EVERY exception reaching that clause -- not only
# template errors -- the matcher must be inert for unrelated errors:
# * If ``tldw_chatbook.Chunking`` is not resident, no template error can be
#   in flight (an instance of its exception classes cannot exist without the
#   defining modules having been imported), so return ``()`` -- which
#   matches nothing -- WITHOUT importing ~39 Chunking modules as a side
#   effect of handling an unrelated exception.
# * If the imports themselves fail (broken install), also return ``()`` so
#   the ORIGINAL in-flight exception propagates with its own class instead
#   of being replaced by a ModuleNotFoundError raised from the except
#   clause.
# Guarded by ``Tests/App/test_template_error_lazy_matching.py``.
def _template_resolution_errors() -> tuple[type[Exception], ...]:
    """Return the named template-resolution error types, imported lazily.

    Returns:
        ``(TemplateResolutionError, InvalidTemplateError)`` when the
        Chunking package is resident and importable; ``()`` otherwise, so
        that using this as an ``except`` matcher never masks an unrelated
        in-flight exception and never imports Chunking as a side effect.
    """
    if "tldw_chatbook.Chunking" not in sys.modules:
        return ()
    try:
        from tldw_chatbook.Chunking.chunking_interop_library import (
            InvalidTemplateError,
        )
        from tldw_chatbook.Chunking.template_runtime import TemplateResolutionError
    except Exception:
        return ()

    return (TemplateResolutionError, InvalidTemplateError)

_INGEST_LOCAL_STT_PHASE_MESSAGES: dict[WorkerPhase, str] = {
    WorkerPhase.PREPARING: "Preparing import",
    WorkerPhase.LOADING: "Loading source",
    WorkerPhase.TRANSCRIBING: "Transcribing audio",
    WorkerPhase.POST_PROCESSING: "Post-processing audio",
}

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


@dataclass(frozen=True)
class _IngestParsePoolResources:
    """Process-pool resources owned by one ingest parse generation."""

    pool: Any
    progress_queue: Any | None


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


def _response_field(payload: Any, name: str) -> Any:
    """Read ``name`` from a pydantic model or a plain dict response.

    The tldw client returns models, but its own tests exercise the same paths
    with dicts, so both shapes are accepted.
    """
    if isinstance(payload, Mapping):
        return payload.get(name)
    return getattr(payload, name, None)


def _accepts_keyword(func: Any, name: str) -> bool:
    """Report whether ``func`` can be called with the ``name`` keyword.

    Asked up front instead of calling and catching ``TypeError``: that pattern
    cannot tell "this callable has no such parameter" from "a ``TypeError`` was
    raised inside it", so a genuine bug downstream reads as a missing feature
    and degrades silently. That is exactly how the remote ingest poller shipped
    asking for an ``offset`` the client did not yet accept, and paginated
    nothing for it (task-684.2).

    A callable whose signature cannot be read (a C builtin, an exotic mock) is
    reported as accepting the keyword, so real services are not downgraded by
    an unreadable signature; ``**kwargs`` counts as accepting it.
    """
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return True
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return True
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


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
    from ``TldwCli.on_unmount``): (1) ``_ingest_shutdown = True`` + executor
    and pool references detached, synchronously -- callbacks short-circuit
    before ever marshaling; (2) executor close followed by
    ``pool.terminate()`` + ``pool.join()`` on one detached daemon thread,
    never the event-loop thread (deadlock rationale in that method's
    docstring); (3) the writer thread is swept afterward by ``on_unmount``'s
    generic worker cancellation, its in-flight DB write completing as
    before. Steps 2 and 3 run concurrently -- safe because the stages share
    no resources (parse workers never touch ``media_db``; the writer never
    touches either heavy worker).
    """

    def _init_library_ingest_runtime_state(self) -> None:
        """Initialize every host attribute the ingest job loop reads.

        The single source of truth for this mixin's host-state contract:
        ``TldwCli``'s wiring calls this, and the headless test harnesses
        (``Tests/UI/test_library_shell.py``'s ``_LibraryIngestCanvasHarness``,
        ``Tests/Library/test_library_ingest_runner.py``'s
        ``_IngestRunnerHarness``) call the same method, so a new
        ``self._ingest_*`` read added to the coordinator/writer is mirrored
        into the fakes automatically instead of hand-listed (task-3315 --
        the hand-listed harness missed ``_ingest_local_stt_jobs`` when the
        local-STT lane landed and ~20 pilots died with AttributeError).
        ``self.media_db`` is deliberately NOT set here: it is a per-host
        input (see the class docstring), not coordinator state.

        F3 parallel-parse coordinator state: the lazily-created parse-pool
        handle, the parse->write handoff (job_id -> parsed payload dict,
        populated by a pool completion and drained by the writer's claim),
        and the shutdown flag pool callbacks check before touching a
        closing app.
        """
        self.library_ingest_jobs = LibraryIngestJobRegistry()
        self._ingest_parse_pool = None
        self._ingest_parse_pool_generation: int = 0
        self._ingest_parse_jobs_by_generation: dict[int, set[str]] = {}
        self._ingest_parse_pool_stop_event: Optional[threading.Event] = None
        self._ingest_parse_progress_queue: Any | None = None
        self._ingest_parse_progress_thread: threading.Thread | None = None
        self._ingest_parsed_payloads: dict[str, dict] = {}
        # RLock, not Lock: dev's STT dispatch work re-enters this guard.
        self._local_stt_executor_lock = threading.RLock()
        self._local_stt_executor: Optional[LocalSTTExecutor] = None
        self._local_stt_dispatch_coordinator: Optional[LocalSTTDispatchCoordinator] = (
            None
        )
        self._parakeet_source_service: Any | None = None
        self._parakeet_source_registry_listener: Callable[[], None] | None = None
        self._parakeet_submitting_scope_ids: set[str] = set()
        self._ingest_local_stt_jobs: dict[str, tuple[int, str]] = {}
        self._ingest_shutdown: bool = False

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

    def _expand_library_ingest_source(self, source_path: str) -> list[str] | None:
        """Expand a directory source into the files it contains.

        Args:
            source_path: The submitted source: a file path, a URL, or a
                directory.

        Returns:
            ``None`` when ``source_path`` is not a directory (URLs and files
            are submitted as-is), otherwise the list of contained file paths
            -- which is empty when the directory holds nothing ingestible.
        """
        try:
            candidate = Path(source_path).expanduser()
            if not candidate.is_dir():
                return None
        except (OSError, ValueError):
            # Unreadable, over-length, or malformed for this platform (Windows
            # raises ValueError where POSIX raises OSError). Not a directory we
            # can expand -- let the single-source path report the failure.
            return None

        raw_limit = get_cli_setting("library.ingest_directory_scan_limit", 1000)
        try:
            scan_limit = int(raw_limit)
        except (TypeError, ValueError):
            scan_limit = 1000

        files, truncated = collect_directory_files(candidate, scan_limit)
        if truncated:
            logger.warning(
                f"Library ingest directory {source_path!r} exceeded the scan "
                f"limit of {scan_limit}; only the first {len(files)} files "
                "were queued."
            )
        return [str(path) for path in files]

    def submit_library_ingest_job(
        self,
        *,
        source_path: str,
        ingest_options: dict[str, Any] | None = None,
        title: str = "",
        author: str = "",
        keywords: tuple[str, ...] = (),
        perform_analysis: bool = False,
        chunk_enabled: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        batch_id: str | None = None,
        active_duplicate_consent: ActiveIngestConsentScope | None = None,
    ) -> LibraryIngestJob:
        """Submit a new Library ingest job and top up the parse pool.

        UI-thread only. Appends a ``QUEUED`` job to ``self.library_ingest_jobs``.
        When ``self.media_db`` is unavailable, the job is failed immediately
        (with the exact copy ``"Media database is unavailable."``) and it
        never reaches the parse pool.
        ``batch_id`` carries the folder-expansion batch id (task-2221) so
        the queue can group one submission's jobs; ``None`` for single
        files.

        Args:
            source_path: The file path to ingest.
            ingest_options: Per-type ingestion options snapshot captured at
                submit time. This is the canonical source of ingestion
                settings; the older ``perform_analysis``/``chunk_enabled``/
                ``chunk_size`` arguments are deprecated fallbacks.
            title: Optional title form field.
            author: Optional author form field.
            keywords: Keywords form field.
            perform_analysis: Whether to run post-ingest analysis.
            chunk_enabled: Whether to chunk the ingested content.
            chunk_size: Requested chunk size when ``chunk_enabled``.
            active_duplicate_consent: Exact candidate and active-membership scope
                captured by an explicitly confirmed submission.

        Returns:
            The newly created job: ``QUEUED`` normally, or immediately
            ``FAILED`` when ``media_db`` is unavailable. A directory source
            queues one job per contained file and returns the first of them,
            so each file gets its own queue row, its own outcome and its own
            retry -- one unsupported file no longer fails its siblings.
        """
        backend = self._resolve_ingest_backend()
        expanded = self._expand_library_ingest_source(source_path)
        sources = tuple(expanded) if expanded is not None else (source_path,)
        matches = self.library_ingest_jobs.find_active_source_matches(
            sources, origin=backend
        )
        matched_source_keys = set()
        for job in matches:
            try:
                matched_source_keys.add(
                    normalize_active_ingest_source(job.source_path, origin=backend)
                )
            except (TypeError, ValueError, OSError):
                continue
        current_consent = build_active_ingest_consent_scope(
            sources,
            origin=backend,
            active_job_ids=(job.job_id for job in matches),
            active_source_count=len(matched_source_keys),
        )
        candidates_changed = active_duplicate_consent is not None and (
            active_duplicate_consent.origin != current_consent.origin
            or active_duplicate_consent.candidate_digest
            != current_consent.candidate_digest
            or active_duplicate_consent.candidate_count
            != current_consent.candidate_count
        )
        matches_covered = (
            active_duplicate_consent is not None
            and active_duplicate_consent.covers(current_consent)
        )
        if candidates_changed or (matches and not matches_covered):
            raise ActiveIngestSubmissionRefused(
                (ActiveIngestJobRef(job.job_id, job.state) for job in matches),
                consent_scope=current_consent,
                candidate_changed=candidates_changed,
            )

        normalized_options = ingest_options or {}
        if expanded is not None:
            if not expanded:
                empty_job = self.library_ingest_jobs.submit(
                    source_path=source_path,
                    title=title,
                    author=author,
                    keywords=keywords,
                    perform_analysis=perform_analysis,
                    chunk_enabled=chunk_enabled,
                    chunk_size=chunk_size,
                    detected_type="",
                    ingest_options=normalized_options,
                )
                failed = self.library_ingest_jobs.mark_failed(
                    empty_job.job_id,
                    error="No files to import were found in this folder.",
                )
                return failed if failed is not None else empty_job
            first_job: LibraryIngestJob | None = None
            # (task-2221 owner ruling) One batch id per folder submission,
            # so the queue can group this run's rows under one header and
            # the tally can answer "what did THIS run just do".
            folder_batch_id = f"local-{uuid.uuid4().hex[:12]}"
            audio_options = normalized_options.get("audio_video", {})
            scope_id = (
                str(audio_options.get("transcription_external_scope_id") or "").strip()
                if isinstance(audio_options, dict)
                else ""
            )
            submitting_scopes = getattr(self, "_parakeet_submitting_scope_ids", None)
            if submitting_scopes is None:
                submitting_scopes = self._parakeet_submitting_scope_ids = set()
            if scope_id:
                submitting_scopes.add(scope_id)
            try:
                for expanded_path in expanded:
                    job = self._submit_library_ingest_job_admitted(
                        source_path=expanded_path,
                        ingest_options=normalized_options,
                        batch_id=folder_batch_id,
                        # Title is per-file (the ingest form clears it on submit
                        # for exactly this reason), so a folder's files each take
                        # their own filename-derived title rather than all
                        # sharing one. Author and keywords are batch metadata and
                        # do carry across.
                        title="",
                        author=author,
                        keywords=keywords,
                        perform_analysis=perform_analysis,
                        chunk_enabled=chunk_enabled,
                        chunk_size=chunk_size,
                        backend=backend,
                    )
                    if first_job is None:
                        first_job = job
            finally:
                if scope_id:
                    submitting_scopes.discard(scope_id)
                    self._sync_parakeet_source_scopes()
            # ``expanded`` is non-empty here, so the loop always assigns.
            assert first_job is not None
            return first_job

        return self._submit_library_ingest_job_admitted(
            source_path=source_path,
            ingest_options=normalized_options,
            title=title,
            author=author,
            keywords=keywords,
            perform_analysis=perform_analysis,
            chunk_enabled=chunk_enabled,
            chunk_size=chunk_size,
            batch_id=batch_id,
            backend=backend,
        )

    def _submit_library_ingest_job_admitted(
        self,
        *,
        source_path: str,
        ingest_options: dict[str, Any],
        title: str,
        author: str,
        keywords: tuple[str, ...],
        perform_analysis: bool,
        chunk_enabled: bool,
        chunk_size: int,
        batch_id: str | None,
        backend: str,
    ) -> LibraryIngestJob:
        """Route a source already admitted by ``submit_library_ingest_job``."""
        if backend == "server":
            # A web page goes to the clipper, not the ingest-jobs API: that API
            # has no media type for one. A local ingest needs no such branch --
            # classify_ingest_source already routes an article through the
            # pipeline's own extractor.
            submit_remote = (
                self._submit_web_clip_job
                if is_web_clip_source(source_path)
                else self._submit_server_ingest_job
            )
            return submit_remote(
                source_path=source_path,
                ingest_options=ingest_options,
                title=title,
                author=author,
                keywords=keywords,
                perform_analysis=perform_analysis,
            )

        return self._submit_local_library_ingest_job(
            source_path=source_path,
            ingest_options=ingest_options,
            title=title,
            author=author,
            keywords=keywords,
            perform_analysis=perform_analysis,
            chunk_enabled=chunk_enabled,
            chunk_size=chunk_size,
            batch_id=batch_id,
        )

    def _submit_local_library_ingest_job(
        self,
        *,
        source_path: str,
        ingest_options: dict[str, Any],
        title: str,
        author: str,
        keywords: tuple[str, ...],
        perform_analysis: bool,
        chunk_enabled: bool,
        chunk_size: int,
        batch_id: str | None,
    ) -> LibraryIngestJob:
        """Append one admitted local source and top up the parse pool."""
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
            ingest_options=ingest_options,
            batch_id=batch_id,
        )
        if self.media_db is None:
            failed = self.library_ingest_jobs.mark_failed(
                job.job_id, error="Media database is unavailable."
            )
            return failed if failed is not None else job
        self._top_up_ingest_parse_pool()
        return job

    def retry_library_ingest_job(
        self,
        job_id: str,
        *,
        transcription_provider: str | None = None,
    ) -> Optional[LibraryIngestJob]:
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
        replacement_options = None
        if transcription_provider not in {None, "faster-whisper"}:
            return None
        if transcription_provider is not None:
            source = self.library_ingest_jobs.get_job(job_id)
            if source is None:
                return None
            replacement_options = deepcopy(source.ingest_options)
            replacement_options.setdefault("audio_video", {})[
                "transcription_provider"
            ] = transcription_provider
        requeued = self.library_ingest_jobs.requeue(
            job_id,
            ingest_options=replacement_options,
        )
        if requeued is None:
            return None
        if self.media_db is None:
            failed = self.library_ingest_jobs.mark_failed(
                requeued.job_id, error="Media database is unavailable."
            )
            return failed if failed is not None else requeued
        self._top_up_ingest_parse_pool()
        return requeued

    def retry_library_ingest_job_with_provider(
        self,
        job_id: str,
        provider: str,
    ) -> Optional[LibraryIngestJob]:
        """Run the one supported explicit cross-provider recovery action."""

        if provider != "faster-whisper":
            return None
        return self.retry_library_ingest_job(
            job_id,
            transcription_provider=provider,
        )

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
        fake resource bundle (see
        ``Tests/Library/test_library_ingest_runner.py``) so pilots stay
        deterministic without spawning real OS processes. Real callers get a
        spawn-context ``multiprocessing.Pool`` and bounded progress queue.

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
        usable fd, both the queue and Pool are constructed under
        ``contextlib.redirect_stderr`` pointing at a genuinely fd-backed
        stream (``_ingest_pool_real_stderr``: ``sys.__stderr__``, else a
        kept-alive devnull handle). The tracker launches at most once per
        process, so covering construction is sufficient -- and applying
        the redirect on every (re)construction is harmless. Queue and Pool
        creation are one atomic owner operation: if Pool creation fails, the
        already-created queue is closed before the exception escapes.
        """
        ctx = multiprocessing.get_context("spawn")
        processes = self._ingest_parse_worker_count()

        def _construct_resources() -> _IngestParsePoolResources:
            progress_queue = None
            try:
                progress_queue = ctx.Queue(maxsize=INGEST_PARSE_PROGRESS_QUEUE_MAXSIZE)
                pool = ctx.Pool(
                    processes=processes,
                    initializer=initialize_ingest_parse_worker,
                    initargs=(progress_queue,),
                )
            except Exception:
                if progress_queue is not None:
                    for method_name in ("close", "cancel_join_thread"):
                        method = getattr(progress_queue, method_name, None)
                        if method is None:
                            continue
                        try:
                            method()
                        except Exception:
                            logger.error(
                                "Error cleaning up a partially constructed "
                                "Library ingest progress queue "
                                "(operation={}, queue_type={}).",
                                method_name,
                                type(progress_queue).__name__,
                            )
                raise
            return _IngestParsePoolResources(pool, progress_queue)

        # The combined initializer keeps worker import noise off the TUI and
        # installs this generation's progress sink.
        if _stream_fileno(sys.stderr) >= 0:
            return _construct_resources()
        with contextlib.redirect_stderr(_ingest_pool_real_stderr()):
            return _construct_resources()

    def _ensure_ingest_parse_pool(self):
        """Return the current parse pool, lazily creating one if needed.

        UI-thread only.
        """
        if self._ingest_parse_pool is None:
            resources = self._create_ingest_parse_pool()
            pool = resources.pool
            progress_queue = resources.progress_queue
            try:
                sentinels = self._ingest_parse_pool_worker_sentinels(pool)
            except Exception:
                self._terminate_ingest_parse_pool_off_thread(
                    pool,
                    progress_queue,
                    None,
                )
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
            self._ingest_parse_progress_queue = progress_queue
            self._ingest_parse_progress_thread = None
            if progress_queue is not None:
                self._ingest_parse_progress_thread = (
                    self._start_ingest_parse_progress_drain(
                        generation,
                        progress_queue,
                        stop_event,
                    )
                )
            if sentinels:
                self._start_ingest_parse_pool_monitor(generation, sentinels, stop_event)
        return self._ingest_parse_pool

    def _start_ingest_parse_progress_drain(
        self,
        generation: int,
        progress_queue: Any,
        stop_event: threading.Event,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> threading.Thread:
        """Start the bounded, latest-per-job drain for one pool generation."""

        def _drain() -> None:
            coalescer = ParseProgressCoalescer(
                interval=INGEST_PARSE_PROGRESS_FLUSH_SECONDS,
                started_at=clock(),
            )
            while not stop_event.is_set() and not self._ingest_shutdown:
                try:
                    raw_event = progress_queue.get(timeout=0.05)
                except queue.Empty:
                    raw_event = None
                except (EOFError, OSError, ValueError):
                    return
                if stop_event.is_set() or self._ingest_shutdown:
                    return
                if raw_event is not None:
                    try:
                        event = make_parse_progress_event(
                            raw_event.generation,
                            raw_event.job_id,
                            raw_event.phase,
                            raw_event.message,
                            raw_event.percent,
                        )
                    except Exception:
                        event = None
                    if event is not None:
                        coalescer.accept(event)
                batch = coalescer.take_due(clock())
                if batch:
                    if stop_event.is_set() or self._ingest_shutdown:
                        return
                    self._marshal_ingest_pool_call(
                        self._on_ingest_parse_progress_batch,
                        generation,
                        batch,
                    )

        thread = threading.Thread(
            target=_drain,
            name=f"library-ingest-progress-drain-{generation}",
            daemon=True,
        )
        thread.start()
        return thread

    @staticmethod
    def _ingest_parse_pool_worker_sentinels(pool: Any) -> Optional[tuple[Any, ...]]:
        """Snapshot real Pool worker sentinels; injected fakes may opt out."""
        workers = getattr(pool, "_pool", None)
        if workers is None:
            return None
        try:
            sentinels = tuple(worker.sentinel for worker in workers)
        except Exception as exc:
            raise RuntimeError(
                "Could not inspect parse-pool worker sentinels."
            ) from exc
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

    def _ingest_job_options(self, job: LibraryIngestJob) -> Dict[str, Any]:
        """Build ``run_parse_job``'s ``options`` dict from a job's fields.

        ``job.ingest_options`` is the canonical source of ingestion settings.
        It is expected to be a group-keyed snapshot (e.g. ``{"generic": {...},
        "pdf": {...}}``); values from the detected type group override values
        from the ``generic`` group. The older scalar fields
        (``perform_analysis``, ``chunk_enabled``, ``chunk_size``) are used only
        as deprecated fallbacks when ``ingest_options`` is empty or does not
        contain a value.

        (task-3301) The three previously dead controls resolve here:

        * ``encoding`` (generic group) travels to the plaintext/html readers.
        * ``chunk_options`` carries the form's size/overlap as ints (the
          snapshot boundary coerces, but restored/persisted jobs may still
          hold display strings), in both the ``size`` spelling (audio/video
          option maps) and the ``max_size`` spelling
          (``improved_chunking_process``); the overlap fallback is the
          generic schema default -- the value the UI displays -- not a
          hardcoded constant. (task-3301/3303 xhigh review round 2,
          F11+F12) An explicit ``method`` ALWAYS travels: pdf and
          audio/video get ``words`` so the generic size/overlap hint
          ("words · 100-5000") is true everywhere the processors would
          otherwise setdefault sentences (a ~10-30x unit lie); the ebook
          group maps its panel choice ("chapters" -> the chunker's
          ``ebook_chapters``, other names verbatim) and falls back to the
          pre-branch ``sentences`` when the snapshot predates the field
          (fresh snapshots always carry the schema default -- absence IS
          the legacy marker, so a requeued old job keeps its original
          scheme). The text tail's own default is already words.
        * When analysis is requested, the configured analysis provider
          (``[analysis_defaults] provider``) is resolved through the shared
          readiness seam, then constrained to a chat-dispatchable name
          (task-3301 xhigh review round): ready adds ``api_name`` (the
          normalized ``API_CALL_HANDLERS`` key), ``api_key`` (``None`` for
          keyless local providers, paired with the explicit
          ``analysis_keyless_ok`` opt-in the processors' credential gates
          require), and ``analysis_call`` (model/temperature/top_p/min_p/
          max_tokens from ``[analysis_defaults]``, viewer-parity defaults)
          plus ``system_prompt`` when the section configures one; not ready
          (including readiness-ready providers with no chat handler) adds
          ``analysis_skipped_reason`` so the job records WHY analysis is
          absent instead of silently dropping it.

        ``custom_prompt`` and ``system_prompt`` are copied from the generic
        snapshot only when analysis is requested. They remain in the job
        snapshot while analysis is off, but parser options omit them so stale
        instructions cannot reach a backend that will not execute analysis.
        ``metadata`` remains absent (``None`` inside the worker's
        ``options.get(...)`` reads).
        """
        opts = job.ingest_options or {}
        group = get_type_group(job.source_path)

        # Resolve a flat option map from the generic group and the detected
        # type-specific group, with type-specific values taking precedence.
        generic_opts: dict[str, Any] = dict(opts.get("generic", {}))
        flat_opts: dict[str, Any] = dict(generic_opts)
        flat_opts.update(opts.get(group, {}) or {})

        def _as_int(value: Any, fallback: int) -> int:
            """Coerce a possibly-display-string number, falling back."""
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                return fallback

        perform_analysis = bool(flat_opts.get("analyze", job.perform_analysis))
        overlap_default = _as_int(generic_option_default("chunk_overlap", 100), 100)
        chunk_size = _as_int(
            flat_opts.get("chunk_size", job.chunk_size), job.chunk_size
        )
        chunk_enabled = bool(flat_opts.get("chunk", job.chunk_enabled))

        # (task 10, spec §9.1 AC 34) Template resolution -- ingest order:
        # picker/batch choice -> config [chunking] default_template -> plain
        # options. Resolution happens HERE (the app process owns the media
        # DB) and the resolved DICT travels inside chunk_options: the parse
        # worker must stay DB-free. An unresolvable or stored-invalid choice
        # raises a NAMED error (TemplateResolutionError /
        # InvalidTemplateError, AC 37 / AC-24b) -- the ingest dispatch
        # catches both and fails THIS item; there is never a silent
        # fallback to plain chunking.
        # (task 4, auto-selection spec §4.3) A picker choice of the Auto
        # sentinel ("auto") resolves to an AutoDecision instead: the job's
        # ALREADY-KNOWN metadata (detected type / title / filename / URL)
        # feeds resolve_auto -- nothing re-reads file contents at selection
        # time. A template-tier win is consumed exactly like a manual pick;
        # a plan-tier win materializes the planner's options as this
        # parse's defaults; a plain-tier win changes nothing below.
        ingest_template: dict[str, Any] | None = None
        auto_decision: Any = None
        plan_options: dict[str, Any] | None = None
        if chunk_enabled:
            from .Chunking.template_runtime import resolve_ingest_template

            source_is_url = str(job.source_path or "").lower().startswith(
                ("http://", "https://")
            )
            resolved = resolve_ingest_template(
                getattr(self, "media_db", None),
                str(flat_opts.get("chunk_template") or "").strip() or None,
                media_type=str(job.detected_type or "").strip() or None,
                title=str(job.title or "").strip() or None,
                filename=(
                    None
                    if source_is_url
                    else PurePath(job.source_path).name
                ),
                url=str(job.source_path) if source_is_url else None,
            )
            from .Chunking.auto_selection import AutoDecision

            if isinstance(resolved, AutoDecision):
                auto_decision = resolved
                if resolved.tier == "template" and isinstance(
                    resolved.template, dict
                ):
                    ingest_template = resolved.template
                elif resolved.tier == "plan" and isinstance(
                    resolved.chunk_options, dict
                ):
                    plan_options = dict(resolved.chunk_options)
                # plain tier: fall through to today's default options
            else:
                ingest_template = resolved

        if ingest_template is not None or plan_options is not None:
            # (task 10, spec §9.1 AC 35 -- the precedence ruling) A resolved
            # template's chunk-stage options beat the ingest builder's
            # DEFAULTS; only a value the user explicitly CHANGED in the
            # ingest form beats the template. Left as-is, the builder's
            # always-on size/overlap (+ per-group method injection) would
            # arrive at the Chunker as explicit options that override the
            # template on every path -- the picker would be inert.
            # (task 4, auto-selection §4.3) The plan tier rides the SAME
            # ruling: the planner's options are the defaults, a
            # user-changed form value still wins.
            #
            # Mechanism: the form snapshot ALWAYS carries explicit values
            # (``_build_ingest_options_snapshot`` seeds every schema
            # default), so "differs from the schema default" is the only
            # user-changed signal available at this seam. Values equal to
            # the schema default are dropped here and re-derived from the
            # template by the parse seam's materialization
            # (``materialize_template_chunk_options``); values that differ
            # ride along and win at the Chunker's explicit-beats-template
            # merge. A snapshot WITHOUT the key (pre-field legacy jobs) has
            # no user signal at all and defaults to the template winning.
            size_schema_default = _as_int(
                generic_option_default("chunk_size", DEFAULT_CHUNK_SIZE),
                DEFAULT_CHUNK_SIZE,
            )
            overlap_schema_default = overlap_default
            if ingest_template is not None:
                chunk_options: dict[str, Any] = {"template": ingest_template}
            else:
                # Plan tier: the planner's options travel as this parse's
                # chunk-stage defaults; ``size`` mirrors ``max_size`` for
                # the audio/video key-by-key re-projection (the same alias
                # ``materialize_template_chunk_options`` fills).
                chunk_options = dict(plan_options)
                if "max_size" in chunk_options:
                    chunk_options.setdefault("size", chunk_options["max_size"])
            if "chunk_size" in flat_opts and chunk_size != size_schema_default:
                chunk_options["size"] = chunk_size
                chunk_options["max_size"] = chunk_size
            if (
                "chunk_overlap" in flat_opts
                and _as_int(flat_opts.get("chunk_overlap"), overlap_default)
                != overlap_schema_default
            ):
                chunk_options["overlap"] = _as_int(
                    flat_opts.get("chunk_overlap"), overlap_default
                )
        else:
            chunk_options = (
                {
                    "size": chunk_size,
                    "max_size": chunk_size,
                    "overlap": _as_int(
                        flat_opts.get("chunk_overlap", overlap_default),
                        overlap_default,
                    ),
                }
                if chunk_enabled
                else None
            )
        if auto_decision is not None and chunk_options is not None:
            # (task 4, auto-selection spec §4.4) The decision's travel
            # ticket to the persist seam (mode/auto_tier/auto_rationale).
            # The parse seam POPS this key before any branch dispatch, so
            # no processor or the Chunker ever sees it.
            chunk_options["auto"] = {
                "tier": str(auto_decision.tier),
                "rationale": [
                    str(line) for line in (auto_decision.rationale or [])
                ],
            }

        options: dict[str, Any] = {
            "title": job.title or None,
            "author": job.author or None,
            "keywords": list(job.keywords) or None,
            "perform_analysis": perform_analysis,
            # These generic fields intentionally travel independently of the
            # detected type-group branch. The downstream local overwrite/RAG
            # behavior is owned by later work; this seam only makes the form
            # snapshot honest for consumers that already read these options.
            "overwrite_existing": bool(
                generic_opts.get(
                    "overwrite_existing",
                    generic_option_default("overwrite_existing", False),
                )
            ),
            "generate_embeddings": bool(
                generic_opts.get(
                    "generate_embeddings",
                    generic_option_default("generate_embeddings", True),
                )
            ),
            "encoding": flat_opts.get("encoding"),
            "chunk_options": chunk_options,
        }
        # ``template_active`` gates the per-group METHOD injection below:
        # the pdf/audio-video/image "words" and the ebook group mapping are
        # builder DEFAULTS (the user cannot type a method in those panels),
        # so under a resolved template they must not be injected -- the
        # template's method wins via materialization. A user-changed ebook
        # chunk_method (differs from the select's schema default) still
        # travels. (task 4, auto-selection §4.3) An auto PLAN-tier win
        # governs identically: its method is a derived default, not a user
        # choice, so the injection is skipped for it too; the auto PLAIN
        # tier keeps today's injections (it changes nothing).
        template_active = (
            ingest_template is not None or plan_options is not None
        )

        if perform_analysis:
            # Prompts remain in the persisted generic snapshot while analysis
            # is off, but parser options must not carry instructions no
            # backend will execute.
            options["custom_prompt"] = generic_opts.get(
                "custom_prompt", generic_option_default("custom_prompt", "")
            )
            options["system_prompt"] = generic_opts.get(
                "system_prompt", generic_option_default("system_prompt", "")
            )
            resolution = resolve_ingest_analysis_provider(
                getattr(self, "app_config", None)
            )
            if resolution.ready:
                # (task-3301 xhigh review round) The NORMALIZED dispatch
                # name (an `API_CALL_HANDLERS` key) travels, not the
                # display spelling -- it is what `chat_api_call` and the
                # summarizer's alias map accept (F5).
                options["api_name"] = resolution.dispatch_name
                options["api_key"] = resolution.api_key
                if resolution.keyless:
                    # (F8) Explicit keyless opt-in: the processors' analysis
                    # gates only dispatch without a credential when the
                    # readiness seam vouched for keyless operation.
                    options["analysis_keyless_ok"] = True
                # (F10) The full [analysis_defaults] call shape, so an
                # ingest analysis runs with the same model/sampling the
                # Media viewer's analysis panel would use.
                options["analysis_call"] = {
                    "model": resolution.model,
                    "temperature": resolution.temperature,
                    "top_p": resolution.top_p,
                    "min_p": resolution.min_p,
                    "max_tokens": resolution.max_tokens,
                }
                if resolution.system_prompt and not options.get("system_prompt"):
                    options["system_prompt"] = resolution.system_prompt
            else:
                options["analysis_skipped_reason"] = resolution.short_reason

        if group == "pdf":
            if options["chunk_options"] is not None and not template_active:
                # (F12) ``process_pdf`` setdefaults method='sentences',
                # under which the form's "words · 100-5000" size hint is a
                # ~10-30x unit lie (500 SENTENCES ~= one chunk per
                # document). Words is what the hint promises. (task 10)
                # Under a resolved template this injection is a builder
                # DEFAULT and is skipped -- the template's method wins.
                options["chunk_options"]["method"] = "words"
            options["pdf_engine"] = flat_opts.get("engine") or flat_opts.get(
                "pdf_engine"
            )
            options["page_range"] = flat_opts.get("pages")
            options["ocr"] = flat_opts.get("ocr", flat_opts.get("enable_ocr", False))
            options["extract_images"] = flat_opts.get("extract_images", False)
            # (task-3303) OCR detail: language + backend, with the
            # processor's own defaults as the fallbacks. The panel gates
            # the OCR toggle to the docling/docext engines, so a silent
            # OCR-under-pymupdf no-op can no longer be *asked for*; the
            # values themselves always travel (process_pdf ignores them
            # when the parser cannot OCR).
            options["ocr_language"] = flat_opts.get("ocr_language") or "en"
            options["ocr_backend"] = flat_opts.get("ocr_backend") or "auto"
        elif group == "document":
            # (task-3303) The document group layers ON TOP of generic:
            # ``flat_opts`` already merged generic (analyze/chunk/encoding)
            # under these, so document files keep task-3301's chunking and
            # analysis while gaining ``process_document``'s own knobs.
            options["processing_method"] = flat_opts.get("processing_method") or "auto"
            options["enable_ocr"] = flat_opts.get(
                "ocr", flat_opts.get("enable_ocr", False)
            )
            options["ocr_language"] = flat_opts.get("ocr_language") or "en"
        elif group == "audio_video":
            if options["chunk_options"] is not None and not template_active:
                # (F12) The audio/video branch defaults chunk_method to
                # sentences too -- same unit-lie fix as the pdf branch.
                # (task 10) Skipped under a resolved template (a builder
                # default, not a user choice).
                options["chunk_options"]["method"] = "words"
            provider = flat_opts.get("transcription_provider")
            if provider is None:
                provider = "default"
            target_language = flat_opts.get("translation_target_language")
            if target_language is None:
                target_language = flat_opts.get("target_language")
            if (
                target_language is None
                and flat_opts.get("translate_to_english")
                # (task-3303 xhigh review round 2, F9) Honor the checkbox's
                # own schema gate: its value survives in the snapshot after
                # the provider select moves to one that rejects translation
                # (transcribe-cpp/parakeet raise BatchSTTRoutingError, which
                # failed the WHOLE batch at dispatch). The normalized
                # provider is passed so the gate sees the same value the
                # route resolution below will use.
                and field_gate_open(
                    "audio_video",
                    "translate_to_english",
                    {**flat_opts, "transcription_provider": provider},
                )
            ):
                # (task-3303) The panel's translate toggle. An explicit
                # target (retry overrides, older snapshots) stays
                # authoritative; the checkbox only fills the gap.
                target_language = "en"
            route = resolve_batch_stt_route(
                provider=provider,
                language=flat_opts.get("language"),
                target_language=target_language,
                precision=flat_opts.get("transcription_precision"),
            )
            options["transcription_provider"] = route.provider
            selected_model_dir = (
                str(flat_opts.get("transcription_model_dir") or "").strip()
                if route.provider == "parakeet-onnx"
                else ""
            )
            options["transcription_model_dir"] = selected_model_dir or None
            selected_model = route.model
            if selected_model is None and route.requested_provider not in {
                "default",
                "transcribe-cpp",
            }:
                selected_model = flat_opts.get("model") or flat_opts.get(
                    "transcription_model"
                )
                if route.requested_provider == "faster-whisper" and not selected_model:
                    selected_model = "base"
            options["transcription_model"] = selected_model
            options["language"] = route.requested_language
            options["translation_target_language"] = route.target_language
            options["transcription_precision"] = route.precision
            options["transcription_local_files_only"] = route.local_files_only
            options["transcription_batch_route_resolved"] = True
            options["timestamps"] = flat_opts.get("timestamps", True)
            options["diarization"] = flat_opts.get("diarization", False)
            # (task-3303) VAD filter -- travels as its own option; the
            # parse worker hands it to the processors' ``vad_use``.
            options["vad_filter"] = bool(flat_opts.get("vad_filter", False))
            # (task-3306) Time-range trim: format-gated at the option layer
            # (HH:MM:SS or seconds); blank means unbounded on that side.
            start_trim = str(flat_opts.get("start_time") or "").strip()
            end_trim = str(flat_opts.get("end_time") or "").strip()
            options["start_time"] = start_trim or None
            options["end_time"] = end_trim or None
            # (task-3306) Gated URL downloads: a cookies FILE PATH only
            # (yt-dlp cookiefile) -- raw cookie text is a credential, and
            # this options dict persists with the job and echoes into
            # config.toml. Its presence IS the use_cookies flag, so there
            # is no separate toggle to go stale. Only the video (yt-dlp)
            # branch of ``parse_local_file_for_ingest`` consumes it; the
            # audio downloader's cookies parameter has JSON-dict semantics
            # a path would crash.
            # (xhigh review round) Validated here, not forwarded verbatim:
            # an unusable path used to degrade into a silent "Invalid
            # cookie format" debug line inside the downloader.
            cookies_file = str(flat_opts.get("cookies_file") or "").strip()
            cookies_path, cookies_problem = _resolve_ingest_cookies_file(cookies_file)
            options["use_cookies"] = bool(cookies_path)
            options["cookies"] = cookies_path
            if cookies_problem:
                options["cookies_problem"] = cookies_problem
            # (task-3306) Recursive map-reduce summary; the processors'
            # analysis tail consumes it only when analysis actually runs,
            # so an idle True is inert rather than a stale hazard.
            options["summarize_recursively"] = bool(
                flat_opts.get("summarize_recursively", False)
            )
            failed_attempt = job.retry_source_failure_provenance
            options["transcription_context"] = {
                "attempt_id": f"{job.job_id}-attempt-{job.retry_count + 1}",
                "batch_id": job.batch_id,
                "job_id": job.job_id,
                "retry_of_attempt_id": failed_attempt.get("attempt_id")
                if failed_attempt
                else None,
                "retry_of_job_id": job.retry_of_job_id,
                "retry_source_failure_provenance": failed_attempt,
            }
            external_scope_id = str(
                flat_opts.get("transcription_external_scope_id") or ""
            ).strip()
            if external_scope_id:
                options["transcription_context"]["external_scope_id"] = (
                    external_scope_id
                )
            if route.provider == "transcribe-cpp":
                configured_path = get_cli_setting(
                    "transcription.transcribe_cpp.model_path"
                )
                options["transcription_context"]["model_path"] = (
                    configured_path
                    if isinstance(configured_path, str) and configured_path
                    else None
                )
        elif group == "image":
            # (task-3307) The image panel's OCR knobs travel under the
            # names the parse branch reads; fallbacks mirror
            # ``process_image``'s own declared defaults. OCR defaults ON:
            # the extracted text IS the imported content, and a no-text
            # parse fails honestly at the persist seam.
            if options["chunk_options"] is not None and not template_active:
                # (F12 parity) ``process_image`` chunks the OCR text via
                # ``improved_chunking_process``; an explicit words method
                # keeps the generic "words · 100-5000" size hint true here
                # too. (task 10) Skipped under a resolved template.
                options["chunk_options"]["method"] = "words"
            options["ocr"] = flat_opts.get("ocr", flat_opts.get("enable_ocr", True))
            options["ocr_language"] = flat_opts.get("ocr_language") or "en"
            options["ocr_backend"] = flat_opts.get("ocr_backend") or "auto"
        elif group == "ebook":
            options["extraction_method"] = (
                flat_opts.get("extraction_method")
                or flat_opts.get("method")
                or flat_opts.get("html_converter")
            )
            options["split_chapters"] = flat_opts.get("split_chapters", True)
            options["include_toc"] = flat_opts.get(
                "include_toc", flat_opts.get("extract_toc", True)
            )
            # (task-3303) The panel's chunk-method choice: the human
            # "chapters" maps to the chunker's real ``ebook_chapters``
            # method; the other names travel verbatim. Only meaningful when
            # chunking is on.
            ebook_chunk_method = str(flat_opts.get("chunk_method") or "").strip()
            if options["chunk_options"] is not None:
                if template_active and ebook_chunk_method in (
                    "",
                    "chapters",  # the ebook select's schema default
                ):
                    # (task 10, AC 35) The select's schema default
                    # ("chapters") is a builder default: under a resolved
                    # template the template's method wins and nothing is
                    # injected. An ABSENT value under a template also lets
                    # the template win (the template IS the scheme the
                    # user picked; there is no legacy scheme to preserve).
                    pass
                elif ebook_chunk_method:
                    options["chunk_options"]["method"] = (
                        "ebook_chapters"
                        if ebook_chunk_method == "chapters"
                        else ebook_chunk_method
                    )
                else:
                    # (task-3303 xhigh review round 2, F11) No chunk_method
                    # in the snapshot means the job PREDATES the field --
                    # fresh submissions always seed the schema default (see
                    # ``_build_ingest_options_snapshot``). The old builder
                    # forced sentences for every group, so a requeued
                    # legacy job must keep that scheme rather than silently
                    # switching to the processor's chapters default.
                    options["chunk_options"]["method"] = "sentences"

        return options

    def _create_local_stt_executor(self) -> LocalSTTExecutor:
        """Construct the one app-owned heavy STT executor lazily."""

        return LocalSTTExecutor()

    def _ensure_local_stt_executor(self) -> LocalSTTExecutor:
        with self._local_stt_executor_lock:
            if self._ingest_shutdown:
                raise ExecutorUnavailableError("Library ingest is shutting down")
            executor = getattr(self, "_local_stt_executor", None)
            if executor is None:
                executor = self._create_local_stt_executor()
                self._local_stt_executor = executor
            return executor

    def _recycle_idle_local_stt_reference(self, reference: "ArtifactRef") -> bool:
        """Recycle an existing idle STT resident that leases ``reference``."""

        with self._local_stt_executor_lock:
            if self._ingest_shutdown:
                return False
            executor = getattr(self, "_local_stt_executor", None)
        if executor is None:
            return False
        return executor.recycle_idle_managed_reference(
            (reference.artifact_id, reference.revision, reference.variant)
        )

    def _create_parakeet_source_service(self) -> Any:
        """Construct the shared download-free Parakeet source service lazily."""

        from tldw_chatbook.STT.parakeet_sources import ParakeetSourceService

        return ParakeetSourceService()

    def _ensure_parakeet_source_service(self) -> Any:
        """Return the one app-owned Parakeet source service."""

        with self._local_stt_executor_lock:
            if self._ingest_shutdown:
                raise ExecutorUnavailableError("Local STT is shutting down")
            service = getattr(self, "_parakeet_source_service", None)
            if service is None:
                service = self._create_parakeet_source_service()
                listener = self._sync_parakeet_source_scopes
                self._parakeet_source_service = service
                self._parakeet_source_registry_listener = listener
                self.library_ingest_jobs.add_listener(listener)
                listener()
            return service

    @staticmethod
    def _parakeet_scope_id_for_job(job: LibraryIngestJob) -> str:
        """Return the path-free verifier owner captured for one Library job."""

        audio_options = (job.ingest_options or {}).get("audio_video", {})
        if isinstance(audio_options, dict):
            scope_id = audio_options.get("transcription_external_scope_id")
            if isinstance(scope_id, str) and scope_id.strip():
                return scope_id.strip()
        return job.batch_id or job.job_id

    def _sync_parakeet_source_scopes(self) -> None:
        """Release only source scopes the registry observed and then settled."""

        service = getattr(self, "_parakeet_source_service", None)
        if service is None:
            return
        active_states = {
            IngestJobState.QUEUED,
            IngestJobState.PARSING,
            IngestJobState.WRITING,
        }
        active = {
            self._parakeet_scope_id_for_job(job)
            for job in self.library_ingest_jobs.jobs()
            if job.state in active_states
        }
        active.update(getattr(self, "_parakeet_submitting_scope_ids", ()))
        service.release_scopes_except(active)

    def _ensure_local_stt_dispatch_coordinator(
        self,
    ) -> LocalSTTDispatchCoordinator:
        """Return the one app-owned admission coordinator lazily."""

        with self._local_stt_executor_lock:
            if self._ingest_shutdown:
                raise ExecutorUnavailableError("Local STT is shutting down")
            executor = self._ensure_local_stt_executor()
            coordinator = getattr(self, "_local_stt_dispatch_coordinator", None)
            if coordinator is None:
                coordinator = LocalSTTDispatchCoordinator(
                    executor,
                    on_dictation_idle=lambda: self._marshal_local_stt_call(
                        self._top_up_ingest_parse_pool
                    ),
                )
                self._local_stt_dispatch_coordinator = coordinator
            return coordinator

    def _create_console_dictation_service(self, **kwargs: Any) -> Any:
        """Build Console dictation without importing its native stack eagerly."""

        from tldw_chatbook.Audio.dictation_service_lazy import (
            LazyLiveDictationService,
        )
        from tldw_chatbook.Local_Ingestion.transcription_service import (
            TranscriptionService,
        )

        app_loop_running = getattr(self, "_loop", None) is not None
        on_app_thread = threading.get_ident() == getattr(self, "_thread_id", None)
        if app_loop_running and not on_app_thread:
            source_service = self.call_from_thread(self._ensure_parakeet_source_service)
        else:
            source_service = self._ensure_parakeet_source_service()
        return LazyLiveDictationService(
            **kwargs,
            transcription_service_factory=lambda: TranscriptionService(
                local_stt_dispatcher=self._ensure_local_stt_dispatch_coordinator(),
                parakeet_source_service=source_service,
            ),
        )

    def _build_local_stt_dispatch(
        self,
        job: LibraryIngestJob,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve the exact private model identity for one eligible job."""

        provider = options["transcription_provider"]
        attempt_id = f"{job.job_id}-attempt-{job.retry_count + 1}"
        local_source = None
        managed_store_root = None
        managed_artifact_ref = None
        managed_dependency_refs: tuple[tuple[str, str, str], ...] = ()
        root_revision = None
        closure_fingerprint = None
        device = ExecutionDevice.CPU

        if provider == "transcribe-cpp":
            from tldw_chatbook.Model_Artifacts.gguf_admission import (
                validate_local_gguf,
            )

            context = options.get("transcription_context") or {}
            configured_path = (
                context.get("model_path") if isinstance(context, dict) else None
            )
            model_id = "local-gguf:unavailable"
            if isinstance(configured_path, str) and configured_path:
                admission = validate_local_gguf(Path(configured_path))
                local_source = snapshot_local_source((admission.path,))
                model_id = f"local-gguf:{admission.metadata.architecture}"
            device = ExecutionDevice.AUTO
        else:
            model_id = options.get("transcription_model") or PARAKEET_V2_MODEL
            precision = options.get("transcription_precision") or "int8"
            selected_dir = options.get("transcription_model_dir")
            from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

            resolved = self._ensure_parakeet_source_service().resolve(
                ParakeetSourceKey.from_values(model_id, precision),
                override=selected_dir,
                scope_id=self._parakeet_scope_id_for_job(job),
            )
            options.update(resolved.option_updates)
            return {
                "attempt_id": attempt_id,
                "identity": resolved.identity,
                "local_source": resolved.local_source,
                "managed_store_root": resolved.managed_store_root,
                "managed_artifact_ref": resolved.managed_artifact_ref,
                "managed_dependency_refs": resolved.managed_dependency_refs,
            }

        identity = ModelIdentity(
            provider_id=provider,
            model_id=model_id,
            root_revision=root_revision,
            closure_fingerprint=closure_fingerprint,
            precision=options.get("transcription_precision") or "int8",
            device=device,
            local_snapshot_token=(
                local_source.token if local_source is not None else None
            ),
        )
        return {
            "attempt_id": attempt_id,
            "identity": identity,
            "local_source": local_source,
            "managed_store_root": managed_store_root,
            "managed_artifact_ref": managed_artifact_ref,
            "managed_dependency_refs": managed_dependency_refs,
        }

    def _submit_local_stt_job(
        self,
        job: LibraryIngestJob,
        options: dict[str, Any],
    ) -> None:
        if options.get("transcription_provider") == "parakeet-onnx":
            self._ensure_parakeet_source_service()
        attempt_id = f"{job.job_id}-attempt-{job.retry_count + 1}"
        self._ingest_local_stt_jobs[job.job_id] = (0, attempt_id)
        thread = threading.Thread(
            target=self._dispatch_local_stt_job,
            args=(job, options, attempt_id),
            name=f"library-local-stt-dispatch-{job.job_id}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            self._ingest_local_stt_jobs.pop(job.job_id, None)
            raise

    def _dispatch_local_stt_job(
        self,
        job: LibraryIngestJob,
        options: dict[str, Any],
        attempt_id: str,
    ) -> None:
        """Build identity and perform the bounded spawn handshake off-loop."""

        try:
            dispatch = self._build_local_stt_dispatch(job, options)
            if dispatch["attempt_id"] != attempt_id:
                raise RuntimeError("Local STT attempt identity changed")
            coordinator = self._ensure_local_stt_dispatch_coordinator()
            generation = coordinator.submit_library(
                attempt_id=attempt_id,
                job_id=job.job_id,
                source=FileAudioSource(Path(job.source_path)),
                identity=dispatch["identity"],
                options=options,
                local_source=dispatch["local_source"],
                managed_store_root=dispatch["managed_store_root"],
                managed_artifact_ref=dispatch["managed_artifact_ref"],
                managed_dependency_refs=dispatch["managed_dependency_refs"],
                on_event=functools.partial(self._ingest_local_stt_event, job.job_id),
                on_result=functools.partial(self._ingest_local_stt_result, job.job_id),
                on_failure=functools.partial(
                    self._ingest_local_stt_failure, job.job_id
                ),
                explicit_retry=job.retry_count > 0,
            )
        except ExecutorBusyError:
            self._marshal_local_stt_call(
                self._on_ingest_local_stt_deferred,
                job.job_id,
                attempt_id,
            )
            return
        except Exception as exc:
            provider = str(options.get("transcription_provider") or "")
            code, actions = self._classify_local_stt_dispatch_error(provider, exc)
            self._marshal_local_stt_call(
                self._on_ingest_local_stt_dispatch_failure,
                job.job_id,
                attempt_id,
                code,
                actions,
                type(exc).__name__,
            )
            return
        self._marshal_local_stt_call(
            self._on_ingest_local_stt_submitted,
            job.job_id,
            generation,
            attempt_id,
        )

    @staticmethod
    def _classify_local_stt_dispatch_error(
        provider: str,
        error: BaseException,
    ) -> tuple[TranscriptionFailureCode, tuple[str, ...]]:
        from tldw_chatbook.STT.parakeet_sources import (
            ParakeetSourceError,
            ParakeetSourceErrorCode,
        )

        missing_model = isinstance(error, ParakeetSourceError) and error.code in {
            ParakeetSourceErrorCode.VAD_UNAVAILABLE,
            ParakeetSourceErrorCode.MANAGED_UNAVAILABLE,
        }
        unavailable = isinstance(error, (ExecutorBusyError, ExecutorUnavailableError))
        if missing_model:
            code = TranscriptionFailureCode.MODEL_NOT_INSTALLED
        elif unavailable:
            code = TranscriptionFailureCode.PROVIDER_UNAVAILABLE
        else:
            code = TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE
        actions = ["retry_faster_whisper"]
        if provider == "transcribe-cpp":
            actions.insert(0, "choose_another_gguf")
        return code, tuple(actions)

    def _marshal_local_stt_call(
        self,
        callback: Callable[..., Any],
        *args: Any,
    ) -> None:
        if self._ingest_shutdown:
            return
        try:
            self.call_from_thread(callback, *args)
        except RuntimeError:
            if not self._ingest_shutdown:
                callback_name = getattr(
                    callback,
                    "__name__",
                    type(callback).__name__,
                )
                logger.error(
                    "Library local STT callback could not be marshaled (callback={}).",
                    callback_name,
                )

    def _on_ingest_local_stt_submitted(
        self,
        job_id: str,
        generation: int,
        attempt_id: str,
    ) -> None:
        binding = self._ingest_local_stt_jobs.get(job_id)
        if (
            self._ingest_shutdown
            or binding is None
            or binding[1] != attempt_id
            or generation <= binding[0]
        ):
            return
        if self._claim_ingest_local_stt_job(job_id) is None:
            self._ingest_local_stt_jobs.pop(job_id, None)
            return
        self._ingest_local_stt_jobs[job_id] = (generation, attempt_id)

    def cancel_local_ingest_job(self, job_id: str) -> bool:
        """Request cooperative cancellation for one bound local STT attempt."""

        binding = self._ingest_local_stt_jobs.get(job_id)
        executor = getattr(self, "_local_stt_executor", None)
        job = self.library_ingest_jobs.get_job(job_id)
        if (
            binding is None
            or binding[0] <= 0
            or executor is None
            or job is None
            or job.state is not IngestJobState.PARSING
        ):
            return False
        if not executor.cancel(binding[1]):
            return False
        progress = dict(job.progress or {})
        progress["cancel_requested"] = True
        self.library_ingest_jobs.update_progress(job_id, progress=progress)
        return True

    def force_stop_local_ingest_job(self, job_id: str) -> bool:
        """Force-stop one cancel-requested local STT attempt off the UI thread."""

        binding = self._ingest_local_stt_jobs.get(job_id)
        executor = getattr(self, "_local_stt_executor", None)
        job = self.library_ingest_jobs.get_job(job_id)
        if (
            binding is None
            or binding[0] <= 0
            or executor is None
            or job is None
            or job.state is not IngestJobState.PARSING
            or not bool((job.progress or {}).get("cancel_requested"))
        ):
            return False
        thread = threading.Thread(
            target=self._force_stop_local_stt_attempt,
            args=(executor, binding[1]),
            name=f"library-local-stt-force-stop-{job_id}",
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError:
            return False
        return True

    def _force_stop_local_stt_attempt(
        self,
        executor: LocalSTTExecutor,
        attempt_id: str,
    ) -> None:
        if not executor.force_stop(attempt_id):
            return
        if executor.wait_for_retirement(10.0):
            self._marshal_local_stt_call(self._top_up_ingest_parse_pool)

    def _on_ingest_local_stt_deferred(
        self,
        job_id: str,
        attempt_id: str,
    ) -> None:
        """Release a provisional Library claim blocked by dictation admission."""

        if self._ingest_shutdown or self._ingest_local_stt_jobs.get(job_id) != (
            0,
            attempt_id,
        ):
            return
        self._ingest_local_stt_jobs.pop(job_id, None)
        self._top_up_ingest_parse_pool()

    def _claim_ingest_local_stt_job(
        self,
        job_id: str,
    ) -> LibraryIngestJob | None:
        """Publish a provisionally dispatched local-STT job as parsing."""

        current = self.library_ingest_jobs.get_job(job_id)
        if current is None:
            return None
        if current.state is IngestJobState.PARSING:
            return current
        if current.state is not IngestJobState.QUEUED:
            return None
        return self.library_ingest_jobs.mark_parsing(
            job_id,
            detected_type=current.detected_type,
        )

    def _on_ingest_local_stt_dispatch_failure(
        self,
        job_id: str,
        attempt_id: str,
        code: TranscriptionFailureCode,
        actions: tuple[str, ...],
        error_type: str,
    ) -> None:
        if self._ingest_shutdown or self._ingest_local_stt_jobs.get(job_id) != (
            0,
            attempt_id,
        ):
            return
        self._ingest_local_stt_jobs.pop(job_id, None)
        message = TRANSCRIPTION_FAILURE_CONTRACT[code][0]
        logger.error(
            "Library local STT dispatch failed "
            f"(job_id={job_id}, error_type={error_type})."
        )
        self.library_ingest_jobs.mark_failed(
            job_id,
            error=message,
            permanent=False,
            error_detail={
                "category": "stt_failure",
                "code": code.value,
                "message": message,
                "actions": list(actions),
            },
        )
        self._top_up_ingest_parse_pool()

    def _marshal_local_stt_callback(
        self,
        callback: Callable[..., Any],
        job_id: str,
        envelope: ExecutorEvent | ExecutorResult | ExecutorFailure,
    ) -> None:
        self._marshal_local_stt_call(callback, job_id, envelope)

    def _ingest_local_stt_event(self, job_id: str, event: ExecutorEvent) -> None:
        self._marshal_local_stt_callback(self._on_ingest_local_stt_event, job_id, event)

    def _ingest_local_stt_result(self, job_id: str, result: ExecutorResult) -> None:
        self._marshal_local_stt_callback(
            self._on_ingest_local_stt_result, job_id, result
        )

    def _ingest_local_stt_failure(
        self,
        job_id: str,
        failure: ExecutorFailure,
    ) -> None:
        self._marshal_local_stt_callback(
            self._on_ingest_local_stt_failure, job_id, failure
        )

    def _local_stt_callback_matches(
        self,
        job_id: str,
        envelope: ExecutorEvent | ExecutorResult | ExecutorFailure,
    ) -> bool:
        return self._ingest_local_stt_jobs.get(job_id) == (
            envelope.generation,
            envelope.attempt_id,
        )

    def _local_stt_terminal_matches(
        self,
        job_id: str,
        envelope: ExecutorResult | ExecutorFailure,
    ) -> bool:
        """Adopt the first controller-fenced generation before submit returns."""

        if self._local_stt_callback_matches(job_id, envelope):
            return True
        binding = self._ingest_local_stt_jobs.get(job_id)
        if binding == (0, envelope.attempt_id) and envelope.generation > 0:
            self._ingest_local_stt_jobs[job_id] = (
                envelope.generation,
                envelope.attempt_id,
            )
            return True
        return False

    def _on_ingest_local_stt_event(
        self,
        job_id: str,
        event: ExecutorEvent,
    ) -> None:
        if self._ingest_shutdown:
            return
        if not self._local_stt_callback_matches(job_id, event):
            binding = self._ingest_local_stt_jobs.get(job_id)
            if (
                binding is None
                or binding[1] != event.attempt_id
                or event.generation <= binding[0]
                or event.phase is not WorkerPhase.PREPARING
            ):
                return
            self._ingest_local_stt_jobs[job_id] = (
                event.generation,
                event.attempt_id,
            )
        if self._claim_ingest_local_stt_job(job_id) is None:
            return
        existing = self.library_ingest_jobs.get_job(job_id)
        progress: dict[str, Any] = {
            "phase": event.phase.value,
            "message": _INGEST_LOCAL_STT_PHASE_MESSAGES[event.phase],
        }
        if (
            existing is not None
            and (existing.progress or {}).get("cancel_requested") is True
        ):
            progress["cancel_requested"] = True
        self.library_ingest_jobs.update_progress(
            job_id,
            progress=progress,
            persist=False,
        )

    def _on_ingest_local_stt_result(
        self,
        job_id: str,
        result: ExecutorResult,
    ) -> None:
        if self._ingest_shutdown or not self._local_stt_terminal_matches(
            job_id, result
        ):
            return
        if self._claim_ingest_local_stt_job(job_id) is None:
            self._ingest_local_stt_jobs.pop(job_id, None)
            return
        self._ingest_local_stt_jobs.pop(job_id, None)
        self._ingest_parsed_payloads[job_id] = result.payload
        self._start_library_ingest_queue_if_idle()
        self._top_up_ingest_parse_pool()

    def _on_ingest_local_stt_failure(
        self,
        job_id: str,
        failure: ExecutorFailure,
    ) -> None:
        if self._ingest_shutdown or not self._local_stt_terminal_matches(
            job_id, failure
        ):
            return
        if self._claim_ingest_local_stt_job(job_id) is None:
            self._ingest_local_stt_jobs.pop(job_id, None)
            return
        self._ingest_local_stt_jobs.pop(job_id, None)
        message = TRANSCRIPTION_FAILURE_CONTRACT[failure.code][0]
        if failure.code is TranscriptionFailureCode.CANCELLED:
            self.library_ingest_jobs.mark_cancelled(job_id, reason=message)
        else:
            self.library_ingest_jobs.mark_failed(
                job_id,
                error=message,
                permanent=False,
                error_detail={
                    "category": "stt_failure",
                    "code": failure.code.value,
                    "message": message,
                    "actions": list(failure.recovery_actions),
                },
                stt_failure_provenance=failure.failed_attempt,
            )
        executor = getattr(self, "_local_stt_executor", None)
        if executor is None or not executor.retiring:
            self._top_up_ingest_parse_pool()

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
        # Read the total + heavy in-flight counts ONCE, then include local-STT
        # jobs provisionally owned by an off-loop dispatch thread. Those rows
        # remain QUEUED until coordinator admission succeeds, but still consume
        # capacity; otherwise a later top-up could overfill the pool with light
        # work while identity resolution is in flight.
        parsing_count = self.library_ingest_jobs.counts().get("parsing", 0)
        heavy_parsing_count = self.library_ingest_jobs.parsing_count_for_types(
            _INGEST_HEAVY_TYPES
        )
        provisional_local_jobs = []
        for provisional_job_id in self._ingest_local_stt_jobs:
            provisional = self.library_ingest_jobs.get_job(provisional_job_id)
            if provisional is not None and provisional.state is IngestJobState.QUEUED:
                provisional_local_jobs.append(provisional)
        parsing_count += len(provisional_local_jobs)
        heavy_parsing_count += sum(
            job.detected_type in _INGEST_HEAVY_TYPES for job in provisional_local_jobs
        )
        while parsing_count < worker_count:
            # LocalSTTExecutor intentionally accepts one request at a time.
            # A legacy heavy-lane override above one must not turn the next
            # queued audio/video job into a spurious ExecutorBusyError.
            local_stt_busy = bool(self._ingest_local_stt_jobs)
            coordinator = getattr(self, "_local_stt_dispatch_coordinator", None)
            dictation_reserved = bool(
                coordinator is not None and coordinator.dictation_reserved
            )
            heavy_full = (
                heavy_parsing_count >= heavy_cap or local_stt_busy or dictation_reserved
            )
            job = self.library_ingest_jobs.next_queued(
                skip_types=_INGEST_HEAVY_TYPES if heavy_full else frozenset()
            )
            if job is None:
                return
            try:
                options = self._ingest_job_options(job)
            except BatchSTTRoutingError as exc:
                error_text = _sanitize_library_ingest_error_text(str(exc))
                failure_text = error_text or "Batch transcription routing failed."
                logger.warning(
                    "Library ingest batch STT routing failed "
                    f"(job_id={job.job_id}, "
                    f"detected_type={job.detected_type}, "
                    f"error={failure_text})."
                )
                self.library_ingest_jobs.mark_failed(
                    job.job_id,
                    error=failure_text,
                    permanent=False,
                )
                continue
            except _template_resolution_errors() as exc:
                # (task 10, AC 37/AC-24b) A template choice that no longer
                # resolves (or a stored-invalid body) FAILS THIS ITEM with
                # the named error -- never a silent fallback to plain
                # chunking, which is how a library gets chunked two ways
                # without the user knowing. Not permanent: re-creating or
                # re-naming the template makes a retry succeed.
                failure_text = _sanitize_library_ingest_error_text(str(exc)) or (
                    "Chunking template resolution failed."
                )
                logger.warning(
                    "Library ingest template resolution failed "
                    f"(job_id={job.job_id}, "
                    f"detected_type={job.detected_type}, "
                    f"error={failure_text})."
                )
                self.library_ingest_jobs.mark_failed(
                    job.job_id,
                    error=failure_text,
                    permanent=False,
                )
                continue
            job_id = job.job_id
            source_path = job.source_path
            if options.get("transcription_provider") in {
                "parakeet-onnx",
                "transcribe-cpp",
            }:
                try:
                    self._submit_local_stt_job(job, options)
                except Exception as exc:
                    code, recovery_actions = self._classify_local_stt_dispatch_error(
                        str(options.get("transcription_provider") or ""), exc
                    )
                    message = TRANSCRIPTION_FAILURE_CONTRACT[code][0]
                    logger.error(
                        "Library local STT dispatch failed "
                        f"(job_id={job_id}, provider="
                        f"{options.get('transcription_provider')}, "
                        f"error_type={type(exc).__name__})."
                    )
                    self.library_ingest_jobs.mark_failed(
                        job_id,
                        error=message,
                        permanent=False,
                        error_detail={
                            "category": "stt_failure",
                            "code": code.value,
                            "message": message,
                            "actions": list(recovery_actions),
                        },
                    )
                    continue
                parsing_count += 1
                if job.detected_type in _INGEST_HEAVY_TYPES:
                    heavy_parsing_count += 1
                continue
            claimed = self.library_ingest_jobs.mark_parsing(
                job.job_id, detected_type=job.detected_type
            )
            if claimed is None:
                logger.error(
                    f"Library ingest coordinator: mark_parsing rejected "
                    f"job {job.job_id} (expected QUEUED) -- abandoning "
                    f"this top-up pass."
                )
                break
            parsing_count += 1
            if job.detected_type in _INGEST_HEAVY_TYPES:
                heavy_parsing_count += 1
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
                    (source_path, options, (generation, job_id)),
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

    def _marshal_ingest_pool_call(
        self,
        callback: Callable[..., Any],
        *args: Any,
    ) -> None:
        """Marshal a pool callback, tolerating only shutdown cancellation."""

        if self._ingest_shutdown:
            return
        try:
            self.call_from_thread(callback, *args)
        except concurrent.futures.CancelledError:
            if not self._ingest_shutdown:
                raise

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
        self._marshal_ingest_pool_call(
            self._on_ingest_parse_complete, generation, job_id, result
        )

    def _ingest_pool_error_callback(
        self, generation: int, job_id: str, exc: BaseException
    ) -> None:
        """``apply_async`` ``error_callback``: same thread + shutdown
        contract as ``_ingest_pool_callback`` (see its docstring)."""
        self._marshal_ingest_pool_call(
            self._handle_broken_ingest_parse_pool, generation, job_id, exc
        )

    def _on_ingest_parse_progress_batch(
        self,
        generation: int,
        events: tuple[ParseProgressEvent, ...],
    ) -> None:
        """Apply one validated progress batch for the current parse generation.

        Progress and terminal results travel on separate channels, so this
        UI-thread boundary rechecks every piece of coordinator authority after
        IPC. Unknown or malformed queue data is ignored; local live telemetry
        is projected in memory only.
        """
        if self._ingest_shutdown or generation != self._ingest_parse_pool_generation:
            return
        generation_jobs = self._ingest_parse_jobs_by_generation.get(generation)
        if generation_jobs is None:
            return

        for raw_event in events:
            try:
                event = make_parse_progress_event(
                    raw_event.generation,
                    raw_event.job_id,
                    raw_event.phase,
                    raw_event.message,
                    raw_event.percent,
                )
            except Exception:
                continue
            if event is None:
                continue
            job = self.library_ingest_jobs.get_job(event.job_id)
            if (
                event.generation != generation
                or event.job_id not in generation_jobs
                or event.job_id in self._ingest_parsed_payloads
                or job is None
                or job.state is not IngestJobState.PARSING
            ):
                continue
            progress: dict[str, Any] = {
                "phase": event.phase,
                "message": event.message,
            }
            if event.percent is not None:
                progress["percent"] = event.percent
            self.library_ingest_jobs.update_progress(
                event.job_id,
                progress=progress,
                persist=False,
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
                str(result.get("error") or "Library import parsing failed.")
            )
            error_detail = result.get("error_detail")
            # (task-2220 owner ruling) An unsupported file was never
            # attempted -- it records as SKIPPED, a neutral terminal
            # outcome; "failed" is reserved for files the pipeline tried.
            if (
                isinstance(error_detail, dict)
                and error_detail.get("category") == "unsupported_file_type"
            ):
                self.library_ingest_jobs.mark_skipped(
                    job_id,
                    reason=error_text or "Unsupported file type.",
                    error_detail=error_detail,
                )
            else:
                self.library_ingest_jobs.mark_failed(
                    job_id,
                    error=error_text or "Library import parsing failed.",
                    permanent=bool(result.get("permanent", False)),
                    error_detail=error_detail,
                    stt_failure_provenance=result.get("stt_failure_provenance"),
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
        progress_queue = self._ingest_parse_progress_queue
        progress_thread = self._ingest_parse_progress_thread
        if stop_event is not None:
            stop_event.set()
        self._ingest_parse_pool_stop_event = None
        self._ingest_parse_pool = None
        self._ingest_parse_progress_queue = None
        self._ingest_parse_progress_thread = None
        if any(
            resource is not None for resource in (pool, progress_queue, progress_thread)
        ):
            self._terminate_ingest_parse_pool_off_thread(
                pool,
                progress_queue,
                progress_thread,
            )

        logger.opt(exception=exc).error(f"Library ingest parse pool failed: {exc}")
        for job in self.library_ingest_jobs.jobs():
            if job.job_id not in affected_jobs or job.state != IngestJobState.PARSING:
                continue
            if job.job_id in self._ingest_parsed_payloads:
                # Parse already finished -- the payload is waiting for the
                # writer; the broken pool can't hurt this job anymore.
                continue
            self.library_ingest_jobs.mark_failed(
                job.job_id,
                error="Library import parse pool failed unexpectedly; retry to resume.",
                permanent=False,
            )
        if self._ingest_parsed_payloads:
            self._start_library_ingest_queue_if_idle()

    @staticmethod
    def _terminate_ingest_parse_pool_off_thread(
        pool: Any | None,
        progress_queue: Any | None = None,
        progress_thread: threading.Thread | None = None,
    ) -> threading.Thread:
        """Clean up one detached parse generation away from the UI thread."""
        return LibraryIngestQueueMixin._shutdown_ingest_workers_off_thread(
            None,
            None,
            None,
            pool,
            progress_queue,
            progress_thread,
        )

    def _shutdown_ingest_parse_pool(self) -> Optional[threading.Thread]:
        """Quit-path teardown: flag up, pool detached, terminate off-loop.

        Called from ``TldwCli.on_unmount`` (i.e. on the app's event-loop
        thread). Synchronously: sets ``_ingest_shutdown = True`` FIRST (so
        pool callbacks -- ``_ingest_pool_callback``/
        ``_ingest_pool_error_callback``, running on the pool's
        result-handler thread -- short-circuit before marshaling from this
        point on) and drops every worker reference (nothing can submit to
        them anymore). Source/coordinator/executor close, parse-pool
        terminate/join, queue cleanup, and bounded drain-thread join then run
        sequentially on one detached daemon thread,
        NEVER on the caller's (loop) thread: verifier close may wait and
        CPython's ``Pool._terminate_pool`` does an unbounded
        ``result_handler.join()``, and if that result-handler thread is at
        that moment parked inside a ``call_from_thread`` it entered just
        before the flag went up, joining it from the loop thread would
        deadlock (the loop can't drain the marshaled call it is itself waiting
        behind). Off-loop, the loop stays free: the in-flight marshaled call
        runs, no-ops via the flag, the result-handler thread unblocks, and the
        join completes. The daemon thread is deliberately not joined by the
        caller -- worst case it outlives the app briefly and dies with the
        process.

        Returns:
            The one teardown thread that owns every detached ingest resource,
            or ``None`` when no ingest resource was ever created. The
            shutdown flag is still set in that case.
        """
        self._ingest_shutdown = True
        with self._local_stt_executor_lock:
            source_service = getattr(self, "_parakeet_source_service", None)
            source_listener = getattr(self, "_parakeet_source_registry_listener", None)
            coordinator = getattr(self, "_local_stt_dispatch_coordinator", None)
            executor = getattr(self, "_local_stt_executor", None)
            self._parakeet_source_service = None
            self._parakeet_source_registry_listener = None
            self._local_stt_dispatch_coordinator = None
            self._local_stt_executor = None
            if source_listener is not None:
                self.library_ingest_jobs.remove_listener(source_listener)
        local_jobs = getattr(self, "_ingest_local_stt_jobs", None)
        if local_jobs is None:
            self._ingest_local_stt_jobs = {}
        else:
            local_jobs.clear()
        pool = getattr(self, "_ingest_parse_pool", None)
        stop_event = getattr(self, "_ingest_parse_pool_stop_event", None)
        progress_queue = getattr(self, "_ingest_parse_progress_queue", None)
        progress_thread = getattr(self, "_ingest_parse_progress_thread", None)
        if stop_event is not None:
            stop_event.set()
        self._ingest_parse_pool_stop_event = None
        self._ingest_parse_pool = None
        self._ingest_parse_progress_queue = None
        self._ingest_parse_progress_thread = None
        if all(
            resource is None
            for resource in (
                source_service,
                coordinator,
                executor,
                pool,
                progress_queue,
                progress_thread,
            )
        ):
            return None
        return self._shutdown_ingest_workers_off_thread(
            source_service,
            coordinator,
            executor,
            pool,
            progress_queue,
            progress_thread,
        )

    @staticmethod
    def _shutdown_ingest_workers_off_thread(
        source_service: Any | None,
        coordinator: Any | None,
        executor: Any | None,
        pool: Any | None,
        progress_queue: Any | None,
        progress_thread: threading.Thread | None,
    ) -> threading.Thread:
        """Close detached ingest workers without blocking the UI thread.

        Executor shutdown remains ahead of parse-pool teardown. The parse pool
        is terminated and joined before its queue is closed/cancelled, then the
        already-stopped daemon drain receives only a bounded join.
        """

        def _shutdown_workers() -> None:
            if source_service is not None:
                try:
                    source_service.close()
                except Exception:
                    logger.error("Error closing the Parakeet source service.")
            if coordinator is not None:
                try:
                    coordinator.close()
                except Exception:
                    logger.error("Error closing the local STT dispatch coordinator.")
            if executor is not None:
                try:
                    executor.close()
                except Exception:
                    logger.opt(exception=True).error(
                        "Error closing the Library local STT executor."
                    )
            if pool is not None:
                try:
                    pool.terminate()
                    pool.join()
                except Exception:
                    logger.opt(exception=True).error(
                        "Error terminating the Library ingest parse pool."
                    )
            if progress_queue is not None:
                close = getattr(progress_queue, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        logger.error(
                            "Error cleaning up the Library ingest progress queue "
                            "(operation={}, queue_type={}).",
                            "close",
                            type(progress_queue).__name__,
                        )
                cancel_join = getattr(progress_queue, "cancel_join_thread", None)
                if cancel_join is not None:
                    try:
                        cancel_join()
                    except Exception:
                        logger.error(
                            "Error cleaning up the Library ingest progress queue "
                            "(operation={}, queue_type={}).",
                            "cancel_join_thread",
                            type(progress_queue).__name__,
                        )
            if progress_thread is not None:
                try:
                    progress_thread.join(timeout=1.0)
                except Exception:
                    logger.error(
                        "Error joining the Library ingest progress drain thread."
                    )

        thread = threading.Thread(
            target=_shutdown_workers,
            name="library-ingest-workers-shutdown",
            daemon=True,
        )
        thread.start()
        return thread

    # -- Remote poller (server-origin jobs) --------------------------------

    #: Seconds between remote status polls. Server ingests are minutes-long
    #: (transcription, OCR), so a slow cadence is plenty and keeps this off the
    #: server's back.
    REMOTE_INGEST_POLL_SECONDS: float = 5.0

    #: Cap on status pages fetched per batch per pass, so a server
    #: reporting has_more forever cannot pin the loop.
    REMOTE_INGEST_MAX_PAGES: int = 20

    def _resolve_ingest_backend(self) -> str:
        """Return the backend a new ingest should run on: ``local`` or ``server``.

        Deliberately its **own** preference rather than the Media destination's
        browse scope. Reusing the browse scope looked tidier -- one notion of
        "which backend am I on" -- but it
        would mean a user who switched scope to look at server-side media and
        then imported a file had that file leave their machine without ever
        asking for it. ``build_library_ingest_state``'s own contract is explicit
        that ingest "always targets the local media store regardless of browsing
        scope", and quietly inverting that is not a change to make on the user's
        behalf.

        So sending an ingest to a server is an explicit opt-in, and anything
        unrecognised or unset means local -- the backend that always works and
        keeps the file where it already is.
        """
        raw = get_cli_setting("library.ingest", "backend", "local")
        if str(raw or "local").strip().lower() != "server":
            return "local"
        # The opt-in is necessary but not sufficient. Runtime policy declares
        # ``media.ingestion_jobs.launch.server`` as ``required_source="server"``,
        # so the service refuses the launch while the Library runtime is local.
        # Honouring that here means an opted-in user whose runtime is local gets
        # a local ingest -- the file stays put and the canvas explains how to
        # enable server imports -- rather than a job that fails with "requires
        # server mode" (seen live against a real server).
        runtime_state = getattr(getattr(self, "runtime_policy", None), "state", None)
        active_source = (
            str(getattr(runtime_state, "active_source", "local") or "local")
            .strip()
            .lower()
        )
        return "server" if active_source == "server" else "local"

    def _submit_server_ingest_job(
        self,
        *,
        source_path: str,
        ingest_options: dict[str, Any],
        title: str,
        author: str,
        keywords: tuple[str, ...],
        perform_analysis: bool,
    ) -> LibraryIngestJob:
        """Queue a ``server``-origin job and send it to the server.

        The registry row is created synchronously so the queue shows the job the
        moment the user starts it, then an async worker performs the submission
        and records the ids the server issues. A source the server has no
        handler for fails immediately, with the reason, rather than being sent
        and rejected later.

        Returns:
            The queued job, or an already-``FAILED`` one when the source cannot
            be sent at all.
        """
        try:
            kwargs = build_server_ingest_kwargs(
                source_path,
                options=ingest_options,
                title=title,
                author=author,
                keywords=keywords,
                perform_analysis=perform_analysis,
            )
        except ServerIngestUnsupported as exc:
            job = self.library_ingest_jobs.submit(
                source_path=source_path,
                title=title,
                author=author,
                keywords=keywords,
                perform_analysis=perform_analysis,
                origin="server",
                ingest_options=ingest_options,
            )
            failed = self.library_ingest_jobs.mark_failed(
                job.job_id, error=str(exc), permanent=True
            )
            return failed if failed is not None else job

        job = self.library_ingest_jobs.submit(
            source_path=source_path,
            title=title,
            author=author,
            keywords=keywords,
            perform_analysis=perform_analysis,
            detected_type=str(kwargs.get("media_type") or ""),
            origin="server",
            ingest_options=ingest_options,
        )
        self._send_server_ingest_job(job.job_id, kwargs)
        return job

    def _submit_web_clip_job(
        self,
        *,
        source_path: str,
        ingest_options: dict[str, Any],
        title: str,
        author: str,
        keywords: tuple[str, ...],
        perform_analysis: bool,
    ) -> LibraryIngestJob:
        """Queue a ``server``-origin job that clips a web page.

        A page cannot go through the ingest-jobs API -- it has no media type for
        one -- so this uses the clipper endpoint instead. That endpoint is
        synchronous and issues no job or batch id, so unlike a server file
        ingest there is nothing to attach or poll: the job settles when the call
        returns (task-684.3).

        Returns:
            The queued job, or an already-``FAILED`` one when the source cannot
            be clipped at all.
        """
        try:
            kwargs = build_web_clip_kwargs(
                source_path,
                options=ingest_options,
                title=title,
                author=author,
                keywords=keywords,
            )
        except NotAWebClipSource as exc:
            job = self.library_ingest_jobs.submit(
                source_path=source_path,
                title=title,
                author=author,
                keywords=keywords,
                perform_analysis=perform_analysis,
                origin="server",
                ingest_options=ingest_options,
            )
            failed = self.library_ingest_jobs.mark_failed(
                job.job_id, error=str(exc), permanent=True
            )
            return failed if failed is not None else job

        job = self.library_ingest_jobs.submit(
            source_path=source_path,
            title=title,
            author=author,
            keywords=keywords,
            perform_analysis=perform_analysis,
            detected_type="web",
            origin="server",
            ingest_options=ingest_options,
        )
        self._send_web_clip_job(job.job_id, kwargs)
        return job

    @work(group="library_ingest_remote_submit")
    async def _send_web_clip_job(self, job_id: str, kwargs: dict[str, Any]) -> None:
        """Clip a page on the server and settle the job on the answer.

        Shares the submit worker group with ``_send_server_ingest_job``: both are
        one-shot submissions on the user's behalf, and neither should be able to
        pile up.
        """
        service = getattr(self, "server_media_reading_service", None)
        clip = getattr(service, "ingest_web_content", None)
        if not callable(clip):
            self.library_ingest_jobs.mark_failed(
                job_id,
                error=(
                    "No server backend is configured, so this page cannot be "
                    "clipped on the server. Configure one in Settings, or switch "
                    "this Library to Local."
                ),
                permanent=True,
            )
            return

        self.library_ingest_jobs.mark_parsing(job_id, detected_type="web")
        try:
            response = await clip(**kwargs)
        except Exception as exc:
            logger.opt(exception=True).warning(f"Web clip failed for job {job_id}.")
            self.library_ingest_jobs.mark_failed(
                job_id, error=f"The server could not clip the page: {exc}"
            )
            return

        # A 200 is not a captured page: the endpoint reports its outcome in the
        # body, so an extraction that found nothing arrives as success.
        reason = clip_failure_reason(response)
        if reason is not None:
            self.library_ingest_jobs.mark_failed(job_id, error=reason)
            return

        # No media id comes back, so this finishes like a remote job: done, with
        # "Open in Library" withheld because the content is in the server's.
        self.library_ingest_jobs.mark_remote_done(job_id)

    @work(group="library_ingest_remote_submit")
    async def _send_server_ingest_job(
        self, job_id: str, kwargs: dict[str, Any]
    ) -> None:
        """Submit to the server, then attach the ids it issued.

        Async for the same reason the poller is: the service call is a
        coroutine, so staying on the event loop keeps every registry mutation
        on the UI thread without marshalling.
        """
        service = getattr(self, "server_media_reading_service", None)
        submit = getattr(service, "submit_ingest_jobs", None) or getattr(
            service, "submit_media_ingest_jobs", None
        )
        if not callable(submit):
            self.library_ingest_jobs.mark_failed(
                job_id,
                error=(
                    "No server backend is configured, so this import cannot run "
                    "on the server. Configure one in Settings, or switch this "
                    "Library to Local."
                ),
                permanent=True,
            )
            return

        try:
            response = await submit(**kwargs)
        except Exception as exc:
            logger.opt(exception=True).warning(
                f"Server ingest submission failed for job {job_id}."
            )
            self.library_ingest_jobs.mark_failed(
                job_id, error=f"The server refused the import: {exc}"
            )
            return

        batch_id = _response_field(response, "batch_id")
        jobs = _response_field(response, "jobs") or []
        remote_job_id = None
        if jobs:
            remote_job_id = _response_field(jobs[0], "id")
        self.library_ingest_jobs.attach_remote(
            job_id,
            remote_job_id=None if remote_job_id is None else str(remote_job_id),
            batch_id=None if batch_id is None else str(batch_id),
        )
        errors = _response_field(response, "errors") or []
        if errors and not jobs:
            self.library_ingest_jobs.mark_failed(
                job_id, error=f"The server rejected the import: {errors[0]}"
            )
            return

        # Following a remote job needs BOTH ids: ``pending_remote_batches``
        # decides what to poll from ``batch_id``, and the reconciler matches
        # statuses to jobs by ``remote_job_id``. Without the first, the job is
        # never polled; without the second, the batch is polled forever while no
        # status can ever be matched to it. Either way the row sits at "queued"
        # indefinitely -- the same never-resolves failure the mistyped ``result``
        # field caused, and not something a queue may do quietly.
        if not batch_id or remote_job_id is None:
            self.library_ingest_jobs.mark_failed(
                job_id,
                error=(
                    "The server accepted this import but did not return the ids "
                    "needed to track it, so its progress cannot be followed. It "
                    "may still be running on the server; check there before "
                    "importing again."
                ),
                permanent=True,
            )
            return

        self.poll_remote_ingest_jobs()

    async def _reconcile_remote_batch(self, service: Any, batch_id: str) -> None:
        """Fetch every page of ``batch_id``'s statuses and reconcile them.

        The server's list response is paginated (``has_more``/``next_offset``,
        per its OpenAPI schema). Reading only the first page would leave later
        jobs unreconciled -- and since they stay unsettled, the poller would
        keep re-fetching that batch forever, which is what the stop condition
        exists to prevent.

        A transient failure is logged and left for the next pass rather than
        killing the poller; the jobs stay visibly unfinished meanwhile, which is
        the recoverable direction.
        """
        lister = service.list_media_ingest_jobs
        supports_offset = _accepts_keyword(lister, "offset")

        offset = 0
        for _ in range(self.REMOTE_INGEST_MAX_PAGES):
            if self._ingest_shutdown:
                return
            try:
                response = (
                    await lister(batch_id, offset=offset)
                    if supports_offset
                    else await lister(batch_id)
                )
            except Exception:
                logger.opt(exception=True).debug(
                    f"Remote ingest poll failed for batch {batch_id!r}; "
                    "will retry on the next pass."
                )
                return

            self._reconcile_page(response)
            if not _response_field(response, "has_more"):
                return
            if not supports_offset:
                # Without an offset there is no way to ask for page two, and
                # re-asking would just re-read page one until the cap.
                logger.debug(
                    f"Batch {batch_id!r} has more statuses but "
                    f"{type(service).__name__}.list_media_ingest_jobs takes no "
                    "offset; reconciled the first page only."
                )
                return
            next_offset = _response_field(response, "next_offset")
            if next_offset is None or next_offset == offset:
                # Server says there is more but gives no way forward; stop
                # rather than spin on the same page.
                logger.debug(
                    f"Batch {batch_id!r} reports has_more with no usable "
                    "next_offset; stopping pagination."
                )
                return
            offset = int(next_offset)
        else:
            logger.warning(
                f"Batch {batch_id!r} exceeded {self.REMOTE_INGEST_MAX_PAGES} "
                "pages of statuses; the rest will be picked up next pass."
            )

    def _reconcile_page(self, response: Any) -> None:
        """Hand one page of statuses to the reconciler, if it has any."""
        statuses = _response_field(response, "jobs")
        if statuses:
            reconcile_remote_ingest_jobs(self.library_ingest_jobs, statuses)

    def cancel_remote_ingest_batch(self, batch_id: str) -> None:
        """Ask the server to cancel every job in ``batch_id``.

        UI-thread entry point. Deliberately does *not* mark the local jobs
        cancelled: the request is asynchronous and may be refused, so the queue
        must not claim an outcome the server has not confirmed. The poller
        records the real state when the server reports it -- which is also why
        polling is (re)started here, so a batch cancelled while nothing was
        being watched still gets its outcome.

        Args:
            batch_id: The server batch to cancel. An empty value is ignored, so
                a queue row that never received a batch id cannot send a cancel
                for every job on the server.
        """
        if not batch_id:
            return
        self._request_remote_ingest_cancel(batch_id)
        self.poll_remote_ingest_jobs()

    @work(group="library_ingest_remote_cancel")
    async def _request_remote_ingest_cancel(self, batch_id: str) -> None:
        """Send the cancel request. Async for the same reason the poller is."""
        service = getattr(self, "server_media_reading_service", None)
        cancel = getattr(service, "cancel_media_ingest_jobs_batch", None)
        if not callable(cancel):
            logger.debug(
                "Remote ingest cancel requested but no server seam is available."
            )
            return
        try:
            # Keyword-only on both the client and the service wrapper; a
            # positional call raises TypeError at runtime.
            await cancel(batch_id=batch_id)
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to cancel remote ingest batch {batch_id!r}."
            )
            self.notify(
                "Could not reach the server to cancel that import.",
                severity="warning",
            )

    def poll_remote_ingest_jobs(self) -> None:
        """Start watching server-origin ingest jobs, if any are outstanding.

        Idempotent: the worker is ``exclusive`` within its own group, so calling
        this again while a poll loop is already running is a no-op rather than a
        second poller.
        """
        if not pending_remote_batches(self.library_ingest_jobs):
            return
        self._run_remote_ingest_poll()

    @work(exclusive=True, group="library_ingest_remote_poll")
    async def _run_remote_ingest_poll(self) -> None:
        """Poll server ingest batches until none are outstanding.

        Deliberately an **async** worker rather than a thread worker. The
        service calls are already coroutines, and running on the event loop
        means every registry mutation here is already on the UI thread -- so
        this needs no ``call_from_thread`` at all, and therefore cannot hit the
        quit-path deadlock documented on ``_ingest_pool_callback`` (that
        marshal blocks the calling thread and does not observe loop shutdown).
        The ``await`` points are network I/O and a sleep, neither of which
        blocks the loop.

        Exits when every server batch has settled, on shutdown, or when the
        server seam is unavailable -- never spins on an answer that cannot
        change.
        """
        service = getattr(self, "server_media_reading_service", None)
        if service is None:
            logger.debug("Remote ingest poll: no server media service; not polling.")
            return

        while not self._ingest_shutdown:
            batches = pending_remote_batches(self.library_ingest_jobs)
            if not batches:
                return

            for batch_id in batches:
                if self._ingest_shutdown:
                    return
                await self._reconcile_remote_batch(service, batch_id)

            await asyncio.sleep(self.REMOTE_INGEST_POLL_SECONDS)

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
                    generic_options = (job.ingest_options or {}).get("generic", {})
                    overwrite_existing = bool(
                        generic_options.get(
                            "overwrite_existing",
                            generic_option_default("overwrite_existing", False),
                        )
                        if isinstance(generic_options, dict)
                        else generic_option_default("overwrite_existing", False)
                    )
                    generate_embeddings = bool(
                        generic_options.get(
                            "generate_embeddings",
                            generic_option_default("generate_embeddings", True),
                        )
                        if isinstance(generic_options, dict)
                        else generic_option_default("generate_embeddings", True)
                    )
                    media_id, _media_uuid, _message = persist_parsed_media(
                        payload,
                        self.media_db,
                        overwrite_existing=overwrite_existing,
                        generate_embeddings=generate_embeddings,
                    )
                    # ``add_media_with_keywords`` returns ``media_id=None`` on
                    # exactly one success path: the duplicate skip ("already
                    # exists. Overwrite not enabled."). A same-path re-ingest
                    # resolves by canonical URL; a byte-identical file at a
                    # DIFFERENT path has a different URL, so fall back to the
                    # content hash -- otherwise the row is a done-without-
                    # media_id husk with no "Open in Library" and nothing
                    # telling the user the file was already there (task-2013).
                    # ``self.media_db`` is unreachable-``None`` here in
                    # practice (submit already fails the job before this point
                    # when it's absent), but the guard is cheap insurance
                    # against an ``AttributeError`` on a stale/racy reference.
                    was_duplicate = media_id is None
                    content_hash = payload.get("content_hash")
                    if media_id is None and self.media_db is not None:
                        existing = self.media_db.get_media_by_url(payload["url"])
                        if existing is None:
                            if content_hash is None and isinstance(
                                payload.get("content"), str
                            ):
                                # The parse payload carries no hash; the DB
                                # computes sha256(content) itself inside
                                # ``add_media_with_keywords``. Mirror that
                                # exact computation, but only on this
                                # duplicate-with-URL-miss path -- never on
                                # the plain success path, which runs on the
                                # single-SQLite-writer critical path and
                                # would pay a second O(n) pass per file.
                                content_hash = hashlib.sha256(
                                    payload["content"].encode()
                                ).hexdigest()
                            if content_hash:
                                try:
                                    existing = self.media_db.get_media_by_hash(
                                        content_hash
                                    )
                                except (
                                    MediaDatabaseError,
                                    MediaInputError,
                                ) as exc:
                                    # The media row exists (the DB deduped
                                    # against it), so a failed lookup must
                                    # not fail the job -- but a silent miss
                                    # leaves a DONE row with no media_id and
                                    # no diagnostic trail.
                                    logger.warning(
                                        "Library ingest duplicate-resolution "
                                        "hash lookup failed "
                                        f"(job_id={job.job_id}, "
                                        f"source={job.source_path}, "
                                        f"hash={content_hash[:12]}…): {exc}"
                                    )
                                    existing = None
                        if existing is not None:
                            media_id = existing.get("id")
                            if content_hash is None:
                                content_hash = existing.get("content_hash")
                    # (task-3301) Includes the "analysis skipped: ..."
                    # annotation when the payload carries a skip reason.
                    progress = _library_ingest_done_progress(
                        job.source_path,
                        was_duplicate=was_duplicate,
                        payload=payload,
                    )
                    self.call_from_thread(
                        self.library_ingest_jobs.mark_done,
                        job.job_id,
                        media_id=media_id,
                        progress=progress,
                        content_hash=content_hash,
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
                        error_detail={
                            # (task-14821 / xhigh review round) The stage
                            # covers two different things: refusing an
                            # empty extraction (BEFORE any write) and a
                            # genuine database write failure. Stamping
                            # both "write_error" told users nothing was
                            # saved because of a write problem when there
                            # had been nothing to save -- and, since
                            # "write_error" is the one category that still
                            # earns the optimistic retry advisory, the
                            # blanket DEFAULT smuggled that advisory back
                            # in for every unclassified cause.
                            "category": _library_ingest_write_failure_category(exc),
                            "message": str(exc),
                            "exception_type": exc.__class__.__name__,
                        },
                    )
        finally:
            if not clean_exit:
                self.call_from_thread(self._release_ingest_runner_after_crash)


# --- Main App ---
def _build_generated_video_store():
    from tldw_chatbook.Video_Generation.video_store import VideoStore

    store = VideoStore()
    try:
        store.enforce_retention()
    except Exception as exc:
        logger.warning(
            "Generated-video startup retention failed (error_type={}).",
            type(exc).__name__,
        )
    return store


def _build_notes_scope_service(
    *,
    chachanotes_db: Any,
    local_notes_service: Any,
    server_service: Any,
    policy_enforcer: Any,
    sync_scope_service: Any,
) -> NotesScopeService:
    """Compose the Notes facade over the shared local database.

    Args:
        chachanotes_db: Existing local ChaChaNotes database handle, if available.
        local_notes_service: Local flat-note service implementation.
        server_service: Server-backed Notes service implementation.
        policy_enforcer: Authorization policy enforcer shared by the app.
        sync_scope_service: Optional Sync-v2 scope service.

    Returns:
        A Notes scope facade with one shared local folder repository.
    """
    folder_repository = (
        LocalNoteFolderRepository(chachanotes_db)
        if chachanotes_db is not None
        else None
    )
    return NotesScopeService(
        local_notes_service=local_notes_service,
        server_service=server_service,
        policy_enforcer=policy_enforcer,
        sync_scope_service=sync_scope_service,
        folder_repository=folder_repository,
    )


_SETUP_STARTUP_NETWORKING_ACTIONS = frozenset({"offer", "prompt", "home"})


def setup_owns_startup_networking(
    app_config: Mapping[str, Any], environ: Mapping[str, str]
) -> bool:
    """Return whether first-run setup owns automatic networking this startup."""
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        setup_recovery_action,
    )

    return (
        setup_recovery_action(app_config, environ) in _SETUP_STARTUP_NETWORKING_ACTIONS
    )


def _select_profile_database(notes_service: object | None) -> Any:
    """Return the seeded injected profile DB, or the seeded lazy global DB."""
    injected = getattr(notes_service, "db", None)
    return seed_builtin_content(injected) if injected else get_chachanotes_db_lazy()


class TldwCli(
    # TextSelectionCrashGuard sits before App so its on_event wrapper is the
    # last line of defense against Textual 8.x's text-selection MouseDown
    # crash on a mid-recompose widget (task-14903) -- see the mixin's module
    # docstring for the signature it (and ONLY it) drops.
    TextSelectionCrashGuard,
    LibraryIngestQueueMixin,
    App[None],
):  # Specify return type for run() if needed, None is common
    """A Textual app for interacting with LLMs."""

    _runtime_policy_projection_snapshot: tuple[str, str | None] = ("local", None)

    def action_command_palette(self) -> None:
        """Open the app's stable Textual command palette."""
        if self.use_command_palette and not StableCommandPalette.is_open(self):
            self.push_screen(StableCommandPalette(id="--command-palette"))

    @property
    def current_runtime_backend(self) -> str:
        return self._runtime_policy_projection_snapshot[0]

    @property
    def runtime_backend(self) -> str:
        return self._runtime_policy_projection_snapshot[0]

    @property
    def active_server_id(self) -> str | None:
        return self._runtime_policy_projection_snapshot[1]

    def _publish_runtime_policy_projection(
        self,
        state: RuntimeSourceState,
    ) -> None:
        self._runtime_policy_projection_snapshot = (
            state.active_source,
            state.active_server_id,
        )

    # Product name shown in the terminal title (legacy "tldw CLI" retired).
    TITLE = "tldw chatbook"
    # CSS file paths, read in order. The screen/modal CSS lifted out of Python
    # (TASK-15450) brackets the bundle: the scope-prefixed stream first, so it
    # loses the specificity ties that writing the scope selector out created,
    # and the self stream last, where Textual used to append a screen's `CSS` on
    # first open. They stay separate files, not bundle modules, because Textual
    # accumulates `$variable` definitions per source and several of these blocks
    # carry local `$ds-*` fallbacks that would otherwise clobber the real design
    # tokens for the rest of the bundle. See css/build_css.py.
    CSS_PATH = [
        str(build_css.screen_css_paths(Path(__file__).parent / "css")[0]),
        str(Path(__file__).parent / "css/tldw_cli_modular.tcss"),
        str(build_css.screen_css_paths(Path(__file__).parent / "css")[1]),
    ]

    def _get_default_css(self) -> list[tuple[tuple[str, str], str, int, str]]:
        """Add the consolidated widget-defaults stylesheet as one CSS source.

        TASK-15450: Textual registers a separate stylesheet source per widget
        class that declares ``DEFAULT_CSS``, and its parse cache is an
        ``LRUCache(64)``. A full destination tour used to end at 94 sources, past
        which *every* ``Stylesheet.parse()`` ran fully cold (125-380 ms measured)
        on each first mount of a not-yet-seen widget class. The widget CSS now
        lives in ``css/widget_defaults.tcss``, generated from the class-level
        ``BUNDLED_CSS`` declarations by ``build_css.py``, and is registered here
        as a single source.

        The sheets are added here (rather than as a plain ``DEFAULT_CSS`` class
        attribute) for two reasons: they are read at app start, so a boot-time
        CSS rebuild is picked up by the same run, and each needs its own
        tie-breaker. Selectors that already named their own widget keep
        tie-breaker 0, the cascade position their class's ``DEFAULT_CSS`` had.
        Selectors that gained a written-out scope prefix cost one specificity
        point more than Textual's injected one, so they take a tie-breaker below
        every other default-CSS source and lose the ties that shift created --
        which are exactly the ties they used to lose outright. See
        ``css/widget_css.py`` for the derivation.

        Returns:
            The default-CSS stack, widget defaults first.
        """
        css_dir = Path(__file__).parent / "css"
        sources = build_css.widget_defaults_sources(css_dir)
        if len(sources) != 2:
            # Never fatal: the app still runs, just with unstyled widgets whose
            # CSS was consolidated. Loud, because that is a build/packaging bug,
            # not a user-facing condition.
            loguru_logger.error(
                "Consolidated widget CSS incomplete: generated sheet count {}",
                len(sources),
            )
        return sources + super()._get_default_css()

    # Shell destination hotkey layer: Ctrl+1..Ctrl+9 then Ctrl+0, zipped against
    # SHELL_DESTINATION_ORDER, plus F7/F8/F9 for the remaining destinations
    # (Lab, Logs, Settings) so every destination has a keyboard route.
    SHELL_DESTINATION_HOTKEYS: tuple[str, ...] = tuple(
        f"ctrl+{digit}" for digit in "1234567890"
    )
    SHELL_DESTINATION_FKEYS: tuple[str, ...] = ("f7", "f8", "f9")
    BINDINGS = (
        [
            Binding("ctrl+q", "quit", "Quit App", show=True),
            Binding("ctrl+p", "command_palette", "Palette Menu", show=True),
            Binding("f1", "show_workbench_help", "Help", show=True),
            Binding("f6", "focus_next_workbench_pane", "Next Pane", show=True),
            Binding(
                "ctrl+shift+f",
                FOCUS_TOGGLE_PALETTE_ENTRY[1],
                "Focus Mode",
                show=False,
            ),
        ]
        + [
            Binding(
                key,
                f"shell_destination({index})",
                f"Go to {destination.accessible_label}",
                show=False,
            )
            for index, (key, destination) in enumerate(
                zip(SHELL_DESTINATION_HOTKEYS, SHELL_DESTINATION_ORDER)
            )
        ]
        + [
            # 10 = len of the ctrl+digit layer ("1234567890"); a class-body
            # comprehension cannot reference class-level names, so this stays a
            # literal.
            Binding(
                key,
                f"shell_destination({10 + fkey_index})",
                f"Go to {destination.accessible_label}",
                show=True,
            )
            for fkey_index, (key, destination) in enumerate(
                zip(
                    SHELL_DESTINATION_FKEYS,
                    SHELL_DESTINATION_ORDER[len("1234567890") :],
                )
            )
        ]
    )
    COMMANDS = App.COMMANDS | {
        ThemeProvider,
        TabNavigationProvider,
        LLMProviderProvider,
        QuickActionsProvider,
        SettingsProvider,
        CharacterProvider,
        MediaProvider,
        LibraryIngestProvider,
        SetupWizardProvider,
        DeveloperProvider,
        ConsoleCommandProvider,
        ImageGenCommandProvider,
    }

    # T169: "notes-window" removed -- no widget composes that id anymore (the
    # standalone Notes tab / Notes_Window.py it belonged to is gone, replaced
    # by the Library workbench's Notes canvas), confirmed via
    # `grep -rn 'id="notes-window"' tldw_chatbook/`.
    ALL_MAIN_WINDOW_IDS = [  # Assuming these are your main content window IDs
        # task-577 T4: "chat-window" removed -- id composed nowhere (the
        # ChatWindowEnhanced surface that owned it was retired in T1/T2).
        "conversations_characters_prompts-window",
        "ingest-window",
        "tools_settings-window",
        "llm_management-window",
        "media-window",
        "search-window",
        "logs-window",
        "stats-window",
        "coding-window",
        "stts-window",
        "study-window",
        "chatbooks-window",
    ]

    # Define reactive at class level with a placeholder default and type hint
    current_tab: reactive[str] = reactive("")

    # Splash screen state
    splash_screen_active: reactive[bool] = reactive(False)
    _splash_screen_widget: Optional[SplashScreen] = None

    # --- REACTIVES FOR PROVIDER SELECTS ---
    # Initialize with a dummy value or fetch default from config here
    # Ensure the initial value matches what's set in compose/settings_sidebar
    # Fetching default provider from config:
    _default_rag_expansion_provider = APP_CONFIG.get("chat_defaults", {}).get(
        "provider", "OpenAI"
    )

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

    # DB size/token status updates go to the per-screen shell status line;
    # the DBStatusManager resolves the visible widget on the active screen.
    # DB Size checker - now using AppFooterStatus
    _db_size_status_widget: Optional[AppFooterStatus] = None
    # DB size update timer moved to DBStatusManager
    _token_count_update_timer: Optional[Timer] = None
    ui_responsiveness_monitor: UIResponsivenessMonitor | None = None
    _ui_responsiveness_heartbeat_timer: Optional[Timer] = None

    # Media services and type catalog
    _media_types_for_ui: List[str] = []

    # Add media_types_for_ui to store fetched types
    media_types_for_ui: List[str] = []
    media_db: Optional[MediaDatabase] = None
    selected_note_files_for_import: List[Path]
    parsed_notes_for_preview: List[Dict[str, Any]] = []
    last_note_import_dir: Optional[Path] = None
    # Add attributes to hold the handlers (optional, but can be useful)
    note_import_success_handler: Optional[Callable] = None
    note_import_failure_handler: Optional[Callable] = None

    _prompt_search_timer: Optional[Timer] = None

    llamacpp_server_process: Optional[subprocess.Popen] = None
    llamafile_server_process: Optional[subprocess.Popen] = None
    vllm_server_process: Optional[subprocess.Popen] = None
    ollama_server_process: Optional[subprocess.Popen] = None
    mlx_server_process: Optional[subprocess.Popen] = None
    onnx_server_process: Optional[subprocess.Popen] = None

    # Make API_IMPORTS_SUCCESSFUL accessible if needed by old methods or directly
    API_IMPORTS_SUCCESSFUL = API_IMPORTS_SUCCESSFUL

    # User ID for notes, will be initialized in __init__
    current_user_id: str = "default_user"  # Will be overridden by self.notes_user_id

    def __init__(self):
        # Track startup timing
        self._startup_start_time = time.perf_counter()
        self._startup_phases = {}
        self.server_credential_store_unavailable_reason: str | None = None

        # Tab switching optimization
        self._initialized_tabs = set()  # Track which tabs have been initialized

        # Reduce logging in production
        if not os.environ.get("TLDW_DEBUG"):
            logging.getLogger().setLevel(
                logging.INFO
            )  # Reduce to INFO level in production
            # Disable debug logging for performance
            logging.getLogger("tldw_chatbook").setLevel(logging.INFO)

        # Log initial memory usage only in debug mode
        if os.environ.get("TLDW_DEBUG"):
            log_resource_usage()
        log_counter(
            "app_startup_initiated", 1, documentation="Application startup initiated"
        )

        super().__init__()

        # TASK-21115: a consolidated (BUNDLED_CSS) class adds no stylesheet
        # source at first mount, so a dynamic first mount can resolve against
        # a stale parse in which a base class's defaults still carry
        # tie-breaker 0 and shadow the consolidated sheet's rules (Textual's
        # `add_source` lowers a stored tie-breaker without arming a reparse).
        # This subclass reparses when that happens -- restoring exactly the
        # cascade per-class DEFAULT_CSS produced. See
        # `css/tie_aware_stylesheet.py` for the measured failure shape.
        self.stylesheet = TieAwareStylesheet(variables=self.get_css_variables())

        # Phase 1: Basic initialization
        phase_start = time.perf_counter()
        self.MediaDatabase = MediaDatabase
        self.app_config = load_settings()
        self.library_new_profile_admission = first_profile_created_this_session()
        self.console_image_edit_operations = ImageEditOperationRegistry()
        self._console_image_edit_shutdown_task: asyncio.Task[None] | None = None
        # Persona Buddy controller is built lazily on first access
        # (TASK-21103): constructing it imports Persona_Visual and PIL
        # (1.28 s cold), and both consumers (screen reconcile, Console
        # sink) already tolerate its absence. Slots must exist before
        # `ConsoleRuntime(self)` below, whose constructor reads the
        # persona_buddy_controller property. See that property.
        self._persona_buddy_controller: Any | None = None
        self._persona_buddy_controller_lock = threading.Lock()
        # task-15860 (headless wake): the Console runtime -- chat store,
        # provider gateway, agent bridge, chat controller -- is constructed
        # by the APP, not by `ChatScreen`, and it OUTLIVES every Console
        # screen. Screens are never cached (`_create_navigation_screen`), so
        # anything that must survive a navigation cannot be built on one.
        # `ChatScreen.on_unmount` now ends one VISIT
        # (`leave_console_runtime`); the runtime itself is destroyed once,
        # here, at exit (`_shutdown_console_runtime`).
        self.console_runtime: ConsoleRuntime | None = ConsoleRuntime(self)
        self._console_runtime_shutdown_task: asyncio.Task[None] | None = None
        self.generated_video_store = _build_generated_video_store()
        # TASK-13157: snapshot any TOML parse failure `load_settings()` just
        # hit -- captured here (mirroring `_instance_lock_status` below, the
        # same "detect at __init__, stash, notify once mounted" shape)
        # because `load_settings()`/`load_cli_config_and_ensure_existence()`
        # both silently fall back to in-memory defaults on a parse failure
        # rather than raising; the app has no UI to notify through yet at
        # this point in construction. `_maybe_warn_config_load_failure`
        # turns this into a persistent, file-and-error-naming notification
        # once the initial screen is up -- previously this degradation
        # (including the resolved data directory silently becoming the
        # `default_user` profile) had no visible signal at all.
        self._config_load_failure = get_config_load_failure()
        # RAG-53 (task-7): advisory per-profile instance lock. The profile
        # (and thus its data dir) is final as soon as config is loaded --
        # earliest sound point for this. Detection only: never blocks,
        # never raises, never prevents boot -- the owner runs concurrent
        # instances deliberately, so any acquisition failure here defaults
        # to "acquired" (no false warning) rather than surfacing as a boot
        # error. The status (and its open file handle, when acquired) is
        # kept referenced on the app instance for the process lifetime --
        # closing/GC'ing that handle would silently release the OS lock and
        # disarm detection for any instance that starts afterward.
        try:
            self._instance_lock_status = acquire_profile_instance_lock(
                get_user_data_dir()
            )
        except Exception as _instance_lock_exc:
            logger.debug(
                "Instance lock acquisition failed unexpectedly (%s)",
                type(_instance_lock_exc).__name__,
            )
            self._instance_lock_status = InstanceLockStatus(acquired=True)
        self.tts_service = build_default_tts_service(self.app_config)
        self._tts_binding_active = False
        self._tts_profile_repository = TTSProfileRepository(get_tts_profiles_db_path())
        self._tts_profile_repository_open_task: asyncio.Task[bool] | None = None
        self._tts_profile_repository_close_task: asyncio.Task[None] | None = None
        self._tts_profile_service: TTSProfileService | None = None
        self._audio_cpp_artifact_lease_coordinator: (
            AudioCppArtifactLeaseCoordinator | None
        ) = None
        self._tts_voice_bundle_service: TTSVoiceBundlePortabilityService | None = None
        self._tts_voice_bundle_service_close_task: asyncio.Task[None] | None = None
        self.acp_runtime_process_manager = ACPRuntimeProcessManager.from_app_config(
            self.app_config
        )
        self.acp_runtime_session_state = (
            self.acp_runtime_process_manager.session_state()
        )
        load_runtime_policy_for_app(self)
        self.screen_state_store = ScreenStateStore()
        self.pending_handoffs = PendingHandoffStore()
        self.audio_cpp_model_install_owner = AudioCppModelInstallOwner()
        self.file_notes_session_owner = build_file_notes_session_owner()
        self._file_notes_session_owner_shutdown_task: asyncio.Task[None] | None = None
        #: TASK-1143 (F5): count of Console agent runs/rounds the last
        #: navigation-away teardown killed (``ChatScreen.on_unmount`` ->
        #: ``ConsoleChatController.shutdown()``). The app outlives the
        #: screen instance that recorded it -- screens are never cached
        #: (``_create_navigation_screen``) -- so the NEXT Console mount
        #: reads and clears this one-shot slot to show a single toast.
        #: 0 means nothing to report.
        self._console_fleet_teardown_notice: int = 0
        self.service_policy_enforcer = (
            ServicePolicyEnforcer.from_runtime_policy_context(self.runtime_policy)
        )
        self.ui_policy_engine = PolicyEngine(CAPABILITY_REGISTRY)
        self.home_active_work_adapter = UnavailableHomeActiveWorkAdapter(
            runtime_policy=self.runtime_policy,
        )
        self.loguru_logger = loguru_logger
        self.loguru_logger.info(
            f"Loaded app_config - strip_thinking_tags: {self.app_config.get('chat_defaults', {}).get('strip_thinking_tags', 'NOT SET')}"
        )  # Make loguru_logger an instance variable for handlers
        self.client_id = CLI_APP_CLIENT_ID
        self.prompts_client_id = (
            "tldw_tui_client_v1"  # Store client ID for prompts service
        )
        self.db_status_manager = DBStatusManager(
            self
        )  # Initialize database status manager
        self.ui_responsiveness_monitor = UIResponsivenessMonitor(
            enabled=bool(
                get_cli_setting("diagnostics", "ui_responsiveness_enabled", True)
            ),
            heartbeat_interval_seconds=1.0,
        )
        self._wire_server_context_provider()
        self._startup_phases["basic_init"] = time.perf_counter() - phase_start
        log_histogram(
            "app_startup_phase_duration_seconds",
            self._startup_phases["basic_init"],
            labels={"phase": "basic_init"},
            documentation="Duration of startup phase in seconds",
        )

        # Phase 2: Attribute initialization
        phase_start = time.perf_counter()
        # Initialize screen navigation flag early to prevent AttributeError
        self._use_screen_navigation = True  # ALWAYS use screen-based navigation now
        # Initialize retained Notes ingest attributes.
        self.selected_note_files_for_import = []
        self.parsed_notes_for_preview = []  # <<< INITIALIZATION for notes
        self.last_note_import_dir = None
        # Llama.cpp server process
        self.llamacpp_server_process = None
        # LlamaFile server process
        self.llamafile_server_process = None
        # vLLM server process
        self.vllm_server_process = None
        self.ollama_server_process = None
        self.mlx_server_process = None
        self.onnx_server_process = None
        self._llm_server_launch_claims = {}
        self._llm_server_lifecycle_lock = threading.RLock()
        self._startup_phases["attribute_init"] = time.perf_counter() - phase_start
        log_histogram(
            "app_startup_phase_duration_seconds",
            self._startup_phases["attribute_init"],
            labels={"phase": "attribute_init"},
            documentation="Duration of startup phase in seconds",
        )

        # Phase 3: Parallel initialization of independent services
        phase_start = time.perf_counter()

        # Prepare shared data
        user_name_for_notes = settings.get("USERS_NAME", "default_tui_user")
        self.notes_user_id = user_name_for_notes

        # Run independent initializations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all independent initialization tasks
            futures = {
                executor.submit(
                    self._init_notes_service, user_name_for_notes
                ): "notes_service",
                executor.submit(self._init_providers_models): "providers_models",
                executor.submit(self._init_prompts_service): "prompts_service",
                executor.submit(self._init_media_db): "media_db",
            }

            # Wait for all tasks to complete and log individual timings
            for future in concurrent.futures.as_completed(futures):
                task_name = futures[future]
                try:
                    task_start = time.perf_counter()
                    future.result()
                    task_duration = time.perf_counter() - task_start
                    logger.info(
                        f"Parallel init task '{task_name}' completed in {task_duration:.3f}s"
                    )
                except Exception as e:
                    logger.opt(exception=True).error(
                        f"Parallel init task '{task_name}' failed: {e}"
                    )

        # Log total parallel phase time
        parallel_duration = time.perf_counter() - phase_start
        self._startup_phases["parallel_init"] = parallel_duration
        log_histogram(
            "app_startup_phase_duration_seconds",
            parallel_duration,
            labels={"phase": "parallel_init"},
            documentation="Duration of parallel initialization phase",
        )
        log_resource_usage()  # Check memory after parallel init

        # Providers, prompts, and media DB are initialized in parallel above
        # Just ensure we have defaults if parallel init failed
        if not hasattr(self, "providers_models"):
            self.providers_models = {}

        # --- Initial Tab ---
        initial_tab_from_config = get_cli_setting("general", "default_tab", TAB_CHAT)
        self._initial_tab_value = self._normalize_initial_tab_from_config(
            initial_tab_from_config
        )
        logging.info(
            f"App __init__: Determined initial tab value: {self._initial_tab_value}"
        )
        # current_tab reactive will be set in on_mount after UI is composed

        # --- Focus mode (task-18812) ---
        self.focus_mode = False
        self._focus_mode_config = bool(
            get_cli_setting("general", "focus_mode", False)
        )
        # Set by _resolve_initial_shell_route when onboarding outranks a
        # focus request at startup; restored when the wizard lands on Chat.
        self._deferred_focus_request: bool = False

        self._rich_log_handler: Optional[RichLogHandler] = (
            None  # For the RichLog widget in Logs tab
        )

        # Prompts service is initialized in parallel above
        # Set up timer
        self._prompt_search_timer = None

        # Media DB is initialized in parallel above
        # Ensure we have media types for UI
        if not hasattr(self, "_media_types_for_ui"):
            self._media_types_for_ui = ["Error: Media DB not loaded"]

        self.local_media_reading_service = LocalMediaReadingService(
            self.media_db, app_config=self.app_config
        )
        self.server_media_reading_service = (
            ServerMediaReadingService.from_server_context_provider(
                self.server_context_provider,
                policy_enforcer=self.service_policy_enforcer,
            )
        )
        self._wire_library_collections_services()
        self._wire_workspace_registry_services()
        self._wire_prompt_chatbook_services()
        self._wire_watchlists_and_notifications_services()
        self.media_reading_scope_service = MediaReadingScopeService(
            local_service=self.local_media_reading_service,
            server_service=self.server_media_reading_service,
            policy_enforcer=self.service_policy_enforcer,
            sync_scope_service=self.sync_scope_service,
        )
        self._wire_writing_services()

        self.loguru_logger.debug(
            f"ULTRA EARLY APP INIT: self._media_types_for_ui VALUE: {self._media_types_for_ui}"
        )
        self.loguru_logger.debug(
            f"ULTRA EARLY APP INIT: self._media_types_for_ui TYPE: {type(self._media_types_for_ui)}"
        )

        self._tts_handler = None
        self._stts_handler = None
        self._tts_initialization_task: asyncio.Task | None = None
        self._stts_initialization_task: asyncio.Task | None = None
        self._deferred_startup_tasks: set[asyncio.Task] = set()
        self._screen_preimport_thread: threading.Thread | None = None

        self._ui_ready = False  # Track if UI is fully composed
        self._shutting_down = False  # Track if app is shutting down
        self._quit_in_progress = False

        # --- Assign DB instances for event handlers ---
        if self.prompts_service_initialized:
            # Get the database instance using the get_db_instance() function
            try:
                self.prompts_db = prompts_interop.get_db_instance()
                logging.info(
                    "Assigned prompts_interop.get_db_instance() to self.prompts_db"
                )
            except RuntimeError as e:
                logging.error(f"Error getting prompts_db instance: {e}")
                self.prompts_db = None  # Explicitly set to None
        else:
            self.prompts_db = None  # Ensure it's None if service failed
            logging.warning(
                "Prompts service not initialized, self.prompts_db set to None."
            )
        self.prompt_scope_service = build_prompt_scope_service(
            prompt_db=self.prompts_db,
            app_config=self.app_config,
            policy_enforcer=self.service_policy_enforcer,
            client_provider=self.server_context_provider,
        )

        if getattr(self.notes_service, "db", None):
            self.chachanotes_db = _select_profile_database(self.notes_service)
            logging.info("Assigned self.notes_service.db to self.chachanotes_db")
        else:  # Fallback to global if notes_service didn't set it up as expected on itself
            lazy_db = _select_profile_database(self.notes_service)
            if lazy_db:
                self.chachanotes_db = lazy_db
                logging.info(
                    "Assigned lazy-loaded chachanotes_db to self.chachanotes_db as fallback."
                )
            else:
                logging.error(
                    "ChaChaNotesDB (CharactersRAGDB) instance not found/assigned in app.__init__."
                )
                self.chachanotes_db = None  # Explicitly set to None

        self._wire_chat_conversation_services()

        self.server_notes_workspace_service = (
            ServerNotesWorkspaceService.from_server_context_provider(
                self.server_context_provider,
                policy_enforcer=self.service_policy_enforcer,
            )
        )
        self.notes_scope_service = _build_notes_scope_service(
            chachanotes_db=self.chachanotes_db,
            local_notes_service=self.notes_service,
            server_service=self.server_notes_workspace_service,
            policy_enforcer=self.service_policy_enforcer,
            sync_scope_service=getattr(self, "sync_scope_service", None),
        )
        notes_sync_state_path = get_notes_sync_state_db_path()
        notes_sync_migrator = build_notes_sync_legacy_migrator(
            database_path=notes_sync_state_path,
            legacy_connection=lambda: self.chachanotes_db.get_connection(),
            settings=self.app_config,
            note_scope_id=ScopeType.LOCAL_NOTE.value,
            file_notes_binding=self.file_notes_session_owner.current_binding,
            private_paths=(notes_sync_state_path, get_chachanotes_db_path()),
        )
        notes_sync_watcher_interval, notes_sync_watcher_max_interval = (
            get_notes_sync_watcher_intervals(self.app_config)
        )
        self.notes_sync_runtime_owner = build_notes_sync_runtime_owner(
            notes_scope_service=self.notes_scope_service,
            cutover_admitted=True,
            profile_process_is_sole=self._instance_lock_status.acquired,
            database_path=notes_sync_state_path,
            migrate_legacy=notes_sync_migrator,
            file_notes_binding=self.file_notes_session_owner.current_binding,
            local_user_id=self.notes_user_id,
            recovery_capacity_bytes=get_notes_sync_recovery_capacity_bytes(
                self.app_config
            ),
            # TASK-21112 boot gate: start only on actual configuration — the
            # legacy [notes] sync-directory key (one-time migration path) or
            # a state DB already on disk. Path.exists() never opens or
            # creates the database; a zero-profile boot therefore creates no
            # notes-sync state at all. First-time setup (review_setup)
            # force-starts the runtime on demand. On Python 3.12
            # Path.exists() RAISES PermissionError (pathlib no longer
            # swallows EACCES); on a sandboxed profile that deliberately
            # rides the gate's fail-open path — one full start attempt,
            # which is the safe direction and is memoized.
            start_evidence=(
                lambda settings=self.app_config, state_path=notes_sync_state_path: (
                    legacy_sync_directory_configured(settings)
                    or state_path.exists()
                )
            ),
            watcher_interval_seconds=notes_sync_watcher_interval,
            watcher_max_interval_seconds=notes_sync_watcher_max_interval,
        )
        self._notes_sync_runtime_start_task: asyncio.Task[None] | None = None
        self._notes_sync_runtime_shutdown_task: asyncio.Task[None] | None = None
        # RAG admin trio (server/local/scope) is built lazily on first access
        # (task-254): its legacy UI consumers were deleted and nothing reads
        # these services at startup, so eager construction only added launch
        # cost. See the server_rag_admin_service / local_rag_admin_service /
        # rag_admin_scope_service properties.
        self._server_rag_admin_service: Optional[ServerRAGAdminService] = None
        self._local_rag_admin_service: Optional[LocalRAGAdminService] = None
        self._rag_admin_scope_service: Optional[RAGAdminScopeService] = None
        self._rag_admin_services_lock = threading.Lock()
        self._wire_evaluation_services()
        self._wire_study_services()
        self._wire_research_services()
        self._wire_character_persona_services()
        # Persona Buddy: the controller slot itself is initialized earlier
        # (before ConsoleRuntime construction); see the lazy
        # persona_buddy_controller property (TASK-21103).
        self._persona_buddy_unavailable_authority = None
        self._persona_buddy_shutdown_task: asyncio.Task[None] | None = None

        # --- Initialize worker handler registry ---
        self._init_worker_handlers()

        # Log total initialization time
        total_init_time = time.perf_counter() - self._startup_start_time
        self._startup_phases["total_init"] = total_init_time
        log_histogram(
            "app_startup_total_duration_seconds",
            total_init_time,
            documentation="Total application initialization time in seconds",
        )

        # Log startup summary
        logger.info("=== STARTUP TIMING SUMMARY ===")
        logger.info(f"Total initialization time: {total_init_time:.3f} seconds")
        for phase, duration in self._startup_phases.items():
            if phase != "total_init":
                percentage = (
                    (duration / total_init_time) * 100 if total_init_time > 0 else 0
                )
                logger.info(f"  {phase}: {duration:.3f}s ({percentage:.1f}%)")
        logger.info("==============================")

        # Final memory check
        log_resource_usage()

    def _build_rag_admin_services(self) -> None:
        """Construct the RAG admin service trio on first access (task-254).

        Constructor semantics are identical to the eager wiring this replaced:
        a config-driven ``ServerRAGAdminService.from_config`` with a
        ``client=None`` fallback when config resolution raises ``ValueError``,
        a ``LocalRAGAdminService`` over the media DB and local media reading
        service, and the scope service routing between them with the policy
        enforcer. Built under a lock so a racing first access from a worker
        thread cannot produce a mixed trio; idempotent once built.
        """
        with self._rag_admin_services_lock:
            if self._rag_admin_scope_service is not None:
                return
            try:
                server_service = ServerRAGAdminService.from_config(
                    self.app_config,
                    policy_enforcer=self.service_policy_enforcer,
                )
            except ValueError:
                server_service = ServerRAGAdminService(
                    client=None,
                    policy_enforcer=self.service_policy_enforcer,
                )
            local_service = LocalRAGAdminService(
                self.media_db,
                media_service=self.local_media_reading_service,
            )
            self._server_rag_admin_service = server_service
            self._local_rag_admin_service = local_service
            self._rag_admin_scope_service = RAGAdminScopeService(
                local_service=local_service,
                server_service=server_service,
                policy_enforcer=self.service_policy_enforcer,
            )

    @property
    def server_rag_admin_service(self) -> "ServerRAGAdminService":
        """Server-backed RAG admin service, built lazily and cached (task-254).

        Returns:
            ServerRAGAdminService: The cached service, constructed together
            with the local and scope services on first access.
        """
        if self._server_rag_admin_service is None:
            self._build_rag_admin_services()
        return self._server_rag_admin_service

    @property
    def local_rag_admin_service(self) -> "LocalRAGAdminService":
        """Local RAG admin service, built lazily and cached (task-254).

        Returns:
            LocalRAGAdminService: The cached service, constructed together
            with the server and scope services on first access.
        """
        if self._local_rag_admin_service is None:
            self._build_rag_admin_services()
        return self._local_rag_admin_service

    @property
    def rag_admin_scope_service(self) -> "RAGAdminScopeService":
        """Local/server RAG admin scope router, built lazily and cached (task-254).

        Returns:
            RAGAdminScopeService: The cached scope router wired to the cached
            local and server services, constructed on first access.
        """
        if self._rag_admin_scope_service is None:
            self._build_rag_admin_services()
        return self._rag_admin_scope_service

    def _persona_buddy_configured_enabled(self) -> bool:
        """Report whether ``[persona_buddy] enabled`` is set in config.

        Parses only the stdlib preference contract (``Persona_Buddy.
        preferences`` behind the now-lazy package init) -- never the
        controller chain, so a disabled profile stays PIL-free.

        Returns:
            bool: True when the persisted preferences enable the Buddy.
        """
        from .Persona_Buddy.preferences import (  # noqa: PLC0415 - stdlib-only seam; keeps PIL off the boot path (TASK-21103)
            parse_persona_buddy_preferences,
        )

        config = getattr(self, "app_config", None)
        section = config.get("persona_buddy", {}) if isinstance(config, dict) else {}
        return parse_persona_buddy_preferences(section).enabled

    def _build_persona_buddy_controller(self) -> Any | None:
        """Construct and cache the app-owned Buddy controller (TASK-21103).

        Constructor semantics are identical to the eager wiring this
        replaced. Importing the controller module here is what pulls
        Persona_Visual and PIL, so it must stay out of module scope. Built
        under a lock so a racing first access from a worker thread cannot
        construct two controllers; idempotent once built.

        Returns:
            The cached controller, or None when the persona services this
            controller wires to are not present yet (early in ``__init__``,
            or on skeletal test apps) -- callers retry on next access.
        """
        with self._persona_buddy_controller_lock:
            if self._persona_buddy_controller is not None:
                return self._persona_buddy_controller
            # Only the persona service gates construction: the old eager
            # wiring ran right after _wire_character_persona_services() and
            # passed self.chachanotes_db through as-is (it is legitimately
            # None on test-factory apps; the controller tolerates that).
            local_persona_service = getattr(
                self, "local_character_persona_service", None
            )
            if local_persona_service is None:
                return None
            profile_db = getattr(self, "chachanotes_db", None)
            from .Persona_Buddy.controller import (  # noqa: PLC0415 - imports Persona_Visual + PIL; first feature use only (TASK-21103)
                PersonaBuddyController,
                load_local_persona_portrait,
            )
            from .Persona_Buddy.preferences import (  # noqa: PLC0415
                parse_persona_buddy_preferences,
            )

            self._persona_buddy_controller = PersonaBuddyController(
                preferences=parse_persona_buddy_preferences(
                    self.app_config.get("persona_buddy", {})
                ),
                local_persona_service=local_persona_service,
                portrait_loader=partial(
                    load_local_persona_portrait,
                    local_persona_service,
                ),
                profile_db=profile_db,
                profile_root=get_user_data_dir(),
                reduced_motion=bool(
                    get_cli_setting("appearance", "reduce_motion", False)
                ),
                scheduler=self.call_after_refresh,
            )
            return self._persona_buddy_controller

    def ensure_persona_buddy_controller(self) -> Any | None:
        """Build (if needed) and return the Buddy controller for feature use.

        Explicit Buddy actions (e.g. Personas Workbench "Use for Buddy" on a
        profile whose preferences still say disabled) go through here: unlike
        the passive property, this constructs regardless of the persisted
        ``enabled`` flag so enabling from a disabled state works end to end.

        Returns:
            The controller, or None when its wiring prerequisites are absent.
        """
        return self._build_persona_buddy_controller()

    @property
    def persona_buddy_controller(self) -> Any | None:
        """App-owned Persona Buddy controller, built lazily (TASK-21103).

        Passive consumers (screen reconcile, Console sink, Workbench status)
        read this via ``getattr(app, "persona_buddy_controller", None)`` and
        already tolerate None. While unbuilt, a profile whose preferences
        leave the Buddy disabled gets None back without constructing
        anything, keeping the every-screen-mount reconcile early-out free of
        the Persona_Visual/PIL import cost. First access on an enabled
        profile -- or an explicit ``ensure_persona_buddy_controller()`` call
        from a Buddy action -- performs the one-time construction.

        Returns:
            The cached controller; None when disabled-and-unbuilt or when
            construction prerequisites are not wired yet.
        """
        controller = self._persona_buddy_controller
        if controller is not None:
            return controller
        if not self._persona_buddy_configured_enabled():
            return None
        return self._build_persona_buddy_controller()

    @persona_buddy_controller.setter
    def persona_buddy_controller(self, controller: Any | None) -> None:
        """Inject or clear the controller slot (tests and skeletal doubles).

        Args:
            controller: The controller instance to install, or None to make
                the lazy property construct anew on next enabled access.
        """
        self._persona_buddy_controller = controller

    def _wire_server_context_provider(self) -> None:
        self.unified_mcp_target_store = ConfiguredServerTargetStore(
            get_user_data_dir() / "mcp_server_targets.json",
        )
        self.unified_mcp_target_store.upsert_legacy_config_target(self.app_config)
        try:
            self.server_credential_store = build_default_server_credential_store()
            self.server_credential_store_unavailable_reason = None
        except CredentialStoreUnavailable as exc:
            self.server_credential_store = UnavailableServerCredentialStore(str(exc))
            self.server_credential_store_unavailable_reason = str(exc)
            logger.warning(
                "No secure OS credential store available; server tokens will "
                "remain config-only (reason={}).",
                str(exc),
            )
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
        origin: Optional[str] = None,
    ) -> None:
        """Stage Study handoffs and navigate to the Study screen.

        Args:
            scope_context: Scoped study context to apply, or None to clear
                any pending scope.
            initial_section: Study section to land on, or None to clear any
                pending section.
            origin: Where the user is coming FROM (``STUDY_ORIGINS``:
                "home" or "library"), threaded to StudyScreen so its
                breadcrumb and Escape target name the actual origin
                (task-4011). None clears the channel and StudyScreen falls
                back to its historical Library default (task-2854's one
                considered origin).
        """
        if scope_context is None:
            self.pending_handoffs.clear_pending(HandoffChannel.STUDY_SCOPE)
        elif not self._stage_handoff(
            HandoffChannel.STUDY_SCOPE,
            scope_context,
            recovery="Study scope could not be opened. Try again.",
        ):
            return

        if initial_section is None:
            self.pending_handoffs.clear_pending(HandoffChannel.STUDY_INITIAL_SECTION)
        elif not self._stage_handoff(
            HandoffChannel.STUDY_INITIAL_SECTION,
            initial_section,
            recovery="Study section could not be opened. Try again.",
        ):
            return

        if origin is None:
            self.pending_handoffs.clear_pending(HandoffChannel.STUDY_ORIGIN)
        elif not self._stage_handoff(
            HandoffChannel.STUDY_ORIGIN,
            origin,
            recovery="Study could not be opened. Try again.",
        ):
            return
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
        self.post_message(
            NavigateToScreen(TAB_LIBRARY, {LIBRARY_NAV_CONTEXT_MODE: "notes"})
        )

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
                surfaces, "Use in Console" for Library). Currently unused
                inside this method -- it previously fed the retired
                chat-tabs gate's blocked notify (task-577 U5, which removed
                the gate so handoffs proceed unconditionally); kept for
                caller-signature compatibility.
        """
        if not self._stage_handoff(
            HandoffChannel.CHAT,
            payload,
            recovery="Chat context could not be staged. Try again.",
        ):
            return
        self.post_message(NavigateToScreen(TAB_CHAT))

    def stage_console_prompt_insert(
        self,
        application: PromptVariableApplication,
    ) -> None:
        """Stage a guarded Prompt application and then navigate to Console.

        The typed, memory-only application carries the final selected lanes
        plus destination/session/staleness guards. Console remains the only
        owner allowed to settle the claim and mutate its active draft.

        Args:
            application: Validated Prompt application to stage.
        """
        if not self._stage_handoff(
            HandoffChannel.CONSOLE_PROMPT_INSERT,
            application,
            recovery="Console prompt could not be staged. Review it and try again.",
        ):
            return
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
        if not self._stage_handoff(
            HandoffChannel.CONSOLE_LIVE_WORK,
            {
                "source": source,
                "title": title,
                "payload": payload,
                "status": status,
                "recovery": recovery,
                "action_label": action_label,
            },
            recovery="Console live work could not be staged. Try again.",
        ):
            return
        self.post_message(NavigateToScreen(TAB_CHAT))

    def _stage_handoff(
        self,
        channel: HandoffChannel,
        value: Any,
        *,
        recovery: str,
    ) -> bool:
        """Stage one typed handoff without exposing its value in recovery."""
        try:
            self.pending_handoffs.stage(channel, value)
        except HandoffValueError:
            self.notify(recovery, severity="warning")
            return False
        return True

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
            self.notify(
                "Console action is unavailable for this live-work item.",
                severity="warning",
            )
            return False

        action = resolve_console_live_work_primary_action(normalized_launch)
        if action is None:
            self.notify(
                "Console action is unavailable for this live-work item.",
                severity="warning",
            )
            return False

        if action.target_route == TAB_WATCHLISTS_COLLECTIONS:
            self.post_message(
                NavigateToScreen(
                    TAB_WATCHLISTS_COLLECTIONS,
                    self._watchlists_run_navigation_context(action.target_id),
                )
            )
            return True

        if action.target_route == TAB_ARTIFACTS:
            if not self._stage_handoff(
                HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
                action.target_id,
                recovery="Console action target could not be opened. Try again.",
            ):
                return False
            self.post_message(NavigateToScreen(TAB_ARTIFACTS))
            return True

        if action.target_route == TAB_ACP:
            if not self._stage_handoff(
                HandoffChannel.ACP_SESSION_TARGET,
                action.target_id,
                recovery="Console action target could not be opened. Try again.",
            ):
                return False
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
        adapter = getattr(
            self, "home_active_work_adapter", UnavailableHomeActiveWorkAdapter()
        )
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

    def approve_active_home_item(
        self, *, target_id: str | None = None
    ) -> HomeControlResult:
        """Approve the active Home item through the configured adapter."""
        return self._handle_home_control_action(
            HomeControlAction.APPROVE, target_id=target_id
        )

    def reject_active_home_item(
        self, *, target_id: str | None = None
    ) -> HomeControlResult:
        """Reject the active Home item through the configured adapter."""
        return self._handle_home_control_action(
            HomeControlAction.REJECT, target_id=target_id
        )

    def pause_active_home_item(
        self, *, target_id: str | None = None
    ) -> HomeControlResult:
        """Pause the active Home item through the configured adapter."""
        return self._handle_home_control_action(
            HomeControlAction.PAUSE, target_id=target_id
        )

    def resume_active_home_item(
        self, *, target_id: str | None = None
    ) -> HomeControlResult:
        """Resume the active Home item through the configured adapter."""
        return self._handle_home_control_action(
            HomeControlAction.RESUME, target_id=target_id
        )

    def retry_active_home_item(
        self, *, target_id: str | None = None
    ) -> HomeControlResult:
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
            job_id = str(target_id)[len("local:ingest:") :]
            requeued = self.retry_library_ingest_job(job_id)
            if requeued is None:
                # Unknown job id, or the job is no longer FAILED (e.g. it
                # was already retried/finished by the time the button was
                # pressed) -- ``requeue`` is a documented no-op in that case.
                result = HomeControlResult(
                    action=HomeControlAction.RETRY,
                    status=HomeControlResultStatus.UNAVAILABLE,
                    message="This import job can no longer be retried.",
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
        return self._handle_home_control_action(
            HomeControlAction.RETRY, target_id=target_id
        )

    def open_home_flashcards_review(self) -> None:
        """Open the Study screen directly on the flashcards review surface.

        task-4011: this is the one entry into Study that does NOT come from
        Library's staging canvas, so it declares its origin -- StudyScreen's
        breadcrumb reads "Home ▸ Study" and Escape returns to Home instead
        of a Library canvas the user never visited.
        """
        self.open_study_screen(initial_section="flashcards", origin="home")

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
            if result.target_route in {
                "subscriptions",
                TAB_WATCHLISTS_COLLECTIONS,
            }:
                self.post_message(
                    NavigateToScreen(
                        TAB_WATCHLISTS_COLLECTIONS,
                        self._watchlists_run_navigation_context(
                            result.target_id or target_id
                        ),
                    )
                )
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

    @staticmethod
    def _watchlists_run_navigation_context(
        target_id: str | None,
    ) -> dict[str, object]:
        """Build the destination-owned context for a Watchlists run deep link."""
        context: dict[str, object] = {
            WATCHLISTS_NAV_CONTEXT_SECTION: WATCHLISTS_SECTION_RUNS
        }
        if target_id:
            target_id_text = str(target_id)
            context[WATCHLISTS_NAV_CONTEXT_RUN_ID] = target_id_text
            backend = target_id_text.partition(":watchlist_run:")[0]
            if backend in {"local", "server"}:
                context[WATCHLISTS_NAV_CONTEXT_BACKEND] = backend
        return context

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
        if (
            result.status is HomeControlResultStatus.HANDLED
            and result.console_launch is not None
        ):
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
        self.server_character_persona_service = (
            ServerCharacterPersonaService.from_server_context_provider(
                self.server_context_provider,
                policy_enforcer=self.service_policy_enforcer,
            )
        )
        self.local_character_persona_service = LocalCharacterPersonaService(
            self.chachanotes_db,
            persona_store_path=get_user_data_dir() / "tldw_chatbook_personas.json",
        )
        self.actor_pack_repository = ActorPackRepository(self.chachanotes_db)
        self.persona_actor_pack_coordinator = PersonaActorPackCoordinator(
            self.actor_pack_repository,
            self.local_character_persona_service,
        )
        # task-21106: crash recovery no longer runs here — synchronous SQLite
        # during __init__ cost every boot and crashed the test app factory
        # (which builds the app with chachanotes_db=None), silently disarming
        # the CSS parse-cache cliff guard. `ensure_actor_pack_recovery` now
        # runs it once per app session: kicked on a background thread from
        # `_schedule_deferred_startup_work`, and hard-gated ahead of the
        # Personas screen's first library read and (inside the coordinator)
        # every `create_persona` mutation.
        self.actor_pack_recovery_error: str | None = None
        self.actor_pack_creation_service = ActorPackCreationService(
            self.chachanotes_db,
            self.actor_pack_repository,
            self.persona_actor_pack_coordinator,
        )
        self.actor_pack_export_service = ActorPackExportService(
            self.chachanotes_db,
            self.local_character_persona_service,
            self.actor_pack_repository,
            persona_visual_repository=PersonaVisualRepository(self.chachanotes_db),
            visual_identity_repository=VisualIdentityRepository(self.chachanotes_db),
            profile_root=get_user_data_dir(),
        )
        self.actor_pack_export_controller = ActorPackExportController(
            self.actor_pack_export_service
        )
        self._actor_pack_export_shutdown_task: asyncio.Task[None] | None = None
        self.character_persona_scope_service = CharacterPersonaScopeService(
            local_service=self.local_character_persona_service,
            server_service=self.server_character_persona_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self.server_chat_dictionary_service = (
            ServerChatDictionaryService.from_server_context_provider(
                self.server_context_provider,
                policy_enforcer=self.service_policy_enforcer,
            )
        )
        self.local_chat_dictionary_service = LocalChatDictionaryService(
            self.chachanotes_db,
            history_store_path=get_user_data_dir()
            / "tldw_chatbook_chat_dictionary_history.json",
        )
        self.chat_dictionary_scope_service = ChatDictionaryScopeService(
            local_service=self.local_chat_dictionary_service,
            server_service=self.server_chat_dictionary_service,
            policy_enforcer=self.service_policy_enforcer,
        )

    def ensure_actor_pack_recovery(self) -> None:
        """Run Actor Pack crash recovery once per app session (task-21106).

        Safe to call from any thread and idempotent: the once-guard lives on
        the coordinator (screens are never cached, so a per-mount flag would
        re-run recovery on every Personas visit). Callers that may touch
        recovery-affected state before the deferred startup kick has finished
        call this first — from a worker thread, because a non-trivial recovery
        does real SQLite work.

        Preserves the exact `__init__`-era outcome mapping: a coordination
        failure records ``actor_pack_recovery_failed``; retained quarantined
        intents record ``actor_pack_recovery_blocked``. With no ChaChaNotes DB
        (the test app factory builds the app without one) recovery is skipped
        entirely, matching a boot where the profile store never opened.
        """
        coordinator = getattr(self, "persona_actor_pack_coordinator", None)
        if coordinator is None or getattr(self, "chachanotes_db", None) is None:
            return
        first_run = not coordinator.recovery_attempted
        recovery = coordinator.ensure_recovered()
        if coordinator.recovery_error is not None:
            self.actor_pack_recovery_error = "actor_pack_recovery_failed"
            if first_run:
                self.loguru_logger.error(
                    "Actor Pack recovery failed: actor_pack_recovery_failed"
                )
        elif recovery is not None and recovery.blocked_intent_ids:
            self.actor_pack_recovery_error = "actor_pack_recovery_blocked"
            if first_run:
                self.loguru_logger.warning(
                    "Actor Pack recovery retained quarantined intents: "
                    "actor_pack_recovery_blocked"
                )

    def _wire_chat_conversation_services(self) -> None:
        trace_db = getattr(self, "chachanotes_db", None)
        sidecar_path = get_user_data_dir() / "tldw_chatbook_chat_rag_context.json"
        existing_service = getattr(
            self,
            "local_chat_conversation_service",
            None,
        )
        existing_migration = getattr(
            self,
            "citation_legacy_migration_service",
            None,
        )
        if (
            trace_db is not None
            and existing_service is not None
            and existing_service.db is trace_db
            and existing_service.rag_context_store_path == sidecar_path
            and existing_migration is not None
            and existing_service.citation_legacy_migration is existing_migration
        ):
            repository = getattr(
                existing_migration,
                "repository",
                getattr(self, "citation_trace_repository", None),
            )
            migration = existing_migration
            local_service = existing_service
        elif trace_db is not None:
            coordinator = getattr(
                self,
                "citation_artifact_ownership_coordinator",
                None,
            )
            coordinator_repository = (
                coordinator.trace_repository
                if coordinator is not None
                and coordinator.trace_repository.db is trace_db
                else None
            )
            local_service, repository, migration = (
                build_local_citation_conversation_service(
                    trace_db,
                    sidecar_path=sidecar_path,
                    repository=coordinator_repository,
                )
            )
        else:
            local_service = None
            repository = None
            migration = None
        self.local_chat_conversation_service = local_service
        self.citation_trace_repository = repository
        self.citation_legacy_migration_service = migration
        self.conversation_local_marks_service = (
            ConversationLocalMarksService(trace_db) if trace_db is not None else None
        )
        self.server_chat_conversation_service = (
            ServerChatConversationService.from_server_context_provider(
                self.server_context_provider,
                policy_enforcer=self.service_policy_enforcer,
            )
        )
        self.chat_conversation_scope_service = ChatConversationScopeService(
            local_service=self.local_chat_conversation_service,
            server_service=self.server_chat_conversation_service,
            policy_enforcer=self.service_policy_enforcer,
            sync_scope_service=self.sync_scope_service,
        )
        self._wire_citation_artifact_ownership()

    def _wire_writing_services(self) -> None:
        try:
            self.local_writing_service = LocalWritingService(get_writing_db_path())
        except Exception:
            logger.opt(exception=True).warning(
                "Local writing service unavailable during app wiring"
            )
            self.local_writing_service = None
        self.server_writing_service = ServerWritingService.from_server_context_provider(
            self.server_context_provider,
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

        self.local_chatbook_service = LocalChatbookService(
            self._build_chatbook_db_paths()
        )
        self.server_chatbook_service = (
            ServerChatbookService.from_server_context_provider(
                self.server_context_provider,
                policy_enforcer=self.service_policy_enforcer,
            )
        )

        self.prompt_chatbook_scope_service = PromptChatbookScopeService(
            local_prompt_service=self.local_prompt_service,
            server_prompt_service=self.server_prompt_service,
            local_chatbook_service=self.local_chatbook_service,
            server_chatbook_service=self.server_chatbook_service,
            policy_enforcer=self.service_policy_enforcer,
        )
        self._wire_citation_artifact_ownership()

    def _wire_citation_artifact_ownership(self) -> None:
        """Compose cross-store citation ownership after both stores exist."""

        artifact_store = getattr(self, "local_chatbook_service", None)
        trace_db = getattr(self, "chachanotes_db", None)
        if artifact_store is None or trace_db is None:
            if not hasattr(self, "citation_artifact_ownership_coordinator"):
                self.citation_artifact_ownership_coordinator = None
            return
        repository = getattr(self, "citation_trace_repository", None)
        if repository is None or repository.db is not trace_db:
            conversation_service, repository, migration = (
                build_local_citation_conversation_service(
                    trace_db,
                    sidecar_path=get_user_data_dir()
                    / "tldw_chatbook_chat_rag_context.json",
                )
            )
            self.local_chat_conversation_service = conversation_service
            self.citation_trace_repository = repository
            self.citation_legacy_migration_service = migration
        current = getattr(
            self,
            "citation_artifact_ownership_coordinator",
            None,
        )
        if (
            current is not None
            and current.artifact_store is artifact_store
            and current.trace_repository is repository
        ):
            return
        coordinator = CitationArtifactOwnershipCoordinator(
            artifact_store=artifact_store,
            trace_repository=repository,
        )
        artifact_store.set_citation_ownership_coordinator(coordinator)
        self.citation_artifact_ownership_coordinator = coordinator

    def _wire_evaluation_services(self) -> None:
        self.local_evaluation_service = None
        try:
            self.evaluation_orchestrator = EvaluationOrchestrator(
                client_id="tldw_cli_app"
            )
            self.local_evaluation_service = LocalEvaluationsService(
                self.evaluation_orchestrator.db
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Local evaluation service unavailable during app wiring"
            )
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
            or getattr(self.server_evaluation_service, "client_provider", None)
            is not None
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
        self._init_library_ingest_runtime_state()

    def _wire_research_services(self) -> None:
        """Initialize source-aware research services if the broad parity wiring has not already done so."""
        if hasattr(self, "research_scope_service") and hasattr(
            self, "research_search_scope_service"
        ):
            return

        try:
            self.local_research_service = LocalResearchService(
                get_research_db_path(),
                notification_dispatcher=self.notification_dispatch_service,
                notification_app=self,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Local research service unavailable during app wiring"
            )
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
            self.server_research_search_service = (
                ServerResearchSearchService.from_config(
                    self.app_config,
                    policy_enforcer=self.service_policy_enforcer,
                )
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
        # task-15463: ONE SubscriptionsDB for this whole wiring. `db_factory`
        # used to be `lambda: SubscriptionsDB(...)`, and `LocalWatchlistsService.
        # _db()` called it on every service method -- so nearly every watchlists
        # read rebuilt the database object, paying a ~52-statement schema
        # `executescript` plus migration probes each time (3.4 ms against
        # 0.04 ms on a held instance; 35 ms for the first build; five-plus per
        # screen refresh). The same instance is handed to the projections, the
        # scheduled-check handler and the bundle service below, which already
        # shared one eager instance among themselves.
        #
        # Safe to share across threads: `SubscriptionsDB` connections are
        # thread-local (`DB/Subscriptions_DB.py`'s `conn` property), so each
        # `asyncio.to_thread` worker that touches this instance opens its own
        # connection to the same file. `db_factory` stays a callable because it
        # is the injectable seam tests repoint (`Tests/UI/
        # test_watchlists_inspector.py`).
        subscriptions_db = SubscriptionsDB(
            get_subscriptions_db_path(), CLI_APP_CLIENT_ID
        )
        # Held on the app so the FTS-backfill worker can reuse it instead of
        # constructing a second one -- see `_backfill_subscription_items_fts`,
        # where a concurrent second `_initialize_schema` was measured
        # poisoning a live connection's schema view.
        self.subscriptions_db = subscriptions_db
        # task-19561, Qodo review of PR #1972: the startup reconcile sweep runs
        # as a deferred startup task, i.e. AFTER `on_mount` has already started
        # the scheduler worker -- and the scheduler ticks immediately, so a due
        # watchlist check can have launched a real `queued`/`running` row by the
        # time the sweep looks. Unscoped, the sweep failed that live row as
        # "interrupted".
        #
        # The boundary is captured HERE, in `__init__`'s wiring, rather than
        # moved earlier in `on_mount`, precisely so that no future edit to
        # `on_mount`'s ordering can reintroduce the race: at this point there is
        # no event loop at all, so nothing in this process can yet have inserted
        # into these tables. Everything this process later creates gets a
        # strictly higher AUTOINCREMENT id and is therefore out of the sweep's
        # reach by construction. See `Subscriptions/startup_reconcile.py`.
        from tldw_chatbook.Subscriptions.startup_reconcile import (
            capture_prior_process_boundary,
        )

        self._subscriptions_prior_process_boundary = capture_prior_process_boundary(
            subscriptions_db
        )
        self.local_watchlists_service = LocalWatchlistsService(
            db_factory=lambda: subscriptions_db
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
            self.server_notifications_service = (
                ServerNotificationsService.from_server_context_provider(
                    self.server_context_provider,
                    policy_enforcer=self.service_policy_enforcer,
                )
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
        server_client = SchedulingServerClient(self.server_notifications_service)

        # `subscriptions_db` is the single instance built at the top of this
        # method (task-15463); it used to be constructed here, separately from
        # the service's own per-call construction.
        watchlist_projection = WatchlistProjection(subscriptions_db)

        # `briefing_projection` is built here, BEFORE `SchedulingService`, so
        # it can be passed straight into the constructor (task-1810) rather
        # than after -- `SchedulingService.list_tasks` needs it live from the
        # moment the service exists, unlike `briefing_handler` below (built
        # later; only `SchedulerLoop`, constructed further down this method,
        # consumes it). Constructing it earlier changes nothing about its
        # behavior: it only depends on `subscriptions_db`, already created
        # above.
        briefing_schedules_enabled = get_cli_setting(
            "scheduling", "briefing_schedules_enabled", True
        )
        briefing_projection = (
            BriefingProjection(subscriptions_db) if briefing_schedules_enabled else None
        )

        self.scheduling_service = SchedulingService(
            db=ScheduledTasksDB(get_scheduled_tasks_db_path()),
            server_client=server_client,
            runtime_source="local",
            watchlist_projection=watchlist_projection,
            briefing_projection=briefing_projection,
            # task-18937: reminder mutations must reach the live scheduler
            # queue on the next tick. The loop itself is constructed further
            # down, so the callback resolves it lazily -- wiring
            # `self.scheduler_loop.request_reload` directly here would freeze
            # `None`/AttributeError in before the loop exists (same getter
            # discipline as `BriefingJobHandler`'s chachanotes_db_getter).
            on_queue_changed=lambda: getattr(
                self, "scheduler_loop", None
            ).request_reload()
            if getattr(self, "scheduler_loop", None) is not None
            else None,
        )

        watchlist_checks_enabled = get_cli_setting(
            "scheduling", "watchlist_checks_enabled", True
        )
        watchlist_checks_shadow = get_cli_setting(
            "scheduling", "watchlist_checks_shadow", False
        )

        watchlist_handler = None
        if watchlist_checks_enabled:
            watchlist_handler = WatchlistCheckHandler(
                subscriptions_db=subscriptions_db,
                shadow_mode=watchlist_checks_shadow,
            )

        briefing_handler = None
        if briefing_projection is not None:
            # `self.chachanotes_db` is assigned later in `__init__`,
            # strictly AFTER `_wire_watchlists_and_notifications_services`
            # (this method, called earlier in `__init__`) returns -- so it
            # does not exist as an attribute on `self` yet at this point.
            # A GETTER, not the instance itself, is what makes this safe:
            # `BriefingJobHandler` calls `chachanotes_db_getter()` fresh
            # every time a scheduled generation completes (long after
            # `__init__` has finished), so this lambda's `getattr(self,
            # "chachanotes_db", None)` re-reads whatever `self.
            # chachanotes_db` has become by THEN, not whatever it was (or
            # wasn't) at this wiring call. Passing the instance directly
            # here (review round 1's finding) would freeze `None` into the
            # handler forever, making auto-keep permanently inert in
            # production regardless of what `self.chachanotes_db` later
            # becomes. The handler tolerates the getter returning `None`
            # at any given call -- auto-keep (task-1780, Task 3) simply
            # skips that one attempt; nothing else about scheduled
            # generation depends on it.
            briefing_handler = BriefingJobHandler(
                subscriptions_db=subscriptions_db,
                chachanotes_db_getter=lambda: getattr(self, "chachanotes_db", None),
            )

        # task-19561: shutdown has to be able to reach the generations this
        # handler spawns, and the scheduler loop is not a route to them --
        # they are bare `asyncio.Task`s, not workers, deliberately detached
        # from the tick. Keeping the handler itself on the app is the only
        # handle `on_unmount` has.
        self._briefing_job_handler = briefing_handler

        handlers: dict[str, Handler] = {
            "reminder": ReminderHandler(
                dispatch_service=self.notification_dispatch_service
            ),
        }
        if watchlist_handler is not None:
            handlers["watchlist_job"] = watchlist_handler
        if briefing_handler is not None:
            handlers["briefing_job"] = briefing_handler

        self.scheduler_loop = SchedulerLoop(
            self.scheduling_service.db,
            handlers=handlers,
            poll_interval=get_cli_setting(
                "scheduling",
                "scheduler_poll_interval_seconds",
                SCHEDULER_POLL_INTERVAL_SECONDS,
            ),
            watchlist_projection=(
                watchlist_projection if watchlist_handler is not None else None
            ),
            briefing_projection=(
                briefing_projection if briefing_handler is not None else None
            ),
            missed_fire_grace_seconds=get_cli_setting(
                "scheduling", "missed_fire_grace_seconds", MISSED_FIRE_GRACE_SECONDS
            ),
            handler_timeout_seconds=get_cli_setting(
                "scheduling", "handler_timeout_seconds", HANDLER_TIMEOUT_SECONDS
            ),
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
        self.server_prompt_studio_service = (
            ServerPromptStudioService.from_server_context_provider(
                self.server_context_provider,
                policy_enforcer=self.service_policy_enforcer,
            )
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
            self.server_voice_assistant_service = (
                ServerVoiceAssistantService.from_config(
                    self.app_config,
                    policy_enforcer=self.service_policy_enforcer,
                )
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
            self.server_personalization_service = (
                ServerPersonalizationService.from_config(
                    self.app_config,
                    policy_enforcer=self.service_policy_enforcer,
                )
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
        # Research services: ONE wiring path, not two. This used to duplicate
        # `_wire_research_services` verbatim here (task-16332); the method's
        # own already-wired guard makes calling it from this earlier-in-
        # `__init__` bootstrap equivalent to the old embedded copy, and the
        # later direct `_wire_research_services()` call then early-returns.
        self._wire_research_services()
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
            self.server_collections_feeds_service = (
                ServerCollectionsFeedsService.from_config(
                    self.app_config,
                    policy_enforcer=self.service_policy_enforcer,
                )
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
        local_skills_store_dir = default_local_skills_store_dir(get_user_data_dir())
        trust_store_dir = default_trust_store_dir(local_skills_store_dir)
        trust_account_scope = skill_trust_account_scope(trust_store_dir)
        skill_trust_marker_store, reduced_rollback_protection = (
            build_skill_trust_marker_store_with_fallback(
                fallback_marker_path=trust_store_dir / _SKILL_TRUST_MARKER_FILENAME,
                store_dir=trust_store_dir,
                account_scope=trust_account_scope,
            )
        )
        self.local_skill_trust_service = SkillTrustService(
            skills_dir=local_skills_store_dir / "skills",
            trust_store=SkillTrustStore(
                store_dir=trust_store_dir,
                marker_store=skill_trust_marker_store,
            ),
            key_cache=build_default_skill_trust_key_cache(
                account_scope=trust_account_scope
            ),
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
        self.server_sync_service = ServerSyncService.from_server_context_provider(
            self.server_context_provider,
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
            provider_catalog_loader=lambda: dict(
                getattr(self, "providers_models", {}) or {}
            ),
            local_provider_names=set(LOCAL_PROVIDERS),
            default_provider=get_cli_setting("chat_defaults", "provider", None),
            policy_enforcer=self.service_policy_enforcer,
        )
        # ADR-020: load the disk-backed model catalog cache before selectors build.
        self.model_catalog_disk_store = self._init_model_catalog_disk_store()
        try:
            self.server_llm_provider_catalog_service = (
                ServerLLMProviderCatalogService.from_config(
                    self.app_config,
                    policy_enforcer=self.service_policy_enforcer,
                )
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
            tts_provider_loader=lambda: {
                "chatbook_tts": {"available": True, "source": "local"}
            },
            stt_provider_loader=lambda: {
                "chatbook_stt": {"available": True, "source": "local"}
            },
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
        self.server_auth_account_service = (
            ServerAuthAccountService.from_server_context_provider(
                self.server_context_provider,
                policy_enforcer=self.service_policy_enforcer,
            )
        )
        self.auth_account_scope_service = AuthAccountScopeService(
            server_service=self.server_auth_account_service,
            policy_enforcer=self.service_policy_enforcer,
            server_context_provider=self.server_context_provider,
        )
        try:
            self.server_user_governance_service = (
                ServerUserGovernanceService.from_config(
                    self.app_config,
                    policy_enforcer=self.service_policy_enforcer,
                )
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
        self.watchlist_bundle_service = WatchlistBundleService(subscriptions_db)
        self.local_media_reading_service.notification_dispatcher = (
            self.notification_dispatch_service
        )
        self.local_media_reading_service.notification_app = self
        self.local_watchlists_service.notification_dispatcher = (
            self.notification_dispatch_service
        )
        self.local_watchlists_service.notification_app = self

    def _backfill_subscription_items_fts(self) -> None:
        """Worker body: index subscription_items rows that predate the FTS
        index (task-688). Started from ``on_mount`` via
        ``run_worker(thread=True)`` so a large backlog never blocks app
        startup or screen mount.

        Uses the app's single ``SubscriptionsDB`` (task-15463). It used to
        construct its own, on the theory that a thread-local connection
        cannot be shared -- but thread-locality is exactly what makes sharing
        the INSTANCE safe: this worker thread gets its own connection from it.
        Constructing a second instance re-ran ``_initialize_schema`` -- a
        ~52-statement ``executescript`` plus migrations, measured at 238 ms --
        on a worker thread *while the app was already serving screens*, and
        any connection opened during that window cached a schema view without
        the tables it was rewriting. That is not theoretical: with per-call
        database construction it showed up as the intermittent
        ``OperationalError: no such table: subscription_items`` documented in
        ``Tests/UI/test_watchlists_inspector.py``, self-healing on retry
        because the next call built a new connection; against a held instance
        the poisoned connection survives, and the write that lands on it just
        fails. One instance, one schema initialization, no window.

        ``close()`` below stays, and what it does is worth stating exactly.
        ``SubscriptionsDB.close`` closes only the CALLING thread's connection
        and clears that thread's slot. This body runs on a **pooled** thread
        (Textual's thread workers run on asyncio's default executor, which is
        shared with every ``asyncio.to_thread`` hop in the app), so the
        connection it closes belongs to a pool thread that will later serve
        other watchlists work on this same shared instance. That is safe for
        exactly one reason: the ``conn`` property re-opens lazily, so the next
        hop scheduled onto that thread gets a fresh connection instead of a
        closed one. It is not safe to "improve" this into a close of the
        instance itself.
        """
        db = None
        db_path = get_subscriptions_db_path()
        try:
            db = getattr(self, "subscriptions_db", None)
            if db is None:
                # Only a harness that skipped service wiring gets here.
                db = SubscriptionsDB(db_path, CLI_APP_CLIENT_ID)
            backfill_subscription_items_fts(db)
        except FTSBackfillError as exc:
            logger.opt(exception=True).error(
                "Subscription items FTS backfill failed for database {} "
                "after indexing {} row(s) this run; some pre-existing "
                "items may remain unsearchable until the app is restarted.",
                db_path,
                exc.rows_indexed,
            )
        except Exception:
            logger.opt(exception=True).error(
                "Subscription items FTS backfill failed for database {}; "
                "some pre-existing items may remain unsearchable until the "
                "app is restarted.",
                db_path,
            )
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    logger.opt(exception=True).warning(
                        "Failed to close SubscriptionsDB {} after FTS backfill.",
                        db_path,
                    )

    def _backfill_chachanotes_messages_fts(self) -> None:
        """Worker body: reinsert messages the v45->v46 migration no longer
        indexes inline (task-21100). Started from ``on_mount`` via
        ``run_worker(thread=True)`` so an upgraded profile's index rebuild
        never blocks boot or first paint; each chunk commits in its own
        transaction, so a kill at any point leaves a consistent, resumable
        index (state = ``messages_fts_docsize`` membership, in the DB
        itself).

        Uses the app's shared ``CharactersRAGDB`` singleton -- thread-local
        connections are exactly what makes that safe from a worker thread
        (see ``_backfill_subscription_items_fts`` for the incident that
        taught this). Unlike that worker, no ``close()`` here: pooled threads
        serve ChaChaNotes work constantly, and the thread-local connection
        this run opens is the same one later hops on this thread reuse.

        On an up-to-date database the loop's first chunk finds nothing and
        the whole call is one indexed scan -- cheap, and it doubles as
        self-healing for any run interrupted before completion.
        """
        from tldw_chatbook.DB.chachanotes_fts_backfill import (
            ChaChaNotesFTSBackfillError,
            backfill_chachanotes_messages_fts,
        )

        try:
            db = get_chachanotes_db_lazy()
            if db is None:
                logger.debug(
                    "ChaChaNotes messages FTS backfill skipped: no database instance."
                )
                return
            backfill_chachanotes_messages_fts(db)
        except ChaChaNotesFTSBackfillError as exc:
            logger.opt(exception=True).error(
                "ChaChaNotes messages FTS backfill failed after indexing {} "
                "row(s) this run; older messages may be missing from search "
                "until the next app start resumes it.",
                exc.rows_indexed,
            )
        except Exception:
            logger.opt(exception=True).error(
                "ChaChaNotes messages FTS backfill failed; older messages may "
                "be missing from search until the next app start resumes it."
            )

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
                event_state_repository=EventStateRepository(
                    ":memory:", CLI_APP_CLIENT_ID
                ),
                sync_state_repository=SyncStateRepository(
                    ":memory:", CLI_APP_CLIENT_ID
                ),
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
        runtime_policy = getattr(self, "runtime_policy", None)
        runtime_state = runtime_policy.state if runtime_policy is not None else None
        if isinstance(runtime_state, RuntimeSourceState):
            normalized = str(runtime_state.active_source or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
        return self._resolve_initial_media_runtime_backend()

    def _server_notification_event_scope(self) -> dict[str, str | None]:
        runtime_policy = getattr(self, "runtime_policy", None)
        runtime_state = runtime_policy.state if runtime_policy is not None else None
        active_server_id = getattr(runtime_state, "active_server_id", None)
        authenticated_principal_id = None
        server_context_provider = getattr(self, "server_context_provider", None)
        get_active_context = getattr(
            server_context_provider, "get_active_context", None
        )
        if callable(get_active_context):
            try:
                authenticated_principal_id = event_principal_id_from_active_context(
                    get_active_context()
                )
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
        state = (
            runtime_state_override
            if isinstance(runtime_state_override, RuntimeSourceState)
            else None
        )
        if state is None:
            policy_enforcer = getattr(self, "service_policy_enforcer", None)
            if policy_enforcer is not None and hasattr(
                policy_enforcer, "current_state"
            ):
                state = policy_enforcer.current_state()
        if not isinstance(state, RuntimeSourceState):
            runtime_policy = getattr(self, "runtime_policy", None)
            runtime_state = runtime_policy.state if runtime_policy is not None else None
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

    async def handle_runtime_backend_changed(
        self,
        runtime_backend: str,
        *,
        app_config_override: Mapping[str, Any] | None = None,
    ) -> bool:
        normalized_backend = str(runtime_backend or "").strip().lower()
        if normalized_backend not in {"local", "server"}:
            return False

        previous_server_id = self.runtime_policy.state.active_server_id
        candidate_config = (
            app_config_override if app_config_override is not None else self.app_config
        )
        try:
            updated_state = set_authoritative_runtime_source(
                self.runtime_policy,
                normalized_backend,
                app_config=candidate_config,
            )
        except Exception as exc:
            logger.warning(
                "Runtime source change was not committed (exception_category={}).",
                type(exc).__name__,
            )
            self.notify(
                "Runtime source could not be changed; "
                "the previous source remains active.",
                severity="warning",
            )
            return False

        if app_config_override is not None:
            self.app_config = app_config_override
            self.server_context_provider.rebind_app_config(
                app_config_override,
                previous_server_id=previous_server_id,
                next_server_id=updated_state.active_server_id,
            )
        else:
            self.server_context_provider.invalidate_for_server_switch(
                previous_server_id,
                updated_state.active_server_id,
            )

        resolved_backend = (
            str(self.runtime_policy.state.active_source or normalized_backend)
            .strip()
            .lower()
        )
        active_screen = self.screen
        callback = getattr(active_screen, "handle_runtime_backend_changed", None)
        if callable(callback):
            try:
                await callback(resolved_backend)
            except Exception as exc:
                logger.warning(
                    "Runtime screen callback failed after runtime commit "
                    "(exception_category={}).",
                    type(exc).__name__,
                )
        return True

    def _init_notes_service(self, user_name_for_notes: str) -> None:
        """Initialize notes service - for parallel execution."""
        try:
            # Get the full path to the unified ChaChaNotes DB FILE
            chachanotes_db_file_path = get_chachanotes_db_path()
            logger.info(f"Unified ChaChaNotes DB file path: {chachanotes_db_file_path}")

            # Determine the PARENT DIRECTORY for NotesInteropService's 'base_db_directory'
            actual_base_directory_for_service = chachanotes_db_file_path.parent
            logger.info(
                f"Notes for user '{user_name_for_notes}' will use the unified DB: {chachanotes_db_file_path}"
            )

            self.notes_service = NotesInteropService(
                base_db_directory=actual_base_directory_for_service,
                api_client_id="tldw_tui_client_v1",
                global_db_to_use=get_chachanotes_db_lazy(),
            )
            logger.info(
                f"NotesInteropService successfully initialized for user '{user_name_for_notes}'."
            )
        except Exception as e:
            logger.opt(exception=True).error(
                f"Failed to initialize NotesInteropService: {e}"
            )
            self.notes_service = None

    def _init_providers_models(self) -> None:
        """Initialize providers and models - for parallel execution."""
        try:
            self.providers_models = get_cli_providers_and_models()
            logger.info(
                f"Successfully retrieved providers_models. Count: {len(self.providers_models)}. Keys: {list(self.providers_models.keys())}"
            )
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to get providers and models: {e}")
            self.providers_models = {}

    def _init_prompts_service(self) -> None:
        """Initialize prompts service - for parallel execution."""
        self.prompts_service_initialized = False
        try:
            prompts_db_path = get_prompts_db_path()
            prompts_interop.initialize_interop(
                db_path=prompts_db_path, client_id=self.prompts_client_id
            )
            self.prompts_service_initialized = True
            logger.info(
                f"Prompts Interop Service initialized with DB: {prompts_db_path}"
            )
        except Exception as e:
            self.prompts_service_initialized = False
            logger.opt(exception=True).error(
                f"Failed to initialize Prompts Interop Service: {e}"
            )

    def _init_media_db(self) -> None:
        """Initialize media database - for parallel execution."""
        try:
            media_db_path = get_media_db_path()
            # Get integrity check configuration
            check_integrity = self.app_config.get("database", {}).get(
                "check_integrity_on_startup", False
            )
            self.media_db = MediaDatabase(
                db_path=media_db_path,
                client_id=CLI_APP_CLIENT_ID,
                check_integrity_on_startup=check_integrity,
            )
            logger.info(
                f"Media_DB_v2 initialized successfully for client '{CLI_APP_CLIENT_ID}' at {media_db_path}"
            )

            # Wire ingestion-time RAG indexing (task-247). The hook no-ops
            # when the embeddings_rag extras are missing; indexing failures
            # are logged and surfaced without ever affecting ingestion.
            try:
                from .RAG_Search.ingestion_indexing import install_media_ingest_hook

                install_media_ingest_hook(
                    failure_notifier=self._notify_rag_indexing_failure,
                    guidance_notifier=self._notify_rag_indexing_guidance,
                )
            except Exception as e:
                logger.warning(f"Could not install RAG ingestion-indexing hook: {e}")

            # Pre-fetch media types for UI
            if self.media_db:
                db_types = self.media_db.get_distinct_media_types(
                    include_deleted=False, include_trash=False
                )
                self._media_types_for_ui = ["All Media"] + sorted(list(set(db_types)))
                logger.info(
                    f"Pre-fetched {len(self._media_types_for_ui)} media types for UI."
                )
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

    def _notify_rag_indexing_guidance(self, message: str) -> None:
        """Surface a RAG setup gap as information, not a warning.

        A fresh install has no embedding model, so nothing can be indexed for
        semantic search -- but the import itself succeeded, and presenting that
        as a warning made a new user's first successful action look like a
        failure (task-685). Same marshalling as the failure notifier: called
        from the indexer's worker thread.
        """
        try:
            self.call_from_thread(
                self.notify, message, severity="information", timeout=8
            )
        except Exception as e:
            logger.debug(f"Could not surface RAG indexing guidance in UI: {e}")

    def _init_worker_handlers(self) -> None:
        """Initialize the worker handler registry and register all handlers."""
        self.worker_handler_registry = WorkerHandlerRegistry(self)

        # Native Console owns Chat runs; these handlers serve retained app workers.
        self.worker_handler_registry.register(MiscWorkerHandler(self))

        self.loguru_logger.info("Worker handler registry initialized with all handlers")

    # task-577 PR2 T2: `_build_handler_map`/`button_handler_map` retired --
    # scout finding #3 (write-only, zero readers; `on_button_pressed` is a
    # screen-nav no-op that never consulted the map). The folded
    # *_BUTTON_HANDLERS source dicts remain defined in their own modules,
    # unreferenced here but out of this task's scope.

    def _setup_buffered_logging(self):
        """Set up a persistent buffered logging handler for screen navigation mode.

        TASK-19555 (privacy). This is the ONE choke point every in-app
        diagnostic passes through, and the two stores it fills have different
        jobs and therefore different privacy bars:

        * ``_log_records`` is the LIVE VIEW. It stays descriptive -- redacting
          it would empty the Logs screen of the content the screen exists to
          show -- but every line is first stripped of credentials and of the
          operating-system account name, which are never worth reading and are
          the two things a screenshot or a shoulder-surfer must not capture.
        * ``_log_buffer`` is the SHARE ARTIFACT: the exact payload
          ``LogsWindow._on_copy_all`` joins onto the system clipboard. It holds
          the metadata-only form, because "Copy all" bulk-exports thousands of
          lines the user has never read -- consent that cannot be informed.
          Anything a user deliberately shares, they share by filtering the view
          and pressing "Copy visible".

        Both stores are bounded to the same window, so the share action cannot
        export more history than the screen admits to keeping.
        """
        from collections import deque
        import logging

        from tldw_chatbook.UI.Logs_Window import MAX_LOG_RECORDS
        from tldw_chatbook.Utils.log_sanitizer import (
            REDACTION_MARKER,
            redact_log_line,
        )
        from tldw_chatbook.Utils.persistent_diagnostics import (
            PersistentDiagnosticFilter,
            safe_metadata_token,
        )

        # The clipboard payload for "Copy all". Bounded (TASK-19555): an
        # unbounded session buffer is a memory leak and a disclosure surface,
        # and it let "Copy all" export far more history than the Logs screen
        # itself retains or discloses in its status line.
        if not hasattr(self, "_log_buffer"):
            self._log_buffer = deque(maxlen=MAX_LOG_RECORDS)

        # Structured records (level, name, formatted message) for the Logs
        # screen's filtering; bounded like the RichLog widget itself.
        if not hasattr(self, "_log_records"):
            self._log_records = deque(maxlen=MAX_LOG_RECORDS)

        # The SAME admission rule the rotating file handler uses, so the
        # clipboard and the disk sink cannot drift apart on what counts as
        # metadata-only. Reused as an object, not re-implemented.
        share_admission = PersistentDiagnosticFilter()

        def _share_line(record, formatted, formatter):
            """Return the metadata-only form of one record for the clipboard.

            Schema-validated ADR-029 metadata events pass through verbatim --
            they are already the safe artifact. Everything else keeps its
            timestamp, logger, level and exception type, and loses its message
            body: the body is where interpolated paths, titles, queries,
            prompts, tool arguments and provider payloads live, and no
            sink-side rule can tell those apart from the wording around them.
            """
            if share_admission.filter(record):
                return formatted
            detail = ""
            exc_type = record.exc_info[0] if record.exc_info else None
            if exc_type is not None:
                name = safe_metadata_token(getattr(exc_type, "__name__", ""))
                detail = f" (exception_type={name})"
            stamp = formatter.formatTime(record, formatter.datefmt)
            return (
                f"{stamp} - {record.name} - {record.levelname} - "
                f"{REDACTION_MARKER}{detail}"
            )

        # Create a custom handler that stores logs in the buffer
        class PersistentLogHandler(logging.Handler):
            def __init__(self, buffer, app):
                super().__init__()
                self.buffer = buffer
                self.app = app

            def emit(self, record):
                try:
                    formatted = self.format(record)
                    formatter = self.formatter or logging.Formatter()
                    self.buffer.append(_share_line(record, formatted, formatter))
                    msg = redact_log_line(formatted)
                    self.app._log_records.append((record.levelname, record.name, msg))

                    # Preferred live path: the Logs screen's LogsWindow applies
                    # the user's active filters as records arrive.
                    logs_window = getattr(self.app, "_current_logs_window", None)
                    if logs_window is not None:
                        try:
                            logs_window.append_record(
                                record.levelname, record.name, msg
                            )
                            return
                        except Exception:
                            pass  # Widget might not be mounted

                    # Legacy fallback: write straight to the RichLog widget.
                    if (
                        hasattr(self.app, "_current_log_widget")
                        and self.app._current_log_widget
                    ):
                        try:
                            self.app._current_log_widget.write(msg)
                        except Exception:
                            pass  # Widget might not be mounted
                except Exception:
                    self.handleError(record)

        # Add the persistent handler to the root logger
        if not hasattr(self, "_persistent_log_handler"):
            self._persistent_log_handler = PersistentLogHandler(self._log_buffer, self)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            self._persistent_log_handler.setFormatter(formatter)
            logging.getLogger().addHandler(self._persistent_log_handler)
            logger.info("Persistent logging handler set up for screen navigation")

        # The app logs via loguru and the persistent handler is stdlib-only,
        # but NO bridge is installed here: `Logging_Config._setup_logging`
        # already forwards every loguru record into stdlib logging
        # (`_forward_loguru_to_standard`, level TRACE, diagnose=False per
        # task-2119), and it runs before this method on every boot path —
        # either early at process start or via `configure_application_
        # logging` in `_setup_logging`. A second sink here made every loguru
        # record reach the root logger twice, so the Logs screen showed each
        # application log line — and counted each error — twice
        # (TASK-15422).

        # Initialize current log widget reference
        self._current_log_widget = None

    def _display_buffered_logs(self, log_widget):
        """Display all buffered logs in the RichLog widget.

        Reads ``_log_records`` (the live-view store), NOT ``_log_buffer``
        (the metadata-only clipboard artifact) -- TASK-19555. This legacy
        path currently has no callers; it is pointed at the right store so
        that reviving it shows a maintainer real diagnostics rather than a
        screen of redaction markers.
        """
        if not hasattr(self, "_log_records"):
            return

        # Store reference to current log widget
        self._current_log_widget = log_widget

        # Clear the widget first to avoid duplicates
        log_widget.clear()

        # Write all buffered messages to the widget
        for _level, _name, msg in self._log_records:
            log_widget.write(msg)

        # Scroll to the latest entry
        log_widget.scroll_end()

        logger.debug(f"Displayed {len(self._log_records)} buffered log entries")

    def _setup_logging(self):
        """Set up logging for the application.

        If early logging was already initialized, this will just set up the RichLogHandler
        for the UI log display widget.
        """
        # Check if we're running as a module (via entry point) or as a script
        if (
            hasattr(self, "_early_logging_initialized")
            and self._early_logging_initialized
        ):
            # Early logging was already initialized, just set up the RichLogHandler
            logging.info(
                "Logging already initialized early, setting up UI log handlers only"
            )
            try:
                log_display_widget = self.query_one("#app-log-display", RichLog)
                if not self._rich_log_handler:
                    self._rich_log_handler = RichLogHandler(log_display_widget)
                    rich_log_handler_level_str = (
                        self.app_config.get("logging", {})
                        .get("rich_log_level", "DEBUG")
                        .upper()
                    )
                    rich_log_handler_level = getattr(
                        logging, rich_log_handler_level_str, logging.DEBUG
                    )
                    self._rich_log_handler.setLevel(rich_log_handler_level)
                    logging.getLogger().addHandler(self._rich_log_handler)
                    logging.info(
                        f"Added RichLogHandler to existing logging setup (Level: {logging.getLevelName(self._rich_log_handler.level)})."
                    )
            except QueryError:
                logging.error(
                    "!!! ERROR: Failed to find #app-log-display widget for RichLogHandler setup."
                )
            except Exception as e:
                logging.error(
                    f"!!! ERROR setting up RichLogHandler: {e}", exc_info=True
                )
        else:
            # No early logging, do full initialization
            configure_application_logging(self)

    def compose(self) -> ComposeResult:
        compose_start = time.perf_counter()
        self._ui_compose_start_time = compose_start  # Store for later reference
        logging.debug("App composing UI...")
        log_counter("ui_compose_started", 1, documentation="UI composition started")

        # TASK-2154.19 (AC-01): ASCII-safe status-marker mode for narrow-font
        # terminals. Resolved once at compose so every glyph-production point
        # downstream reads the same module state.
        set_ascii_glyph_mode(get_cli_setting("appearance", "ascii_glyphs", False))

        # Check if splash screen is enabled
        splash_enabled = get_cli_setting("splash_screen", "enabled", True)
        logging.info(f"Splash screen enabled: {splash_enabled}")
        if splash_enabled:
            # Get splash screen configuration
            splash_duration = get_cli_setting("splash_screen", "duration", 1.5)
            splash_skip = get_cli_setting("splash_screen", "skip_on_keypress", True)
            splash_progress = get_cli_setting("splash_screen", "show_progress", True)
            splash_card = get_cli_setting("splash_screen", "card_selection", "random")
            # TASK-2154.10 (AC-04): vestibular-accessible static splash.
            splash_reduced_motion = get_cli_setting(
                "appearance", "reduce_motion", False
            )
            logging.info(
                f"Creating splash screen - duration: {splash_duration}, card: {splash_card}"
            )

            # Create and yield splash screen
            self._splash_screen_widget = SplashScreen(
                card_name=splash_card if splash_card != "random" else None,
                duration=splash_duration,
                skip_on_keypress=splash_skip,
                show_progress=splash_progress,
                reduced_motion=splash_reduced_motion,
                id="app-splash-screen",
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

        # Screen-based navigation is used exclusively: each BaseAppScreen
        # mounts the visible shell chrome (MainNavigationBar, AppFooterStatus,
        # Textual Footer) itself, so the default screen only needs the
        # container screens are pushed over.
        widgets.append(Container(id="screen-container"))

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

    # Legacy alias routes that need a default navigation context applied
    # when navigated to directly (bare ``NavigateToScreen(route)``, no
    # explicit context supplied). Mirrors how ``open_notes_workspace`` builds
    # ``{LIBRARY_NAV_CONTEXT_MODE: "notes"}`` for the retired standalone
    # Notes tab -- except "prompts" (the retired Personas "prompts" mode
    # chip, Task 7), "skills" (the retired standalone Skills tab, Skills
    # sub-project Task 5), "search" (the retired standalone Search
    # screen, RAG UX v2 PR-1 Task 1), and "media" (the retired standalone
    # Media Library screen, task-2851) have no dedicated re-entry action to
    # carry that context, so the bare alias route itself must supply it here.
    # The retired Customize screen folds into Settings > Theme.
    _LEGACY_ROUTE_LIBRARY_NAV_CONTEXT: dict[str, dict[str, str]] = {
        "prompts": {LIBRARY_NAV_CONTEXT_MODE: "prompts"},
        "skills": {LIBRARY_NAV_CONTEXT_MODE: "skills"},
        "search": {LIBRARY_NAV_CONTEXT_MODE: "search"},
        "media": {LIBRARY_NAV_CONTEXT_MODE: "media"},
        "customize": {"category": "theme"},
    }

    # How long the outgoing screen gets to flush pending work before the app
    # gives up on the transition.
    #
    # `handle_screen_navigation` is an `@on` handler on the App itself, so
    # everything it awaits is awaited ON the App's message pump -- while it
    # blocks, the app processes no clicks, no bindings and no further
    # navigation. The flush path reaches genuinely unbounded awaits
    # (`library_screen`'s `await worker.wait()`, and `_run_library_service_call`'s
    # `asyncio.to_thread`, which cannot be cancelled at all), so a save that
    # never completed left the app permanently frozen AND unkillable.
    #
    # Generous enough that a real save is never cut short, small enough that a
    # wedged one costs a few seconds instead of the session.
    NAVIGATION_FLUSH_TIMEOUT_SECONDS: float = 5.0

    @staticmethod
    def _persona_buddy_authority(controller: Any, snapshot: Any) -> tuple[Any, ...]:
        """Return the exact app-lifetime authority for one visual decision."""

        return (
            id(controller),
            snapshot.generation,
            snapshot.selection,
            snapshot.preferences_generation,
            snapshot.profile_generation,
        )

    def is_persona_buddy_confirmed_unavailable(
        self, controller: Any, snapshot: Any
    ) -> bool:
        """Query and clear the app-owned unavailable marker by exact authority."""

        authority = self._persona_buddy_authority(controller, snapshot)
        marker = getattr(self, "_persona_buddy_unavailable_authority", None)
        if marker is not None and marker != authority:
            self._persona_buddy_unavailable_authority = None
            return False
        return marker == authority

    def confirm_persona_buddy_unavailable(
        self,
        *,
        screen: Any,
        view: Any,
        view_generation: int,
        controller: Any,
        snapshot: Any,
        visual: Any,
    ) -> bool:
        """Publish unavailable only for the exact current app/screen/view authority."""

        current_controller = getattr(self, "persona_buddy_controller", None)
        try:
            current_screen = self.screen
        except Exception:
            return False
        if (
            controller is not current_controller
            or current_screen is not screen
            or not screen.is_attached
            or screen._persona_buddy_view is not view
            or screen.persona_buddy_view_generation != view_generation
            or not view.is_attached
        ):
            return False
        current = controller.snapshot()
        if (
            self._persona_buddy_authority(controller, current)
            != self._persona_buddy_authority(controller, snapshot)
            or current.visual is not visual
            or visual is None
            or visual.available
        ):
            return False
        self._persona_buddy_unavailable_authority = self._persona_buddy_authority(
            controller, current
        )
        return True

    async def reconcile_persona_buddy_view(self) -> bool:
        """Reconcile the active screen and report whether its Buddy is absent."""

        from .UI.Navigation.base_app_screen import BaseAppScreen

        try:
            screen = self.screen
        except Exception:
            return False
        if not isinstance(screen, BaseAppScreen) or not screen.is_active:
            return False
        return await screen.reconcile_persona_buddy_view()

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
        owned by ``ScreenStateStore`` through each screen's
        ``save_state``/``restore_state`` boundary, not instance reuse.

        One documented exception, since task-15860: Console's message
        history is NOT in that snapshot. It lives in the app-owned
        ``ConsoleRuntime``'s ``ConsoleChatStore``, which outlives every
        ``ChatScreen``; Console's snapshot carries only view state (image
        view modes, the task-resume projection, the staged live-work
        launch). Two sources of truth is what that snapshot had become --
        a turn that ran while Console was unmounted persisted to
        ChaChaNotes and was then overwritten, unseen, by a snapshot taken
        before it (executed: ``Docs/superpowers/plans/2026-08-14-headless-
        wake-task-0-report.md``, P3b). Screens still die on navigation;
        only the runtime survives.
        """
        return screen_class(self)

    def _valid_startup_route_ids(self) -> set[str]:
        """Return route ids allowed in startup config during the shell migration."""
        from .UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

        shell_routes = {
            destination.primary_route for destination in SHELL_DESTINATION_ORDER
        } | {destination.destination_id for destination in SHELL_DESTINATION_ORDER}
        legacy_aliases = {
            "conversation",
            "llm",
            "subscription",
            "subscriptions",
            "tools_settings",
            "notes",
            "prompts",
        }
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
        """Choose the startup route while keeping first-run orientation explicit.

        TASK-1508: the in-memory ``_first_run`` flag is routinely lost to a
        config force-reload before routing runs, so on a real fresh install
        the old check routed to the configured default tab (Console) and the
        auto-offered wizard opened over the Console's own "Get started" card
        — Esc revealed a second onboarding surface. Route from the same
        decision the wizard offer uses: if the wizard is about to be
        offered, land on Home beneath it.
        """
        # task-18812: record the focus request BEFORE the onboarding branches
        # return Home — a first-run launch defers it (the wizard navigates to
        # the Console on completion, and _handle_first_run_wizard_result then
        # restores the request) instead of silently discarding it. Any
        # non-onboarding route below applies it immediately.
        _focus_requested = bool(
            getattr(self, "_cli_focus_override", False)
            or getattr(self, "_focus_mode_config", False)
        )
        if self.app_config.get("_first_run", False):
            self._deferred_focus_request = _focus_requested
            return TAB_HOME
        try:
            from tldw_chatbook.UI.Wizards.first_run_setup_state import (
                setup_recovery_action,
            )

            if setup_recovery_action(self.app_config, os.environ) in {
                "offer",
                "prompt",
                "home",
            }:
                self._deferred_focus_request = _focus_requested
                return TAB_HOME
        except Exception:
            logger.debug("Wizard startup route check failed (category=runtime)")
        # task-18812: focus mode is Console-only by definition, so a focus
        # request forces the route — onboarding branches ABOVE still win
        # (spec: first-run wins).
        if _focus_requested:
            self.focus_mode = True
            return TAB_CHAT
        return getattr(self, "_initial_tab_value", TAB_CHAT)

    def _set_focus_mode(self, enabled: bool) -> None:
        """Set focus mode and apply it to the Console if it is on screen.

        task-18812 / ADR-071. Duck-types the content screen (it may or may
        not be the Console — do NOT import ChatScreen here; the screen
        registry keeps app.py free of screen imports for circular-import
        reasons). Enabling while elsewhere navigates to the Console first;
        the screen's mount-time ``_apply_focus_chrome`` read then applies
        the chrome. Disabling only clears the flag.
        """
        self.focus_mode = enabled
        content_screen = self._navigation_outgoing_screen()
        apply_chrome = getattr(content_screen, "_apply_focus_chrome", None)
        if callable(apply_chrome):
            apply_chrome()
        elif enabled:
            self.post_message(NavigateToScreen(TAB_CHAT))

    def action_toggle_focus_mode(self) -> None:
        """Ctrl+Shift+F: toggle the chrome-free Console focus mode."""
        self._set_focus_mode(not self.focus_mode)

    def _clear_focus_if_leaving_console(self, screen_name: str) -> None:
        """Single exit rule (ADR-071): focus mode is Console-only — any
        navigation to another route restores normal chrome on arrival."""
        if screen_name != TAB_CHAT:
            self.focus_mode = False

    def _current_runtime_identity(self) -> RuntimeIdentity:
        """Return the screen-snapshot scope from authoritative runtime state."""
        return RuntimeIdentity.from_state(self.runtime_policy.state)

    def console_prompt_target_projection(
        self,
    ) -> ConsolePromptTargetProjection | None:
        """Return the app-owned Console Prompt target for the current runtime.

        Returns:
            The compatible sanitized projection, or ``None`` when Console has
            not published one for the authoritative runtime snapshot.
        """
        return self.screen_state_store.restore_console_prompt_target(
            TAB_CHAT,
            self._current_runtime_identity(),
        )

    def library_rag_search_execution_lock(self) -> asyncio.Lock:
        """Return the app-lifetime admission lock for Library retrieval calls.

        Returns:
            The shared Library-only admission lock for this app session.
        """
        lock = getattr(self, "_library_rag_search_execution_lock_instance", None)
        if lock is None:
            lock = asyncio.Lock()
            self._library_rag_search_execution_lock_instance = lock
        return lock

    def _screen_navigation_lock(self) -> asyncio.Lock:
        """Return the lock serializing `handle_screen_navigation` attempts.

        TASK-1230: `_dispatch_screen_navigation` (the App's real
        ``@on(NavigateToScreen)`` handler) now runs each navigation attempt
        as its own worker instead of awaiting it inline on the App's single
        message-processing task -- see that method's docstring for why.
        Workers are otherwise independent, and running two attempts
        concurrently would let them race on shared state in a way the old
        single-queue dispatch never allowed: ``self.current_tab``,
        ``switch_screen``'s screen stack, and -- inside
        ``_complete_screen_navigation``, itself called from within the
        guarded region -- ``self.screen_state_store.save()`` (snapshotting
        the OUTGOING screen) and ``.restore()`` (rehydrating the INCOMING
        one); two attempts interleaving there could save/restore the wrong
        screen's state or clobber a snapshot the other attempt just wrote.
        This lock preserves the old FIFO ordering: `asyncio.Lock` serves
        waiters in arrival order, so attempts still complete strictly one
        at a time, in the order their ``NavigateToScreen`` messages were
        dispatched -- confirmed by
        ``test_overlapping_navigate_requests_complete_in_fifo_order``,
        which reliably reorders without this lock -- the only change is
        that an attempt waiting on a confirm-navigation dialog no longer
        blocks the App from routing input to that very dialog while it
        waits its turn.
        """
        lock = getattr(self, "_screen_navigation_lock_instance", None)
        if lock is None:
            lock = asyncio.Lock()
            self._screen_navigation_lock_instance = lock
        return lock

    @on(NavigateToScreen)
    def _dispatch_screen_navigation(self, message: NavigateToScreen) -> None:
        """Kick off ``handle_screen_navigation`` as its own worker (TASK-1230).

        F1 (fleet-UX expert review, 2026-07-28): a busy-fleet navigation
        opens a confirm-navigate dialog via ``ChatScreen.confirm_navigation``
        (``push_screen_wait`` inside a worker, its result awaited back out).
        That await used to happen INLINE inside this handler -- and Textual
        dispatches every ``@on``-decorated handler by awaiting it directly
        from the App's own single message-processing task, the SAME task
        solely responsible for routing every subsequent driver-originated
        mouse/key event (``App.on_event`` -> ``screen._forward_event``) to
        whatever is on top of the screen stack, dialog included. Awaiting
        the dialog's result inline therefore starved that task's own event
        loop for the dialog's entire lifetime: no click, key press, or
        Escape could ever reach it, because delivering any of them requires
        this exact task to loop back and dequeue the next message, which it
        cannot do while suspended awaiting `confirm_navigation`. That is the
        zombie-modal soft-lock: reproduced directly (not just theorized) by
        posting a real driver-style MouseDown/MouseUp pair while a confirm
        dialog was open and observing the App's own message queue grow
        without ever draining -- see the task's Implementation Notes.

        Running the full sequence (``handle_screen_navigation``, including
        its own flush/confirm/complete steps) as a decoupled worker keeps
        this task free to keep delivering input the moment ANY confirm
        dialog opens, first one or a subsequent one alike.
        ``handle_screen_navigation`` itself is unchanged and still directly
        awaitable to completion (its own FIFO ordering across overlapping
        attempts is preserved by ``_screen_navigation_lock``), so every
        existing direct caller (tests included) keeps working exactly as
        before; only real navigation -- dispatched through this handler --
        gains the fix.
        """
        self.run_worker(
            self.handle_screen_navigation(message),
            group="screen-navigation",
            exclusive=False,
            exit_on_error=False,
        )

    async def handle_screen_navigation(self, message: NavigateToScreen) -> None:
        """Handle navigation to a different screen using switch_screen for better performance."""
        async with self._screen_navigation_lock():
            try:
                await self._handle_screen_navigation_locked(message)
            except Exception:
                # task-2720: several steps in the locked body are legitimately
                # unguarded (target resolution, runtime identity, snapshot
                # restore, transition admission) and a transient error in any
                # of them used to fail SILENTLY: no message, nav-bar highlight
                # stuck on the destination, retry clicks no-opped. Recover the
                # user-facing state, then re-raise so the worker hook still
                # writes the `worker_failed` diagnostics line (ADR-029).
                self._notify_navigation_failure(message.screen_name)
                raise

    #: Bound on the dismiss-the-overlays loop below. Each pass removes one
    #: pushed screen, and dismissing one can legitimately reveal another
    #: (a picker opened from a dialog); nothing real stacks this deep, so a
    #: stack that will not reduce inside the bound is a stuck stack, not a
    #: busy one.
    _MAX_NAVIGATION_OVERLAY_DISMISSALS: int = 16

    def _navigation_outgoing_screen(self) -> Any:
        """Return the CONTENT screen a navigation is leaving.

        The screen stack is ``[Textual's default screen, the content screen,
        *pushed screens]``: startup pushes exactly one routed screen
        (``_push_initial_screen``) and every navigation replaces it, so
        index 1 is the tab the user is on and anything above it is an
        overlay. ``self.screen`` is the TOP of that stack, which is the
        overlay whenever one is open -- see ``_dismiss_navigation_overlays``
        for why that distinction is load-bearing.

        Returns:
            The content screen at the base of the stack, or ``self.screen``
            when the stack is too short for that position to exist (before
            the initial push, and in tests that drive the handler with no
            mounted stack at all).
        """
        try:
            stack = self._screen_stack
        except Exception:  # pragma: no cover - defensive; no mode, no stack
            return self.screen
        if len(stack) >= 2:
            return stack[1]
        return self.screen

    @staticmethod
    def _navigation_overlay_awaiter_pending(screen: Any) -> bool:
        """Report whether ``screen`` still owes a ``push_screen_wait`` result."""
        callbacks = getattr(screen, "_result_callbacks", None)
        if not callbacks:
            return False
        future = getattr(callbacks[-1], "future", None)
        return future is not None and not future.done()

    async def _dismiss_navigation_overlays(self, screen_name: str) -> bool:
        """Reduce the screen stack to its content screen before switching.

        TASK-16300. Textual's ``App.switch_screen``
        (``textual/app.py:3001-3032``) pops only ``self._screen_stack[-1]``
        and appends the new screen; ``_replace_screen`` then unmounts only
        that popped screen. So switching while ANY pushed screen sits above
        the content screen replaces THE OVERLAY and leaves the content
        screen resident in the stack -- mounted, message pump running,
        ``on_unmount`` never fired, its timers and controllers alive behind
        whatever the user is now looking at, and a second live instance of
        it created the moment they navigate back. That directly violates
        the invariant ``_create_navigation_screen`` documents (screens die
        on navigation; ``ScreenStateStore`` carries continuity instead),
        and it is the state the wake-integrity arc traced two live Console
        failures to (tasks 15970/15971).

        Overlays are dismissed rather than popped because ``switch_screen``
        and ``pop_screen`` both call ``_pop_result_callback()`` WITHOUT
        invoking it (``textual/app.py:3020``): a modal opened through
        ``push_screen_wait`` holds a future in that callback, so discarding
        it uncalled leaves the awaiting worker suspended forever -- it has
        no timeout and nothing else ever resolves it.
        ``Screen.dismiss(None)`` calls the callback first
        (``textual/screen.py:2048-2070``), so the awaiter resumes with the
        same ``None`` every user-driven close already delivers (``Escape``,
        ``action_dismiss``, a bare ``dismiss()``) -- the value existing
        callers, including the ones that map it to a decline, are already
        written against. Refusing to navigate while a modal is awaited was
        the alternative and is worse: awaited modals are the common kind,
        and a nav shortcut that silently no-ops is indistinguishable from a
        wedged app.

        Args:
            screen_name: Route being navigated to, for log context.

        Returns:
            ``True`` when the stack is reduced to its content screen and the
            switch may proceed; ``False`` when an overlay would not leave,
            in which case the caller must abort rather than switch and
            recreate the very leak this exists to prevent.
        """
        for _ in range(self._MAX_NAVIGATION_OVERLAY_DISMISSALS):
            stack = self._screen_stack
            if len(stack) <= 2:
                return True
            overlay = stack[-1]
            logger.info(
                "Dismissing pushed screen before navigating "
                "(route=%s, screen=%s, awaited=%s).",
                screen_name,
                type(overlay).__name__,
                self._navigation_overlay_awaiter_pending(overlay),
            )
            try:
                dismissed = overlay.dismiss(None)
                if inspect.isawaitable(dismissed):
                    await dismissed
            except Exception as exc:
                logger.warning(
                    "Pushed screen refused to dismiss before navigation "
                    "(route=%s, screen=%s, exception_category=%s).",
                    screen_name,
                    type(overlay).__name__,
                    type(exc).__name__,
                )
                return False
            stack = self._screen_stack
            if stack and stack[-1] is overlay:
                logger.warning(
                    "Pushed screen stayed on the stack after dismissal "
                    "(route=%s, screen=%s).",
                    screen_name,
                    type(overlay).__name__,
                )
                return False
        logger.warning(
            "Screen stack did not reduce to its content screen within %s "
            "dismissals (route=%s).",
            self._MAX_NAVIGATION_OVERLAY_DISMISSALS,
            screen_name,
        )
        return False

    async def _handle_screen_navigation_locked(self, message: NavigateToScreen) -> None:
        """Body of `handle_screen_navigation`, run under its FIFO lock."""
        requested_screen = message.screen_name
        if not getattr(self, "_initial_screen_pushed", False):
            # Until the initial screen exists (splash screen still up, or the
            # post-splash startup push still in flight) the screen stack has
            # no result callback to pop and switch_screen raises IndexError.
            # Swallow the request; the user can re-issue it once the app is
            # interactive.
            logger.info(
                f"Ignoring navigation to {requested_screen}: "
                "initial screen not yet mounted"
            )
            return
        screen_name, current_tab_value, screen_class = (
            self._resolve_screen_navigation_target(requested_screen)
        )
        logger.info(f"Navigating to screen: {requested_screen}")

        # NOT ``self.screen`` (TASK-16300): with a pushed screen on top --
        # the nav overflow menu, the command palette, a picker, a confirm
        # dialog -- ``self.screen`` IS that overlay, and every hook below
        # (flush, confirm, transition admission, and ``save_state`` inside
        # ``_complete_screen_navigation``) was asked of it. Overlays answer
        # none of them, so Console's busy-fleet confirmation never ran and
        # the tab being left was never snapshotted.
        current_screen = self._navigation_outgoing_screen()

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
                    # Shielded: giving up on the WAIT must not give up on the
                    # SAVE. The Library File Notes flush persists through
                    # `asyncio.to_thread`, which cannot be cancelled -- an
                    # unshielded `wait_for` killed the coroutine at that await
                    # while the thread kept writing, so `_save_draft` never ran
                    # its reconciliation: `_save_state` stayed "saving" (which
                    # makes `leave_allowed` False *forever*) and the cached
                    # `content_hash` stayed stale, so the next save reported a
                    # spurious conflict.
                    flush_task = asyncio.ensure_future(flush_result)
                    try:
                        flush_result = await asyncio.wait_for(
                            asyncio.shield(flush_task),
                            timeout=self.NAVIGATION_FLUSH_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        self._retain_unfinished_flush(flush_task, screen_name)
                        raise
                if flush_result is False:
                    logger.info(
                        f"Navigation to {screen_name} vetoed by the outgoing "
                        "screen's pending-work flush"
                    )
                    return
            except asyncio.TimeoutError:
                # Fail closed, exactly like a flush that raised: the pending
                # edits may exist ONLY in the outgoing screen, so keep it
                # mounted rather than discarding it on a save we can't
                # confirm. Abandoning the wait does not abandon the save --
                # the note-save worker is a separate task and keeps running.
                logger.warning(
                    "Screen flush timed out after %ss; staying put (route=%s).",
                    self.NAVIGATION_FLUSH_TIMEOUT_SECONDS,
                    screen_name,
                )
                try:
                    self.notify(
                        "Still saving pending changes; staying on this screen. "
                        "Try again in a moment.",
                        severity="warning",
                    )
                except Exception:
                    pass
                return
            except Exception as exc:
                # The outgoing instance may be the only place pending edits
                # still exist, so a failed flush must abort the transition.
                logger.warning(
                    "Screen flush failed (route=%s, exception_category=%s).",
                    screen_name,
                    type(exc).__name__,
                )
                try:
                    self.notify(
                        "Couldn't save pending changes before switching screens.",
                        severity="warning",
                    )
                except Exception:
                    pass
                return

        # TASK-1143 (F5): give the outgoing screen one awaited chance to
        # ASK before it (and whatever it owns) is torn down -- e.g. Console
        # unmounting cancels every in-flight run and denies every pending/
        # parked approval round for its ConsoleChatController (see
        # ChatScreen.on_unmount / ConsoleChatController.shutdown). Mirrors
        # the flush-veto seam immediately above: False keeps the outgoing
        # screen mounted exactly like a flush veto, only here the decision
        # comes from a user-facing confirmation dialog rather than an
        # unresolved save conflict.
        confirm_navigation = getattr(current_screen, "confirm_navigation", None)
        if callable(confirm_navigation):
            try:
                confirm_result = confirm_navigation()
                if inspect.isawaitable(confirm_result):
                    confirm_result = await confirm_result
                if confirm_result is False:
                    logger.info(
                        f"Navigation to {screen_name} vetoed by the outgoing "
                        "screen's confirm_navigation"
                    )
                    return
            except Exception as exc:
                # A broken confirm hook must not silently let navigation
                # proceed and tear down live work the user was never asked
                # about -- fail closed, same as the flush veto above.
                logger.warning(
                    "Screen navigation confirm failed (route=%s, exception_category=%s).",
                    screen_name,
                    type(exc).__name__,
                )
                try:
                    self.notify(
                        "Couldn't confirm leaving this screen; staying put.",
                        severity="warning",
                    )
                except Exception:
                    pass
                return

        release_navigation = None
        acquire_navigation = getattr(
            current_screen,
            "acquire_navigation_transition",
            None,
        )
        if callable(acquire_navigation):
            admission = acquire_navigation()
            if admission is False:
                logger.info(
                    f"Navigation to {screen_name} vetoed by the outgoing "
                    "screen's transition admission"
                )
                return
            release_navigation = admission
        try:
            await self._complete_screen_navigation(
                message=message,
                requested_screen=requested_screen,
                screen_name=screen_name,
                current_tab_value=current_tab_value,
                screen_class=screen_class,
                current_screen=current_screen,
            )
        finally:
            if callable(release_navigation):
                release_navigation()

    def _retain_unfinished_flush(self, flush_task: Any, screen_name: str) -> None:
        """Keep a timed-out flush alive until it finishes on its own.

        The navigation wait is shielded, so the flush keeps running after the
        app stops waiting -- but asyncio only holds a weak reference to a
        running task, so without a strong reference here it could be garbage
        collected mid-save. Retaining it also gives somewhere to consume the
        eventual result, which otherwise surfaces as "exception was never
        retrieved" noise long after the navigation that started it.

        Args:
            flush_task: The still-running flush task.
            screen_name: Route being navigated to, for log context.
        """
        pending = getattr(self, "_pending_flush_tasks", None)
        if pending is None:
            pending = set()
            self._pending_flush_tasks = pending
        pending.add(flush_task)

        def _finished(task: Any) -> None:
            pending.discard(task)
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.warning(
                    "Screen flush eventually failed after navigation gave up "
                    "waiting (route=%s, exception_category=%s).",
                    screen_name,
                    type(exc).__name__,
                )
            else:
                logger.info(
                    "Screen flush eventually completed after navigation gave "
                    "up waiting (route=%s).",
                    screen_name,
                )

        flush_task.add_done_callback(_finished)

    def _notify_navigation_failure(self, screen_name: str) -> None:
        """Tell the user a destination failed to open, without raising.

        Navigation failures are reported where they happen so the user is
        not left staring at an unchanged screen wondering whether the click
        registered. ``notify`` itself is guarded: this runs on the crash
        path, and a failure to display the message must not replace one
        escaping exception with another.
        """
        try:
            self.notify(
                f"Couldn't open {screen_name}. Staying on the current screen.",
                severity="error",
            )
        except Exception:
            logger.debug(f"Could not surface navigation failure for {screen_name!r}.")
        # task-2720: the nav bar highlighted the destination the moment it was
        # clicked, before the navigation worker ran. Roll it back to the screen
        # actually on the stack — otherwise the bar shows a destination that
        # never loaded AND its already-active check swallows every retry click,
        # leaving the destination unreachable until restart.
        #
        # task-2854: use ``nav_bar_active``, not ``screen_name`` -- a screen
        # whose route is folded under another destination for routing/label
        # purposes only (e.g. Study folds under Library) sets
        # ``nav_bar_active`` to a value that clears its own nav bar's
        # highlight instead of falsely re-claiming the owning destination
        # (see ``BaseAppScreen.nav_bar_active``). ``screen_name`` is kept as
        # a fallback for any screen that predates that attribute.
        # ``nav_bar_active`` may legitimately be ``""`` (Study's case), which
        # must still reach ``restore_active`` -- ``resolve_shell_route("")``
        # matches no destination, so every nav button loses ``is-active``
        # rather than the call being skipped and the stale optimistic
        # highlight surviving.
        try:
            current_screen = self.screen
            current_route = getattr(current_screen, "nav_bar_active", None)
            if current_route is None:
                current_route = getattr(current_screen, "screen_name", None)
            if isinstance(current_route, str):
                current_screen.query_one(MainNavigationBar).restore_active(
                    current_route
                )
        except Exception:
            logger.debug(
                f"Could not roll back nav-bar state after failing to open "
                f"{screen_name!r}."
            )

    async def _complete_screen_navigation(
        self,
        *,
        message: NavigateToScreen,
        requested_screen: str,
        screen_name: str,
        current_tab_value: str,
        screen_class: type | None,
        current_screen: Any,
    ) -> None:
        """Save, construct, restore, and switch while transition admission is held."""
        runtime_identity = self._current_runtime_identity()
        outgoing_key = str(self.current_tab or "").strip()
        if not outgoing_key:
            outgoing_screen_name = getattr(current_screen, "screen_name", None)
            if isinstance(outgoing_screen_name, str) and outgoing_screen_name.strip():
                (
                    _outgoing_screen_name,
                    resolved_outgoing_key,
                    outgoing_screen_class,
                ) = self._resolve_screen_navigation_target(outgoing_screen_name.strip())
                if outgoing_screen_class is not None:
                    outgoing_key = resolved_outgoing_key

        # A Console snapshot that fails to be replaced below must not survive
        # with its published prompt-target projection attached (the stale
        # target `publish_console_prompt_target` would otherwise be read back
        # against). Since task-15860 this discards VIEW state only: Console's
        # sessions and transcripts live in the app-owned `ConsoleRuntime`
        # store, which no snapshot lifecycle can drop.
        if outgoing_key == TAB_CHAT:
            self.screen_state_store.discard(outgoing_key)

        save_state = getattr(current_screen, "save_state", None)
        if outgoing_key and callable(save_state):
            try:
                state = save_state()
                if isinstance(state, Mapping):
                    self.screen_state_store.save(
                        outgoing_key,
                        state,
                        runtime_identity,
                    )
                    if outgoing_key == TAB_CHAT:
                        projection_getter = getattr(
                            current_screen,
                            "console_prompt_target_projection",
                            None,
                        )
                        try:
                            projection = (
                                projection_getter()
                                if callable(projection_getter)
                                else None
                            )
                        except Exception:
                            projection = None
                        if isinstance(projection, ConsolePromptTargetProjection):
                            self.screen_state_store.publish_console_prompt_target(
                                outgoing_key,
                                projection,
                                runtime_identity,
                            )
                    logger.debug(
                        "Saved screen snapshot for canonical route: %s",
                        outgoing_key,
                    )
                else:
                    logger.warning(
                        "Screen snapshot save skipped (route=%s, reason=non_mapping).",
                        outgoing_key,
                    )
            except Exception as exc:
                logger.warning(
                    "Screen snapshot save failed (route=%s, exception_category=%s).",
                    outgoing_key,
                    type(exc).__name__,
                )

        if screen_class:
            try:
                new_screen = self._create_navigation_screen(screen_name, screen_class)
            except Exception as exc:
                # A destination that cannot even be constructed is a broken
                # destination, never a dead app. This ran unguarded until
                # 2026-07-28: the MCP canvases read `Select.NULL` (Textual 8+)
                # at construction time, so on an older Textual the
                # AttributeError escaped this handler and Textual exited the
                # whole app rather than the user simply failing to reach MCP.
                logger.opt(exception=True).error(
                    "Screen construction failed (route={}, exception_category={}).",
                    screen_name,
                    type(exc).__name__,
                )
                self._notify_navigation_failure(screen_name)
                return

            restored_state = self.screen_state_store.restore(
                current_tab_value,
                runtime_identity,
            )
            restore_state = getattr(new_screen, "restore_state", None)
            if restored_state is not None and callable(restore_state):
                try:
                    restore_state(restored_state)
                    logger.debug(
                        "Restored screen snapshot for canonical route: %s",
                        current_tab_value,
                    )
                except Exception as exc:
                    self.screen_state_store.discard(current_tab_value)
                    logger.warning(
                        "Screen snapshot restore failed "
                        "(route=%s, exception_category=%s).",
                        current_tab_value,
                        type(exc).__name__,
                    )

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
                except Exception as exc:
                    logger.warning(
                        "Navigation context application failed "
                        "(route=%s, exception_category=%s).",
                        current_tab_value,
                        type(exc).__name__,
                    )

            # TASK-16300: `switch_screen` replaces the TOP of the stack, so
            # the content screen has to BE the top before it runs -- see
            # `_dismiss_navigation_overlays`. Done here, after the veto
            # hooks and the construction of the incoming screen, so a
            # navigation that never happens never costs the user the dialog
            # they had open. Failing to reduce aborts: switching anyway is
            # exactly how the outgoing screen is left resident.
            if not await self._dismiss_navigation_overlays(screen_name):
                logger.warning(
                    "Aborting navigation: a pushed screen would not leave "
                    "the stack (route=%s).",
                    screen_name,
                )
                self._notify_navigation_failure(screen_name)
                return

            # Use switch_screen to replace the current screen
            try:
                await self.switch_screen(new_screen)
            except Exception as exc:
                # Sibling of the construction guard above: a screen can also
                # fail while composing/mounting (the MCP audit canvas reads
                # `Select.NULL` inside compose()), and Textual surfaces that
                # through switch_screen. Same rule -- report the broken
                # destination instead of taking the app down with it.
                logger.opt(exception=True).error(
                    "Screen mount failed (route={}, exception_category={}).",
                    screen_name,
                    type(exc).__name__,
                )
                self._notify_navigation_failure(screen_name)
                return

            # Keep current_tab aligned to canonical tab ids even when routing uses aliases.
            self.current_tab = current_tab_value

            # task-18812: the exit rule runs only once the switch has
            # SUCCEEDED -- flush vetoes, confirmations, admission, and mount
            # failures above all `return` with the Console still resident, so
            # clearing earlier would desync the app flag from the mounted
            # screen's -focus class (the next toggle would do the wrong
            # visible action).
            self._clear_focus_if_leaving_console(screen_name)

            logger.info(f"Successfully switched to {screen_name} screen")
        else:
            logger.error(f"Unknown screen requested: {requested_screen}")

    @on(TTSRequestEvent)
    async def handle_tts_request_event(self, event: TTSRequestEvent) -> None:
        """Handle TTS generation request."""
        self.loguru_logger.info(
            f"TTS request received for text: '{event.text[:50]}...'"
        )
        handler = await self._ensure_tts_handler()
        if handler:
            await handler.handle_tts_request(event)
        else:
            self.loguru_logger.error("TTS handler not initialized")
            self.post_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error="TTS service not available",
                )
            )

    @on(TTSMessageSpeechRequestEvent)
    async def handle_tts_message_speech_request_event(
        self,
        event: TTSMessageSpeechRequestEvent,
    ) -> None:
        """Route a trusted Console snapshot without logging private content."""
        self.loguru_logger.info("Trusted Console speech request received")
        try:
            handler = await self._ensure_tts_handler()
        except asyncio.CancelledError:
            event.report_outcome(False)
            raise
        except Exception as error:
            self.loguru_logger.error(
                "TTS handler initialization failed "
                "(operation=trusted_console_speech, exception_category={})",
                type(error).__name__,
            )
            event.report_outcome(False)
            return
        if handler:
            try:
                await handler.handle_tts_request(event)
            except asyncio.CancelledError:
                event.report_outcome(False)
                raise
            except Exception as error:
                self.loguru_logger.error(
                    "TTS handler request failed "
                    "(operation=trusted_console_speech, exception_category={})",
                    type(error).__name__,
                )
                event.report_outcome(False)
        else:
            self.loguru_logger.error(
                "TTS handler not initialized "
                "(operation=trusted_console_speech, "
                "outcome_code=handler_unavailable)"
            )
            try:
                self.post_message(
                    TTSCompleteEvent(
                        message_id=event.message_id,
                        error="TTS service not available",
                    )
                )
            except Exception as error:
                self.loguru_logger.error(
                    "TTS unavailable notice failed "
                    "(operation=trusted_console_speech, exception_category={})",
                    type(error).__name__,
                )
            finally:
                event.report_outcome(False)

    @on(TTSGlobalOverrideDecisionEvent)
    async def handle_tts_global_override_decision_event(
        self,
        event: TTSGlobalOverrideDecisionEvent,
    ) -> None:
        """Route one opaque character-speech fallback decision.

        Args:
            event: The accepted or rejected message-scoped fallback decision.
        """
        handler = await self._ensure_tts_handler()
        if handler:
            await handler.handle_tts_global_override_decision(event)
        else:
            self.loguru_logger.error(
                "TTS handler not initialized "
                "(operation=global_voice_fallback, "
                "outcome_code=handler_unavailable)"
            )

    async def _offer_tts_global_override(self, token: str) -> None:
        """Prompt for one message-scoped global-voice fallback.

        The dialog's copy names the actual configured-voice domain that
        refused (a per-character assignment vs. the app-wide default voice
        profile) -- looked up, without consuming the token, from the
        issuing handler's still-pending state
        (`TTSEventHandler.peek_global_override_voice_domain`). Review
        round 2: the completion toast already used `event.error`'s
        domain-accurate copy; this dialog previously did not, and always
        said "character" even for a default-profile refusal on a message
        with no character context at all.
        """
        handler = getattr(self, "_tts_handler", None)
        voice_domain = (
            handler.peek_global_override_voice_domain(token)
            if handler is not None
            else None
        )
        message = _TTS_GLOBAL_OVERRIDE_PROMPT_COPY.get(
            voice_domain,
            _TTS_GLOBAL_OVERRIDE_PROMPT_COPY[None],
        )
        decision = False
        try:
            result = await self.push_screen_wait(
                ConfirmationDialog(
                    title="Use global voice?",
                    message=message,
                    confirm_label="Use global",
                    cancel_label="Cancel",
                )
            )
            decision = result is True
        except asyncio.CancelledError:
            self.post_message(TTSGlobalOverrideDecisionEvent(token, accepted=False))
            raise
        except Exception as error:
            self.loguru_logger.warning(
                "TTS global fallback prompt failed (exception_category={})",
                type(error).__name__,
            )
        self.post_message(TTSGlobalOverrideDecisionEvent(token, accepted=decision))

    @on(TTSCompleteEvent)
    async def handle_tts_complete_event(self, event: TTSCompleteEvent) -> None:
        """Handle TTS generation completion."""
        from tldw_chatbook.Widgets.Chat_Widgets.chat_message_enhanced import (  # noqa: PLC0415 - keeps PIL/textual_image off the boot path (TASK-21103)
            ChatMessageEnhanced,
        )

        self.loguru_logger.info(f"TTS complete for message {event.message_id}")
        playback_lifecycle = getattr(event, "playback_lifecycle", None)

        lifecycle_failure_completion = bool(
            event.error
            and playback_lifecycle is not None
            and playback_lifecycle.state == "failed"
        )
        if (
            playback_lifecycle is not None
            and not playback_lifecycle.is_current()
            and not lifecycle_failure_completion
        ):
            handler = getattr(self, "_tts_handler", None)
            discard = getattr(handler, "discard_stale_console_completion", None)
            if callable(discard):
                try:
                    await discard(
                        event.message_id,
                        event.audio_file,
                        playback_lifecycle,
                    )
                except Exception:
                    playback_lifecycle.report_terminal("failed")
            else:
                playback_lifecycle.report_terminal("stopped")
            return

        if event.error:
            if playback_lifecycle is not None:
                playback_lifecycle.report("failed")
            self.notify(f"TTS failed: {event.error}", severity="error")
            # Update widget state back to idle on error
            try:
                if event.message_id:
                    # Find the message widget and update state
                    for message_widget in list(self.query(ChatMessage)) + list(
                        self.query(ChatMessageEnhanced)
                    ):
                        if (
                            getattr(message_widget, "message_id_internal", None)
                            == event.message_id
                        ):
                            # Update TTS state to idle on error
                            if hasattr(message_widget, "update_tts_state"):
                                message_widget.update_tts_state("idle")
                            # Remove TTS generating class
                            text_widget = message_widget.query_one(
                                ".message-text", Markdown
                            )
                            text_widget.remove_class("tts-generating")
                            break
            except Exception as e:
                self.loguru_logger.error(f"Error updating message UI: {e}")
            # The Console transcript's action row renders from the screen's
            # `_console_speaking_message_id`, not from a legacy widget — on
            # failure it must be cleared too, or the row keeps "⏹ Stop
            # speech" with no speech to stop (TASK-15422).
            if playback_lifecycle is None:
                for screen in reversed(tuple(getattr(self, "screen_stack", ()))):
                    if (
                        getattr(screen, "_console_speaking_message_id", None)
                        == event.message_id
                    ):
                        screen._console_speaking_message_id = None
                        sync = getattr(screen, "_sync_native_console_chat_ui", None)
                        if callable(sync):
                            try:
                                await sync()
                            except Exception:
                                self.loguru_logger.error(
                                    "Console speak-state resync failed after a "
                                    "TTS error"
                                )
                        break
            if event.global_override_token is not None:
                self.run_worker(
                    self._offer_tts_global_override(event.global_override_token),
                    name="tts_global_voice_confirmation",
                )
        else:
            # Update widget state to ready with audio file
            if event.audio_file and event.audio_file.exists():
                if (
                    playback_lifecycle is not None
                    and not playback_lifecycle.is_current()
                ):
                    return
                try:
                    widget_found = False
                    if event.message_id:
                        # Find the message widget and update state
                        for message_widget in list(self.query(ChatMessage)) + list(
                            self.query(ChatMessageEnhanced)
                        ):
                            if (
                                getattr(message_widget, "message_id_internal", None)
                                == event.message_id
                            ):
                                widget_found = True
                                # Update TTS state to ready with audio file
                                if hasattr(message_widget, "update_tts_state"):
                                    message_widget.update_tts_state(
                                        "ready", event.audio_file
                                    )
                                # Remove TTS generating class
                                try:
                                    text_widget = message_widget.query_one(
                                        ".message-text", Markdown
                                    )
                                    text_widget.remove_class("tts-generating")
                                except Exception:
                                    pass
                                break
                    if widget_found:
                        # A legacy ChatMessage/ChatMessageEnhanced widget owns
                        # this message and exposes its own play control - let
                        # the user trigger playback explicitly rather than
                        # auto-playing underneath them.
                        self.notify(
                            "TTS audio ready - click play to listen",
                            severity="information",
                        )
                    else:
                        # No legacy widget claims this message (e.g. Console,
                        # which has no per-message playback control), so
                        # there is nothing for the user to click - play the
                        # generated audio immediately instead of going silent.
                        accepted = self.post_message(
                            TTSPlaybackEvent(
                                action="play",
                                message_id=event.message_id,
                                playback_lifecycle=playback_lifecycle,
                            )
                        )
                        if accepted is False and playback_lifecycle is not None:
                            playback_lifecycle.report("failed")
                except Exception as e:
                    if playback_lifecycle is not None:
                        playback_lifecycle.report("failed")
                    self.loguru_logger.error(f"Error playing audio: {e}")
                    self.notify("Failed to play audio", severity="error")
            elif (
                playback_lifecycle is not None
                and playback_lifecycle.state == "generating"
            ):
                playback_lifecycle.report("failed")

            # Remove TTS generating class from message
            try:
                if event.message_id:
                    for message_widget in list(self.query(ChatMessage)) + list(
                        self.query(ChatMessageEnhanced)
                    ):
                        if (
                            getattr(message_widget, "message_id_internal", None)
                            == event.message_id
                        ):
                            text_widget = message_widget.query_one(
                                ".message-text", Markdown
                            )
                            text_widget.remove_class("tts-generating")
                            break
            except Exception as e:
                self.loguru_logger.error(f"Error updating message UI: {e}")

    @on(TTSProgressEvent)
    async def handle_tts_progress_event(self, event: TTSProgressEvent) -> None:
        """Handle TTS generation progress updates."""
        from tldw_chatbook.Widgets.Chat_Widgets.chat_message_enhanced import (  # noqa: PLC0415 - keeps PIL/textual_image off the boot path (TASK-21103)
            ChatMessageEnhanced,
        )

        self.loguru_logger.debug(
            f"TTS progress for message {event.message_id}: {event.progress:.0%} - {event.status}"
        )

        try:
            if event.message_id:
                # Find the message widget and update progress
                for message_widget in list(self.query(ChatMessage)) + list(
                    self.query(ChatMessageEnhanced)
                ):
                    if (
                        getattr(message_widget, "message_id_internal", None)
                        == event.message_id
                    ):
                        # Update TTS progress
                        if hasattr(message_widget, "update_tts_progress"):
                            message_widget.update_tts_progress(
                                event.progress, event.status
                            )
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
        await self.control_tts_playback(event)

    async def control_tts_playback(self, event: TTSPlaybackEvent) -> None:
        """Run playback control directly and preserve handler callback order."""
        try:
            handler = await self._ensure_tts_handler()
            if handler:
                await handler.handle_tts_playback(event)
            else:
                event.report_outcome(False)
                if event.playback_lifecycle is not None:
                    event.playback_lifecycle.report_terminal("failed")
        except asyncio.CancelledError:
            event.report_outcome(False)
            raise
        except Exception:
            event.report_outcome(False)
            if event.playback_lifecycle is not None:
                event.playback_lifecycle.report_terminal("failed")

    @on(STTSPlaygroundGenerateEvent)
    async def handle_stts_playground_generate_event(
        self, event: STTSPlaygroundGenerateEvent
    ) -> None:
        """Handle S/TT/S playground generation request."""
        self.loguru_logger.info(
            "S/TT/S generation request accepted for provider={}",
            event.request.provider_id,
        )
        handler = await self._ensure_stts_handler()
        if handler:
            handler.start_playground_generation(event)
        else:
            self.loguru_logger.error("S/TT/S handler not initialized")
            self.notify("S/TT/S service not available", severity="error")

    @on(STTSSettingsSaveEvent)
    async def handle_stts_settings_save_event(
        self, event: STTSSettingsSaveEvent
    ) -> None:
        """Handle S/TT/S settings save."""
        handler = await self._ensure_stts_handler()
        if handler:
            await handler.handle_settings_save(event)

    @on(STTSProviderConfigurationChanged)
    def handle_stts_provider_configuration_changed(
        self,
        event: STTSProviderConfigurationChanged,
    ) -> None:
        """Forward provider invalidation to the retained STTS handler."""
        handler = getattr(self, "_stts_handler", None)
        if handler is not None:
            handler.on_stts_provider_configuration_changed(event)
        if event.provider_id == "audio_cpp":
            from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

            current_screen = getattr(self, "screen", None)
            if current_screen is not None:
                for window in current_screen.query(LLMManagementWindow):
                    window.refresh_model_library_observations()

    @on(STTSAudioBookGenerateEvent)
    async def handle_stts_audiobook_generate_event(
        self, event: STTSAudioBookGenerateEvent
    ) -> None:
        """Handle audiobook generation request."""
        handler = await self._ensure_stts_handler()
        if handler:
            await handler.handle_audiobook_generate(event)

    def _bind_tts_service(self) -> None:
        """Bind the single TTS service owned by this application."""
        if self._tts_binding_active:
            return
        bind_tts_service(self.tts_service)
        self._tts_binding_active = True

    async def _close_tts_service(self) -> None:
        """Close and unbind the application-owned TTS service once."""
        if not self._tts_binding_active:
            return
        try:
            await close_tts_resources()
        finally:
            self._tts_binding_active = False

    async def _ensure_tts_profile_repository(
        self,
    ) -> TTSProfileRepository | None:
        """Open and return the one app-owned profile repository on first use."""

        repository = getattr(self, "_tts_profile_repository", None)
        if repository is None:
            return None
        if getattr(self, "_tts_profile_repository_close_task", None) is not None:
            return None
        if repository.state is ProfileRepositoryState.OPEN:
            return repository

        open_task = getattr(self, "_tts_profile_repository_open_task", None)
        if open_task is None or open_task.done():

            async def open_repository() -> bool:
                try:
                    await repository.open()
                except Exception as error:
                    error_code = (
                        error.code
                        if isinstance(error, ProfileRepositoryError)
                        else "operation_failed"
                    )
                    self.loguru_logger.warning(
                        "TTS profile repository phase=open failed "
                        f"type={type(error).__name__} code={error_code}"
                    )
                    return False
                return repository.state is ProfileRepositoryState.OPEN

            open_task = asyncio.create_task(
                open_repository(),
                name="open_tts_profile_repository",
            )
            self._tts_profile_repository_open_task = open_task

            def settle_open_task(completed: asyncio.Task[bool]) -> None:
                try:
                    completed.exception()
                except BaseException:
                    pass
                finally:
                    if self._tts_profile_repository_open_task is completed:
                        self._tts_profile_repository_open_task = None

            open_task.add_done_callback(settle_open_task)

        try:
            opened = await asyncio.shield(open_task)
        except asyncio.CancelledError:
            raise

        if (
            not opened
            or repository.state is not ProfileRepositoryState.OPEN
            or getattr(self, "_tts_profile_repository_close_task", None) is not None
        ):
            return None
        return repository

    async def _ensure_tts_profile_service(self) -> TTSProfileService | None:
        """Return one profile service over the existing app-owned dependencies."""

        repository = await self._ensure_tts_profile_repository()
        if repository is None:
            return None

        profile_service = getattr(self, "_tts_profile_service", None)
        if profile_service is None:
            profile_service = TTSProfileService(
                repository,
                self.tts_service,
                artifact_lease_coordinator=(
                    self._ensure_audio_cpp_artifact_lease_coordinator()
                ),
            )
            self._tts_profile_service = profile_service
        return profile_service

    def _saved_audio_cpp_managed_consumers(
        self,
    ) -> tuple[AudioCppManagedConsumerIdentity, ...]:
        """Project only exact managed identities from immutable saved Settings."""

        try:
            config = project_audio_cpp_settings_config(self.app_config)
        except (TypeError, ValueError):
            config = AudioCppSettingsConfig()
        return tuple(
            AudioCppManagedConsumerIdentity(
                recipe_id=package.recipe_id,
                recipe_revision=package.recipe_revision,
                model_id=package.public_model_id,
                managed_artifact=package.managed_artifact,
            )
            for package in config.guided_packages
            if package.managed_artifact is not None
        )

    def _ensure_audio_cpp_artifact_lease_coordinator(
        self,
    ) -> AudioCppArtifactLeaseCoordinator:
        """Return the one app-owned coordinator over the shared artifact owner."""

        coordinator = self._audio_cpp_artifact_lease_coordinator
        if coordinator is None:
            coordinator = AudioCppArtifactLeaseCoordinator(
                managed_service(),
                saved_settings_snapshot=self._saved_audio_cpp_managed_consumers,
            )
            self._audio_cpp_artifact_lease_coordinator = coordinator
        return coordinator

    def _audio_cpp_removal_settings_inputs(
        self,
    ) -> tuple[
        AudioCppSettingsConfig,
        AudioCppSettingsConfig | None,
        TTSPreferencesSnapshot,
        TTSPreferencesSnapshot | None,
    ]:
        """Read saved plus exact detached-or-mounted Speech/TTS draft state."""

        try:
            saved = project_audio_cpp_settings_config(self.app_config)
            saved_preferences = TTSPreferencesSnapshot.from_settings(self.app_config)
        except (TypeError, ValueError):
            raise ProfileRepositoryError("unavailable") from None

        draft_snapshot: SpeechTTSPanelDraftSnapshot | None = None
        stored_settings_state = False
        store = getattr(self, "screen_state_store", None)
        if store is not None:
            try:
                stored = store.restore(TAB_SETTINGS, self._current_runtime_identity())
            except Exception:
                raise ProfileRepositoryError("unavailable") from None
            if stored is not None:
                stored_settings_state = True
                if "speech_tts_panel_draft" in stored:
                    candidate = stored["speech_tts_panel_draft"]
                    if type(candidate) is not SpeechTTSPanelDraftSnapshot:
                        raise ProfileRepositoryError("unavailable")
                    draft_snapshot = candidate
        if draft_snapshot is None and not stored_settings_state:
            current_screen = getattr(self, "screen", None)
            candidate = getattr(current_screen, "_speech_tts_draft_snapshot", None)
            if type(candidate) is SpeechTTSPanelDraftSnapshot:
                draft_snapshot = candidate

        if draft_snapshot is None:
            return saved, None, saved_preferences, None
        try:
            provider = draft_snapshot.state.providers.get("audio_cpp")
            if not isinstance(provider, dict):
                raise ValueError
            draft = AudioCppSettingsConfig.from_mapping(provider)
            draft_preferences = draft_snapshot.state.defaults.snapshot()
        except (TypeError, ValueError):
            raise ProfileRepositoryError("unavailable") from None
        return saved, draft, saved_preferences, draft_preferences

    async def _audio_cpp_model_library_observation_snapshot(
        self,
        references: tuple["ArtifactRef", ...],
    ) -> AudioCppModelLibraryObservationSnapshot:
        """Collect shared evidence once, then project every exact package ref."""

        from tldw_chatbook.Model_Artifacts.service import ArtifactRef

        if type(references) is not tuple or any(
            type(reference) is not ArtifactRef for reference in references
        ):
            raise TypeError("references must be a tuple of ArtifactRef values")
        if len(set(references)) != len(references):
            raise ValueError("references must be unique")
        if not references:
            return AudioCppModelLibraryObservationSnapshot(())

        saved, draft, saved_preferences, draft_preferences = (
            self._audio_cpp_removal_settings_inputs()
        )

        profile_service = await self._ensure_tts_profile_service()
        if profile_service is None:
            raise ProfileRepositoryError("unavailable")
        profiles_with_counts = (
            await profile_service.bounded_profile_assignment_snapshot()
        )

        try:
            configuration = (
                await self.tts_service.registry.provider_configuration_snapshot(
                    "audio_cpp"
                )
            )
            staged_config = (
                None
                if configuration.staged_config is None
                else AudioCppSettingsConfig.from_mapping(configuration.staged_config)
            )
            applied_config = AudioCppSettingsConfig.from_mapping(
                configuration.applied_config
            )
            supervisor = getattr(self.tts_service, "_audio_cpp_supervisor", None)
            admission = None if supervisor is None else supervisor.admission_snapshot()
        except Exception:
            # Runtime evidence is safety-relevant; fail closed without exposing
            # collaborator details through the removal review.
            raise ProfileRepositoryError("unavailable") from None

        def contains(
            config: AudioCppSettingsConfig | None,
            reference: ArtifactRef,
        ) -> bool:
            return config is not None and any(
                package.managed_artifact is not None
                and (
                    package.managed_artifact.artifact_id,
                    package.managed_artifact.revision,
                    package.managed_artifact.variant,
                )
                == (reference.artifact_id, reference.revision, reference.variant)
                for package in config.guided_packages
            )

        live = admission is not None and admission.state in {
            "starting",
            "running",
            "draining",
            "stopping",
        }
        return AudioCppModelLibraryObservationSnapshot(
            tuple(
                project_audio_cpp_artifact_removal_evidence(
                    reference,
                    saved_settings=saved,
                    draft_settings=draft,
                    saved_preferences=saved_preferences,
                    draft_preferences=draft_preferences,
                    profiles=profiles_with_counts,
                    staged_runtime_ids=(
                        (f"settings-generation-{configuration.staged_generation}",)
                        if contains(staged_config, reference)
                        else ()
                    ),
                    live_runtime_ids=(
                        (f"process-generation-{admission.process_generation}",)
                        if live and contains(applied_config, reference)
                        else ()
                    ),
                )
                for reference in references
            )
        )

    async def _audio_cpp_artifact_removal_evidence(
        self,
        reference: "ArtifactRef",
    ) -> AudioCppArtifactRemovalEvidence:
        """Collect Task 9 removal evidence through the shared bulk snapshot."""

        snapshot = await TldwCli._audio_cpp_model_library_observation_snapshot(
            self,
            (reference,),
        )
        return snapshot.observations[0]

    async def _ensure_tts_voice_bundle_service(
        self,
    ) -> TTSVoiceBundlePortabilityService | None:
        """Construct the app-owned portability owner only on first use."""

        if getattr(self, "_tts_voice_bundle_service_close_task", None) is not None:
            return None
        profile_service = await self._ensure_tts_profile_service()
        if profile_service is None:
            return None
        service = getattr(self, "_tts_voice_bundle_service", None)
        if service is None:
            service = TTSVoiceBundlePortabilityService(
                get_user_data_dir() / "tts_voice_bundle_portability",
                self._tts_profile_repository,
                self.tts_service,
                profile_mutation_fence=profile_service.consumer_mutation_fence,
                artifact_lease_coordinator=(
                    self._ensure_audio_cpp_artifact_lease_coordinator()
                ),
            )
            self._tts_voice_bundle_service = service
        return service

    async def _close_tts_voice_bundle_service(self) -> None:
        """Close and join portability before repository authority is released."""

        service = getattr(self, "_tts_voice_bundle_service", None)
        if service is None:
            return
        close_task = getattr(self, "_tts_voice_bundle_service_close_task", None)
        if close_task is None:

            async def close_portability() -> None:
                await service.close()
                await service.wait_closed()

            close_task = asyncio.create_task(
                close_portability(),
                name="close_tts_voice_bundle_service",
            )
            self._tts_voice_bundle_service_close_task = close_task
        await join_retained_task(close_task)

    async def _close_tts_profile_repository(self) -> None:
        """Definitively close the app-owned profile repository once."""

        repository = getattr(self, "_tts_profile_repository", None)
        if repository is None:
            return

        close_task = getattr(self, "_tts_profile_repository_close_task", None)
        if close_task is None:

            async def close_repository() -> None:
                await repository.close()

            close_task = asyncio.create_task(
                close_repository(),
                name="close_tts_profile_repository",
            )
            self._tts_profile_repository_close_task = close_task

        def record_failure_after_cancellation(
            cancellation: BaseException,
            cleanup_error: BaseException,
        ) -> None:
            cancellation.add_note(
                "TTS profile repository cleanup also failed while preserving "
                "shutdown cancellation"
            )
            self.loguru_logger.warning(
                "TTS profile repository phase=close failed while preserving "
                f"cancellation type={type(cleanup_error).__name__} "
                "code=operation_failed"
            )

        await join_retained_task(
            close_task,
            on_failure_after_cancellation=record_failure_after_cancellation,
        )

    async def _close_owned_tts_resources(self) -> None:
        """Close app-owned TTS resources without masking cancellation."""

        failures: list[tuple[str, BaseException]] = []
        if hasattr(self, "_close_tts_voice_bundle_service"):
            try:
                await self._close_tts_voice_bundle_service()
            except BaseException as portability_close_error:
                failures.append(("voice_bundle_service", portability_close_error))

        try:
            await self._close_tts_profile_repository()
        except BaseException as profile_close_error:
            failures.append(("profile_repository", profile_close_error))

        try:
            await self._close_tts_service()
        except BaseException as service_close_error:
            failures.append(("tts_service", service_close_error))

        if not failures:
            return

        control_flow_failures = [
            failure for failure in failures if not isinstance(failure[1], Exception)
        ]
        primary_phase, primary_error = (
            control_flow_failures[0] if control_flow_failures else failures[0]
        )
        for phase, failure_error in failures:
            if failure_error is primary_error:
                continue
            primary_error.add_note(
                "TTS owner cleanup also failed while preserving the primary error"
            )
            self.loguru_logger.warning(
                f"TTS owner cleanup phase={phase} failed while preserving "
                f"phase={primary_phase} type={type(failure_error).__name__} "
                "code=operation_failed"
            )
        raise primary_error

    def _observe_notes_sync_runtime_start(self, task: asyncio.Task[None]) -> None:
        """Consume a detached startup failure without exposing private detail."""

        if task.cancelled():
            return
        if task.exception() is not None:
            logger.error("Notes sync runtime startup failed.")
            return
        screen = self.screen
        refresh = getattr(screen, "refresh_notes_sync_runtime", None)
        if callable(refresh):
            self.call_after_refresh(refresh)

    def on_mount(self) -> None:
        """Configure logging and schedule post-mount setup."""
        self._bind_tts_service()
        self._notes_sync_runtime_start_task = asyncio.create_task(
            self.notes_sync_runtime_owner.start(),
            name="start_notes_sync_runtime",
        )
        self._notes_sync_runtime_start_task.add_done_callback(
            self._observe_notes_sync_runtime_start
        )
        mount_start = time.perf_counter()

        # task-19561: hand the process-level SIGTERM/SIGINT handler this app
        # and its running loop, so a termination signal becomes an ordinary
        # `App.exit()` instead of an `os._exit(0)` through the middle of
        # whatever was writing at the time.
        register_running_app(self)

        # TASK-1240. Anchors a session in the persistent log; its absence dates
        # a crash to before this point. Wrapped: `persist_event` raises on a
        # malformed component and its sink can fail; diagnostics must never be
        # the reason mount does not complete.
        try:
            persist_event(_DIAGNOSTICS_COMPONENT_APP, "app_started")
        except Exception:
            pass

        # Which interpreter's speech stack this run actually has. Dictation
        # degrades silently and differently per missing package -- without
        # `webrtcvad` no segment can finalize mid-capture (so nothing appears
        # until stop and no voice command can fire), and without the
        # configured provider's package the resolver quietly picks another.
        # Both were diagnosed only after several live rounds because the run
        # left no record of its own environment (2026-08-01); one line here
        # dates every future report to a specific interpreter.
        try:
            from importlib.util import find_spec

            from .Chat.console_voice_input import resolve as _resolve_dictation

            # The provider dictation would actually use, not merely one that
            # is installed: the resolver's config precedence is exactly what
            # went wrong before, so recording its answer is the point.
            _effective = _resolve_dictation()
            persist_event(
                "dictation",
                "speech_stack_available",
                status="ok" if find_spec("webrtcvad") is not None else "degraded",
                provider=_effective.provider if _effective else "none",
                model=(_effective.model if _effective else None) or "provider-default",
            )
        except Exception:
            pass

        # Restore persisted Library ingest job history (self.library_ingest_jobs
        # already exists -- constructed store-less in __init__). Never raises:
        # a corrupt/unreadable store falls back to starting empty.
        self._restore_ingest_jobs()

        # Update splash screen progress only if splash screen is active
        if self.splash_screen_active and self._splash_screen_widget:
            try:
                self._splash_screen_widget.update_progress(0.3, "Setting up logging...")
            except Exception as e:
                self.loguru_logger.warning(
                    f"Failed to update splash screen progress: {e}"
                )

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
            log_histogram(
                "app_on_mount_phase_duration_seconds",
                time.perf_counter() - logging_start,
                labels={"phase": "logging_setup"},
                documentation="Duration of on_mount phase in seconds",
            )
        else:
            self.loguru_logger.debug(
                "Deferring logging setup until after splash screen closes"
            )

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
                    self._splash_screen_widget.update_progress(
                        0.5,
                        f"Loading user interface...{splashscreen_message_selection}",
                    )
                except Exception as e:
                    self.loguru_logger.warning(
                        f"Failed to update splash screen progress: {e}"
                    )

        # Only schedule post-mount setup if splash screen is not active
        if not self.splash_screen_active:
            # Schedule setup to run after initial rendering.
            # task-19561: this was a bare `create_task` whose result nobody
            # held. The event loop keeps only a weak reference to a task, so
            # the whole no-splash startup path could be garbage-collected
            # mid-flight. `_create_deferred_startup_task` keeps the strong
            # reference AND puts it in the set shutdown already cancels.
            self._create_deferred_startup_task(
                self._run_no_splash_post_mount_setup(),
                name="no_splash_post_mount_setup",
            )

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
            self.loguru_logger.warning(
                f"Failed to apply default theme '{default_theme}', falling back to 'textual-dark': {e}"
            )
            self.theme = "textual-dark"

        log_histogram(
            "app_on_mount_phase_duration_seconds",
            time.perf_counter() - theme_start,
            labels={"phase": "theme_registration"},
            documentation="Duration of on_mount phase in seconds",
        )

        mount_duration = time.perf_counter() - mount_start
        log_histogram(
            "app_on_mount_duration_seconds",
            mount_duration,
            documentation="Total time for on_mount() method",
        )
        self.loguru_logger.info(f"on_mount completed in {mount_duration:.3f} seconds")

        # Start the background scheduler loop for reminders and scheduled tasks.
        # A COROUTINE worker, never thread=True: scheduled watchlist checks
        # dispatch from this loop, and the watchlists in-flight guard
        # (`local_watchlists_service._IN_FLIGHT_URL_CHECKS`) is lock-free on
        # the invariant that every check entrant runs on the app's one event
        # loop. Moving dispatch off-loop needs a lock there.
        self.scheduler_worker = self.run_worker(
            self.scheduler_loop.run(),
            exclusive=True,
            group="scheduling",
        )

        # ADR-020: setup owns startup networking until it completes, so a
        # stale configured cloud provider cannot be contacted behind it.
        self._schedule_startup_model_catalog_refresh()

        # task-688: index subscription_items rows scraped before the FTS5
        # index existed, so search covers a user's whole back catalogue
        # without any action on their part. thread=True because this does
        # blocking sqlite work; never blocks startup or screen mount since
        # run_worker only schedules it.
        self.run_worker(
            self._backfill_subscription_items_fts,
            thread=True,
            exclusive=True,
            group="subscriptions-fts-backfill",
        )

        # task-21100: reinsert the messages the v45->v46 FTS reset no longer
        # indexes inline, so an upgraded profile's chat history becomes fully
        # searchable again without ever blocking boot on the index rewrite.
        # thread=True for the same reason as the subscriptions backfill above.
        self.run_worker(
            self._backfill_chachanotes_messages_fts,
            thread=True,
            exclusive=True,
            group="chachanotes-fts-backfill",
        )

    def _init_model_catalog_disk_store(self) -> "ModelCatalogDiskStore | None":
        """Build the disk-backed model catalog cache for startup (ADR-020).

        Returns None (with a log line) when the cache path cannot be resolved,
        fails validation against the user data dir, or the on-disk cache cannot
        be loaded; startup continues without persistence in those cases.
        """
        from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
            ModelCatalogDiskStore,
        )
        from tldw_chatbook.Utils.path_validation import get_safe_relative_path

        try:
            user_data_dir = get_user_data_dir()
            cache_path = user_data_dir / "model_catalog_cache.json"
        except Exception as exc:
            logger.error(
                f"Failed to resolve model catalog cache path: {type(exc).__name__}"
            )
            return None
        # get_safe_relative_path (not is_safe_path): the default data dir lives
        # under ~/.local, which validate_path's hidden-component rule rejects.
        if get_safe_relative_path(cache_path, user_data_dir) is None:
            logger.warning(
                f"Ignoring model catalog cache outside the user data dir: {cache_path}"
            )
            return None
        try:
            store = ModelCatalogDiskStore(cache_path)
            store.load_into(self.local_llm_provider_catalog_service.discovery_cache)
        except Exception as exc:
            # No traceback: the log file sink runs with diagnose=True, which
            # would dump frame locals (including the app's config) into the log.
            logger.error(
                f"Failed to load model catalog disk cache {cache_path}: "
                f"{type(exc).__name__}"
            )
            return None
        return store

    async def _refresh_model_catalogs(self) -> None:
        """ADR-020 startup auto-refresh; never blocks or crashes startup."""
        try:
            from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
                format_refresh_notification,
            )
            from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
                AUTO_REFRESH_PROVIDER_LIST_KEYS,
                load_model_catalog_settings,
            )

            if self.model_catalog_disk_store is None:
                return
            catalog_settings = load_model_catalog_settings(load_settings())
            if not catalog_settings.auto_refresh_enabled:
                return
            if not catalog_settings.refresh_consent_recorded:
                # ADR-020 amendment: the startup check is confirm-first.
                # Scheduling-side consent gate normally intercepts this
                # before the worker spawns; keep the check here so the
                # refresh never runs unconsented by any other path.
                return
            report = await self.local_llm_provider_catalog_service.refresh_stale_configured_providers(
                catalog_settings=catalog_settings,
                disk_store=self.model_catalog_disk_store,
                on_config_saved=self._init_providers_models,
            )
            refreshed = {
                outcome.provider_list_key
                for outcome in report.outcomes
                if outcome.status in {"refreshed", "baseline"}
            }
            if refreshed:
                self.post_message(ModelCatalogRefreshed(providers=refreshed))
            message = format_refresh_notification(report)
            if message:
                has_failure = report.disk_write_failed or any(
                    outcome.status == "failed" or outcome.write_failed
                    for outcome in report.outcomes
                )
                self.notify(
                    message,
                    title="Model catalog",
                    severity="warning" if has_failure else "information",
                )
        except Exception as exc:
            # No traceback: the log file sink runs with diagnose=True, which
            # would dump frame locals (potentially API keys) into the log file.
            logger.error(
                "Model catalog auto-refresh failed "
                f"({', '.join(AUTO_REFRESH_PROVIDER_LIST_KEYS)}): "
                f"{type(exc).__name__}"
            )

    def _schedule_startup_model_catalog_refresh(
        self,
        *,
        after_setup_completion: bool = False,
        environ: Mapping[str, str] | None = None,
    ) -> bool:
        """Schedule the automatic catalog pass once when setup releases it.

        ADR-020 amendment (confirm-first): when the user has never answered
        the consent question, a modal is shown instead of the refresh; the
        refresh itself is only scheduled from the consent callback.
        """
        if getattr(self, "_startup_model_catalog_refresh_scheduled", False):
            return False
        if not after_setup_completion and setup_owns_startup_networking(
            self.app_config,
            os.environ if environ is None else environ,
        ):
            return False

        try:
            from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
                load_model_catalog_settings,
            )

            catalog_settings = load_model_catalog_settings(load_settings())
        except Exception as exc:
            logger.error(
                "Failed to load model catalog settings for startup refresh "
                f"scheduling (after_setup_completion={after_setup_completion}): "
                f"{type(exc).__name__}"
            )
            return False
        if catalog_settings.auto_refresh_enabled and (
            not catalog_settings.refresh_consent_recorded
        ):
            self._startup_model_catalog_refresh_scheduled = True
            self.call_after_refresh(self._push_model_catalog_consent_modal)
            return True

        self._startup_model_catalog_refresh_scheduled = True
        self.run_worker(
            self._refresh_model_catalogs,
            exclusive=True,
            group="model-catalog-refresh",
        )
        return True

    def _push_model_catalog_consent_modal(self) -> None:
        """Show the one-time consent dialog for online model-list checks."""
        if self.is_headless:
            # Headless/embedded runs have no user to answer a modal; stay
            # unconsented (no refresh) rather than blocking startup behind
            # an unanswerable dialog.
            return
        try:
            from tldw_chatbook.UI.Screens.model_catalog_consent import (
                ModelCatalogConsentModal,
            )
        except Exception as exc:
            logger.error(
                "Failed to import the model catalog consent modal "
                f"(screen=model_catalog_consent): {type(exc).__name__}"
            )
            return
        self.push_screen(ModelCatalogConsentModal(), self._handle_model_catalog_consent)

    async def _handle_model_catalog_consent(self, allowed: bool | None) -> None:
        """Persist the consent answer; on allow, run the startup refresh."""
        # Only the boolean singleton True counts as consent — truthy garbage
        # (e.g. a non-bool reaching this callback) falls through to the deny
        # path, mirroring the settings parser's strict validator.
        allowed = allowed is True
        try:
            from tldw_chatbook.config import save_settings_to_cli_config

            section = {"refresh_consent_recorded": True}
            if not allowed:
                section["auto_refresh_enabled"] = False
            saved = await asyncio.to_thread(
                save_settings_to_cli_config, {"model_catalog": section}
            )
        except Exception as exc:
            # No traceback: the log file sink runs with diagnose=True, which
            # would dump frame locals (including the app's config) into the log.
            logger.error(
                "Failed to persist model catalog consent "
                f"(allowed={allowed!r}, section=model_catalog): "
                f"{type(exc).__name__}"
            )
            saved = False
        if allowed:
            if not saved:
                self.notify(
                    "Your choice couldn't be saved; you'll be asked again next launch.",
                    title="Model catalog",
                    severity="warning",
                )
            self.run_worker(
                self._refresh_model_catalogs,
                exclusive=True,
                group="model-catalog-refresh",
            )
        else:
            self.notify(
                "Online model-list checks stay off. You can enable them any "
                "time in Settings.",
                title="Model catalog",
            )

    @on(ModelCatalogRefreshed)
    async def on_model_catalog_refreshed(self, event: ModelCatalogRefreshed) -> None:
        # Textual delivers App-posted messages to App handlers only; forward
        # down to a mounted screen that exposes a refresh handler.
        from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import (
            forward_model_catalog_refreshed,
        )

        await forward_model_catalog_refreshed(self, event)

    def _maybe_offer_first_run_wizard(self) -> bool:
        """Offer the setup wizard once; otherwise nudge unfinished setups.

        Returns:
            True iff the wizard OR the recovery dialog was pushed this
            launch (the ``"offer"`` and ``"prompt"`` branches); False for
            every other outcome (already scheduled, the resume-toast
            "none" branch -- which still runs and notifies -- or a caught
            exception). Callers use this to decide whether a lower-
            priority startup offer (e.g. the project-.SKILLS import
            prompt, spec 2026-08-17 §5.4) should defer to next launch
            instead of competing with the wizard OR the recovery dialog
            for the user's attention -- both branches push a screen onto
            the stack, so both must suppress the lower-priority offer
            (final review 2026-08-17, Finding 3: the "prompt" branch used
            to return False here, letting the skills-import modal stack
            on top of a just-pushed ``SetupRecoveryDialog``).
        """
        if getattr(self, "_first_run_startup_action_scheduled", False):
            return False
        try:
            from tldw_chatbook.UI.Wizards.first_run_setup_state import (
                setup_recovery_action,
                should_show_resume_toast,
            )

            action = setup_recovery_action(self.app_config, os.environ)
            if action == "offer":
                self._first_run_startup_action_scheduled = True
                self.call_after_refresh(self._push_first_run_wizard)
                return True
            elif action == "prompt":
                self._first_run_startup_action_scheduled = True
                self.call_after_refresh(self._push_first_run_recovery_dialog)
                return True
            elif action == "none" and should_show_resume_toast(
                self.app_config, os.environ
            ):
                self.notify(
                    "Setup isn't finished — run it any time from "
                    "Settings ▸ Diagnostics ▸ Run setup wizard.",
                    title="Finish setup",
                    severity="information",
                    timeout=8,
                )
        except Exception as exc:
            logger.error(
                "First-run startup action failed (error_type={})",
                type(exc).__name__,
            )
        return False

    def _maybe_warn_config_load_failure(self) -> None:
        """Warn (never block) when boot's config load fell back to defaults.

        TASK-13157: a config.toml that fails to parse was previously a
        completely silent failure -- `load_settings()`/`load_cli_config_and_
        ensure_existence()` both return bare in-memory defaults with no
        signal, which a live-verification incident showed can silently
        resolve the data directory to the `default_user` profile instead of
        the configured one, with no error, toast, or log line a normal user
        would ever see. `self._config_load_failure` was snapshotted in
        `__init__` (before the UI existed to notify through); this surfaces
        it once the initial screen is up, naming the exact file and parse
        error so the user knows their saved settings are NOT the ones
        currently in effect. `timeout=None` only reaches Textual's own
        5-second default (`App.NOTIFICATION_TIMEOUT`), not "persistent", so
        this passes an explicit long timeout instead -- this is not a
        transient event and must not be missed the way the silent fallback
        it replaces was.
        """
        failure = getattr(self, "_config_load_failure", None)
        if failure is None:
            return
        self.notify(
            f"Your configuration file could not be parsed and was NOT used "
            f"this session -- running on built-in defaults instead (this may "
            f"include the wrong user profile). File: {failure.path}  "
            f"Error: {failure.message}",
            title="Config file failed to load",
            severity="error",
            timeout=60,
        )

    def _maybe_warn_second_instance(self) -> None:
        """Warn (never block) when another instance already holds this profile.

        RAG-53 (task-7): several stores (AgentRuns reconcile sweeps, library
        ingest restart sweeps, MCP permission store) are last-write-wins /
        accepted-but-unwarned under concurrent instances by design -- the
        owner runs concurrent instances deliberately. This is a one-time
        advisory toast, never a lock-out.
        """
        status = getattr(self, "_instance_lock_status", None)
        if status is None or status.acquired:
            return
        detail = ""
        if status.holder_pid:
            detail = f" (pid {status.holder_pid})"
        self.notify(
            "Another copy of tldw is already using this profile"
            f"{detail}. Everything keeps working, but the last instance to "
            "change settings or permissions wins, and a restart sweep may mark "
            "the other instance's running jobs as interrupted.",
            title="Profile already open",
            severity="warning",
            timeout=10,
        )

    def _push_first_run_wizard(self) -> None:
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard

        self.push_screen(
            FirstRunSetupWizard(self), self._handle_first_run_wizard_result
        )

    def _maybe_offer_project_skills_import(self) -> None:
        """Offer to import a project's .SKILLS/ folder (spec 2026-08-17 §5.4).

        ``exit_on_error=False`` matches the repo's own precedent for an
        optional, best-effort worker (``action_quit``'s ``_confirm_and_quit``,
        the screen-navigation dispatch worker): Textual's default
        (``exit_on_error=True``) makes ANY unhandled exception in the worker
        exit the whole app, which an optional startup nicety must never do.
        The worker body below additionally never lets an exception reach the
        worker at all -- this is belt AND suspenders.
        """
        try:
            self.run_worker(
                self._discover_project_skills_for_startup,
                thread=True,
                exclusive=True,
                group="project-skills-discovery",
                exit_on_error=False,
            )
        except Exception:
            logger.opt(exception=True).debug("project-skills startup offer failed")

    def _discover_project_skills_for_startup(self) -> None:
        """Worker body: every line here must be exception-safe.

        This runs on a worker thread with ``exit_on_error=False`` set above,
        but that alone still leaves an unhandled exception logged as a
        worker error and the offer silently dropped with a stack trace in
        the logs -- an entirely optional startup nicety earns a clean,
        quiet no-op instead. ``get_cli_setting``/``get_user_data_dir`` (I/O,
        config parsing), ``startup_discovery_for`` (filesystem walk), and
        ``call_from_thread`` (can raise if the app is already shutting down
        mid-walk) are all covered by the one try/except below.
        """
        try:
            from tldw_chatbook.config import get_cli_setting, get_user_data_dir
            from tldw_chatbook.Skills_Interop.project_skills_prompt import (
                startup_discovery_for,
            )

            try:
                cwd = Path.cwd().resolve()
            except OSError:
                return  # launch directory deleted out from under the process
            discovery = startup_discovery_for(
                cwd,
                enabled=bool(
                    get_cli_setting("skills", "project_skills_prompt_enabled", True)
                ),
                ledger_dir=get_user_data_dir(),
            )
            if discovery is None:
                return
            self.call_from_thread(self._push_project_skills_import_modal, discovery)
        except Exception:
            logger.opt(exception=True).debug("project-skills startup discovery failed")

    def _push_project_skills_import_modal(self, discovery) -> None:
        from tldw_chatbook.Widgets.project_skills_import_modal import (
            maybe_offer_project_skills_import,
        )

        maybe_offer_project_skills_import(self, (discovery,))

    def _push_first_run_recovery_dialog(self) -> None:
        from tldw_chatbook.UI.Wizards.first_run_recovery_dialog import (
            SetupRecoveryDialog,
        )

        self.push_screen(SetupRecoveryDialog(), self._handle_first_run_recovery_result)

    def _handle_first_run_recovery_result(self, result: str | None) -> None:
        if result not in {"resume", "start_over", "later"}:
            return
        self.run_worker(
            self._apply_first_run_recovery_result(result),
            exclusive=True,
            group="first-run-recovery",
        )

    async def _apply_first_run_recovery_result(self, result: str) -> None:
        if result == "later":
            return

        from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
        from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import FirstRunSetupWizard
        from tldw_chatbook.config import save_settings_to_cli_config

        resume_draft = None
        if result == "resume":
            draft = wizard_state.read_setup_draft(self.app_config)
            if draft is None or draft.resume_attempted:
                return
            resume_draft = wizard_state.SetupDraft(
                version=draft.version,
                track=draft.track,
                active_step_id=draft.active_step_id,
                values=draft.values,
                resume_attempted=True,
            )
            settings, delete_keys = wizard_state.build_setup_draft_mutation(
                resume_draft
            )
        elif result == "start_over":
            settings, delete_keys = wizard_state.build_setup_draft_mutation(None)
        else:
            return

        try:
            if delete_keys:
                saved = await asyncio.to_thread(
                    save_settings_to_cli_config,
                    settings,
                    delete_keys=delete_keys,
                )
            else:
                saved = await asyncio.to_thread(save_settings_to_cli_config, settings)
        except Exception as exc:
            logger.error(
                "First-run recovery persistence failed (error_type={})",
                type(exc).__name__,
            )
            saved = False
        if not saved:
            self.notify(
                "Setup recovery could not be saved. Try again.",
                severity="error",
            )
            self._schedule_first_run_recovery_retry()
            return

        self._mirror_first_run_setup_mutation(settings, delete_keys)
        self.push_screen(
            FirstRunSetupWizard(self, resume_draft=resume_draft),
            self._handle_first_run_wizard_result,
        )

    def _schedule_first_run_recovery_retry(self) -> None:
        """Reopen one actionable recovery prompt after a failed mutation."""

        if getattr(self, "_first_run_recovery_retry_scheduled", False):
            return
        self._first_run_recovery_retry_scheduled = True
        self._first_run_startup_action_scheduled = False
        self.call_after_refresh(self._show_first_run_recovery_retry)

    def _show_first_run_recovery_retry(self) -> None:
        if not getattr(self, "_first_run_recovery_retry_scheduled", False):
            return
        self._first_run_recovery_retry_scheduled = False
        self._first_run_startup_action_scheduled = True
        current_screen = type(self.screen).__name__
        if current_screen in {"SetupRecoveryDialog", "FirstRunSetupWizard"}:
            return
        self._push_first_run_recovery_dialog()

    def _mirror_first_run_setup_mutation(
        self,
        settings: Mapping[str, Mapping[str, object]],
        delete_keys: Mapping[str, tuple[str, ...]],
    ) -> None:
        """Mirror the exact first-run recovery mutation after a successful write."""

        first_run = self.app_config.setdefault("first_run", {})
        if not isinstance(first_run, dict):
            first_run = {}
            self.app_config["first_run"] = first_run
        values = settings.get("first_run")
        if isinstance(values, Mapping):
            first_run.update(values)
        for key in delete_keys.get("first_run", ()):
            first_run.pop(key, None)

    def _handle_first_run_wizard_result(self, result: dict | None) -> None:
        if type(result) is not dict:
            return  # cancelled / finish-later: recovery state handles next launch
        exit_route = result.get("exit_route")
        completed = result.get("completed")
        exit_context = result.get("exit_context")
        if exit_route is None:
            if completed is not True or exit_context is not None:
                return
            self._schedule_startup_model_catalog_refresh(after_setup_completion=True)
            return
        if type(exit_route) is not str:
            return

        # task-18812: consume a deferred focus request from a first-run
        # launch (--focus / focus_mode config) at the moment the wizard
        # finishes, BEFORE payload validation -- the request's fate must
        # not depend on how valid the wizard's result dict is. Focus is
        # Console-only: it applies when the exit route is Chat, and is
        # simply dropped for any other destination.
        if getattr(self, "_deferred_focus_request", False):
            self._deferred_focus_request = False
            if exit_route == TAB_CHAT:
                self.focus_mode = True
            else:
                self.focus_mode = False

        screen_context: dict[str, object] = {}
        if exit_route == TAB_SETTINGS:
            if completed is not False or type(exit_context) is not dict:
                return
            if set(exit_context) != {"category"}:
                return
            category = exit_context.get("category")
            if type(category) is not str:
                return
            from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
                REQUIRED_STEP_MANUAL_SETTINGS_CATEGORIES,
            )

            if category not in set(REQUIRED_STEP_MANUAL_SETTINGS_CATEGORIES.values()):
                return
            screen_context = {"category": category}
        elif exit_route in {TAB_CHAT, TAB_HOME}:
            if completed is not True:
                return
            if exit_context is not None and (
                type(exit_context) is not dict or exit_context
            ):
                return
        else:
            return

        if completed is True:
            self._schedule_startup_model_catalog_refresh(after_setup_completion=True)

        # Dismissing a rerun over Console already uncovers that same mounted
        # Console. Replacing it here would interrupt first-chat rollback and
        # focus resync. Other destinations still remount to refresh their state.
        if (
            not screen_context
            and exit_route == TAB_CHAT
            and getattr(self, "current_tab", None) == TAB_CHAT
        ):
            # The already-mounted Console kept chrome from its unfocused
            # mount; apply the restored request in place.
            if self.focus_mode:
                apply_chrome = getattr(
                    self._navigation_outgoing_screen(), "_apply_focus_chrome", None
                )
                if callable(apply_chrome):
                    apply_chrome()
            return

        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

        self.post_message(NavigateToScreen(exit_route, screen_context))

    def handle_first_run_wizard_result(self, result: dict | None) -> None:
        """Public alias for ``_handle_first_run_wizard_result``.

        The wizard's re-entry points outside this module -- Settings'
        "Run setup wizard" button and the command-palette provider below --
        need a non-private way to wire this callback into their own
        ``push_screen(FirstRunSetupWizard(...), ...)`` calls, so a truthy
        exit_route from the Summary step still navigates on re-run instead
        of silently being dropped (the auto-offer path already wires
        ``_push_first_run_wizard`` with this same handler).
        """
        self._handle_first_run_wizard_result(result)

    def hide_inactive_windows(self) -> None:
        """Hides all windows that are not the current active tab."""
        initial_tab = self._initial_tab_value
        self.loguru_logger.debug(
            f"Hiding inactive windows, keeping '{initial_tab}-window' visible."
        )
        # Query both actual windows and placeholders
        for window in self.query(".window, .placeholder-window"):
            # Placeholders should always be hidden
            if window.has_class("placeholder-window"):
                window.display = False
                continue
            is_active = window.id == f"{initial_tab}-window"
            window.display = is_active

    async def _push_initial_screen(self) -> None:
        """Push the configured initial screen for screen-based navigation startup."""
        if getattr(self, "_initial_screen_pushed", False):
            return

        initial_tab = self._resolve_initial_shell_route()
        resolved_screen_name, resolved_tab, screen_class = (
            self._resolve_screen_navigation_target(initial_tab)
        )
        if screen_class is None:
            # Report why the configured target failed before falling back --
            # otherwise a broken screen silently redirects to chat forever.
            logger.warning(
                f"Screen navigation: initial target {initial_tab!r} did not resolve"
                f" ({screen_load_error(initial_tab)}); falling back to {TAB_CHAT!r}"
            )
            resolved_screen_name = TAB_CHAT
            resolved_tab = TAB_CHAT
            _, _, screen_class = self._resolve_screen_navigation_target(TAB_CHAT)
            if screen_class is None:
                # Fatal: no screen to show. `resolve_screen_target()` degrades
                # a failed route to None by design, so surface the underlying
                # cause here -- a bare "unable to resolve" names neither the
                # missing dependency nor the module that pulled it in.
                cause = screen_load_error(TAB_CHAT)
                message = f"Unable to resolve default chat screen ({TAB_CHAT!r})"
                if cause is not None:
                    message += f": {type(cause).__name__}: {cause}"
                raise RuntimeError(message) from cause

        new_screen = screen_class(self)

        # A configured default tab that is itself a legacy alias route (e.g.
        # "search"/"prompts"/"skills" -> Library) carries the same nav-context
        # promise on boot as it does when navigated to in-app -- otherwise
        # `default_tab = "search"` silently degrades to generic Library
        # instead of the Search/RAG canvas the alias promises. The table is
        # keyed on the PRE-resolution route id, so `initial_tab` (captured
        # above, before `_resolve_screen_navigation_target` rewrote it) is
        # the correct lookup key. Mirrors the guarded apply in
        # `handle_screen_navigation` (~:6672-6687); the screen is always
        # unmounted here, so `apply_navigation_context` takes its sync path.
        navigation_context = self._LEGACY_ROUTE_LIBRARY_NAV_CONTEXT.get(initial_tab, {})
        if navigation_context and hasattr(new_screen, "apply_navigation_context"):
            try:
                result = new_screen.apply_navigation_context(navigation_context)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                logger.warning(
                    "Initial navigation context application failed "
                    "(route=%s, exception_category=%s).",
                    initial_tab,
                    type(exc).__name__,
                )

        await self.push_screen(new_screen)
        self.current_tab = resolved_tab
        self._initial_screen_pushed = True
        logger.info(
            f"Screen navigation: Pushed initial {screen_class.__name__}"
            f" (target={resolved_screen_name})"
        )
        wizard_offered = self._maybe_offer_first_run_wizard()
        try:
            self._maybe_warn_second_instance()
        except Exception as e:
            logger.error(f"Second-instance warning failed: {e}")
        if not wizard_offered:
            # Spec 2026-08-17 §5.4: wizard wins; .SKILLS offer defers to next launch.
            self._maybe_offer_project_skills_import()
        try:
            self._maybe_warn_config_load_failure()
        except Exception as e:
            logger.error(
                "Config load failure warning failed (error_type=%s)",
                type(e).__name__,
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
        self.loguru_logger.info(
            "App _post_mount_setup: Binding Select widgets and populating dynamic content..."
        )

        # Update splash screen progress (defensive check - shouldn't happen if splash was shown)
        if self.splash_screen_active and self._splash_screen_widget:
            try:
                self._splash_screen_widget.update_progress(
                    0.7, "Configuring providers..."
                )
            except Exception as e:
                self.loguru_logger.warning(
                    f"Failed to update splash screen progress: {e}"
                )

        # Removed populate_llm_help_texts from here - it's called when LLM tab is shown instead
        phase_start = time.perf_counter()
        # LLM help texts are populated when the LLM tab is shown
        log_histogram(
            "app_post_mount_phase_duration_seconds",
            time.perf_counter() - phase_start,
            labels={"phase": "llm_help_texts_skipped"},
            documentation="Duration of post-mount phase in seconds",
        )

        # Widget binding
        phase_start = time.perf_counter()
        log_histogram(
            "app_post_mount_phase_duration_seconds",
            time.perf_counter() - phase_start,
            labels={"phase": "widget_binding"},
            documentation="Duration of post-mount phase in seconds",
        )

        # TTS/STTS services are initialized after readiness or on first use.
        log_histogram(
            "app_post_mount_phase_duration_seconds",
            0.0,
            labels={"phase": "audio_services_deferred"},
            documentation="Duration of post-mount phase in seconds",
        )

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
        log_histogram(
            "app_post_mount_phase_duration_seconds",
            time.perf_counter() - phase_start,
            labels={"phase": "populate_lists"},
            documentation="Duration of post-mount phase in seconds",
        )

        post_mount_duration = time.perf_counter() - post_mount_start
        log_histogram(
            "app_post_mount_duration_seconds",
            post_mount_duration,
            documentation="Total time for _post_mount_setup() method",
        )
        self.loguru_logger.info(
            f"_post_mount_setup completed in {post_mount_duration:.3f} seconds"
        )

        # Log final resource usage
        log_resource_usage()

        # Update splash screen progress to completion (defensive check)
        if self.splash_screen_active and self._splash_screen_widget:
            try:
                self._splash_screen_widget.update_progress(1.0, "Ready!")
            except Exception as e:
                self.loguru_logger.warning(
                    f"Failed to update splash screen progress: {e}"
                )

        # Footer status population is scheduled after readiness so DB-size
        # polling cannot hold the first interactive frame.

        # CRITICAL: Set UI ready state after all bindings and initializations
        self._ui_ready = True
        ui_ready_time = time.perf_counter()

        self.loguru_logger.info("App _post_mount_setup: Post-mount setup completed.")

        # Log UI loading metrics
        if hasattr(self, "_ui_compose_start_time"):
            ui_loading_time = ui_ready_time - self._ui_compose_start_time
            log_histogram(
                "ui_loading_duration_seconds",
                ui_loading_time,
                documentation="Total time from compose start to UI ready",
            )
            log_counter(
                "ui_loading_complete",
                1,
                documentation="UI loading completed successfully",
            )
            self.loguru_logger.info(
                f"UI loading completed in {ui_loading_time:.3f} seconds"
            )

        # Log post-mount setup duration
        post_mount_duration = ui_ready_time - post_mount_start
        log_histogram(
            "app_post_mount_total_duration_seconds",
            post_mount_duration,
            documentation="Total time for post-mount setup",
        )

        # Log total startup time (from __init__ start to fully ready)
        if hasattr(self, "_startup_start_time"):
            total_startup_time = ui_ready_time - self._startup_start_time
            log_histogram(
                "app_startup_complete_duration_seconds",
                total_startup_time,
                documentation="Total time from app initialization start to fully ready",
            )
            log_counter(
                "app_startup_complete",
                1,
                documentation="Application startup completed successfully",
            )

            # Log breakdown of startup phases
            backend_init_time = (
                self._ui_compose_start_time - self._startup_start_time
                if hasattr(self, "_ui_compose_start_time")
                else 0
            )
            ui_compose_time = (
                getattr(self, "_ui_compose_end_time", ui_ready_time)
                - self._ui_compose_start_time
                if hasattr(self, "_ui_compose_start_time")
                else 0
            )

            log_histogram(
                "app_startup_breakdown_seconds",
                backend_init_time,
                labels={"phase": "backend_initialization"},
                documentation="Breakdown of application startup phases",
            )
            log_histogram(
                "app_startup_breakdown_seconds",
                ui_compose_time,
                labels={"phase": "ui_composition"},
                documentation="Breakdown of application startup phases",
            )
            log_histogram(
                "app_startup_breakdown_seconds",
                post_mount_duration,
                labels={"phase": "post_mount_setup"},
                documentation="Breakdown of application startup phases",
            )

            self.loguru_logger.info("=== APPLICATION STARTUP COMPLETE ===")
            self.loguru_logger.info(
                f"Total startup time: {total_startup_time:.3f} seconds"
            )
            self.loguru_logger.info(f"  - Backend init: {backend_init_time:.3f}s")
            self.loguru_logger.info(f"  - UI composition: {ui_compose_time:.3f}s")
            self.loguru_logger.info(f"  - Post-mount setup: {post_mount_duration:.3f}s")
            self.loguru_logger.info("===================================")

            # Final memory usage
            log_resource_usage()

        self._schedule_deferred_startup_work()

    async def update_db_sizes(self) -> None:
        """Updates the database size information in the shell status line."""
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

        # task-21106: Actor Pack crash recovery moved here from __init__ —
        # synchronous SQLite has no place on the construction path. A thread
        # worker (not a coroutine) because recovery does blocking DB I/O; the
        # coordinator's own once-guard makes every later surface-side call
        # (Personas mount, create_persona) a cached no-op.
        self.run_worker(
            self.ensure_actor_pack_recovery,
            name="deferred_actor_pack_recovery",
            group="actor_pack_recovery",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )
        self.set_timer(
            DEFERRED_DB_SIZE_UPDATE_DELAY_SECONDS,
            self._schedule_footer_status_updates,
        )
        self.set_timer(
            DEFERRED_AUDIO_SERVICE_DELAY_SECONDS,
            self._start_deferred_audio_service_initialization,
        )
        self.set_timer(
            DEFERRED_SCREEN_PREIMPORT_DELAY_SECONDS,
            self._schedule_screen_preimport,
        )
        self.schedule_media_cleanup()
        self._create_deferred_startup_task(
            self._reconcile_interrupted_subscription_work(),
            name="deferred_subscription_interrupt_reconcile",
        )
        coordinator = getattr(
            self,
            "citation_artifact_ownership_coordinator",
            None,
        )
        if coordinator is not None and coordinator.writes_enabled:
            self._create_deferred_startup_task(
                self._reconcile_citation_artifact_ownership(),
                name="deferred_citation_artifact_reconciliation",
            )
        migration = getattr(
            self,
            "citation_legacy_migration_service",
            None,
        )
        if migration is not None and migration.ready:
            self._create_deferred_startup_task(
                self._migrate_legacy_citations_idle_unit(),
                name="deferred_legacy_citation_migration",
            )
        self._schedule_launch_wake()

    def _schedule_launch_wake(self) -> None:
        """Deliver a supervisor wake this install already owed at launch.

        task-15860 Task 6. A background sub-agent that finished while the
        app was closed -- or one whose delivery the user quit out from
        under -- used to wait for the next Console visit. It no longer
        does, under the owner's mark-gated ruling: only a conversation
        that already carries a durable ``FLEET_UNSEEN`` mark AND an owed
        ``agent_runs`` row is delivered, behind the existing ``[agents]
        autowake_enabled`` (there is no separate launch switch).

        **The common path costs one indexed read and constructs nothing.**
        With no marks -- every install that has never run a background
        sub-agent, and every one whose results have all been seen -- this
        returns before touching the Console store, provider gateway, agent
        bridge (so ``agent_runs.db`` is not even opened) or controller.
        That is pinned in ``Tests/UI/test_console_launch_wake.py``.
        """
        try:
            from tldw_chatbook.Chat.console_launch_wake import (
                LAUNCH_WAKE_TASK_NAME,
                deliver_launch_wakes,
                marked_conversations_at_launch,
            )

            marked = marked_conversations_at_launch(self)
            if not marked:
                return
            self._create_deferred_startup_task(
                deliver_launch_wakes(self, marked),
                name=LAUNCH_WAKE_TASK_NAME,
            )
        except Exception:  # noqa: BLE001 -- a launch never dies on this
            logger.opt(exception=True).warning(
                "Launch wake scheduling failed; owed wakes stay staged for "
                "the next Console visit."
            )

    async def _reconcile_interrupted_subscription_work(self) -> None:
        """Un-wedge subscriptions rows a previous process never finished.

        task-19561. ``local_watchlist_runs`` (``queued``/``running``),
        ``briefings``/``briefing_scripts``/``briefing_audio``
        (``generating``) all carry a status only the process doing the work
        can move off, and several of them double as one-at-a-time guards --
        so a row stranded by a termination does not merely look wrong, it
        shuts the feature. Until now the only sweep was UI-gated: it ran
        when the user happened to open the matching Watchlists pane, scoped
        to that one watchlist.

        Doing it on the way *in* is what makes it durable. A reconcile on
        the way out can only cover terminations the process survives long
        enough to run it -- never ``SIGKILL``, a crash, or a battery going
        flat, which are exactly the cases that strand a row. Runs on a
        thread (SQLite) and is best-effort: a failed sweep is logged and the
        launch continues.

        Scoped by the boundary ``_wire_watchlists_and_notifications_services``
        captured when it opened the database, so this cannot fail a row the
        scheduler -- started earlier, in ``on_mount`` -- launched moments ago
        (Qodo review of PR #1972). No boundary means no sweep: leaving a row
        wedged is recoverable on the next launch, failing a live one is not.
        """
        db = getattr(self, "subscriptions_db", None)
        if db is None:
            return
        boundary = getattr(self, "_subscriptions_prior_process_boundary", None)
        if boundary is None:
            self.loguru_logger.warning(
                "Startup reconcile skipped: no prior-process boundary was "
                "captured, so an unscoped sweep could fail live rows."
            )
            return
        from tldw_chatbook.Subscriptions.startup_reconcile import (
            reconcile_interrupted_subscription_work,
        )

        try:
            reconciled = await asyncio.to_thread(
                reconcile_interrupted_subscription_work, db, boundary
            )
        except Exception as exc:  # noqa: BLE001 - a launch never dies on this
            self.loguru_logger.warning(
                f"Startup reconcile of interrupted subscriptions work failed "
                f"type={type(exc).__name__}"
            )
            return
        if any(reconciled.values()):
            self.loguru_logger.info(
                f"Startup reconcile failed interrupted subscriptions work: "
                f"{reconciled}"
            )

    async def _reconcile_citation_artifact_ownership(self) -> None:
        """Run one bounded recovery batch without blocking the UI loop."""

        coordinator = getattr(
            self,
            "citation_artifact_ownership_coordinator",
            None,
        )
        if coordinator is None or not coordinator.writes_enabled:
            return
        try:
            result = await asyncio.to_thread(coordinator.reconcile_pending, limit=25)
        except Exception:
            self.loguru_logger.error(
                "Citation artifact reconciliation failed: "
                "artifact_reconciliation_failed"
            )
            return
        if result.failed:
            self.loguru_logger.warning(
                "Citation artifact reconciliation retained pending operations: "
                f"operation_ids={result.operation_ids!r} "
                f"reason_codes={result.reason_codes!r}"
            )

    async def _migrate_legacy_citations_idle_unit(self) -> None:
        """Drain bounded legacy batches while yielding between every idle unit."""

        if getattr(self, "_legacy_citation_migration_in_flight", False):
            return
        self._legacy_citation_migration_in_flight = True
        retry_count = 0
        try:
            while True:
                migration = getattr(
                    self,
                    "citation_legacy_migration_service",
                    None,
                )
                if migration is None or not migration.ready:
                    return
                try:
                    result = await asyncio.to_thread(migration.migrate_idle_unit)
                except Exception:
                    retry_count += 1
                    self.loguru_logger.error(
                        "Legacy citation migration failed: legacy_migration_failed"
                    )
                    if retry_count >= 3:
                        return
                    await asyncio.sleep(2 ** (retry_count - 1))
                    continue
                state = getattr(result.state, "value", result.state)
                if result.reason_code is not None:
                    self.loguru_logger.warning(
                        "Legacy citation migration retained retry state: "
                        f"reason_code={result.reason_code!r}"
                    )
                    if (
                        state == "running"
                        and result.reason_code == "legacy_cutover_guard_failed"
                    ):
                        retry_count += 1
                        if retry_count >= 3:
                            return
                        await asyncio.sleep(2 ** (retry_count - 1))
                        continue
                retry_count = 0
                if state != "running":
                    return
                await asyncio.sleep(0)
        finally:
            self._legacy_citation_migration_in_flight = False

    def _schedule_footer_status_updates(self) -> None:
        """Wire status-line DB/token status updates after UI readiness."""

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
            # The cache is only a pre-first-screen fallback: per-tick updates
            # resolve the ACTIVE screen's footer via `_active_footer_status`.
            # Splash and the first-run wizard mount no AppFooterStatus, so a
            # miss here must not abort timer setup (task-2721: it previously
            # logged two tracebacks per fresh install and left the DB-size and
            # token timers never started for the whole session).
            try:
                self._db_size_status_widget = self.query_one(AppFooterStatus)
                self.loguru_logger.info("AppFooterStatus widget instance acquired.")
            except QueryError:
                self._db_size_status_widget = None
                self.loguru_logger.debug(
                    "Active screen has no AppFooterStatus; footer timers start "
                    "anyway and each tick resolves the active screen's footer."
                )

            self.set_timer(
                DEFERRED_DB_SIZE_UPDATE_DELAY_SECONDS,
                self.update_db_sizes,
            )
            self.db_status_manager.start_periodic_updates(120)
            record_footer_timer("footer-db-size-periodic")
            self.loguru_logger.info(
                "DB size update timer started for the shell status line (interval: 2 minutes)."
            )

            self.set_timer(0.5, self.update_token_count_display)
            record_footer_timer("footer-token-periodic")
            self._token_count_update_timer = self.set_interval(
                10,
                lambda: self.call_after_refresh(self.update_token_count_display),
            )
            self.loguru_logger.info("Token count update timer started (10s interval).")
        except Exception as e_db_size:
            self.loguru_logger.opt(exception=True).error(
                f"Error setting up DB size indicator for the shell status line: {e_db_size}",
            )

    def _start_deferred_audio_service_initialization(self) -> None:
        """Kick off TTS/STTS initialization after startup readiness."""

        self._schedule_tts_initialization()
        self._schedule_stts_initialization()

    def _screen_preimport_enabled(self) -> bool:
        """Whether the background screen-module pre-importer should run.

        On by default. Off under pytest (``PYTEST_CURRENT_TEST`` -- the same
        signal ``Utils/optional_deps.py`` and ``Metrics/metrics_logger.py``
        already gate background/eager behavior on) so the test suite's many
        ``app.run_test()`` instances don't each spin up an extra
        background-import thread for a mechanism most tests never look at.
        ``TLDW_SCREEN_PREIMPORT`` overrides in either direction: ``"0"``/
        ``"false"`` forces it off even outside pytest, ``"1"``/``"true"``
        forces it on even under pytest -- used by this feature's own tests to
        exercise the real scheduling path rather than only the worker method.
        """
        override = os.environ.get("TLDW_SCREEN_PREIMPORT")
        if override is not None:
            return override.strip().lower() not in ("", "0", "false", "no")
        return "PYTEST_CURRENT_TEST" not in os.environ

    def _screen_preimport_route_order(self) -> tuple[ScreenRoute, ...]:
        """Ordered, module-deduplicated routes for the background pre-importer.

        Several canonical route ids share one module (``"ccp"``/``"personas"``
        both target ``personas_screen.PersonasScreen``, ``"tools_settings"``/
        ``"mcp"`` both target ``mcp_screen.MCPScreen``) -- importing each
        module once is enough, a second ``import_module`` call for the same
        name is just a dict lookup, but there's no reason to schedule the
        redundant work. ``SCREEN_PREIMPORT_PRIORITY_ROUTE_IDS`` (chat/
        library/settings, the audit's three multi-thousand-line modules) go
        first; the rest of the registry follows in stable sorted order.

        Route ids that are ALSO a key in the alias table are skipped: at real
        navigation time, ``_lookup_route()`` resolves the alias to a
        *different* canonical route before ever reaching this dict entry
        (e.g. ``"customize"`` -> the ``settings`` route; ``_SCREEN_ROUTES
        ["customize"]``, pointing at a ``customize_screen`` module that no
        longer exists, is unreachable dead metadata kept for history). Task-
        15472 review round 1: pre-importing it anyway logged a "Screen route
        unavailable: customize: No module named ..." warning on every single
        boot -- a route no click can ever reach should not be attempted.
        """
        shadowed_route_ids = set(registered_screen_aliases())
        routes_by_id = {
            route.screen_name: route for route in registered_screen_routes()
        }
        ordered: list[ScreenRoute] = []
        seen_modules: set[str] = set()

        def _consider(route: ScreenRoute | None) -> None:
            if route is None or route.screen_name in shadowed_route_ids:
                return
            if route.module_path in seen_modules:
                return
            ordered.append(route)
            seen_modules.add(route.module_path)

        for route_id in SCREEN_PREIMPORT_PRIORITY_ROUTE_IDS:
            _consider(routes_by_id.get(route_id))
        for route in registered_screen_routes():
            _consider(route)
        return tuple(ordered)

    def _preimport_screens(self, routes: Iterable[ScreenRoute]) -> None:
        """Warm ``sys.modules`` for ``routes``, one route at a time.

        Runs on a background thread (see ``_schedule_screen_preimport``),
        never the asyncio loop -- ``import_module`` is CPU-bound (bytecode
        compile/exec for chat_screen.py's ~20k lines and friends) and would
        stall UI responsiveness if it ran inline on the event loop. Python's
        import system serializes concurrent imports of the same module
        through its own per-module lock, and a completed import is cached in
        ``sys.modules``, so this is safe to race against a real navigation's
        own ``import_module`` call: nothing is ever imported twice for real,
        and a route the user never visits just cost one idle-thread import
        that would otherwise have happened on their first click to it.

        Each route calls ``ScreenRoute.load_screen_class()`` -- the exact
        method the real navigation path calls -- wrapped in its own
        ``try/except Exception``. ``load_screen_class()`` already swallows
        ``ImportError``/``AttributeError`` and logs a warning; the broader
        catch here is belt-and-suspenders so one screen module raising
        something stranger at import time can't kill the thread or block the
        remaining routes. Either way a failed import is never cached in
        ``sys.modules`` (CPython evicts a partially-initialized module on
        import failure), so a pre-import attempt that fails changes nothing
        about what a real navigation to that route does next: it fails again,
        identically (AC #3).

        Args:
            routes: The routes to pre-import, in order. Factored out of
                ``_preimport_heavy_screens`` so tests can target one or two
                routes directly instead of the whole registry.
        """
        for route in routes:
            try:
                route.load_screen_class()
            except Exception as exc:
                self.loguru_logger.debug(
                    "Screen pre-import failed (route={}, error_type={})",
                    route.screen_name,
                    type(exc).__name__,
                )

    def _preimport_heavy_screens(self) -> None:
        """Warm ``sys.modules`` for every registered screen route.

        See ``_preimport_screens`` for the per-route mechanics; this just
        supplies the full, priority-ordered route list.
        """
        self._preimport_screens(self._screen_preimport_route_order())

    def _schedule_screen_preimport(self) -> None:
        """Start the background screen-module pre-importer, at most once."""
        if not self._screen_preimport_enabled():
            return
        if self._shutting_down:
            return
        if self._screen_preimport_thread is not None:
            return
        thread = threading.Thread(
            target=self._preimport_heavy_screens,
            name="tldw-screen-preimport",
            daemon=True,
        )
        self._screen_preimport_thread = thread
        thread.start()

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
            handler = TTSEventHandler(
                profile_service_loader=self._ensure_tts_profile_service,
                default_profile_id_reader=(
                    lambda: get_cli_setting("app_tts", "default_profile_id", None)
                ),
            )
            handler.app = self
            await handler.initialize_tts()
            self._tts_handler = handler
            self.loguru_logger.info("TTS service initialized successfully")
        except Exception as e:
            self.loguru_logger.error(f"Failed to initialize TTS service: {e}")
            self._tts_handler = None
        finally:
            log_histogram(
                "app_post_mount_phase_duration_seconds",
                time.perf_counter() - phase_start,
                labels={"phase": "tts_init_deferred"},
                documentation="Duration of post-mount phase in seconds",
            )
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
            log_histogram(
                "app_post_mount_phase_duration_seconds",
                time.perf_counter() - phase_start,
                labels={"phase": "stts_init_deferred"},
                documentation="Duration of post-mount phase in seconds",
            )
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

    async def on_shutdown_request(self) -> None:  # Use the imported ShutdownRequest
        logging.info("--- App Shutdown Requested ---")

        # Set shutdown flag to prevent new operations
        self._shutting_down = True

        # Cancel all active workers first
        await self._cancel_and_settle_workers("shutdown request")

        if self._rich_log_handler:
            await self._rich_log_handler.stop_processor()
            logging.info("RichLogHandler processor stopped.")

        # --- Stop DB Size Update Timer ---
        self.db_status_manager.stop_periodic_updates()
        self._stop_footer_status_timers()
        self.loguru_logger.info("DB size update timer stopped.")
        # --- End Stop DB Size Update Timer ---

    async def _cancel_and_settle_workers(self, phase: str) -> None:
        """Cancel every live worker and actually wait for them, bounded.

        task-19561. Both shutdown hooks used to cancel their workers and
        then ``await asyncio.sleep(0.1)`` -- a flat wait that is
        simultaneously too long (nothing to wait for on a quiet exit) and
        far too short (a worker mid-``await`` gets one tick, and a thread
        worker gets nothing at all), and which never observed the outcome
        either way. Waiting on the workers themselves is both faster in the
        common case and honest in the uncommon one; the timeout keeps a
        worker that ignores cancellation from turning quit into a hang, and
        says which ones they were.
        """
        try:
            active_workers = [w for w in self.workers if not w.is_finished]
            if not active_workers:
                return
            self.loguru_logger.info(
                f"Cancelling {len(active_workers)} active workers ({phase})"
            )
            for worker in active_workers:
                worker.cancel()
            # `asyncio.wait`, NOT `wait_for(...)`: on expiry `wait_for`
            # cancels what it is waiting on and awaits that cancellation, so
            # anything that does not honour a cancel hangs the very call
            # meant to bound it. `wait` returns the stragglers instead.
            waiters = [asyncio.ensure_future(w.wait()) for w in active_workers]
            _, unsettled = await asyncio.wait(
                waiters, timeout=WORKER_CANCELLATION_GRACE_SECONDS
            )
            for waiter in waiters:
                if waiter.done() and not waiter.cancelled():
                    # WorkerCancelled/WorkerFailed are the expected outcomes
                    # of cancelling at shutdown; retrieve them so they do not
                    # resurface as "exception was never retrieved".
                    waiter.exception()
                else:
                    waiter.cancel()
            if unsettled:
                stragglers = [w.name for w in active_workers if not w.is_finished]
                self.loguru_logger.warning(
                    f"{len(stragglers)} worker(s) did not settle within "
                    f"{WORKER_CANCELLATION_GRACE_SECONDS}s of cancellation "
                    f"({phase}): {stragglers}"
                )
        except Exception as e:
            self.loguru_logger.error(f"Error cancelling workers ({phase}): {e}")

    async def _close_server_context_provider_cached_client(self) -> None:
        server_context_provider = getattr(self, "server_context_provider", None)
        close_cached_client = getattr(
            server_context_provider, "close_cached_client", None
        )
        if callable(close_cached_client):
            await close_cached_client()

    async def _disconnect_local_mcp_client(self) -> None:
        """Best-effort teardown of local MCP client sessions (P5-T6).

        ``local_mcp_control_service.client`` (``LocalMCPControlService.
        client``) stays ``None`` until a local external MCP profile is
        actually connected during this process's lifetime (see
        ``LocalMCPControlService._get_client``'s lazy-init) -- a session-
        free app quit is a no-op here, matching the sibling teardown
        blocks' own guarded style.
        """
        local_mcp_control_service = getattr(self, "local_mcp_control_service", None)
        client = getattr(local_mcp_control_service, "client", None)
        if client is not None and getattr(client, "sessions", None):
            await client.disconnect_all()

    async def _shutdown_file_notes_session_owner(self) -> None:
        """Settle the process-owned File Notes Git lifecycle exactly once."""
        owner = getattr(self, "file_notes_session_owner", None)
        if owner is None:
            return
        task = getattr(self, "_file_notes_session_owner_shutdown_task", None)
        if task is None:
            task = asyncio.create_task(
                owner.shutdown_async(),
                name="shutdown_file_notes_session_owner",
            )
            self._file_notes_session_owner_shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_notes_sync_runtime(self) -> None:
        """Settle the application-owned lasting-sync runtime exactly once."""

        task = getattr(self, "_notes_sync_runtime_shutdown_task", None)
        if task is None:
            task = asyncio.create_task(
                self.notes_sync_runtime_owner.shutdown(),
                name="shutdown_notes_sync_runtime",
            )
            self._notes_sync_runtime_shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_console_image_edits(self) -> None:
        """Cancel and settle app-owned H3 edits exactly once before teardown."""
        task = self._console_image_edit_shutdown_task
        if task is None:
            task = asyncio.create_task(
                self.console_image_edit_operations.shutdown(),
                name="shutdown_console_image_edits",
            )
            self._console_image_edit_shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_persona_buddy(self) -> None:
        """Drain the app-owned Buddy before profile database teardown.

        Peeks the lazy controller slot (TASK-21103): a controller that was
        never built has nothing to drain, and going through the property
        here could CONSTRUCT one (importing Persona_Visual + PIL) purely to
        shut it down.
        """
        task = self._persona_buddy_shutdown_task
        if task is None:
            controller = self._persona_buddy_controller
            if controller is None:
                return
            task = asyncio.create_task(
                controller.shutdown(),
                name="shutdown_persona_buddy",
            )
            self._persona_buddy_shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_actor_pack_export(self) -> None:
        """Cancel and drain Actor Pack export before profile teardown."""

        controller = getattr(self, "actor_pack_export_controller", None)
        if controller is None:
            return
        task = getattr(self, "_actor_pack_export_shutdown_task", None)
        if task is None:
            task = asyncio.create_task(
                controller.shutdown(),
                name="shutdown_actor_pack_export",
            )
            self._actor_pack_export_shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_console_runtime(self) -> None:
        """Destroy the app-owned Console runtime exactly once, at exit.

        task-15860: the runtime survives every navigation away from
        Console, so the unmount Textual performs at exit is no longer what
        ends it -- this is. `ConsoleRuntime.dispose` runs the permanent
        teardown in the order `ChatScreen.on_unmount` used to:
        `controller.shutdown()`, then `gateway.aclose()`. Idempotent: the
        runtime detaches itself from the app on the way out.
        """
        task = self._console_runtime_shutdown_task
        if task is None:
            task = asyncio.create_task(
                dispose_console_runtime(self),
                name="shutdown_console_runtime",
            )
            self._console_runtime_shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_app_owned_lifecycles(self) -> None:
        """Drain durable app-owned work before Textual closes screen state."""
        await self._shutdown_notes_sync_runtime()
        await self._shutdown_actor_pack_export()
        # Console shutdown terminally fences every trusted Buddy producer
        # before Buddy itself closes admission and drains owned work.
        await self._shutdown_console_runtime()
        await self._shutdown_persona_buddy()
        coordinator = getattr(self, "_audio_cpp_artifact_lease_coordinator", None)
        if coordinator is not None:
            await coordinator.shutdown()
        await self.audio_cpp_model_install_owner.shutdown()
        await self._shutdown_console_image_edits()
        await self._shutdown_file_notes_session_owner()

    async def _shutdown(self) -> None:
        """Settle app-owned durable work before Textual closes screens."""
        cancellation: asyncio.CancelledError | None = None
        owner_error: BaseException | None = None
        shutdown_task = asyncio.current_task()
        cancellation_requests = (
            shutdown_task.cancelling() if shutdown_task is not None else 0
        )
        while True:
            try:
                await self._shutdown_app_owned_lifecycles()
            except asyncio.CancelledError as error:
                next_cancellation_requests = (
                    shutdown_task.cancelling() if shutdown_task is not None else 0
                )
                if next_cancellation_requests > cancellation_requests:
                    cancellation = cancellation or error
                    cancellation_requests = next_cancellation_requests
                    continue
                owner_error = error
            except BaseException as error:
                owner_error = error
            break

        shutdown_error: BaseException | None = None
        try:
            await super()._shutdown()
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
        except BaseException as error:
            shutdown_error = error

        if shutdown_error is not None:
            if owner_error is not None:
                shutdown_error.add_note(
                    "App-owned lifecycle shutdown also failed before "
                    "Textual screen teardown"
                )
            if cancellation is not None:
                shutdown_error.add_note(
                    "Application shutdown cancellation was also requested"
                )
            raise shutdown_error
        if owner_error is not None:
            if cancellation is not None:
                owner_error.add_note(
                    "Application shutdown cancellation was delayed while "
                    "preserving the lifecycle shutdown failure"
                )
            raise owner_error
        if cancellation is not None:
            raise cancellation

    def _handle_exception(self, error: Exception) -> None:
        """Record the crash type, then let Textual do what it always did.

        TASK-1240. Names the exception class only -- never the message, which is
        caller-supplied text and may quote user or model content. Calls super()
        unconditionally: Textual sets the return code here, and swallowing that
        would turn a crash into a hang.

        `WorkerFailed` is unwrapped. When a worker raises and `exit_on_error` is
        true (the default), `Worker._run` sets `WorkerState.ERROR` -- posting
        `StateChanged` *asynchronously* -- and then calls this method
        *synchronously* with `WorkerFailed(self._error)`. So this override fires
        first and, without unwrapping, would persist
        `exception_type=WorkerFailed` for every worker crash in the app, while
        `_fatal_error()` -> `_close_messages_no_wait()` races the queued
        `StateChanged` so the `worker_failed` event that carries the real type
        and `operation` may never be delivered. A crashed session's log would
        then read `event=unhandled_exception exception_type=WorkerFailed` and
        nothing else. `WorkerFailed.error` holds the real exception.
        """
        from textual.worker import WorkerFailed

        underlying = (
            getattr(error, "error", None) if isinstance(error, WorkerFailed) else None
        )
        try:
            persist_event(
                _DIAGNOSTICS_COMPONENT_APP,
                "unhandled_exception",
                level=logging.ERROR,
                exception_type=type(
                    underlying if underlying is not None else error
                ).__name__,
            )
        except Exception:
            # Diagnostics must never be the reason a crash handler fails.
            pass
        super()._handle_exception(error)

    async def on_unmount(self) -> None:
        """Clean up logging resources on application exit."""
        import asyncio

        logging.info("--- App Unmounting ---")
        # task-19561: from here to process death, everything is teardown.
        # Arm the bound now rather than at the entry point, so the deadline
        # covers this method too -- and so a quit that wedges inside cleanup
        # is bounded exactly like a SIGTERM that does. Idempotent and
        # monotonic: a signal-armed watchdog already holds a tighter
        # deadline and this call leaves it alone.
        arm_exit_watchdog(reason="app unmount")
        # TASK-1240. Distinguishes a clean exit from a kill: a log whose last
        # line is app_started ended abruptly. Wrapped, and deliberately so:
        # this line sits ABOVE the entire shutdown sequence -- DB closes,
        # worker cancellation, ingest pool teardown. An exception escaping here
        # would skip all of it. Diagnostics must never break the thing they
        # observe.
        try:
            persist_event(_DIAGNOSTICS_COMPONENT_APP, "app_stopping")
        except Exception:
            pass
        try:
            await self._shutdown_app_owned_lifecycles()
        except Exception as error:
            self.loguru_logger.warning(
                "App-owned lifecycle fallback shutdown failed "
                f"type={type(error).__name__}"
            )
        self._ui_ready = False
        self._stop_ui_responsiveness_monitor()

        # F3/TASK-601: shut down both Library ingest worker boundaries. Final
        # shutdown order, explicit:
        #   1. `_ingest_shutdown = True` + executor/pool references detached
        #      (synchronous, inside `_shutdown_ingest_parse_pool`) -- their
        #      callbacks short-circuit before marshaling from this point on.
        #   2. Executor close, then `pool.terminate()` + `pool.join()`, on one
        #      detached daemon thread, NEVER this (loop) thread -- terminating
        #      inline here could deadlock against a result-handler thread
        #      parked inside `call_from_thread` (see that method's docstring).
        #      `terminate()` kills every in-flight light parse worker process
        #      immediately -- no waiting on a possibly-long OCR job.
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
            self.loguru_logger.error(
                f"Error shutting down Library ingest parse pool: {e}"
            )

        # Stop all background services and threads
        service_cleanup_primary: BaseException | None = None
        try:
            deferred_tasks = [
                task
                for task in getattr(self, "_deferred_startup_tasks", set())
                if not task.done()
            ]
            for task in deferred_tasks:
                task.cancel()
            if deferred_tasks:
                await asyncio.gather(*deferred_tasks, return_exceptions=True)

            # Stop audio player if it exists
            if hasattr(self, "audio_player"):
                try:
                    await self.audio_player.cleanup()
                    self.loguru_logger.info("Audio player cleaned up")
                except Exception as e:
                    self.loguru_logger.error(f"Error cleaning up audio player: {e}")

            # Clean up handler-owned TTS tasks and files if initialized.
            if hasattr(self, "_tts_handler") and self._tts_handler:
                try:
                    await self._tts_handler.cleanup_tts_resources()
                except Exception as e:
                    self.loguru_logger.error(f"Error cleaning up TTS handler: {e}")

            # Clean up handler-owned S/TT/S tasks and files if initialized.
            if hasattr(self, "_stts_handler") and self._stts_handler:
                try:
                    if hasattr(self._stts_handler, "cleanup_tts_resources"):
                        await self._stts_handler.cleanup_tts_resources()
                except Exception as e:
                    self.loguru_logger.error(f"Error cleaning up STTS handler: {e}")

            # Stop the background scheduler loop cleanly.
            scheduler_loop = getattr(self, "scheduler_loop", None)
            scheduler_worker = getattr(self, "scheduler_worker", None)
            if scheduler_loop is not None:
                scheduler_loop.stop()
            if scheduler_worker is not None:
                try:
                    if not scheduler_worker.is_finished:
                        # Textual's public cancellation contract cancels the
                        # underlying asyncio task. Worker.wait() has no timeout
                        # parameter, so request cancellation before observing it.
                        scheduler_worker.cancel()
                    await scheduler_worker.wait()
                except WorkerCancelled:
                    # Cancellation is the expected public Textual shutdown
                    # contract for a loop that may be sleeping between polls.
                    pass
                except Exception as e:
                    self.loguru_logger.error(f"Error stopping scheduler worker: {e}")

            # task-19561: stopping the scheduler worker does NOT stop the
            # generations it dispatched. `BriefingJobHandler.handle` spawns
            # each one as a bare `asyncio.Task` (Locked Decision 3 -- a
            # multi-minute LLM call must not stall the tick), so they are
            # absent from `App.workers` and survived every cancellation
            # above, only to be destroyed mid-flight when the loop closed.
            # Cancel them here, while the loop is still alive to deliver it.
            briefing_handler = getattr(self, "_briefing_job_handler", None)
            if briefing_handler is not None:
                try:
                    cancelled = await briefing_handler.shutdown()
                    if cancelled:
                        self.loguru_logger.info(
                            f"Cancelled {cancelled} in-flight scheduled briefing "
                            "generation(s)"
                        )
                except Exception as e:
                    self.loguru_logger.error(
                        f"Error stopping scheduled briefing generations: {e}"
                    )

            # Disconnect local MCP client sessions (P5-T6), if any were ever
            # established this run.
            try:
                await self._disconnect_local_mcp_client()
                self.loguru_logger.info("Local MCP client sessions disconnected")
            except Exception as e:
                self.loguru_logger.error(
                    f"Error disconnecting local MCP client sessions: {e}"
                )

            # Cancel any pending workers and wait for them, bounded.
            await self._cancel_and_settle_workers("unmount")

            # Stop media cleanup timer
            if hasattr(self, "_media_cleanup_timer") and self._media_cleanup_timer:
                self._media_cleanup_timer.stop()
                self.loguru_logger.info("Media cleanup timer stopped")

            try:
                await self._close_server_context_provider_cached_client()
                self.loguru_logger.info("Server context provider cached client closed")
            except Exception as e:
                self.loguru_logger.error(
                    f"Error closing server context provider cached client: {e}"
                )

        except asyncio.CancelledError as error:
            service_cleanup_primary = error
        except Exception as e:
            self.loguru_logger.error(f"Error during service cleanup: {e}")
        except BaseException as error:
            service_cleanup_primary = error
        finally:
            try:
                await self._close_owned_tts_resources()
                self.loguru_logger.info("TTS resources cleaned up properly")
            except BaseException as cleanup_error:
                if service_cleanup_primary is not None:
                    service_cleanup_primary.add_note(
                        "TTS cleanup also failed while preserving the primary "
                        "shutdown error"
                    )
                    self.loguru_logger.warning(
                        "TTS owner cleanup failed while preserving shutdown "
                        f"type={type(cleanup_error).__name__} "
                        "code=operation_failed"
                    )
                elif isinstance(cleanup_error, Exception):
                    self.loguru_logger.warning(
                        "TTS owner cleanup phase=unmount failed "
                        f"type={type(cleanup_error).__name__} "
                        "code=operation_failed"
                    )
                else:
                    raise
        if service_cleanup_primary is not None:
            raise service_cleanup_primary

        # Original cleanup code
        if self._rich_log_handler:  # Ensure it's removed if it exists
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
            import platform

            # On macOS, force kill any afplay processes
            if platform.system() == "Darwin":
                try:
                    # Find and kill any afplay processes spawned by this app
                    import psutil

                    current_pid = os.getpid()
                    for proc in psutil.process_iter(["pid", "name", "ppid"]):
                        try:
                            if (
                                proc.info["name"] == "afplay"
                                and proc.info["ppid"] == current_pid
                            ):
                                self.loguru_logger.info(
                                    f"Killing orphaned afplay process: {proc.info['pid']}"
                                )
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
                            subprocess.run(
                                ["killall", "afplay"], capture_output=True, timeout=1
                            )
                            self.loguru_logger.info("Killed all afplay processes")
                        except Exception as e:
                            self.loguru_logger.debug(
                                f"Could not kill afplay processes: {e}"
                            )

                    # Run in background to avoid blocking
                    self.run_worker(kill_afplay_processes, name="kill_afplay")
            # task-19561: this used to reach into `loop._default_executor`,
            # call `shutdown(wait=False)` and then set the private attribute
            # to `None`. Nulling it is what made the situation worse, not
            # better: `asyncio.run`'s `Runner.close()` ends with `await
            # loop.shutdown_default_executor(THREAD_JOIN_TIMEOUT)`, which
            # JOINS the worker threads while the loop is still alive. With
            # `_default_executor` set to `None` that coroutine returns at its
            # second line, and the very threads this block was trying to
            # hurry along were instead left for `threading._shutdown()` to
            # join with no bound at all. `run_worker(..., thread=True)` runs
            # on that same default executor (Textual's `Worker._run_threaded`
            # ends in `loop.run_in_executor(None, ...)`), so this is not a
            # corner case.
            #
            # Precise about the other half, because it is easy to overclaim:
            # `shutdown_default_executor` sets `_executor_shutdown_called`
            # BEFORE its `if self._default_executor is None: return`, so the
            # "a stray late `run_in_executor` raises" fence applied at the
            # merge base too. What nulling actually cost was the join -- plus
            # a window between this block and `Runner.close()` in which a late
            # `run_in_executor` would build a brand-new pool (that one IS
            # real, `BaseEventLoop.run_in_executor` creates one when
            # `_default_executor` is None and the fence is not yet set).
            #
            # Doing nothing here is the fix: the public, bounded shutdown
            # runs a few milliseconds later, on its own. Verified on CPython
            # 3.12.11, where `constants.THREAD_JOIN_TIMEOUT` is 300s -- far
            # looser than the exit watchdog armed at the top of this method,
            # which is what actually bounds the wait.

            # Clean up any lingering subprocess
            for proc in (
                subprocess._active.copy()
            ):  # Make a copy to avoid modification during iteration
                try:
                    if proc.poll() is None:  # Process is still running
                        self.loguru_logger.warning(
                            f"Terminating lingering subprocess PID: {proc.pid}"
                        )
                        proc.terminate()
                        try:
                            proc.wait(timeout=1.0)  # Give it 1 second to terminate
                        except subprocess.TimeoutExpired:
                            proc.kill()  # Force kill if it doesn't terminate
                            proc.wait()
                except Exception as e:
                    self.loguru_logger.error(f"Error terminating subprocess: {e}")

            # task-19561: a loop that force-set `thread.daemon = True` on
            # every live `ThreadPoolExecutor*`/`AudioPlayer*` thread used to
            # sit here. CPython raises `RuntimeError: cannot set daemon
            # status of active thread` for every one of them, so it changed
            # nothing and logged an ERROR per thread while doing it. The
            # "Active non-daemon threads remaining" warning that followed
            # reported the same threads a moment before the process was
            # going to wait on them anyway, with no way to act on it.
            # Both are gone. What replaces them is the exit watchdog armed
            # at the top of this method: it names the threads still alive
            # at the moment the wait actually becomes a hang, and ends the
            # process rather than merely describing it.

            # Threads that expose a cooperative stop() still get asked --
            # that half was never dead code.
            for thread in threading.enumerate():
                if thread is threading.main_thread() or not thread.is_alive():
                    continue
                stop = getattr(thread, "stop", None)
                if callable(stop):
                    try:
                        stop()
                        self.loguru_logger.info(f"Stopped thread: {thread.name}")
                    except Exception as e:
                        self.loguru_logger.error(
                            f"Error stopping thread {thread.name}: {e}"
                        )
        except Exception as e:
            self.loguru_logger.error(f"Error checking active threads: {e}")

        # Close the persisted Library ingest job history store (after pool
        # shutdown, above -- no more job writes are in flight by this point).
        store = getattr(self, "_library_ingest_jobs_store", None)
        if store is not None:
            store.close()

        # Nothing this app owns is left to ask; a signal from here on has
        # no orderly path to offer and should unwind the main thread.
        unregister_running_app(self)

        logging.shutdown()
        self.loguru_logger.info("--- App Unmounted (Loguru) ---")

    def _log_view_dimensions(self, view, parent):
        """Helper to log view dimensions after refresh."""
        self.loguru_logger.info(
            f"After refresh - View {view.id} dimensions: width={view.size.width}, height={view.size.height}"
        )
        self.loguru_logger.info(
            f"After refresh - Parent dimensions: width={parent.size.width}, height={parent.size.height}"
        )

    ########################################################################
    #
    # --- EVENT DISPATCHERS ---
    #
    ########################################################################
    # Notes editor changes are handled inside the Library screen, not dispatched here.

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

        # TASK-1240. One hook already sees every worker transition, so failures
        # are recorded without touching any of the 398 run_worker call sites.
        # Only ERROR persists: a start or success event here would emit a line
        # per keystroke-triggered search and per timer tick.
        if event.state is WorkerState.ERROR:
            error = getattr(event.worker, "error", None)
            # DO NOT "improve" `operation` to `event.worker.description`.
            # `Worker.name` is code-side -- the method or literal name given at
            # the `run_worker`/`@work` site. `Worker.description` is built by
            # textual's `_work_decorator` as `f"{name}={value!r}"` over the
            # worker's *actual arguments*, so for a chat, tool or provider
            # worker it contains prompts, API keys and tool values verbatim.
            # Persisting it would put exactly what ADR-029 excludes on disk.
            #
            # `else "unknown"` stays. `Worker._run` assigns `self.state =
            # WorkerState.ERROR` -- whose setter posts `StateChanged` -- one
            # line *before* `self._error = error`. Delivery is via the message
            # queue, so `_error` has landed by the time this handler runs in
            # every real interleaving; the branch is a total-function guard for
            # the ordering itself and for duck-typed workers, and it costs one
            # comparison on a path that only runs when something already broke.
            try:
                persist_event(
                    _DIAGNOSTICS_COMPONENT_APP,
                    "worker_failed",
                    level=logging.ERROR,
                    operation=str(worker_name or "unknown"),
                    exception_type=(
                        type(error).__name__ if error is not None else "unknown"
                    ),
                )
            except Exception:
                # Diagnostics must never break the worker hook every worker
                # transition in the app passes through.
                pass

        # Delegate to the handler registry
        handled = await self.worker_handler_registry.handle_event(event)

        if not handled:
            # Log unhandled workers for debugging
            self.loguru_logger.warning(
                f"No handler found for worker '{worker_name}' (Group: {worker_group})"
            )

    def chat_wrapper(self, strip_thinking_tags: bool = True, **kwargs: Any) -> Any:
        """Delegate a retained non-streaming media call.

        Args:
            strip_thinking_tags: Whether the core chat call removes thinking
                tags.
            **kwargs: Arguments forwarded through the retained worker adapter.

        Returns:
            The non-streaming core chat result.

        Raises:
            ValueError: If a caller requests streaming, which is owned by the
                native Console provider gateway.
        """
        return worker_events.chat_wrapper_function(
            self, strip_thinking_tags=strip_thinking_tags, **kwargs
        )

    def schedule_media_cleanup(self) -> None:
        """Schedule periodic media cleanup based on configuration."""
        # TASK-1975: change-review snapshot retention rides the same
        # maintenance path but has its OWN knob ([change_review]
        # retention_days; <=0 disables inside the pass) -- disabling media
        # cleanup must not silently disable snapshot retention.
        try:
            self._change_review_retention_startup_timer = self.set_timer(
                DEFERRED_MEDIA_CLEANUP_DELAY_SECONDS + 60,
                self._perform_change_review_retention,
            )
            self._change_review_retention_timer = self.set_interval(
                24 * 3600, self._perform_change_review_retention
            )
        except Exception:  # noqa: BLE001 -- maintenance must never block boot
            self.loguru_logger.opt(exception=True).warning(
                "Could not schedule change-review retention"
            )
        try:
            # Get cleanup configuration
            cleanup_config = get_cli_setting("media_cleanup", "enabled", True)
            if not cleanup_config:
                self.loguru_logger.info("Media cleanup is disabled in configuration")
                return

            cleanup_interval_hours = get_cli_setting(
                "media_cleanup", "cleanup_interval_hours", 24
            )
            cleanup_on_startup = get_cli_setting(
                "media_cleanup", "cleanup_on_startup", True
            )

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
            self._media_cleanup_timer = self.set_interval(
                cleanup_interval_seconds, self.perform_media_cleanup
            )
            self.loguru_logger.info(
                f"Scheduled media cleanup every {cleanup_interval_hours} hours"
            )

        except Exception as e:
            self.loguru_logger.opt(exception=True).error(
                f"Error scheduling media cleanup: {e}"
            )

    async def _perform_change_review_retention(self) -> None:
        """Run one change-review retention pass off the UI thread (TASK-1975)."""
        try:
            db = getattr(self, "chachanotes_db", None)
            db_path = getattr(db, "db_path", None) if db is not None else None
            if not db_path or str(db_path) == ":memory:":
                return
            from tldw_chatbook.Workspaces.change_retention import (
                run_retention_for_app,
            )

            await asyncio.to_thread(run_retention_for_app, db_path)
        except Exception:  # noqa: BLE001 -- retention must never surface to the UI
            self.loguru_logger.opt(exception=True).warning(
                "Change-review retention pass failed"
            )

    async def perform_media_cleanup(self) -> None:
        """Perform media cleanup based on configuration settings."""
        try:
            if not self.media_db:
                self.loguru_logger.warning("Media database not available for cleanup")
                return

            # Get cleanup configuration
            cleanup_days = get_cli_setting("media_cleanup", "cleanup_days", 30)
            max_items = get_cli_setting("media_cleanup", "max_items_per_cleanup", 100)
            notify_before = get_cli_setting(
                "media_cleanup", "notify_before_cleanup", True
            )

            # Check for candidates first
            candidates = await asyncio.to_thread(
                self.media_db.get_deletion_candidates, cleanup_days
            )

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
                    timeout=5,
                )

            # Perform the cleanup
            deleted_count = await asyncio.to_thread(
                self.media_db.hard_delete_old_media, cleanup_days
            )

            if deleted_count > 0:
                self.loguru_logger.info(
                    f"Media cleanup completed: {deleted_count} items permanently deleted"
                )
                self.notify(
                    f"Media cleanup completed: {deleted_count} items permanently deleted",
                    severity="information",
                    timeout=3,
                )

        except Exception as e:
            self.loguru_logger.opt(exception=True).error(
                f"Error during media cleanup: {e}"
            )
            self.notify(
                f"Error during media cleanup: {str(e)}", severity="error", timeout=5
            )

    async def action_show_workbench_help(self) -> None:
        """Delegate contextual help to the active Workbench screen.

        Screens without a custom handler get a generic help panel generated
        from their own BINDINGS (falling back to the app-level bindings when
        the screen declares none), so F1 always shows truthful help.
        """
        handler = getattr(self.screen, "action_show_workbench_help", None)
        if callable(handler):
            result = handler()
            if inspect.isawaitable(result):
                await result
            return
        self._show_generic_screen_help()

    def _show_generic_screen_help(self) -> None:
        """Show a help panel generated from the active screen's BINDINGS."""
        screen = self.screen
        shortcuts = _bindings_to_shortcuts(getattr(screen, "BINDINGS", ()))
        if not shortcuts:
            shortcuts = _bindings_to_shortcuts(getattr(type(self), "BINDINGS", ()))
        screen_name = type(screen).__name__
        state = WorkbenchHelpState(
            route_id=str(getattr(self, "current_tab", "") or screen_name),
            title=f"{screen_name} Shortcuts",
            shortcuts=shortcuts,
        )
        self.push_screen(WorkbenchHelpPanel(state))

    def action_shell_destination(self, index: int | str) -> None:
        """Navigate to the shell destination at ``index`` (Ctrl+1..9, Ctrl+0 layer).

        Args:
            index: Zero-based destination index. Textual binding action arguments
                are passed as strings, so this is accepted as either an ``int``
                (direct calls/tests) or ``str`` (hotkey bindings) and coerced.
        """
        try:
            idx = int(index)
        except (ValueError, TypeError):
            return
        if idx < 0 or idx >= len(SHELL_DESTINATION_ORDER):
            return
        destination = SHELL_DESTINATION_ORDER[idx]
        self.post_message(NavigateToScreen(destination.primary_route))

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
        """Dispatch one guarded asynchronous pre-quit confirmation worker."""

        if self._quit_in_progress:
            return
        self._quit_in_progress = True
        quit_flow = self._confirm_and_quit()
        try:
            self.run_worker(
                quit_flow,
                group="application-quit",
                exclusive=True,
                exit_on_error=False,
            )
        except Exception:
            quit_flow.close()
            self._quit_in_progress = False
            loguru_logger.warning(
                "Application quit worker could not start; staying in the app"
            )

    async def _confirm_and_quit(self) -> None:
        """Confirm the active screen, then execute one approved cleanup pass."""

        loguru_logger.info("Application quit initiated")
        try:
            current_screen = self.screen
            confirm_quit = getattr(current_screen, "confirm_quit", None)
            if callable(confirm_quit):
                decision = confirm_quit()
                if inspect.isawaitable(decision):
                    decision = await decision
                if decision is False:
                    self._quit_in_progress = False
                    return
        except Exception:
            loguru_logger.warning("Pre-quit confirmation failed; staying in the app")
            self._quit_in_progress = False
            try:
                self.notify(
                    "Couldn't confirm quitting; staying in Chatbook.",
                    severity="warning",
                )
            except Exception:
                pass
            return

        try:
            prepare_for_quit = getattr(current_screen, "prepare_for_quit", None)
            if callable(prepare_for_quit):
                preparation = prepare_for_quit()
                if inspect.isawaitable(preparation):
                    await preparation
        except Exception:
            loguru_logger.warning("Pre-quit shutdown guard failed; staying in the app")
            self._quit_in_progress = False
            try:
                self.notify(
                    "Couldn't prepare a safe shutdown; staying in Chatbook.",
                    severity="warning",
                )
            except Exception:
                pass
            return

        self._shutting_down = True
        await self._run_approved_quit_cleanup()

    async def _run_approved_quit_cleanup(self) -> None:
        """Preserve quit ordering without blocking the Textual event loop."""

        try:
            await self._cleanup_audio_for_quit()
            media_timer = getattr(self, "_media_cleanup_timer", None)
            if media_timer is not None:
                try:
                    media_timer.stop()
                except Exception:
                    loguru_logger.warning(
                        "Media cleanup timer could not stop during quit"
                    )
            try:
                await asyncio.to_thread(self._run_blocking_quit_persistence)
            except Exception:
                loguru_logger.warning("Blocking quit persistence failed")
        finally:
            self.exit()

    async def _cleanup_audio_for_quit(self) -> None:
        """Stop and release app-owned audio before the final exit."""

        audio_player = getattr(self, "audio_player", None)
        if audio_player is None:
            return
        try:
            await asyncio.wait_for(audio_player.stop(), timeout=0.5)
        except asyncio.TimeoutError:
            loguru_logger.warning("Audio stop timed out")
        except Exception:
            loguru_logger.warning("Audio stop failed during quit")
        try:
            await asyncio.wait_for(audio_player.cleanup(), timeout=0.5)
        except asyncio.TimeoutError:
            loguru_logger.warning("Audio cleanup timed out")
        except Exception:
            loguru_logger.warning("Audio cleanup failed during quit")

    @staticmethod
    def _save_shutdown_caches_with_timeout() -> None:
        """Retain the existing bounded cache-save compatibility pass."""

        loguru_logger.debug("Cache saving skipped - handled by simplified RAG service")

    def _run_blocking_quit_persistence(self) -> None:
        """Run timed joins and configuration persistence off the app loop."""

        try:
            save_thread = threading.Thread(
                target=self._save_shutdown_caches_with_timeout,
                name="chatbook-quit-cache-save",
                daemon=True,
            )
            save_thread.start()
            save_thread.join(timeout=2.0)
            if save_thread.is_alive():
                loguru_logger.warning("Cache save timed out - proceeding with quit")
        except Exception:
            loguru_logger.warning("Error in quit cache handler")

        try:
            persisted = persist_cli_config_for_shutdown()
        except Exception:
            loguru_logger.warning("Configuration shutdown persistence raised an error")
        else:
            if not persisted:
                loguru_logger.warning("Configuration shutdown persistence failed")

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


def _is_source_tree(package_root: Path) -> bool:
    """Return whether package files are inside a build-capable source tree."""

    return (package_root.parent / "pyproject.toml").is_file()


#: A class-level ``BUNDLED_CSS`` / ``BUNDLED_SCREEN_CSS`` *assignment*, which is
#: what makes a module an input to the generated stylesheets. Anchored on the
#: assignment rather than matching the bare name anywhere in the file: four
#: package modules -- including this one, via ``_generated_css_is_stale``'s own
#: docstring -- discuss the marker while declaring nothing, and a plain substring
#: test made every edit to any of them rebuild the CSS on the next boot, quietly
#: rewriting the committed bundle's ``Generated:`` timestamp. A module that has
#: just *gained* a declaration is still caught: a declaration is an assignment.
_BUNDLED_CSS_DECLARATION_RE = re.compile(r"^\s*BUNDLED_(?:SCREEN_)?CSS\s*[:=]", re.M)


def _load_css_build_manifest(css_dir: Path) -> dict[str, list] | None:
    """Load the builder's content manifest, or ``None`` when absent/invalid.

    The manifest is written by ``build_css.write_build_manifest`` beside the
    generated sheets; see TASK-18910. Each entry is ``[sha256, mtime_at_build]``.
    Any read/parse/shape problem returns ``None`` so the caller falls back to
    the legacy mtime rule -- a broken manifest costs one spurious rebuild,
    never a missed one.
    """
    try:
        import json

        with open(css_dir / build_css.BUILD_MANIFEST_FILENAME, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data:
        # An empty manifest is treated as absent: the max() over its entries
        # would raise, and an empty build is not a state the builder can
        # produce (it always records at least the CSS_MODULES that exist).
        return None
    manifest: dict[str, list] = {}
    for key, value in data.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], str)
            or not isinstance(value[1], (int, float))
        ):
            return None  # unknown shape: treat as absent
        manifest[key] = value
    return manifest


def _save_css_build_manifest(css_dir: Path, manifest: dict[str, list]) -> None:
    """Persist an updated manifest (mtime refreshes after hash confirmation).

    Best-effort: a write failure costs one re-hash on the next boot, never a
    missed or spurious rebuild -- the in-memory decision has already been
    made with the correct data.
    """
    try:
        import json

        (css_dir / build_css.BUILD_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _generated_css_is_stale(package_root: Path) -> tuple[bool, str]:
    """Return whether the generated stylesheets need rebuilding, and why.

    Source-tree boots rebuild the CSS when its inputs have moved on. Before
    TASK-15450 every input was a ``.tcss`` module, so checking those mtimes was
    exhaustive. Four of the five generated sheets are now built from class-level
    ``BUNDLED_CSS`` / ``BUNDLED_SCREEN_CSS`` literals in Python modules, so a
    widget-CSS edit would otherwise have *no effect* until someone remembered to
    run ``build_css.py`` by hand -- where editing ``DEFAULT_CSS`` used to take
    effect on the very next run, because Textual read it straight off the class.

    A Python module counts as an input only if it is *newer than the build* and
    actually mentions the marker. Both halves matter. Treating every ``.py`` as
    an input was tried first and is wrong: editing ``app.py`` -- or any of the
    ~1,640 files in this package -- would then re-run the build subprocess on
    every single developer boot. Reading files to find the marker is likewise
    only affordable because the mtime test has already narrowed the set, which is
    normally empty. Checking the marker rather than the list of modules the
    sheets currently name is what catches a module that has just *gained* a
    ``BUNDLED_CSS`` declaration -- exactly the file a "nothing happened" bug
    report starts from.

    Cost: one ``os.walk`` of the package, ~0.3 ms warm for ~1,640 files, plus a
    read of each file changed since the last build (normally none). It runs only
    under ``_is_source_tree`` -- for developers, never for a wheel install -- and
    never on the per-frame or per-keystroke paths.

    Known gap: *deleting* a module that carried ``BUNDLED_CSS`` leaves no newer
    file behind, so it is not detected here. The CSS bundle guard in CI covers
    that; this check is a dev-loop convenience, not the authority.

    Args:
        package_root: The installed ``tldw_chatbook`` package directory.

    Returns:
        ``(stale, reason)``; ``reason`` is a log-ready phrase, empty when fresh.
    """
    css_dir = package_root / "css"
    generated = [
        css_dir / "tldw_cli_modular.tcss",
        css_dir / build_css.WIDGET_DEFAULTS_SELF_FILENAME,
        css_dir / build_css.WIDGET_DEFAULTS_SCOPED_FILENAME,
        css_dir / build_css.SCREEN_CSS_SELF_FILENAME,
        css_dir / build_css.SCREEN_CSS_SCOPED_FILENAME,
    ]
    missing = [path.name for path in generated if not path.is_file()]
    if missing:
        return True, f"generated stylesheet(s) not found: {', '.join(missing)}"

    # Compare against the OLDEST generated sheet: any one of them being behind
    # its sources is enough to require a rebuild.
    oldest = min(path.stat().st_mtime for path in generated)

    # TASK-18910: when the builder's content manifest is present it is
    # AUTHORITATIVE. Each recorded input is mtime-compared first and hashed
    # when its mtime differs from the recorded build time IN EITHER
    # DIRECTION -- which removes the false positives (branch switch /
    # ``git checkout`` / stash pop rewrite mtimes without changing content;
    # each cost a ~0.7 s synchronous rebuild) while still catching content
    # restored with a preserved or backdated timestamp (``cp -p``,
    # rsync -a), which a "newer than the build" test alone would treat as
    # unchanged. It also closes a masking gap the pure-mtime rule had: a
    # pull that brings regenerated sheets (new sheet mtimes) together with
    # a source edit made without a local rebuild never fired, because the
    # edited source was no longer "newer than the build". Inputs whose
    # hash confirms unchanged content have their recorded mtime refreshed
    # so a one-time mtime move does not re-hash on every later boot. No
    # manifest (first boot after the change, or a wheel install) keeps the
    # legacy mtime rule; the manifest self-heals on the next rebuild.
    manifest = _load_css_build_manifest(css_dir)
    if manifest is not None:
        import hashlib

        from .Utils.path_validation import validate_path

        def _sha256(path: Path) -> str | None:
            digest = hashlib.sha256()
            try:
                with open(path, "rb") as handle:
                    for chunk in iter(
                        lambda: handle.read(build_css.HASH_CHUNK_SIZE_BYTES), b""
                    ):
                        digest.update(chunk)
            except OSError:
                return None
            return digest.hexdigest()

        # A "newer than the build" reference for the declaration scan below:
        # the newest mtime recorded in the manifest (any input mtime past it
        # is one the build never saw, whether or not it is in the manifest).
        newest_recorded = max(entry[1] for entry in manifest.values())

        manifest_dirty = False
        seen = set()
        for key, entry in sorted(manifest.items()):
            recorded_hash, recorded_mtime = entry[0], entry[1]
            # Manifest keys are joined into filesystem paths; a hand-edited
            # manifest must not be able to point the stat/hash reads outside
            # the package (Qodo security finding on PR #1831).
            try:
                source = validate_path(key, package_root, allow_hidden=True)
            except ValueError:
                return True, f"{key} in the build manifest escapes the package"
            try:
                source_mtime = source.stat().st_mtime
            except OSError:
                # Deleted input: the sheets still carry its rules, so a
                # rebuild is required (the pre-manifest code could not see
                # deletions at all -- see its "Known gap" note).
                return True, f"{key} (recorded in the build manifest) was deleted"
            seen.add(key)
            if source_mtime == recorded_mtime:
                continue  # unchanged since the build; skip hashing
            if _sha256(source) != recorded_hash:
                return True, f"{key} changed since the build"
            # Hash-confirmed unchanged: refresh the recorded mtime so this
            # mtime move is not re-hashed on every subsequent boot.
            manifest[key] = [recorded_hash, source_mtime]
            manifest_dirty = True

        if manifest_dirty:
            _save_css_build_manifest(css_dir, manifest)

        # A module that has GAINED a BUNDLED_CSS declaration since the build
        # is not in the manifest; catch it by scanning declarations in any
        # .py newer than the newest recorded build input. A backdated NEW
        # carrier cannot be distinguished from pre-build files by mtime, so
        # the scan also admits files older than the build when they were
        # not part of the recorded set and sit in a CSS-declaring
        # neighbourhood -- bounded by the manifest's own key set: any .py
        # NOT in the manifest is either new or predates the manifest, and
        # reading it once is cheap relative to a rebuild.
        skip = {"__pycache__", *widget_css.EXCLUDED_DIRS}
        for dirpath, dirnames, filenames in os.walk(package_root):
            dirnames[:] = [name for name in dirnames if name not in skip]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                source = os.path.join(dirpath, filename)
                key = Path(source).relative_to(package_root).as_posix()
                if key in seen:
                    continue  # already verified above
                try:
                    if os.stat(source).st_mtime <= newest_recorded:
                        continue
                    with open(source, "r", encoding="utf-8", errors="ignore") as handle:
                        text = handle.read()
                except OSError:
                    continue
                if _BUNDLED_CSS_DECLARATION_RE.search(text):
                    return (
                        True,
                        f"{filename} gained a BUNDLED_CSS declaration since the build",
                    )
        return False, ""

    # Legacy path (no manifest): the pre-TASK-18910 mtime rule, unchanged.
    for subdir in ("core", "layout", "components", "features", "utilities"):
        subdir_path = css_dir / subdir
        if not subdir_path.is_dir():
            continue
        for module in subdir_path.glob("*.tcss"):
            if module.stat().st_mtime > oldest:
                return True, f"CSS module {module.name} is newer than the build"

    skip = {"__pycache__", *widget_css.EXCLUDED_DIRS}
    for dirpath, dirnames, filenames in os.walk(package_root):
        # Match the builder's own view of what an input is: `iter_blocks` skips
        # these directories, so a vendored file mentioning the marker must not
        # trigger a rebuild that would then ignore it.
        dirnames[:] = [name for name in dirnames if name not in skip]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            source = os.path.join(dirpath, filename)
            try:
                if os.stat(source).st_mtime <= oldest:
                    continue
                with open(source, "r", encoding="utf-8", errors="ignore") as handle:
                    text = handle.read()
            except OSError:
                continue  # vanished mid-walk; not our problem to report
            if _BUNDLED_CSS_DECLARATION_RE.search(text):
                return True, f"{filename} carries widget CSS newer than the build"

    return False, ""


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the tldw-cli argument parser (extracted from main_cli_runner() for testability)."""
    parser = argparse.ArgumentParser(
        description="tldw chatbook - A Textual TUI for chatting with LLMs",
        prog="tldw-cli",
    )
    parser.add_argument(
        "--serve", action="store_true", help="Run the application as a web server"
    )
    parser.add_argument(
        "--host", type=str, help="Host address for web server (default: localhost)"
    )
    parser.add_argument("--port", type=int, help="Port for web server (default: 8000)")
    parser.add_argument("--web-title", type=str, help="Title for the web page")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for web server"
    )
    parser.add_argument(
        "--focus",
        action="store_true",
        help="Start chrome-free in the Console (hides nav bar and workbench header)",
    )
    return parser


# --- Main execution block ---
if __name__ == "__main__":
    # Record the launch directory first, before anything can chdir -- the
    # `python -m tldw_chatbook.app` path does not route through
    # main_cli_runner, so it needs its own capture (set-once; harmless if
    # already recorded). See workspace_context_note for why this matters.
    from tldw_chatbook.Tools.workspace_file_roots import (
        set_launch_cwd as _set_launch_cwd,
    )

    _set_launch_cwd()

    # Initialize logging first
    early_logging_app = initialize_early_logging()

    try:
        load_cli_config_and_ensure_existence()
    except Exception as e_cfg_main:
        logging.error(
            f"Could not ensure creation of effective config file: {e_cfg_main}",
            exc_info=True,
        )

    # --- Initialize Metrics Systems ---
    # Initialize Prometheus metrics server
    try:
        # Start Prometheus metrics server on port 8000 (or configure via env/config)
        metrics_port = int(os.environ.get("METRICS_PORT", "8000"))
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
    emoji_is_supported = supports_emoji()  # Call it once
    loguru_logger.info(f"Terminal emoji support detected: {emoji_is_supported}")
    loguru_logger.info(
        f"Using brain: {get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}"
    )
    loguru_logger.info("-" * 30)

    # --- CSS File Handling ---
    package_root = Path(__file__).parent
    if _is_source_tree(package_root):
        try:
            css_dir = package_root / "css"
            css_dir.mkdir(exist_ok=True)

            # Check if modular CSS needs to be built
            build_script_path = css_dir / "build_css.py"

            # Check whether any input -- a .tcss module or a Python module
            # carrying BUNDLED_CSS -- has moved on since the last build.
            should_rebuild, reason = _generated_css_is_stale(package_root)
            if should_rebuild:
                logging.info("Generated CSS is stale during module entry; rebuilding")

            if should_rebuild and build_script_path.exists():
                logging.info("Building modular CSS...")
                import subprocess

                # Build CSS synchronously before starting the app
                result = subprocess.run(
                    [sys.executable, str(build_script_path)],
                    cwd=str(css_dir),
                    capture_output=True,
                    text=True,
                )
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
                            on_cancel=lambda: None,
                        ),
                        wait_for_dismiss=True,
                    )

                    if password:
                        # Verify password
                        from tldw_chatbook.Utils.config_encryption import (
                            config_encryption,
                        )

                        password_verifier = encryption_config.get(
                            "password_verifier", ""
                        )
                        if password_verifier and config_encryption.verify_password(
                            password, password_verifier
                        ):
                            self.password = password
                            self.exit()
                        else:
                            self.notify(
                                "Invalid password. Please try again.", severity="error"
                            )
                            # Re-show the dialog
                            await self.on_mount()
                    else:
                        # User cancelled
                        loguru_logger.error(
                            "Password required but not provided. Exiting."
                        )
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

    # task-1650: resolve textual_image's rendering protocol NOW, while the
    # terminal still answers escape queries. Textual takes raw mode in
    # run() below, after which the query silently fails and every image
    # surface degrades to half-cell rendering.
    from .Utils.terminal_utils import warm_up_image_protocol

    warm_up_image_protocol()

    # argparse terminates here on --help (exit 0) and invalid arguments
    # (exit 2), same as the console-script path -- no guard: swallowing
    # SystemExit would print usage and then launch the TUI anyway.
    _main_args = _build_arg_parser().parse_args()

    # task-18908: --serve historically only worked via the console-script
    # entry; this __main__ path parsed the flags and then ignored them,
    # silently binding the config default port. Route them exactly like
    # main_cli_runner does.
    if _main_args.serve:
        from .Web_Server.serve import check_web_server_available, run_web_server

        if not check_web_server_available():
            loguru_logger.error("Web server feature is not available!")
            loguru_logger.error("Install with: pip install tldw_chatbook[web]")
            raise SystemExit(1)

        loguru_logger.info("Starting tldw_chatbook in web server mode")
        run_web_server(
            host=_main_args.host,
            port=_main_args.port,
            title=_main_args.web_title,
            debug=_main_args.debug,
        )
        raise SystemExit(0)

    # task-19561: `python -m tldw_chatbook.app` installed no signal handlers
    # at all, so SIGTERM took the process out with the kernel default -- even
    # more abrupt than the console script's `os._exit(0)`. Both entry points
    # now share one bounded, graceful mechanism.
    install_termination_handlers()

    # task-21100: pending ChaChaNotes migrations replay inside TldwCli's
    # constructor, before anything can paint -- the terminal is the only
    # surface that exists at this phase, so say what the pause is there.
    from tldw_chatbook.Utils.db_upgrade_notice import (
        print_db_upgrade_notice_if_pending,
    )

    print_db_upgrade_notice_if_pending()

    # Create instance with early logging flag
    app_instance = TldwCli()
    app_instance._cli_focus_override = bool(_main_args.focus)
    # Set the early logging flag so _setup_logging knows logging was already initialized
    app_instance._early_logging_initialized = True
    try:
        app_instance.run()
    except KeyboardInterrupt:
        loguru_logger.info("--- KeyboardInterrupt received ---")
    except Exception:
        loguru_logger.exception("--- CRITICAL ERROR DURING app.run() ---")
        traceback.print_exc()  # Make sure traceback prints
    finally:
        # This might run even if app exits early internally in run()
        loguru_logger.info("--- FINALLY block after app.run() ---")
        # Everything from here is interpreter teardown -- `asyncio.run`'s
        # executor join, `threading._shutdown()`, `atexit`. None of it is
        # interruptible from Python, so this is the last point a bound can
        # be placed on it. Idempotent: a SIGTERM-armed watchdog already
        # holds a tighter deadline and this call leaves it alone.
        arm_exit_watchdog(reason="interpreter exit")

    loguru_logger.info("--- AFTER app.run() call (if not crashed hard) ---")


# Entry point for the tldw-chatbook command
def get_app():
    """Entry point for textual serve.

    Returns the TldwCli app instance without running it.
    """
    # Configure logging to suppress verbose debug messages early
    import logging

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

    package_root = Path(__file__).parent
    if _is_source_tree(package_root):
        css_dir = package_root / "css"
        build_script_path = css_dir / "build_css.py"

        # Same staleness rule as the main entry points: a missing generated
        # sheet, or any input newer than the build (TASK-15450).
        stale, reason = _generated_css_is_stale(package_root)
        if stale and build_script_path.exists():
            print(f"Building modular CSS: {reason}")
            import subprocess

            subprocess.run([sys.executable, str(build_script_path)], check=True)

    return TldwCli()


def main_cli_runner():
    """Entry point for the tldw-chatbook command.

    This function is referenced in pyproject.toml as the entry point for the tldw-chatbook command.
    It initializes logging early and then runs the TldwCli app.
    """
    # Record the launch directory at the earliest point in the process, before
    # anything can chdir. The workspace-context note appended to agent prompts
    # expresses workspace roots relative to this (never as absolute host
    # paths). Set-once: harmless if another entry path already recorded it.
    from tldw_chatbook.Tools.workspace_file_roots import set_launch_cwd

    set_launch_cwd()

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

    # task-19561: SIGTERM used to be answered here by `os._exit(0)` from
    # inside the handler, after an `atexit`-registered `force_cleanup` that
    # tried to daemonize already-started threads (a `RuntimeError` every
    # time) and cleared `concurrent.futures.thread._threads_queues` (which
    # only ever robs `_python_exit` of the sentinels that let idle executor
    # threads finish). `os._exit` skipped Textual's `on_unmount` entirely --
    # no database closed, no transaction rolled back, and any row already
    # flipped to `running` stranded there permanently. The handlers below
    # run the ordinary shutdown path and keep a hard exit only as the
    # bounded, last-resort escape. See `Utils/app_shutdown.py`.
    install_termination_handlers()

    # Initialize logging first
    initialize_early_logging()

    try:
        load_cli_config_and_ensure_existence()
    except Exception as e_cfg_main:
        logging.error(
            f"Could not ensure creation of effective config file: {e_cfg_main}",
            exc_info=True,
        )

    # --- Emoji Check ---
    emoji_is_supported = supports_emoji()  # Call it once
    loguru_logger.info(f"Terminal emoji support detected: {emoji_is_supported}")
    loguru_logger.info(
        f"Using brain: {get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}"
    )
    loguru_logger.info("-" * 30)

    # --- CSS File Handling ---
    package_root = Path(__file__).parent
    if _is_source_tree(package_root):
        try:
            css_dir = package_root / "css"
            css_dir.mkdir(exist_ok=True)

            # Check if modular CSS needs to be built
            build_script_path = css_dir / "build_css.py"

            # Check whether any input -- a .tcss module or a Python module
            # carrying BUNDLED_CSS -- has moved on since the last build.
            should_rebuild, reason = _generated_css_is_stale(package_root)
            if should_rebuild:
                logging.info("Generated CSS is stale during CLI entry; rebuilding")

            if should_rebuild and build_script_path.exists():
                logging.info("Building modular CSS...")
                import subprocess

                # Build CSS synchronously before starting the app
                result = subprocess.run(
                    [sys.executable, str(build_script_path)],
                    cwd=str(css_dir),
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    logging.info("Successfully built modular CSS")
                else:
                    logging.error(f"Failed to build modular CSS: {result.stderr}")

        except Exception as e_css_main:
            logging.error(f"Error handling CSS file: {e_css_main}", exc_info=True)

    # Parse command line arguments
    args = _build_arg_parser().parse_args()

    # If --serve flag is provided, run as web server
    if args.serve:
        # Check if web server dependencies are available
        from .Web_Server.serve import check_web_server_available, run_web_server

        if not check_web_server_available():
            loguru_logger.error("\n" + "=" * 60)
            loguru_logger.error("Web server feature is not available!")
            loguru_logger.error("=" * 60)
            loguru_logger.error(
                "\nThe required dependency 'textual-serve' is not installed."
            )
            loguru_logger.error("\nTo install it, run:")
            loguru_logger.error("  pip install tldw_chatbook[web]")
            loguru_logger.error("\nFor development installations:")
            loguru_logger.error('  pip install -e ".[web]"')
            loguru_logger.error("\n" + "=" * 60 + "\n")
            return

        loguru_logger.info("Starting tldw_chatbook in web server mode")
        run_web_server(
            host=args.host, port=args.port, title=args.web_title, debug=args.debug
        )
        return  # Exit after web server stops

    # Otherwise, run as normal TUI app
    # task-1650: resolve textual_image's rendering protocol NOW, while the
    # terminal still answers escape queries. Textual takes raw mode in
    # run() below, after which the query silently fails and every image
    # surface degrades to half-cell rendering.
    from .Utils.terminal_utils import warm_up_image_protocol

    warm_up_image_protocol()

    # task-21100: pending ChaChaNotes migrations replay inside TldwCli's
    # constructor, before anything can paint -- the terminal is the only
    # surface that exists at this phase, so say what the pause is there.
    from .Utils.db_upgrade_notice import print_db_upgrade_notice_if_pending

    print_db_upgrade_notice_if_pending()

    # Create instance with early logging flag
    app_instance = TldwCli()
    app_instance._cli_focus_override = bool(args.focus)
    # Set the early logging flag so _setup_logging knows logging was already initialized
    app_instance._early_logging_initialized = True
    try:
        app_instance.run()
    except KeyboardInterrupt:
        loguru_logger.info("--- KeyboardInterrupt received ---")
    except Exception:
        loguru_logger.exception("--- CRITICAL ERROR DURING app.run() ---")
        traceback.print_exc()  # Make sure traceback prints
    finally:
        # This might run even if app exits early internally in run()
        loguru_logger.info("--- FINALLY block after app.run() ---")
        # Bound interpreter teardown (see the identical call in the
        # `__main__` block for why this is the last placeable bound).
        arm_exit_watchdog(reason="interpreter exit")

    loguru_logger.info("--- AFTER app.run() call (if not crashed hard) ---")


#
# End of app.py
#######################################################################################################################
