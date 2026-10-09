# ADR-126: fence recovery and enroll before any runtime/config imports.
from tldw_chatbook.Backup_Recovery.storage_admission import admit_startup
if __name__ == "__main__": __import__("tldw_chatbook.Utils.launch_options").Utils.launch_options.adopt_config_flag()  # noqa: E701 -- TASK-34100.16: `--config PATH` picks the profile the fence admits; one line keeps app.py's size row
admit_startup()
if __name__ == "__main__":
    # TASK-34100.4: `python -m tldw_chatbook.app` unlocks through the same
    # pre-TUI startup unlock as `tldw-cli`, and at the same point: before this
    # module imports config, which would otherwise load the still-encrypted
    # file (and warn) ahead of the prompt. Spawn workers (`__mp_main__`) skip it.
    # The terminal is quieted first, as `tldw-cli` does before its unlock:
    # installing the password imports config, whose DEBUG/INFO wall otherwise
    # printed between the prompt and the app (TASK-34100.4 review round 2).
    from tldw_chatbook.Utils.startup_logging import quiet_startup_stderr

    quiet_startup_stderr()
    from tldw_chatbook.Backup_Recovery.launcher import startup_unlock

    _unlock_exit = startup_unlock()
    if _unlock_exit is not None:
        raise SystemExit(_unlock_exit)

# tldw_cli - Textual CLI for LLMs
# Description: This file contains the main application logic for the tldw_cli, a Textual-based CLI for interacting with various LLM APIs.
#
# Disable progress bars early to prevent interference with TUI
import os
from typing import ClassVar

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

# TASK-21147 (UAT G-7): when this module IS the entry point
# (``python -m tldw_chatbook.app``), cap terminal logging at WARNING
# before the heavy import chain below emits its DEBUG/INFO wall — a cold
# start's first paint must not be internal debug spew. The packaged CLI
# entry (tldw_chatbook.cli) makes the same call before importing us;
# TLDW_VERBOSE_STARTUP=1 restores the historical verbose startup.
if __name__ == "__main__":
    from tldw_chatbook.Utils.startup_logging import quiet_startup_stderr

    quiet_startup_stderr()

# Imports
import concurrent.futures
import inspect
import logging
import logging.handlers
import random
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Optional, Any, Dict, List, Callable, Iterable, Mapping
from textual.widget import Widget

#
# 3rd-Party Libraries
import asyncio
from loguru import logger as loguru_logger, logger
from textual import on
from textual.app import App, ComposeResult, ScreenStackError
from textual.events import AppFocus, Resize
from textual.widgets import RichLog
from textual.containers import Container
from textual.reactive import reactive
from textual.worker import Worker
from textual.binding import Binding
from textual.timer import Timer
from textual.css.query import NoMatches, QueryError

# Install the ordered-candidate fast path on `Stylesheet.apply` before any App
# exists, so every style application in the process takes it. Upstream's apply
# walks the whole rule list per node to recover source order, which made CSS
# matching the #1 sampled frame during 399 ms screen-switch stalls (2026-08-29
# holistic perf review). Idempotent; see the module for the A/B numbers and
# Tests/Performance/test_textual_css_fastpath.py for the fidelity guard.
from tldw_chatbook.Utils.textual_css_fastpath import install_stylesheet_fastpath

install_stylesheet_fastpath()

from pathlib import Path

from tldw_chatbook.Chat.citation_artifact_ownership import (
    CitationArtifactOwnershipCoordinator,
)
from tldw_chatbook.Chat.console_image_edit_operations import (
    ImageEditOperationRegistry,
)
from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_settings_defaults import ConsoleDefaultDurabilityState
from tldw_chatbook.Chat.console_settings_durability import (
    ConsoleSettingsDurabilityOwner,
)
from tldw_chatbook.Constants import (
    ALL_TABS,
    DEFAULT_SPLASH_DURATION_SECONDS,
    LIBRARY_NAV_CONTEXT_INGEST,
    LIBRARY_NAV_CONTEXT_MODE,
    LIBRARY_NAV_CONTEXT_NOTES_CREATE,
    TAB_CHAT,
    TAB_EVALS,
    TAB_HOME,
    TAB_LIBRARY,
    TAB_SCHEDULES,
    TAB_SETTINGS,
    TAB_WATCHLISTS_COLLECTIONS,
    TAB_WORKFLOWS,
    WIDE_VIEWPORT_COLUMNS,
)
from tldw_chatbook.css import build_css
from tldw_chatbook.css.Themes.themes import ALL_THEMES, ThemeVariableDefaultsMixin
from tldw_chatbook.css.tie_aware_stylesheet import TieAwareStylesheet
from tldw_chatbook.DB.Client_Media_DB_v2 import (
    MediaDatabase,
)
from tldw_chatbook.config import CLI_APP_CLIENT_ID
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Chatbooks import LocalChatbookService
from tldw_chatbook.Home.active_work_adapter import (
    HomeControlAction,
    HomeControlResult,
    UnavailableHomeActiveWorkAdapter,
)

# TASK-33011: the Library ingest queue (``LibraryIngestQueueMixin`` and its
# module-level helpers) lives in ``app_ingest_queue.py``. It is a base class of
# ``TldwCli``, so it is imported eagerly. The private helpers are re-exported
# for existing ``from tldw_chatbook.app import ...`` callers; patch the names
# the moved code reads on ``app_ingest_queue``, never here.
from tldw_chatbook.app_ingest_queue import (
    LibraryIngestQueueMixin,
    _accepts_keyword,  # noqa: F401 -- re-export
    _IngestParsePoolResources,  # noqa: F401 -- re-export
    _library_ingest_done_progress,  # noqa: F401 -- re-export
    _library_ingest_write_failure_category,  # noqa: F401 -- re-export
    _template_resolution_errors,  # noqa: F401 -- re-export
)
from tldw_chatbook.Logging_Config import RichLogHandler

# from tldw_chatbook.css.css_loader import load_modular_css  # Removed - reverting to original CSS
from tldw_chatbook.Metrics.metrics import (
    log_counter,
    log_histogram,
    log_resource_usage,
)
from tldw_chatbook.Prompt_Management import (
    Prompts_Interop as prompts_interop,
)
from tldw_chatbook.TTS import TTSProfileService
from tldw_chatbook.TTS.adapter_bootstrap import build_default_tts_service
from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
    AudioCppArtifactLeaseCoordinator,
)
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_types import ProfileRepositoryState
from tldw_chatbook.Utils.app_shutdown import (
    register_running_app,
)
from tldw_chatbook.Utils.boot_worker_policy import (
    BOOT_WORKER_KEY_BY_IDENTITY,
    MAX_CONCURRENT_STAGGERED_BOOT_WORKERS,
    STAGGERED_BOOT_WORKER_KEYS,
    StaggeredBootWorkerGate,
)
from tldw_chatbook.Utils.db_status_manager import DBStatusManager
from tldw_chatbook.Utils.instance_lock import (
    InstanceLockStatus,
    acquire_profile_instance_lock,
)
from tldw_chatbook.Utils.persistent_diagnostics import persist_event
from tldw_chatbook.Utils.text_selection_crash_guard import TextualAppGuards
from tldw_chatbook.Utils.ui_responsiveness import (
    UIResponsivenessMonitor,
    freeze_long_lived_heap,
)

#
# --- Local API library Imports ---
from .config import (
    first_profile_created_this_session,
    get_cli_setting,
    get_media_db_path,
    get_prompts_db_path,
    get_subscriptions_db_path,  # noqa: F401 - shared app-module compatibility alias.
    get_tts_profiles_db_path,
    get_user_data_dir,
    save_setting_to_cli_config,
)
from .Logging_Config import (
    configure_application_logging,
    sync_loguru_forward_level,
)

# TASK-21108: `TTS/voice_bundle_service` (1,857 lines) is imported
# function-locally in `_ensure_tts_voice_bundle_service` -- the only place
# that constructs it, on first use, long after first paint. The name below is
# TYPE_CHECKING-only, so every annotation that mentions it must stay a string
# (app.py has no `from __future__ import annotations`, and PEP 526
# annotations on attribute targets ARE evaluated at runtime).
if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.voice_bundle_service import (
        TTSVoiceBundlePortabilityService,
    )
from tldw_chatbook.TTS.TTS_Generation import (
    bind_tts_service,
)
from tldw_chatbook.Event_Handlers.worker_handlers import (
    WorkerHandlerRegistry,
    MiscWorkerHandler,
)
from .config import (
    load_settings,
    get_cli_providers_and_models,
    get_config_load_failure,
    get_config_schema_conflict,
)
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSGlobalOverrideDecisionEvent,
    TTSMessageSpeechRequestEvent,
    TTSRequestEvent,
    TTSCompleteEvent,
    TTSPlaybackEvent,
    TTSProgressEvent,
)
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSPlaygroundGenerateEvent,
    STTSProviderConfigurationChanged,
    STTSSettingsSaveEvent,
    STTSAudioBookGenerateEvent,
)
from .Notes.Notes_Library import NotesInteropService
from .Notes.file_notes_git_service import build_file_notes_session_owner
from .Notes.note_folder_repository import LocalNoteFolderRepository
from .Notes.notes_scope_service import NotesScopeService

# TASK-21108: `notes_sync_runtime` (and `notes_sync_legacy`, which the
# TASK-21112 start gate reads) are imported inside
# `_construct_notes_sync_runtime_owner`, the single place that needs them, so the
# lasting-sync chain leaves the app import closure. The name below is
# TYPE_CHECKING-only: annotations mentioning it must stay strings.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from .Notes.notes_sync_runtime import NotesSyncRuntimeOwner
from .Notes.server_notes_workspace_service import ServerNotesWorkspaceService
from .Character_Chat.local_character_persona_service import LocalCharacterPersonaService
from .Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService

# Persona_Buddy is deliberately NOT imported at module scope (TASK-21103):
# its controller drags Persona_Visual and PIL (1.28 s cold) onto the boot
# path. See the lazy persona_buddy_controller property.
from .RAG_Admin.local_rag_admin_service import LocalRAGAdminService
from .RAG_Admin.rag_admin_scope_service import RAGAdminScopeService
from .RAG_Admin.server_rag_admin_service import ServerRAGAdminService
from .Scheduling.constants import (
    HANDLER_TIMEOUT_SECONDS,
    SCHEDULER_POLL_INTERVAL_SECONDS,
)
from .ACP_Interop.runtime_process import ACPRuntimeProcessManager
from tldw_chatbook.Widgets.glyph_fallback import set_ascii_glyph_mode
from .Widgets.AppFooterStatus import AppFooterStatus
from .Widgets.splash_screen import SplashScreen
from tldw_chatbook.config import (
    get_chachanotes_db_path,
    settings,
    get_chachanotes_db_lazy,
    seed_builtin_content,
)
from .UI.Navigation.main_navigation import MainNavigationBar, NavigateToScreen
from .UI.Navigation.audio_cpp_model_handoff import AudioCppModelInstallOwner
from .UI.Navigation.pending_handoff_store import (
    HandoffChannel,
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
    resolve_screen_route,
    screen_load_error,
)
from .UI.Navigation.shell_destinations import (
    ARTIFACTS_COMPATIBILITY_SHORTCUT,
    SHELL_DESTINATION_ORDER,
    SHELL_DESTINATION_SHORTCUTS,
)
from .UI.Workbench.help import WorkbenchHelpPanel, WorkbenchHelpState

# task-24458: import the MESSAGE, not the deprecated window. Importing
# `Tools_Settings_Window` here dragged `Agents.local_tool_provider` ->
# `Tools.workspace_tool_executor` and 7 further modules onto the boot
# import path for a window that is nav-unreachable (TASK-1346).
from .UI.console_command_provider import ConsoleCommandProvider  # noqa: E402
from .UI.image_gen_command_provider import ImageGenCommandProvider  # noqa: E402
from tldw_chatbook.LLM_Provider_Catalog.model_auto_refresh import ModelCatalogRefreshed  # noqa: E402
from tldw_chatbook.Media import (  # noqa: E402
    LocalMediaReadingService,
    MediaReadingScopeService,
    ServerMediaReadingService,
)
from tldw_chatbook.Prompt_Management.prompt_scope_service import (  # noqa: E402
    build_prompt_scope_service,
)

# NOTE (boot budget, ADR-097): `Workspaces.agent_provisioning` is imported
# lazily inside `_wire_workspace_agent_provisioning` (in
# `app_service_wiring.py`; itself deferred to a post-ready timer) so it stays
# out of the UI-ready module census. `Subscriptions.fts_backfill`,
# `UI.stable_command_palette` and `Event_Handlers.worker_events` are imported
# inside their one post-ready user for the same reason (TASK-33011).
from tldw_chatbook.Subscriptions.watchlists_operation_coordinator import (  # noqa: E402
    WatchlistsOperationCoordinator,
)
from tldw_chatbook.Evaluations_Interop import (  # noqa: E402
    LocalEvaluationsService,
)
from tldw_chatbook.runtime_policy.bootstrap import (  # noqa: E402
    load_runtime_policy_for_app,
    set_authoritative_runtime_source,
)
from tldw_chatbook.runtime_policy.engine import PolicyEngine  # noqa: E402
from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer  # noqa: E402
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY  # noqa: E402
from tldw_chatbook.runtime_policy.types import PolicyDecision, RuntimeSourceState  # noqa: E402

# TASK-33011: TldwCli's service composition (the lazy service properties, the
# ``_wire_*`` methods and their Notes-sync/Collections-capture helpers) lives in
# ``app_service_wiring.py``. It is a base class of ``TldwCli``, so it is imported
# eagerly -- last, once every module it needs is loaded. Patch the names the
# moved code reads on ``app_service_wiring`` (on both modules where app.py still
# reads the name), never on this module alone.
from tldw_chatbook.app_service_wiring import (
    ServiceWiringMixin,
    _build_terminal_backend,  # noqa: F401 -- re-export
    _extract_collections_article,  # noqa: F401 -- re-export
    _install_deferred_notes_sync_facades,
    _read_app_raw_cli_permitted,
    _resolve_collections_media_reference,  # noqa: F401 -- re-export
    _resolve_collections_note_reference,  # noqa: F401 -- re-export
    _wire_notes_sync_services,
)

# TASK-33011: TldwCli's lifecycle, shutdown and quit flow lives in
# ``app_lifecycle.py`` (``LifecycleMixin``), a base class of ``TldwCli``.
from tldw_chatbook.app_lifecycle import LifecycleMixin, _DIAGNOSTICS_COMPONENT_APP
from tldw_chatbook.app_navigation import NavigationMixin

# TASK-33011: the command-palette providers (and their key-display helpers) live in
# ``app_command_providers.py``. ``TldwCli.COMMANDS`` names the classes, so they are
# imported eagerly and re-exported here for callers that import them from app.py.
from tldw_chatbook.app_command_providers import (
    FOCUS_TOGGLE_PALETTE_ENTRY,
    CharacterProvider,
    DeveloperProvider,
    LibraryIngestProvider,
    MediaProvider,
    PatternGalleryProvider,
    QuickActionsProvider,
    SettingsProvider,
    SetupWizardProvider,
    TabNavigationProvider,
    ThemeProvider,
    _bindings_to_shortcuts,
)
from tldw_chatbook.app_feature_glue import (
    FeatureGlueMixin,
    setup_owns_startup_networking,  # noqa: F401 -- re-export
)

if TYPE_CHECKING:
    from tldw_chatbook.Terminal.backend import TerminalBackend
    from tldw_chatbook.Terminal.session_manager import TerminalSessionManager
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
else:
    TerminalBackend = Any

# Annotation-only for the cluster D stubs (TASK-33011); the bodies that use
# these at runtime live in ``app_destinations``.
if TYPE_CHECKING:
    from .ACP_Interop.runtime_session import ACPRuntimeSessionState
    from .Chat.chat_handoff_models import ChatHandoffPayload
    from .Prompt_Management.prompt_variables import PromptVariableApplication
    from .UI.Screens.study_scope_models import StudyScopeContext

# Annotation-only for the speech stubs (TASK-33011 PR-E); the bodies that use
# these at runtime live in ``app_speech``.
if TYPE_CHECKING:
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalEvidence,
        AudioCppManagedConsumerIdentity,
        AudioCppModelLibraryObservationSnapshot,
    )
    from tldw_chatbook.TTS.audio_cpp_guided_config import AudioCppSettingsConfig
    from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot

_PERSONAL_CONTEXT_SERVICE_BOOTSTRAP_LOCK = threading.Lock()


def _destinations():
    """Import ``app_destinations`` on first use (never at boot; ADR-097).

    TldwCli's destination launchers, handoffs, Home controls, Roleplay
    character-conversation activation and Personal Context launchers are thin
    stubs that delegate here (TASK-33011). Tests that patch a module-level
    name those bodies use must patch it on ``tldw_chatbook.app_destinations``.
    """
    from tldw_chatbook import app_destinations

    return app_destinations


def _speech():
    """Import ``app_speech`` on first use (never before ``_ui_ready``; ADR-097).

    TldwCli's TTS/STTS handlers, speech resource owners, speech
    initialization and delivery admission are thin stubs that delegate here
    (TASK-33011). Tests that patch a module-level name those bodies use must
    patch it on ``tldw_chatbook.app_speech``.
    """
    from tldw_chatbook import app_speech

    return app_speech


DEFERRED_AUDIO_SERVICE_DELAY_SECONDS = 0.1
#: Collections capture persistence and remote adapters are first-use work.
#: Compose them after the first interactive frame, or synchronously when a
#: caller enters Collections before this timer fires.
DEFERRED_COLLECTIONS_CAPTURE_WIRING_DELAY_SECONDS = 0.1
#: Notes organization composition is not needed for the first interactive
#: frame. Give it the same explicit post-startup window as other idle
#: maintenance: a 0.1 s timer could expire while synchronous post-ready setup
#: was still running and race the ADR-097 UI-ready module census.
DEFERRED_NOTES_ORGANIZATION_WIRING_DELAY_SECONDS = 5.0
#: Workspace agent provisioning (task-8) deferral: after `_ui_ready` so
#: `Workspaces.agent_provisioning` stays out of the UI-ready census
#: (ADR-097); same 0.1-0.2 s non-essential-startup window as audio.
DEFERRED_WORKSPACE_AGENT_PROVISIONING_DELAY_SECONDS = 0.2
DEFERRED_DB_SIZE_UPDATE_DELAY_SECONDS = 0.1


# TASK-22215: how often the staggered boot fleet reconciles its admission
# slots against the workers actually holding them. This is a BACKSTOP for a
# terminal transition that never reaches `on_worker_state_changed`, not the
# primary mechanism -- so it is deliberately slow (it costs one dict walk over
# at most `MAX_CONCURRENT_STAGGERED_BOOT_WORKERS` entries) and stops itself the
# moment the gate drains. Without it, one lost event would strand every
# remaining member of the fleet for the whole session: exactly the failure a
# stagger policy must not introduce.
BOOT_WORKER_RECONCILE_INTERVAL_SECONDS = 2.0

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

# task-21110: the timer above cannot help the FIRST screen. With the splash
# enabled (the default) boot is strictly serial -- the splash owns the loop for
# its full duration, THEN the initial screen's module is imported
# synchronously on that same loop, and only after the screen is up does
# `_post_mount_setup` arm the deferred pre-importer above. So the initial
# route's module gets its own, much earlier kick: scheduled from `on_mount`
# while the splash is still on screen, onto the same daemon-thread mechanism.
#
# Why 0.2 and not 0. The splash animation ticks on the event loop at 20 Hz
# (`Widgets/splash_screen.py`, `animation_speed` default 0.05s) and the import
# thread holds the GIL, so this trades a little splash smoothness for a lot of
# boot time. Measured, interleaved arms x10 boots, isolated profile, M-series
# (frames = animation frames rendered during a 1.5s splash, ideal 30):
#
#   arm      frames  worst gap  p95 gap  gaps>100ms/10 boots  close->usable
#   no warm    30      51.0ms    50.9ms          0               1.410s
#   0.0s       28     111.5ms    69.8ms          6               1.106s
#   0.2s       30      86.8ms    52.9ms          2               1.083s
#   0.5s       30      83.6ms    51.8ms          2               1.087s
#
# 0.2s recovers the dropped frames and nearly all of the p95 that a 0s start
# costs, for no measurable boot-time difference. 0.5s is no better and eats
# overlap headroom that the case with the most to gain cannot spare: on a
# first boot after an upgrade the import is bytecode-compiling and takes
# ~0.98s, which fits inside the splash from 0.2s but not from 0.5s.
SPLASH_INITIAL_SCREEN_PREIMPORT_DELAY_SECONDS = 0.2

# Chat/Library/Settings are the three screens the audit measured as
# multi-thousand-line modules -- import them first so a thread that gets cut
# short (app quit shortly after startup) still banked the highest-value work
# before spending time on the rest of the registry.
#
# TASK-21113 considered reordering this to start with the CONFIGURED DEFAULT
# TAB and measured the idea dead: the whole-registry pass is armed by
# `_schedule_deferred_startup_work()`, the last statement of
# `_post_mount_setup()`, and BOTH boot paths run `_push_initial_screen()` to
# completion first (`_run_no_splash_post_mount_setup` awaits them in order;
# the splash path pushes, then `call_after_refresh(self._post_mount_setup)`).
# So the configured default tab's module is always already in `sys.modules`
# before this list is consulted -- and if the initial push raised, this pass
# never runs at all. Reordering would have moved a `sys.modules` dict hit.
#
# TASK-22214 considered the opposite reordering -- biggest routes LAST, so
# the first seconds after mount only carry the 18 cheap (~5-20 ms) routes --
# and rejected it: the pre-import exists to protect exactly the first click
# to Library/Settings, and pushing their imports minutes of route-list later
# widens the window where that click pays a synchronous import on the event
# loop (the thing this machinery removes). Heavy-first costs little under
# proportional pacing: chat is a dict hit at pass time, so its gap is ~0 and
# library/settings are warm within the pass's first ~0.5 s warm.
SCREEN_PREIMPORT_PRIORITY_ROUTE_IDS: tuple[str, ...] = ("chat", "library", "settings")

# TASK-21113 pacing for the whole-registry pre-importer. The pass is a
# GIL-holding CPU burst on a daemon thread; on a 1-2 core machine the event
# loop is sharing a core with it for the whole post-boot window. Measured on
# a fast M-series with the initial screen already warm (the real boot
# condition, see above): 21 routes, **361 ms** total, of which library
# 110.6 ms + settings 95.9 ms + personas 82.3 ms are **80%** -- the other 18
# routes cost 72.8 ms between them.
#
# That skew is what these constants answer. A flat inter-route sleep would
# treat a 0.5 ms module exactly like a 110 ms one, so the gap is instead
# proportional to the time the previous import just took: hand the event loop
# back (ratio x) what was just taken from it, capped. On the numbers above
# that inserts ~0.35 s of quiet across the pass, i.e. it stops being a
# continuous competitor and becomes a ~50%-duty-cycle one, at zero cost to
# anything that waits on it (nothing does).
#
# What a gap CANNOT do is subdivide a single `import_module`: those three
# 80-110 ms bursts are indivisible, and on constrained hardware they are the
# multi-hundred-millisecond stretches that actually hurt. That is what the
# low-core tier is for -- same mechanism, 3x the yield and a much higher cap,
# so a 400 ms import on a slow box is followed by ~1.2 s of quiet.
#
# TASK-22214 re-measured after the payload grew +99 modules / +74.5k LOC:
# the pass now warms 715 modules / 564,326 LOC beyond the app import (478 /
# 365,692 of it beyond app+chat, which is what the budget guard pins --
# Tests/Performance/test_screen_preimport_payload_budget.py). At that size
# the 0.10 s cap had quietly turned the proportional yield back INTO the
# flat sleep it was designed to replace: library alone costs 156-183 ms
# warm and 525-615 ms on a bytecode-compiling boot (M-series; slower
# hardware proportionally worse), so every heavy route asked for a
# cost-sized gap and got 0.10 s. Observed directly in the requested-gap
# series on a cold pass: BEFORE `[0.0, 0.1, 0.1, 0.002, 0.003, 0.1]` --
# clipped flat exactly on the expensive routes -- AFTER `[0.0, 0.529,
# 0.245, 0.003, 0.113, 0.303]`, tracking cost.
#
# So the caps moved from "binds on every heavy route" to "binds only on
# pathology". They are kept, rather than removed, purely as a boundedness
# guard: a pathological multi-second import (or a wild clock reading) must
# not strand the daemon thread in a minutes-long sleep. 2.0 s sits above
# the largest single-route cost measured on fast hardware with room for a
# slower box; 6.0 s is the same 3x multiple the low-core tier applies
# everywhere else.
#
# Measured, interleaved A/B in both orders with an A/A control first
# (in-pass GIL duty = import time / pass wall time, from a headless Pilot
# boot instrumented on both sides; n=2-4 per arm):
#
#   arm                     duty before   duty after   worst 1 s busy
#   normal tier, warm       49.7-58.0%    47.4-47.8%   wash (~465 ms both)
#   normal tier, cold       66.2-66.6%    47.8-48.5%   783 -> 681 ms
#   low-core tier, warm     23.4-23.5%    23.6-24.2%   WASH (overlapping)
#   low-core tier, cold     24.1-25.0%    23.7-24.1%   WASH (overlapping)
#
# Read honestly: the win is entirely on the NORMAL tier, and the low-core
# tier is a wash in both cache states -- at ratio 3.0 the old 1.5 s cap was
# already nearly non-binding (3 x 525 ms = 1.58 s), so raising it to 6.0 s
# clips one route's gap slightly less. That half is design hardening for
# hardware slower than anything measurable here, not a measured gain, and
# the A/A control (58.5% vs 59.8%) says the noise floor is ~1.5 points.
#
# The accepted cost is a longer total pass: warm 0.90-0.99 -> 1.14-1.24 s,
# cold 2.43-2.48 -> 3.43-3.51 s, i.e. the LAST route becomes warm ~254 ms
# (warm) / ~1.07 s (cold) later than before. Nothing waits on the pass, and
# first-navigation protection is deliberately not traded away: library is
# route #2 and its warm-at time is unchanged (351 -> 371 ms warm, 700 ->
# 693 ms cold), settings slips 499 -> 616 ms warm / 1152 -> 1510 ms cold,
# and a click landing MID-pass is measurably faster than before (Library
# first-nav at 0.35 s after ready: 63.5 -> 17.8 ms median), because the
# thread is now usually in a gap rather than mid-import. The gap sleep is
# sliced (see `_pause_between_preimports`) so a quit never waits one out.
SCREEN_PREIMPORT_YIELD_RATIO = 1.0
SCREEN_PREIMPORT_MAX_ROUTE_GAP_SECONDS = 2.0
SCREEN_PREIMPORT_LOW_CORE_YIELD_RATIO = 3.0
SCREEN_PREIMPORT_LOW_CORE_MAX_ROUTE_GAP_SECONDS = 6.0
# Below this many usable CPUs the pass is throttled rather than switched off:
# disabling it would push each screen's import back onto the event loop at
# first navigation, which is work the user has actually asked for, on the
# machines least able to absorb it. Throttling keeps the win and drops the
# pressure.
SCREEN_PREIMPORT_LOW_CORE_THRESHOLD = 4
# While a screen navigation holds `_screen_navigation_lock`, the event loop is
# doing its own import + compose + mount; the speculative pass steps aside
# until it finishes. Bounded, so a lock that is never released (a navigation
# blocked on a confirm dialog the user leaves open) throttles the pass instead
# of stranding it.
SCREEN_PREIMPORT_NAVIGATION_POLL_SECONDS = 0.05
SCREEN_PREIMPORT_NAVIGATION_PARK_LIMIT_SECONDS = 5.0
SCREEN_PREIMPORT_MAX_NAVIGATION_POLLS = max(
    1,
    round(
        SCREEN_PREIMPORT_NAVIGATION_PARK_LIMIT_SECONDS
        / SCREEN_PREIMPORT_NAVIGATION_POLL_SECONDS
    ),
)


def _usable_cpu_count() -> int:
    """How many CPUs this process may actually run on.

    Prefers the scheduler affinity mask where the platform has one (a
    container pinned to one core reports the host's core count from
    ``os.cpu_count()``), and falls back to ``os.cpu_count()``. Returns 1 when
    neither will answer -- the conservative direction here, since the only
    consequence of guessing low is that a background pass nothing waits on
    paces itself more politely.
    """
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


# Home's open-eval-runs feed queries pending and failed statuses separately;
# this cap bounds both queries (a count, not a listing -- anything beyond it
# still reads as "runs need attention").
_HOME_EVAL_RUN_QUERY_LIMIT = 50
#
#######################################################################################################################
#
# Functions:


# --- Global variable for config ---
APP_CONFIG = load_settings()

# Early logging configuration removed - handled by configure_application_logging() during app initialization


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


def _select_profile_database(notes_service: object | None) -> Any:
    """Return the seeded injected profile DB, or the seeded lazy global DB."""
    injected = getattr(notes_service, "db", None)
    return seed_builtin_content(injected) if injected else get_chachanotes_db_lazy()


class WideViewportTierMixin:
    """App-wide responsive wide tier: one class toggle for every modal.

    The Chat settings modal (PR #2670) and the Alt+M model popover
    (PR #2672) each shipped a private viewport-width tier with its own
    Python toggle. The repo-wide rollout replaces per-modal toggles with
    this single one: at >= 150 terminal columns the App gains the
    ``-wide-viewport`` CSS class, re-synced on every resize. Every widget
    is a descendant of the App, so one ``App.-wide-viewport #<modal-id>``
    selector per modal (see ``components/_agentic_terminal.tcss``) reaches
    every modal regardless of where its base geometry lives -- shared
    sheets or in-file ``DEFAULT_CSS`` -- because app CSS outranks widget
    ``DEFAULT_CSS`` and the descendant selector outspecifies each base
    rule. The tier keys off the app viewport, never a modal's own width:
    sizing a container from the container would oscillate.

    The threshold matches the two shipped per-surface tiers so all three
    mechanisms coexist; the shipped toggles stay (their geometry contracts
    pin them), and the shared tier reproduces their values.
    """

    #: Single source of truth: ``tldw_chatbook.Constants.WIDE_VIEWPORT_COLUMNS``
    #: (the class attribute keeps the threshold reachable from the mixin and
    #: re-binds the shared constant so tests and app cannot drift).
    WIDE_VIEWPORT_COLUMNS = WIDE_VIEWPORT_COLUMNS

    def on_resize(self, event: Resize) -> None:
        """Re-sync the wide tier as the terminal resizes.

        Args:
            event: Terminal resize event; ``event.size.width`` supplies the
                new viewport width compared against ``WIDE_VIEWPORT_COLUMNS``
                to add or remove the ``-wide-viewport`` class on the App.
        """
        self.set_class(
            event.size.width >= self.WIDE_VIEWPORT_COLUMNS,
            "-wide-viewport",
        )


class TldwCli(
    # TextualAppGuards sits before App: its TextSelectionCrashGuard on_event
    # wrapper is the last defense against Textual 8.x's text-selection
    # MouseDown crash on a mid-recompose widget (task-14903; its docstring
    # names the ONE signature). It also owns the loop's default executor
    # (ThreadWorkerContextGuard, TASK-33264).
    WideViewportTierMixin,
    TextualAppGuards,
    LibraryIngestQueueMixin,
    ServiceWiringMixin,
    LifecycleMixin,
    NavigationMixin,
    FeatureGlueMixin,
    ThemeVariableDefaultsMixin,  # TASK-33003.6 ruling 15: guard-name fallback
    App[None],
):  # Specify return type for run() if needed, None is common
    """A Textual app for interacting with LLMs."""

    _runtime_policy_projection_snapshot: tuple[str, str | None] = ("local", None)

    def action_command_palette(self) -> None:
        """Open the app's stable Textual command palette."""
        from .UI.stable_command_palette import StableCommandPalette

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
        # ADR-161 task 10: the console vocabulary (previously the
        # TASK-25812 console sheet, which always rode this boot parse
        # because the Console is the initial tab) now rides the bundle
        # itself via features/_console{,_panels}.tcss -- one fewer boot
        # source, no duplicated variable preamble, and the first Console
        # mount stays free of restyle work exactly as before.
        str(build_css.screen_css_paths(Path(__file__).parent / "css")[1]),
    ]

    def _stamp_new_profile_library_lifecycle(self) -> None:
        """Persist the Library lifecycle at profile CREATION (task-32059).

        ``coerce_library_lifecycle`` reads an absent
        ``[library.rail_state] lifecycle`` as ``expanded`` for any profile the
        current run did not create -- so a user who completed first-run setup,
        quit, and relaunched before ever opening Library never saw the
        documented compact "Get started" rail. Writing ``unknown`` here (the
        same value the screen would have derived on a first visit in THIS run)
        makes the second launch read the fact instead of inferring it from a
        missing key. A profile that already carries a lifecycle is untouched.
        """
        library_config = self.app_config.get("library")
        if not isinstance(library_config, dict):
            library_config = {}
            self.app_config["library"] = library_config
        rail_state = library_config.get("rail_state")
        if not isinstance(rail_state, dict):
            rail_state = {}
            library_config["rail_state"] = rail_state
        if "lifecycle" in rail_state:
            return
        # One name for the value both the in-memory state and the file get:
        # "unknown" is LibraryLifecycle.UNKNOWN.value, spelled out rather than
        # imported: pulling the Library package in here would put its modules
        # on every boot for one string (see the module-census ratchet).
        lifecycle = "unknown"
        rail_state["lifecycle"] = lifecycle
        try:
            # `save_setting_to_cli_config` REPORTS a write failure rather than
            # raising it, and an unstamped profile reads back as `expanded` on
            # the next launch -- so the return value is the failure signal.
            saved = save_setting_to_cli_config(
                "library.rail_state", "lifecycle", lifecycle
            )
        except Exception:
            saved = False
        if not saved:
            # A profile whose config cannot be written still gets the correct
            # in-memory lifecycle for this run; boot must not fail over it.
            logger.warning("Could not stamp the Library lifecycle for a new profile.")

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

    # Shell shortcuts are keyed by stable destination ID so inserting a new
    # destination cannot transfer an existing shortcut to another screen.
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit App", show=True, priority=True),  # ADR-031
        Binding("ctrl+p", "command_palette", "Palette Menu", show=True),
        Binding("f1", "show_workbench_help", "Help", show=True),
        Binding("f6", "focus_next_workbench_pane", "Next Pane", show=True),
        # ADR-172: preserve muscle memory after Artifacts folds into Library.
        Binding(
            ARTIFACTS_COMPATIBILITY_SHORTCUT,
            "library_artifacts",
            "Library Artifacts",
            show=False,
        ),
        Binding(
            "ctrl+shift+f",
            FOCUS_TOGGLE_PALETTE_ENTRY[1],
            "Focus Mode",
            show=False,
        ),
    ] + [
        Binding(
            SHELL_DESTINATION_SHORTCUTS[destination.destination_id],
            f"shell_destination({destination.destination_id!r})",
            f"Go to {destination.accessible_label}",
            show=SHELL_DESTINATION_SHORTCUTS[destination.destination_id].startswith(
                "f"
            ),
        )
        for destination in SHELL_DESTINATION_ORDER
        if destination.destination_id != "artifacts"
    ]
    COMMANDS = App.COMMANDS | {
        ThemeProvider,
        TabNavigationProvider,
        QuickActionsProvider,
        SettingsProvider,
        CharacterProvider,
        MediaProvider,
        LibraryIngestProvider,
        SetupWizardProvider,
        DeveloperProvider,
        ConsoleCommandProvider,
        ImageGenCommandProvider,
        PatternGalleryProvider,
    }

    # T169: "notes-window" removed -- no widget composes that id anymore (the
    # standalone Notes tab / Notes_Window.py it belonged to is gone, replaced
    # by the Library workbench's Notes canvas), confirmed via
    # `grep -rn 'id="notes-window"' tldw_chatbook/`.

    # Define reactive at class level with a placeholder default and type hint
    current_tab: reactive[str] = reactive("")

    # Splash screen state
    splash_screen_active: reactive[bool] = reactive(False)
    _splash_screen_widget: Optional[SplashScreen] = None

    # --- REACTIVES FOR PROVIDER SELECTS ---
    # Initialize with a dummy value or fetch default from config here
    # Ensure the initial value matches what's set in compose/settings_sidebar
    # Fetching default provider from config:

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
    # DB size update timer moved to DBStatusManager; the 10 s token-count
    # timer that used to live here was retired by task-21133 (its consumer
    # surface went with task-17653).
    ui_responsiveness_monitor: UIResponsivenessMonitor | None = None
    _ui_responsiveness_heartbeat_timer: Optional[Timer] = None

    # Media services and type catalog
    _media_types_for_ui: List[str] = []

    media_db: Optional[MediaDatabase] = None
    selected_note_files_for_import: List[Path]
    parsed_notes_for_preview: List[Dict[str, Any]] = []
    last_note_import_dir: Optional[Path] = None
    # Add attributes to hold the handlers (optional, but can be useful)

    llamacpp_server_process: Optional[subprocess.Popen] = None
    llamafile_server_process: Optional[subprocess.Popen] = None
    vllm_server_process: Optional[subprocess.Popen] = None
    ollama_server_process: Optional[subprocess.Popen] = None
    mlx_server_process: Optional[subprocess.Popen] = None
    onnx_server_process: Optional[subprocess.Popen] = None

    # User ID for notes, will be initialized in __init__
    current_user_id: str = "default_user"  # Will be overridden by self.notes_user_id

    def __init__(self):
        # Track startup timing
        self._startup_start_time = time.perf_counter()
        self._startup_phases = {}
        # Real per-task durations of the phase-3 parallel initializers,
        # stamped on the worker thread by `_timed_init_task` (TASK-21111).
        self._startup_parallel_tasks: dict[str, float] = {}
        # Backing slots for the lazily-resolved credential store
        # (TASK-21111(b)); see the `server_credential_store` property.
        self._server_credential_store: Any | None = None
        self._server_credential_store_unavailable_reason: str | None = None

        # Tab switching optimization

        # Reduce logging in production
        if not os.environ.get("TLDW_DEBUG"):
            logging.getLogger().setLevel(
                logging.INFO
            )  # Reduce to INFO level in production
            # Disable debug logging for performance
            logging.getLogger("tldw_chatbook").setLevel(logging.INFO)
            # ...which reaches loguru only once its forwarder is re-levelled:
            # at TRACE every dropped debug call still cost ~7-8 us (PERF-03).
            sync_loguru_forward_level()

        # Log initial memory usage only in debug mode
        if os.environ.get("TLDW_DEBUG"):
            log_resource_usage()
        log_counter(
            "app_startup_initiated", 1, documentation="Application startup initiated"
        )

        super().__init__()

        # A textual-serve child receives a one-use, per-AppService control
        # capability through its spawn environment. A non-secret launch marker
        # persists without a broker; native launches need no served transport.
        canvas_control_keys = (
            "CHATBOOK_CANVAS_CONTROL_HOST",
            "CHATBOOK_CANVAS_CONTROL_PORT",
            "CHATBOOK_CANVAS_CONTROL_CHILD_ID",
            "CHATBOOK_CANVAS_CONTROL_SECRET",
            "CHATBOOK_CANVAS_CONTROL_VERSION",
        )
        self.served_canvas_handler = None
        self.served_canvas_control = None
        self._served_canvas_mode = "CHATBOOK_SERVED_CHILD" in os.environ or any(
            key in os.environ for key in canvas_control_keys
        )
        if self._served_canvas_mode:
            from .Canvas.control_protocol import (
                CanvasControlClient,
                ControlProtocolError,
            )
            from .Canvas.gateway import ServedCanvasControlHandler
            from .Canvas.profiles import (
                load_application_profile_snapshot,
                runtime_snapshot_id,
            )

            self._canvas_profile_snapshot = load_application_profile_snapshot()
            self.served_canvas_handler = ServedCanvasControlHandler()
            try:
                self.served_canvas_control = CanvasControlClient.from_environment(
                    os.environ,
                    runtime_snapshot_id=runtime_snapshot_id(
                        self._canvas_profile_snapshot
                    ),
                    handler=self.served_canvas_handler.handle,
                )
            except ControlProtocolError:
                loguru_logger.warning(
                    "Served Canvas control disabled code=invalid_spawn_environment"
                )
        self._served_canvas_control_start_task: asyncio.Task[None] | None = None

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
        self.raw_cli_runtime = RawCliRuntime(lambda: _read_app_raw_cli_permitted(self))
        self._raw_cli_runtime_shutdown_task: asyncio.Task[Any] | None = None
        # App-owned, but first-use: importing Terminal here spent three
        # first-paint modules before a user opened or armed a session.
        self._terminal_session_manager: "TerminalSessionManager | None" = None
        self._terminal_session_manager_lock = threading.Lock()
        self._terminal_session_manager_shutdown_task: asyncio.Task[None] | None = None
        # Default-save failures belong to the application lifetime rather
        # than whichever Console screen happens to be mounted.  New-chat
        # generation advances only after a Make Default intent is fully
        # published into this running process.
        self.console_default_durability_state = ConsoleDefaultDurabilityState()
        self.console_new_chat_default_generation = 0
        self.console_settings_durability_owner = ConsoleSettingsDurabilityOwner()
        self.console_settings_durability_tasks = (
            self.console_settings_durability_owner.tasks
        )
        self.console_default_recovery_inflight: set[tuple[int, str]] = set()
        self.library_new_profile_admission = first_profile_created_this_session()
        if self.library_new_profile_admission:
            self._stamp_new_profile_library_lifecycle()
        self.console_image_edit_operations = ImageEditOperationRegistry()
        self._console_image_edit_shutdown_task: asyncio.Task[None] | None = None
        self._backup_maintenance_monitor_task: asyncio.Task[None] | None = None
        self._backup_maintenance_error: str | None = None
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
        self.console_needs_attention = False
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
        # TASK-26040 (lane-7 review Important #3): snapshot a newer-than-
        # supported config schema version the loader detected. Served
        # untouched (never mangled by a downgrade); surfaced once mounted,
        # mirroring the parse-failure notification above.
        self._config_schema_conflict = get_config_schema_conflict()
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
        self._tts_profile_repository_path = get_tts_profiles_db_path()
        self._tts_profile_repository: TTSProfileRepository | None = None
        self._tts_profile_repository_close_requested = False
        self._tts_profile_repository_open_task: asyncio.Task[bool] | None = None
        self._tts_profile_repository_close_task: asyncio.Task[None] | None = None
        self._tts_profile_service: TTSProfileService | None = None
        self._audio_cpp_artifact_lease_coordinator: (
            AudioCppArtifactLeaseCoordinator | None
        ) = None
        self._tts_voice_bundle_service: "TTSVoiceBundlePortabilityService | None" = None
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
        # Built lazily by the `meeting_session_owner` property so the meeting
        # modules are not resident at `_ui_ready` (UI-ready module census).
        self._meeting_session_owner = None
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
        self._wire_llamacpp_snapshot_service()
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

        # Run independent initializations in parallel.
        #
        # TASK-21111(a): each task is timed AROUND ITS OWN EXECUTION, on the
        # worker thread, and the duration is stashed in
        # `self._startup_parallel_tasks`. The previous shape started the clock
        # in the `as_completed` loop immediately before `future.result()` --
        # by which point `as_completed` had already yielded the future
        # *because it was done*, so `result()` returned instantly and every
        # task logged 0.000s. The parallel phase (measured here at 82% of
        # construction on a fresh profile) could not be attributed at all.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all independent initialization tasks
            futures = {
                executor.submit(
                    self._timed_init_task,
                    "notes_service",
                    self._init_notes_service,
                    user_name_for_notes,
                ): "notes_service",
                executor.submit(
                    self._timed_init_task,
                    "providers_models",
                    self._init_providers_models,
                ): "providers_models",
                executor.submit(
                    self._timed_init_task,
                    "prompts_service",
                    self._init_prompts_service,
                ): "prompts_service",
                executor.submit(
                    self._timed_init_task, "media_db", self._init_media_db
                ): "media_db",
            }

            # Wait for all tasks to complete and log their real durations.
            for future in concurrent.futures.as_completed(futures):
                task_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    # The duration is still recorded (the wrapper stamps it in
                    # a `finally`), and a slow FAILING task is exactly the one
                    # worth timing.
                    logger.opt(exception=True).error(
                        f"Parallel init task '{task_name}' failed after "
                        f"{self._startup_parallel_tasks.get(task_name, 0.0):.3f}s: {e}"
                    )
                    continue
                logger.info(
                    f"Parallel init task '{task_name}' completed in "
                    f"{self._startup_parallel_tasks.get(task_name, 0.0):.3f}s"
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
        self._focus_mode_config = bool(get_cli_setting("general", "focus_mode", False))
        # Set by _resolve_initial_shell_route when onboarding outranks a
        # focus request at startup; restored when the wizard lands on Chat.
        self._deferred_focus_request: bool = False

        self._rich_log_handler: Optional[RichLogHandler] = (
            None  # For the RichLog widget in Logs tab
        )

        # Prompts service is initialized in parallel above
        # Set up timer

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
        self._actor_pack_recovery_reads: set[Any] = set()
        self._actor_pack_recovery_closed = False
        # Portable Tool Packs are unavailable until first Tool Profiles use
        # composes every authority owner and attaches one complete guard.
        self.tool_pack_service: Any | None = None
        self.tool_pack_service_unavailable_reason: str | None = "not_ready"
        self.tool_pack_receipt_reconciliation_unavailable_reason: str | None = "not_run"
        self._tool_pack_wiring_started = False
        self._tool_pack_composition_worker: Worker | None = None
        self._tool_profile_operations = None
        self._tool_profile_operations_closed = False
        self._screen_preimport_thread: threading.Thread | None = None
        # task-21110: the splash-overlapped warm-up of the INITIAL route's
        # module. Separate from `_screen_preimport_thread` (the whole-registry
        # pass that starts after first paint) because the two run at different
        # times for different reasons; both are idempotent on their own handle.
        self._initial_screen_preimport_thread: threading.Thread | None = None

        self._initial_screen_pushed = False
        self._ui_ready = False  # Track if UI is fully composed
        self._shutting_down = False  # Track if app is shutting down
        self._quit_in_progress = False

        # Lazily initialized by the Workflows destination; survives screen replacement.
        self._workflow_authoring = None
        self._workflow_session = None
        self._workflow_database_path = None
        self.workflow_documents = None
        self.workflow_drafts = None

        # TASK-22215: staggered boot-worker fleet state. The gate is built at
        # `_ui_ready` (`_start_staggered_boot_workers`); until then there is
        # deliberately nothing to admit, because every member of the fleet is
        # post-first-paint work by policy.
        self._boot_worker_gate: StaggeredBootWorkerGate | None = None
        self._boot_worker_handles: dict[str, Worker] = {}
        self._boot_worker_reconcile_timer: Optional[Timer] = None

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

        if self.chachanotes_db is not None:
            self.notes_organization_repository = None
            self.local_first_sync_service.notes_organization_repository = None
            self.sync_restore_service.notes_organization_repository = None

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
        TldwCli._reset_collections_capture_services(self)
        # TASK-21108: the lasting-sync runtime is built on FIRST ACCESS, not
        # here. Its construction is what drags `Notes/notes_sync_runtime` and
        # (through the TASK-21112 start gate) `Notes/notes_sync_legacy` --
        # together 15 modules and ~21 ms, measured 2026-08-23 -- into the
        # `import tldw_chatbook.app` closure, for an object nothing reads until
        # `on_mount` starts it. The property below still accepts assignment,
        # so a test can substitute a runtime double exactly as before.
        #
        # BE HONEST ABOUT WHAT THIS BUYS. `on_mount` reads the property
        # unconditionally to call `.start()`, and Textual dispatches Mount
        # inside `batch_update()` with `_ready()`/first paint in the `finally`
        # after it (textual/app.py:3428-3457). So on a real boot these 15
        # modules are RELOCATED from import time to mount time, still before
        # first paint -- measured: 0/15 resident after `import
        # tldw_chatbook.app`, 15/15 after `run_test()` on a zero-profile
        # boot. The TASK-21112 gate suppresses STARTING, not CONSTRUCTING.
        # What this does buy is a clean import closure (so the guard can see
        # future drift) and no cost at all for consumers that import the
        # module without running the app. Gating construction on the same
        # evidence would make it a real win, but the evidence lives in
        # `notes_sync_legacy` -- reading it imports 12 of the 15 -- and it
        # would split the "configured?" decision that TASK-21112 centralised
        # in `_start_once`. Tracked as a follow-up, deliberately not done here.
        #
        # The two collaborators the eager build READ here are captured here
        # too, so deferring WHEN the owner is built does not also change
        # WHICH objects it binds. This is not hypothetical: the File Notes
        # lifecycle tests replace `app.file_notes_session_owner` between
        # construction and mount, and a build that re-read the attribute at
        # mount would bind the replacement (and, there, crash on it).
        self._notes_sync_file_notes_binding = (
            self.file_notes_session_owner.current_binding
        )
        self._notes_sync_scope_service = self.notes_scope_service
        self._notes_sync_runtime_owner: "NotesSyncRuntimeOwner | None" = None
        self._notes_sync_runtime_owner_lock = threading.Lock()
        self._notes_sync_runtime_start_task: asyncio.Task[None] | None = None
        self._notes_sync_runtime_shutdown_task: asyncio.Task[None] | None = None
        self._recover_inflight_transfers_task: asyncio.Task[None] | None = None
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
        # Workspace agent provisioning (task-8) is deferred to a post-ready
        # timer (see `_schedule_deferred_startup_work`) so the provisioning
        # module stays out of the UI-ready module census (ADR-097); the
        # startup backfill there covers workspaces created before the hook
        # is attached.
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
                if phase == "parallel_init":
                    # Sub-phases: these overlap each other and their parent,
                    # so they are indented and NOT additive with the phases
                    # above (TASK-21111).
                    for task, task_duration in sorted(
                        self._startup_parallel_tasks.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    ):
                        task_share = (
                            (task_duration / duration) * 100 if duration > 0 else 0
                        )
                        logger.info(
                            f"    - {task}: {task_duration:.3f}s "
                            f"({task_share:.1f}% of parallel_init)"
                        )
        logger.info("==============================")

        # Final memory check
        log_resource_usage()

    # Clusters C, E and O (TASK-33011): the lazy service properties and
    # builders, the ``_wire_*`` composition and the misc wiring are inherited
    # from ``ServiceWiringMixin`` (``app_service_wiring.py``).

    # Cluster D (TASK-33011): destination launchers, typed handoffs and Home
    # controls. Each stub delegates to the same-named ``app_destinations``
    # function; the ``_local_*_count`` Home providers stay here because the
    # boot-time wiring hands them to the Home adapter.
    def open_study_screen(
        self,
        scope_context: "StudyScopeContext | None" = None,
        *,
        initial_section: str | None = None,
        origin: str | None = None,
    ) -> None:
        return _destinations().open_study_screen(
            self,
            scope_context,
            initial_section=initial_section,
            origin=origin,
        )

    def open_notes_workspace(
        self,
        workspace_id: str,
        subview: Any = None,
    ) -> None:
        return _destinations().open_notes_workspace(self, workspace_id, subview)

    def open_conversation_archive(
        self, query: str = "", archive_scope: str = "archived"
    ) -> None:
        return _destinations().open_conversation_archive(self, query, archive_scope)

    def resume_console_conversation(self, conversation_id: str) -> None:
        return _destinations().resume_console_conversation(self, conversation_id)

    def open_chat_with_handoff(
        self,
        payload: "ChatHandoffPayload",
        *,
        action_label: str = "Use in Chat",
    ) -> None:
        return _destinations().open_chat_with_handoff(
            self,
            payload,
            action_label=action_label,
        )

    def stage_console_prompt_insert(
        self,
        application: "PromptVariableApplication",
    ) -> None:
        return _destinations().stage_console_prompt_insert(self, application)

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
        return _destinations().open_console_for_live_work(
            self,
            source=source,
            title=title,
            payload=payload,
            status=status,
            recovery=recovery,
            action_label=action_label,
        )

    def _stage_handoff(
        self,
        channel: HandoffChannel,
        value: Any,
        *,
        recovery: str,
    ) -> bool:
        return _destinations()._stage_handoff(self, channel, value, recovery=recovery)

    def get_acp_runtime_session_state(self) -> "ACPRuntimeSessionState":
        return _destinations().get_acp_runtime_session_state(self)

    def open_console_live_work_primary_action(self, launch: Any) -> bool:
        return _destinations().open_console_live_work_primary_action(self, launch)

    def _handle_home_control_action(
        self,
        action: HomeControlAction,
        *,
        target_id: str | None = None,
        target_route: str | None = None,
    ) -> HomeControlResult:
        return _destinations()._handle_home_control_action(
            self,
            action,
            target_id=target_id,
            target_route=target_route,
        )

    def approve_active_home_item(
        self, *, target_id: str | None = None
    ) -> HomeControlResult:
        return _destinations().approve_active_home_item(self, target_id=target_id)

    def reject_active_home_item(
        self, *, target_id: str | None = None
    ) -> HomeControlResult:
        return _destinations().reject_active_home_item(self, target_id=target_id)

    def pause_active_home_item(
        self, *, target_id: str | None = None
    ) -> HomeControlResult:
        return _destinations().pause_active_home_item(self, target_id=target_id)

    def resume_active_home_item(
        self, *, target_id: str | None = None
    ) -> HomeControlResult:
        return _destinations().resume_active_home_item(self, target_id=target_id)

    def retry_active_home_item(
        self, *, target_id: str | None = None
    ) -> HomeControlResult:
        return _destinations().retry_active_home_item(self, target_id=target_id)

    def open_home_flashcards_review(self) -> None:
        return _destinations().open_home_flashcards_review(self)

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
        finally:
            if (
                type(db) is CharactersRAGDB
                and not db.is_memory_db
                and threading.current_thread() is not threading.main_thread()
            ):
                db.close_connection()

    def _local_eval_open_run_counts(self) -> dict[str, int]:
        """Count pending/failed local eval runs for Home (spec §4).

        Never counts 'running' -- a crashed app orphans running rows
        forever, which would permanently pin the review suggestion.
        """
        service = getattr(self, "local_evaluation_service", None)
        list_runs = getattr(service, "list_runs", None)
        if not callable(list_runs):
            return {"pending": 0, "failed": 0}
        from .Backup_Recovery.participants import run_finite_local_worker
        from .DB.Evals_DB import EvalsDB

        def counts():
            return {
                "pending": len(
                    list_runs(status="pending", limit=_HOME_EVAL_RUN_QUERY_LIMIT)
                ),
                "failed": len(
                    list_runs(status="failed", limit=_HOME_EVAL_RUN_QUERY_LIMIT)
                ),
            }

        try:
            if (
                type(service) is LocalEvaluationsService
                and type(service.db) is EvalsDB
                and getattr(list_runs, "__func__", None)
                is LocalEvaluationsService.list_runs
            ):
                return run_finite_local_worker(counts)
            return counts()
        except Exception:
            logger.opt(exception=True).debug("Home eval run counts failed.")
            return {"pending": 0, "failed": 0}

    def _local_read_later_count(self) -> int | None:
        """Count read-it-later media for Home; None when the DB is absent.

        Uses the scalar ``COUNT(*)`` seam rather than materializing the
        id list -- Home needs only the total.
        """
        db = getattr(self, "media_db", None)
        counter = getattr(db, "count_read_it_later_media", None)
        if not callable(counter):
            return None
        from .Backup_Recovery.participants import run_finite_local_worker

        try:
            if (
                type(db) is MediaDatabase
                and getattr(counter, "__func__", None)
                is MediaDatabase.count_read_it_later_media
            ):
                return int(run_finite_local_worker(counter))
            return int(counter())
        except Exception:
            logger.opt(exception=True).debug("Home read-it-later count failed.")
            return None

    def open_active_home_item_details(
        self,
        *,
        target_id: str | None = None,
        target_route: str = TAB_CHAT,
    ) -> HomeControlResult:
        return _destinations().open_active_home_item_details(
            self,
            target_id=target_id,
            target_route=target_route,
        )

    @staticmethod
    def _watchlists_run_navigation_context(
        target_id: str | None,
    ) -> dict[str, object]:
        return _destinations()._watchlists_run_navigation_context(target_id)

    def open_active_home_item_in_console(
        self,
        *,
        target_id: str | None = None,
        target_route: str = TAB_CHAT,
    ) -> HomeControlResult:
        return _destinations().open_active_home_item_in_console(
            self,
            target_id=target_id,
            target_route=target_route,
        )

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
        _wire_notes_sync_services(self)
        self.ensure_collections_capture_services()
        self._activate_collections_capture_authority()

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
        """Initialize notes service and retire this startup thread's connection."""
        notes_db = None
        try:
            # Get the full path to the unified ChaChaNotes DB FILE
            chachanotes_db_file_path = get_chachanotes_db_path()
            logger.info(f"Unified ChaChaNotes DB file path: {chachanotes_db_file_path}")

            # Determine the PARENT DIRECTORY for NotesInteropService's 'base_db_directory'
            actual_base_directory_for_service = chachanotes_db_file_path.parent
            logger.info(
                f"Notes for user '{user_name_for_notes}' will use the unified DB: {chachanotes_db_file_path}"
            )

            notes_db = get_chachanotes_db_lazy()
            self.notes_service = NotesInteropService(
                base_db_directory=actual_base_directory_for_service,
                api_client_id="tldw_tui_client_v1",
                global_db_to_use=notes_db,
            )
            logger.info(
                f"NotesInteropService successfully initialized for user '{user_name_for_notes}'."
            )
        except Exception as e:
            logger.opt(exception=True).error(
                f"Failed to initialize NotesInteropService: {e}"
            )
            self.notes_service = None
        finally:
            if notes_db is not None:
                notes_db.close_connection()

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
        finally:
            if prompts_interop.is_initialized():
                prompts_interop.get_db_instance().close_connection()

    def _init_media_db(self) -> None:
        """Initialize media database and retire this startup thread's connection."""
        media_db = None
        try:
            media_db_path = get_media_db_path()
            # Get integrity check configuration
            check_integrity = self.app_config.get("database", {}).get(
                "check_integrity_on_startup", False
            )
            media_db = self.media_db = MediaDatabase(
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
        finally:
            if media_db is not None:
                media_db.close_connection()

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

        Both the live view and Copy all use the same credential/PII-redacted
        diagnostic text. Their stores are bounded to the same session window.
        """
        from collections import deque

        from tldw_chatbook.Logging_Config import LogsBufferHandler
        from tldw_chatbook.UI.Logs_Window import MAX_LOG_RECORDS

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

        # Add the persistent handler to the root logger. It shares the private
        # file's single redaction pass (PERF-03; see LogsBufferHandler).
        if not hasattr(self, "_persistent_log_handler"):
            self._persistent_log_handler = LogsBufferHandler(self)
            logging.getLogger().addHandler(self._persistent_log_handler)
            logger.info("Persistent logging handler set up for screen navigation")

        # The app logs via loguru and the persistent handler is stdlib-only,
        # but NO bridge is installed here: `Logging_Config._setup_logging`
        # already forwards every loguru record into stdlib logging
        # (`_forward_loguru_to_standard`, diagnose=False per task-2119), and
        # it runs before this method on every boot path —
        # either early at process start or via `configure_application_
        # logging` in `_setup_logging`. A second sink here made every loguru
        # record reach the root logger twice, so the Logs screen showed each
        # application log line — and counted each error — twice
        # (TASK-15422).

        # Initialize current log widget reference
        self._current_log_widget = None

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

        no_splash = getattr(self, "_cli_no_splash", False)  # --no-splash (TASK-34100.16)
        splash_enabled = not no_splash and get_cli_setting("splash_screen", "enabled", True)
        logging.info(f"Splash screen enabled: {splash_enabled}")
        if splash_enabled:
            # Get splash screen configuration
            splash_duration = get_cli_setting(
                "splash_screen", "duration", DEFAULT_SPLASH_DURATION_SECONDS
            )
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
                # Attribute stalls from the timer's install, not its first beat.
                self.ui_responsiveness_monitor.arm()
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
        """Clear the footer status timers' diagnostic entries.

        The timer object itself is owned by ``DBStatusManager`` and stopped
        by its ``stop_periodic_updates()``; both shutdown hooks call that
        immediately before this. task-21133 removed the second, token-count
        timer this method also used to own, so there is no longer a handle
        to stop here.
        """
        self._record_ui_responsiveness_timer_stopped("footer-db-size-periodic")

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
        "artifacts": {LIBRARY_NAV_CONTEXT_MODE: "artifacts-all"},
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

    #: TASK-24459: feature sheets split off the boot bundle
    #: (``build_css.SCREEN_OWNED_SPLITS``) that the APP parses on first
    #: navigation to the owning route. Loaded here rather than via the
    #: screens' ``CSS_PATH`` because Textual loads ``CSS_PATH`` under ANY
    #: app -- including the UI-test harnesses that deliberately model the
    #: unstyled tier (``Tests/UI/consolidated_css.py`` loads no app bundle);
    #: a ``CSS_PATH`` styled harness-mounted screens with only the MOVED
    #: half of the module and flipped three destination-shell geometry
    #: tests (2026-09-04, paired arms). The agentic split sheets predate
    #: this seam and keep their TASK-25812 wiring (console on the boot
    #: path; library/settings via their screens' ``CSS_PATH``).
    _SCREEN_OWNED_ROUTE_CSS: ClassVar[dict[str, tuple[str, ...]]] = {
        TAB_SCHEDULES: ("screen_feature_scheduling.tcss",),
        TAB_EVALS: ("screen_feature_evals.tcss",),
        TAB_WATCHLISTS_COLLECTIONS: ("screen_feature_watchlists.tcss",),
        TAB_WORKFLOWS: ("screen_feature_workflows.tcss",),
    }

    # Cluster K2 (TASK-33011): Roleplay-to-Console character-conversation
    # activation, delegated to ``app_destinations``.
    async def activate_character_conversation_from_roleplay(
        self,
        request: object,
        cancellation: asyncio.Event,
        phase_changed: Callable[[str], None],
    ) -> object:
        return await _destinations().activate_character_conversation_from_roleplay(
            self,
            request,
            cancellation,
            phase_changed,
        )

    @staticmethod
    async def _await_character_conversation_post_commit(
        operation: asyncio.Task[Any],
    ) -> Any:
        return await _destinations()._await_character_conversation_post_commit(
            operation,
        )

    async def _complete_character_conversation_post_commit(
        self,
        candidate: Any,
        caller: Any,
        request: Any,
        runtime_identity: Any,
    ) -> Any:
        return await _destinations()._complete_character_conversation_post_commit(
            self,
            candidate,
            caller,
            request,
            runtime_identity,
        )

    async def _transfer_pushed_console_to_content(
        self,
        candidate: Any,
        caller: Any,
    ) -> None:
        return await _destinations()._transfer_pushed_console_to_content(
            self,
            candidate,
            caller,
        )

    async def _remove_promoted_screen_caller(self, caller: Any) -> None:
        return await _destinations()._remove_promoted_screen_caller(self, caller)

    # Cluster K3 (TASK-33011): Personal Context launchers, delegated to
    # ``app_destinations``. ``_load_personal_context_sync_runtime`` and
    # ``get_personal_context_service`` stay here: startup wiring reaches them.
    def prepare_personal_context_interview_request(
        self,
        *,
        kind: str,
        mode: str = "fixed",
        scope_id: str | None = None,
        local_workspace_id: str | None = None,
        workspace_label: str = "",
        source: str | None = None,
    ):
        return _destinations().prepare_personal_context_interview_request(
            self,
            kind=kind,
            mode=mode,
            scope_id=scope_id,
            local_workspace_id=local_workspace_id,
            workspace_label=workspace_label,
            source=source,
        )

    def build_personal_context_interview_screen(self, request):
        return _destinations().build_personal_context_interview_screen(self, request)

    def launch_personal_context_interview(
        self,
        kind: str,
        scope_id: str,
        mode: str = "fixed",
    ) -> None:
        return _destinations().launch_personal_context_interview(
            self,
            kind,
            scope_id,
            mode,
        )

    def _reload_personal_context_settings_panel(self) -> None:
        return _destinations()._reload_personal_context_settings_panel(self)

    def launch_personal_context_link(self) -> None:
        return _destinations().launch_personal_context_link(self)

    def _load_personal_context_sync_runtime(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
    ) -> None:
        """Restore exact protected Personal Context Sync collaborators after restart."""

        from .Personal_Context.link_key_custody import (
            KeyringPersonalContextLinkKeyCustodian,
        )
        from .Personal_Context.link_service import (
            PersonalContextLinkService,
            authenticate_legacy_completed_link_artifacts,
            cleanup_completed_link_artifacts,
        )

        link = self.sync_state_repository.get_personal_context_link_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
        )
        if link is None or link["state"] != "complete":
            raise ValueError("personal_context_link_incomplete")
        custodian = KeyringPersonalContextLinkKeyCustodian()
        storage_key = custodian.load_storage_key(
            **PersonalContextLinkService._key_binding(link)
        )
        service = self.get_personal_context_service(retry_locked=True)
        authenticate_legacy_completed_link_artifacts(service, custodian, link)
        cleanup_completed_link_artifacts(service, link)
        custodian.delete(**PersonalContextLinkService._key_binding(link))
        dispatcher = service.build_personal_context_outbox_dispatcher(
            state_repository=self.sync_state_repository,
            integrity_key_id=str(link["integrity_key_id"]),
        )
        self.sync_v2_dataset_keys[str(link["dataset_id"])] = storage_key
        self.local_first_sync_service.personal_context_outbox_dispatcher = dispatcher
        self.local_first_sync_service.personal_context_service = service

    async def _run_personal_context_link(self) -> None:
        return await _destinations()._run_personal_context_link(self)

    def get_personal_context_service(self, *, retry_locked: bool = False):
        """Return the app-owned service, explicitly retrying a locked facade."""

        service = getattr(self, "_personal_context_service", None)
        status = getattr(service, "status", None)
        if service is not None and not (
            retry_locked and callable(status) and status().state == "locked"
        ):
            return service
        with _PERSONAL_CONTEXT_SERVICE_BOOTSTRAP_LOCK:
            service = getattr(self, "_personal_context_service", None)
            status = getattr(service, "status", None)
            if service is None or (
                retry_locked and callable(status) and status().state == "locked"
            ):
                from .Personal_Context.bootstrap import (
                    bootstrap_personal_context_service,
                )

                service = bootstrap_personal_context_service()
                self._personal_context_service = service
        return service

    def _create_research_workspace_screen(self, screen_class: type):
        """Late-bind the foundation to the currently active owner services."""

        from .Research_Workspace import (
            LocalResearchWorkspaceAdapter,
            ResearchPresentationOverlayStore,
            ResearchWorkspaceController,
            ServerResearchWorkspaceAdapter,
            WorkspaceDataSource,
        )

        ports = {}
        local_service = getattr(self, "workspace_registry_service", None)
        media_scope_service = getattr(self, "media_reading_scope_service", None)
        operation_store = getattr(self, "research_source_operation_store", None)
        association_scheduler = getattr(
            self, "research_source_association_scheduler", None
        )
        if local_service is not None:
            ports[WorkspaceDataSource.LOCAL] = LocalResearchWorkspaceAdapter(
                local_service,
                media_scope_service=media_scope_service,
                operation_store=operation_store,
                association_scheduler=association_scheduler,
                notes_scope_service=getattr(self, "notes_scope_service", None),
                notes_user_id=getattr(self, "notes_user_id", ""),
            )
        server_service = getattr(self, "server_notes_workspace_service", None)
        server_context_provider = getattr(self, "server_context_provider", None)
        if server_service is not None and server_context_provider is not None:
            ports[WorkspaceDataSource.SERVER] = ServerResearchWorkspaceAdapter(
                server_service,
                server_context_provider,
                media_scope_service=media_scope_service,
                operation_store=operation_store,
                association_scheduler=association_scheduler,
            )
        controller = ResearchWorkspaceController(ports)
        overlay_store = ResearchPresentationOverlayStore(
            get_user_data_dir() / "research_workspace_overlay.json"
        )
        return screen_class(
            self,
            controller=controller,
            overlay_store=overlay_store,
            operation_store=operation_store,
            association_scheduler=association_scheduler,
            paste_staging_store=getattr(self, "research_paste_staging_store", None),
        )

    async def _reconcile_research_quick_notes_startup(self) -> None:
        """Resume one bounded global Local Quick Note receipt page."""

        from .Research_Workspace import LocalResearchWorkspaceAdapter

        registry = getattr(self, "workspace_registry_service", None)
        notes_scope = getattr(self, "notes_scope_service", None)
        notes_user_id = str(getattr(self, "notes_user_id", "") or "").strip()
        if registry is None or notes_scope is None or not notes_user_id:
            return
        try:
            await LocalResearchWorkspaceAdapter(
                registry,
                notes_scope_service=notes_scope,
                notes_user_id=notes_user_id,
            ).reconcile_quick_notes()
        except Exception as exc:  # noqa: BLE001 - startup recovery must degrade safely
            logger.warning(
                "Research Quick Note startup reconciliation deferred: {}",
                type(exc).__name__,
            )

    def _valid_startup_route_ids(self) -> set[str]:
        """Return route ids allowed in startup config during the shell migration."""

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

    def set_console_attention_projection(self, needs_attention: bool) -> None:
        """Publish one boolean Console-attention value to mounted shell chrome."""
        self.console_needs_attention = bool(needs_attention)
        for screen in tuple(getattr(self, "_screen_stack", ())):
            sync_screen = getattr(screen, "sync_console_attention", None)
            if callable(sync_screen):
                try:
                    sync_screen(self.console_needs_attention)
                except Exception:
                    logger.debug("Console overflow attention projection failed")
            try:
                nav_bars = tuple(screen.query(MainNavigationBar))
            except Exception:
                continue
            for nav_bar in nav_bars:
                try:
                    nav_bar.sync_console_attention(self.console_needs_attention)
                except Exception:
                    logger.debug("Console navigation attention projection failed")

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
        if getattr(self, "_screen_navigation_paused", False) or getattr(
            self, "_shutting_down", False
        ):
            return
        worker = self.run_worker(
            self._run_admitted_screen_navigation(message),
            group="screen-navigation",
            exclusive=False,
            exit_on_error=False,
        )
        self._screen_navigation_workers = {
            prior
            for prior in getattr(self, "_screen_navigation_workers", ())
            if not prior.is_finished
        }
        self._screen_navigation_workers.add(worker)

    #: Bound on the dismiss-the-overlays loop below. Each pass removes one
    #: pushed screen, and dismissing one can legitimately reveal another
    #: (a picker opened from a dialog); nothing real stacks this deep, so a
    #: stack that will not reduce inside the bound is a stuck stack, not a
    #: busy one.
    _MAX_NAVIGATION_OVERLAY_DISMISSALS: int = 16

    # Cluster N (TASK-33011): TTS/STTS handlers and speech resource owners.
    # Each stub delegates to the same-named ``app_speech`` function; the
    # ``@on`` decorators stay here because Textual dispatches only decorated
    # methods of the App class. ``_bind_tts_service`` (on_mount) and the
    # three ``_ensure_tts_*`` builders ``TTS/profile_source.py`` identifies
    # by code object stay in this class.
    @on(TTSRequestEvent)
    async def handle_tts_request_event(self, event: TTSRequestEvent) -> None:
        return await _speech().handle_tts_request_event(self, event)

    @on(TTSMessageSpeechRequestEvent)
    async def handle_tts_message_speech_request_event(
        self,
        event: TTSMessageSpeechRequestEvent,
    ) -> None:
        return await _speech().handle_tts_message_speech_request_event(self, event)

    @on(TTSGlobalOverrideDecisionEvent)
    async def handle_tts_global_override_decision_event(
        self,
        event: TTSGlobalOverrideDecisionEvent,
    ) -> None:
        return await _speech().handle_tts_global_override_decision_event(self, event)

    async def _offer_tts_global_override(self, token: str) -> None:
        return await _speech()._offer_tts_global_override(self, token)

    @on(TTSCompleteEvent)
    async def handle_tts_complete_event(self, event: TTSCompleteEvent) -> None:
        return await _speech().handle_tts_complete_event(self, event)

    async def _deliver_tts_complete_event(self, event: TTSCompleteEvent) -> None:
        return await _speech()._deliver_tts_complete_event(self, event)

    @on(TTSProgressEvent)
    async def handle_tts_progress_event(self, event: TTSProgressEvent) -> None:
        return await _speech().handle_tts_progress_event(self, event)

    @on(TTSPlaybackEvent)
    async def handle_tts_playback_event(self, event: TTSPlaybackEvent) -> None:
        return await _speech().handle_tts_playback_event(self, event)

    async def control_tts_playback(self, event: TTSPlaybackEvent) -> None:
        return await _speech().control_tts_playback(self, event)

    @on(STTSPlaygroundGenerateEvent)
    async def handle_stts_playground_generate_event(
        self, event: STTSPlaygroundGenerateEvent
    ) -> None:
        return await _speech().handle_stts_playground_generate_event(self, event)

    @on(STTSSettingsSaveEvent)
    async def handle_stts_settings_save_event(
        self, event: STTSSettingsSaveEvent
    ) -> None:
        return await _speech().handle_stts_settings_save_event(self, event)

    @on(STTSProviderConfigurationChanged)
    def handle_stts_provider_configuration_changed(
        self,
        event: STTSProviderConfigurationChanged,
    ) -> None:
        return _speech().handle_stts_provider_configuration_changed(self, event)

    def _deliver_stts_provider_configuration_changed(
        self, event: STTSProviderConfigurationChanged
    ) -> None:
        return _speech()._deliver_stts_provider_configuration_changed(self, event)

    @on(STTSAudioBookGenerateEvent)
    async def handle_stts_audiobook_generate_event(
        self, event: STTSAudioBookGenerateEvent
    ) -> None:
        return await _speech().handle_stts_audiobook_generate_event(self, event)

    def _bind_tts_service(self) -> None:
        """Bind the single TTS service owned by this application."""
        if self._tts_binding_active:
            return
        bind_tts_service(self.tts_service)
        self._tts_binding_active = True

    async def _close_tts_service(self) -> None:
        return await _speech()._close_tts_service(self)

    async def _ensure_tts_profile_repository(
        self,
    ) -> "TTSProfileRepository | None":
        """Open and return the one app-owned profile repository on first use."""

        if getattr(self, "_tts_profile_repository_close_requested", False):
            return None
        repository = getattr(self, "_tts_profile_repository", None)
        if repository is None:
            from tldw_chatbook.TTS.profile_repository import TTSProfileRepository

            repository = TTSProfileRepository(self._tts_profile_repository_path)
            self._tts_profile_repository = repository
            from tldw_chatbook.TTS.profile_source import bind_app_repository

            bind_app_repository(self)
        if getattr(self, "_tts_profile_repository_close_task", None) is not None:
            return None
        if getattr(repository, "_configured_source", None) is not None:
            from tldw_chatbook.TTS.profile_source import check_repository_source

            try:
                check_repository_source(repository)
            except ProfileRepositoryError:
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
            or getattr(self, "_tts_profile_repository_close_requested", False)
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
            from tldw_chatbook.TTS.profile_source import bind_app_profile_service

            bind_app_profile_service(self)
        return profile_service

    def _saved_audio_cpp_managed_consumers(
        self,
    ) -> "tuple[AudioCppManagedConsumerIdentity, ...]":
        return _speech()._saved_audio_cpp_managed_consumers(self)

    def _ensure_audio_cpp_artifact_lease_coordinator(
        self,
    ) -> AudioCppArtifactLeaseCoordinator:
        return _speech()._ensure_audio_cpp_artifact_lease_coordinator(self)

    def _audio_cpp_removal_settings_inputs(
        self,
    ) -> (
        "tuple[AudioCppSettingsConfig, AudioCppSettingsConfig | None, "
        "TTSPreferencesSnapshot, TTSPreferencesSnapshot | None]"
    ):
        return _speech()._audio_cpp_removal_settings_inputs(self)

    async def _audio_cpp_model_library_observation_snapshot(
        self,
        references: tuple["ArtifactRef", ...],
    ) -> "AudioCppModelLibraryObservationSnapshot":
        return await _speech()._audio_cpp_model_library_observation_snapshot(
            self,
            references,
        )

    async def _audio_cpp_artifact_removal_evidence(
        self,
        reference: "ArtifactRef",
    ) -> "AudioCppArtifactRemovalEvidence":
        return await _speech()._audio_cpp_artifact_removal_evidence(self, reference)

    async def _ensure_tts_voice_bundle_service(
        self,
    ) -> "TTSVoiceBundlePortabilityService | None":
        """Construct the app-owned portability owner only on first use."""

        if getattr(self, "_tts_voice_bundle_service_close_task", None) is not None:
            return None
        profile_service = await self._ensure_tts_profile_service()
        if profile_service is None:
            return None
        service = getattr(self, "_tts_voice_bundle_service", None)
        if service is None:
            # TASK-21108: deferred to this single construction site so the
            # 1,857-line module stays off the app import path.
            from tldw_chatbook.TTS.voice_bundle_service import (  # noqa: PLC0415
                TTSVoiceBundlePortabilityService,
            )

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
            from tldw_chatbook.TTS.profile_source import bind_app_bundle_service

            bind_app_bundle_service(self)
        return service

    async def _close_tts_voice_bundle_service(self) -> None:
        return await _speech()._close_tts_voice_bundle_service(self)

    async def _close_tts_profile_repository(self) -> None:
        return await _speech()._close_tts_profile_repository(self)

    async def _close_owned_tts_resources(self) -> None:
        return await _speech()._close_owned_tts_resources(self)

    def on_mount(self) -> None:
        """Configure logging and schedule post-mount setup."""
        self._start_persona_buddy_overlay()
        runtime = self.console_runtime
        if runtime is not None:
            runtime.start_async_lifecycles()
        self._start_served_canvas_control()
        self.watchlists_operation_coordinator = WatchlistsOperationCoordinator(
            local_service=self.local_watchlists_service,
            briefing_db=self.subscriptions_db,
        )
        self.watchlists_operation_coordinator.bind_running_loop()
        self._wire_watchlists_command_service()
        self._bind_tts_service()
        self._start_backup_maintenance_monitor()
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
        self._restore_ingest_jobs_and_schedule_research_sources()
        self.run_worker(
            self._reconcile_research_quick_notes_startup(),
            group="research-quick-notes-startup-reconciliation",
            exclusive=True,
            exit_on_error=False,
        )

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
            self._initial_screen_setup_task = self._create_deferred_startup_task(
                self._run_no_splash_post_mount_setup(),
                name="no_splash_post_mount_setup",
            )
        else:
            # task-21110: with the splash up, the branch above schedules
            # nothing -- the initial screen is pushed only once
            # `SplashScreen.Closed` arrives, and its module is imported
            # synchronously on this loop at that moment. Overlap that import
            # with the splash instead of serializing behind it.
            #
            # The zero branch is not hypothetical tidiness: Textual 8's
            # `set_timer(0.0)` divides by the interval inside `Timer._run`,
            # so a 0s delay raises ZeroDivisionError in the timer's own task
            # and the callback NEVER fires -- silently, because nobody
            # retrieves that task's exception. Measured while A/B-ing this
            # delay: the "0.0s" arm looked like a clean no-stutter win purely
            # because no pre-import had happened at all.
            if SPLASH_INITIAL_SCREEN_PREIMPORT_DELAY_SECONDS > 0:
                self.set_timer(
                    SPLASH_INITIAL_SCREEN_PREIMPORT_DELAY_SECONDS,
                    self._schedule_initial_screen_preimport,
                )
            else:
                self.call_after_refresh(self._schedule_initial_screen_preimport)

        # Theme registration
        theme_start = time.perf_counter()
        for theme_name in ALL_THEMES:
            self.register_theme(theme_name)
        # TASK-31250: saved user themes (Settings > Theme > Save) load like
        # shipped ones so general.default_theme can name them.
        from .config import get_user_themes_dir
        from .css.Themes.themes import load_user_themes

        for user_theme in load_user_themes(get_user_themes_dir()):
            self.register_theme(user_theme)

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

        from tldw_chatbook.Backup_Recovery.activation import execution_allowed

        if execution_allowed(("db.scheduled_tasks",), self.scheduler_loop.db.db_path):
            # Stale-run reconciliation (spec §4.1): an app killed mid-run must
            # not leave a phantom `running`/`queued` automation_runs row in the
            # UI forever. Cutoff is generous on purpose -- longer than any run
            # this process itself would let live (handler timeout) plus two
            # poll intervals of scheduling slack -- so a run still genuinely
            # in flight is never reconciled out from under itself. Guarded:
            # a diagnostics-adjacent startup step must never block the
            # scheduler from starting.
            try:
                self.scheduling_service.db.reconcile_stale_automation_runs(
                    older_than_seconds=HANDLER_TIMEOUT_SECONDS
                    + 2 * SCHEDULER_POLL_INTERVAL_SECONDS
                )
            except Exception:
                self.loguru_logger.exception(
                    "Failed to reconcile stale automation runs at startup"
                )

            # Transfer-machine startup recovery (spec §6.1.3): a row stuck
            # `to_server_sent` across a crash/restart is the one case
            # SyncEngine's own push replay deliberately refuses to touch --
            # this is its only recovery path. Fire-and-forget: on_mount is not
            # async, and `recover_inflight_transfers` itself never raises
            # (each sub-step is independently exception-guarded, same
            # "must never block the scheduler from starting" discipline as
            # the stale-run reconciliation just above).
            self._recover_inflight_transfers_task = asyncio.create_task(
                self.scheduling_service.recover_inflight_transfers(),
                name="recover_inflight_transfers",
            )

        # TASK-22215: the two FTS backfills (task-688 subscription_items,
        # task-21100 messages) used to start HERE, before first paint, next
        # to the scheduler. They are whole-table re-tokenizations that
        # nothing waits on and that resume from a frontier in their own
        # database, so they belong in the staggered tier -- see
        # `Utils/boot_worker_policy.py` and `_start_staggered_boot_workers`.

    def _start_served_canvas_control(self) -> None:
        """Attach this authoritative child to its parent transport."""

        client = self.served_canvas_control
        if client is None or self._served_canvas_control_start_task is not None:
            return
        task = asyncio.create_task(client.start(), name="start_served_canvas_control")
        self._served_canvas_control_start_task = task

        def observe(completed: asyncio.Task[None]) -> None:
            try:
                completed.result()
            except asyncio.CancelledError:
                pass
            except Exception as error:
                self.loguru_logger.warning(
                    "Served Canvas control unavailable type={} code=connection_failed",
                    type(error).__name__,
                )

        task.add_done_callback(observe)

    async def _stop_served_canvas_control(self) -> None:
        """Close the private child channel without affecting terminal exit."""

        task = self._served_canvas_control_start_task
        self._served_canvas_control_start_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        client = self.served_canvas_control
        if client is not None:
            await client.aclose()

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
                env_keys_that_silenced_first_run,
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
            elif action == "none" and (
                env_key_names := env_keys_that_silenced_first_run(
                    self.app_config, os.environ
                )
            ):
                # TASK-21147 (UAT E-1): the env-key install skipped the
                # wizard silently — say so exactly once, and where the
                # wizard's other value (voice, tools, encryption) lives.
                shown = ", ".join(env_key_names[:2]) + (
                    " (and more)" if len(env_key_names) > 2 else ""
                )
                self.notify(
                    f"Found {shown} — you're ready to chat. Run setup any "
                    "time: Settings ▸ Diagnostics ▸ Run setup wizard.",
                    title="Provider key detected",
                    severity="information",
                    timeout=10,
                )
                self._persist_env_key_notice_flag()
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

    def _maybe_warn_config_schema_conflict(self) -> None:
        """Warn (never block) when the config is from a newer app version.

        TASK-26040 AC#5: a config carrying a schema version newer than this
        build understands is served untouched rather than migrated (a
        downgrade could silently drop keys). This surfaces the detected
        conflict once the UI is up so the user knows why newer settings may
        not take effect, mirroring `_maybe_warn_config_load_failure`.
        """
        conflict = getattr(self, "_config_schema_conflict", None)
        if not conflict:
            return
        self.notify(
            f"Your configuration was written by a newer version of this "
            f"application and was left unchanged (not migrated). Some newer "
            f"settings may not take effect until you upgrade. {conflict}",
            title="Config is from a newer version",
            severity="warning",
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

    def _persist_env_key_notice_flag(self) -> None:
        """Record the one-time env-key notice (TASK-21147, UAT E-1)."""

        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            ENV_KEY_NOTICE_KEY,
            WIZARD_STATE_SECTION,
        )

        app_config = self.app_config
        if isinstance(app_config, dict):
            app_config.setdefault(WIZARD_STATE_SECTION, {})[ENV_KEY_NOTICE_KEY] = True

        def _write() -> None:
            from tldw_chatbook.config import save_settings_to_cli_config

            try:
                saved = save_settings_to_cli_config(
                    {WIZARD_STATE_SECTION: {ENV_KEY_NOTICE_KEY: True}}
                )
            except Exception as exc:
                logger.warning(
                    "Failed to persist env-key notice flag "
                    f"(category=persistence, error_type={type(exc).__name__})"
                )
                return
            if not saved:
                logger.warning(
                    "Failed to persist env-key notice flag "
                    "(category=persistence, error_type=save_returned_false)"
                )

        self.run_worker(
            _write,
            thread=True,
            group="first-run-env-key-notice-flag",
            exit_on_error=False,
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

        async def apply() -> None:
            # TASK-34100.1: setup work never exits the app. A failure reading
            # the draft or opening the wizard re-offers the prompt instead.
            try:
                await self._apply_first_run_recovery_result(result)
            except Exception as exc:  # noqa: BLE001 - the prompt is the recovery
                # ``result`` is one of the three fixed choices checked above.
                logger.error(
                    "First-run recovery failed (choice={}, error_type={})",
                    result,
                    type(exc).__name__,
                )
                self.notify("Setup could not open. Try again.", severity="error")
                self._schedule_first_run_recovery_retry()

        self.run_worker(
            apply(), exclusive=True, group="first-run-recovery", exit_on_error=False
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

    def _handle_first_run_wizard_result(
        self, result: dict | None, *, cancel_to_console: bool = True
    ) -> None:
        """Optionally chain personalization before the existing continuation.

        ``cancel_to_console`` is forwarded for cancellation routing
        (TASK-31813); dict results are unaffected by it.
        """

        if type(result) is dict and result.get("offer_profile_interview") is True:
            completed = result.get("completed")
            exit_route = result.get("exit_route")
            exit_context = result.get("exit_context")
            eligible = completed is True and exit_route in {None, TAB_CHAT, TAB_HOME}
            if exit_route is None:
                eligible = eligible and exit_context is None
            else:
                eligible = eligible and (
                    exit_context is None
                    or (type(exit_context) is dict and not exit_context)
                )
            if eligible:

                def continuation() -> None:
                    TldwCli._continue_first_run_wizard_result(
                        self, result, cancel_to_console=cancel_to_console
                    )

                try:
                    request = self.prepare_personal_context_interview_request(
                        kind="personal",
                        mode="fixed",
                        source="setup",
                    )
                except Exception:
                    logger.opt(exception=True).warning(
                        "First-run profile interview preparation failed"
                    )
                    notify = getattr(self, "notify", None)
                    if callable(notify):
                        try:
                            notify(
                                "Setup was saved, but profile personalization is unavailable.",
                                severity="warning",
                            )
                        except Exception:
                            logger.opt(exception=True).warning(
                                "First-run profile failure notification failed"
                            )
                    continuation()
                    return
                from .Personal_Context.interview_launch import (
                    launch_profile_interview_after_commit,
                )

                launch_profile_interview_after_commit(self, request, continuation)
                return
        TldwCli._continue_first_run_wizard_result(
            self, result, cancel_to_console=cancel_to_console
        )

    def _continue_first_run_wizard_result(
        self, result: dict | None, *, cancel_to_console: bool = True
    ) -> None:
        """Preserve the pre-interview first-run result handling byte-for-byte.

        TASK-31813: the cancel branch changed. Esc-exiting the boot-offered
        wizard used to strand the user on Home (the screen the wizard was
        pushed over, per the first-run startup route); cancelling now lands
        on the Console workbench. Settings/command-palette RE-RUNS opt out
        via ``cancel_to_console=False`` so cancelling a re-run leaves the
        caller's screen alone.
        """

        if type(result) is not dict:
            # Cancelled / finish-later: recovery state handles the NEXT
            # launch; this landing is for the user pressing Esc now.
            if not cancel_to_console:
                return
            # task-18812 parity: consume a deferred focus request under the
            # same Chat-route rule the completed paths use.
            if getattr(self, "_deferred_focus_request", False):
                self._deferred_focus_request = False
                self.focus_mode = True

            self.post_message(NavigateToScreen(TAB_CHAT, {}))
            return
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
        elif exit_route == TAB_LIBRARY:
            # task-32072: the wizard's "Add your first document" exit. The
            # route always means Import -- the destination is fixed here
            # rather than trusted from the wizard's payload.
            if completed is not True:
                return
            if exit_context is not None and (
                type(exit_context) is not dict or exit_context
            ):
                return
            screen_context = {LIBRARY_NAV_CONTEXT_INGEST: True}
        else:
            from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
                EXIT_ROUTE_LIBRARY_NOTES,
            )

            if exit_route != EXIT_ROUTE_LIBRARY_NOTES:
                return
            # task-32140: the wizard's "Write your first note" exit -- a
            # local-first user without a provider still gets a concrete
            # first action. Not a real tab id; rewritten to TAB_LIBRARY
            # below (same sentinel-then-rewrite shape as the ingest exit
            # above), since two different Library destinations both need
            # to travel as one wizard exit_route.
            if completed is not True:
                return
            if exit_context is not None and (
                type(exit_context) is not dict or exit_context
            ):
                return
            exit_route = TAB_LIBRARY
            screen_context = {LIBRARY_NAV_CONTEXT_NOTES_CREATE: True}

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
            self._schedule_startup_model_catalog_refresh(after_setup_completion=True)
            return

        if completed is not True:
            self.post_message(NavigateToScreen(exit_route, screen_context))
            return

        async def navigate_then_schedule_catalog_consent() -> None:
            try:
                await self.handle_screen_navigation(
                    NavigateToScreen(exit_route, screen_context)
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                self._schedule_startup_model_catalog_refresh(
                    after_setup_completion=True
                )
                raise
            self._schedule_startup_model_catalog_refresh(after_setup_completion=True)

        self.run_worker(
            navigate_then_schedule_catalog_consent(),
            group="first-run-exit-navigation",
            exclusive=True,
            exit_on_error=False,
        )

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

    def action_run_setup_wizard(self) -> None:
        """Open the setup wizard for a re-run (TASK-21145, UAT H-3).

        An app-level action so any surface can offer it as an action link
        (e.g. the Console composer's "Send blocked — finish provider setup"
        strip renders "[@click=app.run_setup_wizard]Open setup[/]"), not
        just the Settings button and the command palette.
        """
        try:
            if any(
                type(screen).__name__ == "FirstRunSetupWizard"
                for screen in self.screen_stack
            ):
                return
            from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
                FirstRunSetupWizard,
            )

            self.push_screen(
                FirstRunSetupWizard(self, rerun=True),
                self.handle_first_run_wizard_result,
            )
        except Exception as exc:
            self.notify(f"Failed to open setup wizard: {exc}", severity="error")

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

        # TASK-24459: an initial tab of schedules/evals needs its split-off
        # feature CSS exactly like an in-app navigation does.
        self._ensure_screen_owned_css(resolved_tab)

        if resolved_tab == TAB_CHAT:
            from tldw_chatbook.Chat.console_runtime import _initial_receipt_preparer

            runtime = self.console_runtime
            prepare = _initial_receipt_preparer(runtime)
            if prepare is not None:
                owner_task, owner_loop = asyncio.current_task(), asyncio.get_running_loop()
                owner_thread = threading.current_thread()
                runtime_identity = self._current_runtime_identity()
                initial_stack = tuple(self.screen_stack)
                initial_current_tab = self.current_tab
                initial_database = self.chachanotes_db
                initial_database_path = getattr(initial_database, "db_path", None)
                initial_marks = self.conversation_local_marks_service
                initial_route_value = getattr(self, "_initial_tab_value", TAB_CHAT)

                def require_initial_owner() -> None:
                    if _initial_receipt_preparer(runtime) is None:
                        raise RuntimeError("initial_receipt_source_changed")
                    current_stack = tuple(self.screen_stack)
                    if (
                        asyncio.current_task() is not owner_task
                        or asyncio.get_running_loop() is not owner_loop
                        or threading.current_thread() is not owner_thread
                        or self.console_runtime is not runtime
                        or runtime._app is not self
                        or runtime._disposed
                        or self.chachanotes_db is not initial_database
                        or getattr(initial_database, "db_path", None) != initial_database_path
                        or self.conversation_local_marks_service is not initial_marks
                        or getattr(self, "_initial_tab_value", TAB_CHAT) != initial_route_value
                        or self._current_runtime_identity() != runtime_identity
                        or self.current_tab != initial_current_tab
                        or len(current_stack) != len(initial_stack)
                        or any(now is not before for now, before in zip(current_stack, initial_stack))
                        or getattr(self, "_initial_screen_pushed", False)
                        or self._shutting_down
                        or self._exit
                    ):
                        raise RuntimeError("initial_screen_owner_changed")

                # Existing initial-setup custody is already counted by the
                # navigation drain. Accepted startup may finish while a new
                # maintenance/navigation admission fence is closed.
                async with self._screen_navigation_lock():
                    require_initial_owner()
                    if _initial_receipt_preparer(runtime) is None:
                        raise RuntimeError("initial_receipt_source_changed")
                    await prepare(self, require_current=require_initial_owner)
                    require_initial_owner()

        new_screen = screen_class(self)
        # TASK-31520: retain the initial screen exactly like a navigated-to
        # one. Without this, a reusable initial tab (chat is the default!)
        # built here was never installed, so the first navigation away
        # unmounted it and the first RETURN paid one full re-mint -- the
        # exact cost reuse exists to retire, on the app's most common
        # route, for every user.
        initial_route = resolve_screen_route(resolved_screen_name)
        if initial_route is not None and initial_route.reusable:
            self._retain_reusable_navigation_screen(
                resolved_tab,
                self._current_runtime_identity(),
                new_screen,
            )

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

        # Schedule after splash and the initial screen, before optional startup
        # offers; ADR-020 consent owns this launch when it is still required.
        self._schedule_startup_model_catalog_refresh()
        if not wizard_offered and not getattr(
            self, "_startup_model_catalog_consent_required", False
        ):
            # Spec 2026-08-17 §5.4: wizard wins; .SKILLS offer defers to next launch.
            self._maybe_offer_project_skills_import()
        try:
            self._maybe_warn_config_load_failure()
            self._maybe_warn_config_schema_conflict()
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
        except Exception as e:
            logger.opt(exception=True).error(f"No-splash post-mount setup failed: {e}")

    async def _post_mount_setup(self) -> None:
        """Operations to perform after the main UI is expected to be fully mounted."""
        # A delayed setup callback must not admit startup work after quit.
        # Textual exit() sets _exit before ShutdownRequest sets our flag.
        if self._shutting_down or self._exit:
            return
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
        freeze_long_lived_heap("ui_ready")  # ADR-198: keep the boot heap out of gen-2 scans

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

        # The first scheduler tick loads emergency-stop and heartbeat support.
        # Start only after `_ui_ready`: launching in on_mount let its queue
        # reads finish during slow UI setup and spend first-frame budget
        # nondeterministically (ADR-097).
        # Start the background scheduler loop for reminders and scheduled tasks.
        # A COROUTINE worker, never thread=True: scheduled watchlist checks
        # dispatch from this loop, and the watchlists in-flight guard
        # (`local_watchlists_service._IN_FLIGHT_URL_CHECKS`) is lock-free on
        # the invariant that every check entrant runs on the app's one event
        # loop. Moving dispatch off-loop needs a lock there.
        from tldw_chatbook.Backup_Recovery.activation import execution_allowed

        if execution_allowed(("db.scheduled_tasks",), self.scheduler_loop.db.db_path):
            self.scheduler_worker = self.run_worker(
                self.scheduler_loop.run(),
                exclusive=True,
                group="scheduling",
            )

        # dreams phase 1: Dreams DB + projection + boot catch-up, all after
        # `_ui_ready` (ADR-097 boot-census ratchet; guarded internally, a
        # disabled Dreams stops at two cheap settings reads).
        self._wire_dreams_scheduler_integration()

        self._schedule_deferred_startup_work()
        from .Backup_Recovery.profile_open import acknowledge_mounted

        self.call_after_refresh(acknowledge_mounted, self)

    async def update_db_sizes(self) -> None:
        """Updates the database size information in the shell status line."""
        await self.db_status_manager.update_db_sizes()

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

        # TASK-22215: the boot-time thread fleet starts here, under the
        # explicit order/concurrency policy in `Utils/boot_worker_policy.py`,
        # rather than all at once (and rather than partly from `on_mount`,
        # ahead of first paint, which is where the two FTS backfills used to
        # start).
        self._start_staggered_boot_workers()
        self.set_timer(
            DEFERRED_DB_SIZE_UPDATE_DELAY_SECONDS,
            self._schedule_footer_status_updates,
        )
        self.set_timer(
            DEFERRED_AUDIO_SERVICE_DELAY_SECONDS,
            self._start_deferred_audio_service_initialization,
        )
        self.set_timer(
            DEFERRED_COLLECTIONS_CAPTURE_WIRING_DELAY_SECONDS,
            self._deferred_wire_collections_capture_services,
        )
        # Workspace agent provisioning (task-8): best-effort hook attach +
        # startup backfill, deferred past `_ui_ready` so the provisioning
        # module stays out of the UI-ready module census (ADR-097).
        self.set_timer(
            DEFERRED_WORKSPACE_AGENT_PROVISIONING_DELAY_SECONDS,
            self._deferred_wire_workspace_agent_provisioning,
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
        # Schedule Notes/Sync last.  Earlier placement let the nominal 0.1 s
        # delay expire while the remaining synchronous setup below it was
        # still running, so its import graph could win the race against the
        # first-interactive-frame census on slower starts.
        _install_deferred_notes_sync_facades(self)
        self.set_timer(
            DEFERRED_NOTES_ORGANIZATION_WIRING_DELAY_SECONDS,
            self._deferred_wire_notes_sync_services,
        )

    # ------------------------------------------------------------------
    # TASK-22215: the staggered boot-worker fleet
    # ------------------------------------------------------------------

    def boot_worker_starters(self) -> dict[str, Callable[[], Optional[Worker]]]:
        """The start callables for every staggered boot worker, by policy key.

        One table, so the policy (``Utils/boot_worker_policy.py``) and the
        code that starts the fleet cannot drift apart: a key with no starter
        -- or a starter with no key -- is a test failure, not a worker that
        silently never runs.

        Returns:
            Policy key -> zero-argument callable returning the started
            ``Worker`` (or ``None`` when there was nothing to start).
        """

        def start_actor_pack_recovery() -> Worker:
            # task-21106: Actor Pack crash recovery, moved out of __init__ --
            # synchronous SQLite has no place on the construction path.
            # Retain the finite blocking callback through worker cancellation;
            # the coordinator's own once-guard makes every later
            # surface-side call (Personas mount, create_persona) a cached
            # no-op -- which is also why this may be staggered at all.
            return self.run_worker(
                self._run_actor_pack_recovery_owned(self.ensure_actor_pack_recovery),
                name="deferred_actor_pack_recovery",
                group="actor_pack_recovery",
                exclusive=True,
                exit_on_error=False,
            )

        def start_actor_pack_staging_sweep() -> Worker:
            # task-22216: the Actor Pack staging crash-sweep, moved out of
            # ActorPackImportService.__init__ (synchronous filesystem I/O on
            # the construction path). The service's once-gate also fires at
            # the entry of inspect_archive, so whichever comes first sweeps
            # and the other is a cached no-op.
            return self.run_worker(
                self.ensure_actor_pack_staging_sweep,
                name="deferred_actor_pack_staging_sweep",
                group="actor_pack_staging_sweep",
                thread=True,
                exclusive=True,
                exit_on_error=False,
            )

        def start_chachanotes_fts_backfill() -> Worker:
            # task-21100: reinsert the messages the v45->v46 FTS reset no
            # longer indexes inline, so an upgraded profile's chat history
            # becomes fully searchable again. thread=True: blocking sqlite.
            # The name is explicit so the (name, group) identity the boot
            # census pins cannot drift with a method rename.
            return self.run_worker(
                self._backfill_chachanotes_messages_fts,
                name="_backfill_chachanotes_messages_fts",
                group="chachanotes-fts-backfill",
                thread=True,
                exclusive=True,
            )

        def start_subscriptions_fts_backfill() -> Worker | None:
            # The thread body resolves its path and takes fresh admission.
            # Repeating that preflight here blocks the UI during boot.
            # task-688: index subscription_items rows scraped before the FTS5
            # index existed, so search covers a user's whole back catalogue
            # without any action on their part.
            return self.run_worker(
                self._backfill_subscription_items_fts,
                name="_backfill_subscription_items_fts",
                group="subscriptions-fts-backfill",
                thread=True,
                exclusive=True,
            )

        return {
            "actor_pack_recovery": start_actor_pack_recovery,
            "actor_pack_staging_sweep": start_actor_pack_staging_sweep,
            "chachanotes_fts_backfill": start_chachanotes_fts_backfill,
            "subscriptions_fts_backfill": start_subscriptions_fts_backfill,
        }

    def _start_boot_worker(self, key: str) -> Optional[Worker]:
        """Start one staggered boot worker.

        Args:
            key: A key from ``STAGGERED_BOOT_WORKER_KEYS``.

        Returns:
            The started worker, or None if the key has no starter (which is a
            wiring bug the policy test catches, not a runtime failure).
        """
        starter = self.boot_worker_starters().get(key)
        if starter is None:
            self.loguru_logger.warning(
                f"No starter registered for staggered boot worker {key!r}"
            )
            return None
        return starter()

    def _start_staggered_boot_workers(self) -> None:
        """Open the admission gate for the post-readiness boot fleet.

        Called once, from ``_schedule_deferred_startup_work`` (the last
        statement of ``_post_mount_setup``, i.e. after ``_ui_ready``).
        """
        if getattr(self, "_shutting_down", False):
            return
        self._boot_worker_gate = StaggeredBootWorkerGate(
            STAGGERED_BOOT_WORKER_KEYS,
            MAX_CONCURRENT_STAGGERED_BOOT_WORKERS,
        )
        self._boot_worker_handles = {}
        self._admit_staggered_boot_workers()

    def _admit_staggered_boot_workers(self) -> None:
        """Start whatever the gate admits, then arm the reconcile timer.

        Loops because a starter that raises (or declines to start anything)
        frees its slot immediately -- the queue must advance past it in the
        same pass rather than waiting for a completion that will never come.
        """
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        gate = getattr(self, "_boot_worker_gate", None)
        if gate is None:
            return
        if getattr(self, "_shutting_down", False):
            self._close_boot_worker_gate("shutdown")
            return
        while True:
            admitted = gate.admit()
            if not admitted:
                break
            for index, key in enumerate(admitted):
                worker: Optional[Worker] = None
                try:
                    worker = self._start_boot_worker(key)
                except Exception as error:
                    if (
                        isinstance(error, RecoveryRequired)
                        and str(error) == "storage_locally_paused"
                    ):
                        # Native intent can arrive after admission but before a
                        # starter reads its config. Retain all unstarted keys;
                        # the existing reconcile timer retries after readmission.
                        gate.defer(admitted[index:])
                        self._arm_boot_worker_reconcile()
                        return
                    self.loguru_logger.opt(exception=True).warning(
                        f"Staggered boot worker {key!r} failed to start"
                    )
                if worker is None:
                    # Nothing is in flight for this key, so no terminal
                    # transition will ever arrive: release the slot now.
                    gate.complete(key)
                    continue
                self._boot_worker_handles[key] = worker
        self._arm_boot_worker_reconcile()

    def _release_boot_worker_slot(self, worker: Any) -> None:
        """Free the slot a finished boot worker held and admit the next.

        Args:
            worker: The worker whose state just went terminal. Anything that
                is not a policy member is ignored, so this is safe to call
                from the app-wide ``Worker.StateChanged`` hook.
        """
        gate = getattr(self, "_boot_worker_gate", None)
        if gate is None:
            return
        key = BOOT_WORKER_KEY_BY_IDENTITY.get(
            (getattr(worker, "name", ""), getattr(worker, "group", ""))
        )
        if key is None or not gate.complete(key):
            return
        self._boot_worker_handles.pop(key, None)
        self._admit_staggered_boot_workers()

    def _arm_boot_worker_reconcile(self) -> None:
        """Keep a slow reconcile running while the fleet is outstanding.

        The gate advances on ``Worker.StateChanged``. This is the backstop
        for the one thing that hook cannot cover: a terminal transition that
        never reaches the handler (a worker whose message is dropped during a
        screen swap, a duck-typed worker). Without it a lost event would
        strand every remaining member of the fleet for the whole session --
        the failure mode a stagger policy must not introduce. It stops itself
        as soon as the gate is drained.
        """
        if getattr(self, "_boot_worker_reconcile_timer", None) is not None:
            return
        gate = getattr(self, "_boot_worker_gate", None)
        if gate is None or gate.is_drained or gate.is_closed:
            return
        try:
            self._boot_worker_reconcile_timer = self.set_interval(
                BOOT_WORKER_RECONCILE_INTERVAL_SECONDS,
                self._reconcile_boot_worker_slots,
            )
        except Exception:  # noqa: BLE001 -- boot never dies on a backstop
            self.loguru_logger.opt(exception=True).debug(
                "Could not arm the staggered boot worker reconcile"
            )

    def _reconcile_boot_worker_slots(self) -> None:
        """Release slots held by workers that already finished, then advance."""
        gate = getattr(self, "_boot_worker_gate", None)
        if gate is None:
            self._stop_boot_worker_reconcile()
            return
        released = False
        for key, worker in list(self._boot_worker_handles.items()):
            finished = bool(getattr(worker, "is_finished", False)) or bool(
                getattr(worker, "is_cancelled", False)
            )
            if not finished:
                continue
            self._boot_worker_handles.pop(key, None)
            released = gate.complete(key) or released
        if released or gate.pending:
            self._admit_staggered_boot_workers()
        if gate.is_drained or gate.is_closed:
            self._stop_boot_worker_reconcile()

    def _stop_boot_worker_reconcile(self) -> None:
        """Stop the reconcile timer if one is running."""
        timer = getattr(self, "_boot_worker_reconcile_timer", None)
        if timer is None:
            return
        self._boot_worker_reconcile_timer = None
        try:
            timer.stop()
        except Exception:  # noqa: BLE001 -- teardown must not raise
            pass

    def _close_boot_worker_gate(self, reason: str) -> None:
        """Stop admitting staggered boot workers (quit/shutdown).

        Whatever never started is not lost: each staggered member is either
        re-run by the surface that gates on it (the actor-pack pair) or
        resumes from a frontier in its own database on the next boot (both
        FTS backfills). Workers already in flight are cancelled by the normal
        shutdown path, not here.

        Args:
            reason: Logged, so a quit-time drop is explainable.
        """
        self._stop_boot_worker_reconcile()
        gate = getattr(self, "_boot_worker_gate", None)
        if gate is None or gate.is_closed:
            return
        dropped = gate.close()
        if dropped:
            self.loguru_logger.debug(
                f"Staggered boot workers not started before {reason}: "
                f"{', '.join(dropped)} (each resumes or re-runs on demand)"
            )

    def _schedule_launch_wake(self) -> None:
        """Discover saved child results in existing history and audit before waking.

        ADR-135 makes durable attempts and causal lineage the authority. Unseen
        badges are only a projection. An absent or empty runs database does not
        construct the Console runtime; the existing autowake switch still gates
        automatic launch work.
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
        scheduler -- started earlier in post-mount setup -- launched moments ago
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
        try:
            coordinator = getattr(self, "watchlists_operation_coordinator", None)
            if coordinator is None:
                return
            reconciled = await coordinator.reconcile_startup(boundary)
        except Exception as exc:  # noqa: BLE001 - a launch never dies on this
            self.loguru_logger.warning(
                f"Startup reconcile of interrupted subscriptions work failed "
                f"type={type(exc).__name__}"
            )
            return
        if any(reconciled.values()):
            self.loguru_logger.info(
                f"Startup reconcile failed interrupted subscriptions work: {reconciled}"
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
        from tldw_chatbook.Chat.citation_trace_repository import CitationTraceRepository

        repository = getattr(coordinator, "trace_repository", None)
        db = getattr(repository, "db", None)
        retire = (
            type(coordinator) is CitationArtifactOwnershipCoordinator
            and type(coordinator.artifact_store) is LocalChatbookService
            and type(repository) is CitationTraceRepository
            and type(db) is CharactersRAGDB
            and not db.is_memory_db
        )
        reconcile = coordinator.reconcile_pending

        def reconcile_in_worker():
            try:
                return reconcile(limit=25)
            finally:
                if retire:
                    db.close_connection()

        try:
            result = await asyncio.to_thread(reconcile_in_worker)
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
                from tldw_chatbook.Chat.citation_legacy_migration import (
                    CitationLegacyMigrationService,
                )
                from tldw_chatbook.Chat.citation_trace_repository import (
                    CitationTraceRepository,
                )

                repository = getattr(migration, "repository", None)
                db = getattr(migration, "db", None)
                retire = (
                    type(migration) is CitationLegacyMigrationService
                    and type(repository) is CitationTraceRepository
                    and repository.db is db
                    and type(db) is CharactersRAGDB
                    and not db.is_memory_db
                )
                migrate = migration.migrate_idle_unit

                def migrate_in_worker():
                    try:
                        return migrate()
                    finally:
                        if retire:
                            db.close_connection()

                try:
                    result = await asyncio.to_thread(migrate_in_worker)
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
        """Wire the status-line DB-size updates after UI readiness.

        task-21133: this used to arm a second pair of timers -- a 0.5 s
        one-shot and a 10 s interval -- for a token counter whose entire
        consumer surface task-17653 removed. Nothing armed the footer's
        ``#footer-token-count`` chip any more (``BaseAppScreen`` composes
        every ``AppFooterStatus`` with ``show_token_count=False``, and that
        is the only construction site in the package), so each tick resolved
        the active footer, ran three ``query_one`` selectors that no live
        screen composes, and threw the answer away in a debug log. The
        interval, its handle, and the whole chain behind it are gone; the
        DB-size timers below are unchanged.
        """

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
            # logged two tracebacks per fresh install and left the DB-size
            # timers never started for the whole session).
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
        except Exception as e_db_size:
            self.loguru_logger.opt(exception=True).error(
                f"Error setting up DB size indicator for the shell status line: {e_db_size}",
            )

    # U/U2 (TASK-33011): speech initialization, delegated to ``app_speech``.
    def _start_deferred_audio_service_initialization(self) -> None:
        return _speech()._start_deferred_audio_service_initialization(self)

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

        TASK-21113 added pacing BETWEEN routes: after each import the thread
        hands the event loop back a slice proportional to what it just took
        (see ``SCREEN_PREIMPORT_YIELD_RATIO`` and friends), and parks
        entirely while a screen navigation is resolving. Both are strictly
        between-route, so the single-route call this method also serves --
        task-21110's initial-screen warm-up, racing the splash -- reaches its
        one ``load_screen_class()`` with nothing added in front of it. The
        loop also drops out on ``_shutting_down`` so quit does not wait on a
        daemon thread's remaining registry.

        Args:
            routes: The routes to pre-import, in order. Factored out of
                ``_preimport_heavy_screens`` so tests can target one or two
                routes directly instead of the whole registry.
        """
        yield_ratio, max_gap = self._screen_preimport_pacing()
        previous_cost = 0.0
        for index, route in enumerate(routes):
            if index:
                self._pause_between_preimports(
                    min(previous_cost * yield_ratio, max_gap)
                )
            if getattr(self, "_shutting_down", False):
                return
            started = time.monotonic()
            try:
                route.load_screen_class()
            except Exception as exc:
                self.loguru_logger.debug(
                    "Screen pre-import failed (route={}, error_type={})",
                    route.screen_name,
                    type(exc).__name__,
                )
            previous_cost = time.monotonic() - started
        freeze_long_lived_heap("screen_preimport")  # ADR-198: imported screens are long-lived

    def _screen_preimport_pacing(self) -> tuple[float, float]:
        """``(yield_ratio, max_gap_seconds)`` for the between-route pause.

        One helper so the core-count question is answered in exactly one
        place. It governs the SPECULATIVE whole-registry pass only, and
        deliberately not task-21110's initial-screen warm-up, which shares
        ``_preimport_screens`` but passes a single route: that import is work
        the boot is certainly going to pay either way, and moving it off the
        event loop is worth more, not less, on a slow machine (task-21110
        measured splash-close-to-usable -46% on a cold first boot). Slowing
        or skipping it would put a certain cost back on the loop to avoid a
        speculative one. A single-route list has no between-route gap, so
        that separation needs no branch.
        """
        if _usable_cpu_count() < SCREEN_PREIMPORT_LOW_CORE_THRESHOLD:
            return (
                SCREEN_PREIMPORT_LOW_CORE_YIELD_RATIO,
                SCREEN_PREIMPORT_LOW_CORE_MAX_ROUTE_GAP_SECONDS,
            )
        return (SCREEN_PREIMPORT_YIELD_RATIO, SCREEN_PREIMPORT_MAX_ROUTE_GAP_SECONDS)

    def _screen_navigation_in_progress(self) -> bool:
        """Whether a screen navigation currently holds the FIFO nav lock.

        Read from the pre-import thread, so it must not touch the loop:
        ``asyncio.Lock.locked()`` is a plain attribute read, and the
        attribute is read directly rather than through
        ``_screen_navigation_lock()`` so a probe never *constructs* a lock
        off-loop. Absent lock (nothing has navigated yet) means not
        navigating.
        """
        lock = getattr(self, "_screen_navigation_lock_instance", None)
        if lock is None:
            return False
        try:
            return bool(lock.locked())
        except Exception:
            return False

    def _pause_between_preimports(self, gap_seconds: float) -> None:
        """Yield the CPU between two route imports, then wait out any nav.

        Runs on the pre-import daemon thread. The park is bounded by
        ``SCREEN_PREIMPORT_NAVIGATION_PARK_LIMIT_SECONDS`` and abandoned
        immediately on ``_shutting_down`` so neither a wedged navigation nor
        a quit can leave this thread sleeping in a loop.

        The gap sleep itself is sliced into navigation-poll-sized steps with
        a ``_shutting_down`` check between slices (TASK-22214, the 22200
        ``_interruptible_sleep`` precedent): with the caps at 2.0 s / 6.0 s
        a single ``time.sleep(gap)`` would leave a quit waiting out the
        whole gap before ``_preimport_screens``'s own shutdown check could
        run. Sliced, the thread notices a quit within one 0.05 s slice.
        """
        remaining = gap_seconds
        while remaining > 0:
            if getattr(self, "_shutting_down", False):
                return
            step = min(SCREEN_PREIMPORT_NAVIGATION_POLL_SECONDS, remaining)
            time.sleep(step)
            remaining -= step
        # Counted, not accumulated: summing 0.05 a hundred times lands either
        # side of 5.0 depending on float rounding, which would make the bound
        # off by one at random.
        polls = 0
        while (
            polls < SCREEN_PREIMPORT_MAX_NAVIGATION_POLLS
            and not getattr(self, "_shutting_down", False)
            and self._screen_navigation_in_progress()
        ):
            time.sleep(SCREEN_PREIMPORT_NAVIGATION_POLL_SECONDS)
            polls += 1

    def _preimport_heavy_screens(self) -> None:
        """Warm ``sys.modules`` for every registered screen route.

        See ``_preimport_screens`` for the per-route mechanics; this just
        supplies the full, priority-ordered route list.
        """
        self._preimport_screens(self._screen_preimport_route_order())

    def _initial_screen_preimport_route(self) -> ScreenRoute | None:
        """The route whose module ``_push_initial_screen`` is about to import.

        Resolved through ``resolve_screen_route()`` -- the same alias /
        shell-destination lookup ``_push_initial_screen`` itself goes through
        via ``resolve_screen_target()``, minus the ``load_screen_class()``
        call that would do the import here, on the loop, which is the whole
        thing being avoided. If the two ever disagree the warm-up simply
        warms the wrong module and the real push pays its import as it does
        today; it can never push a different screen.

        Returns ``None`` when the configured target is not routable, in which
        case there is nothing to warm: ``_push_initial_screen`` handles that
        case by falling back to chat, and reproducing that fallback here
        would duplicate a rare error path for no measurable gain.
        """
        try:
            return resolve_screen_route(self._resolve_initial_shell_route())
        except Exception as exc:
            self.loguru_logger.debug(
                "Initial-screen pre-import route resolution failed (error_type={})",
                type(exc).__name__,
            )
            return None

    def _schedule_initial_screen_preimport(self) -> None:
        """Warm the initial screen's module while the splash is still up.

        task-21110. Boot with the splash enabled (the default) is strictly
        serial: the splash owns the event loop for its full duration, and only
        when it closes does ``_push_initial_screen`` synchronously
        ``import_module`` the initial route's module on that same loop --
        measured at 0.31s warm and 0.94s on a first boot after an upgrade,
        for the 306 in-package modules chat_screen adds on top of the
        636-module boot closure. The existing pre-importer cannot
        help: it is armed by ``_schedule_deferred_startup_work`` at the tail of
        ``_post_mount_setup``, which itself only runs *after* that push.

        This moves a start time, not machinery: the work is the exact
        ``_preimport_screens`` body the whole-registry pass already uses, with
        its per-module-lock race semantics (a real navigation racing this
        thread blocks on CPython's own import lock and then finds the finished
        module in ``sys.modules``; a failed import is never cached, so the real
        push fails identically to today). Worst case if the user skips the
        splash mid-import, the push blocks on that same lock -- no worse than
        the synchronous import it replaces.

        Gated on ``_screen_preimport_enabled()`` so the pre-import feature has
        exactly one on/off switch (``TLDW_SCREEN_PREIMPORT``, default off under
        pytest), and re-checked against ``splash_screen_active`` because a
        keypress can close the splash inside the scheduling delay -- past that
        point the push either already happened or is imminent, and a second
        thread would only contend with it.

        Deliberately NOT gated on core count (TASK-21113). The two
        pre-importers share ``_preimport_screens`` and one enable switch, but
        the core-count question has opposite answers for them: the
        whole-registry pass is speculative work for screens the user may
        never open, so a slow machine should be throttled; this one is the
        initial screen's own import, which the boot pays either way, and
        moving it off the event loop is worth *more* on a slow machine
        (task-21110 measured close-to-usable -46% on a cold first boot).
        Throttling it would put a certain cost back on the loop to dodge a
        speculative one. See ``_screen_preimport_pacing``.
        """
        if not self._screen_preimport_enabled():
            return
        if self._shutting_down:
            return
        if self._initial_screen_preimport_thread is not None:
            return
        if not self.splash_screen_active:
            return
        if getattr(self, "_initial_screen_pushed", False):
            return
        route = self._initial_screen_preimport_route()
        if route is None:
            return
        thread = threading.Thread(
            target=self._preimport_screens,
            args=((route,),),
            name="tldw-initial-screen-preimport",
            daemon=True,
        )
        if not self._start_preimport_thread(thread):
            return
        self._initial_screen_preimport_thread = thread

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
        if not self._start_preimport_thread(thread):
            return
        self._screen_preimport_thread = thread

    def _start_preimport_thread(self, thread: threading.Thread) -> bool:
        """Start a pre-import thread; report whether it is running.

        Args:
            thread: The unstarted daemon thread to run.

        Returns:
            ``True`` when the thread started. ``False`` when the interpreter
            refused to spawn it -- thread exhaustion, or a start during
            interpreter shutdown -- in which case the caller must NOT record a
            handle.

        Both callers run from the splash-path timer/deferred-startup callback,
        not a request/response path, so a ``RuntimeError`` out of ``start()``
        would surface as an unhandled exception in a Textual timer task during
        boot. Losing a speculative warm-up is the correct outcome there: every
        module this would have pre-imported is still imported normally on
        first navigation. Recording the handle only after a successful start
        also keeps the once-guard honest -- a failed attempt leaves ``None``,
        so a later call can try again.
        """
        try:
            thread.start()
        except RuntimeError as exc:
            self.loguru_logger.debug(
                "Screen pre-import thread could not start (name={}, error_type={})",
                thread.name,
                type(exc).__name__,
            )
            return False
        return True

    # U/U2 (TASK-33011): speech initialization, delegated to ``app_speech``.
    def _schedule_tts_initialization(self) -> None:
        return _speech()._schedule_tts_initialization(self)

    def _schedule_stts_initialization(self) -> None:
        return _speech()._schedule_stts_initialization(self)

    def _speech_initialization_allowed(self, kind: str) -> bool:
        return _speech()._speech_initialization_allowed(self, kind)

    # U/U2 (TASK-33011): speech delivery/initialization admission and the
    # TTS/STTS handler initializers, delegated to ``app_speech``.
    def _speech_delivery_close_admission(self) -> None:
        return _speech()._speech_delivery_close_admission(self)

    async def _speech_delivery_drain(self, deadline: float) -> bool:
        return await _speech()._speech_delivery_drain(self, deadline)

    def _speech_delivery_resume(self) -> None:
        return _speech()._speech_delivery_resume(self)

    def _defer_speech_playback(self, event: TTSPlaybackEvent) -> None:
        return _speech()._defer_speech_playback(self, event)

    def _post_speech_delivery(self, event) -> bool:
        return _speech()._post_speech_delivery(self, event)

    async def _settle_speech_delivery(self, event, deliver) -> None:
        return await _speech()._settle_speech_delivery(self, event, deliver)

    def _speech_initialization_close_admission(self) -> None:
        return _speech()._speech_initialization_close_admission(self)

    async def _speech_initialization_drain(self, deadline: float) -> bool:
        return await _speech()._speech_initialization_drain(self, deadline)

    def _speech_initialization_resume(self) -> None:
        return _speech()._speech_initialization_resume(self)

    async def _settle_speech_initialization(self) -> asyncio.CancelledError | None:
        return await _speech()._settle_speech_initialization(self)

    async def _run_speech_initialization(self, kind: str, initialize):
        return await _speech()._run_speech_initialization(self, kind, initialize)

    async def _initialize_tts_service(self):
        return await _speech()._initialize_tts_service(self)

    async def _initialize_tts_service_owned(self):
        return await _speech()._initialize_tts_service_owned(self)

    async def _initialize_stts_service(self):
        return await _speech()._initialize_stts_service(self)

    async def _initialize_stts_service_owned(self):
        return await _speech()._initialize_stts_service_owned(self)

    async def _ensure_tts_handler(self):
        return await _speech()._ensure_tts_handler(self)

    async def _ensure_stts_handler(self):
        return await _speech()._ensure_stts_handler(self)

    def on_app_focus(self, event: AppFocus) -> None:
        """Forward a terminal focus regain to the active screen that wants it.

        ``AppFocus`` is declared ``bubble=False`` and the driver posts it ONLY
        to the App -- and events travel UP the DOM, never down, so a
        screen-level ``@on(AppFocus)`` handler can never fire. (task-13 review
        C1: one shipped as dead code precisely because its test called the
        handler method directly instead of posting the real event.) The App is
        therefore the only place this can be observed, and forwarding is the
        only way a screen can react to it.

        Duck-typed rather than isinstance-checked against ChatScreen: this
        stays a one-line opt-in for any future screen, and avoids importing a
        screen module into the app's hot import path. Never raises -- a focus
        event must not be able to take the app down.

        Args:
            event: The focus-regained event; not consumed, so Textual's own
                ``App._on_app_focus`` still runs (both the private framework
                handler and this public one are dispatched, from different
                classes in the MRO).
        """
        try:
            screen = self.screen
        except ScreenStackError:
            return
        notify = getattr(screen, "notify_terminal_focus_regained", None)
        if notify is None:
            return
        try:
            notify()
        except Exception:  # noqa: BLE001 -- a focus nudge must never crash the app
            logger.warning("app: terminal-focus-regained forwarding failed")

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

    ########################################################
    # --- End of Watchers and Helper Methods ---
    # ######################################################


# TASK-33011: the process entry points (early logging, the CSS build
# manifest, the argument parser, ``get_app``, ``main_cli_runner`` and the
# ``python -m`` body) live in ``tldw_chatbook.app_entry``. They run only from
# ``cli.py``'s lazy import or ``python -m``, never on the in-process boot path,
# so app.py must not import that module at module scope. ``__getattr__`` keeps
# ``from tldw_chatbook.app import <name>`` working for every moved name. Patch
# those names -- and anything their bodies call -- on ``app_entry``: a patch on
# this module no longer reaches them.
_APP_ENTRY_EXPORTS = frozenset(
    {
        "_BUNDLED_CSS_DECLARATION_RE",
        "_build_arg_parser",
        "_generated_css_is_stale",
        "_is_source_tree",
        "_load_css_build_manifest",
        "_save_css_build_manifest",
        "get_app",
        "initialize_early_logging",
        "main_cli_runner",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve a moved entry-point name from ``app_entry`` on first use (PEP 562)."""
    if name in _APP_ENTRY_EXPORTS:
        from tldw_chatbook import app_entry

        return getattr(app_entry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Defining App and Runtime identities for finite Console skill preparation.
_CONSOLE_SKILL_APP_SOURCE = (
    TldwCli,
    globals(),
    TldwCli._create_deferred_startup_task,
    TldwCli._create_deferred_startup_task.__code__,
    ConsoleRuntime,
)


# --- Main execution block ---
if __name__ == "__main__":
    # ``python -m tldw_chatbook.app``. app_entry imports ``TldwCli`` from
    # ``tldw_chatbook.app``; register this already-executed module under that
    # name first, so the import binds THIS module instead of executing app.py a
    # second time. Skipped when this namespace is not a registered module
    # (``runpy.run_module(..., alter_sys=False)``).
    _this_module = sys.modules.get(__name__)
    if _this_module is not None and vars(_this_module) is globals():
        sys.modules.setdefault("tldw_chatbook.app", _this_module)
    from tldw_chatbook.app_entry import _run_module_main

    _run_module_main()


#
# End of app.py
#######################################################################################################################
