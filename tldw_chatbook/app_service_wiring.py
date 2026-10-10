"""TldwCli's service composition: ``ServiceWiringMixin`` and its helpers.

Moved verbatim from ``app.py`` (TASK-33011, PR-D): the lazy service
properties and builders (cluster C), the ``_wire_*`` service composition that
``TldwCli.__init__`` runs (cluster E), and the misc wiring (cluster O: the
Notes-sync start observer, the watchlists command service, the briefing
schedules and the llama.cpp snapshot service), plus the module-level
Notes-sync and Collections-capture composition helpers they call.
``TldwCli`` mixes the class in before ``App``, and ``tldw_chatbook.app``
re-exports the helpers callers import from it.

Patch the names this code reads (``get_cli_setting``, ``get_user_data_dir``,
the ``get_*_db_path`` helpers, the service classes and so on) HERE: the bodies
resolve free names through this module's globals, so a patch on
``tldw_chatbook.app`` alone no longer reaches them. Where ``app.py`` still
reads the same name, patch both modules.
``Tests/Architecture/test_app_extracted_patch_targets.py`` fails on an
app-module patch that can only have been meant for code that moved out.
"""

import asyncio
import functools
import hashlib
import os
import threading
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Mapping, NamedTuple  # noqa: UP035

from loguru import logger
from textual.worker import Worker

from tldw_chatbook.Actor_Packs.activation import ActorPackActivationService
from tldw_chatbook.Actor_Packs.controller import ActorPackExportController
from tldw_chatbook.Actor_Packs.creation import ActorPackCreationService
from tldw_chatbook.Actor_Packs.export import ActorPackExportService
from tldw_chatbook.Actor_Packs.import_controller import ActorPackImportController
from tldw_chatbook.Actor_Packs.importer import (
    ActorPackImportError,
    ActorPackImportService,
)
from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Audio_Services_Interop import (
    AudioServicesScopeService,
    LocalAudioServicesService,
    ServerAudioServicesService,
)
from tldw_chatbook.Auth_Account_Interop import (
    AuthAccountScopeService,
    ServerAuthAccountService,
)
from tldw_chatbook.Character_Chat.character_persona_scope_service import (
    CharacterPersonaScopeService,
)
from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import (
    ChatDictionaryScopeService,
)
from tldw_chatbook.Character_Chat.server_character_persona_service import (
    ServerCharacterPersonaService,
)
from tldw_chatbook.Character_Chat.server_chat_dictionary_service import (
    ServerChatDictionaryService,
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
from tldw_chatbook.Chat.server_chat_conversation_service import (
    ServerChatConversationService,
)
from tldw_chatbook.Chat_Grammars_Interop import (
    ChatGrammarsScopeService,
    LocalChatGrammarsService,
    ServerChatGrammarsService,
)
from tldw_chatbook.Chatbooks import LocalChatbookService, ServerChatbookService
from tldw_chatbook.Claims_Interop import ClaimsScopeService, ServerClaimsService
from tldw_chatbook.Collections_Interop import (
    CollectionsFeedsScopeService,
    ServerCollectionsFeedsService,
)
from tldw_chatbook.Companion_Interop import (
    CompanionScopeService,
    ServerCompanionService,
)
from tldw_chatbook.config import (
    CLI_APP_CLIENT_ID,
    LOCAL_PROVIDERS,
    get_chachanotes_db_path,
    get_cli_setting,
    get_dreams_db_path,
    get_library_collections_db_path,
    get_media_db_path,
    get_notes_sync_recovery_capacity_bytes,
    get_notes_sync_state_db_path,
    get_notes_sync_watcher_intervals,
    get_notifications_db_path,
    get_prompts_db_path,
    get_research_db_path,
    get_scheduled_tasks_db_path,
    get_subscriptions_db_path,
    get_user_data_dir,
    get_workspaces_db_path,
    get_writing_db_path,
)
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator
from tldw_chatbook.Evaluations_Interop import (
    EvaluationScopeService,
    LocalEvaluationsService,
    ServerEvaluationsService,
)
from tldw_chatbook.External_Connectors_Interop import (
    ConnectorsScopeService,
    ServerConnectorsService,
)
from tldw_chatbook.Feedback_Interop import (
    FeedbackScopeService,
    LocalFeedbackService,
    ServerFeedbackService,
)
from tldw_chatbook.Home.active_work_adapter import (
    LocalNotificationHomeActiveWorkAdapter,
)
from tldw_chatbook.Kanban_Interop import (
    KanbanScopeService,
    LocalKanbanService,
    ServerKanbanService,
)
from tldw_chatbook.Library import LocalLibraryCollectionsService
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)
from tldw_chatbook.LLM_Provider_Catalog import (
    LLMProviderCatalogScopeService,
    LocalLLMProviderCatalogService,
    ServerLLMProviderCatalogService,
)
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)
from tldw_chatbook.MCP_Governance_Interop import (
    MCPGovernanceScopeService,
    ServerMCPGovernanceService,
)
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingBackend
from tldw_chatbook.Meetings_Interop import MeetingsScopeService, ServerMeetingsService
from tldw_chatbook.Notes.notes_scope_service import ScopeType
from tldw_chatbook.Notifications import (
    ClientNotificationsDB,
    ClientNotificationsService,
    NotificationDispatchService,
    NotificationsScopeService,
    ServerNotificationsService,
)
from tldw_chatbook.Outputs_Interop import OutputsScopeService, ServerOutputsService
from tldw_chatbook.Personalization_Interop import (
    PersonalizationScopeService,
    ServerPersonalizationService,
)
from tldw_chatbook.Prompt_Management import (
    LocalPromptService,
    PromptChatbookScopeService,
    ServerPromptService,
)
from tldw_chatbook.Prompt_Management import Prompts_Interop as prompts_interop
from tldw_chatbook.Prompt_Studio_Interop import (
    PromptStudioScopeService,
    ServerPromptStudioService,
)
from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService
from tldw_chatbook.RAG_Admin.rag_admin_scope_service import RAGAdminScopeService
from tldw_chatbook.RAG_Admin.server_rag_admin_service import ServerRAGAdminService
from tldw_chatbook.Research_Interop import (
    LocalResearchSearchService,
    LocalResearchService,
    ResearchScopeService,
    ResearchSearchScopeService,
    ServerResearchSearchService,
    ServerResearchService,
)
from tldw_chatbook.Research_Workspace.paste_staging import ResearchPasteStagingStore
from tldw_chatbook.Research_Workspace.source_association import (
    ResearchSourceAssociationCoordinator,
    ResearchSourceAssociationScheduler,
)
from tldw_chatbook.Research_Workspace.source_operation_store import (
    ResearchSourceOperationStore,
)
from tldw_chatbook.Research_Workspace.source_readiness import (
    ResearchSourceReadinessCoordinator,
)
from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client
from tldw_chatbook.runtime_policy.server_capabilities import (
    ActiveServerCapabilityService,
)
from tldw_chatbook.runtime_policy.server_context import (
    RuntimeServerContextProvider,
    default_server_credential_profile_id,
)
from tldw_chatbook.runtime_policy.server_credentials import (
    CredentialStoreUnavailable,
    UnavailableServerCredentialStore,
    build_default_server_credential_store,
)
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)
from tldw_chatbook.Scheduling.constants import (
    HANDLER_TIMEOUT_SECONDS,
    MISSED_FIRE_GRACE_SECONDS,
    SCHEDULER_POLL_INTERVAL_SECONDS,
)
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler import (
    BriefingJobHandler,
)
from tldw_chatbook.Scheduling.scheduler.handlers.reminder_handler import ReminderHandler
from tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler import (
    WatchlistCheckHandler,
)
from tldw_chatbook.Scheduling.scheduler.loop import Handler, SchedulerLoop
from tldw_chatbook.Scheduling.services.briefing_projection import BriefingProjection
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import SchedulingServerClient
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection
from tldw_chatbook.Server_Runtime_Interop import (
    ServerRuntimeScopeService,
    ServerRuntimeService,
)
from tldw_chatbook.Sharing_Interop import ServerSharingService, SharingScopeService
from tldw_chatbook.Skills_Interop import (
    LocalSkillsService,
    ServerSkillsService,
    SkillsScopeService,
    SkillTrustService,
    default_local_skills_store_dir,
)
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    MARKER_FILENAME as _SKILL_TRUST_MARKER_FILENAME,
)
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    SkillTrustStore,
    build_default_skill_trust_key_cache,
    build_skill_trust_marker_store_with_fallback,
    default_trust_store_dir,
    skill_trust_account_scope,
)
from tldw_chatbook.Study_Interop import (
    LocalQuizService,
    LocalStudyService,
    QuizScopeService,
    ServerQuizService,
    ServerStudyService,
    StudyScopeService,
)
from tldw_chatbook.Subscriptions import (
    LocalWatchlistsService,
    ServerWatchlistsService,
    WatchlistScopeService,
)
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.Sync_Interop import (
    LocalFirstSyncService,
    ManualSyncControlService,
    ServerSyncService,
    SyncRestoreService,
    SyncScopeService,
)
from tldw_chatbook.Text2SQL_Interop import ServerText2SQLService, Text2SQLScopeService
from tldw_chatbook.Tools_Interop import ServerToolsService, ToolsScopeService
from tldw_chatbook.Translation_Interop import (
    ServerTranslationService,
    TranslationScopeService,
)
from tldw_chatbook.User_Governance_Interop import (
    ServerUserGovernanceService,
    UserGovernanceScopeService,
)
from tldw_chatbook.Voice_Assistant_Interop import (
    ServerVoiceAssistantService,
    VoiceAssistantScopeService,
)
from tldw_chatbook.Web_Clipper_Interop import (
    ServerWebClipperService,
    WebClipperScopeService,
)
from tldw_chatbook.Web_Scraping_Interop import (
    ServerWebScrapingService,
    WebScrapingScopeService,
)
from tldw_chatbook.Workspaces import (
    ChangeReviewConsentService,
    DeferredWorkspaceToolProfileGuard,
    LocalWorkspaceRegistryService,
)
from tldw_chatbook.Writing_Interop import (
    LocalWritingService,
    ServerWritingService,
    WritingScopeService,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Library.collections_capture_models import (
        ExternalMediaReference,
        ExternalNoteReference,
        ExternalReferenceAvailability,
    )
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner
    from tldw_chatbook.Terminal.backend import TerminalBackend
    from tldw_chatbook.Terminal.session_manager import TerminalSessionManager
    from tldw_chatbook.tldw_api import MCPUnifiedClient
else:
    TerminalBackend = Any


class _TldwCliClassProxy:
    """Stand-in for ``TldwCli`` in the moved bodies.

    A few bodies call ``TldwCli.<method>(owner)`` on purpose, so fakes and
    class-level patches take effect. ``tldw_chatbook.app`` imports this module
    before it defines ``TldwCli``, so the class is looked up on each use.
    """

    def __getattr__(self, name: str) -> Any:
        from tldw_chatbook.app import TldwCli as app_class

        return getattr(app_class, name)


TldwCli = _TldwCliClassProxy()


def _read_app_raw_cli_permitted(app: object) -> bool:
    """Read the latest app config and accept only the literal boolean true."""
    config = getattr(app, "app_config", None)
    if not isinstance(config, Mapping):
        return False
    console = config.get("console")
    return isinstance(console, Mapping) and console.get("raw_cli_permitted") is True


def _disabled_builtin_skills(config: Any) -> frozenset[str]:
    """Built-in skills disabled in config; imported lazily (off boot path, ADR-097)."""
    from tldw_chatbook.Skills_Interop.builtin_skills import (
        disabled_builtins_from_config,
    )

    return disabled_builtins_from_config(config)


def _read_app_disabled_builtin_skills(app: object) -> frozenset[str]:
    """Read the current stock app config without invoking a UI callback."""
    return _disabled_builtin_skills(getattr(app, "app_config", None))


_STOCK_DISABLED_BUILTINS_READER = (
    _read_app_disabled_builtin_skills,
    _read_app_disabled_builtin_skills.__code__,
)


def _read_app_published_plugin_service(app: object) -> Any:
    """Return only the already resident plugin metadata owner."""
    return getattr(app, "_plugin_service", None)


_STOCK_PUBLISHED_PLUGIN_READER = (
    _read_app_published_plugin_service,
    _read_app_published_plugin_service.__code__,
)


def _build_terminal_backend() -> "TerminalBackend":
    """Build the supported platform backend without eager platform imports."""
    if os.name != "posix":
        raise OSError("persistent Terminal backend unavailable")
    from tldw_chatbook.Terminal.posix_backend import PosixTerminalBackend

    return PosixTerminalBackend()


def _active_notes_sync_server_profile_id(app: Any) -> str:
    """Return the authoritative server profile eligible for Notes Sync.

    Args:
        app: Application composition owner.

    Returns:
        Active server profile identity, or an empty string for local runtime.
    """

    runtime_state = getattr(getattr(app, "runtime_policy", None), "state", None)
    server_is_authoritative = runtime_state is None or (
        getattr(runtime_state, "active_source", None) == "server"
    )
    if not server_is_authoritative:
        return ""
    return str(
        getattr(app, "active_server_id", None)
        or getattr(runtime_state, "active_server_id", None)
        or ""
    ).strip()


class _DeferredNotesSyncFacade:
    """Load deferred Notes organization wiring on first real collaborator use.

    Args:
        app: Application composition owner.
        target_path: Attribute path to the real collaborator after wiring.
        lock: Shared re-entrant lock for the facade group.
    """

    def __init__(
        self,
        app: Any,
        target_path: tuple[str, ...],
        lock: Any,
    ) -> None:
        self._app = app
        self._target_path = target_path
        self._lock = lock

    def _target(self) -> Any:
        with self._lock:
            _wire_notes_sync_services(self._app)
            target = self._app
            for attribute in self._target_path:
                target = getattr(target, attribute, None)
            if target is None or target is self:
                raise RuntimeError("notes_organization_sync_unavailable")
            return target

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self._target(), attribute)


def _install_deferred_notes_sync_facades(app: Any) -> bool:
    """Protect the post-ready delay with first-use Notes Sync wiring.

    Args:
        app: Application composition owner whose collaborators are deferred.

    Returns:
        True when deferred facades are installed or already active.
    """

    if not _active_notes_sync_server_profile_id(app):
        return False
    notes_scope_service = getattr(app, "notes_scope_service", None)
    if (
        getattr(app, "chachanotes_db", None) is None
        or getattr(app, "sync_state_repository", None) is None
        or notes_scope_service is None
    ):
        return False
    current = getattr(app, "notes_organization_sync_service", None)
    if current is not None:
        return isinstance(current, _DeferredNotesSyncFacade)

    lock = threading.RLock()
    repository = _DeferredNotesSyncFacade(
        app,
        ("notes_organization_repository",),
        lock,
    )
    service = _DeferredNotesSyncFacade(
        app,
        ("notes_organization_sync_service",),
        lock,
    )
    producer = _DeferredNotesSyncFacade(
        app,
        ("notes_scope_service", "sync_v2_notes_producer"),
        lock,
    )
    app.notes_organization_repository = repository
    app.notes_organization_sync_service = service
    notes_scope_service.sync_v2_notes_producer = producer
    notes_scope_service.organization_sync_service = service
    local_notes = getattr(notes_scope_service, "local_notes_service", None)
    if local_notes is not None:
        local_notes.organization_sync_service = service
    local_chat = getattr(app, "local_chat_conversation_service", None)
    if local_chat is not None:
        local_chat.organization_sync_service = service
    local_first = getattr(app, "local_first_sync_service", None)
    if local_first is not None:
        local_first.notes_organization_repository = repository
        local_first.notes_organization_sync_service = service
    restore = getattr(app, "sync_restore_service", None)
    if restore is not None:
        restore.notes_organization_repository = repository
    manual = getattr(app, "manual_sync_control_service", None)
    if manual is not None:
        manual.notes_organization_sync_service = service
        manual.notes_repository = repository
    return True


def _wire_notes_sync_services(app: Any) -> None:
    """Finish Notes Sync composition after both SQLite owners exist."""

    from tldw_chatbook.Notes.agent_lessons import initialize_agent_lessons_folder
    from tldw_chatbook.Notes.notes_organization_repository import (
        NotesOrganizationRepository,
    )
    from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
        NotesOrganizationSyncService,
    )
    from tldw_chatbook.Sync_Interop.notes_outbox_producer import (
        NotesSyncV2OutboxProducer,
    )

    notes_db = getattr(app, "chachanotes_db", None)
    state_repository = getattr(app, "sync_state_repository", None)
    notes_scope_service = getattr(app, "notes_scope_service", None)
    active_server_profile_id = _active_notes_sync_server_profile_id(app)
    if not active_server_profile_id:
        app.notes_organization_repository = None
        app.notes_organization_sync_service = None
        if notes_scope_service is not None:
            notes_scope_service.organization_sync_service = None
        local_notes = getattr(notes_scope_service, "local_notes_service", None)
        if local_notes is not None:
            local_notes.organization_sync_service = None
        local_chat = getattr(app, "local_chat_conversation_service", None)
        if local_chat is not None:
            local_chat.organization_sync_service = None
        local_first = getattr(app, "local_first_sync_service", None)
        if local_first is not None:
            local_first.notes_organization_repository = None
            local_first.notes_organization_sync_service = None
        restore = getattr(app, "sync_restore_service", None)
        if restore is not None:
            restore.notes_organization_repository = None
        manual = getattr(app, "manual_sync_control_service", None)
        if manual is not None:
            manual.notes_organization_sync_service = None
            manual.notes_repository = None
        if notes_db is not None:
            initialize_agent_lessons_folder(
                notes_db,
                scope_mode="local_only",
                profile_id="local",
                dataset_id="local",
            )
        return
    if notes_db is None or state_repository is None or notes_scope_service is None:
        return
    repository = getattr(app, "notes_organization_repository", None)
    if isinstance(repository, _DeferredNotesSyncFacade):
        repository = None
    if (
        repository is None
        or getattr(repository, "db", None) is not notes_db
        or getattr(repository, "server_profile_id", None) != active_server_profile_id
    ):
        repository = NotesOrganizationRepository(
            notes_db,
            server_profile_id=active_server_profile_id,
        )
    producer = NotesSyncV2OutboxProducer(
        state_repository=state_repository,
        dataset_keys=getattr(app, "sync_v2_dataset_keys", {}),
        notes_db=notes_db,
    )
    organization_service = NotesOrganizationSyncService(
        notes_repository=repository,
        state_repository=state_repository,
        notes_producer=producer,
    )
    app.notes_organization_repository = repository
    app.notes_organization_sync_service = organization_service
    notes_scope_service.sync_v2_notes_producer = producer
    notes_scope_service.organization_sync_service = organization_service
    local_notes = getattr(notes_scope_service, "local_notes_service", None)
    if local_notes is not None:
        local_notes.organization_sync_service = organization_service
    local_chat = getattr(app, "local_chat_conversation_service", None)
    if local_chat is not None:
        local_chat.organization_sync_service = organization_service
    local_first = getattr(app, "local_first_sync_service", None)
    if local_first is not None:
        local_first.notes_organization_repository = repository
        local_first.notes_organization_sync_service = organization_service
    restore = getattr(app, "sync_restore_service", None)
    if restore is not None:
        restore.notes_organization_repository = repository
    manual = getattr(app, "manual_sync_control_service", None)
    if manual is not None:
        manual.notes_organization_sync_service = organization_service
        manual.notes_repository = repository

    for profile in state_repository.list_sync_v2_profile_states():
        dataset_id = str(profile.get("dataset_id") or "")
        if (
            profile.get("server_profile_id") != active_server_profile_id
            or profile.get("profile_mode") != "local_first"
            or not dataset_id
        ):
            continue
        seed = notes_db.get_connection().execute(
            "SELECT state FROM agent_lessons_seed_state WHERE profile_id = ? "
            "AND dataset_id = ?",
            (active_server_profile_id, dataset_id),
        ).fetchone()
        if seed is not None and seed["state"] != "unknown":
            organization_service.initialize_agent_lessons_seed(
                server_profile_id=active_server_profile_id,
                dataset_id=dataset_id,
            )


def _external_reference_payload_id(payload: Any, *keys: str) -> str | None:
    """Read one owner identifier without trusting a response's concrete type."""
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        values = payload
    else:
        model_dump = getattr(payload, "model_dump", None)
        values = model_dump(mode="json") if callable(model_dump) else None
    for key in keys:
        value = values.get(key) if isinstance(values, Mapping) else getattr(payload, key, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _external_reference_failure_reason(error: Exception, owner: str) -> str:
    """Map owner failures to bounded provenance reasons."""
    status_code = getattr(error, "status_code", None)
    if status_code in {401, 403} or isinstance(error, PermissionError):
        return f"{owner}_reference_unauthorized"
    if status_code == 404 or isinstance(error, (KeyError, LookupError)):
        return f"{owner}_reference_missing"
    return "reference_resolution_retryable"


def _collections_reference_authority_kind(app: Any, authority_key: str) -> str | None:
    """Return the active owner kind for an exact opaque capture authority."""
    for kind, attribute in (
        ("local", "local_collections_capture_authority"),
        ("server", "server_collections_capture_authority"),
    ):
        authority = getattr(app, attribute, None)
        if authority is not None and getattr(authority, "key", None) == authority_key:
            return kind
    return None


async def _resolve_collections_media_reference(
    app: Any,
    reference: "ExternalMediaReference",
) -> "ExternalReferenceAvailability":
    """Resolve a capture's stored backing-Media identity through its owner."""
    from tldw_chatbook.Library.collections_capture_models import (
        ExternalReferenceAvailability,
    )

    kind = _collections_reference_authority_kind(app, reference.authority_key)
    if kind is None:
        return ExternalReferenceAvailability(
            "unavailable", "media_reference_authority_mismatch"
        )
    mode = (
        MediaReadingBackend.LOCAL
        if kind == "local"
        else MediaReadingBackend.SERVER
    )
    try:
        payload = await app.media_reading_scope_service.get_backing_media_item(
            mode=mode,
            media_id=reference.item_id,
            include_content=False,
            include_versions=False,
        )
    except Exception as exc:
        return ExternalReferenceAvailability(
            "unavailable", _external_reference_failure_reason(exc, "media")
        )
    resolved_id = _external_reference_payload_id(payload, "id", "media_id")
    if resolved_id is None:
        return ExternalReferenceAvailability("unavailable", "media_reference_missing")
    if resolved_id != reference.item_id:
        return ExternalReferenceAvailability(
            "unavailable", "media_reference_identity_mismatch"
        )
    return ExternalReferenceAvailability("available")


async def _resolve_collections_note_reference(
    app: Any,
    reference: "ExternalNoteReference",
) -> "ExternalReferenceAvailability":
    """Resolve a capture's note link through Local or Server Notes only."""
    from tldw_chatbook.Library.collections_capture_models import (
        ExternalReferenceAvailability,
    )

    kind = _collections_reference_authority_kind(app, reference.authority_key)
    if kind is None:
        return ExternalReferenceAvailability(
            "unavailable", "note_reference_authority_mismatch"
        )
    scope = ScopeType.LOCAL_NOTE if kind == "local" else ScopeType.SERVER_NOTE
    try:
        payload = await app.notes_scope_service.get_note_detail(
            scope=scope,
            note_id=reference.note_id,
            user_id=getattr(app, "notes_user_id", None) if kind == "local" else None,
        )
    except Exception as exc:
        return ExternalReferenceAvailability(
            "unavailable", _external_reference_failure_reason(exc, "note")
        )
    resolved_id = _external_reference_payload_id(payload, "id", "note_id")
    if resolved_id is None:
        return ExternalReferenceAvailability("unavailable", "note_reference_missing")
    if resolved_id != reference.note_id:
        return ExternalReferenceAvailability(
            "unavailable", "note_reference_identity_mismatch"
        )
    return ExternalReferenceAvailability("available")


def _extract_collections_article(url: str) -> Mapping[str, Any]:
    """Fetch one capture through the existing guarded article extractor."""
    from tldw_chatbook.Local_Ingestion.web_article_ingestion import (
        extract_article_for_ingest,
    )
    from tldw_chatbook.Utils.egress import UrlProvenance

    # A capture URL is what the user typed into quick-capture (TASK-20973).
    return extract_article_for_ingest(
        url, {}, url_provenance=UrlProvenance.USER_ENTERED
    )


class _DeferredCollectionsCaptureScope:
    """Stable app seam that composes the real capture scope on first use."""

    def __init__(self, owner: Any) -> None:
        object.__setattr__(self, "_owner", owner)

    def _resolve(self) -> Any:
        owner = object.__getattribute__(self, "_owner")
        scope = TldwCli.ensure_collections_capture_services(owner)
        if scope is None or scope is self:
            raise RuntimeError("collections_capture_scope_unavailable")
        return scope

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._resolve(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._resolve(), name)


class _CollectionsSetupObsolete(RuntimeError):
    """The original deferred initializer no longer owns publication."""


def _collections_setup_sources_current(sources):
    import inspect
    import sys

    for namespace, entries in sources:
        module = sys.modules.get(namespace["__name__"])
        if module is None or vars(module) is not namespace:
            return False
        for owner, name, expected, records in entries:
            actual = (
                owner.get(name)
                if type(owner) is dict  # noqa: E721 - exact stock compatibility boundary
                else inspect.getattr_static(owner, name, None)
            )
            if actual is not expected:
                return False
            for (
                function,
                code,
                defining,
                defaults,
                keywords,
                items,
                closure,
                cells,
                wrapped,
            ) in records:
                defining_module = sys.modules.get(defining["__name__"])
                if (
                    defining_module is None
                    or vars(defining_module) is not defining
                    or function.__code__ is not code
                    or function.__globals__ is not defining
                    or function.__defaults__ is not defaults
                    or function.__kwdefaults__ is not keywords
                    or len(keywords or {}) != len(items)
                    or any(
                        key not in (keywords or {}) or keywords[key] is not value
                        for key, value in items
                    )
                    or function.__closure__ is not closure
                    or any(cell.cell_contents is not value for cell, value in cells)
                    or vars(function).get("__wrapped__") is not wrapped
                ):
                    return False
    return True


def _capture_deferred_collections_setup(app):
    import inspect
    import sys
    from types import SimpleNamespace
    from tldw_chatbook import config
    from tldw_chatbook.DB import Library_Collections_DB
    from tldw_chatbook.Library import (
        collections_capture_repository,
        collections_capture_service,
        collections_legacy_recovery,
        collections_offline_store,
    )

    app_module = sys.modules["tldw_chatbook.app"]
    app_type, defining, factory, factory_code, _runtime_type = (
        app_module._CONSOLE_SKILL_APP_SOURCE
    )
    fields = vars(app)
    database = fields.get("local_library_collections_db")
    scope = fields.get("collections_capture_scope_service")
    if (
        defining is not vars(app_module)
        or app_module.TldwCli is not app_type
        or type(app) is not app_type
        or factory.__code__ is not factory_code
        or inspect.getattr_static(app, "_create_deferred_startup_task") is not factory
        or type(database) is not LibraryCollectionsDB
        or database.is_memory_db
        or type(scope) is not _DeferredCollectionsCaptureScope
        or fields.get("_collections_capture_initializer_closed", False)
        or fields.get("_shutting_down", False)
        or fields.get("_exit", False)
        or type(fields.get("_deferred_startup_tasks")) is not set  # noqa: E721 - exact stock compatibility boundary
        or any(
            fields.get(name) is not None for name in _COLLECTIONS_CAPTURE_RESULT_FIELDS
        )
    ):
        return None
    sources = (
        _COLLECTIONS_SETUP_SOURCE,
        config._COLLECTIONS_SETUP_SOURCE,
        Library_Collections_DB._COLLECTIONS_SETUP_SOURCE,
        collections_capture_repository._COLLECTIONS_SETUP_SOURCE,
        collections_capture_service._COLLECTIONS_SETUP_SOURCE,
        collections_legacy_recovery._COLLECTIONS_SETUP_SOURCE,
        collections_offline_store._COLLECTIONS_SETUP_SOURCE,
    )
    checker, checker_code = _COLLECTIONS_SOURCE_CHECKER
    if (
        _collections_setup_sources_current is not checker
        or checker.__code__ is not checker_code
        or not checker(sources)
    ):
        return None
    receivers = tuple(
        (app, name, original) for name, original in _COLLECTIONS_APP_METHODS
    )
    receivers += ((app, "_create_deferred_startup_task", factory),)
    receivers += tuple(
        (database, name, original)
        for owner, name, original, _records in Library_Collections_DB._COLLECTIONS_SETUP_SOURCE[
            1
        ]
        if owner is LibraryCollectionsDB
    )
    if any(
        inspect.getattr_static(owner, name, None) is not original
        for owner, name, original in receivers
    ):
        return None
    policy = app.runtime_policy
    if policy is None:
        return None
    request = SimpleNamespace(
        app=app,
        scope=scope,
        database=database,
        local=database._thread_local,
        path=database.db_path,
        participant=database._maintenance_participant,
        policy=policy,
        state=policy.state,
        identity=config.current_config_identity(),
        sources=sources,
        receivers=receivers,
        factory=factory,
        factory_code=factory_code,
        reads=set(),
        initializer=None,
        loop=asyncio.get_running_loop(),
        thread=threading.current_thread(),
    )
    require = _require_collections_setup_current
    checker = _collections_setup_sources_current
    require_code, checker_code = require.__code__, checker.__code__

    def current():
        if (
            globals().get("_require_collections_setup_current") is not require
            or require.__code__ is not require_code
            or globals().get("_collections_setup_sources_current") is not checker
            or checker.__code__ is not checker_code
        ):
            raise _CollectionsSetupObsolete("collections_capture_setup_source_changed")
        require(request)

    request.require_current = current
    return request


def _require_collections_setup_current(request):
    import inspect
    from tldw_chatbook import config

    app, database = request.app, request.database
    if (
        vars(app).get("_collections_capture_setup") is not request
        or vars(app).get("_collections_capture_initializer_task")
        is not request.initializer
        or type(request.initializer) is not _COLLECTIONS_TASK
        or request.initializer.get_loop() is not request.loop
        or request.factory.__code__ is not request.factory_code
        or vars(app).get("_collections_capture_initializer_closed", False)
        or vars(app).get("_shutting_down", False)
        or vars(app).get("_exit", False)
        or any(
            inspect.getattr_static(owner, name, None) is not original
            for owner, name, original in request.receivers
        )
        or app.collections_capture_scope_service is not request.scope
        or app.local_library_collections_db is not database
        or database._thread_local is not request.local
        or database.db_path != request.path
        or database._maintenance_participant is not request.participant
        or app.runtime_policy is not request.policy
        or request.policy.state is not request.state
        or not _collections_setup_sources_current(request.sources)
        or config.current_config_identity() != request.identity
        or any(
            vars(app).get(name) is not None
            for name in _COLLECTIONS_CAPTURE_RESULT_FIELDS
        )
    ):
        raise _CollectionsSetupObsolete("collections_capture_setup_changed")
    if threading.current_thread() is request.thread and (
        asyncio.get_running_loop() is not request.loop
        or asyncio.current_task() is not request.initializer
    ):
        raise _CollectionsSetupObsolete("collections_capture_initializer_changed")


def _build_deferred_collections_capture_parts(request):
    from tldw_chatbook.Backup_Recovery.participants import (
        _core_cached_connection,
        _core_closing,
    )
    from tldw_chatbook.Utils.private_paths import lexical_path

    request.require_current()
    database_path = get_library_collections_db_path()
    request.require_current()
    if lexical_path(database_path) != request.path:
        raise _CollectionsSetupObsolete(
            "collections_capture_database_selection_changed"
        )
    data_root = get_user_data_dir()
    request.require_current()
    database, local = request.database, request.local
    previous = _core_cached_connection(database, getattr(local, "conn", None))
    connection = None
    try:
        request.require_current()
        connection = database._held_connection()
        request.require_current()
        # Each original repository/filesystem call keeps its own admission.
        # A database-only outer scope would reject the separate archive root.
        return _collections_capture_parts(
            database,
            database_path,
            data_root,
            require_current=request.require_current,
        )
    finally:
        if connection is not None and connection is not previous:
            with _core_closing(database, connection) as allowed:
                if not allowed:
                    raise RuntimeError("collections_capture_connection_not_retired")
                connection.close()
                if getattr(local, "conn", None) is connection:
                    local.conn = None


async def _initialize_deferred_collections_capture(request):
    app = request.app
    try:
        parts = await _COLLECTIONS_RUN_PREPARATION(
            lambda: _build_deferred_collections_capture_parts(request),
            creator=app,
            session_id=None,
            reads=request.reads,
            require_current=request.require_current,
        )
        request.require_current()
        scope = _collections_capture_scope(app)
        service = _local_collections_capture_service(parts)
        request.require_current()
    except _CollectionsSetupObsolete:
        return
    except asyncio.CancelledError:
        raise
    except Exception:
        try:
            request.require_current()
        except _CollectionsSetupObsolete:
            return
        app.collections_capture_scope_service = _collections_capture_scope(app)
        logger.opt(exception=True).warning(
            "Local Collections capture service unavailable during app wiring"
        )
        return
    else:
        app.collections_capture_scope_service = scope
        _publish_collections_capture_parts(app, parts, service)
        app._create_deferred_startup_task(
            app._reconcile_collections_capture_startup(),
            name="deferred_collections_capture_reconciliation",
        )
    finally:
        if vars(app).get("_collections_capture_setup") is request:
            app._collections_capture_setup = None


async def _retire_deferred_collections_capture(app):
    app._collections_capture_initializer_closed = True
    task = vars(app).get("_collections_capture_initializer_task")
    cancellation = None
    if task is None:
        return None
    if not task.done():
        task.cancel()
    while not task.done():
        try:
            # wait does not propagate child cancellation or cancel the child.
            await asyncio.wait({task})
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
    if not task.cancelled():
        task.result()
    return cancellation


def _collections_capture_scope(app):
    from tldw_chatbook.Library.collections_capture_service import (
        CollectionsCaptureScopeService,
    )

    return CollectionsCaptureScopeService(
        resolve_media_reference=functools.partial(
            _resolve_collections_media_reference, app
        ),
        resolve_note_reference=functools.partial(
            _resolve_collections_note_reference, app
        ),
    )


def _collections_capture_parts(
    database, database_path, data_root, *, require_current=None
):
    from tldw_chatbook.Library.collections_capture_repository import (
        CollectionsCaptureRepository,
    )
    from tldw_chatbook.Library.collections_capture_service import (
        build_local_capture_authority,
    )
    from tldw_chatbook.Library.collections_legacy_recovery import (
        LegacyCollectionsRecovery,
        LegacyCollectionsRecoveryError,
    )
    from tldw_chatbook.Library.collections_offline_store import CollectionsOfflineStore

    if require_current is not None:
        require_current()
    authority = build_local_capture_authority(
        profile_id=str(data_root.resolve()),
        database_identity=str(database_path.resolve()),
    )
    if require_current is not None:
        require_current()
    repository = CollectionsCaptureRepository(database, authority_key=authority.key)
    if require_current is not None:
        require_current()
    offline_store = CollectionsOfflineStore(
        repository,
        data_root=data_root,
        authority_fingerprint=authority.fingerprint,
    )
    if require_current is not None:
        require_current()
    legacy_recovery = LegacyCollectionsRecovery(database)
    try:
        legacy_recovery.list_collections(page=1, size=1)
        legacy_recovery_available = True
    except LegacyCollectionsRecoveryError:
        legacy_recovery_available = False
    if require_current is not None:
        require_current()
    return (
        authority,
        repository,
        offline_store,
        legacy_recovery,
        legacy_recovery_available,
    )


def _local_collections_capture_service(parts):
    from tldw_chatbook.Library.collections_capture_service import (
        LocalCollectionsCaptureService,
    )

    authority, repository, offline_store, legacy_recovery, legacy_available = parts
    service = LocalCollectionsCaptureService(
        authority,
        repository,
        offline_store=offline_store,
        extractor=_extract_collections_article,
        legacy_recovery_available=legacy_available,
    )
    return service


def _publish_collections_capture_parts(app, parts, service):
    authority, repository, offline_store, legacy_recovery, _legacy_available = parts
    app.collections_capture_repository = repository
    app.collections_offline_store = offline_store
    app.collections_legacy_recovery_service = legacy_recovery
    app.local_collections_capture_authority = authority
    app.local_collections_capture_service = service
    TldwCli._activate_collections_capture_authority(app)


class ServiceWiringMixin:
    """TldwCli's service composition (clusters C, E and O; TASK-33011).

    A plain mixin: ``TldwCli.__init__`` still runs the ``_wire_*`` methods
    through ``self``, and it lists this class before ``App`` in its bases.
    """

    @property
    def terminal_session_manager(self) -> "TerminalSessionManager":
        """Return the single app-owned Terminal manager, creating it on use.

        Returns:
            The app-owned terminal session manager.
        """

        manager = self._terminal_session_manager
        if manager is not None:
            return manager
        with self._terminal_session_manager_lock:
            manager = self._terminal_session_manager
            if manager is None:
                from tldw_chatbook.Terminal.session_manager import (
                    TerminalSessionManager,
                )

                manager = TerminalSessionManager(
                    lambda: _read_app_raw_cli_permitted(self),
                    _build_terminal_backend,
                )
                self._terminal_session_manager = manager
        return manager

    @terminal_session_manager.setter
    def terminal_session_manager(self, manager: Any) -> None:
        """Replace the app-owned manager for lifecycle tests and adapters.

        Args:
            manager: Replacement terminal session manager.
        """

        self._terminal_session_manager = manager

    def _timed_init_task(self, task_name: str, func: Callable[..., Any], *args: Any):
        """Run one phase-3 initializer and record how long IT took.

        Args:
            task_name: Key under which the duration is recorded in
                ``self._startup_parallel_tasks``.
            func: The initializer to run.
            *args: Positional arguments forwarded to ``func``.

        Returns:
            Whatever ``func`` returns.

        The timing is taken on the worker thread, around the call itself, so
        it survives however long the future then sits completed before
        ``as_completed`` yields it (TASK-21111). Recorded in a ``finally`` so
        a failing task is timed too. ``dict`` item assignment is atomic under
        the GIL and each task writes a distinct key, so no lock is needed.
        """
        task_start = time.perf_counter()
        try:
            return func(*args)
        finally:
            self._startup_parallel_tasks[task_name] = time.perf_counter() - task_start

    def _construct_notes_sync_runtime_owner(self) -> "NotesSyncRuntimeOwner":
        """Build the application-owned lasting-sync runtime (TASK-21108).

        Named ``_construct_`` rather than the house ``_build_`` prefix on
        purpose: ``Tests/Notes/test_notes_sync_cutover.py`` fences the cutover
        keywords by matching call names that END WITH
        ``build_notes_sync_runtime_owner``, and a ``_build_...`` wrapper would
        register as a second such call and defeat the fence.

        Moved out of ``__init__`` so `Notes/notes_sync_runtime` and
        `Notes/notes_sync_legacy` leave the app import closure; the body is
        the one this app has always run, including the TASK-21112 start gate.
        Construction performs no I/O: ``NotesDeviceStateStore`` only records
        the path, and the gate's ``Path.exists()`` neither opens nor creates
        the database.

        Returns:
            NotesSyncRuntimeOwner: The unstarted runtime owner.
        """
        from .Notes.notes_sync_legacy import (  # noqa: PLC0415
            legacy_sync_directory_configured,
        )
        from .Notes.notes_sync_runtime import (  # noqa: PLC0415
            build_notes_sync_legacy_migrator,
            build_notes_sync_runtime_owner,
        )

        notes_sync_state_path = get_notes_sync_state_db_path()
        notes_sync_migrator = build_notes_sync_legacy_migrator(
            database_path=notes_sync_state_path,
            legacy_connection=lambda: self.chachanotes_db.get_connection(),
            settings=self.app_config,
            note_scope_id=ScopeType.LOCAL_NOTE.value,
            file_notes_binding=self._notes_sync_file_notes_binding,
            private_paths=(notes_sync_state_path, get_chachanotes_db_path()),
        )
        notes_sync_watcher_interval, notes_sync_watcher_max_interval = (
            get_notes_sync_watcher_intervals(self.app_config)
        )
        return build_notes_sync_runtime_owner(
            notes_scope_service=self._notes_sync_scope_service,
            cutover_admitted=True,
            profile_process_is_sole=self._instance_lock_status.acquired,
            database_path=notes_sync_state_path,
            migrate_legacy=notes_sync_migrator,
            file_notes_binding=self._notes_sync_file_notes_binding,
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
                    legacy_sync_directory_configured(settings) or state_path.exists()
                )
            ),
            watcher_interval_seconds=notes_sync_watcher_interval,
            watcher_max_interval_seconds=notes_sync_watcher_max_interval,
        )

    @property
    def notes_sync_runtime_owner(self) -> "NotesSyncRuntimeOwner":
        """The lasting-sync runtime owner, built lazily and cached.

        Built under a lock so a racing first access cannot produce two
        runtimes over the same state database. ``on_mount`` is the first
        reader in production.

        Returns:
            NotesSyncRuntimeOwner: The cached runtime owner.
        """
        owner = self._notes_sync_runtime_owner
        if owner is None:
            with self._notes_sync_runtime_owner_lock:
                owner = self._notes_sync_runtime_owner
                if owner is None:
                    owner = self._construct_notes_sync_runtime_owner()
                    self._notes_sync_runtime_owner = owner
        return owner

    @notes_sync_runtime_owner.setter
    def notes_sync_runtime_owner(self, owner: "NotesSyncRuntimeOwner") -> None:
        """Substitute the runtime owner (tests install doubles this way).

        Takes the same lock as the getter so the slot is coherent in both
        directions: an assignment racing a first read cannot interleave with
        the build. Non-reentrant is safe here because the build never assigns
        through this property.
        """
        with self._notes_sync_runtime_owner_lock:
            self._notes_sync_runtime_owner = owner

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
            # Independent Buddy artwork needs only the profile DB. The local
            # Persona service remains optional compatibility for legacy choices.
            local_persona_service = getattr(self, "local_character_persona_service", None)
            profile_db = getattr(self, "chachanotes_db", None)
            if local_persona_service is None and profile_db is None:
                return None
            from .Persona_Buddy.controller import (  # noqa: PLC0415 - imports Persona_Visual + PIL; first feature use only (TASK-21103)
                PersonaBuddyController,
                load_local_persona_portrait,
            )
            from .Persona_Buddy.interaction import parse_preferences
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
                reduced_motion=lambda: bool(
                    self.app_config.get("appearance", {}).get("reduce_motion", False)
                    or not parse_preferences(
                        self.app_config.get("buddy_interaction", {})
                    ).animated
                ),
                scheduler=self.call_after_refresh,
                on_change=self._notify_persona_buddy_changed,
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
        self.server_context_provider = RuntimeServerContextProvider(
            runtime_context=self.runtime_policy,
            target_store=self.unified_mcp_target_store,
            credential_store_factory=lambda: self.server_credential_store,
            app_config=self.app_config,
            credential_profile_id=default_server_credential_profile_id(),
        )

    def _build_local_skill_trust_service(self) -> Any:
        """Build the skill trust service. Performs OS keyring discovery.

        Split out of the eager wiring (TASK-21111(b)): it is the only part
        of the local skills stack that touches the keyring -- twice, once
        for the rollback marker store's secure-backend probe and once for
        the trust key cache -- and nothing at startup asks a trust question.
        Deferring the whole SERVICE was not enough on its own: the Console's
        agent bridge takes the skills scope facade during Chat screen mount,
        which merely relocated the discovery from ``__init__`` to mount.
        ``LocalSkillsService`` therefore takes this as a FACTORY and calls it
        on the first trust decision.
        """
        local_skills_store_dir = default_local_skills_store_dir(get_user_data_dir())
        trust_store_dir = default_trust_store_dir(local_skills_store_dir)
        from .Skills_Interop.recovery_activation import is_recovered

        recovered_skills = is_recovered(
            local_skills_store_dir / "skills", trust_store_dir
        )
        if recovered_skills:
            from .Skills_Interop.skill_trust_store import (
                FileSkillTrustGenerationMarkerStore,
            )

            skill_trust_marker_store = FileSkillTrustGenerationMarkerStore(
                trust_store_dir / _SKILL_TRUST_MARKER_FILENAME, store_dir=trust_store_dir
            )
            reduced_rollback_protection = True
            skill_key_cache = None
        else:
            trust_account_scope = skill_trust_account_scope(trust_store_dir)
            skill_trust_marker_store, reduced_rollback_protection = (
                build_skill_trust_marker_store_with_fallback(
                    fallback_marker_path=trust_store_dir / _SKILL_TRUST_MARKER_FILENAME,
                    store_dir=trust_store_dir,
                    account_scope=trust_account_scope,
                )
            )
            skill_key_cache = build_default_skill_trust_key_cache(
                account_scope=trust_account_scope
            )
        return SkillTrustService(
            skills_dir=local_skills_store_dir / "skills",
            trust_store=SkillTrustStore(
                store_dir=trust_store_dir,
                marker_store=skill_trust_marker_store,
            ),
            key_cache=skill_key_cache,
            keyring_convenience_enabled=False,
            reduced_rollback_protection=reduced_rollback_protection,
        )

    def _build_plugin_service(self) -> Any:
        """Build one IO-free facade; its worker starts on explicit plugin use."""
        service = getattr(self, "_plugin_service", None)
        if service is None:
            from tldw_chatbook.Plugins.service import PluginService

            service = PluginService(
                get_user_data_dir(),
                workspace_lookup=self.workspace_registry_service.get_workspace,
                mcp_mapping_owner=self.local_mcp_control_service,
            )
            self._plugin_service = service
        return service

    def _build_local_skills_stack(self) -> None:
        """Build the local skills service + scope facade. Idempotent.

        Body moved out of ``_wire_watchlists_and_notifications_services``
        (TASK-21111(b)). Keyring-free: the trust service is handed over as a
        factory. The collaborators it reads were captured at construction
        time, not re-read now, so deferring changes WHEN it runs and not
        WHAT it binds (the TASK-21108 trap).

        Each slot is filled only if still unset, so an injected double (a
        test assigning one of them between construction and first read) is
        never clobbered by a later sibling access.
        """
        if None not in (self._local_skills_service, self._skills_scope_service):
            return
        policy_enforcer, server_skills_service = self._local_skills_stack_inputs
        if self._local_skills_service is None:
            self._local_skills_service = LocalSkillsService(
                store_dir=default_local_skills_store_dir(get_user_data_dir()),
                policy_enforcer=policy_enforcer,
                trust_service_factory=lambda: self.local_skill_trust_service,
                plugin_service_factory=self._build_plugin_service,
                # In-memory config read: this runs on every skills read,
                # including the Console's per-send capture.
                builtin_disabled_loader=partial(
                    _read_app_disabled_builtin_skills, self
                ),
            )
        if self._skills_scope_service is None:
            self._skills_scope_service = SkillsScopeService(
                local_service=self._local_skills_service,
                server_service=server_skills_service,
                policy_enforcer=policy_enforcer,
            )

    @property
    def local_skill_trust_service(self) -> Any:
        """Local skill trust service, built on first access (TASK-21111(b))."""
        if self._local_skill_trust_service is None:
            self._local_skill_trust_service = self._build_local_skill_trust_service()
        return self._local_skill_trust_service

    async def ensure_local_skill_trust_service(
        self,
        *,
        _source_current: Callable[[], bool] | None = None,
        _owner_current: Callable[[Any], bool] | None = None,
        _read_observers: tuple[set, ...] = (),
    ) -> Any:
        """First-use trust service build, OFF the UI event loop (task-33081).

        The build performs OS keyring backend discovery (SecretService/D-Bus
        on Linux -- the deferred-wiring notes' timing figure never measured
        that platform). The sync property keeps its contract for callers
        already on worker threads; UI-loop callers must come here so the
        discovery never blocks the loop.

        Returns:
            The shared local skill trust service, built once; concurrent
            first callers await the same build under the build lock.
        """
        if _owner_current is not None and _owner_current() is not True:
            raise RuntimeError("console_skill_trust_owner_changed")
        if self._local_skill_trust_service is not None:
            return self._local_skill_trust_service
        async with self._local_skill_trust_service_build_lock:
            if (
                _owner_current is not None
                and _owner_current(self._local_skill_trust_service) is not True
            ):
                raise RuntimeError("console_skill_trust_owner_changed")
            if self._local_skill_trust_service is None:
                from .Chat.console_preparation_reads import run_preparation_read

                # Hold singleflight ownership until the original executor
                # callback physically returns, even through repeated cancel.
                def require_current():
                    if _source_current is not None and _source_current() is not True:
                        raise RuntimeError("console_skill_trust_source_changed")

                service = await run_preparation_read(
                    self._build_local_skill_trust_service,
                    creator=self,
                    session_id=None,
                    reads=set(),
                    observers=_read_observers,
                    require_current=require_current,
                )
                if (
                    _owner_current is not None
                    and _owner_current(self._local_skill_trust_service) is not True
                ):
                    raise RuntimeError("console_skill_trust_owner_changed")
                # Preserve a ready/injected winner installed during the build.
                if self._local_skill_trust_service is None:
                    self._local_skill_trust_service = service
            return self._local_skill_trust_service

    @local_skill_trust_service.setter
    def local_skill_trust_service(self, service: Any) -> None:
        self._local_skill_trust_service = service

    @property
    def local_skills_service(self) -> Any:
        """Local skills service, built on first access (TASK-21111(b))."""
        self._build_local_skills_stack()
        return self._local_skills_service

    @local_skills_service.setter
    def local_skills_service(self, service: Any) -> None:
        self._local_skills_service = service

    @property
    def skills_scope_service(self) -> Any:
        """Skills scope facade, built on first access (TASK-21111(b))."""
        self._build_local_skills_stack()
        return self._skills_scope_service

    @skills_scope_service.setter
    def skills_scope_service(self, service: Any) -> None:
        self._skills_scope_service = service

    def _resolve_server_credential_store(self) -> None:
        """Build the OS-backed credential store, or the unavailable stand-in.

        The body ``_wire_server_context_provider`` used to run inline. It is
        deferred because ``build_default_server_credential_store()`` calls
        ``keyring.get_keyring()``, whose first invocation performs backend
        discovery (11.3 ms on macOS, including the Security.framework ctypes
        load) -- work no boot needs unless the user actually uses server
        mode. TASK-21111(b).

        Sets both ``_server_credential_store`` and
        ``_server_credential_store_unavailable_reason``; the fallback choice
        and its warning are unchanged, only their timing.
        """
        try:
            self._server_credential_store = build_default_server_credential_store()
            self._server_credential_store_unavailable_reason = None
        except CredentialStoreUnavailable as exc:
            self._server_credential_store = UnavailableServerCredentialStore(str(exc))
            self._server_credential_store_unavailable_reason = str(exc)
            logger.warning(
                "No secure OS credential store available; server tokens will "
                "remain config-only (reason={}).",
                str(exc),
            )

    @property
    def server_credential_store(self) -> Any:
        """The app's credential store, resolved on first use (TASK-21111(b))."""
        if self._server_credential_store is None:
            self._resolve_server_credential_store()
        return self._server_credential_store

    @server_credential_store.setter
    def server_credential_store(self, store: Any) -> None:
        """Inject a credential store (tests, explicit reconfiguration).

        Keeps the reason consistent with the store, so the pair can never
        disagree the way two independently-assigned attributes could.
        """
        self._server_credential_store = store
        self._server_credential_store_unavailable_reason = (
            store.message
            if isinstance(store, UnavailableServerCredentialStore)
            else None
        )

    @property
    def server_credential_store_unavailable_reason(self) -> str | None:
        """Why no OS credential store is in use, or None. Resolves on read."""
        if self._server_credential_store is None:
            self._resolve_server_credential_store()
        return self._server_credential_store_unavailable_reason

    @server_credential_store_unavailable_reason.setter
    def server_credential_store_unavailable_reason(self, reason: str | None) -> None:
        self._server_credential_store_unavailable_reason = reason

    def _wire_character_persona_services(self) -> None:
        from .Backup_Recovery.chat_source_participants import (
            build_persona_service, build_dictionary_service,
        )
        from .DB.VisualIdentity_DB import VisualIdentityRepository
        from .Persona_Visual.repository import PersonaVisualRepository

        self.server_character_persona_service = (
            ServerCharacterPersonaService.from_server_context_provider(
                self.server_context_provider,
                policy_enforcer=self.service_policy_enforcer,
            )
        )
        self.local_character_persona_service = build_persona_service(self.chachanotes_db)
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
        self.actor_pack_import_service = None
        self.actor_pack_activation_service = None
        self.actor_pack_import_controller = None
        if self.chachanotes_db is not None:
            # task-22216: this construction is pure — the staging crash
            # sweep no longer runs inside ActorPackImportService.__init__
            # (a secure_private_directory walk + scandir on every boot,
            # the task-21106 class). `ensure_actor_pack_staging_sweep`
            # runs it once per app session from the deferred startup
            # worker; the service itself gates `inspect_archive` on the
            # same once-lock, so an import racing the worker still sweeps
            # first. Guarded by
            # Tests/App/test_boot_construct_fs_side_effects.py.
            actor_pack_profile_root = get_user_data_dir()
            self.actor_pack_import_service = ActorPackImportService(
                self.actor_pack_repository,
                staging_root=actor_pack_profile_root / "actor_pack_imports",
                profile_root=actor_pack_profile_root,
                local_service=self.local_character_persona_service,
            )
            self.actor_pack_activation_service = ActorPackActivationService(
                self.chachanotes_db,
                self.local_character_persona_service,
                self.actor_pack_repository,
                self.persona_actor_pack_coordinator,
                self.actor_pack_import_service,
            )
            self.actor_pack_import_controller = ActorPackImportController(
                self.actor_pack_import_service,
                self.actor_pack_activation_service,
                refresh_callbacks=(self._refresh_after_actor_pack_import,),
            )
        self._actor_pack_import_shutdown_task: asyncio.Task[None] | None = None
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
        self.local_chat_dictionary_service = build_dictionary_service(self.chachanotes_db)
        self.chat_dictionary_scope_service = ChatDictionaryScopeService(
            local_service=self.local_chat_dictionary_service,
            server_service=self.server_chat_dictionary_service,
            policy_enforcer=self.service_policy_enforcer,
        )

    async def _run_actor_pack_recovery_owned(self, callback: Callable[[], None]) -> None:
        """Keep the original startup callback alive until its thread retires."""
        from .Chat.console_preparation_reads import run_preparation_read

        def require_open() -> None:
            if self._actor_pack_recovery_closed:
                raise asyncio.CancelledError

        await run_preparation_read(
            callback,
            creator=self,
            session_id=None,
            reads=self._actor_pack_recovery_reads,
            require_current=require_open,
        )

    async def _shutdown_actor_pack_recovery(self) -> asyncio.CancelledError | None:
        """Close startup admission and drain its exact physical callbacks."""
        self._actor_pack_recovery_closed = True
        reads = getattr(self, "_actor_pack_recovery_reads", None)
        if reads:
            from .Chat.console_preparation_reads import drain_preparation_reads

            if await drain_preparation_reads(reads):
                return asyncio.CancelledError()
        return None

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
        from .DB.base_db import operation_owned_connection

        with operation_owned_connection(self.chachanotes_db):
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

            if self.actor_pack_recovery_error is None:
                service = getattr(self, "local_character_persona_service", None)
                try:
                    from .Persona_Buddy.library import BuddyLibrary
                    from .Persona_Buddy.preferences import (
                        parse_persona_buddy_preferences,
                        persist_persona_buddy_preferences,
                        serialize_persona_buddy_preferences,
                    )

                    library = BuddyLibrary(
                        self.chachanotes_db,
                        get_user_data_dir(),
                        persona_reader=getattr(service, "get_persona_profile", None),
                    )
                    legacy_retired = False
                    if service is not None:
                        try:
                            legacy_builtin = service._find_persona_profile(
                                "local-persona-builtin-pixel-migu", include_deleted=True
                            )
                        except ValueError:
                            legacy_builtin = None
                        legacy_retired = bool(
                            legacy_builtin
                            and (
                                legacy_builtin.get("deleted")
                                or legacy_builtin.get("is_active", True) is not True
                                or library.repository.get_active_persona_pack(
                                    "local-persona-builtin-pixel-migu"
                                )
                                is None
                            )
                        )
                    library.ensure_builtin(legacy_retired=legacy_retired)
                    controller = getattr(self, "_persona_buddy_controller", None)
                    if controller is None:
                        config = getattr(self, "app_config", {})
                        previous = parse_persona_buddy_preferences(
                            config.get("persona_buddy", {})
                        )

                        def persist_unclaimed_migration(candidate):
                            # First construction reads app_config under this same lock.
                            # Keep disk admission and publication together so it sees
                            # the committed owner, or takes over migration itself.
                            with self._persona_buddy_controller_lock:
                                if (
                                    self._persona_buddy_controller is not None
                                    or parse_persona_buddy_preferences(
                                        config.get("persona_buddy", {})
                                    )
                                    != previous
                                    or not persist_persona_buddy_preferences(candidate)
                                ):
                                    return False
                                config["persona_buddy"] = serialize_persona_buddy_preferences(
                                    candidate
                                )
                                return True

                        library.migrate_legacy_selection(
                            previous,
                            writer=persist_unclaimed_migration,
                        )
                        controller = getattr(self, "_persona_buddy_controller", None)
                    if controller is not None:

                        def schedule_migration():
                            self.run_worker(
                                controller.migrate_legacy_selection(library),
                                group="buddy-legacy-migration",
                                exclusive=False,
                            )

                        if threading.get_ident() == getattr(self, "_thread_id", None):
                            schedule_migration()
                        else:
                            self.call_from_thread(schedule_migration)
                except Exception:
                    self.loguru_logger.warning(
                        "Independent Buddy installation/migration deferred; existing choices retained"
                    )

    def ensure_actor_pack_staging_sweep(self) -> None:
        """Run the Actor Pack staging crash-sweep once per session (task-22216).

        Safe to call from any thread: the once-gate (and the lock that
        serializes it against a first ``inspect_archive``) lives on the
        import service. Called from the deferred startup worker; runs on a
        thread because the sweep does real filesystem I/O.

        A sweep failure is absorbed and logged rather than raised — the
        pre-move behavior (the sweep ran inside ``TldwCli.__init__`` via
        the service constructor, so a failure aborted app construction
        outright) is deliberately softened to match the task-21106
        recovery seam: the app stays up, the service's gate stays open,
        and the next import attempt retries the sweep and surfaces the
        same categorized error to the user.
        """
        service = getattr(self, "actor_pack_import_service", None)
        if service is None:
            return
        try:
            service.ensure_staging_swept()
        except ActorPackImportError as exc:
            # Category tokens only — the importer's errors are path-free by
            # contract, and this sink is persistent (TASK-15103 rules).
            self.loguru_logger.warning(
                "Actor Pack staging sweep failed (will retry on first "
                f"import use): {exc.category}"
            )
        except Exception as exc:
            self.loguru_logger.warning(
                "Actor Pack staging sweep failed (will retry on first "
                f"import use): {type(exc).__name__}"
            )

    def _deferred_wire_workspace_agent_provisioning(self) -> None:
        """Timer callback: run the (best-effort, non-fatal) provisioning wiring."""
        from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID

        try:
            registry = getattr(self, "workspace_registry_service", None)
            if (
                registry is not None
                and not registry.db.is_agent_backfill_complete()
                and any(
                    record.workspace_id != DEFAULT_WORKSPACE_ID
                    and not record.archived
                    and record.assistant_defaults is None
                    and not record.assistant_defaults_explicit_none
                    for record in registry.list_workspaces()
                )
            ):
                # Existing workspaces are a real first use; an empty startup
                # must not eagerly initialize the portable-profile subsystem.
                self.run_worker(
                    self.ensure_workspace_agent_provisioning(),
                    group="workspace-agent-provisioning",
                    exclusive=True,
                    exit_on_error=False,
                )
                return
            self._wire_workspace_agent_provisioning()
        except Exception as exc:
            self.loguru_logger.warning(
                "Deferred workspace agent provisioning wiring failed; error_type={}",
                type(exc).__name__,
            )

    async def ensure_workspace_agent_provisioning(self) -> None:
        """Await app-owned profile authority before automatic workspace setup."""
        worker = self._deferred_wire_tool_pack_service()
        if worker is not None:
            try:
                # Modal cancellation must not cancel shared app composition.
                await asyncio.shield(worker.wait())
            except Exception as exc:  # noqa: BLE001 - convenience setup remains nonfatal
                self.loguru_logger.warning(
                    "Workspace profile initialization failed; "
                    f"error_type={type(exc).__name__}"
                )
        try:
            self._wire_workspace_agent_provisioning()
        except Exception as exc:  # noqa: BLE001 - convenience setup remains nonfatal
            self.loguru_logger.warning(
                "Workspace agent provisioning wiring failed; "
                f"error_type={type(exc).__name__}"
            )

    def _get_tool_profile_operations(self):
        """Lazily own admitted Tool Profile writes for this app session."""
        from .Tool_Packs.operations import (
            ToolProfileOperations,
            ToolProfileWriteUnavailable,
        )

        if getattr(self, "_tool_profile_operations_closed", False):
            raise ToolProfileWriteUnavailable("shutdown")
        owner = getattr(self, "_tool_profile_operations", None)
        if owner is None:
            owner = ToolProfileOperations()
            self._tool_profile_operations = owner
        return owner

    def _deferred_wire_tool_pack_service(self) -> Worker | None:
        """Schedule one complete Tool Pack composition on first feature use."""
        if not getattr(self, "_ui_ready", False) or getattr(
            self, "tool_pack_service", None
        ) is not None:
            return
        if getattr(self, "_tool_pack_wiring_started", False):
            return getattr(self, "_tool_pack_composition_worker", None)
        self._tool_pack_wiring_started = True
        self.tool_pack_service_unavailable_reason = "starting"
        try:
            worker = self.run_worker(
                self._compose_tool_pack_service_off_thread,
                name="deferred_tool_pack_service_composition",
                group="tool-pack-service-composition",
                thread=True,
                exclusive=True,
                exit_on_error=False,
            )
            self._tool_pack_composition_worker = worker
            return worker
        except Exception:
            self._tool_pack_wiring_started = False
            self.tool_pack_service_unavailable_reason = "composition_unavailable"
            return None

    def _compose_tool_pack_service_off_thread(self) -> None:
        """Compose, activate, then reconcile away from the Textual event loop."""
        try:
            unified = getattr(self, "unified_mcp_service", None)
            local_control = getattr(self, "local_mcp_control_service", None)
            registry = getattr(self, "workspace_registry_service", None)
            bootstrap = getattr(self, "_tool_pack_guard_bootstrap", None)
            permission_store = getattr(unified, "permission_store", None)
            if (
                permission_store is None
                or local_control is None
                or registry is None
                or type(bootstrap) is not DeferredWorkspaceToolProfileGuard
                or getattr(registry, "tool_profile_guard", None) is not bootstrap
            ):
                self.call_from_thread(
                    self._mark_tool_pack_service_unavailable,
                    "prerequisites_unavailable",
                )
                return

            # Deferred imports are intentional: service composition pulls in
            # archive, receipt, import, activation, and removal owners.
            from tldw_chatbook.MCP.local_server_tools import (
                resolve_server_workspace_root,
            )
            from tldw_chatbook.Tool_Packs.catalog_snapshot import (
                PermissionInventoryRegistry,
            )
            from tldw_chatbook.Tool_Packs.service import ToolPackService

            inventory = PermissionInventoryRegistry.v1(
                local_control,
                fallback_root=resolve_server_workspace_root(),
            )
            service = ToolPackService.compose(
                permission_store=permission_store,
                inventory=inventory,
                workspace_registry=registry,
                receipt_root=get_user_data_dir() / "tool_pack_receipts",
            )
            attached = self.call_from_thread(
                self._attach_tool_pack_service,
                service,
                registry,
                bootstrap,
            )
            if attached is not True:
                return
            recovery = service.reconcile_receipts()
            self.call_from_thread(
                self._record_tool_pack_receipt_reconciliation,
                service,
                recovery.unavailable_category,
            )
        except Exception:
            self.call_from_thread(
                self._mark_tool_pack_service_unavailable,
                "composition_unavailable",
            )

    def _attach_tool_pack_service(
        self,
        service: object,
        registry: object,
        bootstrap: object,
    ) -> bool:
        """Atomically activate one complete guard for the captured registry."""
        if getattr(self, "tool_pack_service", None) is not None:
            return False
        if (
            getattr(self, "workspace_registry_service", None) is not registry
            or getattr(self, "_tool_pack_guard_bootstrap", None) is not bootstrap
            or type(bootstrap) is not DeferredWorkspaceToolProfileGuard
            or getattr(registry, "tool_profile_guard", None) is not bootstrap
        ):
            self._mark_tool_pack_service_unavailable("prerequisites_unavailable")
            return False
        try:
            guard = service.binding_guard  # type: ignore[attr-defined]
            if bootstrap.activate(guard) is not True:
                raise RuntimeError("Tool Pack guard was already active")
            if bootstrap.active_guard is not guard:
                raise RuntimeError("Tool Pack guard activation was not atomic")
        except Exception:
            self._mark_tool_pack_service_unavailable("composition_unavailable")
            return False
        self.tool_pack_service = service
        self.tool_pack_service_unavailable_reason = None
        self.tool_pack_receipt_reconciliation_unavailable_reason = "pending"
        return True

    def _record_tool_pack_receipt_reconciliation(
        self, service: object, unavailable_category: str | None
    ) -> None:
        """Record recovery only for the service that still owns authority."""
        if getattr(self, "tool_pack_service", None) is service:
            self.tool_pack_receipt_reconciliation_unavailable_reason = (
                unavailable_category
            )

    def _mark_tool_pack_service_unavailable(self, category: str) -> None:
        """Expose one stable unavailable category without diagnostic detail."""
        if getattr(self, "tool_pack_service", None) is None:
            self._tool_pack_wiring_started = False
            self.tool_pack_service_unavailable_reason = category

    def _deferred_wire_notes_sync_services(self) -> None:
        """Compose Notes organization Sync after the first interactive frame."""

        try:
            _wire_notes_sync_services(self)
        except Exception as exc:
            self.loguru_logger.warning(
                "Deferred Notes organization Sync wiring failed; error_type={}",
                type(exc).__name__,
            )

    def _wire_workspace_agent_provisioning(self) -> None:
        """Attach the workspace agent provisioner and run the startup backfill.

        Task-8 (workspace assistant defaults): every explicit workspace gets
        a reference-backed default agent (persona + ``ws-<id>`` permission
        profile) without user wiring. The registry is constructed before
        persona services exist, so the hook is attached post-construction via
        ``set_agent_provisioner``; the backfill then covers workspaces
        created before this wiring ran. Strictly best-effort: skipped when
        the registry, local persona service, or the unified MCP service's
        permission store is unavailable, and never raises.
        """
        registry = getattr(self, "workspace_registry_service", None)
        persona_service = getattr(self, "local_character_persona_service", None)
        unified_service = getattr(self, "unified_mcp_service", None)
        if registry is not None:
            guard = registry.tool_profile_guard
            if (
                isinstance(guard, DeferredWorkspaceToolProfileGuard)
                and guard.active_guard is None
            ):
                # Do not create Persona/profile records that cannot yet be
                # bound. The create dialog or eligible startup backfill awaits
                # the existing Tool Pack composition before retrying wiring.
                return
        permission_store = getattr(unified_service, "permission_store", None)
        # Lazy import (boot budget, ADR-097): this wiring runs on a
        # post-ready timer, and importing at module scope would make
        # `Workspaces.agent_provisioning` resident at `_ui_ready`.
        from tldw_chatbook.Workspaces.agent_provisioning import (
            WorkspaceAgentProvisioner,
            run_workspace_agent_backfill,
        )
        if registry is None or persona_service is None or permission_store is None:
            logger.debug(
                "Workspace agent provisioning skipped: registry, persona "
                "service, or permission store unavailable"
            )
            return
        provisioner = WorkspaceAgentProvisioner(persona_service, permission_store)
        registry.set_agent_provisioner(provisioner.provision)
        self.workspace_agent_provisioner = provisioner
        try:
            provisioned = run_workspace_agent_backfill(
                registry=registry,
                provisioner=provisioner,
            )
        except Exception as exc:
            logger.warning(
                "Workspace agent backfill failed during app wiring; error_type={}",
                type(exc).__name__,
            )
            return
        if provisioned:
            logger.info(
                "Workspace agent backfill provisioned {} workspace(s)",
                provisioned,
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
        if local_service is not None and migration is not None:
            from .Backup_Recovery.chat_source_participants import bind_citation_services

            bind_citation_services(local_service, migration)
        self.local_chat_conversation_service = local_service
        self.citation_trace_repository = repository
        self.citation_legacy_migration_service = migration
        self.conversation_local_marks_service = (
            ConversationLocalMarksService(trace_db) if trace_db is not None else None
        )
        runtime = getattr(self, "console_runtime", None)
        recompute_attention = getattr(
            runtime, "recompute_console_attention", None
        )
        if callable(recompute_attention):
            recompute_attention(force_projection=True)
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
        if self.local_chat_conversation_service is not None:
            self.local_chat_conversation_service.organization_sync_service = getattr(
                self, "notes_organization_sync_service", None
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

        # dreams phase 1: the Dreams DB itself is NOT constructed here --
        # every Dreams import must stay off the `_ui_ready` module census
        # (ADR-097 boot ratchet; the budget is at its limit), and a disabled
        # Dreams must not create storage, so `get_dreams_db` builds it on
        # first use while `[dreams] enabled` is set.
        self.dreams_db = None
        self._dreams_db_lock = threading.Lock()

    def _wire_collections_capture_services(self) -> None:
        """Compose the profile-owned Local capture authority and scope seam."""
        TldwCli._reset_collections_capture_services(self)
        self.collections_capture_scope_service = _collections_capture_scope(self)
        try:
            database_path = get_library_collections_db_path()
            database = getattr(self, "local_library_collections_db", None)
            if not isinstance(database, LibraryCollectionsDB):
                database = LibraryCollectionsDB(database_path, CLI_APP_CLIENT_ID)
                self.local_library_collections_db = database
            parts = _collections_capture_parts(database, database_path, get_user_data_dir())
            service = _local_collections_capture_service(parts)
        except Exception:
            logger.opt(exception=True).warning(
                "Local Collections capture service unavailable during app wiring"
            )
            return
        _publish_collections_capture_parts(self, parts, service)

    def _reset_collections_capture_services(self) -> None:
        """Install inert capture seams without importing their implementations."""
        self.collections_capture_repository = None
        self.collections_offline_store = None
        self.collections_legacy_recovery_service = None
        self.local_collections_capture_authority = None
        self.local_collections_capture_service = None
        self.server_collections_capture_authority = None
        self.server_collections_capture_service = None
        self.collections_capture_scope_service = _DeferredCollectionsCaptureScope(self)

    def ensure_collections_capture_services(self) -> Any | None:
        """Compose capture services once, after readiness or on first use."""
        scope = getattr(self, "collections_capture_scope_service", None)
        if scope is None or isinstance(scope, _DeferredCollectionsCaptureScope):
            TldwCli._wire_collections_capture_services(self)
            scope = getattr(self, "collections_capture_scope_service", None)
            initializer = getattr(self, "_collections_capture_initializer_task", None)
            if (
                initializer is not None
                and not initializer.done()
                and not any(
                    getattr(self, flag, False)
                    for flag in (
                        "_collections_capture_initializer_closed", "_shutting_down", "_exit"
                    )
                )
                and getattr(self, "collections_capture_repository", None) is not None
            ):
                # First use wins publication; the displaced initializer cannot
                # schedule reconciliation for this independently composed owner.
                self._create_deferred_startup_task(
                    self._reconcile_collections_capture_startup(),
                    name="deferred_collections_capture_reconciliation",
                )
        return scope

    def _deferred_wire_collections_capture_services(self) -> None:
        """Prepare stock file-backed capture services away from the UI loop."""
        if any(vars(self).get(flag, False) for flag in (
            "_collections_capture_initializer_closed", "_shutting_down", "_exit",
        )):
            return
        task = vars(self).get("_collections_capture_initializer_task")
        if task is not None and not task.done():
            return
        request = _capture_deferred_collections_setup(self)
        if request is not None:
            self._collections_capture_setup = request
            task = _COLLECTIONS_TASK(
                _initialize_deferred_collections_capture(request),
                name="deferred_collections_capture_setup",
            )
            request.initializer = self._collections_capture_initializer_task = task
            self._deferred_startup_tasks.add(task)
            task.add_done_callback(self._deferred_startup_tasks.discard)
            return
        self.ensure_collections_capture_services()
        if getattr(self, "collections_capture_repository", None) is not None:
            self._create_deferred_startup_task(
                self._reconcile_collections_capture_startup(),
                name="deferred_collections_capture_reconciliation",
            )

    def _activate_collections_capture_authority(self) -> None:
        """Activate the capture owner selected by the committed runtime source."""
        scope = getattr(self, "collections_capture_scope_service", None)
        if scope is None:
            return
        runtime_policy = getattr(self, "runtime_policy", None)
        runtime_state = getattr(runtime_policy, "state", None)
        source = str(getattr(runtime_state, "active_source", "local") or "local")
        if source.strip().lower() != "server":
            authority = getattr(self, "local_collections_capture_authority", None)
            service = getattr(self, "local_collections_capture_service", None)
            if authority is not None and service is not None:
                scope.activate(authority, service)
            else:
                deactivate = getattr(scope, "deactivate", None)
                if callable(deactivate):
                    deactivate()
            return

        from tldw_chatbook.Library.collections_capture_service import (
            build_server_capture_authority,
        )
        from tldw_chatbook.Library.server_collections_capture_service import (
            ServerCollectionsCaptureService,
        )

        provider = getattr(self, "server_context_provider", None)
        try:
            context = provider.get_active_context()
            profile_id = str(getattr(context, "active_server_id", "") or "").strip()
            principal_id = event_principal_id_from_active_context(context) or ""
            if not profile_id or not principal_id:
                raise ValueError("server_capture_identity_unavailable")
            authority = build_server_capture_authority(profile_id, principal_id)
            client = provider.build_client()
            token = str(getattr(context, "auth_token", "") or "")
            credential_fingerprint = hashlib.sha256(
                token.encode("utf-8")
            ).hexdigest()[:24]
            service = getattr(self, "server_collections_capture_service", None)
            prior_fingerprint = getattr(
                self, "_server_collections_credential_fingerprint", None
            )
            if not (
                isinstance(service, ServerCollectionsCaptureService)
                and service.authority == authority
                and service.client is client
                and prior_fingerprint == credential_fingerprint
            ):
                async def docs_info_provider() -> Any:
                    get_docs_info = getattr(client, "get_server_docs_info", None)
                    if not callable(get_docs_info):
                        raise RuntimeError("server_docs_info_unavailable")
                    return await get_docs_info()

                service = ServerCollectionsCaptureService(
                    authority,
                    client,
                    docs_info_provider=docs_info_provider,
                    credential_fingerprint=credential_fingerprint,
                )
                self.server_collections_capture_authority = authority
                self.server_collections_capture_service = service
                self._server_collections_credential_fingerprint = (
                    credential_fingerprint
                )
            scope.activate(authority, service)
        except Exception:
            logger.opt(exception=True).warning(
                "Server Collections capture authority unavailable during activation"
            )
            deactivate = getattr(scope, "deactivate", None)
            if callable(deactivate):
                deactivate()

    async def _reconcile_collections_capture_startup(self) -> None:
        """Repair interrupted capture state with retained physical callbacks."""
        from .Chat.console_preparation_reads import run_preparation_read
        from .DB.base_db import operation_owned_connection

        def require_open() -> None:
            if any(getattr(self, flag, False) for flag in (
                "_collections_capture_initializer_closed", "_shutting_down", "_exit",
            )):
                raise asyncio.CancelledError

        require_open()
        reads = getattr(self, "_collections_capture_reconciliation_reads", None)
        if reads is None:
            reads = self._collections_capture_reconciliation_reads = set()
        repository = getattr(self, "collections_capture_repository", None)
        if repository is not None:
            def interrupt_in_worker():
                with operation_owned_connection(getattr(repository, "db", None)):
                    return repository.interrupt_stale_extractions()

            await run_preparation_read(
                interrupt_in_worker, creator=self, session_id=None,
                reads=reads, require_current=require_open,
            )
        offline_store = getattr(self, "collections_offline_store", None)
        if offline_store is not None:
            def reconcile_in_worker():
                with operation_owned_connection(getattr(getattr(offline_store, "repository", None), "db", None)):
                    return offline_store.reconcile_batch(limit=25)

            await run_preparation_read(
                reconcile_in_worker, creator=self, session_id=None,
                reads=reads, require_current=require_open,
            )

    async def _shutdown_collections_capture_runtime(self) -> None:
        """Fence capture authority and retire its finite setup before disposal."""
        self._collections_capture_initializer_closed = True
        scope = getattr(self, "collections_capture_scope_service", None)
        if scope is not None and not isinstance(scope, _DeferredCollectionsCaptureScope):
            deactivate = getattr(scope, "deactivate", None)
            if callable(deactivate):
                deactivate()
        cancellation = await _retire_deferred_collections_capture(self)
        reads = getattr(self, "_collections_capture_reconciliation_reads", None)
        if reads:
            from .Chat.console_preparation_reads import drain_preparation_reads

            if await drain_preparation_reads(reads):
                cancellation = cancellation or asyncio.CancelledError()
        local_service = getattr(self, "local_collections_capture_service", None)
        if local_service is not None:
            await local_service.cancel_extractions()
        if cancellation is not None:
            raise cancellation

    def _wire_workspace_registry_services(self) -> None:
        self.change_review_consent_service = None
        self._tool_pack_guard_bootstrap = None
        try:
            self.local_workspace_db = WorkspaceDB(
                get_workspaces_db_path(),
                CLI_APP_CLIENT_ID,
            )
            self.workspace_registry_service = LocalWorkspaceRegistryService(
                self.local_workspace_db,
            )
            self._tool_pack_guard_bootstrap = DeferredWorkspaceToolProfileGuard()
            self.workspace_registry_service.attach_tool_profile_guard(
                self._tool_pack_guard_bootstrap
            )
            self.workspace_registry_service.ensure_default_workspace()
            self.change_review_consent_service = ChangeReviewConsentService(
                self.workspace_registry_service
            )
            self.workspace_registry_service.attach_change_review_consent_service(
                self.change_review_consent_service
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Local workspace registry service unavailable during app wiring",
            )
            self.local_workspace_db = None
            self.workspace_registry_service = None
            self._tool_pack_guard_bootstrap = None

    def _wire_research_source_association(self) -> None:
        """Compose durable post-ingest association services."""

        try:
            self.research_paste_staging_store = ResearchPasteStagingStore(
                get_user_data_dir() / "research_paste_staging"
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Private Research paste staging unavailable"
            )
            self.research_paste_staging_store = None
        try:
            from .Research_Workspace import (
                LocalResearchWorkspaceAdapter,
                ServerResearchWorkspaceAdapter,
                WorkspaceDataSource,
            )

            if self.local_workspace_db is None:
                raise RuntimeError("Workspace database is unavailable.")
            operation_store = ResearchSourceOperationStore(self.local_workspace_db)
            coordinator = ResearchSourceAssociationCoordinator(
                operation_store=operation_store,
                ingest_jobs=self.library_ingest_jobs,
                local_registry=self.workspace_registry_service,
                server_service=self.server_notes_workspace_service,
                server_context_provider=self.server_context_provider,
                catalog_requeuer=self._requeue_research_source_catalog_job,
                catalog_dispatcher=self._dispatch_research_source_catalog_job,
            )
            readiness_adapters = {}
            if self.workspace_registry_service is not None:
                readiness_adapters[WorkspaceDataSource.LOCAL] = (
                    LocalResearchWorkspaceAdapter(
                        self.workspace_registry_service,
                        media_scope_service=getattr(
                            self, "media_reading_scope_service", None
                        ),
                    )
                )
            if (
                self.server_notes_workspace_service is not None
                and self.server_context_provider is not None
            ):
                readiness_adapters[WorkspaceDataSource.SERVER] = (
                    ServerResearchWorkspaceAdapter(
                        self.server_notes_workspace_service,
                        self.server_context_provider,
                        media_scope_service=getattr(
                            self, "media_reading_scope_service", None
                        ),
                    )
                )
            readiness_coordinator = ResearchSourceReadinessCoordinator(
                operation_store=operation_store,
                adapters=readiness_adapters,
            )
            scheduler = ResearchSourceAssociationScheduler(
                coordinator=coordinator,
                operation_store=operation_store,
                readiness_coordinator=readiness_coordinator,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Research source association unavailable during app wiring"
            )
            self.research_source_operation_store = None
            self.research_source_association_coordinator = None
            self.research_source_readiness_coordinator = None
            self.research_source_association_scheduler = None
            return
        self.research_source_operation_store = operation_store
        self.research_source_association_coordinator = coordinator
        self.research_source_readiness_coordinator = readiness_coordinator
        self.research_source_association_scheduler = scheduler

    def _build_chatbook_db_paths(self) -> dict[str, str]:
        import sys
        from types import FunctionType

        from tldw_chatbook import config
        from tldw_chatbook.Backup_Recovery import config_participants as life

        callbacks = (get_chachanotes_db_path, get_media_db_path, get_prompts_db_path)
        names = ("get_chachanotes_db_path", "get_media_db_path", "get_prompts_db_path")
        selected = None
        retained = life.__dict__.get("_STARTUP_PATH_READER_ORIGINAL")
        tuple_type = tuple
        if type(retained) is tuple_type and len(retained) == 3:
            reader, code, namespace = retained
            if (
                type(reader) is FunctionType
                and life.__dict__.get("_startup_path_config_bundle") is reader
                and reader.__code__ is code
                and reader.__globals__ is namespace
                and namespace is life.__dict__
                and reader.__defaults__ is None
                and reader.__kwdefaults__ is None
                and reader.__closure__ is None
            ):
                selected = reader(
                    config,
                    tuple(
                        (sys.modules[__name__], name, callback)
                        for name, callback in zip(names, callbacks)
                    ),
                )
        if selected is None:
            return {
                "ChaChaNotes": str(get_chachanotes_db_path()),
                "Media": str(get_media_db_path()),
                "Prompts": str(get_prompts_db_path()),
            }
        bundle, check = selected
        operation, identity = bundle.operation, bundle.checked_identity
        capture, publish, key, error = (
            bundle.capture_publication,
            bundle.check_publication,
            bundle.key,
            bundle.error,
        )
        check()
        with operation(config) as active:
            check()
            owner = capture(config, active)
            paths = {}
            for label, callback in zip(("ChaChaNotes", "Media", "Prompts"), callbacks):
                check()
                paths[label] = str(callback())
                check()
                if identity(config, active) != (key[1], key[3]):
                    raise error("chatbook_path_source_changed")
        with owner[-1]:
            check(publication=True)
            publish(config, owner)
            return paths

    def _wire_prompt_chatbook_services(self) -> None:
        self.local_prompt_service = LocalPromptService(prompts_interop)
        self.server_prompt_service = ServerPromptService.from_server_context_provider(
            self.server_context_provider,
            policy_enforcer=self.service_policy_enforcer,
        )

        self.local_chatbook_service = LocalChatbookService(
            self._build_chatbook_db_paths()
        )
        # ArtifactShareController is created lazily on first share use via
        # _get_artifact_share_controller(): importing its module chain at boot
        # breaches the UI-ready module census ratchet (ADR-097 — the budget
        # never rises; new imports must be deferred).
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
        self._wire_research_source_association()

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

    @property
    def daily_report_demo_service(self) -> Any:
        """Build the opt-in report demo only when a demo entry point uses it."""

        service = getattr(self, "_daily_report_demo_service", None)
        if service is None:
            from .Subscriptions.daily_report_demo import DailyReportDemoService

            service = DailyReportDemoService(
                subscriptions_db=self.subscriptions_db,
                local_watchlists_getter=lambda: getattr(
                    self, "local_watchlists_service", None
                ),
                dispatch_service=self.notification_dispatch_service,
                app_getter=lambda: self,
                tts_service_getter=lambda: getattr(self, "tts_service", None),
                tts_profile_service_getter=lambda: getattr(
                    self, "_tts_profile_service", None
                ),
            )
            self._daily_report_demo_service = service
        return service

    @daily_report_demo_service.setter
    def daily_report_demo_service(self, service: Any) -> None:
        """Preserve the public injection seam used by screens and tests."""

        self._daily_report_demo_service = service

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
            on_queue_changed=lambda: (
                getattr(self, "scheduler_loop", None).request_reload()
                if getattr(self, "scheduler_loop", None) is not None
                else None
            ),
            # schedules-handoff PR-2, Task 6: `run_automation_now` (manual
            # dispatch) resolves the app for read-time health
            # (`compute_local_health`) and the automation handler through
            # these getters rather than importing the handler module
            # itself (ADR-097 boot-census rule) or holding either
            # reference directly -- both are late-binding lambdas closing
            # over `self`, same discipline as `on_queue_changed` above:
            # `self` is still mid-`__init__` here, and
            # `_get_automation_definition_handler` reads
            # `self.scheduling_service` (assigned by this very statement)
            # and `self.notification_dispatch_service`, neither resolved
            # until the getter is actually called.
            app_getter=lambda: self,
            automation_handler_getter=lambda: self._get_automation_definition_handler(),
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
                dispatch_service=self.notification_dispatch_service,
                notification_app_getter=lambda: self,
                # TASK-26027: group repeat brief failures into one incident
                # (the ScheduledTasks DB owns the durable state machine).
                incident_recorder=getattr(
                    self.scheduling_service, "db", None
                ),
            )

        # task-19561: shutdown has to be able to reach the generations this
        # handler spawns, and the scheduler loop is not a route to them --
        # they are bare `asyncio.Task`s, not workers, deliberately detached
        # from the tick. Keeping the handler itself on the app is the only
        # handle `on_unmount` has.
        self._briefing_job_handler = briefing_handler

        handlers: dict[str, Handler] = {
            "reminder": ReminderHandler(
                dispatch_service=self.notification_dispatch_service,
                app_getter=lambda: self,
            ),
        }
        if watchlist_handler is not None:
            handlers["watchlist_job"] = watchlist_handler
        if briefing_handler is not None:
            handlers["briefing_job"] = briefing_handler

        # dreams phase 1: the Dreams cycle handler is dispatched through a
        # lazy closure (the `_dispatch_automation_definition` pattern above)
        # so its import chain stays off the `_ui_ready` module census
        # (ADR-097; the boot budget is at its limit). The projection that
        # feeds it is attached to the live queue after `_ui_ready` in
        # `_wire_dreams_scheduler_integration`, before the scheduler's
        # first queue load, so a task can only ever dispatch after both
        # exist. Cycle deps are built per dispatch via
        # `_dreams_cycle_deps` (getter-lambda discipline -- most handles do
        # not exist yet at wiring time; the getter tolerates None and the
        # handler no-ops).
        async def _dispatch_dreams_cycle(task: dict[str, Any]) -> None:
            handler = self._get_dreams_cycle_handler()
            await handler.handle(task)

        handlers["dreams_cycle"] = _dispatch_dreams_cycle

        # schedules-handoff PR-2, Task 5: local automation definitions
        # (`family="recurring_question"` in v1) are real queue rows, not
        # gated behind a config flag the way watchlist/briefing checks are
        # -- `PriorityQueue.load` only ever arms a row that is already
        # `lifecycle="configured"` with a real `next_run_at`, so there is
        # nothing here for an absent handler to leave silently unhandled.
        #
        # ADR-097 (boot-census ratchet): `AutomationDefinitionHandler`'s
        # import chain (the handler module + `schedule_compute` +
        # `slot_keys`) stays OFF the boot path -- constructed lazily on
        # the first dispatched `automation_definition` row, not here at
        # wiring time, then cached on `self` so every later dispatch
        # reuses the SAME instance. The overlap-claim guard
        # (`_claimed`/`_pending` on the handler) only works across calls
        # if it is the same object each time; building a fresh handler
        # per dispatch would silently defeat that guard.
        async def _dispatch_automation_definition(task: dict[str, Any]) -> None:
            handler = self._get_automation_definition_handler()
            await handler.handle(task)

        handlers["automation_definition"] = _dispatch_automation_definition

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
            # dreams phase 1: no projection here -- attaching one would
            # import it before `_ui_ready` (census ratchet).
            missed_fire_grace_seconds=get_cli_setting(
                "scheduling", "missed_fire_grace_seconds", MISSED_FIRE_GRACE_SECONDS
            ),
            handler_timeout_seconds=get_cli_setting(
                "scheduling", "handler_timeout_seconds", HANDLER_TIMEOUT_SECONDS
            ),
            # UAT finding 3a: `on_queue_changed` (above) only reloads the
            # loop's OWN in-memory queue -- it never reaches a screen.
            # This is the fallback the plan calls for: a lightweight
            # `post_message` bridge so a fired reminder's row repaints
            # without navigating away and back.
            on_reminder_dispatched=self._post_reminder_dispatched,
        )
        # The report demo is opt-in. Keep its audio stack off first paint and
        # let the property above build it on the first demo entry-point read.
        self._daily_report_demo_service = None
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
            # Open-task queue feeds (spec §4); same lazy-self closure reason
            # as ingest_jobs_provider -- local_evaluation_service and
            # media_db are assigned later in __init__.
            eval_open_runs_provider=lambda: self._local_eval_open_run_counts(),
            read_later_count_provider=lambda: self._local_read_later_count(),
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
        # The local skills stack (trust service -> local service -> scope
        # facade) is built on first access, not here: constructing the trust
        # service performs OS keyring backend discovery TWICE (marker store +
        # key cache) for a feature most boots never touch (TASK-21111(b)).
        # Every consumer reads these through `getattr(app_instance, ...)` at
        # UI time, so a property is a drop-in.
        self._local_skill_trust_service: Any | None = None
        # task-33081: serializes the off-loop first build above.
        self._local_skill_trust_service_build_lock = asyncio.Lock()
        self._local_skills_service: Any | None = None
        self._skills_scope_service: Any | None = None
        # Captured NOW, at the timing the eager build had: `_build_local_
        # skills_stack` must not re-read collaborators that a test (or a
        # later boot step) may reassign between construction and first use
        # (the TASK-21108 deferral trap).
        self._local_skills_stack_inputs = (
            self.service_policy_enforcer,
            self.server_skills_service,
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
        self.notes_organization_repository = None
        self.local_first_sync_service = LocalFirstSyncService(
            server_service=self.server_sync_service,
            state_repository=self.sync_state_repository,
            local_store=getattr(self, "sync_v2_local_store", None),
            dataset_keys=self.sync_v2_dataset_keys,
            notes_organization_repository=self.notes_organization_repository,
            personal_context_runtime_loader=(
                self._load_personal_context_sync_runtime
            ),
        )
        self.sync_restore_service = SyncRestoreService(
            server_service=self.server_sync_service,
            local_store=getattr(self, "sync_v2_local_store", None),
            dataset_keys=self.sync_v2_dataset_keys,
            notes_organization_repository=self.notes_organization_repository,
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

    def _get_dreams_cycle_handler(self) -> Any:  # dreams phase 1
        """Lazily construct and memoize the Dreams cycle handler.

        ADR-097 (boot-census ratchet): the handler module's import chain
        stays off the boot path -- built on first dispatch, then cached on
        `self` (the `_get_automation_definition_handler` pattern). Shutdown
        reachability does not depend on which instance runs a cycle:
        spawned cycles are held in the handler module's own set, which
        `on_unmount` reaches directly.
        """
        handler = getattr(self, "_dreams_cycle_handler", None)
        if handler is None:
            from .Scheduling.scheduler.handlers.dreams_handler import (
                DreamsCycleHandler,
            )

            handler = DreamsCycleHandler(
                deps_getter=lambda: self._dreams_cycle_deps()
            )
            self._dreams_cycle_handler = handler
        return handler

    def get_dreams_db(self) -> Any:  # dreams phase 1
        """The Dreams DB, created on first use while Dreams is enabled.

        Returns ``None`` while ``[dreams] enabled`` is off (an off-by-default
        feature must not create storage) or when the database cannot open.
        Called from the UI thread and from the scheduler's queue-load
        worker thread, hence the lock around first construction.
        """
        from .Dreams.settings import dreams_setting

        if not dreams_setting("enabled"):
            return getattr(self, "dreams_db", None)
        with self._dreams_db_lock:
            if getattr(self, "dreams_db", None) is None:
                try:
                    from .DB.Dreams_DB import DreamsDB

                    self.dreams_db = DreamsDB(
                        get_dreams_db_path(), CLI_APP_CLIENT_ID
                    )
                except Exception:
                    logger.opt(exception=True).warning(
                        "Dreams DB unavailable",
                    )
                    return None
            return self.dreams_db

    def _dreams_cycle_deps(self):  # dreams phase 1
        """Build ``CycleDeps`` for a scheduled or catch-up Dreams cycle.

        Returns ``None`` when Dreams cannot run right now (disabled, or no
        Dreams database) so every caller -- the scheduled handler's
        ``deps_getter`` and the boot catch-up -- skips cleanly instead of
        constructing a cycle over missing handles. Every collaborator is a
        GETTER LAMBDA resolved at cycle time, never an instance captured
        here: this method is first reachable from the handler wiring inside
        ``__init__`` (same construction-order trap the
        ``chachanotes_db_getter`` comment block documents), and most of the
        handles below do not exist yet at wiring time. The imports are
        deferred to first real use (ADR-097 boot-census ratchet): the
        cycle-service chain loads only when a cycle actually runs.
        """
        from .Dreams.cycle_service import CycleDeps
        from .Dreams.settings import dreams_setting
        from .Dreams.story_service import resolve_dreams_chat

        if not dreams_setting("enabled"):
            return None
        dreams_db = self.get_dreams_db()
        if dreams_db is None:
            return None
        return CycleDeps(
            dreams_db=dreams_db,
            chachanotes_db_getter=lambda: getattr(self, "chachanotes_db", None),
            media_db_getter=lambda: getattr(self, "media_db", None),
            subs_db_getter=lambda: getattr(self, "subscriptions_db", None),
            # Raw attribute read, not `get_personal_context_service()`:
            # that method BOOTSTRAPS the service (imports the Personal
            # Context stack on first call) and the discovery cycle only
            # carries this handle for the profile-refresh flows -- a
            # missing service must stay a cheap None.
            pc_service_getter=lambda: getattr(
                self, "_personal_context_service", None
            ),
            # May raise RuntimeError when no provider resolves; run_cycle
            # catches that and degrades the cycle by design.
            chat_getter=resolve_dreams_chat,
        )

    def _start_dreams_boot_catchup(self) -> None:  # dreams phase 1
        """Run the Dreams boot catch-up worker when enabled and due.

        Guarded so a disabled Dreams never constructs deps -- or even
        imports the cycle chain (ADR-097). A COROUTINE worker, non-
        exclusive: the catch-up cycle runs alongside the scheduler rather
        than contending with its worker group.
        """
        from .Dreams.settings import dreams_setting

        if not (
            dreams_setting("enabled") and dreams_setting("catchup_enabled")
        ):
            return
        deps = self._dreams_cycle_deps()
        if deps is None:
            return
        from .Dreams.cycle_service import run_catchup_if_due

        self.run_worker(
            run_catchup_if_due(deps),
            exclusive=False,
            group="dreams",
        )

    def _wire_dreams_scheduler_integration(self) -> None:  # dreams phase 1
        """Wire Dreams into the live scheduler after `_ui_ready`.

        Every Dreams import -- the DB, the projection, the cycle chain --
        stays off the first-paint module census by landing here, AFTER the
        census's synchronous `_ui_ready` snapshot is taken but in the same
        slice as the scheduler worker start, so the worker's first queue
        load already sees the projection (the worker coroutine cannot run
        before this slice yields). The queue attribute is the same post-hoc
        seam the briefing settings refresh already uses
        (`self.scheduler_loop.queue.briefing_projection = ...`); the
        projection is built unconditionally because its `[dreams] enabled`
        gate and cadence are read live on every `tasks()` call. The Dreams
        DB itself is NOT built here: `get_dreams_db` creates it on first
        use while `[dreams] enabled` is set, so a default (disabled)
        install never creates `dreams.sqlite`.
        """
        from .Scheduling.services.dreams_projection import DreamsProjection

        self.scheduler_loop.queue.dreams_projection = DreamsProjection(
            self.get_dreams_db
        )
        self._start_dreams_boot_catchup()

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

    def _wire_watchlists_command_service(self) -> None:
        """Share one Console/UI Watchlists command facade over app owners."""
        from tldw_chatbook.Subscriptions.briefing_service import (
            resolve_persisted_briefing_defaults,
        )
        from tldw_chatbook.Tools.watchlists_command_service import (
            WatchlistsCommandService,
        )
        from tldw_chatbook.runtime_policy.bootstrap import (
            load_default_runtime_source_state,
        )

        scheduler = self.scheduler_loop
        coordinator = self.watchlists_operation_coordinator
        self.watchlists_command_service = WatchlistsCommandService(
            runtime_source_loader=load_default_runtime_source_state,
            create_sources_batch=self.local_watchlists_service.create_sources_exact_batch_sync,
            create_collection=self.watchlist_bundle_service.create_with_sources,
            update_collection_sources=self.watchlist_bundle_service.update_sources,
            accept_source_checks=coordinator.submit_checks,
            accept_briefing=coordinator.submit_briefing,
            resolve_collection_sources=self.watchlist_bundle_service.list_sources,
            set_briefing_schedule=self.subscriptions_db.set_watchlist_briefing_settings,
            briefing_schedules_enabled=lambda: bool(
                get_cli_setting("scheduling", "briefing_schedules_enabled", True)
            ),
            scheduler_running=lambda: bool(scheduler.running),
            request_scheduler_reload=scheduler.request_reload,
            wait_scheduler_reload=lambda token, timeout: (
                scheduler.wait_for_reload_blocking(token, timeout=timeout)
            ),
            default_briefing_defaults=resolve_persisted_briefing_defaults,
        )

    def apply_briefing_schedules_enabled(self, enabled: bool) -> Any:
        """Apply the persisted global briefing gate to existing runtime owners."""
        if type(enabled) is not bool:  # noqa: E721 -- persisted briefing gate requires an exact bool.
            raise TypeError("enabled must be a bool")
        projection = BriefingProjection(self.subscriptions_db) if enabled else None
        self.scheduling_service.briefing_projection = projection
        self.scheduler_loop.queue.briefing_projection = projection
        return self.scheduler_loop.request_reload()

    def _wire_llamacpp_snapshot_service(self) -> None:
        """Reserve lazy owner slots without importing the snapshot stack at boot."""
        self._llamacpp_snapshot_service = None
        self._llamacpp_snapshot_setup_task = None

    @property
    def llamacpp_snapshot_service(self) -> Any:
        """Compose snapshots on first Models/launch use (ADR-097, ADR-119)."""
        owner = self._llamacpp_snapshot_service
        if owner is None:
            from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
                snapshot_claim_is_live,
            )
            from tldw_chatbook.LLM_Management.snapshot_service import (
                LlamaCppSnapshotService,
            )

            owner = LlamaCppSnapshotService(
                None, lambda claim: snapshot_claim_is_live(self, claim)
            )
            self._llamacpp_snapshot_service = owner
        if self.is_running and self._llamacpp_snapshot_setup_task is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Pre-run callers and launch workers can inspect the owner safely.
                pass
            else:
                self._llamacpp_snapshot_setup_task = loop.create_task(
                    owner.initialize(
                        lambda: get_user_data_dir() / "llamacpp_snapshots"
                    ),
                    name="initialize_llamacpp_snapshots",
                )
        return owner

    @llamacpp_snapshot_service.setter
    def llamacpp_snapshot_service(self, service: Any) -> None:
        """Preserve the public injection seam used by screens and tests."""
        self._llamacpp_snapshot_service = service


# Stock Console capture reads only this factory's already resident facade.
_STOCK_PLUGIN_SERVICE_FACTORY = (
    ServiceWiringMixin._build_plugin_service,
    ServiceWiringMixin._build_plugin_service.__code__,
)


def _console_skill_metadata_current(metadata):
    """Check the three reused proof helpers before invoking their bodies."""
    import sys
    from importlib.machinery import ModuleSpec
    from types import FunctionType, MappingProxyType, ModuleType

    if type(metadata) is not ModuleType:
        return False
    if sys.modules.get(__package__ + ".Widgets.compact_model_bar") is not metadata:
        return False
    namespace = vars(metadata)
    source = namespace.get("_WIDGET_SOURCE")
    if type(source) is not tuple or len(source) != 6 or type(source[5]) is not tuple:
        return False
    defining, path, spec, origin, bindings, _records = source
    if (
        defining is not namespace
        or type(path) is not str  # noqa: E721 -- exact source metadata
        or type(origin) is not str  # noqa: E721 -- exact plain metadata; no custom dispatch
        or type(namespace.get("__file__")) is not str  # noqa: E721 -- exact source metadata
        or namespace.get("__file__") != path  # noqa: E721 -- exact plain metadata; no custom dispatch
        or namespace.get("__spec__") is not spec
        or type(spec) is not ModuleSpec
        or type(spec.origin) is not str  # noqa: E721 -- exact source metadata
        or spec.origin != origin
        or origin != path  # noqa: E721 -- exact plain metadata; no custom dispatch
        or type(bindings) is not tuple
    ):
        return False
    for row in bindings:
        if (
            type(row) is not tuple
            or len(row) != 3
            or type(row[1]) is not str  # noqa: E721 -- exact plain metadata; no custom dispatch
            or (type(row[0]) is not dict and type(row[0]) is not MappingProxyType)  # noqa: E721 -- exact plain metadata; no custom dispatch
            or row[0].get(row[1]) is not row[2]
        ):
            return False
    names = {"_plain_fields", "_source_current", "_function_current"}
    records = tuple(
        row
        for row in source[5]
        if type(row) is tuple
        and len(row) == 11
        and row[0] is namespace
        and type(row[1]) is str  # noqa: E721 -- plain capsule key
        and row[1] in names
    )
    return len(records) == 3 and all(
        namespace.get(row[1]) is row[2]
        and type(row[2]) is FunctionType
        and row[2].__code__ is row[3]
        and row[2].__globals__ is row[4] is namespace
        and row[2].__defaults__ is None
        and row[2].__kwdefaults__ is None
        and row[2].__closure__ is None
        for row in records
    )


from textual.dom import DOMNode as _SkillMessagePump  # noqa: E402
from textual.app import App as _SkillAppBase  # noqa: E402
from textual.message_pump import MessagePump as _SkillAppAccessor  # noqa: E402
from textual.worker_manager import WorkerManager as _SkillWorkerManager  # noqa: E402

_CONSOLE_SKILL_RUN_WORKER = _SkillMessagePump.run_worker
_CONSOLE_SKILL_WAIT = Worker.wait
_CONSOLE_SKILL_WORKERS = _SkillAppBase.__dict__["workers"]
_CONSOLE_SKILL_WORKERS_GET = _CONSOLE_SKILL_WORKERS.fget
_CONSOLE_SKILL_APP = _SkillAppAccessor.__dict__["app"]
_CONSOLE_SKILL_APP_GET = _CONSOLE_SKILL_APP.fget
_CONSOLE_SKILL_NEW_WORKER = _SkillWorkerManager._new_worker
_CONSOLE_SKILL_ADD_WORKER = _SkillWorkerManager.add_worker
_CONSOLE_SKILL_START_WORKER = Worker._start
_CONSOLE_SKILL_SCOPE_SLOTS = (
    "get_context",
    "_call",
    "_enforce_policy",
    "_require_service",
    "_normalize_mode",
    "_normalize_response",
    "_maybe_await",
    "_source_action_id",
    "_normalize_item",
    "_with_record_id",
)
_CONSOLE_SKILL_LOCAL_SLOTS = ("get_context", "trust_service")


def _console_skill_source_current(metadata, source):
    """Reuse the existing finite source capsule checker for this stock route."""
    if type(source) is not tuple or len(source) != 6:
        return False
    from importlib.machinery import ModuleSpec

    namespace, path, spec, origin, bindings, records = source
    if type(namespace) is not dict or type(path) is not str or type(origin) is not str:  # noqa: E721 -- exact stock metadata; no custom dispatch
        return False
    if (
        type(spec) is not ModuleSpec
        or type(spec.origin) is not str  # noqa: E721 -- exact stock metadata; no custom dispatch
        or type(namespace.get("__name__")) is not str  # noqa: E721 -- exact stock metadata; no custom dispatch
        or type(namespace.get("__file__")) is not str  # noqa: E721 -- exact stock metadata; no custom dispatch
        or namespace.get("__file__") != path
        or spec.origin != origin
        or path != origin
    ):
        return False
    if type(bindings) is not tuple or type(records) is not tuple:
        return False
    # Reject foreign names/containers before the existing checker hashes keys.
    for row in bindings:
        if type(row) is not tuple or len(row) != 3 or type(row[1]) is not str:  # noqa: E721 -- exact stock metadata; no custom dispatch
            return False
        if type(row[0]) is not dict and type(row[0]) is not _SkillMappingProxyType:  # noqa: E721 -- exact stock metadata; no custom dispatch
            return False
    for row in records:
        if type(row) is not tuple or len(row) != 11 or type(row[1]) is not str:  # noqa: E721 -- exact stock metadata; no custom dispatch
            return False
        if type(row[0]) is not dict and type(row[0]) is not _SkillMappingProxyType:  # noqa: E721 -- exact stock metadata; no custom dispatch
            return False
        if type(row[2]) is not _SkillFunctionType:
            return False
        defaults = row[2].__kwdefaults__
        if defaults is not None and (
            type(defaults) is not dict or any(type(key) is not str for key in defaults)  # noqa: E721 -- exact stock metadata; no custom dispatch
        ):
            return False
    return metadata._source_current(source)


def _capture_console_skill_trust_source(app, service):
    """Capture resident service sources without a UI scheduling dependency."""
    import inspect
    import sys
    from types import FunctionType, ModuleType

    metadata = sys.modules.get("tldw_chatbook.Widgets.compact_model_bar")
    if not _console_skill_metadata_current(metadata):
        return None
    app_module = sys.modules.get("tldw_chatbook.app")
    scope_module = sys.modules.get("tldw_chatbook.Skills_Interop.skills_scope_service")
    local_module = sys.modules.get("tldw_chatbook.Skills_Interop.local_skills_service")
    config_module = sys.modules.get("tldw_chatbook.config")
    if any(
        type(module) is not ModuleType
        for module in (app_module, scope_module, local_module, config_module)
    ):
        return None
    app_namespace = vars(app_module)
    app_record = app_namespace.get("_CONSOLE_SKILL_APP_SOURCE")
    if type(app_record) is not tuple or len(app_record) != 5:
        return None
    app_type, defining, factory, factory_code, runtime_type = app_record
    if (
        defining is not app_namespace
        or type(app) is not app_type
        or app_namespace.get("TldwCli") is not app_type
        or type(factory) is not FunctionType
        or factory.__globals__ is not defining
        or factory.__code__ is not factory_code
        or factory.__defaults__ is not None
        or factory.__kwdefaults__ is not None
        or factory.__closure__ is not None
        or inspect.getattr_static(app_type, "_create_deferred_startup_task", None)
        is not factory
    ):
        return None
    source = _CONSOLE_SKILL_WIRING_SOURCE
    scope_source = vars(scope_module).get("_CONSOLE_SKILL_CONTEXT_SOURCE")
    local_source = vars(local_module).get("_CONSOLE_SKILL_CONTEXT_SOURCE")
    config_source = vars(config_module).get("_COMPACT_MODEL_CONFIG_SOURCE")
    if source is not globals().get("_CONSOLE_SKILL_CONTEXT_SOURCE"):
        return None
    sources = (source, scope_source, local_source, config_source)
    if not all(_console_skill_source_current(metadata, row) for row in sources):
        return None
    app_slots = (
        "_build_local_skill_trust_service",
        "_build_local_skills_stack",
        "ensure_local_skill_trust_service",
        "local_skill_trust_service",
        "local_skills_service",
        "skills_scope_service",
    )
    if any(
        inspect.getattr_static(app_type, name, None)
        is not ServiceWiringMixin.__dict__[name]
        for name in app_slots
    ):
        return None
    scope_type = vars(scope_module).get("SkillsScopeService")
    local_type = vars(local_module).get("LocalSkillsService")
    app_fields = metadata._plain_fields(
        app,
        app_type,
        (
            "_local_skill_trust_service",
            "_local_skill_trust_service_build_lock",
            "_local_skills_service",
            "_skills_scope_service",
            "_local_skills_stack_inputs",
            "app_config",
            "_shutting_down",
            "_exit",
            "_workers",
            "_thread_id",
            "console_runtime",
            "_console_runtime_shutdown_task",
        ),
    )
    scope_fields = metadata._plain_fields(
        service, scope_type, ("local_service", "server_service", "policy_enforcer")
    )
    if app_fields is None or scope_fields is None:
        return None
    runtime = app_fields.get("console_runtime")
    if (
        runtime is None
        or app_fields.get("_console_runtime_shutdown_task") is not None
        or app_fields.get("_shutting_down") is not False
        or app_fields.get("_exit") is not False
    ):
        raise RuntimeError("console_skill_trust_runtime_closed")
    runtime_fields = metadata._plain_fields(
        runtime, runtime_type, ("_disposed", "_preparation_reads", "_app")
    )
    if (
        runtime_fields is None
        or type(runtime_fields.get("_preparation_reads")) is not set  # noqa: E721 -- exact source metadata
    ):  # noqa: E721 -- exact observer ownership
        return None
    runtime_reads = runtime_fields["_preparation_reads"]

    def runtime_current():
        return (
            app_namespace.get("ConsoleRuntime") is runtime_type
            and metadata._plain_fields(
                runtime, runtime_type, ("_disposed", "_preparation_reads", "_app")
            )
            is runtime_fields
            and app_fields.get("console_runtime") is runtime
            and app_fields.get("_console_runtime_shutdown_task") is None
            and app_fields.get("_shutting_down") is False
            and app_fields.get("_exit") is False
            and runtime_fields.get("_app") is app
            and runtime_fields.get("_disposed") is False
            and runtime_fields.get("_preparation_reads") is runtime_reads
        )

    # An original closing Runtime is a refusal, not eligibility for direct IO.
    if not runtime_current():
        raise RuntimeError("console_skill_trust_runtime_closed")
    local = scope_fields.get("local_service")
    local_fields = metadata._plain_fields(
        local,
        local_type,
        (
            "_trust_service",
            "_trust_service_factory",
            "store_dir",
            "skills_dir",
            "policy_enforcer",
        ),
    )
    if local_fields is None:
        return None
    if (
        any(name in app_fields for name in app_slots)
        or any(name in scope_fields for name in _CONSOLE_SKILL_SCOPE_SLOTS)
        or any(name in local_fields for name in _CONSOLE_SKILL_LOCAL_SLOTS)
    ):
        return None
    if (
        app_fields.get("_skills_scope_service") is not service
        or app_fields.get("_local_skills_service") is not local
        or app_fields.get("_shutting_down") is not False
        or app_fields.get("_exit") is not False
        or type(app_fields.get("app_config")) is not dict  # noqa: E721 -- exact stock metadata; no custom dispatch
    ):
        return None
    lazy = local_fields.get("_trust_service_factory")
    if (
        type(lazy) is not FunctionType
        or lazy.__code__ is not _CONSOLE_SKILL_TRUST_FACTORY_CODE
        or lazy.__globals__ is not globals()
        or lazy.__defaults__ is not None
        or lazy.__kwdefaults__ is not None
        or lazy.__code__.co_freevars != ("self",)
        or type(lazy.__closure__) is not tuple
        or len(lazy.__closure__) != 1
        or lazy.__closure__[0].cell_contents is not app
    ):
        return None
    lazy_closure = lazy.__closure__
    # Preserve the original captured policy collaborators; no verdict is reused.
    inputs = app_fields.get("_local_skills_stack_inputs")
    if type(inputs) is not tuple or len(inputs) != 2:
        return None
    if (
        scope_fields.get("policy_enforcer") is not inputs[0]
        or local_fields.get("policy_enforcer") is not inputs[0]
        or scope_fields.get("server_service") is not inputs[1]
    ):
        return None
    mapping = app_fields["app_config"]
    lock = app_fields["_local_skill_trust_service_build_lock"]
    if type(lock) is not asyncio.Lock:
        return None
    store_dir, skills_dir = (
        local_fields.get("store_dir"),
        local_fields.get("skills_dir"),
    )
    identity = config_module.current_config_identity()
    cache = vars(config_module).get("_SETTINGS_CACHE")
    posture = vars(config_module).get("_SETTINGS_CACHE_POSTURE")
    selector = tuple(
        os.environ.get(key)
        for key in (
            "TLDW_CONFIG_PATH",
            "HOME",
            "USERPROFILE",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        )
    )
    loop, thread = asyncio.get_running_loop(), threading.current_thread()
    if (
        type(app_fields.get("_thread_id")) is not int  # noqa: E721 -- exact stock metadata; no custom dispatch
        or app_fields.get("_thread_id") != thread.ident
    ):
        return None

    metadata_check = _console_skill_metadata_current
    source_check = _console_skill_source_current
    support = tuple(
        row for row in source[5] if row[2] is metadata_check or row[2] is source_check
    )
    if len(support) != 2 or any(
        row[5] is not None or row[6] is not None or row[8] is not None
        for row in support
    ):
        return None

    def source_current():
        # Check captured helper records without invoking either helper first.
        # Bound references alone do not reject an in-place body replacement.
        for row in support:
            namespace, name, function = row[:3]
            if (
                namespace.get(name) is not function
                or type(function) is not FunctionType
                or function.__code__ is not row[3]
                or function.__globals__ is not row[4]
                or function.__defaults__ is not None
                or function.__kwdefaults__ is not None
                or function.__closure__ is not None
                or type(vars(function)) is not dict  # noqa: E721 -- plain defining metadata
                or vars(function).get("__wrapped__") is not row[10]
            ):
                return False
        if not metadata_check(metadata):
            return False
        if not all(source_check(metadata, row) for row in sources):
            return False
        # Captured plain fields only: no Textual accessor or cached authority.
        if not (
            runtime_current()
            and sys.modules.get("tldw_chatbook.app") is app_module
            and app_namespace.get("_CONSOLE_SKILL_APP_SOURCE") is app_record
            and app_namespace.get("TldwCli") is app_type
            and factory.__code__ is factory_code
            and factory.__defaults__ is None
            and factory.__kwdefaults__ is None
            and factory.__closure__ is None
            and inspect.getattr_static(app_type, "_create_deferred_startup_task", None)
            is factory
            and config_module.current_config_identity() == identity
            and vars(config_module).get("_SETTINGS_CACHE") is cache
            and vars(config_module).get("_SETTINGS_CACHE_POSTURE") is posture
            and tuple(
                os.environ.get(key)
                for key in (
                    "TLDW_CONFIG_PATH",
                    "HOME",
                    "USERPROFILE",
                    "XDG_CONFIG_HOME",
                    "XDG_DATA_HOME",
                )
            )
            == selector
        ):
            return False
        return (
            metadata._plain_fields(
                app,
                app_type,
                (
                    "_local_skill_trust_service",
                    "_local_skill_trust_service_build_lock",
                    "_local_skills_service",
                    "_skills_scope_service",
                    "_local_skills_stack_inputs",
                    "app_config",
                    "_shutting_down",
                    "_exit",
                    "_workers",
                    "_thread_id",
                    "console_runtime",
                    "_console_runtime_shutdown_task",
                ),
            )
            is app_fields
            and metadata._plain_fields(
                service,
                scope_type,
                ("local_service", "server_service", "policy_enforcer"),
            )
            is scope_fields
            and metadata._plain_fields(
                local,
                local_type,
                (
                    "_trust_service",
                    "_trust_service_factory",
                    "store_dir",
                    "skills_dir",
                    "policy_enforcer",
                ),
            )
            is local_fields
            and app_fields.get("_skills_scope_service") is service
            and app_fields.get("_local_skills_service") is local
            and app_fields.get("_local_skill_trust_service_build_lock") is lock
            and app_fields.get("app_config") is mapping
            and type(app_fields.get("_thread_id")) is int  # noqa: E721 -- original exact source field
            and app_fields.get("_thread_id") == thread.ident
            and app_fields.get("_shutting_down") is False
            and app_fields.get("_exit") is False
            and all(
                inspect.getattr_static(app_type, name, None)
                is ServiceWiringMixin.__dict__[name]
                for name in app_slots
            )
            and not any(name in app_fields for name in app_slots)
            and local_fields.get("_trust_service_factory") is lazy
            and lazy.__code__ is _CONSOLE_SKILL_TRUST_FACTORY_CODE
            and lazy.__globals__ is globals()
            and lazy.__defaults__ is None
            and lazy.__kwdefaults__ is None
            and lazy.__closure__ is lazy_closure
            and lazy_closure[0].cell_contents is app
            and local_fields.get("store_dir") is store_dir
            and local_fields.get("skills_dir") is skills_dir
            and not any(name in scope_fields for name in _CONSOLE_SKILL_SCOPE_SLOTS)
            and not any(name in local_fields for name in _CONSOLE_SKILL_LOCAL_SLOTS)
            and scope_fields.get("server_service") is inputs[1]
            and app_fields.get("_local_skills_stack_inputs") is inputs
            and scope_fields.get("local_service") is local
            and scope_fields.get("policy_enforcer") is inputs[0]
            and local_fields.get("policy_enforcer") is inputs[0]
        )

    def current():
        try:
            return (
                asyncio.get_running_loop() is loop
                and threading.current_thread() is thread
                and source_current()
            )
        except (AttributeError, TypeError, ValueError):
            return False

    if not current():
        return None
    return _ConsoleSkillTrustSource(
        app, service, local, source_current, current, loop, thread, runtime_reads
    )


class _ConsoleSkillTrustSource(NamedTuple):
    """One stock service proof; no UI owner or permission verdict is captured."""

    app: Any
    scope: Any
    local: Any
    source_current: Callable[[], bool]
    current: Callable[[], bool]
    loop: Any
    thread: Any
    runtime_reads: set


def _capture_console_skill_trust_preparation(
    app, service, owner_current, controller_source
):
    """Add the original skill-discovery App-worker scheduling contract."""
    import inspect
    import sys

    source = _capture_console_skill_trust_source(app, service)
    if source is None:
        return None
    app_fields = vars(app)
    if (
        app_fields.get("_local_skill_trust_service") is not None
        or vars(source.local).get("_trust_service") is not None
    ):
        return None
    metadata = sys.modules.get("tldw_chatbook.Widgets.compact_model_bar")
    if not _console_skill_source_current(metadata, controller_source):
        return None
    manager = app_fields.get("_workers")
    manager_fields = metadata._plain_fields(
        manager, _SkillWorkerManager, ("_app", "_workers")
    )
    if manager_fields is None:
        return None
    manager_workers = manager_fields.get("_workers")

    def source_current():
        return source.source_current() and _console_skill_source_current(
            metadata, controller_source
        )

    def current(expected_trust=None):
        try:
            return (
                source.current()
                and source_current()
                and owner_current() is True
                and source.loop.get_task_factory() is None
                and app_fields.get("_local_skill_trust_service") is expected_trust
                and app_fields.get("_workers") is manager
                and metadata._plain_fields(
                    manager, _SkillWorkerManager, ("_app", "_workers")
                )
                is manager_fields
                and manager_fields.get("_app") is app
                and type(manager_workers) is set  # noqa: E721 -- original exact worker owner
                and manager_fields.get("_workers") is manager_workers
                and not any(
                    name in manager_fields for name in ("_new_worker", "add_worker")
                )
                and "workers" not in app_fields
                and "app" not in app_fields
                and inspect.getattr_static(type(app), "workers", None)
                is _CONSOLE_SKILL_WORKERS
                and inspect.getattr_static(type(app), "app", None) is _CONSOLE_SKILL_APP
                and _CONSOLE_SKILL_APP_GET(app) is app
                and "run_worker" not in app_fields
                and inspect.getattr_static(type(app), "run_worker", None)
                is _CONSOLE_SKILL_RUN_WORKER
            )
        except (AttributeError, TypeError, ValueError):
            return False

    if not current():
        return None
    return (
        app,
        source_current,
        current,
        (manager, manager_fields, source.loop, source.runtime_reads),
    )


async def _prepare_console_skill_trust_service(captured):
    """Retain a stock preparation in the App's selected worker drain scope."""
    app, source_current, current, custody = captured
    if not current():
        raise RuntimeError("console_skill_trust_owner_changed")
    preparation = ServiceWiringMixin.ensure_local_skill_trust_service(
        app,
        _source_current=source_current,
        _owner_current=current,
        _read_observers=(custody[3],),
    )
    try:
        worker = _CONSOLE_SKILL_RUN_WORKER(
            app,
            preparation,
            group="console-skill-trust-setup",
            exclusive=False,
            exit_on_error=False,
        )
    except BaseException:
        preparation.close()
        raise
    from textual.worker import WorkerCancelled, WorkerFailed

    if type(worker) is not Worker:
        raise RuntimeError("console_skill_trust_worker_changed")
    manager, manager_fields, loop, _runtime_reads = custody
    values = vars(worker)
    task = values.get("_task")
    if (
        vars(app).get("_workers") is not manager
        or vars(manager) is not manager_fields
        or manager_fields.get("_app") is not app
        or values.get("_node") is not app
        or values.get("_work") is not preparation
        or type(task) is not asyncio.Task
        or task.get_loop() is not loop
        or not any(item is worker for item in manager_fields["_workers"])
    ):
        raise RuntimeError("console_skill_trust_worker_changed")
    try:
        value = await _CONSOLE_SKILL_WAIT(worker)
    except WorkerCancelled:
        error = vars(worker).get("_error")
        if isinstance(error, asyncio.CancelledError):
            raise error
        raise asyncio.CancelledError
    except WorkerFailed:
        error = vars(worker).get("_error")
        if isinstance(error, BaseException):
            raise error
        raise
    if not current(value):
        raise RuntimeError("console_skill_trust_owner_changed")


# Definition-time originals for stock Console trust preparation only.
from types import (  # noqa: E402
    FunctionType as _SkillFunctionType,
    MappingProxyType as _SkillMappingProxyType,
)  # noqa: E402


_CONSOLE_SKILL_FUNCTIONS = {
    "_build_local_skill_trust_service": ServiceWiringMixin.__dict__[
        "_build_local_skill_trust_service"
    ],
    "_build_local_skills_stack": ServiceWiringMixin.__dict__[
        "_build_local_skills_stack"
    ],
    "ensure_local_skill_trust_service": ServiceWiringMixin.__dict__[
        "ensure_local_skill_trust_service"
    ],
    "local_skill_trust_service": ServiceWiringMixin.__dict__[
        "local_skill_trust_service"
    ].fget,
    "local_skills_service": ServiceWiringMixin.__dict__["local_skills_service"].fget,
    "skills_scope_service": ServiceWiringMixin.__dict__["skills_scope_service"].fget,
    "_console_skill_metadata_current": _console_skill_metadata_current,
    "_console_skill_source_current": _console_skill_source_current,
    "_capture_console_skill_trust_source": _capture_console_skill_trust_source,
    "_capture_console_skill_trust_preparation": _capture_console_skill_trust_preparation,
    "_prepare_console_skill_trust_service": _prepare_console_skill_trust_service,
    "_CONSOLE_SKILL_RUN_WORKER": _CONSOLE_SKILL_RUN_WORKER,
    "_CONSOLE_SKILL_WAIT": _CONSOLE_SKILL_WAIT,
    "_CONSOLE_SKILL_WORKERS_GET": _CONSOLE_SKILL_WORKERS_GET,
    "_CONSOLE_SKILL_APP_GET": _CONSOLE_SKILL_APP_GET,
    "_CONSOLE_SKILL_NEW_WORKER": _CONSOLE_SKILL_NEW_WORKER,
    "_CONSOLE_SKILL_ADD_WORKER": _CONSOLE_SKILL_ADD_WORKER,
    "_CONSOLE_SKILL_START_WORKER": _CONSOLE_SKILL_START_WORKER,
}
_CONSOLE_SKILL_CONTEXT_SOURCE = (
    globals(),
    __file__,
    __spec__,
    getattr(__spec__, "origin", None),
    (
        (globals(), "ServiceWiringMixin", ServiceWiringMixin),
        (globals(), "_CONSOLE_SKILL_FUNCTIONS", _CONSOLE_SKILL_FUNCTIONS),
        (
            ServiceWiringMixin.__dict__,
            "_build_local_skill_trust_service",
            ServiceWiringMixin.__dict__["_build_local_skill_trust_service"],
        ),
        (
            ServiceWiringMixin.__dict__,
            "_build_local_skills_stack",
            ServiceWiringMixin.__dict__["_build_local_skills_stack"],
        ),
        (
            ServiceWiringMixin.__dict__,
            "ensure_local_skill_trust_service",
            ServiceWiringMixin.__dict__["ensure_local_skill_trust_service"],
        ),
        (
            ServiceWiringMixin.__dict__,
            "local_skill_trust_service",
            ServiceWiringMixin.__dict__["local_skill_trust_service"],
        ),
        (
            ServiceWiringMixin.__dict__,
            "local_skills_service",
            ServiceWiringMixin.__dict__["local_skills_service"],
        ),
        (
            ServiceWiringMixin.__dict__,
            "skills_scope_service",
            ServiceWiringMixin.__dict__["skills_scope_service"],
        ),
        (globals(), "_ConsoleSkillTrustSource", _ConsoleSkillTrustSource),
        (
            globals(),
            "_capture_console_skill_trust_source",
            _capture_console_skill_trust_source,
        ),
        (globals(), "_console_skill_metadata_current", _console_skill_metadata_current),
        (globals(), "_console_skill_source_current", _console_skill_source_current),
        (
            globals(),
            "_capture_console_skill_trust_preparation",
            _capture_console_skill_trust_preparation,
        ),
        (
            globals(),
            "_prepare_console_skill_trust_service",
            _prepare_console_skill_trust_service,
        ),
        (globals(), "_CONSOLE_SKILL_RUN_WORKER", _CONSOLE_SKILL_RUN_WORKER),
        (globals(), "_CONSOLE_SKILL_WAIT", _CONSOLE_SKILL_WAIT),
        (globals(), "_CONSOLE_SKILL_WORKERS_GET", _CONSOLE_SKILL_WORKERS_GET),
        (globals(), "_CONSOLE_SKILL_APP_GET", _CONSOLE_SKILL_APP_GET),
        (globals(), "_CONSOLE_SKILL_NEW_WORKER", _CONSOLE_SKILL_NEW_WORKER),
        (globals(), "_CONSOLE_SKILL_ADD_WORKER", _CONSOLE_SKILL_ADD_WORKER),
        (globals(), "_CONSOLE_SKILL_START_WORKER", _CONSOLE_SKILL_START_WORKER),
        (globals(), "get_user_data_dir", get_user_data_dir),
        (globals(), "LocalSkillsService", LocalSkillsService),
        (globals(), "SkillsScopeService", SkillsScopeService),
        (globals(), "Worker", Worker),
        (globals(), "_CONSOLE_SKILL_SCOPE_SLOTS", _CONSOLE_SKILL_SCOPE_SLOTS),
        (globals(), "_CONSOLE_SKILL_LOCAL_SLOTS", _CONSOLE_SKILL_LOCAL_SLOTS),
        (globals(), "_SkillWorkerManager", _SkillWorkerManager),
        (globals(), "_CONSOLE_SKILL_WORKERS", _CONSOLE_SKILL_WORKERS),
        (globals(), "_CONSOLE_SKILL_APP", _CONSOLE_SKILL_APP),
        (_SkillAppBase.__dict__, "workers", _SkillAppBase.__dict__["workers"]),
        (_SkillAppAccessor.__dict__, "app", _SkillAppAccessor.__dict__["app"]),
        (
            _CONSOLE_SKILL_APP_GET.__globals__,
            "active_app",
            _CONSOLE_SKILL_APP_GET.__globals__["active_app"],
        ),
        (_CONSOLE_SKILL_NEW_WORKER.__globals__, "Worker", Worker),
        (
            _SkillWorkerManager.__dict__,
            "_new_worker",
            _SkillWorkerManager.__dict__["_new_worker"],
        ),
        (
            _SkillWorkerManager.__dict__,
            "add_worker",
            _SkillWorkerManager.__dict__["add_worker"],
        ),
        (Worker.__dict__, "_start", Worker.__dict__["_start"]),
        (Worker.__dict__, "wait", Worker.__dict__["wait"]),
        (
            _SkillMessagePump.__dict__,
            "run_worker",
            _SkillMessagePump.__dict__["run_worker"],
        ),
    ),
    tuple(
        (
            _CONSOLE_SKILL_FUNCTIONS,
            _skill_name,
            _skill_function,
            _skill_function.__code__,
            _skill_function.__globals__,
            _skill_function.__defaults__,
            _skill_function.__kwdefaults__,
            tuple((_skill_function.__kwdefaults__ or {}).items()),
            _skill_function.__closure__,
            tuple(
                (cell, cell.cell_contents) for cell in _skill_function.__closure__ or ()
            ),
            vars(_skill_function).get("__wrapped__"),
        )
        for _skill_name, _skill_function in _CONSOLE_SKILL_FUNCTIONS.items()
        if type(_skill_function) is _SkillFunctionType
    ),
)

_CONSOLE_SKILL_WIRING_SOURCE = _CONSOLE_SKILL_CONTEXT_SOURCE
_CONSOLE_SKILL_ENTRY = (
    _capture_console_skill_trust_preparation,
    _prepare_console_skill_trust_service,
)
_CONSOLE_SKILL_TRUST_FACTORY_CODE = next(
    code
    for code in ServiceWiringMixin._build_local_skills_stack.__code__.co_consts
    if isinstance(code, type(ServiceWiringMixin._build_local_skills_stack.__code__))
    and code.co_name == "<lambda>"
    and code.co_names == ("local_skill_trust_service",)
)
_CONSOLE_SKILL_CONTEXT_SOURCE = (
    *_CONSOLE_SKILL_CONTEXT_SOURCE[:4],
    _CONSOLE_SKILL_CONTEXT_SOURCE[4]
    + (
        (globals(), "_CONSOLE_SKILL_ENTRY", _CONSOLE_SKILL_ENTRY),
        (
            globals(),
            "_CONSOLE_SKILL_TRUST_FACTORY_CODE",
            _CONSOLE_SKILL_TRUST_FACTORY_CODE,
        ),
    ),
    _CONSOLE_SKILL_CONTEXT_SOURCE[5],
)
_CONSOLE_SKILL_WIRING_SOURCE = _CONSOLE_SKILL_CONTEXT_SOURCE


# Definition-time compatibility boundary for the finite Collections initializer.
from tldw_chatbook.Chat.console_preparation_reads import (  # noqa: E402
    run_preparation_read as _COLLECTIONS_RUN_PREPARATION,
)

_COLLECTIONS_TASK = asyncio.Task
_COLLECTIONS_SOURCE_CHECKER = (
    _collections_setup_sources_current,
    _collections_setup_sources_current.__code__,
)
_COLLECTIONS_CAPTURE_RESULT_FIELDS = (
    "collections_capture_repository",
    "collections_offline_store",
    "collections_legacy_recovery_service",
    "local_collections_capture_authority",
    "local_collections_capture_service",
)
_COLLECTIONS_APP_METHODS = tuple(
    (name, ServiceWiringMixin.__dict__[name])
    for name in (
        "_wire_collections_capture_services",
        "_reset_collections_capture_services",
        "ensure_collections_capture_services",
        "_deferred_wire_collections_capture_services",
        "_activate_collections_capture_authority",
        "_shutdown_collections_capture_runtime",
        "_reconcile_collections_capture_startup",
    )
)


# Original callbacks eligible for finite deferred Collections setup only.
def _record_collections_setup_source(entries):
    from types import FunctionType

    rows = []
    for owner, name in entries:
        original = owner[name] if type(owner) is dict else getattr(owner, name)  # noqa: E721 - exact stock compatibility boundary
        function = getattr(original, "__func__", original)
        records = []
        while type(function) is FunctionType:
            records.append(
                (
                    function,
                    function.__code__,
                    function.__globals__,
                    function.__defaults__,
                    function.__kwdefaults__,
                    tuple((function.__kwdefaults__ or {}).items()),
                    function.__closure__,
                    tuple(
                        (cell, cell.cell_contents)
                        for cell in function.__closure__ or ()
                    ),
                    vars(function).get("__wrapped__"),
                )
            )
            function = vars(function).get("__wrapped__")
        rows.append((owner, name, original, tuple(records)))
    return globals(), tuple(rows)


_COLLECTIONS_SETUP_SOURCE = _record_collections_setup_source(
    (
        (globals(), "_collections_setup_sources_current"),
        (globals(), "_capture_deferred_collections_setup"),
        (globals(), "_require_collections_setup_current"),
        (globals(), "_build_deferred_collections_capture_parts"),
        (globals(), "_initialize_deferred_collections_capture"),
        (globals(), "_retire_deferred_collections_capture"),
        (globals(), "_collections_capture_scope"),
        (globals(), "_collections_capture_parts"),
        (globals(), "_local_collections_capture_service"),
        (globals(), "_publish_collections_capture_parts"),
        (globals(), "get_library_collections_db_path"),
        (globals(), "get_user_data_dir"),
        (globals(), "LibraryCollectionsDB"),
        (globals(), "_COLLECTIONS_RUN_PREPARATION"),
        (globals(), "_COLLECTIONS_TASK"),
        (globals(), "_resolve_collections_media_reference"),
        (globals(), "_resolve_collections_note_reference"),
        (globals(), "_extract_collections_article"),
        (ServiceWiringMixin, "_wire_collections_capture_services"),
        (ServiceWiringMixin, "_reset_collections_capture_services"),
        (ServiceWiringMixin, "ensure_collections_capture_services"),
        (ServiceWiringMixin, "_deferred_wire_collections_capture_services"),
        (ServiceWiringMixin, "_activate_collections_capture_authority"),
        (ServiceWiringMixin, "_shutdown_collections_capture_runtime"),
        (ServiceWiringMixin, "_reconcile_collections_capture_startup"),
    )
)
del _record_collections_setup_source
