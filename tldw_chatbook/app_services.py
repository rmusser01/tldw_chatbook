"""The typed application services bundle (ADR-250, TASK-36947.2).

``TldwCli.__init__`` calls ``ServiceWiringMixin._publish_application_services``
once, after composition; that sets ``app.services`` from
``assemble_application_services``. The app attributes that composition assigns
stay in place as aliases of the bundle's fields (ADR-250 section 4).

The roots outside the TUI (the standalone MCP server, the mobile service and the
RAG backfill CLI) open their profile databases through ``open_profile_storage``
and close them with ``ProfileStorage.close()`` (ADR-250 section 6, TASK-36947.7).

Only the standard library is imported at module level (ADR-097 boot budget);
every service type is a ``TYPE_CHECKING`` import, and each opener imports its
database class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from .ACP_Interop.runtime_process import ACPRuntimeProcessManager
    from .Actor_Packs.activation import ActorPackActivationService
    from .Actor_Packs.controller import ActorPackExportController
    from .Actor_Packs.creation import ActorPackCreationService
    from .Actor_Packs.export import ActorPackExportService
    from .Actor_Packs.import_controller import ActorPackImportController
    from .Actor_Packs.importer import ActorPackImportService
    from .Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
    from .Actor_Packs.repository import ActorPackRepository
    from .Audio_Services_Interop.audio_services_scope_service import (
        AudioServicesScopeService,
    )
    from .Audio_Services_Interop.local_audio_services_service import (
        LocalAudioServicesService,
    )
    from .Audio_Services_Interop.server_audio_services_service import (
        ServerAudioServicesService,
    )
    from .Auth_Account_Interop.auth_account_scope_service import AuthAccountScopeService
    from .Auth_Account_Interop.server_auth_account_service import (
        ServerAuthAccountService,
    )
    from .Character_Chat.character_persona_scope_service import (
        CharacterPersonaScopeService,
    )
    from .Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService
    from .Character_Chat.local_character_persona_service import (
        LocalCharacterPersonaService,
    )
    from .Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
    from .Character_Chat.server_character_persona_service import (
        ServerCharacterPersonaService,
    )
    from .Character_Chat.server_chat_dictionary_service import (
        ServerChatDictionaryService,
    )
    from .Chat.chat_conversation_scope_service import ChatConversationScopeService
    from .Chat.chat_conversation_service import ChatConversationService
    from .Chat.citation_artifact_ownership import CitationArtifactOwnershipCoordinator
    from .Chat.citation_legacy_migration import CitationLegacyMigrationService
    from .Chat.citation_trace_repository import CitationTraceRepository
    from .Chat.conversation_local_marks_service import ConversationLocalMarksService
    from .Chat.server_chat_conversation_service import ServerChatConversationService
    from .Chat_Grammars_Interop.chat_grammars_scope_service import (
        ChatGrammarsScopeService,
    )
    from .Chat_Grammars_Interop.local_chat_grammars_service import (
        LocalChatGrammarsService,
    )
    from .Chat_Grammars_Interop.server_chat_grammars_service import (
        ServerChatGrammarsService,
    )
    from .Chatbooks.local_chatbook_service import LocalChatbookService
    from .Chatbooks.server_chatbook_service import ServerChatbookService
    from .Claims_Interop.claims_scope_service import ClaimsScopeService
    from .Claims_Interop.server_claims_service import ServerClaimsService
    from .Collections_Interop.collections_feeds_scope_service import (
        CollectionsFeedsScopeService,
    )
    from .Collections_Interop.server_collections_feeds_service import (
        ServerCollectionsFeedsService,
    )
    from .Companion_Interop.companion_scope_service import CompanionScopeService
    from .Companion_Interop.server_companion_service import ServerCompanionService
    from .DB.AgentRuns_DB import AgentRunsDB
    from .DB.ChaChaNotes_DB import CharactersRAGDB
    from .DB.Client_Media_DB_v2 import MediaDatabase
    from .DB.Prompts_DB import PromptsDatabase
    from .DB.Subscriptions_DB import SubscriptionsDB
    from .DB.Workspace_DB import WorkspaceDB
    from .Evaluations_Interop.evaluation_scope_service import EvaluationScopeService
    from .Evaluations_Interop.local_evaluations_service import LocalEvaluationsService
    from .Evaluations_Interop.server_evaluations_service import ServerEvaluationsService
    from .External_Connectors_Interop.connectors_scope_service import (
        ConnectorsScopeService,
    )
    from .External_Connectors_Interop.server_connectors_service import (
        ServerConnectorsService,
    )
    from .Feedback_Interop.feedback_scope_service import FeedbackScopeService
    from .Feedback_Interop.local_feedback_service import LocalFeedbackService
    from .Feedback_Interop.server_feedback_service import ServerFeedbackService
    from .LLM_Provider_Catalog.llm_provider_catalog_scope_service import (
        LLMProviderCatalogScopeService,
    )
    from .LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from .LLM_Provider_Catalog.model_discovery_disk_cache import ModelCatalogDiskStore
    from .LLM_Provider_Catalog.server_llm_provider_catalog_service import (
        ServerLLMProviderCatalogService,
    )
    from .Library.library_collections_service import LocalLibraryCollectionsService
    from .Library.library_local_rag_search_service import LibraryLocalRagSearchService
    from .MCP.local_control_service import LocalMCPControlService
    from .MCP.local_store import LocalMCPStore
    from .MCP.server_target_store import ConfiguredServerTargetStore
    from .MCP.server_unified_service import ServerUnifiedMCPService
    from .MCP.unified_context_store import UnifiedMCPContextStore
    from .MCP.unified_control_plane_service import UnifiedMCPControlPlaneService
    from .MCP_Governance_Interop.mcp_governance_scope_service import (
        MCPGovernanceScopeService,
    )
    from .MCP_Governance_Interop.server_mcp_governance_service import (
        ServerMCPGovernanceService,
    )
    from .Media.local_media_reading_service import LocalMediaReadingService
    from .Media.media_reading_scope_service import MediaReadingScopeService
    from .Media.server_media_reading_service import ServerMediaReadingService
    from .Meetings_Interop.meetings_scope_service import MeetingsScopeService
    from .Meetings_Interop.server_meetings_service import ServerMeetingsService
    from .Notes.Notes_Library import NotesInteropService
    from .Notes.file_notes_session_owner import FileNotesSessionOwner
    from .Notes.notes_scope_service import NotesScopeService
    from .Notes.server_notes_workspace_service import ServerNotesWorkspaceService
    from .Notifications.client_notifications_db import ClientNotificationsDB
    from .Notifications.client_notifications_service import ClientNotificationsService
    from .Notifications.event_state_repository import EventStateRepository
    from .Notifications.notification_dispatch_service import NotificationDispatchService
    from .Notifications.notifications_scope_service import NotificationsScopeService
    from .Notifications.server_notifications_service import ServerNotificationsService
    from .Outputs_Interop.outputs_scope_service import OutputsScopeService
    from .Outputs_Interop.server_outputs_service import ServerOutputsService
    from .Personalization_Interop.personalization_scope_service import (
        PersonalizationScopeService,
    )
    from .Personalization_Interop.server_personalization_service import (
        ServerPersonalizationService,
    )
    from .Prompt_Management.local_prompt_service import LocalPromptService
    from .Prompt_Management.prompt_chatbook_scope_service import (
        PromptChatbookScopeService,
    )
    from .Prompt_Management.prompt_scope_service import PromptScopeService
    from .Prompt_Management.server_prompt_service import ServerPromptService
    from .Prompt_Studio_Interop.prompt_studio_scope_service import (
        PromptStudioScopeService,
    )
    from .Prompt_Studio_Interop.server_prompt_studio_service import (
        ServerPromptStudioService,
    )
    from .Research_Interop.local_research_search_service import (
        LocalResearchSearchService,
    )
    from .Research_Interop.local_research_service import LocalResearchService
    from .Research_Interop.research_scope_service import ResearchScopeService
    from .Research_Interop.research_search_scope_service import (
        ResearchSearchScopeService,
    )
    from .Research_Interop.server_research_search_service import (
        ServerResearchSearchService,
    )
    from .Research_Interop.server_research_service import ServerResearchService
    from .Research_Workspace.paste_staging import ResearchPasteStagingStore
    from .Research_Workspace.source_association import (
        ResearchSourceAssociationCoordinator,
    )
    from .Research_Workspace.source_operation_store import ResearchSourceOperationStore
    from .Research_Workspace.source_readiness import ResearchSourceReadinessCoordinator
    from .Scheduling.services.scheduling_service import SchedulingService
    from .Server_Runtime_Interop.server_runtime_scope_service import (
        ServerRuntimeScopeService,
    )
    from .Server_Runtime_Interop.server_runtime_service import ServerRuntimeService
    from .Sharing_Interop.server_sharing_service import ServerSharingService
    from .Sharing_Interop.sharing_scope_service import SharingScopeService
    from .Study_Interop.local_quiz_service import LocalQuizService
    from .Study_Interop.local_study_service import LocalStudyService
    from .Study_Interop.quiz_scope_service import QuizScopeService
    from .Study_Interop.server_quiz_service import ServerQuizService
    from .Study_Interop.server_study_service import ServerStudyService
    from .Study_Interop.study_scope_service import StudyScopeService
    from .Subscriptions.local_watchlists_service import LocalWatchlistsService
    from .Subscriptions.server_watchlists_service import ServerWatchlistsService
    from .Subscriptions.watchlist_bundle_service import WatchlistBundleService
    from .Subscriptions.watchlist_scope_service import WatchlistScopeService
    from .Sync_Interop.local_first_sync_service import LocalFirstSyncService
    from .Sync_Interop.manual_sync_control import ManualSyncControlService
    from .Sync_Interop.restore_service import SyncRestoreService
    from .Sync_Interop.server_sync_service import ServerSyncService
    from .Sync_Interop.sync_scope_service import SyncScopeService
    from .Sync_Interop.sync_state_repository import SyncStateRepository
    from .TTS.TTS_Generation import TTSService
    from .Text2SQL_Interop.server_text2sql_service import ServerText2SQLService
    from .Text2SQL_Interop.text2sql_scope_service import Text2SQLScopeService
    from .Tools_Interop.server_tools_service import ServerToolsService
    from .Tools_Interop.tools_scope_service import ToolsScopeService
    from .Translation_Interop.server_translation_service import ServerTranslationService
    from .Translation_Interop.translation_scope_service import TranslationScopeService
    from .UI.Navigation.audio_cpp_model_handoff import AudioCppModelInstallOwner
    from .User_Governance_Interop.server_user_governance_service import (
        ServerUserGovernanceService,
    )
    from .User_Governance_Interop.user_governance_scope_service import (
        UserGovernanceScopeService,
    )
    from .Video_Generation.video_store import VideoStore
    from .Voice_Assistant_Interop.server_voice_assistant_service import (
        ServerVoiceAssistantService,
    )
    from .Voice_Assistant_Interop.voice_assistant_scope_service import (
        VoiceAssistantScopeService,
    )
    from .Web_Clipper_Interop.server_web_clipper_service import ServerWebClipperService
    from .Web_Clipper_Interop.web_clipper_scope_service import WebClipperScopeService
    from .Web_Scraping_Interop.server_web_scraping_service import (
        ServerWebScrapingService,
    )
    from .Web_Scraping_Interop.web_scraping_scope_service import WebScrapingScopeService
    from .Workspaces.change_review_consent import ChangeReviewConsentService
    from .Workspaces.registry_service import LocalWorkspaceRegistryService
    from .Writing_Interop.local_writing_service import LocalWritingService
    from .Writing_Interop.server_writing_service import ServerWritingService
    from .Writing_Interop.writing_scope_service import WritingScopeService
    from .runtime_policy.enforcement import ServicePolicyEnforcer
    from .runtime_policy.server_capabilities import ActiveServerCapabilityService
    from .runtime_policy.server_context import RuntimeServerContextProvider


@dataclass(frozen=True, slots=True, kw_only=True)
class ScopedServices[L, S, C]:
    """A feature with a server service and a scope service, and optionally a local one."""

    local: L | None = None
    server: S
    scope: C


@dataclass(frozen=True, slots=True, kw_only=True)
class CoreServices:
    server_context_provider: RuntimeServerContextProvider
    service_policy_enforcer: ServicePolicyEnforcer
    unified_mcp_target_store: ConfiguredServerTargetStore


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileStorage:
    """The profile databases (ADR-250 section 2).

    ``agent_runs`` is filled only by the mobile engine; in the TUI it is ``None``.
    The TUI's other databases stay in their feature slices or off the bundle.
    """

    chachanotes: CharactersRAGDB | None = None
    media: MediaDatabase | None = None
    prompts: PromptsDatabase | None = None
    workspaces: WorkspaceDB | None = None
    agent_runs: AgentRunsDB | None = None

    def close(self) -> None:
        """Close the calling thread's connection on each opened database.

        A failure is logged by type name and never raised (ADR-250 section 5).
        """
        for database in (
            self.chachanotes,
            self.media,
            self.prompts,
            self.workspaces,
            self.agent_runs,
        ):
            if database is None:
                continue
            try:
                database.close()
            except Exception as exc:  # noqa: BLE001 - teardown closes every database
                from loguru import logger

                logger.warning(
                    "profile_storage_close_failed database={} error={}",
                    type(database).__name__,
                    type(exc).__name__,
                )


# The shared factory (ADR-250 section 6). The openers are the only code that
# constructs these databases for the standalone MCP server, the mobile service
# and the RAG backfill CLI; every other construction is pinned by the census in
# Tests/Architecture/test_application_services_bundle.py (Q2).


def open_chachanotes_db(
    path: Path,
    *,
    client_id: str,
    check_integrity: bool = False,
    seed_builtin_content: bool = False,
) -> CharactersRAGDB:
    """Open ChaChaNotes; the Console Library migration seed is always passed."""
    from .config import load_console_library_migration_seed
    from .config import seed_builtin_content as seed_builtins
    from .DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(
        path,
        client_id=client_id,
        check_integrity_on_startup=check_integrity,
        console_library_migration_seed=load_console_library_migration_seed(),
    )
    if seed_builtin_content:
        seed_builtins(db)
    return db


def open_media_db(
    path: Path, *, client_id: str, check_integrity: bool = False
) -> MediaDatabase:
    """Open Media; ``check_integrity`` runs its startup integrity check."""
    from .DB.Client_Media_DB_v2 import MediaDatabase

    return MediaDatabase(
        path, client_id=client_id, check_integrity_on_startup=check_integrity
    )


def open_prompts_db(
    path: Path, *, client_id: str, check_integrity: bool = False
) -> PromptsDatabase:
    """Open Prompts; ``check_integrity`` runs its startup integrity check."""
    from .DB.Prompts_DB import PromptsDatabase

    return PromptsDatabase(
        path, client_id=client_id, check_integrity_on_startup=check_integrity
    )


def open_workspaces_db(path: Path, *, client_id: str) -> WorkspaceDB:
    """Open the workspaces database (it has no startup integrity check)."""
    from .DB.Workspace_DB import WorkspaceDB

    return WorkspaceDB(path, client_id=client_id)


def open_agent_runs_db(path: Path, *, client_id: str) -> AgentRunsDB:
    """Open the agent-runs database (it has no startup integrity check)."""
    from .DB.AgentRuns_DB import AgentRunsDB

    return AgentRunsDB(path, client_id=client_id)


def open_profile_storage(
    *,
    client_id: str,
    chachanotes: Path | None = None,
    media: Path | None = None,
    prompts: Path | None = None,
    workspaces: Path | None = None,
    agent_runs: Path | None = None,
    seed_builtin_content: bool = False,
    check_integrity: bool = False,
) -> ProfileStorage:
    """Open each database whose path is given, with the same options for every root.

    The others stay ``None``. ``check_integrity`` reaches the databases that have
    a startup integrity check (ChaChaNotes, Media, Prompts). Built-in content is
    seeded only when ``seed_builtin_content`` is set; only the TUI sets it.

    Raises:
        Exception: Whatever an opener raised, after the databases already
            opened have been closed.
    """
    opened: dict[str, Any] = {}
    try:
        if chachanotes is not None:
            opened["chachanotes"] = open_chachanotes_db(
                chachanotes,
                client_id=client_id,
                check_integrity=check_integrity,
                seed_builtin_content=seed_builtin_content,
            )
        if media is not None:
            opened["media"] = open_media_db(
                media, client_id=client_id, check_integrity=check_integrity
            )
        if prompts is not None:
            opened["prompts"] = open_prompts_db(
                prompts, client_id=client_id, check_integrity=check_integrity
            )
        if workspaces is not None:
            opened["workspaces"] = open_workspaces_db(workspaces, client_id=client_id)
        if agent_runs is not None:
            opened["agent_runs"] = open_agent_runs_db(agent_runs, client_id=client_id)
    except BaseException:
        ProfileStorage(**opened).close()
        raise
    return ProfileStorage(**opened)


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatServices:
    conversation: ScopedServices[
        ChatConversationService,
        ServerChatConversationService,
        ChatConversationScopeService,
    ]
    dictionary: ScopedServices[
        LocalChatDictionaryService,
        ServerChatDictionaryService,
        ChatDictionaryScopeService,
    ]
    grammars: ScopedServices[
        LocalChatGrammarsService, ServerChatGrammarsService, ChatGrammarsScopeService
    ]
    conversation_local_marks_service: ConversationLocalMarksService | None = None
    citation_trace_repository: CitationTraceRepository | None = None
    citation_artifact_ownership_coordinator: (
        CitationArtifactOwnershipCoordinator | None
    ) = None
    citation_legacy_migration_service: CitationLegacyMigrationService | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class NotesServices:
    server: ServerNotesWorkspaceService
    scope: NotesScopeService
    file_notes_session_owner: FileNotesSessionOwner
    local: NotesInteropService | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkspaceServices:
    workspace_registry_service: LocalWorkspaceRegistryService
    change_review_consent_service: ChangeReviewConsentService


@dataclass(frozen=True, slots=True, kw_only=True)
class McpServices:
    unified_mcp_service: UnifiedMCPControlPlaneService
    server_unified_mcp_service: ServerUnifiedMCPService
    local_mcp_control_service: LocalMCPControlService
    local_mcp_store: LocalMCPStore
    unified_mcp_context_store: UnifiedMCPContextStore
    governance: ScopedServices[
        None, ServerMCPGovernanceService, MCPGovernanceScopeService
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class SyncServices:
    server_sync_service: ServerSyncService
    sync_scope_service: SyncScopeService
    local_first_sync_service: LocalFirstSyncService
    manual_sync_control_service: ManualSyncControlService
    sync_state_repository: SyncStateRepository
    sync_restore_service: SyncRestoreService


@dataclass(frozen=True, slots=True, kw_only=True)
class WatchlistsServices:
    local: LocalWatchlistsService
    server: ServerWatchlistsService
    scope: WatchlistScopeService
    subscriptions_db: SubscriptionsDB
    watchlist_bundle_service: WatchlistBundleService


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeServices:
    server: ServerRuntimeService
    scope: ServerRuntimeScopeService
    active_server_capability_service: ActiveServerCapabilityService


@dataclass(frozen=True, slots=True, kw_only=True)
class ResearchServices:
    local: LocalResearchService
    server: ServerResearchService
    scope: ResearchScopeService
    research_paste_staging_store: ResearchPasteStagingStore
    research_source_association_coordinator: ResearchSourceAssociationCoordinator
    research_source_operation_store: ResearchSourceOperationStore
    research_source_readiness_coordinator: ResearchSourceReadinessCoordinator


@dataclass(frozen=True, slots=True, kw_only=True)
class NotificationsServices:
    server: ServerNotificationsService
    scope: NotificationsScopeService
    client_notifications_db: ClientNotificationsDB
    client_notifications_service: ClientNotificationsService
    notification_dispatch_service: NotificationDispatchService


@dataclass(frozen=True, slots=True, kw_only=True)
class LibraryServices:
    library_rag_search_service: LibraryLocalRagSearchService
    local_library_collections_service: LocalLibraryCollectionsService | None = None
    library_collections_service: LocalLibraryCollectionsService | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ActorPackServices:
    actor_pack_repository: ActorPackRepository
    actor_pack_creation_service: ActorPackCreationService
    actor_pack_export_service: ActorPackExportService
    actor_pack_export_controller: ActorPackExportController
    persona_actor_pack_coordinator: PersonaActorPackCoordinator
    actor_pack_import_service: ActorPackImportService | None = None
    actor_pack_activation_service: ActorPackActivationService | None = None
    actor_pack_import_controller: ActorPackImportController | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PlatformServices:
    acp_runtime_process_manager: ACPRuntimeProcessManager
    audio_cpp_model_install_owner: AudioCppModelInstallOwner
    tts_service: TTSService
    generated_video_store: VideoStore
    model_catalog_disk_store: ModelCatalogDiskStore
    event_state_repository: EventStateRepository
    scheduling_service: SchedulingService


@dataclass(frozen=True, slots=True, kw_only=True)
class ApplicationServices:
    """The typed bundle ``TldwCli.services`` holds. Never mutated after publication."""

    core: CoreServices
    storage: ProfileStorage
    chat: ChatServices
    notes: NotesServices
    media: ScopedServices[
        LocalMediaReadingService, ServerMediaReadingService, MediaReadingScopeService
    ]
    workspace: WorkspaceServices
    mcp: McpServices
    sync: SyncServices
    watchlists: WatchlistsServices
    quiz: ScopedServices[LocalQuizService, ServerQuizService, QuizScopeService]
    chatbook: ScopedServices[
        LocalChatbookService, ServerChatbookService, PromptChatbookScopeService
    ]
    runtime: RuntimeServices
    research: ResearchServices
    notifications: NotificationsServices
    library: LibraryServices
    actor_packs: ActorPackServices
    platform: PlatformServices
    audio_services: ScopedServices[
        LocalAudioServicesService, ServerAudioServicesService, AudioServicesScopeService
    ]
    auth_account: ScopedServices[
        None, ServerAuthAccountService, AuthAccountScopeService
    ]
    character_persona: ScopedServices[
        LocalCharacterPersonaService,
        ServerCharacterPersonaService,
        CharacterPersonaScopeService,
    ]
    claims: ScopedServices[None, ServerClaimsService, ClaimsScopeService]
    collections_feeds: ScopedServices[
        None, ServerCollectionsFeedsService, CollectionsFeedsScopeService
    ]
    companion: ScopedServices[None, ServerCompanionService, CompanionScopeService]
    connectors: ScopedServices[None, ServerConnectorsService, ConnectorsScopeService]
    evaluation: ScopedServices[
        LocalEvaluationsService, ServerEvaluationsService, EvaluationScopeService
    ]
    feedback: ScopedServices[
        LocalFeedbackService, ServerFeedbackService, FeedbackScopeService
    ]
    llm_provider_catalog: ScopedServices[
        LocalLLMProviderCatalogService,
        ServerLLMProviderCatalogService,
        LLMProviderCatalogScopeService,
    ]
    meetings: ScopedServices[None, ServerMeetingsService, MeetingsScopeService]
    outputs: ScopedServices[None, ServerOutputsService, OutputsScopeService]
    personalization: ScopedServices[
        None, ServerPersonalizationService, PersonalizationScopeService
    ]
    prompts: ScopedServices[LocalPromptService, ServerPromptService, PromptScopeService]
    prompt_studio: ScopedServices[
        None, ServerPromptStudioService, PromptStudioScopeService
    ]
    research_search: ScopedServices[
        LocalResearchSearchService,
        ServerResearchSearchService,
        ResearchSearchScopeService,
    ]
    sharing: ScopedServices[None, ServerSharingService, SharingScopeService]
    study: ScopedServices[LocalStudyService, ServerStudyService, StudyScopeService]
    text2sql: ScopedServices[None, ServerText2SQLService, Text2SQLScopeService]
    tools: ScopedServices[None, ServerToolsService, ToolsScopeService]
    translation: ScopedServices[None, ServerTranslationService, TranslationScopeService]
    user_governance: ScopedServices[
        None, ServerUserGovernanceService, UserGovernanceScopeService
    ]
    voice_assistant: ScopedServices[
        None, ServerVoiceAssistantService, VoiceAssistantScopeService
    ]
    web_clipper: ScopedServices[None, ServerWebClipperService, WebClipperScopeService]
    web_scraping: ScopedServices[
        None, ServerWebScrapingService, WebScrapingScopeService
    ]
    writing: ScopedServices[
        LocalWritingService, ServerWritingService, WritingScopeService
    ]


def _optional(app: Any, name: str) -> Any:
    """Read a service that composition leaves ``None`` when it is unavailable.

    The name is not a literal at the ``getattr`` call, so these reads stay out of
    ``test_app_string_lookup_census`` (they are composition output, not consumers).
    """
    return getattr(app, name, None)


def assemble_application_services(app: Any) -> ApplicationServices:
    """Read the services composition has just assigned onto ``app`` into a bundle."""
    return ApplicationServices(
        core=CoreServices(
            server_context_provider=app.server_context_provider,
            service_policy_enforcer=app.service_policy_enforcer,
            unified_mcp_target_store=app.unified_mcp_target_store,
        ),
        storage=ProfileStorage(
            chachanotes=_optional(app, "chachanotes_db"),
            media=_optional(app, "media_db"),
            prompts=_optional(app, "prompts_db"),
            workspaces=_optional(app, "local_workspace_db"),
        ),
        chat=ChatServices(
            conversation=ScopedServices(
                local=_optional(app, "local_chat_conversation_service"),
                server=app.server_chat_conversation_service,
                scope=app.chat_conversation_scope_service,
            ),
            dictionary=ScopedServices(
                local=app.local_chat_dictionary_service,
                server=app.server_chat_dictionary_service,
                scope=app.chat_dictionary_scope_service,
            ),
            grammars=ScopedServices(
                local=app.local_chat_grammars_service,
                server=app.server_chat_grammars_service,
                scope=app.chat_grammars_scope_service,
            ),
            conversation_local_marks_service=_optional(app, "conversation_local_marks_service"),
            citation_trace_repository=_optional(app, "citation_trace_repository"),
            citation_artifact_ownership_coordinator=_optional(app, "citation_artifact_ownership_coordinator"),
            citation_legacy_migration_service=_optional(app, "citation_legacy_migration_service"),
        ),
        notes=NotesServices(
            local=_optional(app, "notes_service"),
            server=app.server_notes_workspace_service,
            scope=app.notes_scope_service,
            file_notes_session_owner=app.file_notes_session_owner,
        ),
        media=ScopedServices(
            local=app.local_media_reading_service,
            server=app.server_media_reading_service,
            scope=app.media_reading_scope_service,
        ),
        workspace=WorkspaceServices(
            workspace_registry_service=app.workspace_registry_service,
            change_review_consent_service=app.change_review_consent_service,
        ),
        mcp=McpServices(
            unified_mcp_service=app.unified_mcp_service,
            server_unified_mcp_service=app.server_unified_mcp_service,
            local_mcp_control_service=app.local_mcp_control_service,
            local_mcp_store=app.local_mcp_store,
            unified_mcp_context_store=app.unified_mcp_context_store,
            governance=ScopedServices(
                server=app.server_mcp_governance_service,
                scope=app.mcp_governance_scope_service,
            ),
        ),
        sync=SyncServices(
            server_sync_service=app.server_sync_service,
            sync_scope_service=app.sync_scope_service,
            local_first_sync_service=app.local_first_sync_service,
            manual_sync_control_service=app.manual_sync_control_service,
            sync_state_repository=app.sync_state_repository,
            sync_restore_service=app.sync_restore_service,
        ),
        watchlists=WatchlistsServices(
            local=app.local_watchlists_service,
            server=app.server_watchlists_service,
            scope=app.watchlist_scope_service,
            subscriptions_db=app.subscriptions_db,
            watchlist_bundle_service=app.watchlist_bundle_service,
        ),
        quiz=ScopedServices(
            local=_optional(app, "local_quiz_service"),
            server=app.server_quiz_service,
            scope=app.study_quiz_scope_service,
        ),
        chatbook=ScopedServices(
            local=app.local_chatbook_service,
            server=app.server_chatbook_service,
            scope=app.prompt_chatbook_scope_service,
        ),
        runtime=RuntimeServices(
            server=app.server_runtime_service,
            scope=app.server_runtime_scope_service,
            active_server_capability_service=app.active_server_capability_service,
        ),
        research=ResearchServices(
            local=app.local_research_service,
            server=app.server_research_service,
            scope=app.research_scope_service,
            research_paste_staging_store=app.research_paste_staging_store,
            research_source_association_coordinator=app.research_source_association_coordinator,
            research_source_operation_store=app.research_source_operation_store,
            research_source_readiness_coordinator=app.research_source_readiness_coordinator,
        ),
        notifications=NotificationsServices(
            server=app.server_notifications_service,
            scope=app.notifications_scope_service,
            client_notifications_db=app.client_notifications_db,
            client_notifications_service=app.client_notifications_service,
            notification_dispatch_service=app.notification_dispatch_service,
        ),
        library=LibraryServices(
            local_library_collections_service=_optional(app, "local_library_collections_service"),
            library_collections_service=_optional(app, "library_collections_service"),
            library_rag_search_service=app.library_rag_search_service,
        ),
        actor_packs=ActorPackServices(
            actor_pack_repository=app.actor_pack_repository,
            actor_pack_creation_service=app.actor_pack_creation_service,
            actor_pack_export_service=app.actor_pack_export_service,
            actor_pack_export_controller=app.actor_pack_export_controller,
            actor_pack_import_service=_optional(app, "actor_pack_import_service"),
            actor_pack_activation_service=_optional(app, "actor_pack_activation_service"),
            actor_pack_import_controller=_optional(app, "actor_pack_import_controller"),
            persona_actor_pack_coordinator=app.persona_actor_pack_coordinator,
        ),
        platform=PlatformServices(
            acp_runtime_process_manager=app.acp_runtime_process_manager,
            audio_cpp_model_install_owner=app.audio_cpp_model_install_owner,
            tts_service=app.tts_service,
            generated_video_store=app.generated_video_store,
            model_catalog_disk_store=app.model_catalog_disk_store,
            event_state_repository=app.event_state_repository,
            scheduling_service=app.scheduling_service,
        ),
        audio_services=ScopedServices(
            local=app.local_audio_services_service,
            server=app.server_audio_services_service,
            scope=app.audio_services_scope_service,
        ),
        auth_account=ScopedServices(
            server=app.server_auth_account_service, scope=app.auth_account_scope_service
        ),
        character_persona=ScopedServices(
            local=app.local_character_persona_service,
            server=app.server_character_persona_service,
            scope=app.character_persona_scope_service,
        ),
        claims=ScopedServices(
            server=app.server_claims_service, scope=app.claims_scope_service
        ),
        collections_feeds=ScopedServices(
            server=app.server_collections_feeds_service,
            scope=app.collections_feeds_scope_service,
        ),
        companion=ScopedServices(
            server=app.server_companion_service, scope=app.companion_scope_service
        ),
        connectors=ScopedServices(
            server=app.server_connectors_service, scope=app.connectors_scope_service
        ),
        evaluation=ScopedServices(
            local=app.local_evaluation_service,
            server=app.server_evaluation_service,
            scope=app.evaluation_scope_service,
        ),
        feedback=ScopedServices(
            local=app.local_feedback_service,
            server=app.server_feedback_service,
            scope=app.feedback_scope_service,
        ),
        llm_provider_catalog=ScopedServices(
            local=app.local_llm_provider_catalog_service,
            server=app.server_llm_provider_catalog_service,
            scope=app.llm_provider_catalog_scope_service,
        ),
        meetings=ScopedServices(
            server=app.server_meetings_service, scope=app.meetings_scope_service
        ),
        outputs=ScopedServices(
            server=app.server_outputs_service, scope=app.outputs_scope_service
        ),
        personalization=ScopedServices(
            server=app.server_personalization_service,
            scope=app.personalization_scope_service,
        ),
        prompts=ScopedServices(
            local=app.local_prompt_service,
            server=app.server_prompt_service,
            scope=app.prompt_scope_service,
        ),
        prompt_studio=ScopedServices(
            server=app.server_prompt_studio_service,
            scope=app.prompt_studio_scope_service,
        ),
        research_search=ScopedServices(
            local=app.local_research_search_service,
            server=app.server_research_search_service,
            scope=app.research_search_scope_service,
        ),
        sharing=ScopedServices(
            server=app.server_sharing_service, scope=app.sharing_scope_service
        ),
        study=ScopedServices(
            local=_optional(app, "local_study_service"),
            server=app.server_study_service,
            scope=app.study_scope_service,
        ),
        text2sql=ScopedServices(
            server=app.server_text2sql_service, scope=app.text2sql_scope_service
        ),
        tools=ScopedServices(
            server=app.server_tools_service, scope=app.tools_scope_service
        ),
        translation=ScopedServices(
            server=app.server_translation_service, scope=app.translation_scope_service
        ),
        user_governance=ScopedServices(
            server=app.server_user_governance_service,
            scope=app.user_governance_scope_service,
        ),
        voice_assistant=ScopedServices(
            server=app.server_voice_assistant_service,
            scope=app.voice_assistant_scope_service,
        ),
        web_clipper=ScopedServices(
            server=app.server_web_clipper_service, scope=app.web_clipper_scope_service
        ),
        web_scraping=ScopedServices(
            server=app.server_web_scraping_service, scope=app.web_scraping_scope_service
        ),
        writing=ScopedServices(
            local=app.local_writing_service,
            server=app.server_writing_service,
            scope=app.writing_scope_service,
        ),
    )
