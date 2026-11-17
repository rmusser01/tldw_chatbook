"""Open the Console backend's original collaborators for one selected profile."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import config
from ..app_services import (
    ProfileStorage,
    open_chachanotes_db,
    open_prompts_db,
    open_workspaces_db,
)
from ..Backup_Recovery import storage_admission
from ..Backup_Recovery.chat_source_participants import bind_citation_services
from ..Backup_Recovery.participants import retired_core_connections
from ..MCP.local_control_service import LocalMCPControlService
from ..MCP.local_store import LocalMCPStore
from ..MCP.server_target_store import ConfiguredServerTargetStore
from ..MCP.server_unified_service import ServerUnifiedMCPService
from ..MCP.unified_context_store import UnifiedMCPContextStore
from ..MCP.unified_control_plane_service import UnifiedMCPControlPlaneService
from ..Prompt_Management.prompt_scope_service import (
    PromptScopeService,
    build_prompt_scope_service,
)
from ..runtime_policy.bootstrap import (
    LegacyConfigServerClientProvider,
    RuntimePolicyContext,
    build_runtime_api_client,
    default_runtime_policy_path,
    synchronize_runtime_source_state_with_app_config,
)
from ..runtime_policy.enforcement import ServicePolicyEnforcer
from ..runtime_policy.server_context import (
    RuntimeServerContextProvider,
    default_server_credential_profile_id,
)
from ..runtime_policy.server_credentials import (
    CredentialStoreUnavailable,
    ServerCredentialStore,
    UnavailableServerCredentialStore,
    build_default_server_credential_store,
)
from ..runtime_policy.server_parity_state import SYNC_STATE_DB_FILENAME
from ..runtime_policy.source_state import RuntimeSourceStateStore
from ..runtime_policy.types import RuntimeSourceState
from ..Skills_Interop import (
    LocalSkillsService,
    ServerSkillsService,
    SkillsScopeService,
    default_local_skills_store_dir,
)
from ..Sync_Interop import ServerSyncService, SyncScopeService
from ..Sync_Interop.sync_state_repository import SyncStateRepository
from ..Utils.private_paths import lexical_path, verify_trusted_directory
from ..Workspaces import (
    ChangeReviewConsentService,
    DeferredWorkspaceToolProfileGuard,
    LocalWorkspaceRegistryService,
)
from ..Workspaces.models import DEFAULT_WORKSPACE_ID
from .chat_conversation_scope_service import ChatConversationScopeService
from .citation_service_factory import build_local_citation_conversation_service
from .citation_trace_repository import CitationTraceRepository
from .conversation_local_marks_service import ConversationLocalMarksService
from .server_chat_conversation_service import ServerChatConversationService

if TYPE_CHECKING:
    from ..tldw_api import TLDWAPIClient


@dataclass(slots=True)
class _PolicyPublication:
    state: RuntimeSourceState

    def publish(self, state: RuntimeSourceState) -> None:
        self.state = state


class ConsoleBackendProfileCleanupUncertain(RuntimeError):
    """An original creator-owned connection did not positively close."""

    def __init__(self, repository: object, connection: sqlite3.Connection) -> None:
        super().__init__("Console backend profile connection retirement is unproven")
        self.repository = repository
        self.connection = connection


class ConsoleBackendProfileOpenFailure(RuntimeError):
    """Retain real partially acquired owners when their cleanup fails."""

    def __init__(
        self,
        *,
        acquisition_error: BaseException,
        cleanup_error: BaseException,
        storage: ProfileStorage,
        sync_repository: SyncStateRepository | None,
        server_context: RuntimeServerContextProvider,
        control_plane: UnifiedMCPControlPlaneService | None,
    ) -> None:
        super().__init__(
            "Console backend profile acquisition failed with unresolved cleanup"
        )
        self.acquisition_error = acquisition_error
        self.cleanup_error = cleanup_error
        self.storage = storage
        self.sync_repository = sync_repository
        self.server_context = server_context
        self.control_plane = control_plane


def _close_repository(repository, *, local):
    current = threading.current_thread()
    participant = getattr(repository, "_maintenance_participant", None)
    cache = getattr(local, "conn", None)
    registry = getattr(repository, "_connection_quiescence", None)
    with storage_admission._lock:
        leases = {} if participant is None else participant.connections
        owned = {
            connection: lease
            for connection, lease in leases.items()
            if lease.resource_thread is current
        }
        if cache is not None:
            owned.setdefault(cache, leases.get(cache))

    repository.close()
    for connection, lease in owned.items():
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            pass
        else:
            raise ConsoleBackendProfileCleanupUncertain(repository, connection)
        with storage_admission._lock:
            retired = (
                connection in retired_core_connections
                and connection not in leases
                and (
                    lease is None
                    or (
                        lease not in storage_admission._live_leases
                        and lease._key is None
                    )
                )
            )
        if not retired or (registry is not None and registry.is_registered(connection)):
            raise ConsoleBackendProfileCleanupUncertain(repository, connection)


def _close_repositories(
    storage: ProfileStorage,
    sync_repository: SyncStateRepository | None,
) -> BaseException | None:
    first_error = None
    for repository, cache_attr in (
        (storage.workspaces, "_thread_local"),
        (storage.prompts, "_local"),
        (storage.chachanotes, "_local"),
        (sync_repository, None),
    ):
        if repository is None:
            continue
        try:
            local = getattr(repository, cache_attr) if cache_attr is not None else None
            _close_repository(repository, local=local)
        except BaseException as error:
            if first_error is None:
                first_error = error
    return first_error


@dataclass(slots=True)
class _ProfileCleanup:
    attempted: bool = False
    error: BaseException | None = None
    unresolved_cached_clients: list[TLDWAPIClient] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ConsoleBackendProfile:
    """Concrete original profile owners acquired on the backend owner thread."""

    configuration: dict[str, Any] = field(repr=False)
    configuration_identity: tuple[int, str] = field(repr=False)
    config_path: Path = field(repr=False)
    data_directory: Path = field(repr=False)
    storage: ProfileStorage = field(repr=False)
    policy_context: RuntimePolicyContext = field(repr=False)
    policy_enforcer: ServicePolicyEnforcer = field(repr=False)
    server_context: RuntimeServerContextProvider = field(repr=False)
    target_store: ConfiguredServerTargetStore = field(repr=False)
    workspace_registry: LocalWorkspaceRegistryService = field(repr=False)
    consent_service: ChangeReviewConsentService = field(repr=False)
    conversation_scope_service: ChatConversationScopeService = field(repr=False)
    conversation_marks: ConversationLocalMarksService = field(repr=False)
    citation_repository: CitationTraceRepository = field(repr=False)
    prompt_scope: PromptScopeService = field(repr=False)
    local_skills: LocalSkillsService = field(repr=False)
    skills_scope: SkillsScopeService = field(repr=False)
    control_plane: UnifiedMCPControlPlaneService = field(repr=False)
    tool_profile_guard: DeferredWorkspaceToolProfileGuard = field(repr=False)
    sync_scope: SyncScopeService = field(repr=False)
    _sync_repository: SyncStateRepository = field(repr=False)
    _policy_publication: _PolicyPublication = field(repr=False)
    _creator: threading.Thread = field(
        init=False, repr=False, default_factory=threading.current_thread
    )
    _close_lock: asyncio.Lock = field(
        init=False, repr=False, default_factory=asyncio.Lock
    )
    _cleanup: _ProfileCleanup = field(
        init=False, repr=False, default_factory=_ProfileCleanup
    )

    @property
    def published_policy_state(self) -> RuntimeSourceState:
        """Return the state published by the original policy owner's callback."""
        return self._policy_publication.state

    @property
    def unresolved_cached_clients(self) -> tuple[TLDWAPIClient, ...]:
        """Retain the acquired clients whose original close could not settle."""
        return tuple(self._cleanup.unresolved_cached_clients)

    async def _disconnect_local_mcp(self) -> None:
        client = self.control_plane.local_service.client
        if client is not None and client.sessions:
            await client.disconnect_all()

    async def aclose(self) -> None:
        """Close acquired collaborators after the worker has settled its work."""
        if threading.current_thread() is not self._creator:
            raise RuntimeError("Console backend profile requires its creator thread")
        async with self._close_lock:
            if self._cleanup.attempted:
                if self._cleanup.error is not None:
                    raise self._cleanup.error
                return
            first_error = None
            close_clients = [
                (
                    self.server_context.close_cached_client,
                    self.server_context._cached_client,
                ),
                (self._disconnect_local_mcp, None),
            ]
            skills_provider = self.skills_scope.server_service.client_provider
            if type(skills_provider) is LegacyConfigServerClientProvider:
                close_clients.append(
                    (
                        skills_provider.close_cached_client,
                        skills_provider._cached_client,
                    )
                )
            for close, cached_client in close_clients:
                try:
                    await close()
                except BaseException as error:
                    if cached_client is not None:
                        self._cleanup.unresolved_cached_clients.append(cached_client)
                    if first_error is None:
                        first_error = error
            repository_error = _close_repositories(self.storage, self._sync_repository)
            self._cleanup.error = (
                repository_error if first_error is None else first_error
            )
            # Original client close can clear its cache before raising. A retry
            # through that empty slot cannot settle the original uncertainty.
            self._cleanup.attempted = True
            if self._cleanup.error is not None:
                raise self._cleanup.error


def open_console_backend_profile(
    *, expected_config_path: Path, expected_data_directory: Path
) -> ConsoleBackendProfile:
    """Open the real selected profile after the original App finished setup."""
    for expected in (expected_config_path, expected_data_directory):
        if not isinstance(expected, Path) or not expected.is_absolute():
            raise ValueError("Expected Console profile paths must be absolute")
    config_path = config.get_cli_config_path()
    if lexical_path(expected_config_path) != config_path:
        raise RuntimeError("Console backend config selection changed")
    configuration = config.load_settings()
    configuration_identity = config.current_config_identity()
    data_directory = config.get_user_data_dir()
    if lexical_path(expected_data_directory) != data_directory:
        raise RuntimeError("Console backend data directory selection changed")

    policy_store = RuntimeSourceStateStore(
        lexical_path(default_runtime_policy_path()),
        application_owned_directory=config.application_owned_config_directory(
            config_path
        ),
    )
    state = policy_store.load()
    if synchronize_runtime_source_state_with_app_config(state, configuration) != state:
        raise RuntimeError("Console backend runtime policy setup is not current")
    publication = _PolicyPublication(state)
    policy_context = RuntimePolicyContext(
        state, policy_store, publish=publication.publish
    )
    policy_enforcer = ServicePolicyEnforcer.from_runtime_policy_context(policy_context)

    target_store = ConfiguredServerTargetStore(
        data_directory / "mcp_server_targets.json"
    )
    credential_store: ServerCredentialStore | None = None

    def server_credentials() -> ServerCredentialStore:
        nonlocal credential_store
        if credential_store is None:
            try:
                credential_store = build_default_server_credential_store()
            except CredentialStoreUnavailable as error:
                credential_store = UnavailableServerCredentialStore(str(error))
        return credential_store

    server_context = RuntimeServerContextProvider(
        runtime_context=policy_context,
        target_store=target_store,
        credential_store_factory=server_credentials,
        app_config=configuration,
        credential_profile_id=default_server_credential_profile_id(),
    )
    chat_database = None
    prompt_database = None
    workspace_database = None
    sync_repository = None
    control_plane = None
    try:
        client_id = config.CLI_APP_CLIENT_ID
        verify_trusted_directory(data_directory, allow_shared_sticky=False)
        sync_repository = SyncStateRepository(
            data_directory / SYNC_STATE_DB_FILENAME, client_id
        )
        chat_database = open_chachanotes_db(
            config.get_chachanotes_db_path(),
            client_id=client_id,
            check_integrity=configuration.get("database", {}).get(
                "check_integrity_on_startup", False
            ),
        )
        prompt_database = open_prompts_db(
            config.get_prompts_db_path(), client_id="tldw_tui_client_v1"
        )
        workspace_database = open_workspaces_db(
            config.get_workspaces_db_path(), client_id=client_id
        )
        storage = ProfileStorage(
            chachanotes=chat_database,
            prompts=prompt_database,
            workspaces=workspace_database,
        )

        registry = LocalWorkspaceRegistryService(workspace_database)
        guard = DeferredWorkspaceToolProfileGuard()
        registry.attach_tool_profile_guard(guard)
        active = registry.get_active_workspace()
        default = registry.get_workspace(DEFAULT_WORKSPACE_ID)
        if active is None or (
            active.workspace_id == DEFAULT_WORKSPACE_ID
            and (default is None or default.archived)
        ):
            raise RuntimeError("Console backend workspace setup is not current")
        consent = ChangeReviewConsentService(registry)
        registry.attach_change_review_consent_service(consent)

        server_sync = ServerSyncService.from_server_context_provider(
            server_context,
            policy_enforcer=policy_enforcer,
            state_repository=sync_repository,
        )
        sync_scope = SyncScopeService(
            server_service=server_sync,
            policy_enforcer=policy_enforcer,
            state_repository=sync_repository,
        )
        local_conversations, citation_repository, migration = (
            build_local_citation_conversation_service(
                chat_database,
                sidecar_path=data_directory / "tldw_chatbook_chat_rag_context.json",
                defer_key=True,
            )
        )
        bind_citation_services(local_conversations, migration)
        conversation_marks = ConversationLocalMarksService(chat_database)
        conversation_scope = ChatConversationScopeService(
            local_service=local_conversations,
            server_service=ServerChatConversationService.from_server_context_provider(
                server_context, policy_enforcer=policy_enforcer
            ),
            policy_enforcer=policy_enforcer,
            sync_scope_service=sync_scope,
        )
        prompt_scope = build_prompt_scope_service(
            prompt_db=prompt_database,
            app_config=configuration,
            policy_enforcer=policy_enforcer,
            client_provider=server_context,
        )
        try:
            server_skills = ServerSkillsService.from_config(
                configuration, policy_enforcer=policy_enforcer
            )
        except ValueError:
            server_skills = ServerSkillsService(
                client=None, policy_enforcer=policy_enforcer
            )
        local_skills = LocalSkillsService(
            store_dir=default_local_skills_store_dir(data_directory),
            policy_enforcer=policy_enforcer,
        )
        skills_scope = SkillsScopeService(
            local_service=local_skills,
            server_service=server_skills,
            policy_enforcer=policy_enforcer,
        )

        def server_mcp_client(target):
            from ..tldw_api import MCPUnifiedClient

            if getattr(target, "auth_reference", None) == "legacy:tldw_api":
                client = build_runtime_api_client(
                    app_config=configuration,
                    endpoint_url=target.base_url,
                    auth_method=target.auth_mode,
                )
            else:
                client = build_runtime_api_client(
                    endpoint_url=target.base_url,
                    auth_token=target.auth_reference,
                    auth_method=target.auth_mode,
                )
            return MCPUnifiedClient(client)

        local_mcp = LocalMCPControlService(
            store=LocalMCPStore(data_directory / "local_mcp_store.json"),
            policy_enforcer=policy_enforcer,
            storage_provider=lambda: storage,
        )
        control_plane = UnifiedMCPControlPlaneService(
            target_store=target_store,
            context_store=UnifiedMCPContextStore(
                data_directory / "unified_mcp_context.json"
            ),
            local_service=local_mcp,
            server_service=ServerUnifiedMCPService(
                client_factory=server_mcp_client,
                policy_enforcer=policy_enforcer,
                target_store=target_store,
            ),
        )
        if control_plane.permission_store is None:
            raise RuntimeError("Console backend permission store is unavailable")
        if (
            config.get_cli_config_path() != config_path
            or config.get_user_data_dir() != data_directory
            or config.current_config_identity() != configuration_identity
        ):
            raise RuntimeError("Console backend profile changed during construction")
        return ConsoleBackendProfile(
            configuration=configuration,
            configuration_identity=configuration_identity,
            config_path=config_path,
            data_directory=data_directory,
            storage=storage,
            policy_context=policy_context,
            policy_enforcer=policy_enforcer,
            server_context=server_context,
            target_store=target_store,
            workspace_registry=registry,
            consent_service=consent,
            conversation_scope_service=conversation_scope,
            conversation_marks=conversation_marks,
            citation_repository=citation_repository,
            prompt_scope=prompt_scope,
            local_skills=local_skills,
            skills_scope=skills_scope,
            control_plane=control_plane,
            tool_profile_guard=guard,
            sync_scope=sync_scope,
            _sync_repository=sync_repository,
            _policy_publication=publication,
        )
    except BaseException as error:
        # The original server and MCP client factories are still lazy here.
        storage = ProfileStorage(
            chachanotes=chat_database,
            prompts=prompt_database,
            workspaces=workspace_database,
        )
        cleanup_error = _close_repositories(storage, sync_repository)
        if cleanup_error is not None:
            raise ConsoleBackendProfileOpenFailure(
                acquisition_error=error,
                cleanup_error=cleanup_error,
                storage=storage,
                sync_repository=sync_repository,
                server_context=server_context,
                control_plane=control_plane,
            ) from error
        raise
