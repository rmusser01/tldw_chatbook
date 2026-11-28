"""Compose the Research Workspace's Local and Server authority adapters."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from .Research_Workspace import (
    LocalResearchWorkspaceAdapter,
    ResearchPresentationOverlayStore,
    ResearchWorkspaceController,
    ServerResearchWorkspaceAdapter,
    WorkspaceDataSource,
)

from .app_services import (
    read_chat_database,
    read_console_runtime,
    read_server_context_provider,
    read_workspace_registry,
)

if TYPE_CHECKING:
    from .app import TldwCli
    from .Research_Workspace.local_chat import LocalResearchChat
    from .UI.Screens.research_workspace_screen import ResearchWorkspaceScreen
    from .Workspaces.registry_service import LocalWorkspaceRegistryService


def _optional_service[T](read: Callable[[], T]) -> T | None:
    """Read a typed app attribute that may be unavailable during startup."""
    try:
        return read()
    except AttributeError:
        return None


def _local_chat(
    app: TldwCli,
    registry: LocalWorkspaceRegistryService,
    overlay: ResearchPresentationOverlayStore,
) -> LocalResearchChat | None:
    conversations = _optional_service(lambda: app.local_chat_conversation_service)
    database = read_chat_database(app)
    runtime = read_console_runtime(app)
    if conversations is None or database is None or runtime is None:
        return None
    from .Chat.chat_persistence_service import ChatPersistenceService
    from .Research_Workspace.local_chat import LocalResearchChat, search_local

    return LocalResearchChat(
        ChatPersistenceService(
            database,
            workspace_registry=registry,
            citation_repository=_optional_service(lambda: app.citation_trace_repository),
        ),
        conversations,
        runtime.ensure_provider_gateway,
        partial(search_local, app),
        overlay,
    )


def create_research_workspace_screen(
    app: TldwCli, screen_class: type[ResearchWorkspaceScreen], user_data_dir: Path
) -> ResearchWorkspaceScreen:
    """Construct a fresh Research screen with the currently active services."""
    ports = {}
    local_service = read_workspace_registry(app)
    media_scope_service = _optional_service(lambda: app.media_reading_scope_service)
    operation_store = _optional_service(lambda: app.research_source_operation_store)
    association_scheduler = _optional_service(lambda: app.research_source_association_scheduler)
    chatbook_service = _optional_service(lambda: app.local_chatbook_service)
    policy_enforcer = _optional_service(lambda: app.service_policy_enforcer)
    overlay_store = ResearchPresentationOverlayStore(
        user_data_dir / "research_workspace_overlay.json"
    )
    if local_service is not None:
        from .Research_Workspace.local_studio import LocalStudioAdapter

        chat = _local_chat(app, local_service, overlay_store)
        ports[WorkspaceDataSource.LOCAL] = LocalResearchWorkspaceAdapter(
            local_service,
            media_scope_service=media_scope_service,
            operation_store=operation_store,
            association_scheduler=association_scheduler,
            notes_scope_service=_optional_service(lambda: app.notes_scope_service),
            notes_user_id=_optional_service(lambda: app.notes_user_id) or "",
            chat=chat,
            studio=LocalStudioAdapter(
                local_service,
                chatbook_service,
                _optional_service(lambda: app.local_study_service),
                _optional_service(lambda: app.local_quiz_service),
                audio=_optional_service(lambda: app.audio_services_scope_service),
            )
            if chat
            else None,
        )
    server_service = _optional_service(lambda: app.server_notes_workspace_service)
    server_context_provider = read_server_context_provider(app)
    if server_service is not None and server_context_provider is not None:
        ports[WorkspaceDataSource.SERVER] = ServerResearchWorkspaceAdapter(
            server_service,
            server_context_provider,
            media_scope_service=media_scope_service,
            operation_store=operation_store,
            association_scheduler=association_scheduler,
            chat_overlay=overlay_store,
            chat_policy_enforcer=policy_enforcer,
        )
    copy_workspace = None
    if WorkspaceDataSource.LOCAL in ports and WorkspaceDataSource.SERVER in ports:
        from .Research_Workspace.copy_workspace import ResearchWorkspaceCopy

        copy_workspace = ResearchWorkspaceCopy(
            ports[WorkspaceDataSource.LOCAL],
            ports[WorkspaceDataSource.SERVER],
            local_service,
            chatbook_service,
            policy_enforcer=policy_enforcer,
        )
    return screen_class(
        app,
        controller=ResearchWorkspaceController(ports),
        overlay_store=overlay_store,
        operation_store=operation_store,
        association_scheduler=association_scheduler,
        paste_staging_store=_optional_service(lambda: app.research_paste_staging_store),
        copy_workspace=copy_workspace,
    )
