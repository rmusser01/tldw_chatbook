"""Compose the Research Workspace's Local and Server authority adapters."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from .Research_Workspace import (
    LocalResearchWorkspaceAdapter,
    ResearchPresentationOverlayStore,
    ResearchWorkspaceController,
    ServerResearchWorkspaceAdapter,
    WorkspaceDataSource,
)


def _local_chat(app: Any, registry: Any, overlay: ResearchPresentationOverlayStore):
    conversations = getattr(app, "local_chat_conversation_service", None)
    database = getattr(app, "chachanotes_db", None)
    runtime = getattr(app, "console_runtime", None)
    if conversations is None or database is None or runtime is None:
        return None
    from .Chat.chat_persistence_service import ChatPersistenceService
    from .Research_Workspace.local_chat import LocalResearchChat, search_local

    return LocalResearchChat(
        ChatPersistenceService(
            database,
            workspace_registry=registry,
            citation_repository=getattr(app, "citation_trace_repository", None),
        ),
        conversations,
        runtime.ensure_provider_gateway,
        partial(search_local, app),
        overlay,
    )


def create_research_workspace_screen(app: Any, screen_class: type, user_data_dir: Path):
    """Construct a fresh Research screen with the currently active services."""
    ports = {}
    local_service = getattr(app, "workspace_registry_service", None)
    media_scope_service = getattr(app, "media_reading_scope_service", None)
    operation_store = getattr(app, "research_source_operation_store", None)
    association_scheduler = getattr(app, "research_source_association_scheduler", None)
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
            notes_scope_service=getattr(app, "notes_scope_service", None),
            notes_user_id=getattr(app, "notes_user_id", ""),
            chat=chat,
            studio=LocalStudioAdapter(
                local_service,
                getattr(app, "local_chatbook_service", None),
                getattr(app, "local_study_service", None),
                getattr(app, "local_quiz_service", None),
            )
            if chat
            else None,
        )
    server_service = getattr(app, "server_notes_workspace_service", None)
    server_context_provider = getattr(app, "server_context_provider", None)
    if server_service is not None and server_context_provider is not None:
        ports[WorkspaceDataSource.SERVER] = ServerResearchWorkspaceAdapter(
            server_service,
            server_context_provider,
            media_scope_service=media_scope_service,
            operation_store=operation_store,
            association_scheduler=association_scheduler,
            chat_overlay=overlay_store,
            chat_policy_enforcer=getattr(app, "service_policy_enforcer", None),
        )
    copy_workspace = None
    if WorkspaceDataSource.LOCAL in ports and WorkspaceDataSource.SERVER in ports:
        from .Research_Workspace.copy_workspace import ResearchWorkspaceCopy

        copy_workspace = ResearchWorkspaceCopy(
            ports[WorkspaceDataSource.LOCAL],
            ports[WorkspaceDataSource.SERVER],
            local_service,
            getattr(app, "local_chatbook_service", None),
            policy_enforcer=getattr(app, "service_policy_enforcer", None),
        )
    return screen_class(
        app,
        controller=ResearchWorkspaceController(ports),
        overlay_store=overlay_store,
        operation_store=operation_store,
        association_scheduler=association_scheduler,
        paste_staging_store=getattr(app, "research_paste_staging_store", None),
        copy_workspace=copy_workspace,
    )
