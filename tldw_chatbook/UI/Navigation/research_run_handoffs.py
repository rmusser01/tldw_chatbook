"""Qualify native Runs navigation against the live Server profile."""

from collections.abc import Mapping

from ...Research_Workspace.contracts import QualifiedWorkspaceRef, WorkspaceDataSource
from ...Research_Workspace.deep_research_bridge import DeepResearchBridge
from ...runtime_policy.server_context import ServerContextError
from ...runtime_policy.server_event_scope import event_principal_id_from_active_context


def active_server_identity(app) -> tuple[str, str] | None:
    provider = getattr(app, "server_context_provider", None)
    if provider is None:
        return None
    try:
        context = provider.get_active_context()
        principal = event_principal_id_from_active_context(context)
        if principal:
            return context.active_server_id, principal
    except (ServerContextError, TypeError, ValueError):
        pass
    return None


def active_server_workspace(app, workspace_id: str) -> QualifiedWorkspaceRef | None:
    if identity := active_server_identity(app):
        try:
            return QualifiedWorkspaceRef("server", workspace_id, *identity)
        except (TypeError, ValueError):
            pass
    return None


def research_run_target_current(app, ref: QualifiedWorkspaceRef) -> bool:
    return (
        ref.data_source is WorkspaceDataSource.LOCAL
        or active_server_workspace(app, ref.workspace_id) == ref
    )


def workspace_target_from_run(app, run):
    origin = None
    if isinstance(run, Mapping) and run.get("source") == "server":
        launch = run.get("workspace_launch")
        if isinstance(launch, Mapping) and isinstance(launch.get("workspace_id"), str):
            origin = active_server_workspace(app, launch["workspace_id"])
    return DeepResearchBridge.target_from_run(run, server_origin=origin)
