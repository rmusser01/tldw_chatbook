"""Pure display state for workspace-aware Console surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from loguru import logger

from tldw_chatbook.Sync_Interop.sync_promotion_state import build_sync_promotion_state
from tldw_chatbook.Sync_Interop.sync_readiness import (
    DEFAULT_SYNC_ELIGIBILITY_REGISTRY,
    build_sync_readiness_report,
)

from .models import (
    RuntimeBindingStatus,
    WorkspaceAuthority,
    WorkspaceEligibility,
    WorkspaceMembership,
    WorkspaceOperation,
    WorkspaceRecord,
    WorkspaceRuntimeBinding,
    WorkspaceSyncStatus,
)
from .eligibility import evaluate_workspace_eligibility

logger = logger.bind(module="WorkspaceDisplayState")

LIBRARY_WORKSPACE_VISIBILITY_COPY = (
    "Browse/search: all Library and Notes items remain visible."
)
LIBRARY_WORKSPACE_COLLECTIONS_COPY = (
    "Collections: browse and organize; staging is read-only"
)
LIBRARY_WORKSPACE_IMPORT_EXPORT_COPY = "Import/Export: copy or reference sources"


@dataclass(frozen=True)
class ConsoleWorkspaceConversationRow:
    """One conversation row shown inside the Console workspace context tray."""

    conversation_id: str
    title: str
    status: str = ""
    selected: bool = False


@dataclass(frozen=True)
class ConsoleWorkspaceContextState:
    """Renderable Console workspace context snapshot."""

    heading: str
    workspace_label: str
    authority_label: str
    sync_label: str
    runtime_label: str
    conversation_rows: tuple[ConsoleWorkspaceConversationRow, ...]
    conversation_empty_copy: str
    change_workspace_enabled: bool
    change_workspace_recovery: str
    new_conversation_enabled: bool
    new_conversation_recovery: str
    recovery_copy: str


@dataclass(frozen=True)
class LibraryWorkspaceSourceRow:
    """One visible Library source row annotated with workspace-context authority."""

    item_type: str
    item_id: str
    title: str
    workspace_ids: tuple[str, ...]
    workspace_label: str
    visible: bool
    active_context_eligible: bool
    authority_label: str
    context_label: str
    recovery_copy: str = ""


@dataclass(frozen=True)
class LibraryWorkspaceDepthState:
    """Renderable Library workspace-depth snapshot."""

    heading: str
    workspace_label: str
    workspace_name: str
    visibility_label: str
    handoff_label: str
    context_handoff_enabled: bool
    context_handoff_tooltip: str
    source_authority_label: str
    collections_membership_label: str
    import_export_label: str
    source_rows: tuple[LibraryWorkspaceSourceRow, ...]
    recovery_copy: str = ""


def build_console_workspace_state(
    *,
    registry_service: Any,
    current_conversation: str | None,
    conversations: Iterable[ConsoleWorkspaceConversationRow] | None = None,
) -> ConsoleWorkspaceContextState:
    """Build Console workspace display state from the local registry seam.

    Args:
        registry_service: Local workspace registry service used to read the
            active workspace, runtime bindings, and workspace memberships. A
            missing or failing service produces a safe degraded state.
        current_conversation: Active conversation id used to mark the selected
            conversation row, when it belongs to the active workspace.
        conversations: Optional prebuilt conversation rows. When omitted, rows
            are derived from `conversation` workspace memberships.

    Returns:
        Renderable Console workspace context state.
    """

    if registry_service is None:
        return ConsoleWorkspaceContextState(
            heading="Convos & Workspaces",
            workspace_label="No workspace selected",
            authority_label="Authority: unavailable",
            sync_label="Sync: unavailable",
            runtime_label="Runtime: unavailable",
            conversation_rows=(),
            conversation_empty_copy="Workspace conversations are unavailable.",
            change_workspace_enabled=False,
            change_workspace_recovery="Workspace service not ready.",
            new_conversation_enabled=False,
            new_conversation_recovery="Workspace service not ready.",
            recovery_copy="Workspace service not ready. Restart or open Settings if this persists.",
        )

    try:
        active_workspace = registry_service.get_active_workspace()
    except Exception:
        logger.warning(
            "Failed to read active workspace for Console context rail",
            exc_info=True,
        )
        return ConsoleWorkspaceContextState(
            heading="Convos & Workspaces",
            workspace_label="No workspace selected",
            authority_label="Authority: unavailable",
            sync_label="Sync: unavailable",
            runtime_label="Runtime: unavailable",
            conversation_rows=(),
            conversation_empty_copy="Workspace conversations are unavailable.",
            change_workspace_enabled=False,
            change_workspace_recovery="Workspace registry could not be read.",
            new_conversation_enabled=False,
            new_conversation_recovery="Workspace registry could not be read.",
            recovery_copy="Workspace registry could not be read. Check local workspace storage.",
        )

    if active_workspace is None:
        return ConsoleWorkspaceContextState(
            heading="Convos & Workspaces",
            workspace_label="Workspace: Local Default",
            authority_label="Authority: local registry ready",
            sync_label="Sync: not configured",
            runtime_label="Runtime: none",
            conversation_rows=(),
            conversation_empty_copy="No active workspace conversations.",
            change_workspace_enabled=False,
            change_workspace_recovery="Workspace switching is read-only in this slice.",
            new_conversation_enabled=False,
            new_conversation_recovery="Conversation creation is read-only until workspace selection is wired.",
            recovery_copy="Workspace switching: locked",
        )

    runtime_bindings = _safe_runtime_bindings(registry_service, active_workspace)
    source_rows = (
        tuple(conversations)
        if conversations is not None
        else _conversation_rows_from_memberships(registry_service, active_workspace)
    )
    rows = tuple(_select_conversation(row, current_conversation) for row in source_rows)
    return ConsoleWorkspaceContextState(
        heading="Convos & Workspaces",
        workspace_label=f"Workspace: {active_workspace.name}",
        authority_label=f"Authority: {active_workspace.authority.value}",
        sync_label=_workspace_sync_label(active_workspace),
        runtime_label=_runtime_label(runtime_bindings),
        conversation_rows=rows,
        conversation_empty_copy="No conversations in this workspace yet.",
        change_workspace_enabled=False,
        change_workspace_recovery="Workspace switching is not wired yet.",
        new_conversation_enabled=False,
        new_conversation_recovery="Workspace conversation creation lands in a later slice.",
        recovery_copy="Workspace switching is read-only in this slice.",
    )


def build_library_workspace_depth_state(
    *,
    registry_service: Any,
    source_records: Mapping[str, Iterable[Mapping[str, Any]]],
) -> LibraryWorkspaceDepthState:
    """Build Library workspace depth state without filtering visible records.

    The Library remains a global browse/search/edit surface. Workspace context
    only changes whether a source can be staged into active Console/RAG/agent
    context.

    Args:
        registry_service: Workspace registry service used to read the active
            workspace, item memberships, and workspace labels. A missing or
            failing service produces a safe degraded state.
        source_records: Visible Library records grouped by source type. Records
            without a stable identifier remain globally visible elsewhere, but
            are omitted from workspace staging calculations.

    Returns:
        Renderable Library workspace-depth snapshot for the Library workbench.
    """

    if registry_service is None:
        rows = _library_source_rows_without_workspace(source_records)
        return LibraryWorkspaceDepthState(
            heading="Workspaces",
            workspace_label="Workspace: unavailable",
            workspace_name="unavailable",
            visibility_label=LIBRARY_WORKSPACE_VISIBILITY_COPY,
            handoff_label="Console/RAG handoff: blocked until workspace registry is ready",
            context_handoff_enabled=False,
            context_handoff_tooltip="Workspace registry is unavailable.",
            source_authority_label="Source authority: unavailable",
            collections_membership_label=LIBRARY_WORKSPACE_COLLECTIONS_COPY,
            import_export_label=LIBRARY_WORKSPACE_IMPORT_EXPORT_COPY,
            source_rows=rows,
            recovery_copy="Workspace registry is unavailable. Retry after local workspace storage is ready.",
        )

    try:
        active_workspace = registry_service.get_active_workspace()
    except Exception:
        logger.warning(
            "Failed to read active workspace for Library workspace depth",
            exc_info=True,
        )
        rows = _library_source_rows_without_workspace(source_records)
        return LibraryWorkspaceDepthState(
            heading="Workspaces",
            workspace_label="Workspace: unavailable",
            workspace_name="unavailable",
            visibility_label=LIBRARY_WORKSPACE_VISIBILITY_COPY,
            handoff_label="Console/RAG handoff: blocked until workspace registry can be read",
            context_handoff_enabled=False,
            context_handoff_tooltip="Workspace registry could not be read.",
            source_authority_label="Source authority: unavailable",
            collections_membership_label=LIBRARY_WORKSPACE_COLLECTIONS_COPY,
            import_export_label=LIBRARY_WORKSPACE_IMPORT_EXPORT_COPY,
            source_rows=rows,
            recovery_copy="Workspace registry could not be read. Check local workspace storage.",
        )

    if active_workspace is None:
        rows = _library_source_rows_without_workspace(source_records)
        return LibraryWorkspaceDepthState(
            heading="Workspaces",
            workspace_label="Workspace: Local Default",
            workspace_name="Local Default",
            visibility_label=LIBRARY_WORKSPACE_VISIBILITY_COPY,
            handoff_label=(
                "Console/RAG handoff: local default source snapshot"
                if rows
                else "Console/RAG handoff: unavailable until sources exist"
            ),
            context_handoff_enabled=bool(rows),
            context_handoff_tooltip=(
                "Stage local Library source context in Console."
                if rows
                else "Add Library sources before staging context in Console."
            ),
            source_authority_label="Source authority: local default",
            collections_membership_label=LIBRARY_WORKSPACE_COLLECTIONS_COPY,
            import_export_label=LIBRARY_WORKSPACE_IMPORT_EXPORT_COPY,
            source_rows=rows,
            recovery_copy="Select a workspace to preview workspace-specific context eligibility.",
        )

    workspace_name_cache: dict[str, str] = {}
    rows: list[LibraryWorkspaceSourceRow] = []
    for source_type, records in source_records.items():
        for record in records:
            row = _library_workspace_source_row(
                registry_service,
                active_workspace,
                source_type,
                record,
                workspace_name_cache,
            )
            if row is not None:
                rows.append(row)
    eligible_count = sum(row.active_context_eligible for row in rows)
    blocked_count = len(rows) - eligible_count
    handoff_enabled = bool(rows) and blocked_count == 0
    tooltip = (
        "Stage active-workspace Library source context in Console."
        if handoff_enabled
        else (
            "Copy or link blocked Library sources into the active workspace before "
            "using them in Console."
        )
    )
    return LibraryWorkspaceDepthState(
        heading="Workspaces",
        workspace_label=f"Workspace: {active_workspace.name}",
        workspace_name=active_workspace.name,
        visibility_label=LIBRARY_WORKSPACE_VISIBILITY_COPY,
        handoff_label=(
            f"Console/RAG handoff: {eligible_count} eligible"
            + (f", {blocked_count} blocked" if blocked_count else "")
        ),
        context_handoff_enabled=handoff_enabled,
        context_handoff_tooltip=tooltip,
        source_authority_label=f"Source authority: active workspace {active_workspace.workspace_id}",
        collections_membership_label=LIBRARY_WORKSPACE_COLLECTIONS_COPY,
        import_export_label=LIBRARY_WORKSPACE_IMPORT_EXPORT_COPY,
        source_rows=tuple(rows),
        recovery_copy=(
            "Workspace switching changes context eligibility, not Library visibility."
        ),
    )


def _safe_runtime_bindings(
    registry_service: Any,
    active_workspace: WorkspaceRecord,
) -> tuple[WorkspaceRuntimeBinding, ...]:
    try:
        runtime_bindings = registry_service.list_runtime_bindings(active_workspace.workspace_id)
        if not runtime_bindings:
            return ()
        return tuple(runtime_bindings)
    except Exception:
        logger.warning(
            "Failed to read workspace runtime bindings for Console context rail",
            exc_info=True,
        )
        return ()


def _conversation_rows_from_memberships(
    registry_service: Any,
    active_workspace: WorkspaceRecord,
) -> tuple[ConsoleWorkspaceConversationRow, ...]:
    try:
        memberships = registry_service.list_workspace_memberships(active_workspace.workspace_id)
    except Exception:
        logger.warning(
            "Failed to read workspace memberships for Console context rail",
            exc_info=True,
        )
        return ()
    if not memberships:
        return ()
    rows: list[ConsoleWorkspaceConversationRow] = []
    for membership in memberships:
        if membership.item_type != "conversation":
            continue
        rows.append(
            ConsoleWorkspaceConversationRow(
                conversation_id=membership.item_id,
                title=membership.title or membership.item_id,
                status=membership.role,
            )
        )
    return tuple(rows)


def _library_source_rows_without_workspace(
    source_records: Mapping[str, Iterable[Mapping[str, Any]]],
) -> tuple[LibraryWorkspaceSourceRow, ...]:
    rows: list[LibraryWorkspaceSourceRow] = []
    for source_type, records in source_records.items():
        for record in records:
            item_type = _library_workspace_item_type(source_type)
            item_id = _library_source_record_id(record)
            if not item_id:
                continue
            rows.append(
                LibraryWorkspaceSourceRow(
                    item_type=item_type,
                    item_id=item_id,
                    title=_library_source_record_title(record, item_id),
                    workspace_ids=(),
                    workspace_label="Unscoped",
                    visible=True,
                    active_context_eligible=True,
                    authority_label="Workspace: unscoped",
                    context_label="Console/RAG: local default",
                )
            )
    return tuple(rows)


def _library_workspace_source_row(
    registry_service: Any,
    active_workspace: WorkspaceRecord,
    source_type: str,
    record: Mapping[str, Any],
    workspace_name_cache: dict[str, str],
) -> LibraryWorkspaceSourceRow | None:
    item_type = _library_workspace_item_type(source_type)
    item_id = _library_source_record_id(record)
    if not item_id:
        return None
    title = _library_source_record_title(record, item_id)
    memberships = _safe_item_memberships(registry_service, item_type, item_id)
    workspace_ids = tuple(membership.workspace_id for membership in memberships)
    decision = evaluate_workspace_eligibility(
        active_workspace_id=active_workspace.workspace_id,
        item_workspace_ids=workspace_ids,
        item_type=item_type,
        operation=WorkspaceOperation.STAGE_IN_CONSOLE,
    )
    return LibraryWorkspaceSourceRow(
        item_type=item_type,
        item_id=item_id,
        title=title,
        workspace_ids=workspace_ids,
        workspace_label=_library_source_workspace_label(
            registry_service,
            workspace_ids,
            workspace_name_cache,
        ),
        visible=decision.visible,
        active_context_eligible=decision.active_context_eligible,
        authority_label=_library_source_authority_label(workspace_ids),
        context_label=(
            "Console/RAG: eligible"
            if decision.active_context_eligible
            else "Console/RAG: blocked"
        ),
        recovery_copy=decision.recovery_copy,
    )


def _safe_item_memberships(
    registry_service: Any,
    item_type: str,
    item_id: str | None,
) -> tuple[WorkspaceMembership, ...]:
    if not item_id:
        return ()
    get_item_memberships = getattr(registry_service, "get_item_memberships", None)
    if not callable(get_item_memberships):
        return ()
    try:
        memberships = get_item_memberships(item_type, item_id)
    except Exception:
        logger.warning(
            "Failed to read Library source workspace memberships",
            item_type=item_type,
            item_id=item_id,
            exc_info=True,
        )
        return ()
    return tuple(memberships or ())


def _library_workspace_item_type(source_type: str) -> str:
    return {
        "notes": "note",
        "media": "media",
        "conversations": "conversation",
    }.get(source_type, source_type.rstrip("s") or "source")


def _library_source_record_id(record: Mapping[str, Any]) -> str | None:
    for key in (
        "id",
        "uuid",
        "record_id",
        "backing_id",
        "source_id",
        "item_id",
        "media_id",
        "note_id",
        "conversation_id",
        "backing_media_id",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return None


def _library_source_record_title(record: Mapping[str, Any], fallback: str) -> str:
    for key in ("title", "name", "label"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _library_source_authority_label(workspace_ids: tuple[str, ...]) -> str:
    if not workspace_ids:
        return "Workspace: unassigned"
    return f"Workspace: {', '.join(workspace_ids)}"


def _library_source_workspace_label(
    registry_service: Any,
    workspace_ids: tuple[str, ...],
    workspace_name_cache: dict[str, str],
) -> str:
    if not workspace_ids:
        return "Unassigned"
    get_workspace = getattr(registry_service, "get_workspace", None)
    if not callable(get_workspace):
        return ", ".join(workspace_ids)
    labels: list[str] = []
    for workspace_id in workspace_ids:
        cached_label = workspace_name_cache.get(workspace_id)
        if cached_label is not None:
            labels.append(cached_label)
            continue
        try:
            workspace = get_workspace(workspace_id)
        except Exception:
            workspace = None
        workspace_name = getattr(workspace, "name", None)
        label = workspace_name if workspace_name else workspace_id
        workspace_name_cache[workspace_id] = label
        labels.append(label)
    return ", ".join(labels)


def _select_conversation(
    row: ConsoleWorkspaceConversationRow,
    current_conversation: str | None,
) -> ConsoleWorkspaceConversationRow:
    return ConsoleWorkspaceConversationRow(
        conversation_id=row.conversation_id,
        title=row.title,
        status=row.status,
        selected=bool(current_conversation) and row.conversation_id == current_conversation,
    )


def _runtime_label(bindings: tuple[WorkspaceRuntimeBinding, ...]) -> str:
    if not bindings:
        return "Runtime: none"
    ready_count = sum(binding.status == RuntimeBindingStatus.READY for binding in bindings)
    return (
        f"Runtime: {len(bindings)} {_plural('binding', len(bindings))}, "
        f"{ready_count} ready"
    )


def _workspace_sync_label(active_workspace: WorkspaceRecord) -> str:
    sync_status = active_workspace.sync_status
    if sync_status == WorkspaceSyncStatus.NOT_CONFIGURED:
        return "Sync: not configured"
    if sync_status == WorkspaceSyncStatus.SYNCING:
        return "Sync: syncing"
    if sync_status == WorkspaceSyncStatus.BLOCKED:
        return "Sync: blocked"
    if sync_status == WorkspaceSyncStatus.CONFLICT:
        return "Sync: conflict review required"

    readiness = build_sync_readiness_report(
        domain="workspaces",
        server_profile_id=None,
        workspace_id=active_workspace.workspace_id,
        registry=DEFAULT_SYNC_ELIGIBILITY_REGISTRY,
    )
    source_authority = (
        "server"
        if active_workspace.authority
        in {
            WorkspaceAuthority.SERVER_BACKED,
            WorkspaceAuthority.SYNCING_FROM_SERVER,
            WorkspaceAuthority.REMOTE_ONLY,
        }
        else "local"
    )
    return build_sync_promotion_state(
        domain="workspaces",
        surface_label="Workspaces",
        readiness=readiness,
        source_authority=source_authority,
        workspace_id=active_workspace.workspace_id,
    ).sync_label


def _plural(label: str, count: int) -> str:
    return label if count == 1 else f"{label}s"
