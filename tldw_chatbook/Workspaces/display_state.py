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
    DEFAULT_WORKSPACE_ID,
    RuntimeBindingStatus,
    WorkspaceAuthority,
    WorkspaceEligibility,
    WorkspaceMembership,
    WorkspaceOperation,
    WorkspaceRecord,
    WorkspaceRuntimeBinding,
    WorkspaceSyncStatus,
    WorkspaceTransferPolicy,
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
class ConsoleWorkspaceServerAdapterState:
    """Server adapter availability for future workspace hydration paths.

    Attributes:
        available: Whether a server workspace adapter is available for the
            active workspace context.
        detail: Human-readable readiness or recovery detail for the adapter.
    """

    available: bool
    detail: str = ""


@dataclass(frozen=True)
class ConsoleWorkspaceHandoffRow:
    """One source/conversation/artifact handoff eligibility row.

    Attributes:
        item_type: Type of item being considered for workspace handoff.
        item_id: Stable item identifier from the source registry.
        title: Display title shown in the Console workspace context.
        transfer_policy: Copy/reference/metadata/local-only handoff policy.
        handoff_label: User-facing policy label.
        portable: Whether the item can be staged into the active workspace.
        detail: Recovery or audit detail explaining the handoff state.
    """

    item_type: str
    item_id: str
    title: str
    transfer_policy: WorkspaceTransferPolicy
    handoff_label: str
    portable: bool
    detail: str


@dataclass(frozen=True)
class ConsoleWorkspaceACPHandoffState:
    """Visible readiness state for future ACP task/run package handoff.

    Attributes:
        status: Machine-readable readiness state such as unavailable, ready,
            blocked, or failed.
        detail: User-facing readiness or recovery copy.
        audit_detail: Audit summary for the last ACP package handoff attempt.
    """

    status: str = "unavailable"
    detail: str = "ACP task/run package handoff is not wired."
    audit_detail: str = "Audit: no ACP package was sent."


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
    server_readiness_label: str = "Server: local fallback"
    server_readiness_detail: str = (
        "Local registry is authoritative. No background sync is running."
    )
    handoff_rows: tuple[ConsoleWorkspaceHandoffRow, ...] = ()
    acp_handoff_label: str = "ACP task/run: unavailable"
    acp_handoff_detail: str = "ACP task/run package handoff is not wired."
    acp_handoff_audit: str = "Audit: no ACP package was sent."


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
    server_adapter_state: ConsoleWorkspaceServerAdapterState | None = None,
    acp_handoff_state: ConsoleWorkspaceACPHandoffState | None = None,
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
        server_adapter_state: Optional server adapter readiness snapshot used
            to render unavailable, ready, conflict, or remote-only workspace
            hydration states without starting sync.
        acp_handoff_state: Optional ACP task/run package handoff snapshot used
            to render unavailable, ready, failed, blocked, and audit states.

    Returns:
        Renderable Console workspace context state.
    """

    if registry_service is None:
        acp_state = _acp_handoff_state(acp_handoff_state)
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
            server_readiness_label="Server: unavailable",
            server_readiness_detail=(
                "Workspace registry service is not ready. Server-backed hydration "
                "remains behind the workspace adapter boundary. No background sync "
                "is running."
            ),
            acp_handoff_label=acp_state[0],
            acp_handoff_detail=acp_state[1],
            acp_handoff_audit=acp_state[2],
        )

    try:
        active_workspace = registry_service.get_active_workspace()
    except Exception:
        logger.warning(
            "Failed to read active workspace for Console context rail",
            exc_info=True,
        )
        acp_state = _acp_handoff_state(acp_handoff_state)
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
            server_readiness_label="Server: unavailable",
            server_readiness_detail=(
                "Workspace registry could not be read. Server-backed hydration "
                "remains behind the workspace adapter boundary. No background sync "
                "is running."
            ),
            acp_handoff_label=acp_state[0],
            acp_handoff_detail=acp_state[1],
            acp_handoff_audit=acp_state[2],
        )

    if active_workspace is None:
        workspaces = _safe_workspaces(registry_service)
        can_switch = bool(workspaces)
        acp_state = _acp_handoff_state(acp_handoff_state)
        return ConsoleWorkspaceContextState(
            heading="Convos & Workspaces",
            workspace_label="Workspace: Local Default",
            authority_label="Authority: local registry ready",
            sync_label="Sync: not configured",
            runtime_label="Runtime: none",
            conversation_rows=(),
            conversation_empty_copy="No active workspace conversations.",
            change_workspace_enabled=can_switch,
            change_workspace_recovery=(
                "" if can_switch else "Create a workspace in Library > Workspaces before switching."
            ),
            new_conversation_enabled=False,
            new_conversation_recovery="Conversation creation is read-only until workspace selection is wired.",
            recovery_copy=(
                "" if can_switch else "Workspace switching: locked"
            ),
            server_readiness_label="Server: local fallback",
            server_readiness_detail=(
                "Local registry fallback is active. No background sync is running."
            ),
            acp_handoff_label=acp_state[0],
            acp_handoff_detail=acp_state[1],
            acp_handoff_audit=acp_state[2],
        )

    runtime_bindings = _safe_runtime_bindings(registry_service, active_workspace)
    workspaces = _safe_workspaces(registry_service)
    can_switch = len(workspaces) > 1
    is_default_workspace = active_workspace.workspace_id == DEFAULT_WORKSPACE_ID
    source_rows = (
        tuple(conversations)
        if conversations is not None
        else _conversation_rows_from_memberships(registry_service, active_workspace)
    )
    rows = tuple(_select_conversation(row, current_conversation) for row in source_rows)
    server_label, server_detail = _server_readiness(
        active_workspace,
        server_adapter_state,
    )
    acp_state = _acp_handoff_state(acp_handoff_state)
    return ConsoleWorkspaceContextState(
        heading="Convos & Workspaces",
        workspace_label=f"Workspace: {active_workspace.name}",
        authority_label=f"Authority: {active_workspace.authority.value}",
        sync_label=_workspace_sync_label(active_workspace),
        runtime_label=(
            "Runtime: none, file tools disabled"
            if is_default_workspace and not runtime_bindings
            else _runtime_label(runtime_bindings)
        ),
        conversation_rows=rows,
        conversation_empty_copy="No conversations in this workspace yet.",
        change_workspace_enabled=can_switch,
        change_workspace_recovery=(
            "" if can_switch else "Add another workspace before switching."
        ),
        new_conversation_enabled=False,
        new_conversation_recovery="Workspace conversation creation lands in a later slice.",
        recovery_copy=(
            ""
            if can_switch or is_default_workspace
            else "Workspace switching: only one workspace available."
        ),
        server_readiness_label=server_label,
        server_readiness_detail=server_detail,
        handoff_rows=_handoff_rows_from_memberships(registry_service, active_workspace),
        acp_handoff_label=acp_state[0],
        acp_handoff_detail=acp_state[1],
        acp_handoff_audit=acp_state[2],
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
    workspace_display_name = (
        "Local Default"
        if active_workspace.workspace_id == DEFAULT_WORKSPACE_ID
        else active_workspace.name
    )
    if not rows:
        return LibraryWorkspaceDepthState(
            heading="Workspaces",
            workspace_label=f"Workspace: {workspace_display_name}",
            workspace_name=workspace_display_name,
            visibility_label=LIBRARY_WORKSPACE_VISIBILITY_COPY,
            handoff_label="Console/RAG handoff: unavailable until sources exist",
            context_handoff_enabled=False,
            context_handoff_tooltip="Add Library sources before staging context in Console.",
            source_authority_label=f"Source authority: active workspace {active_workspace.workspace_id}",
            collections_membership_label=LIBRARY_WORKSPACE_COLLECTIONS_COPY,
            import_export_label=LIBRARY_WORKSPACE_IMPORT_EXPORT_COPY,
            source_rows=(),
            recovery_copy=(
                "Workspace switching changes context eligibility, not Library visibility."
            ),
        )
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
        workspace_label=f"Workspace: {workspace_display_name}",
        workspace_name=workspace_display_name,
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


def _safe_workspaces(registry_service: Any) -> tuple[WorkspaceRecord, ...]:
    if registry_service is None:
        return ()
    try:
        workspaces = registry_service.list_workspaces()
    except Exception:
        logger.warning("Failed to list workspaces for display state", exc_info=True)
        return ()
    return tuple(workspaces or ())


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
    conversation_memberships = tuple(
        membership for membership in memberships if membership.item_type == "conversation"
    )
    duplicate_titles = _duplicate_membership_titles(conversation_memberships)
    rows: list[ConsoleWorkspaceConversationRow] = []
    for membership in conversation_memberships:
        rows.append(
            ConsoleWorkspaceConversationRow(
                conversation_id=membership.item_id,
                title=_membership_display_title(membership, duplicate_titles),
                status=membership.role,
            )
        )
    return tuple(rows)


def _handoff_rows_from_memberships(
    registry_service: Any,
    active_workspace: WorkspaceRecord,
) -> tuple[ConsoleWorkspaceHandoffRow, ...]:
    try:
        memberships = registry_service.list_workspace_memberships(active_workspace.workspace_id)
    except Exception:
        logger.warning(
            "Failed to read workspace memberships for Console handoff readiness",
            exc_info=True,
        )
        return ()
    memberships_seq = tuple(memberships or ())
    duplicate_titles = _duplicate_membership_titles(memberships_seq)
    rows: list[ConsoleWorkspaceHandoffRow] = []
    for membership in memberships_seq:
        rows.append(_handoff_row_from_membership(membership, duplicate_titles))
    return tuple(rows)


def _handoff_row_from_membership(
    membership: WorkspaceMembership,
    duplicate_titles: set[str] | None = None,
) -> ConsoleWorkspaceHandoffRow:
    policy = membership.transfer_policy
    label = f"Handoff: {policy.value}"
    portable = policy != WorkspaceTransferPolicy.LOCAL_ONLY
    details = {
        WorkspaceTransferPolicy.COPY: (
            "Eligible by copying source content into an explicit handoff package."
        ),
        WorkspaceTransferPolicy.REFERENCE: (
            "Eligible by uploading a stable source reference or pointer."
        ),
        WorkspaceTransferPolicy.METADATA_ONLY: (
            "Eligible as metadata only; source content is not copied."
        ),
        WorkspaceTransferPolicy.LOCAL_ONLY: (
            "Local-only; not portable to server or ACP package handoff."
        ),
    }
    return ConsoleWorkspaceHandoffRow(
        item_type=membership.item_type,
        item_id=membership.item_id,
        title=_membership_display_title(membership, duplicate_titles or set()),
        transfer_policy=policy,
        handoff_label=label,
        portable=portable,
        detail=details[policy],
    )


def _duplicate_membership_titles(
    memberships: Iterable[WorkspaceMembership],
) -> set[str]:
    counts: dict[str, int] = {}
    for membership in memberships:
        title = _membership_base_title(membership)
        counts[title] = counts.get(title, 0) + 1
    return {title for title, count in counts.items() if count > 1}


def _membership_display_title(
    membership: WorkspaceMembership,
    duplicate_titles: set[str],
) -> str:
    title = _membership_base_title(membership)
    if title not in duplicate_titles:
        return title
    short_id = str(membership.item_id or "").strip()[:8]
    return f"{title} [{short_id}]" if short_id else title


def _membership_base_title(membership: WorkspaceMembership) -> str:
    return str(membership.title or membership.item_id).strip()


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
    missing_count = sum(binding.status == RuntimeBindingStatus.MISSING for binding in bindings)
    label = (
        f"Runtime: {len(bindings)} {_plural('binding', len(bindings))}, "
        f"{ready_count} ready"
    )
    if missing_count:
        label = f"{label}, {missing_count} missing"
    return label


def _server_readiness(
    active_workspace: WorkspaceRecord,
    server_adapter_state: ConsoleWorkspaceServerAdapterState | None,
) -> tuple[str, str]:
    no_sync = "No background sync is running."
    adapter_boundary = "Server-backed hydration remains behind the workspace adapter boundary."
    if server_adapter_state is not None and not server_adapter_state.available:
        detail = server_adapter_state.detail.strip() or "No server workspace adapter is available."
        return "Server: unavailable", f"{detail} {adapter_boundary} {no_sync}"

    authority = active_workspace.authority
    if authority == WorkspaceAuthority.REMOTE_ONLY:
        return (
            "Server: remote-only",
            f"Remote workspace is visible but not materialized locally. {adapter_boundary} {no_sync}",
        )
    if authority == WorkspaceAuthority.CONFLICT:
        return (
            "Server: conflict",
            f"Local and server workspace state disagree; review required. {no_sync}",
        )
    if authority == WorkspaceAuthority.RUNTIME_MISSING:
        return (
            "Server: runtime missing",
            f"Workspace metadata exists but the runtime binding cannot be restored. {no_sync}",
        )
    if authority in {WorkspaceAuthority.SERVER_BACKED, WorkspaceAuthority.SYNCING_FROM_SERVER}:
        return (
            "Server: adapter ready",
            f"Server identity exists, but hydration still requires an explicit adapter action. {no_sync}",
        )
    if authority == WorkspaceAuthority.SYNCING_TO_SERVER:
        return (
            "Server: handoff pending",
            f"Workspace is marked for explicit handoff; automatic sync is not active. {no_sync}",
        )
    if authority == WorkspaceAuthority.DETACHED:
        return (
            "Server: detached",
            f"Workspace was previously server-backed but cannot verify server identity. {no_sync}",
        )
    return (
        "Server: local fallback",
        f"Local registry is authoritative for this workspace. {no_sync}",
    )


def _acp_handoff_state(
    state: ConsoleWorkspaceACPHandoffState | None,
) -> tuple[str, str, str]:
    if state is None:
        state = ConsoleWorkspaceACPHandoffState()
    status = str(state.status or "unavailable").strip().lower() or "unavailable"
    return (
        f"ACP task/run: {status}",
        state.detail.strip() or "ACP task/run package handoff is not wired.",
        state.audit_detail.strip() or "Audit: no ACP package was sent.",
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
