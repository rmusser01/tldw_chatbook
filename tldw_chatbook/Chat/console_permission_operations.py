"""Finite in-process operations over the existing permission singleton.

This is an opt-in local surface. It does not install a transport, dispatch loop,
second store, publication mirror, or session-grant owner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import (
    EffectiveToolState,
    MCPPermissionStore,
    PermissionStoreSnapshot,
    profile_lifecycle_disposition,
    profile_policy_chain_digest,
)
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)
from tldw_chatbook.Tool_Packs.binding import profile_policy_digest

if TYPE_CHECKING:
    from tldw_chatbook.Tool_Packs.activation import ToolPackActivationResult
    from tldw_chatbook.Tool_Packs.binding import ToolProfileBindingReview
    from tldw_chatbook.Tool_Packs.importer import ServerMapping, ToolPackImportReview
    from tldw_chatbook.Tool_Packs.removal import ToolProfileRemovalResult
    from tldw_chatbook.Tool_Packs.service import ToolProfileListing
    from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults


@dataclass(frozen=True, slots=True)
class PermissionRenderCapture:
    """Detached display observation; never an execution grant."""

    snapshot: PermissionStoreSnapshot
    effective_states: Mapping[tuple[str, str], EffectiveToolState]
    profile_identity: tuple[str, int | None, str] | None


def _profile_identity(snapshot, profile_id):
    profile = snapshot.payload.get("profiles", {}).get(profile_id)
    if not isinstance(profile, Mapping):
        return None
    disposition = profile_lifecycle_disposition(profile)
    if disposition in {"invalid", "tombstone"}:
        return None
    digest = profile_policy_digest(profile)
    revision = None
    if disposition == "imported":
        lifecycle = profile["tool_pack_lifecycle"]
        if lifecycle["policy_digest"] != digest:
            return None
        revision = lifecycle["revision"]
    return digest, revision, profile_policy_chain_digest(snapshot.payload, profile_id)


class ConsolePermissionOperations:
    """Keep complete permission actions on one actual control-plane owner."""

    def __init__(
        self,
        control_plane: UnifiedMCPControlPlaneService,
        *,
        runtime: Any = None,
        tool_packs: Any = None,
    ) -> None:
        if type(control_plane) is not UnifiedMCPControlPlaneService:
            raise TypeError("Custom control planes retain their legacy route")
        if runtime is not None:
            from .console_runtime import ConsoleRuntime

            if type(runtime) is not ConsoleRuntime or runtime.domain_services is None:
                raise TypeError(
                    "Selected operations require the actual explicit runtime"
                )
        # Selection requires the already supplied singleton; no lazy receiver
        # construction or second store is performed on the owner loop.
        store = vars(control_plane).get("_permission_store")
        if type(store) is not MCPPermissionStore:
            raise TypeError("An existing stock permission store is required")
        self._control_plane = control_plane
        self._store = store
        from tldw_chatbook.Backup_Recovery import mcp_source_participants

        self._source_receipt = mcp_source_participants.source_receipt(store)
        self._publication = store._permission_publication
        self._path_lock = store._path_lock
        self._grants = control_plane._session_approvals
        self._grant_lock = control_plane._session_approvals_lock
        self._runtime = runtime
        self._domain = None if runtime is None else runtime.domain_services
        self._controller = None if runtime is None else runtime._chat_controller
        self._host = (
            None if self._controller is None else self._controller._interrupt_host
        )
        self._tool_packs = tool_packs
        self._reads: set[Any] = set()
        self._unresolved: list[Any] = []
        self._require_current()

    def _require_current(self) -> None:
        if any(not operation.retired for operation in self._unresolved):
            raise RecoveryRequired("permission_operation_unresolved")
        from tldw_chatbook.MCP import permission_store as permissions
        from tldw_chatbook.MCP import unified_control_plane_service as unified
        from tldw_chatbook.Backup_Recovery import mcp_source_participants

        service, store = self._control_plane, self._store
        if (
            type(service) is not UnifiedMCPControlPlaneService
            or type(store) is not MCPPermissionStore
            or vars(service).get("_permission_store") is not store
            or store._permission_publication is not self._publication
            or store._path_lock is not self._path_lock
            or service._session_approvals is not self._grants
            or service._session_approvals_lock is not self._grant_lock
            or mcp_source_participants.source_receipt(store) != self._source_receipt
            or not permissions._permission_process_definitions_current(store)
            or any(
                name in vars(service)
                or vars(UnifiedMCPControlPlaneService).get(name) is not member
                or not all(
                    function.__code__ is code
                    and unified._capture_controller_inputs(function) == inputs
                    for function, code, inputs in records
                )
                for name, member, records in unified._LOCAL_PERMISSION_DEFINITIONS
            )
        ):
            raise RecoveryRequired("permission_operation_source_changed")
        if self._domain is not None:
            self._domain.require_current()
            if (
                self._runtime.domain_services is not self._domain
                or self._domain.sources.control_plane is not service
                or self._domain.sources.tool_pack_service is not self._tool_packs
                or self._runtime._disposed
            ):
                raise RecoveryRequired("permission_operation_domain_changed")
        if self._tool_packs is not None:
            from tldw_chatbook.Tool_Packs.service import (
                ToolPackService,
                _CONSOLE_COMPOSITION_FIELDS,
                _CONSOLE_TOOL_PACK_DEFINITIONS,
            )
            from tldw_chatbook.Utils.definition_records import definitions_current
            from tldw_chatbook.Workspaces.registry_service import (
                DeferredWorkspaceToolProfileGuard,
            )

            packs = self._tool_packs
            actors = (
                packs,
                packs._activation,
                packs._importer,
                packs._removal,
                packs.binding_guard,
                packs.lifecycle,
                packs._receipt_store,
                packs._references,
            )
            attached = packs._workspace_registry.tool_profile_guard
            if type(attached) is DeferredWorkspaceToolProfileGuard:
                attached = attached.active_guard
            if (
                type(packs) is not ToolPackService
                or attached is not packs.binding_guard
                or packs._permission_store is not store
                or packs._activation._permission_store is not store
                or packs._importer._permission_store is not store
                or packs._removal._permission_store is not store
                or packs.binding_guard._permission_store is not store
                or packs._activation._lifecycle is not packs.lifecycle
                or packs._removal._lifecycle is not packs.lifecycle
                or packs.binding_guard.lifecycle is not packs.lifecycle
                or packs._activation._importer is not packs._importer
                or packs._activation._inventory is not packs._inventory
                or packs._importer._inventory is not packs._inventory
                or packs.binding_guard._inventory is not packs._inventory
                or packs._activation._receipt_store is not packs._receipt_store
                or packs._removal._receipt_store is not packs._receipt_store
                or packs._activation._reference_checker is not packs._references
                or packs._importer._reference_checker is not packs._references
                or packs._removal._reference_checker is not packs._references
                or packs._references._registry is not packs._workspace_registry
                or getattr(
                    packs.binding_guard._workspace_defaults_reader, "__self__", None
                )
                is not packs._references
                or getattr(
                    packs.binding_guard._workspace_defaults_reader, "__func__", None
                )
                is not type(packs._references).defaults_for
                or tuple(name for name, _ in getattr(packs, "_console_composition", ()))
                != _CONSOLE_COMPOSITION_FIELDS
                or any(
                    getattr(packs, name) is not value
                    for name, value in packs._console_composition
                )
                or not definitions_current(_CONSOLE_TOOL_PACK_DEFINITIONS)
                or tuple(
                    (owner, name)
                    for owner, name, *_ in getattr(packs, "_console_callbacks", ())
                )
                != (
                    (packs, "_now"),
                    (packs._importer, "_now"),
                    (packs._activation, "_now"),
                    (packs.binding_guard, "_now"),
                    (packs.binding_guard, "_workspace_defaults_reader"),
                )
                or not definitions_current(packs._console_callbacks)
                or any(
                    not any(
                        type(actor) is owner
                        for owner, *_ in _CONSOLE_TOOL_PACK_DEFINITIONS
                    )
                    or any(
                        name in vars(actor)
                        for owner, name, *_ in _CONSOLE_TOOL_PACK_DEFINITIONS
                        if type(actor) is owner
                    )
                    for actor in actors
                )
            ):
                raise RecoveryRequired("permission_profile_owner_changed")

    async def _native(self, body):
        """Retain this method's private fixed body through caller cancellation."""
        from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import (
            run_preparation_read,
        )
        from tldw_chatbook.MCP.local_permission_operation import (
            LocalPermissionOperation,
        )

        # The original wrappers use this module's enrollment hooks. Loading it
        # creates no broker; only its exact selected source bypasses transport.
        from tldw_chatbook.MCP import permission_process  # noqa: F401
        from contextlib import nullcontext

        controls = (
            self._runtime._execution_service._local_controls
            if self._runtime is not None
            else None
        )

        def invoke():
            operation = LocalPermissionOperation(self._store, self._require_current)
            effect = (
                controls.permission_effect(self, operation)
                if controls is not None
                else nullcontext()
            )
            with effect:
                try:
                    with operation:
                        return body()
                finally:
                    if not operation.retired:
                        self._unresolved.append(operation)

        observers = () if self._runtime is None else (self._runtime._preparation_reads,)
        return await run_preparation_read(
            invoke,
            creator=self,
            session_id=None,
            reads=self._reads,
            observers=observers,
            source=self._store,
            require_current=self._require_current,
        )

    async def read_permission_snapshot(
        self, *, strict: bool = True
    ) -> PermissionStoreSnapshot | dict[str, Any]:
        """Preserve the caller's strict or compatibility read contract."""
        if type(strict) is not bool:
            raise TypeError("strict must be bool")
        return await self._native(
            lambda: self._store.read_snapshot_strict() if strict else self._store.load()
        )

    async def read_permission_inventory(self) -> PermissionStoreSnapshot:
        return await self._native(lambda: self._store.read_profile_inventory_snapshot())

    async def read_kill_switch(self) -> bool:
        return await self._native(lambda: self._control_plane.get_kill_switch())

    async def read_effective_states(
        self, tools: list[HubTool], *, profile_id: str = "default"
    ) -> dict[tuple[str, str], EffectiveToolState]:
        return await self._native(
            lambda: self._control_plane.effective_tool_states(
                tools, profile_id=profile_id
            )
        )

    async def capture_permission_render(
        self, tools: list[HubTool], *, profile_id: str = "default"
    ) -> PermissionRenderCapture:
        """Keep inventory, downgrade audit and successor recapture in one fence."""

        def capture():
            with self._store.mutation_fence():
                snapshot = self._store.read_profile_inventory_snapshot()
                identity = _profile_identity(snapshot, profile_id)
                fail_closed = {
                    (tool.server_key, tool.name): EffectiveToolState(
                        state="ask", origin="global_default"
                    )
                    for tool in tools
                }
                if identity is None:
                    return PermissionRenderCapture(snapshot, fail_closed, None)
                for _attempt in range(2):
                    states = self._control_plane.effective_tool_states(
                        tools, profile_id=profile_id
                    )
                    snapshot = self._store.read_profile_inventory_snapshot()
                    successor = _profile_identity(snapshot, profile_id)
                    if successor == identity:
                        return PermissionRenderCapture(snapshot, states, successor)
                    if successor is None:
                        break
                    identity = successor
                return PermissionRenderCapture(snapshot, fail_closed, None)

        return await self._native(capture)

    def resolve_approval_round(
        self, decisions: dict[str, str], *, round_id: str | None
    ) -> None:
        """Use the real retained host's identity/decision path on its owner loop."""
        from .console_interrupt_rounds import (
            InterruptRoundHost,
            _LOCAL_APPROVAL_RESOLUTION,
        )
        from tldw_chatbook.Utils.definition_records import definitions_current

        self._require_current()
        if (
            self._runtime is None
            or self._controller is None
            or self._runtime._chat_controller is not self._controller
            or self._controller._interrupt_host is not self._host
            or type(self._host) is not InterruptRoundHost
            or "resolve_pending_approval" in vars(self._host)
            or not definitions_current(_LOCAL_APPROVAL_RESOLUTION)
        ):
            raise RecoveryRequired("permission_approval_host_unavailable")
        self._host.resolve_pending_approval(decisions, round_id=round_id)

    async def list_session_approvals(
        self, *, profile_id: str = "default"
    ) -> list[tuple[str, str]]:
        return await self._native(
            lambda: self._control_plane.list_session_approvals(profile_id=profile_id)
        )

    async def revoke_session_approval(
        self, server_key: str, tool_name: str, *, profile_id: str = "default"
    ) -> bool:
        return await self._native(
            lambda: self._control_plane.revoke_session_approval(
                server_key, tool_name, profile_id=profile_id
            )
        )

    async def clear_session_approvals(self, *, profile_id: str | None = None) -> None:
        return await self._native(
            lambda: self._control_plane.clear_session_approvals(profile_id=profile_id)
        )

    async def set_tool_state(
        self,
        server_key: str,
        tool_name: str,
        ui_state: str | None,
        *,
        tool: HubTool | None = None,
        profile_id: str = "default",
        expected_profile_digest: str | None = None,
        expected_revision: int | None = None,
        expected_policy_chain_digest: str | None = None,
    ) -> None:
        return await self._native(
            lambda: self._control_plane.set_tool_state(
                server_key,
                tool_name,
                ui_state,
                tool=tool,
                profile_id=profile_id,
                expected_profile_digest=expected_profile_digest,
                expected_revision=expected_revision,
                expected_policy_chain_digest=expected_policy_chain_digest,
            )
        )

    async def set_server_default(
        self,
        server_key: str,
        state: str | None,
        *,
        profile_id: str = "default",
        expected_profile_digest: str | None = None,
        expected_revision: int | None = None,
    ) -> None:
        return await self._native(
            lambda: self._control_plane.set_server_default(
                server_key,
                state,
                profile_id=profile_id,
                expected_profile_digest=expected_profile_digest,
                expected_revision=expected_revision,
            )
        )

    async def set_global_default(
        self,
        state: str,
        *,
        profile_id: str = "default",
        expected_profile_digest: str | None = None,
        expected_revision: int | None = None,
    ) -> None:
        return await self._native(
            lambda: self._control_plane.set_global_default(
                state,
                profile_id=profile_id,
                expected_profile_digest=expected_profile_digest,
                expected_revision=expected_revision,
            )
        )

    async def set_kill_switch(self, value: bool) -> None:
        return await self._native(lambda: self._control_plane.set_kill_switch(value))

    async def add_arg_rule(
        self,
        server_key: str,
        tool_name: str,
        *,
        args: Mapping[str, Any],
        tool: HubTool | None = None,
        profile_id: str = "default",
    ) -> None:
        return await self._native(
            lambda: self._control_plane.add_tool_arg_rule(
                server_key, tool_name, args=args, tool=tool, profile_id=profile_id
            )
        )

    async def remove_arg_rule(
        self,
        server_key: str,
        tool_name: str,
        rule_id: str,
        *,
        profile_id: str = "default",
    ) -> bool:
        return await self._native(
            lambda: self._control_plane.remove_tool_arg_rule(
                server_key, tool_name, rule_id, profile_id=profile_id
            )
        )

    async def read_arg_rules(
        self, server_key: str, tool_name: str, *, profile_id: str = "default"
    ) -> list[dict[str, Any]]:
        return await self._native(
            lambda: self._control_plane.list_tool_arg_rules(
                server_key, tool_name, profile_id=profile_id
            )
        )

    async def dismiss_suggestion(self, server_key: str, tool_name: str) -> None:
        return await self._native(
            lambda: self._control_plane.dismiss_permission_suggestion(
                server_key, tool_name
            )
        )

    async def restore_suggestion(self, server_key: str, tool_name: str) -> None:
        return await self._native(
            lambda: self._control_plane.restore_permission_suggestion(
                server_key, tool_name
            )
        )

    async def clear_suggestion_dismissals(self) -> None:
        return await self._native(
            lambda: self._control_plane.clear_permission_suggestion_dismissals()
        )

    def _profiles(self):
        self._require_current()
        if self._tool_packs is None:
            raise RecoveryRequired("permission_profile_owner_unavailable")
        return self._tool_packs

    async def inspect_import(
        self,
        archive_path: Path,
        *,
        destination_id: str,
        mappings: Sequence[ServerMapping] = (),
    ) -> ToolPackImportReview:
        packs = self._profiles()
        return await self._native(
            lambda: packs.inspect_import(
                archive_path, destination_id=destination_id, mappings=mappings
            )
        )

    async def import_unbound_profile(
        self, review: ToolPackImportReview
    ) -> ToolPackActivationResult:
        packs = self._profiles()
        return await self._native(lambda: packs.import_unbound(review))

    async def remove_imported_profile(
        self, profile_id: str, *, expected_revision: int
    ) -> ToolProfileRemovalResult:
        packs = self._profiles()
        return await self._native(
            lambda: packs.remove_profile(
                profile_id, expected_revision=expected_revision
            )
        )

    async def list_tool_profiles(self) -> ToolProfileListing:
        packs = self._profiles()
        return await self._native(packs.list_profiles)

    async def review_first_bind(
        self,
        workspace_id: str,
        intended_defaults: WorkspaceAssistantDefaults,
        *,
        action: str,
    ) -> ToolProfileBindingReview:
        packs = self._profiles()
        return await self._native(
            lambda: packs.review_first_bind(
                workspace_id, intended_defaults, action=action
            )
        )

    async def confirm_first_bind(self, review: ToolProfileBindingReview) -> str:
        """Return the original guard token; the actual workspace setter consumes it."""
        packs = self._profiles()
        return await self._native(lambda: packs.confirm_first_bind(review))
