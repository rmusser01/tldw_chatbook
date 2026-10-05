"""Finite fresh MCP data preparation for asynchronous Console snapshots."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import MappingProxyType, MethodType
from typing import Any, Callable

from loguru import logger

from tldw_chatbook.Backup_Recovery import mcp_source_participants as sources
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

from .execution_log import (
    MCPExecutionLog,
    _CONSOLE_STANDARD_METHODS as _execution_methods,
)
from .hub_tool_catalog import (
    HubTool,
    builtin_tools_from_inventory,
    local_tools_from_record,
)
from .local_control_service import (
    LocalMCPControlService,
    _CONSOLE_STANDARD_METHODS as _local_control_methods,
)
from .local_store import (
    LocalMCPStore,
    _CONSOLE_STANDARD_METHODS as _local_store_methods,
)
from .permission_store import (
    MCPPermissionStore,
    EffectiveToolState,
    _CONSOLE_STANDARD_METHODS as _permission_methods,
    definition_hash,
    resolve_effective_state,
)
from .unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
    _CONSOLE_STANDARD_METHODS as _unified_methods,
)


# Callable provenance only: native source admission remains fresh for every read.
_ORIGINAL_METHODS = MappingProxyType(
    {
        (owner, name): function
        for owner, methods in (
            (UnifiedMCPControlPlaneService, _unified_methods),
            (LocalMCPControlService, _local_control_methods),
            (LocalMCPStore, _local_store_methods),
            (MCPPermissionStore, _permission_methods),
            (MCPExecutionLog, _execution_methods),
        )
        for name, function in methods
    }
)


def _source_identity(source: Any) -> tuple[Any, ...]:
    """Name exact installed metadata; the actual reader still proves it natively."""
    bound = sources._BINDINGS.get(source)
    if bound is None or type(source) is not bound.source_type:
        raise RecoveryRequired("mcp_source_not_installed")
    config = bound.config
    return (
        source,
        type(source),
        Path(source.path),
        bound,
        config,
        sys.modules.get("tldw_chatbook.config"),
        config.current_config_identity(),
        config._CONFIG_CACHE_SOURCE,
        bound.profile,
        bound.selected,
    )


def _original_method(callback: Any, receiver: Any, name: str, owner: type) -> bool:
    return (
        isinstance(callback, MethodType)
        and callback.__func__ is _ORIGINAL_METHODS.get((owner, name))
        and callback.__self__ is receiver
    )


def _original_bound_method(receiver: Any, name: str, owner: type) -> bool:
    return _original_method(getattr(receiver, name, None), receiver, name, owner)


def _original_class_member(owner: type, name: str) -> bool:
    return vars(owner).get(name) is _ORIGINAL_METHODS.get((owner, name))


def _same_callback(current: Any, original: Any) -> bool:
    return current is original or (
        isinstance(current, MethodType)
        and isinstance(original, MethodType)
        and current.__func__ is original.__func__
        and current.__self__ is original.__self__
    )


_LOCAL_CATALOG_METHODS = ("get_external_servers", "_project_external_catalog")
_STORE_CATALOG_METHODS = ("get_catalog_bundle", "get_external_catalog", "load")
_PERMISSION_METHODS = ("load", "mark_config_changed", "get_kill_switch")


def standard_local_catalog_sources(local: Any, store: Any) -> bool:
    """Keep declared local/store custom readers on their preceding route."""
    if type(local) is not LocalMCPControlService or type(store) is not LocalMCPStore:
        return False
    for receiver, owner, names in (
        (local, LocalMCPControlService, _LOCAL_CATALOG_METHODS),
        (store, LocalMCPStore, _STORE_CATALOG_METHODS),
    ):
        if not all(
            _original_class_member(owner, name)
            and _original_bound_method(receiver, name, owner)
            for name in names
        ):
            return False
    try:
        _source_identity(store)
    except RecoveryRequired:
        return False
    return True


def standard_console_sources(service: Any) -> bool:
    """Select only the concrete source route, without granting file authority."""
    if type(service) is not UnifiedMCPControlPlaneService:
        return False
    for name in ("permission_store", "execution_log"):
        if not _original_class_member(UnifiedMCPControlPlaneService, name):
            return False
    for name in (
        "get_kill_switch",
        "effective_tool_states",
        "_audit_downgrade_if_fresh",
    ):
        if not (
            _original_class_member(UnifiedMCPControlPlaneService, name)
            and _original_bound_method(service, name, UnifiedMCPControlPlaneService)
        ):
            return False
    local = service.local_service
    store = getattr(local, "store", None)
    if not standard_local_catalog_sources(local, store):
        return False
    if not (
        _original_class_member(LocalMCPControlService, "get_inventory")
        and _original_bound_method(local, "get_inventory", LocalMCPControlService)
    ):
        return False
    permission = service._permission_store
    log = service._execution_log
    # Absent lazy receivers must qualify before construction and publication too.
    for receiver, owner, names in (
        (permission, MCPPermissionStore, _PERMISSION_METHODS),
        (log, MCPExecutionLog, ("append",)),
    ):
        if receiver is not None and type(receiver) is not owner:
            return False
        if not all(
            _original_class_member(owner, name)
            and (receiver is None or _original_bound_method(receiver, name, owner))
            for name in names
        ):
            return False
        if receiver is not None:
            try:
                _source_identity(receiver)
            except RecoveryRequired:
                return False
    return True


def standard_console_catalog_sources(service: Any) -> bool:
    """Preserve explicit custom catalog callbacks' late receiver contract."""
    return (
        standard_console_sources(service)
        and _original_class_member(
            UnifiedMCPControlPlaneService, "local_external_catalog"
        )
        and _original_bound_method(
            service, "local_external_catalog", UnifiedMCPControlPlaneService
        )
    )


async def _owned_worker(call: Callable[[], Any]) -> Any:
    worker = asyncio.create_task(asyncio.to_thread(call))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise


def _checked_read(source: Any, call: Callable[[], Any]) -> tuple[Any, tuple[Any, ...]]:
    # The receipt is captured from the actual issued scope while the original
    # source method and all its fresh native guards execute inside that scope.
    with raw._scope(source, sources.ROUTE, writing=True) as operation:
        state = raw._states[operation]
        if state.source is not source or state.participant is None:
            raise RecoveryRequired("mcp_source_not_installed")
        before = _source_identity(source)
        payload = call()
        raw._check(operation, state.selected)
        after = _source_identity(source)
        if before != after or Path(source.path) != state.selected:
            raise RecoveryRequired("mcp_source_selection_changed")
        return payload, after


class _CapturedLocalCatalog:
    """Retain one exact local/store worker and its caller-owned projection."""

    def __init__(self, local, store):
        self.local = local
        self.store = store
        self.identity = _source_identity(store)
        self.binding = self.identity[3]
        self.policy = local.policy_enforcer
        self.callbacks = self._callbacks()
        self.catalog_reader = self.callbacks[2]
        self.load_reader = self.callbacks[4]
        self.projector = self.callbacks[1]
        self.require_current()

    def _callbacks(self):
        return (
            self.local.get_external_servers,
            self.local._project_external_catalog,
            self.store.get_catalog_bundle,
            self.store.get_external_catalog,
            self.store.load,
        )

    def require_current(self):
        if not (
            standard_local_catalog_sources(self.local, self.store)
            and self.local.store is self.store
            and self.local.policy_enforcer is self.policy
            and sources._BINDINGS.get(self.store) is self.binding
            and _source_identity(self.store) == self.identity
            and all(
                _same_callback(current, original)
                for current, original in zip(self._callbacks(), self.callbacks)
            )
        ):
            raise RecoveryRequired("mcp_source_selection_changed")

    def read_bundle(self):
        self.require_current()
        bundle = self.catalog_reader(_captured_load=self.load_reader)
        self.require_current()
        return bundle


class _CapturedSources:
    """Ephemeral exact receivers for one accepted finite preparation."""

    def __init__(self, service, *, capture_catalog: bool = False):
        self.service = service
        self._capture_catalog = capture_catalog
        if capture_catalog and not standard_console_catalog_sources(service):
            raise RecoveryRequired("mcp_source_selection_changed")
        self.local = service.local_service
        self.store = self.local.store
        self.permission = service._permission_store
        self.log = service._execution_log
        self.store_identity = _source_identity(self.store)
        self.permission_identity = (
            _source_identity(self.permission) if self.permission is not None else None
        )
        self.log_identity = _source_identity(self.log) if self.log is not None else None
        self.store_binding = self.store_identity[3]
        self.permission_binding = (
            self.permission_identity[3]
            if self.permission_identity is not None
            else None
        )
        self.log_binding = (
            self.log_identity[3] if self.log_identity is not None else None
        )
        self.manifest_provider = self.local.manifest_provider
        self.policy_enforcer = self.local.policy_enforcer
        self.method_bindings = self._method_bindings()
        self.catalog_callback = (
            self.service.local_external_catalog if capture_catalog else None
        )
        self.catalog_reader = self.method_bindings["store.bundle"]
        self.store_loader = self.method_bindings["store.load"]
        self.inventory_reader = self.method_bindings["local.inventory"]
        self.effective_reader = self.method_bindings["service.effective"]
        self.audit_writer = self.method_bindings["service.audit"]
        self.permission_reader = self.method_bindings["permission.load"]
        self.require_current()

    def _method_bindings(self):
        return {
            "service.kill": self.service.get_kill_switch,
            "service.effective": self.service.effective_tool_states,
            "service.audit": self.service._audit_downgrade_if_fresh,
            "local.external": self.local.get_external_servers,
            "local.inventory": self.local.get_inventory,
            "local.projection": self.local._project_external_catalog,
            "store.bundle": self.store.get_catalog_bundle,
            "store.external": self.store.get_external_catalog,
            "store.load": self.store.load,
            "permission.load": self.permission.load
            if self.permission is not None
            else None,
            "permission.mark": self.permission.mark_config_changed
            if self.permission is not None
            else None,
            "permission.kill": self.permission.get_kill_switch
            if self.permission is not None
            else None,
            "log.append": self.log.append if self.log is not None else None,
        }

    def require_current(self):
        # Refuse replaced class descriptors before any dynamic getter lookup.
        qualifier = (
            standard_console_catalog_sources
            if self._capture_catalog
            else standard_console_sources
        )
        if not qualifier(self.service):
            raise RecoveryRequired("mcp_source_selection_changed")
        current_bindings = self._method_bindings()
        if not (
            all(
                _same_callback(current_bindings[name], original)
                for name, original in self.method_bindings.items()
            )
            and (
                not self._capture_catalog
                or _same_callback(
                    self.service.local_external_catalog, self.catalog_callback
                )
            )
            and self.local.manifest_provider is self.manifest_provider
            and self.local.policy_enforcer is self.policy_enforcer
            and sources._BINDINGS.get(self.store) is self.store_binding
            and (
                self.permission is None
                or sources._BINDINGS.get(self.permission) is self.permission_binding
            )
            and (
                self.log is None or sources._BINDINGS.get(self.log) is self.log_binding
            )
            and self.service.local_service is self.local
            and self.local.store is self.store
            and self.service._permission_store is self.permission
            and self.service._execution_log is self.log
            and _source_identity(self.store) == self.store_identity
            and (
                self.permission is None
                or _source_identity(self.permission) == self.permission_identity
            )
            and (self.log is None or _source_identity(self.log) == self.log_identity)
        ):
            raise RecoveryRequired("mcp_source_selection_changed")

    def permission_owner(self):
        self.require_current()
        if self.permission is None:
            canonical = Path(
                getattr(self.store, "_recovery_original_path", self.store.path)
            )
            candidate = MCPPermissionStore(canonical.with_name("mcp_permissions.json"))
            self.require_current()
            callbacks = {
                "permission.load": candidate.load,
                "permission.mark": candidate.mark_config_changed,
                "permission.kill": candidate.get_kill_switch,
            }
            identity = _source_identity(candidate)
            with self.service._execution_log_init_lock:
                self.require_current()
                if not all(
                    _original_method(
                        callbacks[key], candidate, name, MCPPermissionStore
                    )
                    for key, name in (
                        ("permission.load", "load"),
                        ("permission.mark", "mark_config_changed"),
                        ("permission.kill", "get_kill_switch"),
                    )
                ):
                    raise RecoveryRequired("mcp_source_selection_changed")
                self.service._permission_store = candidate
                self.permission = candidate
                self.permission_identity = identity
                self.permission_reader = callbacks["permission.load"]
                self.permission_binding = identity[3]
                self.method_bindings = {**self.method_bindings, **callbacks}
                self.require_current()
        return self.permission

    def permission_call(self, call):
        permission = self.permission_owner()
        result, receipt = _checked_read(permission, lambda: call(permission))
        self.require_current()
        if receipt != self.permission_identity:
            raise RecoveryRequired("mcp_source_selection_changed")
        return result

    def log_owner(self):
        self.require_current()
        if self.log is None:
            candidate = MCPExecutionLog(
                self.store_identity[2].with_name("mcp_execution_log.jsonl")
            )
            self.require_current()
            append = candidate.append
            identity = _source_identity(candidate)
            with self.service._execution_log_init_lock:
                self.require_current()
                if not _original_method(append, candidate, "append", MCPExecutionLog):
                    raise RecoveryRequired("mcp_source_selection_changed")
                self.service._execution_log = candidate
                self.log = candidate
                self.log_identity = identity
                self.log_binding = identity[3]
                self.method_bindings = {**self.method_bindings, "log.append": append}
                self.require_current()
        self.require_current()
        return self.log


async def read_console_kill_switch(
    service: UnifiedMCPControlPlaneService,
    *,
    _captured_sources: _CapturedSources | None = None,
) -> bool:
    """Read one fresh compose-time switch; retain native custody on cancellation."""
    captured = (
        _captured_sources
        if _captured_sources is not None
        else _CapturedSources(service)
    )
    if captured.service is not service:
        raise RecoveryRequired("mcp_source_selection_changed")
    captured.require_current()
    with service._producer_lifetime.operation():
        payload = await _owned_worker(
            lambda: captured.permission_call(lambda owner: captured.permission_reader())
        )
        captured.require_current()
        return bool(payload.get("kill_switch", False))


async def capture_console_effective_states(
    service: UnifiedMCPControlPlaneService,
    tools: list[HubTool],
    *,
    profile_id: str = "default",
    _captured_sources: _CapturedSources | None = None,
) -> dict[tuple[str, str], EffectiveToolState]:
    """Keep original state/audit behavior on exact captured native receivers."""
    captured = (
        _captured_sources
        if _captured_sources is not None
        else _CapturedSources(service)
    )
    if captured.service is not service:
        raise RecoveryRequired("mcp_source_selection_changed")
    captured.require_current()
    with service._producer_lifetime.operation():
        states = await _owned_worker(
            lambda: captured.permission_call(
                lambda owner: captured.effective_reader(
                    tools,
                    profile_id=profile_id,
                    _captured_permission_store=owner,
                    _captured_execution_log=captured.log_owner,
                )
            )
        )
        captured.require_current()
        return states


async def capture_console_definition_maximum(
    service: UnifiedMCPControlPlaneService,
    exclusions: frozenset[str],
) -> dict[str, str]:
    """Prepare one narrowing maximum; dispatch still uses its fresh live gates."""
    captured = _CapturedSources(service)

    def read_sources():
        payload = captured.permission_call(lambda owner: captured.permission_reader())
        bundle = None
        if not bool(payload.get("kill_switch", False)):
            bundle, receipt = _checked_read(
                captured.store,
                lambda: captured.catalog_reader(_captured_load=captured.store_loader),
            )
            captured.require_current()
            if receipt != captured.store_identity:
                raise RecoveryRequired("mcp_source_selection_changed")
        return payload, bundle

    # Entry itself refuses a closed producer. Accepted read uncertainty retains
    # the existing empty-maximum behavior, without suppressing cancellation.
    with service._producer_lifetime.operation():
        try:
            captured.local._require_allowed("mcp.external_profiles.list.local")
            payload, bundle = await _owned_worker(read_sources)
            captured.require_current()
            if bool(payload.get("kill_switch", False)):
                return {}
            captured.local._require_allowed("mcp.external_profiles.list.local")
            tools = []
            for profile in bundle["profiles"]:
                tools.extend(
                    local_tools_from_record(
                        {
                            **profile,
                            "discovery_snapshot": bundle["discovery_snapshots"].get(
                                str(profile["profile_id"]).strip()
                            ),
                        }
                    )
                )
            inventory = captured.inventory_reader()
            if isinstance(inventory, dict):
                tools.extend(builtin_tools_from_inventory(inventory))
            states = {
                (tool.server_key, tool.name): resolve_effective_state(
                    payload, tool, profile_id="default"
                )
                for tool in tools
            }
            changed = [
                tool
                for tool in tools
                if states[(tool.server_key, tool.name)].config_changed
            ]
            if changed:

                def audit_changes():
                    for tool in changed:
                        captured.require_current()
                        captured.audit_writer(
                            captured.permission,
                            tool,
                            profile_id="default",
                            _captured_execution_log=captured.log_owner,
                        )

                await _owned_worker(audit_changes)
            captured.require_current()
            return {
                tool.tool_id: definition_hash(tool.description, tool.input_schema)
                for tool in tools
                if not (
                    tool.server_key == "builtin:tldw_chatbook"
                    and tool.name in exclusions
                )
                and states[(tool.server_key, tool.name)].state != "deny"
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "Console MCP maximum capture unavailable (exception_type={})",
                type(exc).__name__,
            )
            return {}
