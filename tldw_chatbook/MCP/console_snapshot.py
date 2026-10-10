"""Finite fresh MCP data preparation for asynchronous Console snapshots."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import FunctionType, MappingProxyType, MethodType
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
    _CONSOLE_OWNED_LOAD_HELPERS as _permission_load_helpers,
    _CONSOLE_CONTROLLER_PERMISSION_MODULE as _permission_switch_module,
    _CONSOLE_CONTROLLER_PERMISSION_CLASS as _permission_switch_class,
    _CONSOLE_CONTROLLER_PERMISSION_METHODS as _permission_switch_methods,
    definition_hash,
    resolve_effective_state,
)
from .unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
    _CONSOLE_STANDARD_METHODS as _unified_methods,
    _CONSOLE_CONTROLLER_SWITCH_MODULE as _unified_switch_module,
    _CONSOLE_CONTROLLER_SWITCH_CLASS as _unified_switch_class,
    _CONSOLE_CONTROLLER_SWITCH_METHODS as _unified_switch_methods,
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


def _owned_permission_load_helpers_current() -> bool:
    """Qualify directly consumed owned-load helpers before invoking them."""
    if (
        sys.modules.get(_permission_switch_module.__name__)
        is not _permission_switch_module
        or not _controller_inputs_checker_current()
    ):
        return False
    namespace = vars(_permission_switch_module)
    return all(
        namespace.get(name) is function
        and function.__code__ is code
        and function.__globals__ is defining
        and defining is namespace
        and _controller_inputs_current(function, inputs)
        for name, function, code, defining, inputs in _permission_load_helpers
    )


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

    def read_permission_payload(self):
        """Load stock policy within the existing checked source operation."""
        permission = self.permission_owner()
        reader = self.permission_reader
        helpers = {name: function for name, function, *_ in _permission_load_helpers}
        captured = None
        if _owned_permission_load_helpers_current():
            captured = helpers["_capture_console_owned_load"](permission)

        def read():
            self.require_current()
            if captured is None:
                return reader()
            if not _owned_permission_load_helpers_current() or not helpers[
                "_console_owned_load_current"
            ](permission, captured):
                raise RecoveryRequired("mcp_source_selection_changed")
            return helpers["_load_in_owned_scope"](
                permission, getattr(raw._local, "operation", None), captured
            )

        result, receipt = _checked_read(permission, read)
        self.require_current()
        if (
            receipt != self.permission_identity
            or captured is not None
            and (
                not _owned_permission_load_helpers_current()
                or not helpers["_console_owned_load_current"](permission, captured)
            )
        ):
            raise RecoveryRequired("mcp_source_selection_changed")
        return result

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
        payload = await _owned_worker(captured.read_permission_payload)
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
    *,
    _run_native=None,
) -> dict[str, str]:
    """Prepare one narrowing maximum; dispatch still uses its fresh live gates."""
    captured = _CapturedSources(service)
    run_native = _owned_worker if _run_native is None else _run_native

    def read_sources():
        payload = captured.read_permission_payload()
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
            payload, bundle = await run_native(read_sources)
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

                await run_native(audit_changes)
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


def _capture_controller_inputs(function):
    keyword_defaults = function.__kwdefaults__
    closure = function.__closure__
    return (
        function.__defaults__,
        keyword_defaults,
        tuple(dict.items(keyword_defaults)) if keyword_defaults is not None else (),
        closure,
        tuple((cell, cell.cell_contents) for cell in closure or ()),
    )


def _controller_inputs_current(function, inputs):
    defaults, keyword_defaults, keyword_items, closure, cells = inputs
    if (
        function.__defaults__ is not defaults
        or function.__kwdefaults__ is not keyword_defaults
        or function.__closure__ is not closure
    ):
        return False
    if keyword_defaults is not None:
        if type(keyword_defaults) is not dict:  # noqa: E721 -- exact built-in inputs
            return False
        current_items = tuple(dict.items(keyword_defaults))
        if len(current_items) != len(keyword_items) or not all(
            current_key is key and current_value is value
            for (current_key, current_value), (key, value) in zip(
                current_items, keyword_items
            )
        ):
            return False
    if len(closure or ()) != len(cells):
        return False
    try:
        return all(
            current_cell is cell and current_cell.cell_contents is contents
            for current_cell, (cell, contents) in zip(closure or (), cells)
        )
    except ValueError:
        return False


def _controller_switch_callbacks_current():
    if (
        sys.modules.get(_unified_switch_module.__name__) is not _unified_switch_module
        or vars(_unified_switch_module).get("UnifiedMCPControlPlaneService")
        is not _unified_switch_class
        or UnifiedMCPControlPlaneService is not _unified_switch_class
        or vars(_unified_switch_module).get("_CONSOLE_CONTROLLER_SWITCH_METHODS")
        is not _unified_switch_methods
        or sys.modules.get(_permission_switch_module.__name__)
        is not _permission_switch_module
        or vars(_permission_switch_module).get("MCPPermissionStore")
        is not _permission_switch_class
        or MCPPermissionStore is not _permission_switch_class
        or vars(_permission_switch_module).get("_CONSOLE_CONTROLLER_PERMISSION_METHODS")
        is not _permission_switch_methods
    ):
        return False
    for name, descriptor, function, code, namespace, inputs in _unified_switch_methods:
        current = vars(_unified_switch_class).get(name)
        if not (
            current is descriptor
            and (type(current) is not property or current.fget is function)
            and type(function) is FunctionType
            and function.__code__ is code
            and function.__globals__ is namespace
            and namespace is vars(_unified_switch_module)
            and _controller_inputs_current(function, inputs)
        ):
            return False
    for name, wrapper, bindings in _permission_switch_methods:
        if (
            vars(_permission_switch_class).get(name) is not wrapper
            or len(bindings) != 2
        ):
            return False
        for function, code, namespace, defining_module, inputs in bindings:
            if not (
                type(function) is FunctionType
                and function.__code__ is code
                and function.__globals__ is namespace
                and namespace is vars(defining_module)
                and sys.modules.get(defining_module.__name__) is defining_module
                and _controller_inputs_current(function, inputs)
            ):
                return False
    return True


_CONTROLLER_INPUTS_CHECK = (
    _controller_inputs_current,
    _controller_inputs_current.__code__,
    _controller_inputs_current.__globals__,
)


def _controller_inputs_checker_current():
    original, code, namespace = _CONTROLLER_INPUTS_CHECK
    return (
        type(_controller_inputs_current) is FunctionType
        and _controller_inputs_current is original
        and original.__code__ is code
        and original.__globals__ is namespace
        and namespace is globals()
        and original.__defaults__ is None
        and original.__kwdefaults__ is None
        and original.__closure__ is None
    )


# TASK-34561: defining metadata for the fresh composition switch pipeline.
# This table is callback provenance only; each actual source read stays native.
_CONTROLLER_PRECHECK_MODULE = sys.modules[__name__]
_CONTROLLER_PRECHECK_CLASS = _CapturedSources
_CONTROLLER_PRECHECK_FUNCTIONS = tuple(
    (
        name,
        function,
        function.__code__,
        function.__globals__,
        _capture_controller_inputs(function),
    )
    for name, function in (
        ("_controller_inputs_current", _controller_inputs_current),
        ("_controller_inputs_checker_current", _controller_inputs_checker_current),
        ("_controller_switch_callbacks_current", _controller_switch_callbacks_current),
        ("_source_identity", _source_identity),
        ("_original_method", _original_method),
        ("_original_bound_method", _original_bound_method),
        ("_original_class_member", _original_class_member),
        ("_same_callback", _same_callback),
        ("standard_local_catalog_sources", standard_local_catalog_sources),
        ("standard_console_sources", standard_console_sources),
        ("standard_console_catalog_sources", standard_console_catalog_sources),
        ("_owned_worker", _owned_worker),
        ("_checked_read", _checked_read),
        (
            "_owned_permission_load_helpers_current",
            _owned_permission_load_helpers_current,
        ),
        ("read_console_kill_switch", read_console_kill_switch),
    )
)
_CONTROLLER_PRECHECK_METHODS = tuple(
    (
        name,
        function,
        function.__code__,
        function.__globals__,
        _capture_controller_inputs(function),
    )
    for name in (
        "__init__",
        "_method_bindings",
        "require_current",
        "permission_owner",
        "read_permission_payload",
        "permission_call",
    )
    for function in (vars(_CapturedSources)[name],)
)


def controller_precheck_pipeline_current() -> bool:
    """Check originals without resolving a path or reusing permission data."""
    if (
        sys.modules.get(__name__) is not _CONTROLLER_PRECHECK_MODULE
        or _CapturedSources is not _CONTROLLER_PRECHECK_CLASS
    ):
        return False
    for owner, bindings in (
        (vars(_CONTROLLER_PRECHECK_MODULE), _CONTROLLER_PRECHECK_FUNCTIONS),
        (vars(_CONTROLLER_PRECHECK_CLASS), _CONTROLLER_PRECHECK_METHODS),
    ):
        for name, original, code, namespace, inputs in bindings:
            current = owner.get(name)
            if not (
                type(current) is FunctionType
                and current is original
                and current.__code__ is code
                and current.__globals__ is namespace
                and namespace is vars(_CONTROLLER_PRECHECK_MODULE)
            ):
                return False
    if not _controller_inputs_checker_current():
        return False
    for bindings in (_CONTROLLER_PRECHECK_FUNCTIONS, _CONTROLLER_PRECHECK_METHODS):
        for _name, function, _code, _namespace, inputs in bindings:
            if not _controller_inputs_current(function, inputs):
                return False
    return _controller_switch_callbacks_current()


_CONTROLLER_PIPELINE_CHECK = (
    controller_precheck_pipeline_current,
    controller_precheck_pipeline_current.__code__,
    controller_precheck_pipeline_current.__globals__,
    _capture_controller_inputs(controller_precheck_pipeline_current),
)
