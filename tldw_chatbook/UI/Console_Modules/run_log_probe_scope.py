"""Optional finite scope for the original legacy run-log availability probe.

Only the default resolver route participates. Injected stores, authorities,
registry factories and callbacks retain their original direct lifetime.
"""

from __future__ import annotations

import inspect
import sys
from _thread import LockType, _local
from types import FunctionType, GetSetDescriptorType, ModuleType

_MISSING = object()


def _function_current(record: tuple) -> bool:
    function, code, defining, defaults, kwdefaults, items, closure, cells, wrapped = (
        record
    )
    try:
        return (
            type(function) is FunctionType
            and function.__code__ is code
            and function.__globals__ is defining
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is kwdefaults
            and len(function.__kwdefaults__ or {}) == len(items)
            and all(
                (function.__kwdefaults__ or {}).get(key) is value
                for key, value in items
            )
            and function.__closure__ is closure
            and all(cell.cell_contents is value for cell, value in cells)
            and vars(function).get("__wrapped__") is wrapped
        )
    except (AttributeError, ValueError):
        return False


def _source_current(module: ModuleType) -> bool:
    if type(module) is not ModuleType:
        return False
    defining, path, spec, origin, slots, records = module._RUN_LOG_PROBE_SOURCE
    return (
        type(module) is ModuleType
        and sys.modules.get(defining.get("__name__")) is module
        and vars(module) is defining
        and defining.get("__file__") == path
        and defining.get("__spec__") is spec
        and getattr(spec, "origin", None) == origin
        and all(
            owner.get(name, _MISSING) is value
            if type(owner) is dict  # noqa: E721 -- source namespace identity
            else inspect.getattr_static(owner, name, _MISSING) is value
            for owner, name, value in slots
        )
        and all(_function_current(record) for record in records)
    )


def _plain(
    receiver: object, owner: type, source: tuple, fields: tuple[str, ...]
) -> bool:
    return (
        type(receiver) is owner
        and inspect.getattr_static(owner, "__getattribute__") is object.__getattribute__
        and inspect.getattr_static(owner, "__getattr__", _MISSING) is _MISSING
        and type(inspect.getattr_static(owner, "__dict__", _MISSING))
        is GetSetDescriptorType
        and all(
            inspect.getattr_static(owner, name, _MISSING) is _MISSING for name in fields
        )
        and all(
            name not in vars(receiver)
            for recorded_owner, name, _ in source[4]
            if recorded_owner is owner
        )
    )


def _stock_default_probe(bridge: object) -> bool:
    """Decline custom dispatch before reading fields or invoking any callback."""
    bridge_source = sys.modules.get("tldw_chatbook.Chat.console_agent_bridge")
    if type(bridge_source) is not ModuleType:
        return False

    bridge_type = bridge_source.ConsoleAgentBridge
    if not _source_current(bridge_source) or not _plain(
        bridge,
        bridge_type,
        bridge_source._RUN_LOG_PROBE_SOURCE,
        ("_db", "_store", "_run_log_authorities", "_run_log_authority_lock"),
    ):
        return False
    fields = vars(bridge)
    authorities = fields.get("_run_log_authorities")
    if (
        fields.get("_store") is not None
        or type(authorities) is not dict  # noqa: E721 -- custom mapping declines
        or authorities
        or type(fields.get("_run_log_authority_lock")) is not LockType
    ):
        return False

    from tldw_chatbook import config
    from tldw_chatbook.Agents import run_log, run_log_paging
    from tldw_chatbook.DB import AgentRuns_DB, Workspace_DB
    from tldw_chatbook.Tools import file_operation_tools, workspace_file_roots
    from tldw_chatbook import Workspaces
    from tldw_chatbook.Workspaces import registry_service

    modules = (
        config,
        run_log,
        run_log_paging,
        AgentRuns_DB,
        Workspace_DB,
        file_operation_tools,
        workspace_file_roots,
        registry_service,
    )
    if not all(_source_current(module) for module in modules):
        return False
    if (
        Workspaces.LocalWorkspaceRegistryService
        is not registry_service.LocalWorkspaceRegistryService
    ):
        return False
    database = fields.get("_db")
    if not _plain(
        database,
        AgentRuns_DB.AgentRunsDB,
        AgentRuns_DB._RUN_LOG_PROBE_SOURCE,
        ("is_memory_db", "_thread_local"),
    ) or (
        vars(database).get("is_memory_db") is not False
        or type(vars(database).get("_thread_local")) is not _local
    ):
        return False
    registry = workspace_file_roots._default_registry_instance
    if registry is None:
        return True
    if not _plain(
        registry,
        registry_service.LocalWorkspaceRegistryService,
        registry_service._RUN_LOG_PROBE_SOURCE,
        ("db",),
    ):
        return False
    workspace = vars(registry).get("db")
    return (
        _plain(
            workspace,
            Workspace_DB.WorkspaceDB,
            Workspace_DB._RUN_LOG_PROBE_SOURCE,
            ("is_memory_db", "_thread_local"),
        )
        and vars(workspace).get("is_memory_db") is False
        and type(vars(workspace).get("_thread_local")) is _local
    )


def finite_stock_probe(bridge: object, read):
    """Run the caller's complete synchronous read before retiring new caches."""
    try:
        stock = _stock_default_probe(bridge)
    except (AttributeError, ImportError, TypeError, ValueError):
        stock = False
    if not stock:
        return read()
    from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker

    return run_finite_local_worker(read)
