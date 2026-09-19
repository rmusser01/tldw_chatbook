"""Recovery admission for installed agent runs and their native workers."""

import asyncio
import os
import sys
import threading
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from tldw_chatbook.Backup_Recovery import bootstrap, profile_paths

_active = ContextVar("agent_execution", default=None)


class AgentActivationRequired(PermissionError):
    """Local review is required before starting or resuming this agent."""

    def __init__(self):
        super().__init__("agent_activation_required")


def _identity():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), threading.get_ident(), task


def _sources(service):
    """Observe installed selectors without constructing stores or creating roots."""
    selector = bootstrap.effective_config_path()
    config = sys.modules.get("tldw_chatbook.config")
    cached = getattr(config, "_CONFIG_CACHE", None)
    if getattr(config, "_CONFIG_CACHE_SOURCE", None) != selector:
        cached = None
    data = profile_paths.user_data_dir(cached or {})
    paths = [("config", selector)]
    db = getattr(service, "db", getattr(service, "_db", None))
    db_path = getattr(db, "db_path", None)
    # In-memory runs still consume the selected agent configuration.
    paths.append(
        (
            "db.agent_runs",
            db_path if db_path and not getattr(db, "is_memory_db", False) else selector,
        )
    )
    store = getattr(service, "_store", None)
    persistence = getattr(store, "persistence", None)
    history_db = getattr(persistence, "db", None)
    if history_db is not None and not getattr(history_db, "is_memory_db", False):
        paths.append(("db.chachanotes.primary", history_db.db_path))
    roots_module = sys.modules.get("tldw_chatbook.Tools.workspace_file_roots")
    registry = getattr(roots_module, "_default_registry_instance", None)
    workspace_db = getattr(registry, "db", None)
    workspace_path = getattr(workspace_db, "db_path", None)
    if workspace_path is None:
        workspace_path = profile_paths.database_path(cached or {}, "workspaces_db_path")
    paths.append(("db.workspaces", workspace_path))
    sandbox = profile_paths.setting(cached or {}, "tools", "file_sandbox_root")
    paths.append(("agents.history", sandbox or data / "tool_sandbox"))
    writer = getattr(service, "run_log_writer", None)
    log_dir = getattr(writer, "log_dir", None)
    if log_dir is not None:
        paths.append(("agents.history", log_dir))
    catalog = getattr(service, "registry", getattr(service, "_registry", None))
    for provider in getattr(catalog, "_providers", ()):
        paths.extend(_permission_sources(provider))
        paths.extend(_permission_sources(getattr(provider, "_gate", None)))
    return tuple((owner, profile_paths.lexical_path(path)) for owner, path in paths)


def _permission_sources(consumer):
    """Observe the actual Console/MCP gate store without loading imported rules."""
    plane = getattr(consumer, "_service", None)
    store = getattr(plane, "_permission_store", None)
    path = getattr(store, "path", None)
    if path is None:
        local = getattr(plane, "local_service", None)
        local_path = getattr(getattr(local, "store", None), "path", None)
        if local_path is not None:
            path = Path(local_path).with_name("mcp_permissions.json")
    return (
        (("mcp.permissions", profile_paths.lexical_path(path)),)
        if path is not None
        else ()
    )


@contextmanager
def execution(service=None, *, sources=()):
    """Retain admission through final persistence in this native task/thread."""
    from tldw_chatbook.Backup_Recovery.activation import execution_scope
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

    identity = _identity()
    active = _active.get()
    if active is not None and active[0] != identity:
        raise AgentActivationRequired()
    leases = dict(active[1]) if active else {}
    observed = tuple(
        dict.fromkeys(
            (active[2] if active else ()) + tuple(sources) + _sources(service)
        )
    )
    with ExitStack() as stack:
        try:
            for owner, path in observed:
                if path not in leases:
                    leases[path] = acquire_storage(path)
                    stack.callback(leases[path].close)
                if not stack.enter_context(
                    execution_scope(("config", owner), path, retained=leases[path])
                ):
                    raise AgentActivationRequired()
        except (OSError, ValueError, TypeError, RuntimeError, AttributeError):
            raise AgentActivationRequired() from None
        token = _active.set((identity, leases, observed))
        try:
            yield
        finally:
            _active.reset(token)


def guarded(function):
    """Guard synchronous installed run entry points before their first effect."""

    @wraps(function)
    def call(self, *args, **kwargs):
        sources = tuple(
            source
            for name in ("builtin_gate", "mcp_provider")
            for source in _permission_sources(kwargs.get(name))
        )
        log_dir = getattr(kwargs.get("run_log_writer"), "log_dir", None)
        if log_dir is not None:
            sources += (("agents.history", profile_paths.lexical_path(log_dir)),)
        with execution(self, sources=sources):
            if function.__name__ == "run_turn":
                from .run_log import resolve_log_root

                # This installed selector consults the admitted workspace DB.
                # Check the actual workspace log source before writer.bind can
                # migrate logs or before any model/tool consumes saved history.
                root = resolve_log_root()
                if root is not None:
                    with execution(self, sources=(("agents.history", root),)):
                        return function(self, *args, **kwargs)
            return function(self, *args, **kwargs)

    return call


def _worker_databases(service):
    """Only installed thread-local databases reached by these native workers."""
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    roots = sys.modules.get("tldw_chatbook.Tools.workspace_file_roots")
    registry = getattr(roots, "_default_registry_instance", None)
    return tuple(
        db
        for db, expected in (
            (getattr(service, "db", getattr(service, "_db", None)), AgentRunsDB),
            (getattr(registry, "db", None), WorkspaceDB),
        )
        if type(db) is expected and not db.is_memory_db
    )


def worker_guard(service=None):
    """Capture sources, then acquire independently inside the actual worker."""
    active = _active.get()
    sources = (
        active[2]
        if active is not None and active[0] == _identity()
        else _sources(service)
    )

    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            token = _active.set(None)
            try:
                with execution(service, sources=sources):
                    existing = {
                        db
                        for db in _worker_databases(service)
                        if getattr(db._thread_local, "conn", None) is not None
                    }
                    try:
                        return function(*args, **kwargs)
                    finally:
                        # These fresh native workers own their new thread-local
                        # handles. Existing callers' handles stay caller-owned.
                        # Failed close keeps the DB's actual native lease live.
                        for db in _worker_databases(service):
                            if db not in existing:
                                db.close()
            finally:
                _active.reset(token)

        return call

    return decorate


def async_worker_guard(service=None):
    """Admit the Console provider coroutine on its actual model-call loop."""
    active = _active.get()
    sources = (
        active[2]
        if active is not None and active[0] == _identity()
        else _sources(service)
    )

    def decorate(function):
        @wraps(function)
        async def call(*args, **kwargs):
            token = _active.set(None)
            try:
                with execution(service, sources=sources):
                    return await function(*args, **kwargs)
            finally:
                _active.reset(token)

        return call

    return decorate
