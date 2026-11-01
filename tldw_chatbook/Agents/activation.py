"""Recovery admission for installed agent runs and their native workers."""

import inspect
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import FunctionType, ModuleType

from tldw_chatbook.Backup_Recovery import bootstrap, profile_paths
from tldw_chatbook.Backup_Recovery.admission_runtime import RecoveryAdmissionGuard


class AgentActivationRequired(PermissionError):
    """Local review is required before starting or resuming this agent."""

    def __init__(self):
        super().__init__("agent_activation_required")


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


_guard = RecoveryAdmissionGuard(
    "agent",
    error=AgentActivationRequired,
    sources=_sources,
    # Agent runs admit both the selected profile and each source owner.
    owners=lambda owner: ("config", owner),
)


def execution(service=None, *, sources=()):
    """Retain admission through final persistence in this native task/thread."""
    from .scratch_admission import execution as scratch_execution

    return scratch_execution(_guard, service, sources)


def _explicit_writer_root(service):
    """Select only the unchanged writer that stock run_turn will actually bind."""
    service_source = sys.modules.get("tldw_chatbook.Agents.agent_service")
    log_source = sys.modules.get("tldw_chatbook.Agents.run_log")
    if type(service_source) is not ModuleType or type(log_source) is not ModuleType:
        return None
    AgentService = vars(service_source).get("AgentService")
    record = vars(log_source).get("_RUN_LOG_PATH_WRITER_ORIGINAL")
    if type(record) is not tuple or len(record) != 2:
        return None
    writer_class, methods = record
    if (
        type(AgentService) is not type
        or type(writer_class) is not type
        or type(methods) is not tuple
        or len(methods) != 3
        or not all(
            type(row) is tuple
            and len(row) == 4
            and type(row[0]) is str
            and type(row[1]) is FunctionType
            for row in methods
        )
        or type(service) is not AgentService
        or inspect.getattr_static(AgentService, "__getattribute__")
        is not object.__getattribute__
        or inspect.getattr_static(writer_class, "__getattribute__")
        is not object.__getattribute__
    ):
        return None
    writer = inspect.getattr_static(service, "_injected_run_log_writer", None)
    if type(writer) is not writer_class or not all(
        inspect.getattr_static(writer, name, None) is function
        and function.__code__ is code
        and function.__defaults__ is defaults
        for name, function, code, defaults in methods
    ):
        return None
    root = inspect.getattr_static(writer, "_explicit_root", None)
    return root if type(root) is type(Path()) else None


@contextmanager
def history_execution(service: object) -> Iterator[Path | None]:
    """Admit saved history at selection and retain it through segment loading."""
    from .run_log import resolve_log_root

    with execution(service):
        root = resolve_log_root()
        sources = (("agents.history", root),) if root is not None else ()
        with execution(service, sources=sources):
            yield root


def standard_history_resolver(callback: Callable[..., Path | None]) -> bool:
    """Recognize the recorded reader before supplying its optional root argument."""
    from .run_log import _RUN_LOG_PROBE_SOURCE

    original = next(
        function
        for _owner, name, function in _RUN_LOG_PROBE_SOURCE[4]
        if name == "resolve_existing_log_dir"
    )
    if callback is not original:
        return False
    record = next(row for row in _RUN_LOG_PROBE_SOURCE[5] if row[0] is original)
    _, code, namespace, defaults, kwdefaults, items, closure, cells, wrapped = record
    return (
        callback.__code__ is code
        and callback.__globals__ is namespace
        and callback.__defaults__ is defaults
        and callback.__kwdefaults__ is kwdefaults
        and len(callback.__kwdefaults__ or {}) == len(items)
        and all(
            (callback.__kwdefaults__ or {}).get(key) is value for key, value in items
        )
        and callback.__closure__ is closure
        and all(cell.cell_contents is value for cell, value in cells)
        and vars(callback).get("__wrapped__") is wrapped
    )


def guarded(function):
    """Guard synchronous installed run entry points before their first effect."""

    original_code = function.__code__

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
        # Passive stock-source observation joins the initial fresh admission;
        # custom writers still resolve global history inside the admitted scope.
        explicit_root = (
            _explicit_writer_root(self)
            if function.__name__ == "run_turn" and function.__code__ is original_code
            else None
        )
        if explicit_root is not None:
            sources += (("agents.history", explicit_root),)
        with execution(self, sources=sources):
            if function.__name__ == "run_turn" and explicit_root is None:
                from .run_log import resolve_log_root

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
    from .scratch_admission import capture_worker, worker_entry

    sources = _guard.captured_sources(service)
    handoff = capture_worker()

    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            with worker_entry(handoff), _guard.worker_isolation():
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

        return call

    return decorate


def async_worker_guard(service=None):
    """Admit the Console provider coroutine on its actual model-call loop."""
    from .scratch_admission import capture_worker, worker_entry

    sources = _guard.captured_sources(service)
    handoff = capture_worker()

    def decorate(function):
        @wraps(function)
        async def call(*args, **kwargs):
            with worker_entry(handoff), _guard.worker_isolation():
                with execution(service, sources=sources):
                    return await function(*args, **kwargs)

        return call

    return decorate
