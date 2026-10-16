from typing import (  # noqa: UP035 - preserve original public annotation objects
    Any,
    Callable,
    NamedTuple,
)

from . import app_service_wiring as _source_wiring


class _ConsoleSkillTrustSource(NamedTuple):
    """One stock service proof; no UI owner or permission verdict is captured."""

    app: Any
    scope: Any
    local: Any
    source_current: Callable[[], bool]
    current: Callable[[], bool]
    loop: Any
    thread: Any
    runtime_reads: set


def _disabled_builtin_skills(config: Any) -> frozenset[str]:
    """Built-in skills disabled in config; imported lazily (off boot path, ADR-097)."""
    from tldw_chatbook.Skills_Interop.builtin_skills import (
        disabled_builtins_from_config,
    )

    return disabled_builtins_from_config(config)


def _read_app_disabled_builtin_skills(app: object) -> frozenset[str]:
    """Read the current stock app config without invoking a UI callback."""
    return _source_wiring._disabled_builtin_skills(getattr(app, "app_config", None))


def _read_app_published_plugin_service(app: object) -> Any:
    """Return only the already resident plugin metadata owner."""
    return getattr(app, "_plugin_service", None)


def _console_skill_metadata_current(metadata):
    """Check the three reused proof helpers before invoking their bodies."""
    import sys
    from importlib.machinery import ModuleSpec
    from types import FunctionType, MappingProxyType, ModuleType

    if type(metadata) is not ModuleType:
        return False
    if sys.modules.get(__package__ + ".Widgets.compact_model_bar") is not metadata:
        return False
    namespace = vars(metadata)
    source = namespace.get("_WIDGET_SOURCE")
    if type(source) is not tuple or len(source) != 6 or type(source[5]) is not tuple:
        return False
    defining, path, spec, origin, bindings, _records = source
    if (
        defining is not namespace
        or type(path) is not str  # exact source metadata
        or type(origin) is not str  # exact plain metadata; no custom dispatch
        or type(namespace.get("__file__")) is not str  # exact source metadata
        or namespace.get("__file__") != path  # exact plain metadata; no custom dispatch
        or namespace.get("__spec__") is not spec
        or type(spec) is not ModuleSpec
        or type(spec.origin) is not str  # exact source metadata
        or spec.origin != origin
        or origin != path  # exact plain metadata; no custom dispatch
        or type(bindings) is not tuple
    ):
        return False
    for row in bindings:
        if (
            type(row) is not tuple
            or len(row) != 3
            or type(row[1]) is not str  # exact plain metadata; no custom dispatch
            or (type(row[0]) is not dict and type(row[0]) is not MappingProxyType)  # exact plain metadata; no custom dispatch
            or row[0].get(row[1]) is not row[2]
        ):
            return False
    names = {"_plain_fields", "_source_current", "_function_current"}
    records = tuple(
        row
        for row in source[5]
        if type(row) is tuple
        and len(row) == 11
        and row[0] is namespace
        and type(row[1]) is str  # plain capsule key
        and row[1] in names
    )
    return len(records) == 3 and all(
        namespace.get(row[1]) is row[2]
        and type(row[2]) is FunctionType
        and row[2].__code__ is row[3]
        and row[2].__globals__ is row[4] is namespace
        and row[2].__defaults__ is None
        and row[2].__kwdefaults__ is None
        and row[2].__closure__ is None
        for row in records
    )


def _console_skill_source_current(metadata, source):
    """Reuse the existing finite source capsule checker for this stock route."""
    if type(source) is not tuple or len(source) != 6:
        return False
    from importlib.machinery import ModuleSpec

    namespace, path, spec, origin, bindings, records = source
    if type(namespace) is not dict or type(path) is not str or type(origin) is not str:  # exact stock metadata; no custom dispatch
        return False
    if (
        type(spec) is not ModuleSpec
        or type(spec.origin) is not str  # exact stock metadata; no custom dispatch
        or type(namespace.get("__name__")) is not str  # exact stock metadata; no custom dispatch
        or type(namespace.get("__file__")) is not str  # exact stock metadata; no custom dispatch
        or namespace.get("__file__") != path
        or spec.origin != origin
        or path != origin
    ):
        return False
    if type(bindings) is not tuple or type(records) is not tuple:
        return False
    # Reject foreign names/containers before the existing checker hashes keys.
    for row in bindings:
        if type(row) is not tuple or len(row) != 3 or type(row[1]) is not str:  # exact stock metadata; no custom dispatch
            return False
        if type(row[0]) is not dict and type(row[0]) is not _source_wiring._SkillMappingProxyType:  # exact stock metadata; no custom dispatch
            return False
    for row in records:
        if type(row) is not tuple or len(row) != 11 or type(row[1]) is not str:  # exact stock metadata; no custom dispatch
            return False
        if type(row[0]) is not dict and type(row[0]) is not _source_wiring._SkillMappingProxyType:  # exact stock metadata; no custom dispatch
            return False
        if type(row[2]) is not _source_wiring._SkillFunctionType:
            return False
        defaults = row[2].__kwdefaults__
        if defaults is not None and (
            type(defaults) is not dict or any(type(key) is not str for key in defaults)  # exact stock metadata; no custom dispatch
        ):
            return False
    return metadata._source_current(source)


def _capture_console_skill_trust_source(app, service):
    """Capture resident service sources without a UI scheduling dependency."""
    _source_wiring = globals()["_source_wiring"]
    import inspect
    import sys
    from types import FunctionType, ModuleType

    metadata = sys.modules.get("tldw_chatbook.Widgets.compact_model_bar")
    if not _source_wiring._console_skill_metadata_current(metadata):
        return None
    app_module = sys.modules.get("tldw_chatbook.app")
    scope_module = sys.modules.get("tldw_chatbook.Skills_Interop.skills_scope_service")
    local_module = sys.modules.get("tldw_chatbook.Skills_Interop.local_skills_service")
    config_module = sys.modules.get("tldw_chatbook.config")
    if any(
        type(module) is not ModuleType
        for module in (app_module, scope_module, local_module, config_module)
    ):
        return None
    app_namespace = vars(app_module)
    app_record = app_namespace.get("_CONSOLE_SKILL_APP_SOURCE")
    if type(app_record) is not tuple or len(app_record) != 5:
        return None
    app_type, defining, factory, factory_code, runtime_type = app_record
    if (
        defining is not app_namespace
        or type(app) is not app_type
        or app_namespace.get("TldwCli") is not app_type
        or type(factory) is not FunctionType
        or factory.__globals__ is not defining
        or factory.__code__ is not factory_code
        or factory.__defaults__ is not None
        or factory.__kwdefaults__ is not None
        or factory.__closure__ is not None
        or inspect.getattr_static(app_type, "_create_deferred_startup_task", None)
        is not factory
    ):
        return None
    source = _source_wiring._CONSOLE_SKILL_WIRING_SOURCE
    scope_source = vars(scope_module).get("_CONSOLE_SKILL_CONTEXT_SOURCE")
    local_source = vars(local_module).get("_CONSOLE_SKILL_CONTEXT_SOURCE")
    config_source = vars(config_module).get("_COMPACT_MODEL_CONFIG_SOURCE")
    if source is not vars(_source_wiring).get("_CONSOLE_SKILL_CONTEXT_SOURCE"):
        return None
    sources = (source, _source_wiring._CONSOLE_SKILL_HELPER_SOURCE, scope_source, local_source, config_source)
    if not all(_source_wiring._console_skill_source_current(metadata, row) for row in sources):
        return None
    app_slots = (
        "_build_local_skill_trust_service",
        "_build_local_skills_stack",
        "ensure_local_skill_trust_service",
        "local_skill_trust_service",
        "local_skills_service",
        "skills_scope_service",
    )
    if any(
        inspect.getattr_static(app_type, name, None)
        is not (_source_wiring._skill_wiring.SkillServiceWiringMixin if name in ("_build_local_skill_trust_service", "_build_local_skills_stack", "ensure_local_skill_trust_service") else _source_wiring.ServiceWiringMixin).__dict__[name]
        for name in app_slots
    ):
        return None
    scope_type = vars(scope_module).get("SkillsScopeService")
    local_type = vars(local_module).get("LocalSkillsService")
    app_fields = metadata._plain_fields(
        app,
        app_type,
        (
            "_local_skill_trust_service",
            "_local_skill_trust_service_build_lock",
            "_local_skills_service",
            "_skills_scope_service",
            "_local_skills_stack_inputs",
            "app_config",
            "_shutting_down",
            "_exit",
            "_workers",
            "_thread_id",
            "console_runtime",
            "_console_runtime_shutdown_task",
        ),
    )
    scope_fields = metadata._plain_fields(
        service, scope_type, ("local_service", "server_service", "policy_enforcer")
    )
    if app_fields is None or scope_fields is None:
        return None
    runtime = app_fields.get("console_runtime")
    if (
        runtime is None
        or app_fields.get("_console_runtime_shutdown_task") is not None
        or app_fields.get("_shutting_down") is not False
        or app_fields.get("_exit") is not False
    ):
        raise RuntimeError("console_skill_trust_runtime_closed")
    runtime_fields = metadata._plain_fields(
        runtime, runtime_type, ("_disposed", "_preparation_reads", "_app")
    )
    if (
        runtime_fields is None
        or type(runtime_fields.get("_preparation_reads")) is not set  # exact source metadata
    ):  # exact observer ownership
        return None
    runtime_reads = runtime_fields["_preparation_reads"]

    def runtime_current():
        return (
            app_namespace.get("ConsoleRuntime") is runtime_type
            and metadata._plain_fields(
                runtime, runtime_type, ("_disposed", "_preparation_reads", "_app")
            )
            is runtime_fields
            and app_fields.get("console_runtime") is runtime
            and app_fields.get("_console_runtime_shutdown_task") is None
            and app_fields.get("_shutting_down") is False
            and app_fields.get("_exit") is False
            and runtime_fields.get("_app") is app
            and runtime_fields.get("_disposed") is False
            and runtime_fields.get("_preparation_reads") is runtime_reads
        )

    # An original closing Runtime is a refusal, not eligibility for direct IO.
    if not runtime_current():
        raise RuntimeError("console_skill_trust_runtime_closed")
    local = scope_fields.get("local_service")
    local_fields = metadata._plain_fields(
        local,
        local_type,
        (
            "_trust_service",
            "_trust_service_factory",
            "store_dir",
            "skills_dir",
            "policy_enforcer",
        ),
    )
    if local_fields is None:
        return None
    if (
        any(name in app_fields for name in app_slots)
        or any(name in scope_fields for name in _source_wiring._CONSOLE_SKILL_SCOPE_SLOTS)
        or any(name in local_fields for name in _source_wiring._CONSOLE_SKILL_LOCAL_SLOTS)
    ):
        return None
    if (
        app_fields.get("_skills_scope_service") is not service
        or app_fields.get("_local_skills_service") is not local
        or app_fields.get("_shutting_down") is not False
        or app_fields.get("_exit") is not False
        or type(app_fields.get("app_config")) is not dict  # exact stock metadata; no custom dispatch
    ):
        return None
    lazy = local_fields.get("_trust_service_factory")
    if (
        type(lazy) is not FunctionType
        or lazy.__code__ is not _source_wiring._CONSOLE_SKILL_TRUST_FACTORY_CODE
        or lazy.__globals__ is not vars(_source_wiring._skill_wiring)
        or lazy.__defaults__ is not None
        or lazy.__kwdefaults__ is not None
        or lazy.__code__.co_freevars != ("self",)
        or type(lazy.__closure__) is not tuple
        or len(lazy.__closure__) != 1
        or lazy.__closure__[0].cell_contents is not app
    ):
        return None
    lazy_closure = lazy.__closure__
    # Preserve the original captured policy collaborators; no verdict is reused.
    inputs = app_fields.get("_local_skills_stack_inputs")
    if type(inputs) is not tuple or len(inputs) != 2:
        return None
    if (
        scope_fields.get("policy_enforcer") is not inputs[0]
        or local_fields.get("policy_enforcer") is not inputs[0]
        or scope_fields.get("server_service") is not inputs[1]
    ):
        return None
    mapping = app_fields["app_config"]
    lock = app_fields["_local_skill_trust_service_build_lock"]
    if type(lock) is not _source_wiring.asyncio.Lock:
        return None
    store_dir, skills_dir = (
        local_fields.get("store_dir"),
        local_fields.get("skills_dir"),
    )
    identity = config_module.current_config_identity()
    cache = vars(config_module).get("_SETTINGS_CACHE")
    posture = vars(config_module).get("_SETTINGS_CACHE_POSTURE")
    selector = tuple(
        _source_wiring.os.environ.get(key)
        for key in (
            "TLDW_CONFIG_PATH",
            "HOME",
            "USERPROFILE",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        )
    )
    loop, thread = _source_wiring.asyncio.get_running_loop(), _source_wiring.threading.current_thread()
    if (
        type(app_fields.get("_thread_id")) is not int  # exact stock metadata; no custom dispatch
        or app_fields.get("_thread_id") != thread.ident
    ):
        return None

    metadata_check = _source_wiring._console_skill_metadata_current
    source_check = _source_wiring._console_skill_source_current
    support = tuple(
        row for row in source[5] if row[2] is metadata_check or row[2] is source_check
    )
    if len(support) != 2 or any(
        row[5] is not None or row[6] is not None or row[8] is not None
        for row in support
    ):
        return None

    def source_current():
        if sources[1][0].get("_source_wiring") is not _source_wiring:
            return False
        # Check captured helper records without invoking either helper first.
        # Bound references alone do not reject an in-place body replacement.
        for row in support:
            namespace, name, function = row[:3]
            if (
                namespace.get(name) is not function
                or type(function) is not FunctionType
                or function.__code__ is not row[3]
                or function.__globals__ is not row[4]
                or function.__globals__.get("_source_wiring") is not _source_wiring
                or function.__defaults__ is not None
                or function.__kwdefaults__ is not None
                or function.__closure__ is not None
                or type(vars(function)) is not dict  # plain defining metadata
                or vars(function).get("__wrapped__") is not row[10]
            ):
                return False
        if not metadata_check(metadata):
            return False
        if not all(source_check(metadata, row) for row in sources):
            return False
        # Captured plain fields only: no Textual accessor or cached authority.
        if not (
            runtime_current()
            and sys.modules.get("tldw_chatbook.app") is app_module
            and app_namespace.get("_CONSOLE_SKILL_APP_SOURCE") is app_record
            and app_namespace.get("TldwCli") is app_type
            and factory.__code__ is factory_code
            and factory.__defaults__ is None
            and factory.__kwdefaults__ is None
            and factory.__closure__ is None
            and inspect.getattr_static(app_type, "_create_deferred_startup_task", None)
            is factory
            and config_module.current_config_identity() == identity
            and vars(config_module).get("_SETTINGS_CACHE") is cache
            and vars(config_module).get("_SETTINGS_CACHE_POSTURE") is posture
            and tuple(
                _source_wiring.os.environ.get(key)
                for key in (
                    "TLDW_CONFIG_PATH",
                    "HOME",
                    "USERPROFILE",
                    "XDG_CONFIG_HOME",
                    "XDG_DATA_HOME",
                )
            )
            == selector
        ):
            return False
        return (
            metadata._plain_fields(
                app,
                app_type,
                (
                    "_local_skill_trust_service",
                    "_local_skill_trust_service_build_lock",
                    "_local_skills_service",
                    "_skills_scope_service",
                    "_local_skills_stack_inputs",
                    "app_config",
                    "_shutting_down",
                    "_exit",
                    "_workers",
                    "_thread_id",
                    "console_runtime",
                    "_console_runtime_shutdown_task",
                ),
            )
            is app_fields
            and metadata._plain_fields(
                service,
                scope_type,
                ("local_service", "server_service", "policy_enforcer"),
            )
            is scope_fields
            and metadata._plain_fields(
                local,
                local_type,
                (
                    "_trust_service",
                    "_trust_service_factory",
                    "store_dir",
                    "skills_dir",
                    "policy_enforcer",
                ),
            )
            is local_fields
            and app_fields.get("_skills_scope_service") is service
            and app_fields.get("_local_skills_service") is local
            and app_fields.get("_local_skill_trust_service_build_lock") is lock
            and app_fields.get("app_config") is mapping
            and type(app_fields.get("_thread_id")) is int  # original exact source field
            and app_fields.get("_thread_id") == thread.ident
            and app_fields.get("_shutting_down") is False
            and app_fields.get("_exit") is False
            and all(
                inspect.getattr_static(app_type, name, None)
                is (_source_wiring._skill_wiring.SkillServiceWiringMixin if name in ("_build_local_skill_trust_service", "_build_local_skills_stack", "ensure_local_skill_trust_service") else _source_wiring.ServiceWiringMixin).__dict__[name]
                for name in app_slots
            )
            and not any(name in app_fields for name in app_slots)
            and local_fields.get("_trust_service_factory") is lazy
            and lazy.__code__ is _source_wiring._CONSOLE_SKILL_TRUST_FACTORY_CODE
            and lazy.__globals__ is vars(_source_wiring._skill_wiring)
            and lazy.__defaults__ is None
            and lazy.__kwdefaults__ is None
            and lazy.__closure__ is lazy_closure
            and lazy_closure[0].cell_contents is app
            and local_fields.get("store_dir") is store_dir
            and local_fields.get("skills_dir") is skills_dir
            and not any(name in scope_fields for name in _source_wiring._CONSOLE_SKILL_SCOPE_SLOTS)
            and not any(name in local_fields for name in _source_wiring._CONSOLE_SKILL_LOCAL_SLOTS)
            and scope_fields.get("server_service") is inputs[1]
            and app_fields.get("_local_skills_stack_inputs") is inputs
            and scope_fields.get("local_service") is local
            and scope_fields.get("policy_enforcer") is inputs[0]
            and local_fields.get("policy_enforcer") is inputs[0]
        )

    def current():
        if sources[1][0].get("_source_wiring") is not _source_wiring:
            return False
        try:
            return (
                _source_wiring.asyncio.get_running_loop() is loop
                and _source_wiring.threading.current_thread() is thread
                and source_current()
            )
        except (AttributeError, TypeError, ValueError):
            return False

    if not current():
        return None
    return _source_wiring._ConsoleSkillTrustSource(
        app, service, local, source_current, current, loop, thread, runtime_reads
    )


def _capture_console_skill_trust_preparation(
    app, service, owner_current, controller_source
):
    """Add the original skill-discovery App-worker scheduling contract."""
    _source_wiring = globals()["_source_wiring"]
    import inspect
    import sys

    source = _source_wiring._capture_console_skill_trust_source(app, service)
    if source is None:
        return None
    app_fields = vars(app)
    if (
        app_fields.get("_local_skill_trust_service") is not None
        or vars(source.local).get("_trust_service") is not None
    ):
        return None
    metadata = sys.modules.get("tldw_chatbook.Widgets.compact_model_bar")
    if not _source_wiring._console_skill_source_current(metadata, controller_source):
        return None
    manager = app_fields.get("_workers")
    manager_fields = metadata._plain_fields(
        manager, _source_wiring._SkillWorkerManager, ("_app", "_workers")
    )
    if manager_fields is None:
        return None
    manager_workers = manager_fields.get("_workers")

    def source_current():
        return source.source_current() and _source_wiring._console_skill_source_current(
            metadata, controller_source
        )

    def current(expected_trust=None):
        try:
            return (
                source.current()
                and source_current()
                and owner_current() is True
                and source.loop.get_task_factory() is None
                and app_fields.get("_local_skill_trust_service") is expected_trust
                and app_fields.get("_workers") is manager
                and metadata._plain_fields(
                    manager, _source_wiring._SkillWorkerManager, ("_app", "_workers")
                )
                is manager_fields
                and manager_fields.get("_app") is app
                and type(manager_workers) is set  # original exact worker owner
                and manager_fields.get("_workers") is manager_workers
                and not any(
                    name in manager_fields for name in ("_new_worker", "add_worker")
                )
                and "workers" not in app_fields
                and "app" not in app_fields
                and inspect.getattr_static(type(app), "workers", None)
                is _source_wiring._CONSOLE_SKILL_WORKERS
                and inspect.getattr_static(type(app), "app", None) is _source_wiring._CONSOLE_SKILL_APP
                and _source_wiring._CONSOLE_SKILL_APP_GET(app) is app
                and "run_worker" not in app_fields
                and inspect.getattr_static(type(app), "run_worker", None)
                is _source_wiring._CONSOLE_SKILL_RUN_WORKER
            )
        except (AttributeError, TypeError, ValueError):
            return False

    if not current():
        return None
    return (
        app,
        source_current,
        current,
        (manager, manager_fields, source.loop, source.runtime_reads),
    )


async def _prepare_console_skill_trust_service(captured):
    """Retain a stock preparation in the App's selected worker drain scope."""
    _source_wiring = globals()["_source_wiring"]
    app, source_current, current, custody = captured
    if not current():
        raise RuntimeError("console_skill_trust_owner_changed")
    # Keep the original scope gate ahead of optional lazy trust construction.
    vars(app)["_skills_scope_service"]._enforce_policy("skills.context.list.local")
    if not current():
        raise RuntimeError("console_skill_trust_owner_changed")
    preparation = _source_wiring.ServiceWiringMixin.ensure_local_skill_trust_service(
        app,
        _source_current=source_current,
        _owner_current=current,
        _read_observers=(custody[3],),
    )
    try:
        worker = _source_wiring._CONSOLE_SKILL_RUN_WORKER(
            app,
            preparation,
            group="console-skill-trust-setup",
            exclusive=False,
            exit_on_error=False,
        )
    except BaseException:
        preparation.close()
        raise
    from textual.worker import WorkerCancelled, WorkerFailed

    if type(worker) is not _source_wiring.Worker:
        raise RuntimeError("console_skill_trust_worker_changed")
    manager, manager_fields, loop, _runtime_reads = custody
    values = vars(worker)
    task = values.get("_task")
    if (
        vars(app).get("_workers") is not manager
        or vars(manager) is not manager_fields
        or manager_fields.get("_app") is not app
        or values.get("_node") is not app
        or values.get("_work") is not preparation
        or type(task) is not _source_wiring.asyncio.Task
        or task.get_loop() is not loop
        or not any(item is worker for item in manager_fields["_workers"])
    ):
        raise RuntimeError("console_skill_trust_worker_changed")
    try:
        value = await _source_wiring._CONSOLE_SKILL_WAIT(worker)
    except WorkerCancelled:
        error = vars(worker).get("_error")
        if isinstance(error, _source_wiring.asyncio.CancelledError):
            raise error
        raise _source_wiring.asyncio.CancelledError
    except WorkerFailed:
        error = vars(worker).get("_error")
        if isinstance(error, BaseException):
            raise error
        raise
    if not current(value):
        raise RuntimeError("console_skill_trust_owner_changed")










def _install_skill_source(_source_wiring):
    from textual.app import App as _SkillAppBase
    vars(_source_wiring).update({'_SkillAppBase': _SkillAppBase})
    from textual.dom import DOMNode as _SkillMessagePump
    vars(_source_wiring).update({'_SkillMessagePump': _SkillMessagePump})
    from textual.message_pump import MessagePump as _SkillAppAccessor
    vars(_source_wiring).update({'_SkillAppAccessor': _SkillAppAccessor})
    from textual.worker_manager import WorkerManager as _SkillWorkerManager
    vars(_source_wiring).update({'_SkillWorkerManager': _SkillWorkerManager})

    _CONSOLE_SKILL_RUN_WORKER = _SkillMessagePump.run_worker
    vars(_source_wiring).update({'_CONSOLE_SKILL_RUN_WORKER': _CONSOLE_SKILL_RUN_WORKER})
    _CONSOLE_SKILL_WAIT = _source_wiring.Worker.wait
    vars(_source_wiring).update({'_CONSOLE_SKILL_WAIT': _CONSOLE_SKILL_WAIT})
    _CONSOLE_SKILL_WORKERS = _SkillAppBase.__dict__["workers"]
    vars(_source_wiring).update({'_CONSOLE_SKILL_WORKERS': _CONSOLE_SKILL_WORKERS})
    _CONSOLE_SKILL_WORKERS_GET = _CONSOLE_SKILL_WORKERS.fget
    vars(_source_wiring).update({'_CONSOLE_SKILL_WORKERS_GET': _CONSOLE_SKILL_WORKERS_GET})
    _CONSOLE_SKILL_APP = _SkillAppAccessor.__dict__["app"]
    vars(_source_wiring).update({'_CONSOLE_SKILL_APP': _CONSOLE_SKILL_APP})
    _CONSOLE_SKILL_APP_GET = _CONSOLE_SKILL_APP.fget
    vars(_source_wiring).update({'_CONSOLE_SKILL_APP_GET': _CONSOLE_SKILL_APP_GET})
    _CONSOLE_SKILL_NEW_WORKER = _SkillWorkerManager._new_worker
    vars(_source_wiring).update({'_CONSOLE_SKILL_NEW_WORKER': _CONSOLE_SKILL_NEW_WORKER})
    _CONSOLE_SKILL_ADD_WORKER = _SkillWorkerManager.add_worker
    vars(_source_wiring).update({'_CONSOLE_SKILL_ADD_WORKER': _CONSOLE_SKILL_ADD_WORKER})
    _CONSOLE_SKILL_START_WORKER = _source_wiring.Worker._start
    vars(_source_wiring).update({'_CONSOLE_SKILL_START_WORKER': _CONSOLE_SKILL_START_WORKER})
    _CONSOLE_SKILL_SCOPE_SLOTS = (
        "get_context",
        "_call",
        "_enforce_policy",
        "_require_service",
        "_normalize_mode",
        "_normalize_response",
        "_maybe_await",
        "_source_action_id",
        "_normalize_item",
        "_with_record_id",
    )
    vars(_source_wiring).update({'_CONSOLE_SKILL_SCOPE_SLOTS': _CONSOLE_SKILL_SCOPE_SLOTS})
    _CONSOLE_SKILL_LOCAL_SLOTS = ("get_context", "trust_service")
    vars(_source_wiring).update({'_CONSOLE_SKILL_LOCAL_SLOTS': _CONSOLE_SKILL_LOCAL_SLOTS})


    # Definition-time originals for stock Console trust preparation only.
    from types import (
        FunctionType as _SkillFunctionType,
    )
    vars(_source_wiring).update({'_SkillFunctionType': _SkillFunctionType})
    from types import (
        MappingProxyType as _SkillMappingProxyType,
    )
    vars(_source_wiring).update({'_SkillMappingProxyType': _SkillMappingProxyType})

    _CONSOLE_SKILL_FUNCTIONS = {
        "_build_local_skill_trust_service": _source_wiring._skill_wiring.SkillServiceWiringMixin.__dict__[
            "_build_local_skill_trust_service"
        ],
        "_build_local_skills_stack": _source_wiring._skill_wiring.SkillServiceWiringMixin.__dict__[
            "_build_local_skills_stack"
        ],
        "ensure_local_skill_trust_service": _source_wiring._skill_wiring.SkillServiceWiringMixin.__dict__[
            "ensure_local_skill_trust_service"
        ],
        "local_skill_trust_service": _source_wiring.ServiceWiringMixin.__dict__[
            "local_skill_trust_service"
        ].fget,
        "local_skills_service": _source_wiring.ServiceWiringMixin.__dict__["local_skills_service"].fget,
        "skills_scope_service": _source_wiring.ServiceWiringMixin.__dict__["skills_scope_service"].fget,
        "_console_skill_metadata_current": _source_wiring._console_skill_metadata_current,
        "_console_skill_source_current": _source_wiring._console_skill_source_current,
        "_capture_console_skill_trust_source": _source_wiring._capture_console_skill_trust_source,
        "_capture_console_skill_trust_preparation": _source_wiring._capture_console_skill_trust_preparation,
        "_prepare_console_skill_trust_service": _source_wiring._prepare_console_skill_trust_service,
        "_CONSOLE_SKILL_RUN_WORKER": _CONSOLE_SKILL_RUN_WORKER,
        "_CONSOLE_SKILL_WAIT": _CONSOLE_SKILL_WAIT,
        "_CONSOLE_SKILL_WORKERS_GET": _CONSOLE_SKILL_WORKERS_GET,
        "_CONSOLE_SKILL_APP_GET": _CONSOLE_SKILL_APP_GET,
        "_CONSOLE_SKILL_NEW_WORKER": _CONSOLE_SKILL_NEW_WORKER,
        "_CONSOLE_SKILL_ADD_WORKER": _CONSOLE_SKILL_ADD_WORKER,
        "_CONSOLE_SKILL_START_WORKER": _CONSOLE_SKILL_START_WORKER,
    }
    vars(_source_wiring).update({'_CONSOLE_SKILL_FUNCTIONS': _CONSOLE_SKILL_FUNCTIONS})
    _CONSOLE_SKILL_CONTEXT_SOURCE = (
        vars(_source_wiring),
        _source_wiring.__file__,
        _source_wiring.__spec__,
        getattr(_source_wiring.__spec__, "origin", None),
        (
            (vars(_source_wiring), "ServiceWiringMixin", _source_wiring.ServiceWiringMixin),
            (vars(_source_wiring), "_CONSOLE_SKILL_FUNCTIONS", _CONSOLE_SKILL_FUNCTIONS),
            (
                _source_wiring._skill_wiring.SkillServiceWiringMixin.__dict__,
                "_build_local_skill_trust_service",
                _source_wiring._skill_wiring.SkillServiceWiringMixin.__dict__["_build_local_skill_trust_service"],
            ),
            (
                _source_wiring._skill_wiring.SkillServiceWiringMixin.__dict__,
                "_build_local_skills_stack",
                _source_wiring._skill_wiring.SkillServiceWiringMixin.__dict__["_build_local_skills_stack"],
            ),
            (
                _source_wiring._skill_wiring.SkillServiceWiringMixin.__dict__,
                "ensure_local_skill_trust_service",
                _source_wiring._skill_wiring.SkillServiceWiringMixin.__dict__["ensure_local_skill_trust_service"],
            ),
            (
                _source_wiring.ServiceWiringMixin.__dict__,
                "local_skill_trust_service",
                _source_wiring.ServiceWiringMixin.__dict__["local_skill_trust_service"],
            ),
            (
                _source_wiring.ServiceWiringMixin.__dict__,
                "local_skills_service",
                _source_wiring.ServiceWiringMixin.__dict__["local_skills_service"],
            ),
            (
                _source_wiring.ServiceWiringMixin.__dict__,
                "skills_scope_service",
                _source_wiring.ServiceWiringMixin.__dict__["skills_scope_service"],
            ),
            (vars(_source_wiring), "_ConsoleSkillTrustSource", _source_wiring._ConsoleSkillTrustSource),
            (
                vars(_source_wiring),
                "_capture_console_skill_trust_source",
                _source_wiring._capture_console_skill_trust_source,
            ),
            (vars(_source_wiring), "_console_skill_metadata_current", _source_wiring._console_skill_metadata_current),
            (vars(_source_wiring), "_console_skill_source_current", _source_wiring._console_skill_source_current),
            (
                vars(_source_wiring),
                "_capture_console_skill_trust_preparation",
                _source_wiring._capture_console_skill_trust_preparation,
            ),
            (
                vars(_source_wiring),
                "_prepare_console_skill_trust_service",
                _source_wiring._prepare_console_skill_trust_service,
            ),
            (vars(_source_wiring), "_CONSOLE_SKILL_RUN_WORKER", _CONSOLE_SKILL_RUN_WORKER),
            (vars(_source_wiring), "_CONSOLE_SKILL_WAIT", _CONSOLE_SKILL_WAIT),
            (vars(_source_wiring), "_CONSOLE_SKILL_WORKERS_GET", _CONSOLE_SKILL_WORKERS_GET),
            (vars(_source_wiring), "_CONSOLE_SKILL_APP_GET", _CONSOLE_SKILL_APP_GET),
            (vars(_source_wiring), "_CONSOLE_SKILL_NEW_WORKER", _CONSOLE_SKILL_NEW_WORKER),
            (vars(_source_wiring), "_CONSOLE_SKILL_ADD_WORKER", _CONSOLE_SKILL_ADD_WORKER),
            (vars(_source_wiring), "_CONSOLE_SKILL_START_WORKER", _CONSOLE_SKILL_START_WORKER),
            (vars(_source_wiring), "get_user_data_dir", _source_wiring.get_user_data_dir),
            (vars(_source_wiring), "LocalSkillsService", _source_wiring.LocalSkillsService),
            (vars(_source_wiring), "SkillsScopeService", _source_wiring.SkillsScopeService),
            (vars(_source_wiring), "Worker", _source_wiring.Worker),
            (vars(_source_wiring), "_CONSOLE_SKILL_SCOPE_SLOTS", _CONSOLE_SKILL_SCOPE_SLOTS),
            (vars(_source_wiring), "_CONSOLE_SKILL_LOCAL_SLOTS", _CONSOLE_SKILL_LOCAL_SLOTS),
            (vars(_source_wiring), "_SkillWorkerManager", _SkillWorkerManager),
            (vars(_source_wiring), "_CONSOLE_SKILL_WORKERS", _CONSOLE_SKILL_WORKERS),
            (vars(_source_wiring), "_CONSOLE_SKILL_APP", _CONSOLE_SKILL_APP),
            (_SkillAppBase.__dict__, "workers", _SkillAppBase.__dict__["workers"]),
            (_SkillAppAccessor.__dict__, "app", _SkillAppAccessor.__dict__["app"]),
            (
                _CONSOLE_SKILL_APP_GET.__globals__,
                "active_app",
                _CONSOLE_SKILL_APP_GET.__globals__["active_app"],
            ),
            (_CONSOLE_SKILL_NEW_WORKER.__globals__, "Worker", _source_wiring.Worker),
            (
                _SkillWorkerManager.__dict__,
                "_new_worker",
                _SkillWorkerManager.__dict__["_new_worker"],
            ),
            (
                _SkillWorkerManager.__dict__,
                "add_worker",
                _SkillWorkerManager.__dict__["add_worker"],
            ),
            (_source_wiring.Worker.__dict__, "_start", _source_wiring.Worker.__dict__["_start"]),
            (_source_wiring.Worker.__dict__, "wait", _source_wiring.Worker.__dict__["wait"]),
            (
                _SkillMessagePump.__dict__,
                "run_worker",
                _SkillMessagePump.__dict__["run_worker"],
            ),
        ),
        tuple(
            (
                _CONSOLE_SKILL_FUNCTIONS,
                _skill_name,
                _skill_function,
                _skill_function.__code__,
                _skill_function.__globals__,
                _skill_function.__defaults__,
                _skill_function.__kwdefaults__,
                tuple((_skill_function.__kwdefaults__ or {}).items()),
                _skill_function.__closure__,
                tuple(
                    (cell, cell.cell_contents) for cell in _skill_function.__closure__ or ()
                ),
                vars(_skill_function).get("__wrapped__"),
            )
            for _skill_name, _skill_function in _CONSOLE_SKILL_FUNCTIONS.items()
            if type(_skill_function) is _SkillFunctionType
        ),
    )
    vars(_source_wiring).update({'_CONSOLE_SKILL_CONTEXT_SOURCE': _CONSOLE_SKILL_CONTEXT_SOURCE})

    _CONSOLE_SKILL_WIRING_SOURCE = _CONSOLE_SKILL_CONTEXT_SOURCE
    vars(_source_wiring).update({'_CONSOLE_SKILL_WIRING_SOURCE': _CONSOLE_SKILL_WIRING_SOURCE})
    _CONSOLE_SKILL_ENTRY = (
        _source_wiring._capture_console_skill_trust_preparation,
        _source_wiring._prepare_console_skill_trust_service,
    )
    vars(_source_wiring).update({'_CONSOLE_SKILL_ENTRY': _CONSOLE_SKILL_ENTRY})
    _CONSOLE_SKILL_TRUST_FACTORY_CODE = next(
        code
        for code in _source_wiring.ServiceWiringMixin._build_local_skills_stack.__code__.co_consts
        if isinstance(code, type(_source_wiring.ServiceWiringMixin._build_local_skills_stack.__code__))
        and code.co_name == "<lambda>"
        and code.co_names == ("local_skill_trust_service",)
    )
    vars(_source_wiring).update({'_CONSOLE_SKILL_TRUST_FACTORY_CODE': _CONSOLE_SKILL_TRUST_FACTORY_CODE})
    _CONSOLE_SKILL_CONTEXT_SOURCE = (
        *_CONSOLE_SKILL_CONTEXT_SOURCE[:4],
        _CONSOLE_SKILL_CONTEXT_SOURCE[4]
        + (
            (vars(_source_wiring), "_CONSOLE_SKILL_ENTRY", _CONSOLE_SKILL_ENTRY),
            (
                vars(_source_wiring),
                "_CONSOLE_SKILL_TRUST_FACTORY_CODE",
                _CONSOLE_SKILL_TRUST_FACTORY_CODE,
            ),
        ),
        _CONSOLE_SKILL_CONTEXT_SOURCE[5],
    )
    vars(_source_wiring).update({'_CONSOLE_SKILL_CONTEXT_SOURCE': _CONSOLE_SKILL_CONTEXT_SOURCE})
    _CONSOLE_SKILL_WIRING_SOURCE = _CONSOLE_SKILL_CONTEXT_SOURCE
    vars(_source_wiring).update({'_CONSOLE_SKILL_WIRING_SOURCE': _CONSOLE_SKILL_WIRING_SOURCE})
    _helper_namespace = globals()
    _helper_names = ('_console_skill_metadata_current', '_console_skill_source_current', '_capture_console_skill_trust_source', '_capture_console_skill_trust_preparation', '_prepare_console_skill_trust_service')
    _helper_bindings = tuple((_helper_namespace, name, _helper_namespace[name]) for name in _helper_names) + ((_helper_namespace, '_source_wiring', _source_wiring), (_helper_namespace, '__package__', __package__), (vars(_source_wiring), '_skill_wiring', _source_wiring._skill_wiring))
    _helper_bindings += tuple((_helper_namespace, name, _helper_namespace[name]) for name in ("SkillServiceWiringMixin",))
    _helper_records = tuple((
        _helper_namespace, name, function, function.__code__,
        function.__globals__, function.__defaults__, function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()), function.__closure__,
        tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
        vars(function).get("__wrapped__"),
    ) for name in _helper_names if type(function := _helper_namespace[name]) is _SkillFunctionType)
    _class_names = ('_build_local_skill_trust_service', '_build_local_skills_stack', 'ensure_local_skill_trust_service')
    _class_bindings = tuple((SkillServiceWiringMixin.__dict__, name, SkillServiceWiringMixin.__dict__[name]) for name in _class_names)
    _class_records = tuple((SkillServiceWiringMixin.__dict__, name, function, function.__code__, function.__globals__, function.__defaults__, function.__kwdefaults__, tuple((function.__kwdefaults__ or {}).items()), function.__closure__, tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()), vars(function).get("__wrapped__")) for name in _class_names if type(function := SkillServiceWiringMixin.__dict__[name]) is _SkillFunctionType)
    _CONSOLE_SKILL_HELPER_SOURCE = (_helper_namespace, __file__, __spec__, getattr(__spec__, "origin", None), _helper_bindings + _class_bindings, _helper_records + _class_records)
    vars(_source_wiring).update({'_CONSOLE_SKILL_HELPER_SOURCE': _CONSOLE_SKILL_HELPER_SOURCE})
    vars(_source_wiring).update({
        '_CONSOLE_SKILL_ADD_WORKER': _CONSOLE_SKILL_ADD_WORKER,
        '_CONSOLE_SKILL_APP': _CONSOLE_SKILL_APP,
        '_CONSOLE_SKILL_APP_GET': _CONSOLE_SKILL_APP_GET,
        '_CONSOLE_SKILL_CONTEXT_SOURCE': _CONSOLE_SKILL_CONTEXT_SOURCE,
        '_CONSOLE_SKILL_ENTRY': _CONSOLE_SKILL_ENTRY,
        '_CONSOLE_SKILL_FUNCTIONS': _CONSOLE_SKILL_FUNCTIONS,
        '_CONSOLE_SKILL_LOCAL_SLOTS': _CONSOLE_SKILL_LOCAL_SLOTS,
        '_CONSOLE_SKILL_NEW_WORKER': _CONSOLE_SKILL_NEW_WORKER,
        '_CONSOLE_SKILL_RUN_WORKER': _CONSOLE_SKILL_RUN_WORKER,
        '_CONSOLE_SKILL_SCOPE_SLOTS': _CONSOLE_SKILL_SCOPE_SLOTS,
        '_CONSOLE_SKILL_START_WORKER': _CONSOLE_SKILL_START_WORKER,
        '_CONSOLE_SKILL_TRUST_FACTORY_CODE': _CONSOLE_SKILL_TRUST_FACTORY_CODE,
        '_CONSOLE_SKILL_WAIT': _CONSOLE_SKILL_WAIT,
        '_CONSOLE_SKILL_WIRING_SOURCE': _CONSOLE_SKILL_WIRING_SOURCE,
        '_CONSOLE_SKILL_WORKERS': _CONSOLE_SKILL_WORKERS,
        '_CONSOLE_SKILL_WORKERS_GET': _CONSOLE_SKILL_WORKERS_GET,
        '_SkillAppAccessor': _SkillAppAccessor,
        '_SkillAppBase': _SkillAppBase,
        '_SkillFunctionType': _SkillFunctionType,
        '_SkillMappingProxyType': _SkillMappingProxyType,
        '_SkillMessagePump': _SkillMessagePump,
        '_SkillWorkerManager': _SkillWorkerManager,
        '_CONSOLE_SKILL_HELPER_SOURCE': _CONSOLE_SKILL_HELPER_SOURCE,
    })


def _wire_skill_inputs(self):
    try:
        self.server_skills_service = _source_wiring.ServerSkillsService.from_config(
            self.app_config,
            policy_enforcer=self.service_policy_enforcer,
        )
    except ValueError:
        self.server_skills_service = _source_wiring.ServerSkillsService(
            client=None,
            policy_enforcer=self.service_policy_enforcer,
        )
    # The local skills stack (trust service -> local service -> scope
    # facade) is built on first access, not here: constructing the trust
    # service performs OS keyring backend discovery TWICE (marker store +
    # key cache) for a feature most boots never touch (TASK-21111(b)).
    # Every consumer reads these through `getattr(app_instance, ...)` at
    # UI time, so a property is a drop-in.
    self._local_skill_trust_service: _source_wiring.Any | None = None
    # task-33081: serializes the off-loop first build above.
    self._local_skill_trust_service_build_lock = _source_wiring.asyncio.Lock()
    self._local_skills_service: _source_wiring.Any | None = None
    self._skills_scope_service: _source_wiring.Any | None = None
    # Captured NOW, at the timing the eager build had: `_build_local_
    # skills_stack` must not re-read collaborators that a test (or a
    # later boot step) may reassign between construction and first use
    # (the TASK-21108 deferral trap).
    self._local_skills_stack_inputs = (
        self.service_policy_enforcer,
        self.server_skills_service,
    )


class SkillServiceWiringMixin:
    def _build_local_skill_trust_service(self) -> Any:
        """Build the skill trust service. Performs OS keyring discovery.

        Split out of the eager wiring (TASK-21111(b)): it is the only part
        of the local skills stack that touches the keyring -- twice, once
        for the rollback marker store's secure-backend probe and once for
        the trust key cache -- and nothing at startup asks a trust question.
        Deferring the whole SERVICE was not enough on its own: the Console's
        agent bridge takes the skills scope facade during Chat screen mount,
        which merely relocated the discovery from ``__init__`` to mount.
        ``LocalSkillsService`` therefore takes this as a FACTORY and calls it
        on the first trust decision.
        """
        local_skills_store_dir = _source_wiring.default_local_skills_store_dir(_source_wiring.get_user_data_dir())
        trust_store_dir = _source_wiring.default_trust_store_dir(local_skills_store_dir)
        from .Skills_Interop.recovery_activation import is_recovered
        recovered_skills = is_recovered(
            local_skills_store_dir / "skills", trust_store_dir
        )
        if recovered_skills:
            from .Skills_Interop.skill_trust_store import (
                FileSkillTrustGenerationMarkerStore,
            )
            skill_trust_marker_store = FileSkillTrustGenerationMarkerStore(
                trust_store_dir / _source_wiring._SKILL_TRUST_MARKER_FILENAME, store_dir=trust_store_dir
            )
            reduced_rollback_protection = True
            skill_key_cache = None
        else:
            trust_account_scope = _source_wiring.skill_trust_account_scope(trust_store_dir)
            skill_trust_marker_store, reduced_rollback_protection = (
                _source_wiring.build_skill_trust_marker_store_with_fallback(
                    fallback_marker_path=trust_store_dir / _source_wiring._SKILL_TRUST_MARKER_FILENAME,
                    store_dir=trust_store_dir,
                    account_scope=trust_account_scope,
                )
            )
            skill_key_cache = _source_wiring.build_default_skill_trust_key_cache(
                account_scope=trust_account_scope
            )
        return _source_wiring.SkillTrustService(
            skills_dir=local_skills_store_dir / "skills",
            trust_store=_source_wiring.SkillTrustStore(
                store_dir=trust_store_dir,
                marker_store=skill_trust_marker_store,
            ),
            key_cache=skill_key_cache,
            keyring_convenience_enabled=False,
            reduced_rollback_protection=reduced_rollback_protection,
        )

    @property
    def plugin_source_service(self) -> Any:
        """One lazy source facade, with no construction-time storage or fetching."""
        service = getattr(self, "_plugin_source_service", None)
        if service is None:
            from tldw_chatbook.Plugins.source_service import SourceService

            service = SourceService(self._build_plugin_service())
            self._plugin_source_service = service
        return service

    def _build_plugin_service(self) -> Any:
        """Build one IO-free facade; its worker starts on explicit plugin use."""
        service = getattr(self, "_plugin_service", None)
        if service is None:
            from tldw_chatbook.Plugins.service import PluginService
            service = PluginService(
                _source_wiring.get_user_data_dir(),
                workspace_lookup=self.workspace_registry_service.get_workspace,
                mcp_mapping_owner=self.local_mcp_control_service,
                model_mapping_owner=self.local_llm_provider_catalog_service,
            )
            self._plugin_service = service
        return service

    def _build_local_skills_stack(self) -> None:
        """Build the local skills service + scope facade. Idempotent.

        Body moved out of ``_wire_watchlists_and_notifications_services``
        (TASK-21111(b)). Keyring-free: the trust service is handed over as a
        factory. The collaborators it reads were captured at construction
        time, not re-read now, so deferring changes WHEN it runs and not
        WHAT it binds (the TASK-21108 trap).

        Each slot is filled only if still unset, so an injected double (a
        test assigning one of them between construction and first read) is
        never clobbered by a later sibling access.
        """
        if None not in (self._local_skills_service, self._skills_scope_service):
            return
        policy_enforcer, server_skills_service = self._local_skills_stack_inputs
        if self._local_skills_service is None:
            self._local_skills_service = _source_wiring.LocalSkillsService(
                store_dir=_source_wiring.default_local_skills_store_dir(_source_wiring.get_user_data_dir()),
                policy_enforcer=policy_enforcer,
                trust_service_factory=lambda: self.local_skill_trust_service,
                plugin_service_factory=self._build_plugin_service,
                # In-memory config read: this runs on every skills read,
                # including the Console's per-send capture.
                builtin_disabled_loader=_source_wiring.partial(_source_wiring._read_app_disabled_builtin_skills, self),
            )
        if self._skills_scope_service is None:
            self._skills_scope_service = _source_wiring.SkillsScopeService(
                local_service=self._local_skills_service,
                server_service=server_skills_service,
                policy_enforcer=policy_enforcer,
            )

    async def ensure_local_skill_trust_service(
        self,
        *,
        _source_current: Callable[[], bool] | None = None,
        _owner_current: Callable[[Any], bool] | None = None,
        _read_observers: tuple[set, ...] = (),
    ) -> Any:
        """First-use trust service build, OFF the UI event loop (task-33081).

        The build performs OS keyring backend discovery (SecretService/D-Bus
        on Linux -- the deferred-wiring notes' timing figure never measured
        that platform). The sync property keeps its contract for callers
        already on worker threads; UI-loop callers must come here so the
        discovery never blocks the loop.

        Returns:
            The shared local skill trust service, built once; concurrent
            first callers await the same build under the build lock.
        """
        if _owner_current is not None and _owner_current() is not True:
            raise RuntimeError("console_skill_trust_owner_changed")
        if self._local_skill_trust_service is not None:
            return self._local_skill_trust_service
        async with self._local_skill_trust_service_build_lock:
            if (
                _owner_current is not None
                and _owner_current(self._local_skill_trust_service) is not True
            ):
                raise RuntimeError("console_skill_trust_owner_changed")
            if self._local_skill_trust_service is None:
                from .Chat.console_preparation_reads import run_preparation_read
                def require_current():
                    if _source_current is not None and _source_current() is not True:
                        raise RuntimeError("console_skill_trust_source_changed")
                service = await run_preparation_read(
                    self._build_local_skill_trust_service,
                    creator=self,
                    session_id=None,
                    reads=set(),
                    observers=_read_observers,
                    require_current=require_current,
                )
                if (
                    _owner_current is not None
                    and _owner_current(self._local_skill_trust_service) is not True
                ):
                    raise RuntimeError("console_skill_trust_owner_changed")
                # Another owner may publish while the native builder is held.
                if self._local_skill_trust_service is None:
                    self._local_skill_trust_service = service
            return self._local_skill_trust_service
