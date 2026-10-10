"""Concrete configuration source lifetimes; no public maintenance authority."""

import secrets
import stat
import sys
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path

from tldw_chatbook.Utils.platform_files import fcntl, os

from . import bootstrap, profile_paths
from . import storage_admission as storage

ROUTES = {
    "config",
    "config_snapshot",
    "config_data",
    "config_data_lock",
    "config_default_root",
    "config_chat_dicts",
    "config_models",
}
_STATE_NAMES = (
    "_CONFIG_CACHE",
    "_CONFIG_CACHE_SOURCE",
    "_CONFIG_FILE_STAMP",
    "_CONFIG_STAT_CHECKED_MONOTONIC",
    "_SETTINGS_CACHE",
    "_SETTINGS_CACHE_SOURCE",
    "settings",
    "_CONFIG_GENERATION",
    "_ENCRYPTION_PASSWORD",
    "_FIRST_PROFILE_CREATED_THIS_SESSION",
)


class ConfigOperationBusy(RuntimeError):
    """Presentation entry deferred because another thread owns a config lock."""


def checked_config_identity(source: object, active: object) -> tuple[int, str]:
    """Tag data with its active checked config generation and selected path."""
    from . import raw_participants as raw

    state = raw._check(active)
    if state.source is not source or state.route not in {"config", "config_snapshot"}:
        raise bootstrap.RecoveryRequired("config_operation_source_invalid")
    return source._CONFIG_GENERATION, str(state.selected)


def _sensitive_input_publication_owner(source, active):
    """Capture refusal metadata from the actual issued config reader."""
    from . import raw_participants as raw

    with storage._lock:
        state = raw._live_state(active, None, False)
        if state.source is not source or state.route not in {
            "config",
            "config_snapshot",
        }:
            raise bootstrap.RecoveryRequired("config_operation_source_invalid")
        participant = state.participant
        if participant is None:
            raise bootstrap.RecoveryRequired("raw_participant_not_installed")
        gate = raw._participant_identity(participant)
        if gate.source() is not source or gate.owner != "config":
            raise bootstrap.RecoveryRequired("config_operation_source_invalid")
        return participant, gate, gate.source, gate.selected, storage._lock


def _check_sensitive_input_publication(source, owner):
    """Refuse unavailable publication; never admit another read or native work."""
    from . import raw_participants as raw

    participant, gate, source_ref, selected, lock = owner
    if lock is not storage._lock:
        raise bootstrap.RecoveryRequired("config_publication_source_changed")
    with lock:
        if storage._pause is not None:
            raise bootstrap.RecoveryRequired("storage_locally_paused")
        if (
            gate.source is not source_ref
            or gate.selected is not selected
            or raw._participant_identity(participant) is not gate
            or source_ref() is not source
            or gate.owner != "config"
        ):
            raise bootstrap.RecoveryRequired("config_publication_source_changed")
        if gate.closed:
            raise bootstrap.RecoveryRequired("storage_locally_paused")


def binding(source):
    module = sys.modules.get("tldw_chatbook.config")
    if module is not None and source is module:
        return (
            "config",
            profile_paths.lexical_path(source._get_effective_config_path()),
            True,
        )
    return None


def selection(source, route, target):
    bound = binding(source)
    if bound is None:
        raise bootstrap.RecoveryRequired("raw_source_not_supported")
    _, selected, installed = bound
    if route in {"config", "config_snapshot"}:
        if (
            target is not None
            and route == "config"
            and profile_paths.lexical_path(target) != selected
        ):
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
        return selected, installed, False
    if route == "config_data_lock":
        lock_path = profile_paths.lexical_path(
            profile_paths.default_base_data_dir().parents[2] / profile_paths.DATA_ROOT_LOCK_NAME
        )
        if target is not None and profile_paths.lexical_path(target) != lock_path:
            raise bootstrap.RecoveryRequired("config_directory_selection_changed")
        return lock_path, installed, False
    data = source._CONFIG_CACHE
    if data is None or source._CONFIG_CACHE_SOURCE != selected:
        raise bootstrap.RecoveryRequired("config_directory_selection_unavailable")
    if route == "config_default_root":
        configured = profile_paths.setting(data, "paths", "data_dir")
        if configured is None:
            configured = profile_paths.setting(data, "Paths", "data_dir")
        conventional = profile_paths.lexical_path(profile_paths.default_base_data_dir())
        fallback = conventional.parents[2] / profile_paths.DEFAULT_DATA_FALLBACK_DIRECTORY
        selected_root = profile_paths.lexical_path(target)
        if configured or selected_root not in {conventional, fallback}:
            raise bootstrap.RecoveryRequired("config_directory_selection_changed")
        return selected_root, installed, True
    directory = profile_paths.user_data_dir(data)
    if route == "config_chat_dicts":
        directory /= "chat_dicts"
    elif route == "config_models":
        default = source.DEFAULT_CONFIG_FROM_TOML.get("embedding_config", {}).get(
            "model_cache_dir"
        )
        custom = data.get("embedding_config", {}).get("model_cache_dir", default)
        directory = (
            profile_paths.lexical_path(custom).resolve()
            if custom and custom != default
            else directory / "models" / "embeddings"
        )
    if profile_paths.lexical_path(target) != profile_paths.lexical_path(directory):
        raise bootstrap.RecoveryRequired("config_directory_selection_changed")
    return profile_paths.lexical_path(directory), installed, True


def members(source, selected, route, target):
    paths = (
        selected,
        selected.with_name(selected.name + ".lock"),
        source._advanced_backup_path(selected),
    )
    if route == "config_snapshot":
        paths += (source._config_snapshot_path(selected, target),)
    temporaries = {
        path: path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
        for path in paths
        if path.suffix != ".lock"
    }
    return paths + tuple(temporaries.values()), temporaries


def sibling_selector(source, route, selected):
    """Resolve only installed fixed config siblings, without config IO."""
    from . import raw_participants as raw

    module = sys.modules.get("tldw_chatbook.config")
    if module is None:
        return None
    if route == "sidebar_state":
        path, installed = raw._async_source_selection(source, route)
        leaf = "ui_state.toml"
    elif route == "emoji" and source is sys.modules.get("tldw_chatbook.Widgets.emoji_picker"):
        path, installed, leaf = source._recent_emojis_path(), True, "recent_emojis.json"
    elif route in {"runtime_read", "runtime_state"}:
        bound = raw.settings_files.binding(source)
        if bound is None or bound[0] != "runtime.source_state":
            return None
        _, path, installed = bound
        leaf = "runtime_policy.json"
    else:
        return None
    if not installed:
        return None
    selector = profile_paths.lexical_path(module._get_effective_config_path())
    if profile_paths.lexical_path(path) != selected:
        raise bootstrap.RecoveryRequired("config_companion_scope_changed")
    return selector if selected == selector.parent / leaf else None


def companion_guard(operation, attempt):
    """Keep exact installed config siblings disjoint until their IO retires."""
    from . import raw_participants as raw
    from .native_files import pinned_directory
    from .service_storage import default_control_root, work_root

    state = raw._states[operation]
    selected = state.config_anchor or state.selected
    hold = state.holds[-1]
    if hold is None:
        return None
    root = bootstrap.default_bootstrap_root()
    guard = ExitStack()
    try:
        parent = guard.enter_context(hold.authority._directory())
        guard.enter_context(
            hold.authority._lock(
                parent, "registry.lock", fcntl.LOCK_SH, cancel=attempt.cancel
            )
        )
        attempt.check()
        key = ("companions", str(selected))
        before = storage._derived_before(hold, key)
        reused, metadata = storage._derived_reuse(hold, key, before)
        if reused:
            record, registry = metadata
            pending = ()
        else:
            pending, profiles = bootstrap._records(root)
            record = next(
                (row for row in profiles if row["selector"] == str(selected)), None
            )
        if record is None or tuple(record["namespaces"]) != hold.names:
            guard.close()
            return None
        if (
            pending
            or hold.authority.control_root != root / "admission"
            or not reused
            and storage._scope(root, selected, selected, authority=hold.authority)
            != hold.names
            or (
                sibling_selector(state.source, state.route, state.selected) != selected
                if state.config_anchor is not None
                else binding(state.source) != ("config", selected, True)
                or state.route not in {"config", "config_snapshot"}
            )
            or not state.pinned
        ):
            raise bootstrap.RecoveryRequired("config_companion_scope_changed")
        if not reused:
            registry = bootstrap._registry(root)
        directory = guard.enter_context(pinned_directory(selected.parent))
        info = os.fstat(directory)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise bootstrap.RecoveryRequired("config_companion_parent_unsafe")
        controls = (root, default_control_root(), work_root(default_control_root()))
        foreign = [entry for name, entry in registry.items() if name not in hold.names]
        for member in state.paths:
            if member.parent != selected.parent or any(
                bootstrap._overlap(member, control) for control in controls
            ):
                raise bootstrap.RecoveryRequired("config_companion_scope_changed")
            tokens = {"path:" + str(member), "path:" + str(member.resolve())}
            try:
                member_info = os.stat(member, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(member_info.st_mode) or member_info.st_nlink != 1:
                    raise bootstrap.RecoveryRequired("config_companion_scope_changed")
                tokens.add(bootstrap.inode_token(member_info))
            if any(
                tokens.intersection(bootstrap.identity_view(entry["historical"]))
                or any(
                    bootstrap._overlap(member, Path(path)) for path in entry["roots"]
                )
                or any(
                    bootstrap._overlap(member, Path(token[5:]))
                    for token in entry["historical"]
                    if token.startswith("path:")
                )
                for entry in foreign
            ):
                raise bootstrap.RecoveryRequired("config_companion_scope_changed")
        # Existing parent posture is verified, never delegated for creation.
        state.directories = ()
        state.companion_roots = tuple(Path(path) for path in record["roots"])
        if not reused:
            evidence = storage._metadata_evidence(
                hold, (selected, selected.parent), registry=registry
            )
            storage._note_derived(hold, key, evidence, (record, registry), before)
        return guard
    except BaseException:
        guard.close()
        raise


def verified_companion_parent(source, path):
    """Only a current proven owner operation uses a verify-only parent seam."""
    from . import raw_participants as raw

    operation = getattr(raw._local, "operation", None)
    if operation is None:
        return False
    state = raw._check(operation)
    return (
        state.source is source
        and state.companion_guard is not None
        and (
            state.route in {"config", "config_snapshot"}
            or state.route == "runtime_state" and state.config_anchor is not None
        )
        and state.selected == profile_paths.lexical_path(path)
    )


def verified_user_data_directory(source):
    """Observe an existing bound profile directory without root-selection writes."""
    from . import raw_participants as raw
    from .native_files import pinned_directory

    operation = getattr(raw._local, "operation", None)
    if operation is None:
        return None
    state = raw._check(operation)
    if state.source is not source or state.companion_guard is None:
        return None
    if source._CONFIG_CACHE_SOURCE != state.selected or source._CONFIG_CACHE is None:
        raise bootstrap.RecoveryRequired("config_directory_selection_unavailable")
    selected = profile_paths.user_data_dir(source._CONFIG_CACHE)
    associated = False
    owned_parent = False
    for root in state.companion_roots:
        if root == state.selected:
            continue
        info = os.stat(root, follow_symlinks=False)
        if (
            root.parent == selected and (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode))
            or stat.S_ISDIR(info.st_mode) and (root == selected or root in selected.parents)
        ):
            associated = True
            owned_parent = stat.S_ISDIR(info.st_mode) and root in selected.parents
            break
    if not associated:
        raise bootstrap.RecoveryRequired("config_directory_selection_changed")
    missing = False
    try:
        os.stat(selected, follow_symlinks=False)
    except FileNotFoundError:
        if not owned_parent:
            raise
        missing = True
    with ExitStack() as parents:
        for directory in ((selected.parent,) if missing else (selected.parent, selected)):
            fd = parents.enter_context(pinned_directory(directory))
            info = os.fstat(fd)
            current = os.stat(directory, follow_symlinks=False)
            if (
                info.st_uid != os.geteuid() or info.st_mode & 0o077
                or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino)
            ):
                raise bootstrap.RecoveryRequired("config_directory_selection_changed")
        # Retain the installed default/fallback ambiguity check at both edges.
        if profile_paths.user_data_dir(source._CONFIG_CACHE) != selected:
            raise bootstrap.RecoveryRequired("config_directory_selection_changed")
        raw._check(operation)
        # Creation still belongs to the fully admitted config_data operation.
        return None if missing else selected


@contextmanager
def operation(source, *, route="config", target=None, wait_for_locks: bool = True):
    """Retain a checked config lifetime, optionally refusing busy lock entry."""
    from . import raw_participants as raw

    previous = getattr(raw._local, "operation", None)
    nested = (
        previous is not None
        and raw._check(previous).source is source
        and route in {"config", "config_snapshot"}
    )
    nested_target = target
    if nested and route == "config_snapshot":
        raw._check(
            previous,
            source._config_snapshot_path(raw._check(previous).selected, target),
            writing=True,
        )
        nested_target = None
    if nested:
        before = {
            name: getattr(source, name)
            for name in _STATE_NAMES
            if hasattr(source, name)
        }
        try:
            with raw._scope(
                source, "config", writing=True, selected_read=nested_target
            ) as active:
                yield active
        except BaseException:
            for name, value in before.items():
                setattr(source, name, value)
            raw._states[previous].config_failed = True
            source._CONFIG_PERSISTENCE_ERROR = "config_operation_failed"
            raise
        return
    core = getattr(storage._operation_local, "operation", None)
    if core is not None:
        storage._check_operation(core, core.path)
    storage._operation_local.operation = None
    attempt = None
    acquired = ExitStack()
    before = None
    entered = False
    try:
        attempt = storage._Acquisition()
        # Config snapshots and rebuilds already use REBUILD -> FILE. Keep
        # that order before entering any guarded reader or writer body.
        for lock in (source._settings_rebuild_lock(), source._config_file_lock()):
            if wait_for_locks:
                while not lock.acquire(timeout=0.05):
                    attempt.check()
            elif not lock.acquire(blocking=False):
                attempt.check()
                raise ConfigOperationBusy("config_lock_busy")
            acquired.callback(lock.release)
            attempt.check()
        before = {
            name: getattr(source, name)
            for name in _STATE_NAMES
            if hasattr(source, name)
        }
        with raw._scope(source, route, writing=True, selected_read=target) as active:
            state = raw._states[active]
            entered = True
            yield active
        if active in raw._states:
            raise bootstrap.RecoveryRequired("raw_resources_not_retired")
        if state.config_failed:
            for name, value in before.items():
                setattr(source, name, value)
    except BaseException:
        if before is not None:
            for name, value in before.items():
                setattr(source, name, value)
        if entered:
            source._CONFIG_PERSISTENCE_ERROR = "config_operation_failed"
        raise
    finally:
        acquired.close()
        if attempt is not None:
            attempt.close()
        if core is not None:
            storage._check_operation(core, core.path)
        storage._operation_local.operation = core


# TASK-34404: original wrapper and generator, retained before consumer import.
_STARTUP_PATH_OPERATION_ORIGINAL = tuple(
    (
        namespace,
        name,
        callback,
        callback.__code__,
        callback.__globals__,
        callback.__defaults__,
        callback.__kwdefaults__,
        tuple((callback.__kwdefaults__ or {}).items()),
        callback.__closure__,
        tuple((cell, cell.cell_contents) for cell in callback.__closure__ or ()),
    )
    for namespace, name, callback in (
        (globals(), "operation", operation),
        (operation.__dict__, "__wrapped__", operation.__wrapped__),
    )
)



def _startup_path_config_bundle(source, aliases):
    """Reuse original refusal metadata for one finite stock path selection."""
    from types import FunctionType, ModuleType

    from tldw_chatbook.Utils import sensitive_paths as metadata

    tuple_type, dict_type, str_type, int_type = tuple, dict, str, int
    retained = metadata.__dict__.get("_STARTUP_PATH_METADATA_ORIGINALS")
    if type(retained) is not tuple_type or len(retained) != 3:
        return None
    bundle_type, descriptors, callbacks = retained
    operation_record, reader_owner = (
        _STARTUP_PATH_OPERATION_ORIGINAL,
        _STARTUP_PATH_READER_ORIGINAL,
    )

    def callbacks_current(records):
        for (
            namespace,
            name,
            callback,
            code,
            defining,
            defaults,
            kwdefaults,
            items,
            closure,
            cells,
        ) in records:
            if (
                namespace.get(name) is not callback
                or type(callback) is not FunctionType
                or callback.__code__ is not code
                or callback.__globals__ is not defining
                or callback.__defaults__ is not defaults
                or callback.__kwdefaults__ is not kwdefaults
                or callback.__closure__ is not closure
                or (
                    kwdefaults is not None
                    and (
                        len(kwdefaults) != len(items)
                        or any(kwdefaults.get(key) is not value for key, value in items)
                    )
                )
                or len(closure or ()) != len(cells)
            ):
                return False
            try:
                if any(
                    cell is not original or cell.cell_contents is not value
                    for cell, (original, value) in zip(closure or (), cells)
                ):
                    return False
            except ValueError:
                return False
        return True

    def metadata_current():
        return (
            sys.modules.get(metadata.__name__) is metadata
            and metadata.__dict__.get("_STARTUP_PATH_METADATA_ORIGINALS") is retained
            and metadata.__dict__.get("_SensitiveConfigInputBundle") is bundle_type
            and metadata.__dict__.get("_SensitiveFunctionType") is FunctionType
            and metadata.__dict__.get("_SensitiveModuleType") is ModuleType
            and len(bundle_type.__dict__) == len(descriptors)
            and all(
                bundle_type.__dict__.get(name) is value for name, value in descriptors
            )
            and globals().get("_STARTUP_PATH_OPERATION_ORIGINAL") is operation_record
            and globals().get("_STARTUP_PATH_READER_ORIGINAL") is reader_owner
            and _startup_path_config_bundle is reader_owner[0]
            and reader_owner[0].__code__ is reader_owner[1]
            and reader_owner[0].__globals__ is reader_owner[2]
            and reader_owner[0].__defaults__ is None
            and reader_owner[0].__kwdefaults__ is None
            and reader_owner[0].__closure__ is None
            and callbacks_current(callbacks + operation_record)
        )

    if not metadata_current():
        return None
    readers = {name: callback for _, name, callback, *_ in callbacks}
    bindings = readers["_sensitive_reader_bindings"](source)
    local = readers["_sensitive_reader_bindings"](metadata)
    cached = (
        readers["_sensitive_cached_reader_bindings"](source)
        if bindings is not None
        else None
    )
    if bindings is None or local is None or cached is None:
        return None
    namespace = source.__dict__
    guarded = namespace.get("_SENSITIVE_INPUT_GUARDED_READERS")
    if not readers["_sensitive_guarded_readers_current"](source, guarded):
        return None
    cache, defaults = (
        namespace.get("_CONFIG_CACHE"),
        namespace.get("DEFAULT_CONFIG_FROM_TOML"),
    )
    if type(cache) is not dict_type or type(defaults) is not dict_type:
        return None
    database, default_database = cache.get("database"), defaults.get("database")
    if type(database) is not dict_type or type(default_database) is not dict_type:
        return None
    settings = ("chachanotes_db_path", "media_db_path", "prompts_db_path")
    values, default_values = (
        tuple(database.get(name) for name in settings),
        tuple(default_database.get(name) for name in settings),
    )
    if any(
        type(value) not in (str_type, type(None))
        or type(default) not in (str_type, type(None))
        or (value and value != default)
        for value, default in zip(values, default_values)
    ):
        return None
    for owner, defining, records in namespace["_SENSITIVE_INPUT_OWNERS"]:
        if (
            type(owner) is not ModuleType
            or sys.modules.get(owner.__name__) is not owner
            or owner.__dict__ is not defining
        ):
            return None
        for name, callback, callback_globals, code in records:
            if (
                type(callback) is not FunctionType
                or defining.get(name) is not callback
                or callback.__globals__ is not callback_globals
                or callback.__code__ is not code
            ):
                return None
            bindings += ((owner, defining, name, callback, callback_globals, code),)
    for module, name, callback in aliases:
        if (
            type(module) is not ModuleType
            or module.__dict__.get(name) is not callback
            or namespace.get(name) is not callback
        ):
            return None
        bindings += (
            (
                module,
                module.__dict__,
                name,
                callback,
                callback.__globals__,
                callback.__code__,
            ),
        )
    alias_defaults = tuple(
        (callback, callback.__kwdefaults__) for _, _, callback in aliases
    )
    loader = namespace["load_cli_config_and_ensure_existence"]
    loader_defaults = loader.__defaults__
    if (
        type(loader_defaults) is not tuple_type
        or len(loader_defaults) != 1
        or loader_defaults[0] is not False
    ):
        return None
    environment, cwd = os.environ.copy(), os.getcwd()
    key = (
        cache,
        namespace["_CONFIG_GENERATION"],
        namespace["_CONFIG_CACHE_SOURCE"],
        str(namespace["_get_effective_config_path"]()),
    )
    if type(key[1]) is not int_type:
        return None
    names = metadata.__dict__.get("_SENSITIVE_INPUT_DB_NAMES")
    if (
        type(names) is not tuple_type
        or metadata.__dict__.get("_DB_PATH_ACCESSOR_NAMES") is not names
    ):
        return None
    if not metadata_current():
        return None
    bundle = object.__new__(bundle_type)
    readers["__init__"](
        bundle,
        source,
        metadata,
        key,
        bindings + local,
        cached,
        guarded,
        operation_record[0][2],
        checked_config_identity,
        _sensitive_input_publication_owner,
        _check_sensitive_input_publication,
        names,
        bootstrap.RecoveryRequired,
    )
    fields = tuple(bundle.__dict__.items())
    mappings = tuple(
        (value, tuple(value.items())) for _, value in fields if type(value) is dict_type
    )

    def check(*, publication=False):
        if (
            not metadata_current()
            or os.environ != environment
            or os.getcwd() != cwd
            or len(bundle.__dict__) != len(fields)
            or any(bundle.__dict__.get(name) is not value for name, value in fields)
            or any(
                len(mapping) != len(items)
                or any(mapping.get(name) is not value for name, value in items)
                for mapping, items in mappings
            )
            or namespace.get("_CONFIG_CACHE_SOURCE") is not key[2]
            or namespace.get("DEFAULT_CONFIG_FROM_TOML") is not defaults
            or cache.get("database") is not database
            or defaults.get("database") is not default_database
            or any(
                database.get(name) is not value for name, value in zip(settings, values)
            )
            or any(
                default_database.get(name) is not value
                for name, value in zip(settings, default_values)
            )
            or loader.__defaults__ is not loader_defaults
        ):
            raise bootstrap.RecoveryRequired("chatbook_path_source_changed")
        for callback, kwdefaults in alias_defaults:
            if (
                callback.__defaults__ is not None
                or callback.__closure__ is not None
                or callback.__kwdefaults__ is not kwdefaults
                or type(kwdefaults) is not dict_type
                or tuple(kwdefaults) != ("ignore_override",)
                or kwdefaults["ignore_override"] is not False
            ):
                raise bootstrap.RecoveryRequired("chatbook_path_reader_changed")
        readers["check"](bundle, observe_path=not publication)

    # Preinstalled custom body/default contracts retain the original route.
    try:
        check()
    except bootstrap.RecoveryRequired:
        return None
    return bundle, check


_STARTUP_PATH_READER_ORIGINAL = (
    _startup_path_config_bundle,
    _startup_path_config_bundle.__code__,
    _startup_path_config_bundle.__globals__,
)



def guarded(function):
    """Enclose the actual config reader/cache bodies, including direct helpers."""
    allowed = {
        # PERF-06: the public load_settings / get_runtime_config_snapshot
        # serve warm hits unguarded; their bodies below stay guarded.
        "_load_settings_guarded",
        "_load_settings_uncached",
        "_load_cli_config_bootstrap",
        "_load_cli_config_bootstrap_unlocked",
        "_read_raw_cli_config_unlocked",
        "_write_raw_cli_config_unlocked",
        "_try_read_cli_config_serialized_unlocked",
        "_read_cli_config_serialized_unlocked",
        "read_cli_config_serialized",
        "read_cli_config_backup_serialized",
        "_publish_runtime_config_unlocked",
        "_prepare_config_parent",
        "_get_runtime_config_snapshot_guarded",
        "get_user_data_dir",
        "get_model_cache_dir",
        "replace_cli_config_serialized",
    }
    if function.__name__ not in allowed:
        raise ValueError("config_source_not_supported")

    @wraps(function)
    def wrapped(*args, **kwargs):
        source = sys.modules.get("tldw_chatbook.config")
        if (
            source is None
            or source.__dict__ is not function.__globals__
            or getattr(source, function.__name__, None) is not wrapped
        ):
            raise bootstrap.RecoveryRequired("config_source_not_installed")
        path_helper = function.__name__ in {
            "_prepare_config_parent",
            "_read_raw_cli_config_unlocked",
            "_write_raw_cli_config_unlocked",
            "_try_read_cli_config_serialized_unlocked",
            "_read_cli_config_serialized_unlocked",
        }
        target = (
            (args[0] if args else kwargs.get("config_path")) if path_helper else None
        )
        with operation(source, target=target):
            return function(*args, **kwargs)

    # Definition-time body/closure provenance is refusal metadata only. It is
    # retained before consumers can install custom readers; the guard itself
    # keeps its preceding direct-call behavior.
    wrapped._config_guarded_body = (
        function,
        function.__code__,
        function.__globals__,
        function.__defaults__,
        function.__kwdefaults__,
        function.__closure__,
        tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
        wrapped.__closure__,
        tuple((cell, cell.cell_contents) for cell in wrapped.__closure__ or ()),
    )
    return wrapped


def check_lock_wait(source, operation):
    from . import raw_participants as raw

    state = raw._check(operation)
    if state.source is not source:
        raise bootstrap.RecoveryRequired("config_source_not_installed")
    if storage._pause is not None or (
        state.participant is not None
        and raw._participant_state(state.participant).closed
    ):
        raise bootstrap.RecoveryRequired("storage_locally_paused")
    if any(
        hold is not None and hold.authority.pause_requested(hold.names)
        for hold in state.holds
    ):
        raise bootstrap.RecoveryRequired("storage_locally_paused")
