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
            profile_paths.default_base_data_dir().parents[2] / ".tldw_cli-data-root.lock"
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
        guard.enter_context(hold.authority._lock(
            parent, "registry.lock", fcntl.LOCK_SH, cancel=attempt.cancel
        ))
        attempt.check()
        pending, profiles = bootstrap._records(root)
        record = next((row for row in profiles if row["selector"] == str(selected)), None)
        if record is None or tuple(record["namespaces"]) != hold.names:
            guard.close()
            return None
        if (
            pending
            or hold.authority.control_root != root / "admission"
            or storage._scope(root, selected, selected, authority=hold.authority) != hold.names
            or (
                sibling_selector(state.source, state.route, state.selected) != selected
                if state.config_anchor is not None
                else binding(state.source) != ("config", selected, True)
                or state.route not in {"config", "config_snapshot"}
            )
            or not state.pinned
        ):
            raise bootstrap.RecoveryRequired("config_companion_scope_changed")
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
                tokens.add(f"inode:{member_info.st_dev}:{member_info.st_ino}")
            if any(
                tokens.intersection(entry["historical"])
                or any(bootstrap._overlap(member, Path(path)) for path in entry["roots"])
                or any(
                    bootstrap._overlap(member, Path(token[5:]))
                    for token in entry["historical"] if token.startswith("path:")
                )
                for entry in foreign
            ):
                raise bootstrap.RecoveryRequired("config_companion_scope_changed")
        # Existing parent posture is verified, never delegated for creation.
        state.directories = ()
        state.companion_roots = tuple(Path(path) for path in record["roots"])
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
def operation(source, *, route="config", target=None):
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
    lock = source._config_file_lock()
    acquired = False
    before = None
    entered = False
    try:
        attempt = storage._Acquisition()
        while not lock.acquire(timeout=0.05):
            attempt.check()
        acquired = True
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
        if acquired:
            lock.release()
        if attempt is not None:
            attempt.close()
        if core is not None:
            storage._check_operation(core, core.path)
        storage._operation_local.operation = core


def guarded(function):
    """Enclose the actual config reader/cache bodies, including direct helpers."""
    allowed = {
        "load_settings",
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
        "get_runtime_config_snapshot",
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
