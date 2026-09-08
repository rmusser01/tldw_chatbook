"""Concrete configuration source lifetimes; no public maintenance authority."""

from contextlib import contextmanager
from functools import wraps
import secrets
import sys

from . import bootstrap, profile_paths, storage_admission as storage

ROUTES = {
    "config",
    "config_snapshot",
    "config_data",
    "config_chat_dicts",
    "config_models",
}
_STATE_NAMES = (
    "_CONFIG_CACHE",
    "_CONFIG_CACHE_SOURCE",
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
    data = source._CONFIG_CACHE
    if data is None or source._CONFIG_CACHE_SOURCE != selected:
        raise bootstrap.RecoveryRequired("config_directory_selection_unavailable")
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
