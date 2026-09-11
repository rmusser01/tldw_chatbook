"""Exact MCP local persistence sources; no transport or execution authority."""

from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import wraps
import os
from pathlib import Path
import secrets
import stat
import sys
import threading
import weakref

from . import bootstrap, profile_paths, storage_admission as storage

ROUTE = "mcp_store"
_BINDINGS = weakref.WeakKeyDictionary()
_SOURCES = (
    ("local_store", "LocalMCPStore", "mcp.local", "local_mcp_store.json"),
    (
        "server_target_store",
        "ConfiguredServerTargetStore",
        "mcp.targets",
        "mcp_server_targets.json",
    ),
    (
        "unified_context_store",
        "UnifiedMCPContextStore",
        "mcp.context",
        "unified_mcp_context.json",
    ),
    (
        "permission_store",
        "MCPPermissionStore",
        "mcp.permissions",
        "mcp_permissions.json",
    ),
    ("execution_log", "MCPExecutionLog", "mcp.history", "mcp_execution_log.jsonl"),
)


@dataclass(frozen=True)
class _Binding:
    source_type: type
    config: object
    profile: Path
    selected: Path


def _selected(source, owner, canonical):
    if owner in {"mcp.local", "mcp.permissions", "mcp.context"}:
        from ..MCP.recovery_activation import selected_path

        if getattr(source, "_recovery_original_path", None) != canonical:
            return canonical
        return selected_path(canonical)
    return canonical


def _kind(source):
    for module, name, owner, leaf in _SOURCES:
        cls = getattr(sys.modules.get("tldw_chatbook.MCP." + module), name, None)
        if cls is not None and isinstance(source, cls):
            return cls, owner, leaf
    return None


def bind(source):
    """Capture already-selected real configuration, never load custom config."""
    if source in _BINDINGS:
        raise bootstrap.RecoveryRequired("mcp_source_already_bound")
    cls, owner, leaf = _kind(source)
    source._mcp_source_lock = getattr(
        source, "_mutation_lock", getattr(source, "_lock", threading.RLock())
    )
    source._mcp_persistence_error = None
    config = sys.modules.get("tldw_chatbook.config")
    data = getattr(config, "_CONFIG_CACHE", None)
    from . import raw_participants as raw
    from ..Utils import private_paths

    if not raw._pinned_io_available() or (
        owner == "mcp.history"
        and not (
            private_paths._posix_guards_available()
            and private_paths._atomic_posix_guards_available()
        )
    ):
        return
    if type(source) is not cls or data is None:
        return
    selected = profile_paths.lexical_path(source.path)
    profile = profile_paths.lexical_path(config._get_effective_config_path())
    if (
        config._CONFIG_CACHE_SOURCE == profile
        and selected
        == _selected(source, owner, profile_paths.lexical_path(profile_paths.user_data_dir(data) / leaf))
    ):
        _BINDINGS[source] = _Binding(cls, config, profile, selected)


def binding(source):
    kind = _kind(source)
    if kind is None:
        return None
    cls, owner, leaf = kind
    selected = profile_paths.lexical_path(source.path)
    bound = _BINDINGS.get(source)
    if bound is not None:
        from ..Utils import private_paths

        if owner == "mcp.history" and not (
            private_paths._posix_guards_available()
            and private_paths._atomic_posix_guards_available()
        ):
            raise bootstrap.RecoveryRequired("mcp_source_selection_changed")
        config = bound.config
        data = config._CONFIG_CACHE
        if (
            type(source) is not bound.source_type
            or cls is not bound.source_type
            or selected != bound.selected
            or sys.modules.get("tldw_chatbook.config") is not config
            or profile_paths.lexical_path(config._get_effective_config_path())
            != bound.profile
            or config._CONFIG_CACHE_SOURCE != bound.profile
            or data is None
            or _selected(source, owner, profile_paths.lexical_path(profile_paths.user_data_dir(data) / leaf))
            != selected
        ):
            raise bootstrap.RecoveryRequired("mcp_source_selection_changed")
    return owner, selected, bound is not None


def selection(source):
    bound = binding(source)
    if bound is None:
        raise bootstrap.RecoveryRequired("mcp_source_not_supported")
    return bound[1], bound[2], False


def members(source, selected):
    owner = binding(source)[0]
    if owner == "mcp.history":
        targets = (selected, selected.with_name(selected.name + ".1"))
        temps = {p: p.parent / f".{p.name}.{secrets.token_hex(8)}.tmp" for p in targets}
        return targets + tuple(temps.values()), temps
    paths = (selected, selected.with_suffix(selected.suffix + ".tmp"))
    if owner == "mcp.permissions":
        paths += (selected.with_suffix(selected.suffix + ".bak"),)
    return paths, {}


def _identity(state, path):
    try:
        info = (
            os.stat(path.name, dir_fd=state.pins[path.parent], follow_symlinks=False)
            if state.pinned and path.parent in state.pins
            else path.lstat()
        )
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise bootstrap.RecoveryRequired("raw_not_regular")
    return info.st_dev, info.st_ino


def preflight(state):
    owner = binding(state.source)[0]
    targets = (state.selected,)
    if owner == "mcp.permissions":
        targets += (state.selected.with_suffix(state.selected.suffix + ".bak"),)
    elif owner == "mcp.history":
        targets += (state.selected.with_name(state.selected.name + ".1"),)
    state.mcp_identities = (
        {p: _identity(state, p) for p in targets}
        if state.participant is not None
        else {}
    )
    state.observed_files.update(
        {
            p: identity
            for p, identity in state.mcp_identities.items()
            if identity is not None
        }
    )
    state.mcp_effects = False
    state.mcp_publications = {}
    state.mcp_payload_updates = []


def check_destination(state, path):
    path = profile_paths.lexical_path(path)
    if (
        path in state.mcp_identities
        and _identity(state, path) != state.mcp_identities[path]
    ):
        raise bootstrap.RecoveryRequired("raw_entry_identity_changed")


def published(state, path):
    state.mcp_effects = True
    if state.participant is None:
        return
    path = profile_paths.lexical_path(path)
    identity = _identity(state, path)
    expected = state.mcp_publications.pop(path)
    if identity != expected:
        raise bootstrap.RecoveryRequired("raw_entry_identity_changed")
    state.mcp_identities[path] = identity
    state.observed_files.pop(path, None)
    if state.mcp_identities[path] is not None:
        state.observed_files[path] = state.mcp_identities[path]


def complete(state):
    """Publish caller stamps after positive native close, under the source lock."""
    for payload, stamp in state.mcp_payload_updates:
        payload["updated_at"] = stamp


def drain_ready(source):
    return source._mcp_persistence_error is None


def guarded(function):
    """Decorate only actual instance methods on the five concrete source classes."""

    @wraps(function)
    def wrapped(source, *args, **kwargs):
        from . import raw_participants as raw

        module = sys.modules.get(function.__module__)
        cls = getattr(module, function.__qualname__.split(".")[0], None)
        if (
            cls is None
            or cls.__dict__.get(function.__name__) is not wrapped
            or not isinstance(source, cls)
        ):
            raise bootstrap.RecoveryRequired("mcp_source_not_supported")
        if function.__name__ == "__init__":
            with closing(storage._Acquisition()):
                if source in _BINDINGS:
                    raise bootstrap.RecoveryRequired("mcp_source_already_bound")
                result = function(source, *args, **kwargs)
                bind(source)
                with raw._scope(source, ROUTE, writing=True):
                    return result
        operation = None
        state = None
        try:
            with raw._scope(source, ROUTE, writing=True) as operation:
                state = raw._states[operation]
                result = function(source, *args, **kwargs)
            return result
        except BaseException:
            if state is not None and (state.mcp_effects or state.uncertain):
                source._mcp_persistence_error = "mcp_persistence_incomplete"
            raise

    return wrapped


def _operation(source):
    from . import raw_participants as raw

    operation = getattr(raw._local, "operation", None)
    state = raw._check(operation)
    if state.source is not source or state.route != ROUTE:
        raise bootstrap.RecoveryRequired("mcp_source_not_supported")
    return operation, state


@contextmanager
def reader(source):
    from . import raw_participants as raw

    operation, state = _operation(source)
    if state.participant is None:
        with source.path.open("r", encoding="utf-8") as handle:
            yield handle
        return
    check_destination(state, state.selected)
    with raw._file(operation, state.selected, "r") as handle:
        yield handle


def write_json(source, payload):
    from . import raw_participants as raw
    import json

    operation, state = _operation(source)
    if state.participant is None:
        source.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = source.path.with_suffix(source.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temporary.replace(source.path)
        state.mcp_effects = True
        return
    raw._mkdirs(operation)
    temporary = state.selected.with_suffix(state.selected.suffix + ".tmp")
    try:
        with raw._file(operation, temporary, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        check_destination(state, state.selected)
        raw._replace(operation, temporary, state.selected)
    finally:
        if not state.uncertain:
            raw._remove_temporary(operation, temporary)


def stamp_payload(source, payload, stamp):
    _, state = _operation(source)
    state.mcp_payload_updates.append((payload, stamp))


def backup_corrupt(source):
    from . import raw_participants as raw

    operation, state = _operation(source)
    backup = state.selected.with_suffix(state.selected.suffix + ".bak")
    if state.participant is None:
        source.path.replace(backup)
        state.mcp_effects = True
        return
    raw._check(operation, backup, writing=True)
    check_destination(state, state.selected)
    check_destination(state, backup)
    try:
        if state.pinned:
            fd = state.pins[state.selected.parent]
            os.replace(state.selected.name, backup.name, src_dir_fd=fd, dst_dir_fd=fd)
        else:
            os.replace(state.selected, backup)
    except BaseException:
        state.uncertain = True
        raise bootstrap.RecoveryRequired("raw_publication_uncertain") from None
    state.mcp_publications[state.selected] = None
    state.mcp_publications[backup] = state.mcp_identities[state.selected]
    published(state, state.selected)
    published(state, backup)


def history_operation(state):
    kind = _kind(state.source)
    return kind is not None and kind[1] == "mcp.history" and state.route == ROUTE


def helper_allowed(state, helper, selected):
    """The fixed native helpers actually used by MCPExecutionLog only."""
    if not history_operation(state):
        return False
    selected = profile_paths.lexical_path(selected)
    if helper == "secure_private_directory":
        return selected == state.selected.parent and selected in state.directories
    if helper in {"open_private_binary", "atomic_private_write_bytes"}:
        return selected in state.temporaries
    if helper == "open_private_text_append_stream":
        return selected == state.selected
    return False
