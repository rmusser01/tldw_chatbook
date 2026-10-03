"""Exact MCP local persistence sources; no transport or execution authority."""

import json
import secrets
import stat
import sys
import threading
import weakref
from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import MappingProxyType

from tldw_chatbook.Utils.platform_files import os

from ..Utils.file_durability import flush_directory, fsync_parent_directory
from . import bootstrap, profile_paths
from . import storage_admission as storage

ROUTE = "mcp_store"
_BINDINGS = weakref.WeakKeyDictionary()
_JSON_CACHE_BYTES = 1024 * 1024
_JSON_CACHE_ENTRIES = 16
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
    from ..Utils import private_paths
    from . import raw_participants as raw

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
    if config._CONFIG_CACHE_SOURCE == profile and selected == _selected(
        source,
        owner,
        profile_paths.lexical_path(profile_paths.user_data_dir(data) / leaf),
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
            or _selected(
                source,
                owner,
                profile_paths.lexical_path(profile_paths.user_data_dir(data) / leaf),
            )
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
            else os.stat(path, follow_symlinks=False)
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
    state.mcp_json_updates = {}


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
    if state.mcp_effects:
        _discard_json(state)
        return
    # The raw scope is retired here. Do not revive it or call raw._check.
    for key, (policy, raw_bytes, frozen) in state.mcp_json_updates.items():
        try:
            current = _json_context(state, policy)
        except (OSError, ValueError, RuntimeError):
            continue  # Optional evidence cannot replace the caller's result.
        if current is None or current[0] != key:
            continue
        with storage._lock:
            if not _json_holds_current(state, current[1]):
                continue
            cache = current[1][0].json_evidence
            cache[key] = (raw_bytes, frozen)
            cache.move_to_end(key)
            while len(cache) > _JSON_CACHE_ENTRIES:
                cache.popitem(last=False)


def _json_holds_current(state, holds):
    return (
        state.pid == os.getpid()
        and not state.uncertain
        and not state.mcp_effects
        and storage._pause is None
        and all(
            storage._holds.get(hold.key) is hold and storage._hold_serving(hold)
            for hold in holds
        )
    )


def _json_context(state, policy):
    """Current selection outside the mutex; surviving Hold identity inside it."""
    if os.name == "nt" or state.participant is None or not state.pinned:
        return None
    source = state.source
    selected = binding(source)
    if selected is None or not selected[2] or selected[0] == "mcp.history":
        return None
    execution = storage._execution_selection_for(state.selected)
    holds = tuple(dict.fromkeys(state.holds))
    if not holds or None in holds:
        return None
    with storage._lock:
        if not _json_holds_current(state, holds):
            return None
        groups = tuple(
            (
                hold,
                hold.json_generation,
                hold.names,
                hold.authority._observed_groups.get(hold.names),
            )
            for hold in holds
        )
        if any(not group[3] for group in groups):
            return None
        key = (
            weakref.ref(source),
            _BINDINGS.get(source),
            _BINDINGS[source].config._CONFIG_GENERATION,
            policy,
            execution,
            bootstrap._admission_epoch,
            groups,
        )
        return key, holds


def _discard_json(state):
    state.mcp_json_updates.clear()
    _discard_source_json(state.source, state.holds)


def _discard_source_json(source, holds):
    with storage._lock:
        for hold in holds:
            if hold is not None:
                for key in tuple(hold.json_evidence):
                    if key[0]() is source:
                        del hold.json_evidence[key]


def discard_json(source):
    """A caller's failed shape validation must not publish parsed evidence."""
    _discard_json(_operation(source)[1])


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if type(value) in (str, int, float, bool, type(None)):
        return value
    raise TypeError("json_cache_ineligible")


def _thaw_json(value):
    if isinstance(value, MappingProxyType):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def read_json(source, *, policy="json", exact_bytes=False, **parser_options):
    """Read every current byte; reuse only a positively retired immutable parse.

    The byte limit controls eligibility, not the stores' existing read/error policy.
    Validation and fallback defaults remain at each caller.
    """
    _, state = _operation(source)
    parser = json.loads
    policy = (policy, exact_bytes, parser, tuple(sorted(parser_options.items())))
    try:
        if exact_bytes and state.participant is None:
            raw_bytes = source.path.read_bytes()
            text = raw_bytes.decode("utf-8")
        else:
            with reader(source) as handle:
                if state.participant is not None:
                    raw_bytes = handle.buffer.read()
                    text = raw_bytes.decode("utf-8")
                    if not exact_bytes:
                        text = text.replace("\r\n", "\n").replace("\r", "\n")
                else:
                    text = handle.read()
                    raw_bytes = text.encode("utf-8")
        # reader's positive close precedes cache lookup and staging.
        _operation(source)
        check_destination(state, state.selected)
        context = _json_context(state, policy)
        eligible = context is not None and len(raw_bytes) <= _JSON_CACHE_BYTES
        if eligible:
            key, holds = context
            with storage._lock:
                entry = holds[0].json_evidence.get(key)
            if entry is not None and entry[0] == raw_bytes:
                return raw_bytes, _thaw_json(entry[1])
        payload = parser(text, **parser_options)
        if eligible:
            try:
                frozen = _freeze_json(payload)
            except (RecursionError, TypeError):
                pass  # Preserve a successfully parsed, cache-ineligible payload.
            else:
                state.mcp_json_updates[key] = (policy, raw_bytes, frozen)
        return raw_bytes, payload
    except BaseException:
        _discard_json(state)
        raise


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
            if state is not None:
                _discard_json(state)
            else:
                with storage._lock:
                    _discard_source_json(source, tuple(storage._holds.values()))
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
    import json

    from . import raw_participants as raw

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
            flush_directory(fd)
        else:
            os.replace(state.selected, backup)
            fsync_parent_directory(backup.parent)
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
