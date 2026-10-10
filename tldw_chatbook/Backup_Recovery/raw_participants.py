"""Bounded installed raw-file lifetimes; never recovery/capture authority (ADR-126)."""

import copy
import inspect
import io
import secrets
import stat
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from types import ModuleType

from tldw_chatbook.Utils.platform_files import os

from ..Utils.file_durability import (
    flush_directory,
    flush_file,
    fsync_parent_directory,
)
from ..Utils.private_paths import _open_verified_parent, _posix_guards_available
from . import bootstrap
from . import chat_source_participants as chat_sources
from . import config_participants as config_files
from . import dictionary_file_participants as dictionary_files
from . import mcp_source_participants as mcp_sources
from . import settings_file_participants as settings_files
from . import storage_admission as storage
from .profile_paths import lexical_path, user_data_dir
from .admission import _ORDINARY_PAUSE_REQUEST_BINDING


@dataclass
class _ParticipantState:
    source: object
    source_type: type
    owner: str
    selected: Path
    closed: bool = False


_participants = weakref.WeakKeyDictionary()
_source_participants = weakref.WeakKeyDictionary()
_path_locks = weakref.WeakValueDictionary()
_local = threading.local()


class _RawParticipant:
    __slots__ = ("__weakref__",)

    def __init__(self):
        raise TypeError("raw_participant_is_installed")

    @property
    def owner_id(self):
        with storage._lock:
            return _participant_identity(self).owner

    def close_admission(self):
        with storage._changed:
            state = _participant_identity(self)
            state.closed = True

    def drain(self, deadline):
        with storage._changed:
            checked = _participant_identity(self)
            if not checked.closed:
                raise RuntimeError("participant_admission_not_closed")
            while (
                any(s.participant is self for s in _states.values())
                or storage._pending_acquisitions
                or storage._retiring_holds
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                storage._changed.wait(min(remaining, 0.05))
        state = checked
        source = state.source()
        if state.owner.startswith("mcp."):
            ready = mcp_sources.drain_ready(source)
        elif state.owner == "chat.dictionaries":
            ready = dictionary_files.drain_ready(source)
        elif state.owner in {"personas", "chat.dictionary_history", "chat.rag_context"}:
            ready = chat_sources.drain_ready(source)
        elif state.owner == "config":
            ready = (
                source._CONFIG_PERSISTENCE_ERROR is None
                and source.get_config_load_failure() is None
                and source._CONFIG_CACHE is not None
            )
        elif state.owner == "eval.definitions":
            ready = (
                source._config == source._persisted_config
                and source.persistence_error is None
            )
        elif state.owner == "ui.themes":
            ready = not source.is_modified
        else:
            ready = True
        with storage._changed:
            if _participant_identity(self) is not state or not state.closed:
                raise RuntimeError("participant_admission_not_closed")
            return (
                ready
                and not any(s.participant is self for s in _states.values())
                and not storage._pending_acquisitions
                and not storage._retiring_holds
            )

    def resume(self):
        with storage._changed:
            state = _participant_identity(self)
            if storage._pause is not None:
                raise bootstrap.RecoveryRequired("process_pause_still_active")
            state.closed = False


def _participant_identity(participant):
    """Check only installed identity mappings; never read source metadata."""
    if type(participant) is not _RawParticipant or participant not in _participants:
        raise bootstrap.RecoveryRequired("raw_participant_not_installed")
    state = _participants[participant]
    source = state.source()
    if source is None or type(source) is not state.source_type:
        raise bootstrap.RecoveryRequired("raw_participant_not_installed")
    if _source_participants.get(source) is not participant:
        raise bootstrap.RecoveryRequired("raw_participant_not_installed")
    return state


def _participant_state(participant):
    state = _participant_identity(participant)
    source = state.source()
    if state.owner.startswith("mcp."):
        bound = mcp_sources.binding(source)
        if bound is None or not bound[2] or bound[1] != state.selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    if state.owner == "chat.dictionaries":
        binding = dictionary_files.binding(source)
        if binding is None or binding[1] != state.selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    if state.owner in {"personas", "chat.dictionary_history", "chat.rag_context"}:
        binding = chat_sources.binding(source)
        if binding is None or binding[1] != state.selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    if state.owner == "config":
        binding = config_files.binding(source)
        if binding is None or binding[1] != state.selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    if state.owner in {
        "eval.definitions",
        "notes.templates",
        "ui.themes",
        "tamagotchi.config",
        "runtime.source_state",
    }:
        binding = settings_files.binding(source)
        if binding is None or not binding[2] or binding[1] != state.selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    if state.owner == "chat.prompt_history":
        # Bound-state checks run under the storage lock and during active IO.
        # The default resolver opens a guarded config operation (and creates
        # directories), so calling it here inverts config -> storage locking
        # and reenters the current raw operation. Validate the same canonical
        # mapping from the current bound cache, as the MCP sources do.
        config = sys.modules.get("tldw_chatbook.config")
        data = getattr(config, "_CONFIG_CACHE", None)
        if (
            data is None
            or config._CONFIG_CACHE_SOURCE != config._get_effective_config_path()
            or lexical_path(source.path) != state.selected
            or user_data_dir(data) / "prompt_history.jsonl" != state.selected
        ):
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    if (
        state.owner == "hooks.permissions"
        and _hook_permissions_selection(source) != state.selected
    ):
        raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    if state.owner == "ui.state":
        selected, installed = _async_source_selection(source, "sidebar_state")
        if not installed or selected != state.selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    if state.owner == "notes.file_notes_replica" and source.db_path != state.selected:
        raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    return state


def _hook_permissions_selection(source):
    module = sys.modules.get("tldw_chatbook.Agents.hook_permissions")
    if module is None or type(source) is not module.HookPermissions:
        raise bootstrap.RecoveryRequired("raw_source_not_supported")
    config = sys.modules.get("tldw_chatbook.config")
    data = getattr(config, "_CONFIG_CACHE", None)
    if (
        data is None
        or config._CONFIG_CACHE_SOURCE != config._get_effective_config_path()
    ):
        raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    return lexical_path(user_data_dir(data) / "hook_permissions.json")


def _types():
    from ..Chat_Grammars_Interop.local_chat_grammars_service import (
        LocalChatGrammarsService,
    )
    from ..Feedback_Interop.local_feedback_service import LocalFeedbackService

    return {
        LocalFeedbackService: "feedback",
        LocalChatGrammarsService: "chat.grammars",
    }


def _async_source_selection(source, route, *, _require_installed=False):
    """Resolve only the three actual async file owners; labels grant no authority."""
    if route == "note_templates":
        selected, installed, _ = settings_files.selection(source, route, None)
    elif route == "prompt_history":
        from ..Chat.prompt_history import PromptHistory, default_prompt_history_path

        if not isinstance(source, PromptHistory):
            raise bootstrap.RecoveryRequired("raw_source_not_supported")
        selected = lexical_path(source.path)
        installed = type(source) is PromptHistory and selected == lexical_path(
            default_prompt_history_path()
        )
    else:
        module = sys.modules.get("tldw_chatbook.UI.Screens.chat_screen")
        cls = getattr(module, "ChatScreen", None)
        if route != "sidebar_state" or cls is None or not isinstance(source, cls):
            raise bootstrap.RecoveryRequired("raw_source_not_supported")
        selected = lexical_path(
            module._get_effective_config_path().parent / "ui_state.toml"
        )
        installed = type(source) is cls
    if _require_installed and not installed:
        raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    # An already bound source cannot silently become an ordinary custom source
    # to bypass its closed gate when configuration changes beneath it.
    participant = _source_participants.get(source)
    if participant is not None:
        state = _participants[participant]
        if not installed or state.selected != selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    return selected, installed


def _pinned_io_available():
    """Select a posture before IO; a failed native safety check never falls back."""
    return (
        _posix_guards_available()
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and {os.open, os.stat, os.mkdir, os.rename, os.unlink} <= os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
    )


def _register_raw_participant(source, owner, selected):
    """Register observed metadata; admission and file effects validate separately."""
    selected = lexical_path(selected)
    with storage._lock:
        participant = _source_participants.get(source)
        if participant is None:
            participant = object.__new__(_RawParticipant)
            _participants[participant] = _ParticipantState(
                weakref.ref(source), type(source), owner, selected
            )
            _source_participants[source] = participant
        state = _participant_identity(participant)
        if state.source() is not source:
            raise bootstrap.RecoveryRequired("raw_participant_not_installed")
        if state.owner != owner or state.selected != selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
        return participant


def _raw_participant(source):
    """Only installed actual sources; this does not qualify capture inventory."""
    if not _pinned_io_available():
        raise bootstrap.RecoveryRequired("raw_participant_not_installed")
    settings_binding = (
        mcp_sources.binding(source)
        or dictionary_files.binding(source)
        or chat_sources.binding(source)
        or config_files.binding(source)
        or settings_files.binding(source)
    )
    if settings_binding is not None:
        owner, selected, installed = settings_binding
        if not installed:
            raise bootstrap.RecoveryRequired("raw_participant_not_installed")
    else:
        from ..Notes.file_notes_replica import FileNotesReplica
        from ..Widgets import emoji_picker

        types = _types()
        from ..Chat.prompt_history import PromptHistory

        screen_module = sys.modules.get("tldw_chatbook.UI.Screens.chat_screen")
        screen_type = getattr(screen_module, "ChatScreen", None)
        if type(source) is PromptHistory or (
            screen_type is not None and type(source) is screen_type
        ):
            route = (
                "prompt_history" if type(source) is PromptHistory else "sidebar_state"
            )
            selected, installed = _async_source_selection(source, route)
            if not installed:
                raise bootstrap.RecoveryRequired("raw_participant_not_installed")
            owner = "chat.prompt_history" if route == "prompt_history" else "ui.state"
        elif type(source) is FileNotesReplica:
            owner, selected = "notes.file_notes_replica", source.db_path
        elif source is emoji_picker:
            owner, selected = "ui.emoji_recents", emoji_picker._recent_emojis_path()
        elif type(source).__module__ == "tldw_chatbook.Agents.hook_permissions":
            owner = "hooks.permissions"
            selected = _hook_permissions_selection(source)
        elif type(source) in types:
            owner = types[type(source)]
            selected = source.store_path
        else:
            raise bootstrap.RecoveryRequired("raw_participant_not_installed")
    selected = lexical_path(selected)
    participant = _register_raw_participant(source, owner, selected)
    state = _participant_state(participant)
    with storage._lock:
        if _participant_identity(participant) is not state:
            raise bootstrap.RecoveryRequired("raw_participant_not_installed")
        if state.selected != selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
        return participant


class _RawOperation:
    __slots__ = ("__weakref__",)

    def __init__(self):
        raise TypeError("raw_operation_is_source_issued")


@dataclass
class _State:
    source: object
    participant: object
    selected: Path
    paths: tuple[Path, ...]
    directories: tuple[Path, ...]
    writing: bool
    route: str
    pid: int
    thread: object
    task: object
    pinned: bool = True
    identities: dict = field(default_factory=dict)
    leases: list = field(default_factory=list)
    holds: list = field(default_factory=list)
    pins: dict = field(default_factory=dict)
    files: list = field(default_factory=list)
    descriptors: set = field(default_factory=set)
    created_files: dict = field(default_factory=dict)
    observed_files: dict = field(default_factory=dict)
    # theme_directory only: members refused as non-regular, never observed.
    rejected_files: dict = field(default_factory=dict)
    temporaries: dict = field(default_factory=dict)
    temporary: Path | None = None
    backup: Path | None = None
    backup_temporary: Path | None = None
    active: bool = False
    uncertain: bool = False
    config_failed: bool = False
    config_generation: int | None = None
    config_publication: tuple[Path, tuple[int, int]] | None = None
    companion_guard: object | None = None
    companion_roots: tuple[Path, ...] = ()
    config_anchor: Path | None = None
    mcp_canonical: Path | None = None
    mcp_observation_lease: object | None = None


_states = {}


def _live_state(operation, path, writing):
    """Validate issued operation and lease identities under the coordinator.

    This helper is deliberately free of selected-source and filesystem reads.
    """
    if type(operation) is not _RawOperation or operation not in _states:
        raise bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
    state = _states[operation]
    if (
        not state.active
        or operation not in storage._raw_operations
        or state.pid != os.getpid()
        or state.thread is not threading.current_thread()
        or state.task is not storage._task_identity()
        or state.uncertain
    ):
        raise bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
    if state.participant is None and storage._pause is not None:
        raise bootstrap.RecoveryRequired("storage_locally_paused")
    if writing and not state.writing:
        raise bootstrap.RecoveryRequired("raw_path_outside_scope")
    if path is not None and lexical_path(path) not in state.paths + state.directories:
        raise bootstrap.RecoveryRequired("raw_path_outside_scope")
    for lease, hold in zip(state.leases, state.holds):
        if (
            lease not in storage._live_leases
            or storage._holds.get(lease._key) is not hold
        ):
            raise bootstrap.RecoveryRequired("raw_native_scope_changed")
        if hold is not None and (hold.stop.is_set() or hold.error is not None):
            raise bootstrap.RecoveryRequired("raw_native_scope_changed")
        if storage._pause is not None and hold is None:
            raise bootstrap.RecoveryRequired("raw_native_scope_unqualified")
    return state


def _mcp_observation(source, canonical):
    """Supply only this installed source's currently issued canonical lease."""
    operation = getattr(_local, "operation", None)
    if operation is None:
        return None
    with storage._lock:
        state = _live_state(operation, None, False)
        if (
            state.source is not source
            or state.route != mcp_sources.ROUTE
            or state.participant is None
        ):
            return None
        participant = _participant_identity(state.participant)
        if (
            participant.source() is not source
            or state.mcp_canonical != lexical_path(canonical)
            or state.mcp_observation_lease not in state.leases
        ):
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
        lease = state.mcp_observation_lease
    # This existing gate checks the lease's exact canonical selection. A lease
    # admitted for a recovered destination cannot substitute for the canonical.
    lease.execution_context(canonical)
    with storage._lock:
        if (
            _live_state(operation, None, False) is not state
            or _participant_identity(state.participant) is not participant
            or state.mcp_observation_lease is not lease
            or state.source is not source
            or state.mcp_canonical != lexical_path(canonical)
        ):
            raise bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
    return {lexical_path(canonical): lease}


@contextmanager
def _source_custody_check(operation, path=None, *, writing=False):
    # No virtual validator dispatch and no mutable caller token fields.
    with storage._lock:
        state = _live_state(operation, path, writing)
        participant = state.participant
        source = state.source
    # Binding validation may read native recovery records. It must not
    # monopolize the coordinator while unrelated admitted workers use it.
    participant_state = (
        _participant_state(participant) if participant is not None else None
    )
    if (
        state.config_anchor is not None
        and config_files.sibling_selector(state.source, state.route, state.selected)
        != state.config_anchor
    ):
        raise bootstrap.RecoveryRequired("config_companion_scope_changed")
    if state.route in {
        "config_data",
        "config_default_root",
        "config_chat_dicts",
        "config_models",
    }:
        config_files.selection(state.source, state.route, state.selected)
        if state.source._CONFIG_GENERATION != state.config_generation:
            raise bootstrap.RecoveryRequired("config_directory_generation_changed")
    yield state
    # Revocation/retirement can interleave with either source or parent proof.
    # A valid earlier observation cannot substitute for current issued custody.
    with storage._lock:
        if (
            _live_state(operation, path, writing) is not state
            or state.participant is not participant
            or state.source is not source
        ):
            raise bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
        if participant is not None and (
            type(participant) is not _RawParticipant
            or _participants.get(participant) is not participant_state
            or participant_state.source() is not source
            or type(source) is not participant_state.source_type
            or _source_participants.get(source) is not participant
        ):
            raise bootstrap.RecoveryRequired("raw_participant_not_installed")


def _check_parent_pins(state):
    # Disk checks never occur under the coordinator lock. Native descriptors pin
    # destinations even if an external nonparticipant renames after this check.
    # Existing parent aliases are valid only while they resolve to the same
    # positively checked physical directory; leaf aliases remain refused.
    for directory, identity in tuple(state.identities.items()):
        info = os.stat(directory)
        if (info.st_dev, info.st_ino) != identity or not stat.S_ISDIR(info.st_mode):
            raise bootstrap.RecoveryRequired("raw_parent_identity_changed")
    for directory, fd in tuple(state.pins.items()):
        info = os.stat(directory)
        pinned = os.fstat(fd)
        if (info.st_dev, info.st_ino) != (pinned.st_dev, pinned.st_ino):
            raise bootstrap.RecoveryRequired("raw_parent_identity_changed")
        if state.companion_guard is not None and (
            info.st_uid != os.geteuid() or info.st_mode & 0o077
        ):
            raise bootstrap.RecoveryRequired("config_companion_parent_unsafe")


def _check(operation, path=None, *, writing=False):
    with _source_custody_check(operation, path, writing=writing) as state:
        _check_parent_pins(state)
    return state


def _parent_walk_operation():
    """Select an installed stock owner without repeating native preparation."""
    operation = getattr(_local, "operation", None)
    with storage._lock:
        state = _states.get(operation)
        if state is None or state.participant is None:
            return None
        participant = _participant_identity(state.participant)
        if participant.owner not in {"config", "hooks.permissions", "mcp.history"}:
            return None
        _live_state(operation, None, True)
        return operation, state, state.source, state.participant


def _check_directory_allocation(operation, expected_state, source, participant):
    """Keep source/custody gates while the caller checks the directory walk."""
    if getattr(_local, "operation", None) is not operation:
        raise bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
    with _source_custody_check(operation, writing=True) as state:
        if (
            state is not expected_state
            or state.source is not source
            or state.participant is not participant
        ):
            raise bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
    if getattr(_local, "operation", None) is not operation:
        raise bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
    return state


def _owned_descriptor_retirement_state(fd: int) -> _State | None:
    """Find exact creator custody for cleanup without re-admitting an effect."""
    return _descriptor_retirement_state(getattr(_local, "operation", None), fd)


def _descriptor_retirement_state(operation, fd: int) -> _State | None:
    """Validate exact creator custody for ambient or finite-walk cleanup."""
    if operation is None:
        return None
    with storage._lock:
        if type(operation) is not _RawOperation or operation not in _states:
            return None
        state = _states[operation]
        if (
            not state.active
            or operation not in storage._raw_operations
            or state.pid != os.getpid()
            or state.thread is not threading.current_thread()
            or state.task is not storage._task_identity()
        ):
            raise bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
        # Source/path revocation must not prevent exact owned cleanup. An
        # uncertain close still follows _close_descriptor's retained outcome.
        descriptor_type = type(fd)
        return state if descriptor_type is int and fd in state.descriptors else None


def _close_descriptor(state, fd):
    state.descriptors.add(fd)
    if state.uncertain:
        raise bootstrap.RecoveryRequired("raw_resources_not_retired")
    try:
        os.close(fd)
        state.descriptors.remove(fd)
    except BaseException:
        state.uncertain = True
        raise bootstrap.RecoveryRequired("raw_resources_not_retired") from None


def _retire(state):
    if state.uncertain or state.files or state.descriptors:
        return False
    for path, fd in tuple(state.pins.items()):
        _close_descriptor(state, fd)
        del state.pins[path]
    if state.companion_guard is not None:
        state.companion_guard.close()
        state.companion_guard = None
    for lease in reversed(state.leases):
        lease.close()
    return True


def _selection(source, route, template, user_template, selected_read):
    if route == "hook_permissions":
        selected = _hook_permissions_selection(source)
        if selected_read is not None and lexical_path(selected_read) != selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
        return selected, True, False
    if route == mcp_sources.ROUTE:
        return mcp_sources.selection(source)
    if route == dictionary_files.ROUTE:
        return dictionary_files.selection(source)
    if route in chat_sources.ROUTES:
        return chat_sources.selection(source, route, selected_read)
    if route in config_files.ROUTES:
        return config_files.selection(source, route, selected_read)
    if route in settings_files.ROUTES:
        return settings_files.selection(source, route, selected_read)
    if route in {"prompt_history", "sidebar_state"}:
        selected, installed = _async_source_selection(source, route)
        if selected_read is not None and lexical_path(selected_read) != selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
        return selected, installed, False
    if route == "file_notes_directory":
        from ..Notes.file_notes_replica import FileNotesReplica

        if not isinstance(source, FileNotesReplica) or source.is_memory_db:
            raise bootstrap.RecoveryRequired("raw_source_not_supported")
        return lexical_path(source.db_path), type(source) is FileNotesReplica, False
    types = _types()
    owner = types.get(type(source))
    installed = owner is not None
    if route == "service":
        # Subclasses retain ordinary calls but receive no installed descendants.
        if not any(
            isinstance(source, cls)
            for cls, name in types.items()
            if name in {"feedback", "chat.grammars"}
        ):
            raise bootstrap.RecoveryRequired("raw_source_not_supported")
        return lexical_path(source.store_path), installed, False
    from ..Widgets import emoji_picker

    if route != "emoji" or source is not emoji_picker:
        raise bootstrap.RecoveryRequired("raw_source_not_supported")
    return lexical_path(emoji_picker._recent_emojis_path()), True, False


def _related_member_acquirer():
    """Use only the original acquisition callback for installed member batches."""
    binding = storage._RAW_MEMBER_ACQUIRE_BINDING
    acquire, code, defining, defaults, keywords, items = binding
    if (
        storage.acquire_storage is not acquire
        or acquire.__code__ is not code
        or acquire.__globals__ is not defining
        or defining is not vars(storage)
        or acquire.__defaults__ is not defaults
        or acquire.__kwdefaults__ is not keywords
        or (
            keywords is not None
            and (
                len(keywords) != len(items)
                or any(
                    key not in keywords or keywords[key] is not value
                    for key, value in items
                )
            )
        )
    ):
        return None
    return acquire


def _pending_mcp_acquirer():
    """Qualify the original canonical observation callbacks without calling them."""
    from ..MCP import recovery_activation as activation

    acquire = _related_member_acquirer()
    if (
        acquire is None
        or sys.modules.get("tldw_chatbook.MCP.recovery_activation") is not activation
        or getattr(sys.modules.get("tldw_chatbook.MCP"), "recovery_activation", None)
        is not activation
        or activation.acquire_storage is not acquire
        or vars(activation) is not getattr(activation, "_PENDING_OBSERVATION_NAMESPACE", None)
    ):
        return None
    for name, function, bodies in activation._PENDING_OBSERVATION_BINDINGS:
        if getattr(activation, name) is not function:
            return None
        if (
            len(bodies) > 1
            and getattr(function, "__wrapped__", None) is not bodies[1][0]
        ):
            return None
        for (
            current,
            code,
            defining,
            defaults,
            keywords,
            items,
            closure,
            cells,
        ) in bodies:
            if (
                current.__code__ is not code
                or current.__globals__ is not defining
                or current.__defaults__ is not defaults
                or current.__kwdefaults__ is not keywords
                or current.__closure__ is not closure
                or any(cell.cell_contents is not value for cell, value in cells)
                or (
                    keywords is not None
                    and (
                        len(keywords) != len(items)
                        or any(
                            key not in keywords or keywords[key] is not value
                            for key, value in items
                        )
                    )
                )
            ):
                return None
    return acquire


def _check_pending_mcp_preparation(preparation, source, canonical):
    """Fence one exact pending owner around a fresh canonical witness read."""
    if type(preparation) is not tuple or len(preparation) != 5:
        raise bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
    attempt, issued_source, binding, selected, lease = preparation

    def current():
        with storage._lock:
            if (
                type(attempt) is not storage._Acquisition
                or attempt not in storage._pending_acquisitions
                or getattr(attempt, "_mcp_preparation", None) is not preparation
                or getattr(_local, "pending_mcp_preparation", None) is not preparation
                or attempt.pid != os.getpid()
                or attempt.thread is not threading.current_thread()
                or attempt.task is not storage._task_identity()
                or attempt.operation is not None
                or getattr(storage._operation_local, "operation", None) is not None
                or getattr(_local, "operation", None) is not None
                or issued_source is not source
                or mcp_sources._BINDINGS.get(source) is not binding
                or type(source) is not binding.source_type
                or canonical != selected
                or getattr(source, "_recovery_original_path", None) != selected
                or lease not in storage._live_leases
            ):
                raise bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
            if lexical_path(source.path) != binding.selected:
                raise bootstrap.RecoveryRequired("raw_source_selection_changed")
            attempt.check()
        if (
            _pending_mcp_acquirer() is None
            or mcp_sources.canonical_path(source) != selected
        ):
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")

    current()
    lease.execution_context(selected)
    current()
    return lease


def _begin_mcp_preparation(source, attempt):
    """Own one canonical lease before selectors, leaving member admission separate."""
    binding = mcp_sources._BINDINGS.get(source)
    if (
        binding is None
        or type(source) is not binding.source_type
        or mcp_sources._source_owner(source)
        not in {"mcp.local", "mcp.permissions", "mcp.context"}
        or (acquire := _pending_mcp_acquirer()) is None
    ):
        return None
    canonical = mcp_sources.canonical_path(source)
    if getattr(source, "_recovery_original_path", None) != canonical:
        return None
    attempt.check()
    lease = acquire(canonical)
    # Publish cleanup ownership before any subsequent source or lease validation.
    preparation = (attempt, source, binding, canonical, lease)
    attempt._mcp_preparation = preparation
    _local.pending_mcp_preparation = preparation
    _check_pending_mcp_preparation(preparation, source, canonical)
    return preparation


@contextmanager
def _pending_mcp_observation(source, canonical):
    """Retain admission only; selected_path still observes fresh witnesses."""
    preparation = getattr(_local, "pending_mcp_preparation", None)
    if preparation is None:
        yield None
        return
    lease = _check_pending_mcp_preparation(preparation, source, canonical)
    try:
        yield {canonical: lease}
    finally:
        _check_pending_mcp_preparation(preparation, source, canonical)


def _nested_installed_mcp_state(source, route, previous):
    """Inspect issued MCP custody without repeating its native source proof."""
    if route != mcp_sources.ROUTE or type(previous) is not _RawOperation:
        return None
    with storage._lock:
        state = _states.get(previous)
        if (
            state is None
            or state.source is not source
            or state.route != mcp_sources.ROUTE
            or state.participant is None
        ):
            return None
        state = _live_state(previous, None, False)
        participant = _participant_identity(state.participant)
        if participant.source() is not source:
            raise bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
        if not participant.owner.startswith("mcp."):
            return None
        return state


def _pin_parent(state, anchor):
    """Reuse a complete parent proof; each operation still owns its fresh FD."""
    from tldw_chatbook.Utils import private_paths

    hold = next((h for h in state.holds if h is not None), None)
    # Derived evidence is optional; keep actual operation holds untouched.
    if hold is not None and storage._ordinary_hold(hold.authority) is not hold:
        hold = None
    key = ("raw-pin", str(anchor))
    before = storage._derived_before(hold, key)
    reused, posture = storage._derived_reuse(hold, key, before)
    close = lambda fd: _close_descriptor(state, fd)
    if reused:
        fd = private_paths._native_open(
            anchor, private_paths._DIRECTORY_OPEN_FLAGS | private_paths._NOFOLLOW
        )
        try:
            info = os.fstat(fd)
            current = (
                info.st_dev,
                info.st_ino,
                stat.S_IFMT(info.st_mode),
                stat.S_IMODE(info.st_mode),
                info.st_uid,
            )
            # Complete pathname ancestry must still match AFTER acquisition.
            after = storage._derived_before(hold, key)
            valid, _ = storage._derived_reuse(hold, key, after)
            if valid and current == posture:
                return fd
        except BaseException:
            close(fd)
            raise
        close(fd)
    fd, _ = _open_verified_parent(
        anchor / ".raw-pin", missing_leaf_allowed=True, _close=close
    )
    try:
        evidence = (
            storage._path_evidence(hold.names, anchor) if hold is not None else None
        )
        info = os.fstat(fd)
        posture = (
            info.st_dev,
            info.st_ino,
            stat.S_IFMT(info.st_mode),
            stat.S_IMODE(info.st_mode),
            info.st_uid,
        )
        if evidence is not None and evidence.posture[-1][1] != posture:
            evidence = None
        storage._note_derived(hold, key, evidence, posture, before)
        return fd
    except BaseException:
        close(fd)
        raise


def _raw_holds_pause_requested(holds):
    """Probe each exact stock hold once; custom queries keep their original route."""
    (
        module,
        owner_type,
        original,
        code,
        namespace,
        defaults,
        keywords,
        closure,
        lookup,
        dictionary,
    ) = _ORDINARY_PAUSE_REQUEST_BINDING
    seen = []
    for hold in holds:
        if hold is None:
            continue
        authority, names = hold.authority, hold.names
        stock = (
            type(hold) is storage._Hold
            and type(module) is ModuleType
            and sys.modules.get("tldw_chatbook.Backup_Recovery.admission") is module
            and vars(module) is namespace
            and namespace.get("Admission") is owner_type
            and type(authority) is owner_type
            and vars(owner_type).get("pause_requested") is original
            and original.__code__ is code
            and original.__globals__ is namespace
            and original.__defaults__ is defaults
            and original.__kwdefaults__ is keywords
            and original.__closure__ is closure
            and vars(owner_type).get("__getattribute__") is lookup
            and vars(owner_type).get("__dict__") is dictionary
        )
        if stock:
            values = vars(authority)
            # A replaced instance dictionary can have custom membership behavior.
            stock = type(values) is dict and "pause_requested" not in values  # noqa: E721
        if stock and any(
            hold is previous and authority is owner and names is group
            for previous, owner, group in seen
        ):
            continue
        # Re-evaluate eligibility on every occurrence. Changed callbacks are
        # called through the original dynamic route, never skipped or blessed.
        if authority.pause_requested(names):
            return True
        if stock:
            seen.append((hold, authority, names))
    return False

@contextmanager
def _scope(
    source,
    route,
    *,
    writing=False,
    template=None,
    user_template=True,
    selected_read=None,
    _require_installed=False,
):
    previous = getattr(_local, "operation", None)
    if previous is not None:
        nested_mcp = (
            _nested_installed_mcp_state(source, route, previous)
            if selected_read is None
            else None
        )
        if nested_mcp is not None:
            _check(previous, nested_mcp.selected, writing=writing)
            yield previous
            return
        previous_state = _check(previous)
        if previous_state.source is source and route in {
            "service",
            "prompt_history",
            "sidebar_state",
            "eval_config",
            "note_templates",
            "pet",
            "runtime_state",
            "runtime_read",
            "config",
        } | chat_sources.ROUTES | {mcp_sources.ROUTE}:
            selected = (
                source.store_path
                if route == "service"
                else (
                    previous_state.selected
                    if route == mcp_sources.ROUTE
                    else chat_sources.selection(source, route, selected_read)[0]
                    if route in chat_sources.ROUTES
                    else config_files.selection(source, route, selected_read)[0]
                    if route == "config"
                    else settings_files.selection(source, route, selected_read)[0]
                    if route in settings_files.ROUTES | {"config"}
                    else _async_source_selection(
                        source, route, _require_installed=_require_installed
                    )[0]
                )
            )
            if selected_read is not None and lexical_path(
                selected_read
            ) != lexical_path(selected):
                raise bootstrap.RecoveryRequired("raw_source_selection_changed")
            _check(previous, selected, writing=writing)
            yield previous
            return
    # Core and raw scopes are independent. Keep outer registration/lease live,
    # suspend discovery, and refuse a new scope when either gate has closed.
    core = getattr(storage._operation_local, "operation", None)
    if core is not None:
        storage._check_operation(core, core.path)
    storage._operation_local.operation = None
    _local.operation = None
    attempt = None
    operation = None
    source_lock = None
    locked = False
    body_completed = False
    previous_preparation = getattr(_local, "pending_mcp_preparation", None)
    _local.pending_mcp_preparation = None
    preparation = None
    try:
        attempt = storage._Acquisition()  # before selectors, authority or path IO
        pinned = _pinned_io_available()
        if pinned and route == mcp_sources.ROUTE:
            preparation = _begin_mcp_preparation(source, attempt)
        selected, installed, directory_only = _selection(
            source, route, template, user_template, selected_read
        )
        # A queued default-history job may only narrow this source decision.
        # Check before the platform mask: native guard availability is separate.
        if _require_installed and not installed:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
        if (
            route
            in config_files.ROUTES
            | chat_sources.ROUTES
            | {dictionary_files.ROUTE, mcp_sources.ROUTE}
            and source in _source_participants
            and not pinned
        ):
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
        installed = installed and pinned
        if route == mcp_sources.ROUTE and not installed:
            pinned = False
        paths = () if directory_only else (selected,)
        if (
            route
            in {
                "service",
                "prompt_history",
                "sidebar_state",
                "eval_config",
                "note_templates",
                "theme_file",
                "theme_export",
            }
            | chat_sources.ROUTES
            and writing
        ):
            paths += (selected.with_suffix(selected.suffix + ".tmp"),)
        if route == dictionary_files.ROUTE:
            paths = dictionary_files.members(source)
        temporaries = {}
        if route in {"config", "config_snapshot"}:
            paths, temporaries = config_files.members(
                source, selected, route, selected_read
            )
        if route == mcp_sources.ROUTE:
            paths, temporaries = mcp_sources.members(source, selected)
        if route == "hook_permissions":
            paths += (selected.with_name(selected.name + ".lock"),)
        temporary = None
        if route in {"runtime_state", "hook_permissions"} and writing:
            temporary = selected.parent / f".{selected.name}.{secrets.token_hex(8)}.tmp"
            paths += (temporary,)
        if route == "pet" and writing:
            temporary = selected.with_suffix(".tmp")
            paths += (temporary,)
        parent = selected if directory_only else selected.parent
        missing = []
        anchor = parent
        while not anchor.exists():
            missing.append(anchor)
            anchor = anchor.parent
        anchor_info = os.stat(anchor)
        anchor_identity = (anchor_info.st_dev, anchor_info.st_ino)
        # Read paths never create missing directories. Reserve them only when the
        # actual producer has mkdir behavior (template save historically does not).
        directories = (
            tuple(reversed(missing)) if writing and route != "template_save" else ()
        )
        if route in {"config", "config_snapshot"}:
            owned = source.application_owned_config_directory(selected)
            directories = (
                (directories + ((parent,) if parent not in directories else ()))
                if owned is not None
                else ()
            )
        if route in {"config_data", "config_default_root", "config_chat_dicts", "config_models"}:
            directories += (selected,) if selected not in directories else ()
            if route == "config_data":
                base = selected.parent
                if (
                    base == lexical_path(source._selected_default_base_data_dir())
                    and base not in directories
                ):
                    directories += (base,)
        if (
            route == mcp_sources.ROUTE
            and mcp_sources._source_owner(source) == "mcp.history"
        ):
            directories += (parent,) if parent not in directories else ()
        if route == "runtime_read":
            directories = ()
        if route == "hook_permissions":
            directories += (parent,) if writing and parent not in directories else ()
        if route == "runtime_state":
            owned = source.application_owned_directory
            if owned is not None and lexical_path(owned) != parent:
                raise ValueError(
                    "Application-owned directory must be the target parent"
                )
            directories = (
                (directories + ((parent,) if parent not in directories else ()))
                if writing and owned is not None
                else ()
            )
        participant = None
        if installed:
            # Constructor directory selection is complete before binding.
            if route == "template_directory":
                source.user_templates_dir = selected
            if route == mcp_sources.ROUTE:
                # Initial selection already observed this installed source.
                # Registration only records that metadata; the full pre-lock
                # and post-lock admission gates below remain independent.
                participant = _register_raw_participant(
                    source, mcp_sources._source_owner(source), selected
                )
                binding = _participant_identity(participant)
            else:
                participant = _raw_participant(source)
                binding = _participant_state(participant)
            if (
                binding.owner
                not in {
                    "chunking.templates",
                    "ui.themes",
                    "config",
                    "chat.dictionaries",
                }
                and binding.selected != selected
            ):
                raise bootstrap.RecoveryRequired("raw_source_selection_changed")
        source_key = (
            str(
                (
                    settings_files.binding(source)[1]
                    if route in settings_files.ROUTES
                    else selected
                ).resolve()
            )
            if route
            in (
                {"service", "prompt_history", "sidebar_state"}
                | settings_files.ROUTES
                | (chat_sources.ROUTES - {"citation_sidecar"})
            )
            else None
        )
        if route == mcp_sources.ROUTE:
            source_lock = source._mcp_source_lock
        if route in config_files.ROUTES:
            source_lock = source._config_file_lock()
        gate = _participant_state(participant) if participant is not None else None
        with storage._changed:
            attempt.check()
            if participant is not None:
                if _participant_identity(participant) is not gate:
                    raise bootstrap.RecoveryRequired("raw_participant_not_installed")
                if gate.closed:
                    raise bootstrap.RecoveryRequired("storage_locally_paused")
            if source_key is not None:
                source_lock = _path_locks.get(source_key)
                if source_lock is None:
                    source_lock = threading.RLock()
                    _path_locks[source_key] = source_lock
        if source_lock is not None:
            while not source_lock.acquire(timeout=0.05):
                attempt.check()
            locked = True
        with storage._lock:
            if any(
                state.uncertain and state.selected == selected
                for state in _states.values()
            ):
                raise bootstrap.RecoveryRequired("raw_resources_not_retired")
        state = _State(
            source,
            participant,
            selected,
            paths,
            directories,
            writing,
            route,
            os.getpid(),
            threading.current_thread(),
            storage._task_identity(),
            pinned=pinned,
            temporary=temporary,
            temporaries=temporaries,
            config_generation=(
                source._CONFIG_GENERATION
                if route in {"config_data", "config_default_root", "config_chat_dicts", "config_models"}
                else None
            ),
        )
        operation = object.__new__(_RawOperation)
        with storage._changed:
            _states[operation] = state
            storage._raw_operations.add(operation)
            if preparation is not None:
                lease = preparation[4]
                state.leases.append(lease)
                state.holds.append(storage._holds.get(lease._key))
                state.mcp_canonical = preparation[3]
                state.mcp_observation_lease = lease
        # Every publication/creation target is admitted before any side effect.
        admission_paths = (
            ((parent,) if route in {"pet", "theme_directory"} else ())
            + paths
            + directories
            + ((selected,) if directory_only and not directories else ())
        ) or (selected,)
        state.config_anchor = (
            config_files.sibling_selector(source, route, selected) if installed else None
        )
        if route in {"config", "config_snapshot"} or state.config_anchor is not None:
            attempt.check()
            state.leases.append(storage.acquire_storage(state.config_anchor or selected))
            state.holds.append(storage._holds.get(state.leases[-1]._key))
            state.companion_guard = config_files.companion_guard(operation, attempt)
            if state.companion_guard is None:
                state.leases.append(storage.acquire_storage(
                    admission_paths[0], related_paths=admission_paths[1:]
                ))
                state.holds.append(storage._holds.get(state.leases[-1]._key))
            attempt.check()
        elif route in config_files.ROUTES:
            # All config members use the selected profile's same native group.
            # Check every member together rather than reopening its control tree
            # once per lock, backup and temporary file on each cached config read.
            attempt.check()
            state.leases.append(storage.acquire_storage(
                admission_paths[0], related_paths=admission_paths[1:]
            ))
            state.holds.append(storage._holds.get(state.leases[-1]._key))
            attempt.check()
        elif (
            installed
            and route in {"hook_permissions", mcp_sources.ROUTE}
            and (acquire_members := _related_member_acquirer()) is not None
        ):
            attempt.check()
            state.leases.append(
                acquire_members(admission_paths[0], related_paths=admission_paths[1:])
            )
            state.holds.append(storage._holds.get(state.leases[-1]._key))
            if _related_member_acquirer() is not acquire_members:
                raise bootstrap.RecoveryRequired("raw_source_selection_changed")
            attempt.check()
        else:
            for path in admission_paths:
                attempt.check()
                state.leases.append(storage.acquire_storage(path))
                state.holds.append(storage._holds.get(state.leases[-1]._key))
                attempt.check()
        if (
            route == mcp_sources.ROUTE
            and participant is not None
            and state.mcp_observation_lease is None
        ):
            canonical = mcp_sources.canonical_path(source)
            state.mcp_canonical = canonical
            if canonical == admission_paths[0]:
                state.mcp_observation_lease = state.leases[0]
            else:
                attempt.check()
                state.mcp_observation_lease = storage.acquire_storage(canonical)
                state.leases.append(state.mcp_observation_lease)
                state.holds.append(storage._holds.get(state.leases[-1]._key))
                attempt.check()
        if pinned:
            fd = _pin_parent(state, anchor)
            state.pins[anchor] = fd
            pinned_info = os.fstat(fd)
            if (pinned_info.st_dev, pinned_info.st_ino) != anchor_identity:
                raise bootstrap.RecoveryRequired("raw_parent_identity_changed")
        else:
            # Ordinary path IO cannot prove a pinned recovery/source boundary.
            # Keep the actual admission result, even if native exclusion exists.
            state.identities[anchor] = anchor_identity
        if route == mcp_sources.ROUTE:
            mcp_sources.preflight(state)
        if route == dictionary_files.ROUTE:
            dictionary_files.pin_inputs(state)
        if route == "theme_directory" or (route == "pet" and writing):
            settings_files.preflight(state, route, attempt)
        if route == "theme_export":
            settings_files.check_export_parent(state)
        if _raw_holds_pause_requested(state.holds):
            raise bootstrap.RecoveryRequired("storage_locally_paused")
        gate = _participant_state(participant) if participant is not None else None
        with storage._changed:
            attempt.check()
            if participant is not None:
                if _participant_identity(participant) is not gate:
                    raise bootstrap.RecoveryRequired("raw_participant_not_installed")
                if gate.closed:
                    raise bootstrap.RecoveryRequired("storage_locally_paused")
            _local.pending_mcp_preparation = None
            state.active = True
            _local.operation = operation
        _check(operation)
        settings_files.check_members(state)
        try:
            yield operation
            body_completed = True
        except BaseException:
            if state.created_files:
                # Failed cleanup or interruption must preserve unpublished bytes
                # and their live source admission, not only the Python error.
                state.uncertain = True
            raise
    finally:
        _local.operation = None
        # A validation failure in _begin may happen before its return assignment.
        preparation = getattr(attempt, "_mcp_preparation", None)
        transferred = (
            preparation is not None
            and operation is not None
            and any(lease is preparation[4] for lease in _states[operation].leases)
        )
        try:
            if operation is not None:
                state = _states[operation]
                state.active = False
                if _retire(state):
                    if state.route == mcp_sources.ROUTE and body_completed:
                        mcp_sources.complete(state)
                    with storage._changed:
                        del _states[operation]
                        storage._raw_operations.discard(operation)
                        storage._changed.notify_all()
        finally:
            try:
                if preparation is not None and not transferred:
                    preparation[4].close()
            finally:
                if attempt is not None:
                    attempt._mcp_preparation = None
                _local.pending_mcp_preparation = previous_preparation
                if locked:
                    source_lock.release()
                if attempt is not None:
                    attempt.close()
                if core is not None:
                    storage._check_operation(core, core.path)
                storage._operation_local.operation = core
                _local.operation = previous
                try:
                    if previous is not None:
                        _check(previous)
                except BaseException:
                    _local.operation = None
                    raise


# Only these raw guard calls are omitted by the named stock permission load.
# Capture their defining identities before any Console consumer can replace them.
_CONSOLE_PERMISSION_LOAD_GUARDS = tuple(
    (
        name,
        callback,
        tuple(
            (
                function,
                function.__code__,
                function.__globals__,
                sys.modules[function.__globals__["__name__"]],
                (
                    function.__defaults__,
                    function.__kwdefaults__,
                    tuple(function.__kwdefaults__.items())
                    if function.__kwdefaults__ is not None
                    else (),
                    function.__closure__,
                    tuple(
                        (cell, cell.cell_contents)
                        for cell in (function.__closure__ or ())
                    ),
                ),
            )
            for function in functions
        ),
    )
    for name, callback, functions in (
        ("_scope", _scope, (_scope, _scope.__wrapped__)),
        ("_check", _check, (_check,)),
    )
)


def _selected(operation):
    return _check(operation).selected


def _mkdirs(operation):
    state = _check(operation, writing=True)
    for directory in state.directories:
        if directory in state.pins:
            continue
        _check(operation, directory, writing=True)
        if state.pinned:
            parent = state.pins[directory.parent]
            try:
                os.mkdir(directory.name, mode=0o700, dir_fd=parent)
            except FileExistsError:
                pass
            fd = os.open(
                directory.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            state.pins[directory] = fd
        else:
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                pass
            info = os.stat(directory, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise bootstrap.RecoveryRequired("raw_parent_identity_changed")
            state.identities[directory] = (info.st_dev, info.st_ino)
        _check(operation)


@contextmanager
def _file(operation, path, mode):
    if mode not in {"r", "w", "a"}:
        raise ValueError("raw_file_mode_invalid")
    state = _check(operation, path, writing=mode in {"w", "a"})
    path = lexical_path(path)
    parent = state.pins.get(path.parent)
    if (state.pinned and parent is None) or (
        not state.pinned and not path.parent.is_dir()
    ):
        raise FileNotFoundError("raw_parent_not_created")
    temporary = path in state.paths[1:]
    flags = os.O_RDONLY if mode == "r" else os.O_WRONLY | os.O_CREAT
    if mode == "a":
        flags |= os.O_APPEND
    if mode == "w" and temporary:
        flags |= os.O_EXCL
    flags |= (
        getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    if state.pinned:
        fd = os.open(path.name, flags, 0o600, dir_fd=parent)
    else:
        if path.is_symlink():
            raise bootstrap.RecoveryRequired("raw_path_outside_scope")
        fd = os.open(path, flags, 0o600)
    state.descriptors.add(fd)
    if temporary and mode == "w":
        state.created_files[path] = None
    # Own the actual descriptor before wrapping it. Partial wrapper construction
    # cannot lose or implicitly retire native participation.
    native = None
    try:
        info = os.fstat(fd)
        if temporary and mode == "w":
            state.created_files[path] = (info.st_dev, info.st_ino)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("raw_not_regular")
        expected = state.observed_files.get(path)
        if expected is not None and (info.st_dev, info.st_ino) != expected:
            raise bootstrap.RecoveryRequired("raw_entry_identity_changed")
        if mode == "w":
            os.ftruncate(fd, 0)
        native = io.FileIO(fd, mode, closefd=False)
        text = io.TextIOWrapper(native, encoding="utf-8")
        state.files.append(text)
        try:
            yield text
        finally:
            try:
                io.TextIOWrapper.close(text)
                if mode in {"w", "a"}:
                    # Closing the wrapper only pushes user-space buffers into
                    # the descriptor. _replace goes on to fsync the destination
                    # DIRECTORY, so without this the entry was made durable
                    # while the blocks it points at were not -- the classic
                    # rename-without-fsync corruption, on the user's own MCP,
                    # chat-source and dictionary state. The descriptor is still
                    # the verified one here; _close_descriptor retires it in the
                    # outer finally, after this.
                    flush_file(fd)
            except BaseException:
                # A flush failure is unresolved evidence like any other: mark
                # the operation uncertain so publication cannot proceed.
                state.uncertain = True
                raise
            if not text.closed or not native.closed:
                state.uncertain = True
                raise bootstrap.RecoveryRequired("raw_resources_not_retired")
            # The closed wrapper and canonical durability barrier are complete.
            # The outer finally still owns the exact native descriptor.
            state.files.remove(text)
    finally:
        # closefd=False makes descriptor lifetime independent of wrapper GC.
        if native is not None and not native.closed:
            state.files.append(native)
            state.uncertain = True
        _close_descriptor(state, fd)


_UNCHECKED = object()


def _entry_identity(state, path):
    """``path``'s (dev, inode) without following a leaf link, or None if absent."""
    try:
        info = (
            os.stat(path.name, dir_fd=state.pins[path.parent], follow_symlinks=False)
            if state.pinned
            else os.stat(path, follow_symlinks=False)
        )
    except FileNotFoundError:
        return None
    return (info.st_dev, info.st_ino)


def _replace(operation, temporary, destination, *, expected=_UNCHECKED):
    """Publish ``temporary`` as ``destination``.

    Args:
        operation: The active raw write scope.
        temporary: The admitted temporary this scope created.
        destination: The admitted publication target.
        expected: Omitted, replace whatever is there (the historical
            behaviour). ``None``, create ``destination`` only if it is still
            absent (a hard link, which fails on an existing entry). A
            ``(st_dev, st_ino)`` pair, replace only that observed file.

    Raises:
        FileExistsError: ``destination`` is not the entry ``expected`` names;
            nothing was published and the scope stays usable.
    """
    state = _check(operation, temporary, writing=True)
    _check(operation, destination, writing=True)
    temporary, destination = lexical_path(temporary), lexical_path(destination)
    linked = False
    if expected is not _UNCHECKED:
        if _entry_identity(state, destination) != expected:
            raise FileExistsError(f"{destination.name} changed before it was written")
        if expected is None:
            _check_temporary_identity(state, temporary)
            try:
                if state.pinned:
                    os.link(
                        temporary.name,
                        destination.name,
                        src_dir_fd=state.pins[temporary.parent],
                        dst_dir_fd=state.pins[destination.parent],
                    )
                else:
                    os.link(temporary, destination)
                linked = True
            except FileExistsError:
                raise FileExistsError(
                    f"{destination.name} appeared before it was written"
                ) from None
            except OSError:
                # ponytail: no hard links on this filesystem (FAT/exFAT) --
                # fall back to the checked replace below; its window is the
                # microseconds since the identity check above.
                pass
    try:
        if linked:
            # Published by the link; retire the temporary's own name.
            if state.pinned:
                os.unlink(temporary.name, dir_fd=state.pins[temporary.parent])
                flush_directory(state.pins[destination.parent])
            else:
                os.unlink(temporary)
                fsync_parent_directory(destination.parent)
            state.created_files.pop(temporary)
            return
        if destination == state.backup:
            try:
                info = (
                    os.stat(
                        destination.name,
                        dir_fd=state.pins[destination.parent],
                        follow_symlinks=False,
                    )
                    if state.pinned
                    else os.stat(destination, follow_symlinks=False)
                )
                identity = (info.st_dev, info.st_ino)
            except FileNotFoundError:
                identity = None
            if identity != state.observed_files.get(destination):
                raise bootstrap.RecoveryRequired("raw_entry_identity_changed")
        _check_temporary_identity(state, temporary)
        if state.route == mcp_sources.ROUTE:
            mcp_sources.check_destination(state, destination)
        # task-32896: the rename is atomic but was not durable -- this
        # module had zero fsync calls while writing the user's live config.
        # Barrier placement and the no-tolerance failure mode match
        # config_binding.py's flush_directory-after-replace convention: an
        # unpersisted publication is a recovery event, not a warning.
        if state.pinned:
            os.replace(
                temporary.name,
                destination.name,
                src_dir_fd=state.pins[temporary.parent],
                dst_dir_fd=state.pins[destination.parent],
            )
            # The rename itself is only durable once its directory entry is.
            # One pin covers both ends: `state.pins` holds the single anchor
            # directory, which is why the os.replace above can index it for
            # both parents. A failure here falls into the BaseException
            # handler below and is treated as an unresolved publication,
            # which is what an unproven barrier is. Pinned is the only
            # posture the config, settings, dictionary and MCP routes accept,
            # so every route that writes user-owned files is covered.
            flush_directory(state.pins[destination.parent])
        else:
            os.replace(temporary, destination)
            fsync_parent_directory(destination.parent)
        # Publication consumes this exact temporary object. A following process
        # may now create its own same-name sidecar; our cleanup cannot own it.
        identity = state.created_files.pop(temporary)
        if state.route == mcp_sources.ROUTE:
            state.mcp_publications[destination] = identity
            mcp_sources.published(state, destination)
    except BaseException:
        # An interrupted/ambiguous publication is unresolved source evidence.
        # Keep both the actual lease and any sidecar until explicit recovery.
        state.uncertain = True
        raise bootstrap.RecoveryRequired("raw_publication_uncertain") from None


def _remove_temporary(operation, temporary):
    state = _check(operation, temporary, writing=True)
    temporary = lexical_path(temporary)
    if temporary not in state.created_files:
        return
    try:
        _check_temporary_identity(state, temporary)
        if state.pinned:
            os.unlink(temporary.name, dir_fd=state.pins[temporary.parent])
        else:
            os.unlink(temporary)
    except FileNotFoundError:
        pass
    state.created_files.pop(temporary)


def _check_temporary_identity(state, temporary):
    expected = state.created_files.get(temporary)
    info = (
        os.stat(
            temporary.name, dir_fd=state.pins[temporary.parent], follow_symlinks=False
        )
        if state.pinned
        else os.stat(temporary, follow_symlinks=False)
    )
    if expected is None or (info.st_dev, info.st_ino) != expected:
        state.uncertain = True
        raise bootstrap.RecoveryRequired("raw_temporary_identity_changed")


def _service_mutation(function):
    """Wrap only these sources' complete synchronous-body service operations."""
    writing = function.__name__ != "_load"

    @contextmanager
    def guarded(source):
        valid = any(
            isinstance(source, cls)
            and owner in {"feedback", "chat.grammars"}
            and cls.__dict__.get(function.__name__) is wrapped
            and function.__module__ == cls.__module__
            and function.__qualname__ == f"{cls.__name__}.{function.__name__}"
            and function.__name__
            in {
                "_load",
                "_persist",
                "submit_feedback",
                "update_feedback",
                "delete_feedback",
                "create_grammar",
                "update_grammar",
                "delete_grammar",
            }
            for cls, owner in _types().items()
        )
        if not valid:
            raise bootstrap.RecoveryRequired("raw_source_not_supported")
        with _scope(source, "service", writing=writing):
            before = (
                copy.deepcopy((source._records, source._next_id)) if writing else None
            )
            try:
                yield
            except BaseException:
                if before is not None:
                    source._records, source._next_id = before
                raise

    if inspect.iscoroutinefunction(function):

        @wraps(function)
        async def asynchronous(source, *args, **kwargs):
            with guarded(source):
                return await function(source, *args, **kwargs)

        wrapped = asynchronous
        return wrapped

    @wraps(function)
    def synchronous(source, *args, **kwargs):
        with guarded(source):
            return function(source, *args, **kwargs)

    wrapped = synchronous
    return wrapped


def _service_file(source, mode):
    operation = getattr(_local, "operation", None)
    state = _check(operation, source.store_path, writing=mode == "w")
    if state.source is not source:
        raise bootstrap.RecoveryRequired("raw_operation_provenance_invalid")
    return operation


def _unlink(operation, path):
    """Delete only an admitted regular entry whose observed identity still holds."""
    state = _check(operation, path, writing=True)
    path = lexical_path(path)
    parent = state.pins.get(path.parent)
    info = (
        os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if state.pinned
        else os.stat(path, follow_symlinks=False)
    )
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise bootstrap.RecoveryRequired("raw_not_regular")
    _check(operation, path, writing=True)
    current = (
        os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if state.pinned
        else os.stat(path, follow_symlinks=False)
    )
    expected = state.observed_files.get(path, (info.st_dev, info.st_ino))
    if (current.st_dev, current.st_ino) != expected:
        raise bootstrap.RecoveryRequired("raw_entry_identity_changed")
    if state.pinned:
        os.unlink(path.name, dir_fd=parent)
    else:
        path.unlink()
    state.observed_files.pop(path, None)


def _runtime_operation(path=None):
    """Discover only the live runtime-state or exact config source operation."""
    operation = getattr(_local, "operation", None)
    if operation is None:
        return None
    state = _states.get(operation)
    module = sys.modules.get("tldw_chatbook.runtime_policy.source_state")
    cls = getattr(module, "RuntimeSourceStateStore", None)
    if state is None:
        return None
    if mcp_sources.history_operation(state) and state.participant is not None:
        _check(operation, path, writing=True)
        return operation
    if state.route == "hook_permissions":
        _check(operation, path, writing=True)
        return operation
    if config_files.binding(state.source) is None and (
        cls is None or not isinstance(state.source, cls)
    ):
        return None
    _check(operation, path, writing=True)
    return operation


_RUNTIME_OPERATION_BINDING = (
    _runtime_operation,
    _runtime_operation.__code__,
    _runtime_operation.__defaults__,
)
