"""Bounded installed raw-file lifetimes; never recovery/capture authority (ADR-126)."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
import copy
import inspect
import io
import os
from pathlib import Path
import stat
import secrets
import sys
import threading
import time
import weakref

from . import bootstrap, storage_admission as storage
from .profile_paths import lexical_path
from . import settings_file_participants as settings_files
from . import config_participants as config_files
from . import chat_source_participants as chat_sources
from . import dictionary_file_participants as dictionary_files
from . import mcp_source_participants as mcp_sources
from ..Utils.private_paths import _open_verified_parent, _posix_guards_available


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
        return _participant_state(self).owner

    def close_admission(self):
        with storage._changed:
            _participant_state(self).closed = True

    def drain(self, deadline):
        with storage._changed:
            if not _participant_state(self).closed:
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
            state = _participant_state(self)
            source = state.source()
            if mcp_sources.binding(source) is not None:
                return mcp_sources.drain_ready(source)
            if state.owner == "chat.dictionaries":
                return dictionary_files.drain_ready(source)
            if chat_sources.binding(source) is not None:
                return chat_sources.drain_ready(source)
            if state.owner == "config":
                return (
                    source._CONFIG_PERSISTENCE_ERROR is None
                    and source.get_config_load_failure() is None
                    and source._CONFIG_CACHE is not None
                )
            if state.owner == "eval.definitions":
                return (
                    source._config == source._persisted_config
                    and source.persistence_error is None
                )
            if state.owner == "ui.themes":
                return not source.is_modified
            return True

    def resume(self):
        with storage._changed:
            state = _participant_state(self)
            if storage._pause is not None:
                raise bootstrap.RecoveryRequired("process_pause_still_active")
            state.closed = False


def _participant_state(participant):
    if type(participant) is not _RawParticipant or participant not in _participants:
        raise bootstrap.RecoveryRequired("raw_participant_not_installed")
    state = _participants[participant]
    source = state.source()
    if source is None or type(source) is not state.source_type:
        raise bootstrap.RecoveryRequired("raw_participant_not_installed")
    if _source_participants.get(source) is not participant:
        raise bootstrap.RecoveryRequired("raw_participant_not_installed")
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
    if state.owner in {"chat.prompt_history", "ui.state"}:
        route = (
            "prompt_history"
            if state.owner == "chat.prompt_history"
            else "sidebar_state"
        )
        selected, installed = _async_source_selection(source, route)
        if not installed or selected != state.selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    if state.owner == "notes.file_notes_replica" and source.db_path != state.selected:
        raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    return state


def _types():
    from ..Feedback_Interop.local_feedback_service import LocalFeedbackService
    from ..Chat_Grammars_Interop.local_chat_grammars_service import (
        LocalChatGrammarsService,
    )
    from ..Chunking.chunking_templates import ChunkingTemplateManager

    return {
        LocalFeedbackService: "feedback",
        LocalChatGrammarsService: "chat.grammars",
        ChunkingTemplateManager: "chunking.templates",
    }


def _async_source_selection(source, route):
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
        from ..Widgets import emoji_picker
        from ..Notes.file_notes_replica import FileNotesReplica

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
        elif type(source) in types:
            owner = types[type(source)]
            selected = (
                source.user_templates_dir
                if owner == "chunking.templates"
                else source.store_path
            )
        else:
            raise bootstrap.RecoveryRequired("raw_participant_not_installed")
    selected = lexical_path(selected)
    if owner == "chunking.templates":
        from ..config import get_cli_data_dir

        if selected != lexical_path(get_cli_data_dir() / "chunking_templates"):
            raise bootstrap.RecoveryRequired("raw_participant_not_installed")
    with storage._lock:
        participant = _source_participants.get(source)
        if participant is None:
            participant = object.__new__(_RawParticipant)
            _participants[participant] = _ParticipantState(
                weakref.ref(source), type(source), owner, selected
            )
            _source_participants[source] = participant
        state = _participant_state(participant)
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
    temporaries: dict = field(default_factory=dict)
    temporary: Path | None = None
    backup: Path | None = None
    backup_temporary: Path | None = None
    active: bool = False
    uncertain: bool = False
    config_failed: bool = False
    config_generation: int | None = None


_states = {}


def _check(operation, path=None, *, writing=False):
    # No virtual validator dispatch and no mutable caller token fields.
    with storage._lock:
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
        if state.participant is not None:
            _participant_state(state.participant)
        elif storage._pause is not None:
            raise bootstrap.RecoveryRequired("storage_locally_paused")
        if writing and not state.writing:
            raise bootstrap.RecoveryRequired("raw_path_outside_scope")
        if (
            path is not None
            and lexical_path(path) not in state.paths + state.directories
        ):
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
    if state.route in {"config_data", "config_chat_dicts", "config_models"}:
        config_files.selection(state.source, state.route, state.selected)
        if state.source._CONFIG_GENERATION != state.config_generation:
            raise bootstrap.RecoveryRequired("config_directory_generation_changed")
    # Disk checks never occur under the coordinator lock. Native descriptors pin
    # destinations even if an external nonparticipant renames after this check.
    # Existing parent aliases are valid only while they resolve to the same
    # positively checked physical directory; leaf aliases remain refused.
    for directory, identity in tuple(state.identities.items()):
        info = directory.stat()
        if (info.st_dev, info.st_ino) != identity or not stat.S_ISDIR(info.st_mode):
            raise bootstrap.RecoveryRequired("raw_parent_identity_changed")
    for directory, fd in tuple(state.pins.items()):
        info = os.stat(directory)
        pinned = os.fstat(fd)
        if (info.st_dev, info.st_ino) != (pinned.st_dev, pinned.st_ino):
            raise bootstrap.RecoveryRequired("raw_parent_identity_changed")
    return state


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
    for lease in reversed(state.leases):
        lease.close()
    return True


def _selection(source, route, template, user_template, selected_read):
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
    if route in {"template_directory", "template_save", "template_read"}:
        from ..Chunking.chunking_templates import ChunkingTemplateManager

        if not isinstance(source, ChunkingTemplateManager):
            raise bootstrap.RecoveryRequired("raw_source_not_supported")
        if route == "template_directory":
            from ..config import get_cli_data_dir

            return (
                lexical_path(get_cli_data_dir() / "chunking_templates"),
                installed,
                True,
            )
        from ..config import get_cli_data_dir

        owned_directory = lexical_path(get_cli_data_dir() / "chunking_templates")
        directory = source.user_templates_dir if user_template else source.templates_dir
        if route == "template_read":
            selected = lexical_path(selected_read)
            allowed = selected.parent == owned_directory
            return selected, installed and allowed, False
        name = template.name
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("invalid_template_name")
        return (
            lexical_path(directory / f"{name}.json"),
            installed and user_template and lexical_path(directory) == owned_directory,
            False,
        )
    from ..Widgets import emoji_picker

    if route != "emoji" or source is not emoji_picker:
        raise bootstrap.RecoveryRequired("raw_source_not_supported")
    return lexical_path(emoji_picker._recent_emojis_path()), True, False


@contextmanager
def _scope(
    source,
    route,
    *,
    writing=False,
    template=None,
    user_template=True,
    selected_read=None,
):
    previous = getattr(_local, "operation", None)
    if previous is not None:
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
                    mcp_sources.selection(source)[0]
                    if route == mcp_sources.ROUTE
                    else chat_sources.selection(source, route, selected_read)[0]
                    if route in chat_sources.ROUTES
                    else config_files.selection(source, route, selected_read)[0]
                    if route == "config"
                    else settings_files.selection(source, route, selected_read)[0]
                    if route in settings_files.ROUTES | {"config"}
                    else _async_source_selection(source, route)[0]
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
    try:
        attempt = storage._Acquisition()  # before selectors, authority or path IO
        pinned = _pinned_io_available()
        selected, installed, directory_only = _selection(
            source, route, template, user_template, selected_read
        )
        if (
            route in config_files.ROUTES | chat_sources.ROUTES | {dictionary_files.ROUTE, mcp_sources.ROUTE}
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
            } | chat_sources.ROUTES
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
        temporary = None
        if route == "runtime_state" and writing:
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
        anchor_info = anchor.stat()
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
        if route in {"config_data", "config_chat_dicts", "config_models"}:
            directories += (selected,) if selected not in directories else ()
            if route == "config_data":
                base = selected.parent
                if (
                    base == lexical_path(source._default_base_data_dir())
                    and base not in directories
                ):
                    directories += (base,)
        if route == mcp_sources.ROUTE and mcp_sources.binding(source)[0] == "mcp.history":
            directories += (parent,) if parent not in directories else ()
        if route == "runtime_read":
            directories = ()
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
            participant = _raw_participant(source)
            binding = _participant_state(participant)
            if (
                binding.owner not in {
                    "chunking.templates", "ui.themes", "config", "chat.dictionaries"
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
        with storage._changed:
            attempt.check()
            if participant is not None and _participant_state(participant).closed:
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
        operation = object.__new__(_RawOperation)
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
                if route in {"config_data", "config_chat_dicts", "config_models"}
                else None
            ),
        )
        with storage._changed:
            _states[operation] = state
            storage._raw_operations.add(operation)
        # Every publication/creation target is admitted before any side effect.
        for path in (
            ((parent,) if route in {"pet", "theme_directory"} else ())
            + paths
            + directories
            + ((selected,) if directory_only and not directories else ())
        ):
            attempt.check()
            state.leases.append(storage.acquire_storage(path))
            state.holds.append(storage._holds.get(state.leases[-1]._key))
            attempt.check()
        if not state.leases:
            state.leases.append(storage.acquire_storage(selected))
            state.holds.append(storage._holds.get(state.leases[-1]._key))
        if pinned:
            fd, _ = _open_verified_parent(
                anchor / ".raw-pin",
                missing_leaf_allowed=True,
                _close=lambda fd: _close_descriptor(state, fd),
            )
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
        if any(
            hold is not None and hold.authority.pause_requested(hold.names)
            for hold in state.holds
        ):
            raise bootstrap.RecoveryRequired("storage_locally_paused")
        with storage._changed:
            attempt.check()
            if participant is not None and _participant_state(participant).closed:
                raise bootstrap.RecoveryRequired("storage_locally_paused")
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
            if locked:
                source_lock.release()
            if attempt is not None:
                attempt.close()
            if core is not None:
                storage._check_operation(core, core.path)
            storage._operation_local.operation = core
            if previous is not None:
                _check(previous)
            _local.operation = previous


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
            info = directory.lstat()
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
            except BaseException:
                state.uncertain = True
                raise
            if not text.closed or not native.closed:
                state.uncertain = True
                raise bootstrap.RecoveryRequired("raw_resources_not_retired")
            state.files.remove(text)
    finally:
        # closefd=False makes descriptor lifetime independent of wrapper GC.
        if native is not None and not native.closed:
            state.files.append(native)
            state.uncertain = True
        _close_descriptor(state, fd)


def _replace(operation, temporary, destination):
    state = _check(operation, temporary, writing=True)
    _check(operation, destination, writing=True)
    temporary, destination = lexical_path(temporary), lexical_path(destination)
    try:
        if destination == state.backup:
            try:
                info = (
                    os.stat(
                        destination.name,
                        dir_fd=state.pins[destination.parent],
                        follow_symlinks=False,
                    )
                    if state.pinned
                    else destination.lstat()
                )
                identity = (info.st_dev, info.st_ino)
            except FileNotFoundError:
                identity = None
            if identity != state.observed_files.get(destination):
                raise bootstrap.RecoveryRequired("raw_entry_identity_changed")
        _check_temporary_identity(state, temporary)
        if state.route == mcp_sources.ROUTE:
            mcp_sources.check_destination(state, destination)
        if state.pinned:
            os.replace(
                temporary.name,
                destination.name,
                src_dir_fd=state.pins[temporary.parent],
                dst_dir_fd=state.pins[destination.parent],
            )
        else:
            os.replace(temporary, destination)
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
        else temporary.lstat()
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
        else path.lstat()
    )
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise bootstrap.RecoveryRequired("raw_not_regular")
    _check(operation, path, writing=True)
    current = (
        os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if state.pinned
        else path.lstat()
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
    if config_files.binding(state.source) is None and (
        cls is None or not isinstance(state.source, cls)
    ):
        return None
    _check(operation, path, writing=True)
    return operation
