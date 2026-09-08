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
import threading
import time
import weakref

from . import bootstrap, storage_admission as storage
from .profile_paths import lexical_path
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
    from ..Widgets import emoji_picker

    types = _types()
    if source is emoji_picker:
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
    active: bool = False
    uncertain: bool = False


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
    if state.uncertain or state.files:
        return False
    for path, fd in tuple(state.pins.items()):
        _close_descriptor(state, fd)
        del state.pins[path]
    for lease in reversed(state.leases):
        lease.close()
    return True


def _selection(source, route, template, user_template, selected_read):
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
        if previous_state.source is source and route == "service":
            _check(previous, source.store_path, writing=writing)
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
    try:
        attempt = storage._Acquisition()  # before selectors, authority or path IO
        pinned = _pinned_io_available()
        selected, installed, directory_only = _selection(
            source, route, template, user_template, selected_read
        )
        installed = installed and pinned
        paths = () if directory_only else (selected,)
        if route == "service" and writing:
            paths += (selected.with_suffix(selected.suffix + ".tmp"),)
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
        participant = None
        if installed:
            # Constructor directory selection is complete before binding.
            if route == "template_directory":
                source.user_templates_dir = selected
            participant = _raw_participant(source)
            binding = _participant_state(participant)
            if binding.owner != "chunking.templates" and binding.selected != selected:
                raise bootstrap.RecoveryRequired("raw_source_selection_changed")
        source_key = str(selected.resolve()) if route == "service" else None
        with storage._changed:
            attempt.check()
            if participant is not None and _participant_state(participant).closed:
                raise bootstrap.RecoveryRequired("storage_locally_paused")
            if route == "service":
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
            os.getpid(),
            threading.current_thread(),
            storage._task_identity(),
            pinned=pinned,
        )
        with storage._changed:
            _states[operation] = state
            storage._raw_operations.add(operation)
        # Every publication/creation target is admitted before any side effect.
        for path in (
            paths
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
        try:
            yield operation
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
    state = _check(operation, path, writing=mode == "w")
    path = lexical_path(path)
    parent = state.pins.get(path.parent)
    if (state.pinned and parent is None) or (
        not state.pinned and not path.parent.is_dir()
    ):
        raise FileNotFoundError("raw_parent_not_created")
    temporary = path in state.paths[1:]
    flags = os.O_RDONLY if mode == "r" else os.O_WRONLY | os.O_CREAT
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
        _check_temporary_identity(state, temporary)
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
        state.created_files.pop(temporary)
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
