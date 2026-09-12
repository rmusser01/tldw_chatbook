"""Concrete Shared Visual Identity source/native lifetimes (ADR-126, rulings74–78).

These private scopes are ordinary source admission, never capture authority. File
selection is performed by the actual visual producers before any native mutation.
"""

import errno
import json
import stat
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from functools import wraps
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from . import bootstrap, profile_paths
from . import storage_admission as storage

_local = threading.local()
_sources = weakref.WeakKeyDictionary()
_repositories = weakref.WeakKeyDictionary()
_issued = {}
_states = {}
_producer_codes = set()


@dataclass(eq=False)
class _Source:
    config: object
    config_path: Path
    profile: Path
    closed: bool = False
    native: bool = field(default_factory=lambda: _native_available())
    failures: list = field(default_factory=list)

    @property
    def owner_id(self):
        return "persona.visual_identity"

    def close_admission(self):
        with storage._changed:
            self.closed = True

    def drain(self, deadline):
        with storage._changed:
            if not self.closed:
                raise RuntimeError("participant_admission_not_closed")
            while (
                any(s.source is self for s in _states.values())
                or storage._pending_acquisitions
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                storage._changed.wait(min(remaining, 0.05))
            _validate_source(self)
            return self.native and not self.failures and not _dirty(self)

    def resume(self):
        with storage._changed:
            if storage._pause is not None:
                raise bootstrap.RecoveryRequired("process_pause_still_active")
            _validate_source(self)
            self.closed = False


def _native_available():
    return (
        os.name == "posix"
        and bool(getattr(os, "O_DIRECTORY", 0))
        and bool(getattr(os, "O_NOFOLLOW", 0))
        and {os.open, os.stat, os.rename, os.unlink}.issubset(os.supports_dir_fd)
    )


def _validate_source(source):
    config = source.config
    if source.native and not _native_available():
        raise bootstrap.RecoveryRequired("visual_identity_native_capability_changed")
    if (
        sys.modules.get("tldw_chatbook.config") is not config
        or profile_paths.lexical_path(config._get_effective_config_path())
        != source.config_path
        or config._CONFIG_CACHE_SOURCE != source.config_path
        or config._CONFIG_CACHE is None
        or profile_paths.user_data_dir(config._CONFIG_CACHE) != source.profile
    ):
        raise bootstrap.RecoveryRequired("visual_identity_source_changed")


def source_for(profile):
    """Select only the actual loaded canonical profile; never import config here."""
    config = sys.modules.get("tldw_chatbook.config")
    if config is None or getattr(config, "_CONFIG_CACHE", None) is None:
        return None
    selected = profile_paths.lexical_path(profile)
    expected = profile_paths.user_data_dir(config._CONFIG_CACHE)
    if selected != expected:
        return None
    with storage._changed:
        source = _sources.get(config)
        if source is None:
            source = _Source(
                config,
                profile_paths.lexical_path(config._get_effective_config_path()),
                selected,
            )
            _sources[config] = source
    _validate_source(source)
    return source


def db_source(db):
    from ..DB.ChaChaNotes_DB import CharactersRAGDB
    from .participants import _repository_participant

    try:
        old = _repositories.get(db)
    except TypeError:
        old = None
    if old is not None:
        _validate_source(old)
        if (
            type(db) is not CharactersRAGDB
            or db.is_memory_db
            or profile_paths.database_path(
                old.config._CONFIG_CACHE, "chachanotes_db_path"
            )
            != db.db_path
        ):
            raise bootstrap.RecoveryRequired("visual_identity_repository_changed")
        _repository_participant(db)
        return old
    config = sys.modules.get("tldw_chatbook.config")
    if (
        type(db) is not CharactersRAGDB
        or db.is_memory_db
        or config is None
        or getattr(config, "_CONFIG_CACHE", None) is None
        or profile_paths.database_path(config._CONFIG_CACHE, "chachanotes_db_path")
        != db.db_path
    ):
        return None
    _repository_participant(db)
    source = source_for(profile_paths.user_data_dir(config._CONFIG_CACHE))
    _repositories[db] = source
    return source


def repository_source(repository):
    from ..DB.VisualIdentity_DB import VisualIdentityRepository

    if type(repository) is not VisualIdentityRepository:
        return None
    return db_source(repository.db)


def db_guard(function):
    _producer_codes.add(function.__code__)

    @wraps(function)
    def guarded(db, *args, **kwargs):
        from contextlib import nullcontext

        from ..DB.ChaChaNotes_DB import CharactersRAGDB
        from .participants import _core_operation

        with request():
            source = db_source(db) if isinstance(db, CharactersRAGDB) else None
            if source is not None and source.closed:
                raise bootstrap.RecoveryRequired("storage_locally_paused")
            if (
                source is not None
                and function.__name__ == "publish_visual_identity_candidate"
                and args
            ):
                selected = kwargs.get("user_data_dir")
                candidate_source(
                    args[0],
                    db,
                    source.profile
                    if selected is None
                    else profile_paths.lexical_path(selected),
                )
            with (
                _core_operation(db)
                if isinstance(db, CharactersRAGDB)
                else nullcontext()
            ):
                return function(db, *args, **kwargs)

    return guarded


def repository_guard(function):
    _producer_codes.add(function.__code__)

    @wraps(function)
    def guarded(repository, *args, **kwargs):
        frame = sys._getframe(1)
        module = sys.modules.get("tldw_chatbook.DB.VisualIdentity_DB")
        edges = {
            ("activate_pack", "get_active_actor_pack"),
            ("publish_version", "get_active_actor_pack"),
            ("get_active_actor_pack", "list_version_assets"),
        }
        if (
            module is not None
            and type(repository) is module.VisualIdentityRepository
            and frame.f_globals is module.__dict__
            and frame.f_locals.get("self") is repository
            and (frame.f_code.co_name, function.__name__) in edges
        ):
            caller = getattr(module.VisualIdentityRepository, frame.f_code.co_name)
            callee = getattr(module.VisualIdentityRepository, function.__name__)
            while hasattr(caller, "__wrapped__"):
                caller = caller.__wrapped__
            while hasattr(callee, "__wrapped__"):
                callee = callee.__wrapped__
            operation = getattr(storage._operation_local, "operation", None)
            if (
                caller.__code__ is frame.f_code
                and callee is function
                and type(operation) is storage._Operation
            ):
                from .participants import _repository_participant

                source = db_source(repository.db)
                participant = _repository_participant(repository.db)
                if source is not None and operation.participant is participant:
                    storage._check_operation(operation, participant.path)
                    return function(repository, *args, **kwargs)
        return db_guard(lambda db: function(repository, *args, **kwargs))(repository.db)

    return guarded


@contextmanager
def request():
    """Reserve before selectors; this reservation grants no IO authority."""
    active = getattr(_local, "state", None)
    if active is not None:
        check(active)
        yield active.attempt
        return
    previous_core = getattr(storage._operation_local, "operation", None)
    storage._operation_local.operation = None
    try:
        attempt = storage._Acquisition()
    finally:
        storage._operation_local.operation = previous_core
    try:
        yield attempt
    finally:
        attempt.close()


def _shape(value):
    if is_dataclass(value):
        return type(value), tuple(
            (f.name, _shape(getattr(value, f.name))) for f in fields(value)
        )
    if isinstance(value, dict):
        return tuple(sorted((key, _shape(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_shape(item) for item in value)
    return value


def candidate_shape(value):
    # Stage methods own the mutable edit fields under the original candidate lock.
    names = (
        "actor_kind",
        "actor_id",
        "old_pack_id",
        "old_version_id",
        "old_binding_id",
        "old_binding_version",
        "old_pack_version",
        "source_kind",
        "title",
        "description",
        "original_default_expression_key",
        "source_context",
        "assets",
        "_lock",
        "_maintenance_identity",
    )
    return type(value), tuple((name, _shape(getattr(value, name))) for name in names)


def issue(value, db):
    source = db_source(db)
    if source is not None:
        # This refusal marker survives shallow copies; only the actual registry
        # record grants authority. Copying this marker never creates a record.
        value._maintenance_identity = object()
    with storage._changed:
        _issued[id(value)] = (value, db, source, candidate_shape(value), False)
    return value


def candidate_source(value, db, profile):
    source = db_source(db)
    record = _issued.get(id(value))
    if record is not None and record[0] is value:
        if record[1] is not db or record[3] != candidate_shape(value):
            raise bootstrap.RecoveryRequired("visual_identity_candidate_changed")
        if record[2] is not source:
            raise bootstrap.RecoveryRequired("visual_identity_source_changed")
    elif source is not None:
        raise bootstrap.RecoveryRequired("visual_identity_candidate_not_issued")
    if source is not None and source.profile != profile:
        raise bootstrap.RecoveryRequired("visual_identity_source_changed")
    return source


def _identity(path):
    try:
        info = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)


@dataclass(eq=False)
class _NativeState:
    source: object
    files: tuple
    directories: tuple
    writable: frozenset
    attempt: object
    repository: object = None
    pid: int = field(default_factory=os.getpid)
    thread: object = field(default_factory=threading.current_thread)
    task: object = field(default_factory=storage._task_identity)
    expectations: dict = field(default_factory=dict)
    descriptors: dict = field(default_factory=dict)
    streams: list = field(default_factory=list)
    failed_closes: set = field(default_factory=set)
    leases: list = field(default_factory=list)
    holds: list = field(default_factory=list)
    uncertain: bool = False
    active: bool = False
    result: object = None
    helper: object = None
    owned: dict = field(default_factory=dict)
    body_failed: bool = False
    publication_cleanup: object = None
    publication_candidate: object = None
    candidate: object = None
    candidate_generation: object = None
    native: bool = False


class VisualIdentityNativeError(bootstrap.RecoveryRequired):
    def __init__(self, state):
        self.result = state.result
        self.cleanup_candidate = state.publication_candidate
        super().__init__("visual_identity_native_not_retired")


def current():
    state = getattr(_local, "state", None)
    if state is not None:
        check(state)
    return state


def check(state):
    if (
        state not in _states
        or state not in storage._raw_operations
        or not state.active
        or state.pid != os.getpid()
        or state.thread is not threading.current_thread()
        or state.task is not storage._task_identity()
        or state.uncertain
    ):
        raise VisualIdentityNativeError(state)
    if state.source is not None:
        _validate_source(state.source)
    if state.repository is not None:
        db_source(state.repository)
    if state.candidate is not None:
        candidate = state.candidate
        if state.candidate_generation != (
            candidate_shape(candidate),
            _shape(candidate._replacements),
            frozenset(candidate._cleared),
            candidate.default_expression_key,
        ):
            raise bootstrap.RecoveryRequired("visual_identity_candidate_changed")
    for lease, hold in zip(state.leases, state.holds):
        if (
            lease not in storage._live_leases
            or storage._holds.get(lease._key) is not hold
        ):
            raise VisualIdentityNativeError(state)
        if hold is not None and (hold.stop.is_set() or hold.error is not None):
            raise VisualIdentityNativeError(state)
        if storage._pause is not None and (
            hold is None
            or state.source is None
            or not state.source.native
            or not state.native
        ):
            raise bootstrap.RecoveryRequired("storage_locally_paused")
    return state


def _path(value, dir_fd=None):
    selected = Path(os.fsdecode(value))
    if dir_fd is not None:
        state = current()
        if state is None or dir_fd not in state.descriptors:
            raise bootstrap.RecoveryRequired("visual_identity_descriptor_not_owned")
        selected = state.descriptors[dir_fd] / selected
    return profile_paths.lexical_path(selected)


def _check_path(state, selected, *, writing=False):
    check(state)
    if selected not in state.expectations or (
        writing and selected not in state.writable
    ):
        raise bootstrap.RecoveryRequired("visual_identity_path_outside_scope")
    expected = state.expectations[selected]
    if _identity(selected) != expected:
        raise bootstrap.RecoveryRequired("visual_identity_file_identity_changed")
    for parent in selected.parents:
        if (
            parent in state.expectations
            and _identity(parent) != state.expectations[parent]
        ):
            raise bootstrap.RecoveryRequired("visual_identity_parent_identity_changed")


@contextmanager
def files(
    source, paths, directories=(), *, writing=(), repository=None, candidate=None
):
    """Native scope selected by concrete producers, never an arbitrary tree lease."""
    producer = _check_producer(sys._getframe(2))
    paths = tuple(dict.fromkeys(profile_paths.lexical_path(p) for p in paths))
    directories = tuple(
        dict.fromkeys(profile_paths.lexical_path(p) for p in directories)
    )
    writable = frozenset(profile_paths.lexical_path(p) for p in writing)
    previous = getattr(_local, "state", None)
    if previous is not None:
        check(previous)
        if (
            source is previous.source
            and (repository is None or repository is previous.repository)
            and (candidate is None or candidate is previous.candidate)
            and set(paths + directories).issubset(previous.expectations)
            and writable.issubset(previous.writable)
        ):
            for p in paths + directories:
                _check_path(previous, p, writing=p in writable)
            yield previous
            return
    core = getattr(storage._operation_local, "operation", None)
    storage._operation_local.operation = None
    _local.state = None
    state = None
    try:
        with request() as attempt:
            if source is not None:
                _validate_source(source)
                if source.closed:
                    raise bootstrap.RecoveryRequired("storage_locally_paused")
            state = _NativeState(
                source, paths, directories, writable, attempt, repository
            )
            module = sys.modules["tldw_chatbook.Character_Chat.visual_identity"]
            native = _native_available()
            if producer in {
                "publish_visual_identity_candidate",
                "cleanup_visual_identity_publication_candidate",
            }:
                native = native and module._publication_posix_guards_available()
            elif producer == "_read_user_asset":
                native = native and module._supports_secure_dir_fd()
            state.native = native
            state.candidate = candidate
            if candidate is not None:
                state.candidate_generation = (
                    candidate_shape(candidate),
                    _shape(candidate._replacements),
                    frozenset(candidate._cleared),
                    candidate.default_expression_key,
                )
            with storage._changed:
                if source is not None and any(
                    s.source is source and s.uncertain for s in _states
                ):
                    raise bootstrap.RecoveryRequired(
                        "visual_identity_native_not_retired"
                    )
                _states[state] = state
                storage._raw_operations.add(state)
            for p in paths + directories:
                for ancestor in (p, *p.parents):
                    if ancestor not in state.expectations:
                        state.expectations[ancestor] = _identity(ancestor)
            for p in (paths + directories) or (None,):
                attempt.check()
                lease = storage.acquire_storage(p)
                state.leases.append(lease)
                state.holds.append(storage._holds.get(lease._key))
            for p, identity in state.expectations.items():
                if _identity(p) != identity:
                    raise bootstrap.RecoveryRequired(
                        "visual_identity_preflight_changed"
                    )
            if any(
                h is not None and h.authority.pause_requested(h.names)
                for h in state.holds
            ):
                raise bootstrap.RecoveryRequired("storage_locally_paused")
            with storage._changed:
                attempt.check()
                if source is not None and source.closed:
                    raise bootstrap.RecoveryRequired("storage_locally_paused")
                state.active = True
                _local.state = state
                if (
                    source is not None
                    and source.native
                    and state.native
                    and all(state.holds)
                ):
                    storage._operation_local.operation = core
            try:
                yield state
            except BaseException:
                state.body_failed = True
                raise
    finally:
        # Retire actual owned descriptors even if partial allocation lost a local
        # reference. An explicit uncertain close is never retried here or by GC.
        if state is not None:
            for fd in tuple(state.descriptors):
                if fd not in state.failed_closes:
                    try:
                        _close(state, fd)
                    except BaseException:
                        pass
            if not state.uncertain and not state.descriptors and not state.streams:
                for lease in reversed(state.leases):
                    lease.close()
                with storage._changed:
                    _states.pop(state, None)
                    storage._raw_operations.discard(state)
                    if state.result is True and state.publication_cleanup is not None:
                        record = _publications.pop(id(state.publication_cleanup), None)
                        if (
                            record is not None
                            and record[0] is state.publication_cleanup
                            and record[1] is state.source
                            and state.source is not None
                        ):
                            original = record[3]
                            state.source.failures[:] = [
                                f for f in state.source.failures if f is not original
                            ]
                    if state.candidate is not None and state.candidate._published:
                        _issued.pop(id(state.candidate), None)
                    storage._changed.notify_all()
            if (
                state.source is not None
                and state.body_failed
                and (
                    state.publication_candidate is not None
                    or any(
                        p in state.files and p in state.writable for p in state.owned
                    )
                )
                and state.result is None
            ):
                with storage._changed:
                    state.source.failures.append(state)
            state.active = False
        _local.state = previous
        storage._operation_local.operation = core
        if state is not None and state.uncertain:
            raise VisualIdentityNativeError(state)


# Indirection is intentionally module-local: fault injection must not replace
# process-global os primitives used by capability discovery or native admission.
def _open_native(*args, **kwargs):
    return os.open(*args, **kwargs)


def _close_native(fd):
    os.close(fd)


def native_open(value, flags, mode=0o777, *, dir_fd=None):
    state = current()
    if state is None:
        return _open_native(value, flags, mode, dir_fd=dir_fd)
    selected = _path(value, dir_fd)
    writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    _check_path(state, selected, writing=writing)
    try:
        fd = _open_native(value, flags, mode, dir_fd=dir_fd)
    except OSError as error:
        if error.errno not in {
            errno.ENOENT,
            errno.EEXIST,
            errno.EACCES,
            errno.ENOTDIR,
            errno.ELOOP,
        }:
            state.uncertain = True
        raise
    except BaseException:
        state.uncertain = True
        raise
    state.descriptors[fd] = selected
    info = os.fstat(fd)
    identity = (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))
    if flags & os.O_CREAT and flags & os.O_EXCL:
        state.expectations[selected] = identity
        state.owned[selected] = identity
    if state.expectations[selected] != identity or _identity(selected) != identity:
        raise bootstrap.RecoveryRequired("visual_identity_file_identity_changed")
    return fd


def _close(state, fd):
    if fd in state.failed_closes:
        raise VisualIdentityNativeError(state)
    try:
        _close_native(fd)
    except BaseException:
        state.failed_closes.add(fd)
        state.uncertain = True
        raise VisualIdentityNativeError(state) from None
    state.descriptors.pop(fd, None)


def native_close(fd):
    state = getattr(_local, "state", None)
    if state is None:
        return _close_native(fd)
    if fd not in state.descriptors:
        raise bootstrap.RecoveryRequired("visual_identity_descriptor_not_owned")
    return _close(state, fd)


def mkdir(value, mode=0o777, *, dir_fd=None):
    state = current()
    if state is None:
        return os.mkdir(value, mode, dir_fd=dir_fd)
    selected = _path(value, dir_fd)
    _check_path(state, selected, writing=True)
    os.mkdir(value, mode, dir_fd=dir_fd)
    # Pin the directory immediately using the exact parent/leaf, and carry its
    # owned inode to all later checks (never rediscover a published destination).
    try:
        fd = _open_native(
            value, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd
        )
    except BaseException:
        state.uncertain = True
        raise
    state.descriptors[fd] = selected
    info = os.fstat(fd)
    state.expectations[selected] = (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))
    state.owned[selected] = state.expectations[selected]
    _check_path(state, selected)
    _close(state, fd)


def unlink(value, *, dir_fd=None):
    state = current()
    if state is None:
        return os.unlink(value, dir_fd=dir_fd)
    selected = _path(value, dir_fd)
    _check_path(state, selected, writing=True)
    os.unlink(value, dir_fd=dir_fd)
    state.expectations[selected] = None
    state.owned.pop(selected, None)


def rmdir(value, *, dir_fd=None):
    state = current()
    if state is None:
        return os.rmdir(value, dir_fd=dir_fd)
    selected = _path(value, dir_fd)
    _check_path(state, selected, writing=True)
    os.rmdir(value, dir_fd=dir_fd)
    state.expectations[selected] = None
    state.owned.pop(selected, None)


def published(source, destination):
    """Carry the selected owned native tree identity across its actual rename."""
    state = current()
    if state is None:
        return
    source, destination = Path(source), Path(destination)
    mapping = {
        p: destination / p.relative_to(source)
        for p in state.expectations
        if p == source or source in p.parents
    }
    identities = {p: state.expectations[p] for p in mapping}
    for old, new in mapping.items():
        if new not in state.expectations:
            raise bootstrap.RecoveryRequired("visual_identity_path_outside_scope")
        state.expectations[new] = identities[old]
        state.expectations[old] = None
        if old in state.owned:
            state.owned[new] = state.owned.pop(old)
    for fd, p in tuple(state.descriptors.items()):
        if p in mapping:
            state.descriptors[fd] = mapping[p]
    for new in mapping.values():
        _check_path(state, new)


def private_directory(path):
    state = current()
    if state is None:
        return None
    selected = profile_paths.lexical_path(path)
    if selected not in state.directories:
        raise bootstrap.RecoveryRequired("visual_identity_helper_not_supported")
    _check_path(state, selected, writing=True)
    return state


def directory_created(path, fd):
    state = current()
    if state is not None:
        selected = profile_paths.lexical_path(path)
        info = os.fstat(fd)
        if selected not in state.writable:
            raise bootstrap.RecoveryRequired("visual_identity_path_outside_scope")
        state.expectations[selected] = (
            info.st_dev,
            info.st_ino,
            stat.S_IFMT(info.st_mode),
        )


def safe_point(profile):
    source = source_for(profile)
    if source is None:
        return "unqualified"
    with storage._changed:
        if source.failures or any(s.source is source for s in _states):
            return "incomplete"
    if _dirty(source):
        return "needs-user-save/discard"
    return "source-idle"


_publications = {}


def remember_publication(token, candidate):
    state = current()
    if state is not None and token is not None:
        state.publication_candidate = token
        selected = {
            p: identity
            for p, identity in state.expectations.items()
            if p == candidate or candidate in p.parents
        }
        _publications[id(token)] = (token, state.source, selected, state)
    return token


def cleanup_selection(db, token, profile, candidate):
    source = db_source(db)
    if source is not None and source.profile != profile:
        raise bootstrap.RecoveryRequired("visual_identity_source_changed")
    record = _publications.get(id(token))
    if record is not None and record[0] is token:
        if (
            record[1] is not source
            or source is not None
            and (record[3].repository is not db or source.profile != profile)
        ):
            raise bootstrap.RecoveryRequired("visual_identity_source_changed")
        for path, identity in record[2].items():
            if _identity(path) != identity:
                raise bootstrap.RecoveryRequired(
                    "visual_identity_candidate_identity_changed"
                )
        return source, tuple(record[2]), record
    # Older or copied ordinary relpaths never inherit installed across-pause
    # authority and cannot reconcile a registered failure. Fresh admission only.
    core = getattr(storage._operation_local, "operation", None)
    storage._operation_local.operation = None
    try:
        with request(), storage.acquire_storage(candidate):
            children = tuple(candidate.iterdir())
            if len(children) > 257 or any(p.is_dir() for p in children):
                raise bootstrap.RecoveryRequired("visual_identity_cleanup_denied")
    finally:
        storage._operation_local.operation = core
    return None, children, None


def _check_producer(frame):
    module = sys.modules.get("tldw_chatbook.Character_Chat.visual_identity")
    names = (
        "publish_visual_identity_candidate",
        "cleanup_visual_identity_publication_candidate",
        "_read_builtin_asset",
        "_read_user_asset",
        "ensure_builtin_samira",
        "_read_samira_resource",
    )
    for name in names:
        function = getattr(module, name, None)
        while function is not None:
            if (
                getattr(function, "__code__", None) is frame.f_code
                and frame.f_globals is module.__dict__
            ):
                return name
            function = getattr(function, "__wrapped__", None)
    if (
        frame.f_globals is module.__dict__
        and frame.f_code in _producer_codes
        and frame.f_code.co_name in names
    ):
        return frame.f_code.co_name
    raise bootstrap.RecoveryRequired("visual_identity_producer_not_installed")


@contextmanager
def stream(path, mode):
    state = current()
    selected = profile_paths.lexical_path(path)
    if state is not None:
        _check_path(state, selected, writing=mode != "rb")
    flags = os.O_RDONLY if mode == "rb" else os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = native_open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        handle = os.fdopen(fd, mode, closefd=False)
    except BaseException:
        native_close(fd)
        raise
    if state is not None:
        state.streams.append(handle)
    try:
        yield handle
    finally:
        try:
            handle.close()
            if not handle.closed:
                raise RuntimeError("visual_identity_stream_not_retired")
        except BaseException:
            if state is not None:
                state.uncertain = True
                state.failed_closes.add(fd)
            raise
        if state is not None:
            state.streams.remove(handle)
        native_close(fd)


def reader_guard(function):
    _producer_codes.add(function.__code__)

    @wraps(function)
    def guarded(*args, **kwargs):
        with request():
            return function(*args, **kwargs)

    return guarded


@contextmanager
def resource_stream(candidate):
    """Retain non-filesystem package streams as ordinary unqualified resources."""
    state = current()
    if isinstance(candidate, os.PathLike):
        with stream(candidate, "rb") as handle:
            yield handle
        return
    handle = None
    try:
        try:
            handle = candidate.open("rb")
        except BaseException:
            if state is not None:
                state.uncertain = True
            raise
        if state is not None:
            state.streams.append(handle)
        yield handle
    finally:
        if handle is not None:
            try:
                handle.close()
                if not handle.closed:
                    raise RuntimeError("visual_identity_stream_not_retired")
            except BaseException:
                if state is not None:
                    state.uncertain = True
                raise
            if state is not None:
                state.streams.remove(handle)


def publication_errors(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        from ..Character_Chat.visual_identity import VisualIdentityPublicationError

        try:
            return function(*args, **kwargs)
        except bootstrap.RecoveryRequired as error:
            mapped = VisualIdentityPublicationError(
                "visual_identity_cleanup_denied"
                if function.__name__.startswith("cleanup")
                else "visual_identity_publication_denied",
                cleanup_candidate_relpath=getattr(error, "cleanup_candidate", None),
            )
            mapped.result = getattr(error, "result", None)
            raise mapped from None

    return guarded


def validate_cleanup_contents(fd):
    state = current()
    if state is None:
        return
    directory = state.descriptors[fd]
    _check_path(state, directory)
    names = os.listdir(fd)
    for name in names:
        _check_path(state, directory / name, writing=True)
    _check_path(state, directory)


def _dirty(source):
    return any(
        issuing_source is source
        and not value._published
        and not value._cancelled
        and (restored or value._replacements or value._cleared)
        for value, _db, issuing_source, _original, restored in tuple(_issued.values())
    )


@contextmanager
def restoring_rows(candidate, assets):
    from ..Character_Chat.expression_generation import (
        canonical_visual_identity_reactions,
    )
    from ..UI.Screens.personas_screen import PersonasScreen

    frame = sys._getframe(2)
    actual = PersonasScreen._restore_candidate_reaction_rows
    if (
        frame.f_code is not actual.__code__
        or frame.f_globals is not sys.modules[actual.__module__].__dict__
    ):
        raise bootstrap.RecoveryRequired("visual_identity_candidate_update_denied")
    with request(), candidate._lock:
        record = _issued.get(id(candidate))
        if candidate._maintenance_identity is not None and (
            record is None or record[0] is not candidate
        ):
            raise bootstrap.RecoveryRequired("visual_identity_candidate_not_issued")
        if record is not None and record[0] is candidate and record[2] is not None:
            candidate_source(candidate, record[1], record[2].profile)
            candidate._ensure_stageable()
            known = {str(row["expression_key"]) for row in candidate.assets}
            canonical = {
                reaction.expression_key: (index, reaction)
                for index, reaction in enumerate(canonical_visual_identity_reactions())
            }
            missing = []
            for asset in assets:
                if asset.expression_key in known:
                    continue
                selected = canonical.get(asset.expression_key)
                if selected is None:
                    raise bootstrap.RecoveryRequired(
                        "visual_identity_candidate_update_denied"
                    )
                index, reaction = selected
                if any(
                    row["expression_key"] == asset.expression_key for row in missing
                ):
                    raise bootstrap.RecoveryRequired(
                        "visual_identity_candidate_update_denied"
                    )
                if (
                    asset.asset_id,
                    asset.original_label,
                    asset.display_label,
                    asset.content_type,
                    asset.is_animated,
                ) != (
                    -(index + 1),
                    reaction.original_label,
                    reaction.display_label,
                    "image/png",
                    False,
                ):
                    raise bootstrap.RecoveryRequired(
                        "visual_identity_candidate_update_denied"
                    )
                missing.append(
                    {
                        "id": -(index + 1),
                        "expression_key": reaction.expression_key,
                        "original_expression_key": reaction.original_label,
                        "display_label": reaction.display_label,
                        "bytes": 0,
                        "source_context_json": json.dumps(
                            {"visual_direction": reaction.visual_direction}
                        ),
                    }
                )
            original = candidate.assets
            expected = tuple(missing)
            yield
            # The authenticated method may append exactly its canonical rows.
            # Check the entire old graph again before issuing the new shape.
            updated = candidate_shape(candidate)
            old_shape = dict(record[3][1])
            new_shape = dict(updated[1])
            if (
                new_shape.pop("assets") != _shape(original + expected)
                or old_shape.pop("assets") != _shape(original)
                or new_shape != old_shape
            ):
                raise bootstrap.RecoveryRequired(
                    "visual_identity_candidate_update_denied"
                )
            if expected:
                _issued[id(candidate)] = (
                    candidate,
                    record[1],
                    record[2],
                    candidate_shape(candidate),
                    True,
                )
        else:
            # Original ordinary in-memory API has no installed source authority.
            yield


def forget_cancelled(candidate):
    if candidate._cancelled:
        record = _issued.get(id(candidate))
        if record is not None and record[0] is candidate:
            _issued.pop(id(candidate), None)


def installed_code(function):
    while function is not None:
        if getattr(function, "__code__", None) in _producer_codes:
            return True
        function = getattr(function, "__wrapped__", None)
    return False


def replace_publication(replace, source, destination, **kwargs):
    """Call the public replacement hook without lending source admission."""
    if _check_producer(sys._getframe(1)) != "publish_visual_identity_candidate":
        raise bootstrap.RecoveryRequired("visual_identity_producer_not_installed")
    state = current()
    if state is None:
        raise bootstrap.RecoveryRequired("visual_identity_publication_scope_required")
    _check_path(state, _path(source, kwargs.get("src_dir_fd")), writing=True)
    _check_path(state, _path(destination, kwargs.get("dst_dir_fd")), writing=True)
    core = getattr(storage._operation_local, "operation", None)
    _local.state = None
    storage._operation_local.operation = None
    try:
        replace(source, destination, **kwargs)
    finally:
        _local.state = state
        storage._operation_local.operation = core


def evaluate_repository_publication_guard(guard):
    """Evaluate the two repository hooks without lending their core operation."""
    frame = sys._getframe(1)
    module = sys.modules.get("tldw_chatbook.DB.VisualIdentity_DB")
    if (
        module is None
        or frame.f_globals is not module.__dict__
        or frame.f_code not in _producer_codes
        or frame.f_code.co_name not in {"activate_pack", "publish_version"}
    ):
        raise bootstrap.RecoveryRequired("visual_identity_producer_not_installed")
    state = getattr(_local, "state", None)
    core = getattr(storage._operation_local, "operation", None)
    _local.state = None
    storage._operation_local.operation = None
    try:
        return bool(guard())
    finally:
        _local.state = state
        storage._operation_local.operation = core
