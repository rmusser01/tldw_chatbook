"""Concrete Persona Visual source/native lifetimes (ADR-126, rulings70–73).

These private scopes are ordinary source admission, never capture authority. File
selection is performed by the actual visual producers before any native mutation.
"""

import errno
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


@dataclass(eq=False)
class _Source:
    config: object
    config_path: Path
    profile: Path
    lock: object = field(default_factory=threading.RLock)
    closed: bool = False
    native: bool = field(default_factory=lambda: _native_available())
    failures: list = field(default_factory=list)

    @property
    def owner_id(self):
        return "persona.assets"

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
            return not self.failures

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
        raise bootstrap.RecoveryRequired("persona_visual_native_capability_changed")
    if (
        sys.modules.get("tldw_chatbook.config") is not config
        or profile_paths.lexical_path(config._get_effective_config_path())
        != source.config_path
        or config._CONFIG_CACHE_SOURCE != source.config_path
        or config._CONFIG_CACHE is None
        or profile_paths.user_data_dir(config._CONFIG_CACHE) != source.profile
    ):
        raise bootstrap.RecoveryRequired("persona_visual_source_changed")


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


def bind_repository(repository):
    from ..DB.ChaChaNotes_DB import CharactersRAGDB
    from ..Persona_Visual.repository import PersonaVisualRepository
    from .participants import _repository_participant

    if repository in _repositories:
        raise bootstrap.RecoveryRequired("persona_visual_repository_reinitialized")
    config = sys.modules.get("tldw_chatbook.config")
    db = repository.db
    if (
        type(repository) is not PersonaVisualRepository
        or type(db) is not CharactersRAGDB
        or db.is_memory_db
        or config is None
        or getattr(config, "_CONFIG_CACHE", None) is None
        or profile_paths.database_path(config._CONFIG_CACHE, "chachanotes_db_path")
        != db.db_path
    ):
        return
    _repository_participant(db)
    source = source_for(profile_paths.user_data_dir(config._CONFIG_CACHE))
    _repositories[repository] = (db, source)


def repository_source(repository):
    binding = _repositories.get(repository)
    if binding is None:
        return None
    db, source = binding
    from ..Persona_Visual.repository import PersonaVisualRepository
    from .participants import _repository_participant

    if type(repository) is not PersonaVisualRepository or repository.db is not db:
        raise bootstrap.RecoveryRequired("persona_visual_repository_changed")
    _validate_source(source)
    if (
        profile_paths.database_path(source.config._CONFIG_CACHE, "chachanotes_db_path")
        != db.db_path
    ):
        raise bootstrap.RecoveryRequired("persona_visual_repository_changed")
    _repository_participant(db)
    return source


def repository_guard(function):
    @wraps(function)
    def guarded(repository, *args, **kwargs):
        from .participants import _core_operation

        repository_source(repository)
        with _core_operation(repository.db):
            return function(repository, *args, **kwargs)

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


def issued(value):
    with storage._changed:
        record = _issued.get(id(value))
    if record is None or record[0] is not value:
        return None
    source, shape, pins = record[1:]
    if _shape(value) != shape:
        raise bootstrap.RecoveryRequired("persona_visual_candidate_changed")
    if source is not None:
        _validate_source(source)
    return record


def _shape(value):
    if is_dataclass(value):
        return type(value), tuple(
            (f.name, _shape(getattr(value, f.name))) for f in fields(value)
        )
    if isinstance(value, tuple):
        return tuple(_shape(item) for item in value)
    return value


def issue(value, source, *, replaces=None):
    state = current()
    pins = {}
    if replaces is not None:
        with storage._changed:
            old = _issued.pop(id(replaces), None)
        if old is not None:
            pins.update(old[3])
    if state is not None:
        for p, identity in state.expectations.items():
            if identity is not None and any(
                part.startswith((".draft-", ".import-")) for part in p.parts
            ):
                pins[p] = identity
    with storage._changed:
        _issued[id(value)] = (value, source, _shape(value), pins)
    return value


def forget(value):
    record = issued(value)
    if record is not None:
        with storage._changed:
            _issued.pop(id(value), None)
            if record[1] is not None:
                record[1].failures[:] = [
                    failure for failure in record[1].failures if failure is not value
                ]


def candidate_source(value, profile):
    actual = source_for(profile)
    with storage._changed:
        original = _issued.get(id(value))
    try:
        record = issued(value)
        if record is not None:
            for p, expected in record[3].items():
                if _identity(p) != expected:
                    raise bootstrap.RecoveryRequired(
                        "persona_visual_candidate_identity_changed"
                    )
    except BaseException:
        if original is not None and original[0] is value and original[1] is not None:
            with storage._changed:
                if not any(failure is value for failure in original[1].failures):
                    original[1].failures.append(value)
        raise
    if record is not None and record[1] is not None and record[1] is not actual:
        raise bootstrap.RecoveryRequired("persona_visual_candidate_changed")
    if actual is not None and (record is None or record[1] is not actual):
        raise bootstrap.RecoveryRequired("persona_visual_candidate_not_issued")
    return actual


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
    candidate_cleanup: object = None


class PersonaVisualNativeError(bootstrap.RecoveryRequired):
    def __init__(self, state):
        self.result = state.result
        self.cleanup_candidate = state.publication_candidate
        super().__init__("persona_visual_native_not_retired")


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
        raise PersonaVisualNativeError(state)
    if state.source is not None:
        _validate_source(state.source)
    if state.repository is not None:
        repository_source(state.repository)
    for lease, hold in zip(state.leases, state.holds):
        if (
            lease not in storage._live_leases
            or storage._holds.get(lease._key) is not hold
        ):
            raise PersonaVisualNativeError(state)
        if hold is not None and (hold.stop.is_set() or hold.error is not None):
            raise PersonaVisualNativeError(state)
        if storage._pause is not None and (
            hold is None or state.source is None or not state.source.native
        ):
            raise bootstrap.RecoveryRequired("storage_locally_paused")
    return state


def _path(value, dir_fd=None):
    selected = Path(os.fsdecode(value))
    if dir_fd is not None:
        state = current()
        if state is None or dir_fd not in state.descriptors:
            raise bootstrap.RecoveryRequired("persona_visual_descriptor_not_owned")
        selected = state.descriptors[dir_fd] / selected
    return profile_paths.lexical_path(selected)


def _check_path(state, selected, *, writing=False):
    check(state)
    if selected not in state.expectations or (
        writing and selected not in state.writable
    ):
        raise bootstrap.RecoveryRequired("persona_visual_path_outside_scope")
    expected = state.expectations[selected]
    if _identity(selected) != expected:
        raise bootstrap.RecoveryRequired("persona_visual_file_identity_changed")
    for parent in selected.parents:
        if (
            parent in state.expectations
            and _identity(parent) != state.expectations[parent]
        ):
            raise bootstrap.RecoveryRequired("persona_visual_parent_identity_changed")


@contextmanager
def files(source, paths, directories=(), *, writing=(), repository=None):
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
        if set(paths + directories).issubset(
            previous.expectations
        ) and writable.issubset(previous.writable):
            for p in paths + directories:
                _check_path(previous, p, writing=p in writable)
            yield previous
            return
    core = getattr(storage._operation_local, "operation", None)
    storage._operation_local.operation = None
    _local.state = None
    state = None
    acquired = False
    try:
        with request() as attempt:
            if source is not None:
                _validate_source(source)
                # Publication owns fresh fixed candidate paths and uses the
                # existing optimistic core transaction. Holding this profile
                # mutex across its public guard would invert the actual UI's
                # Persona source lock. Only this authenticated producer skips it.
                if producer != ("publication", "publish_persona_visual"):
                    while not source.lock.acquire(timeout=0.05):
                        attempt.check()
                    acquired = True
                if source.closed:
                    raise bootstrap.RecoveryRequired("storage_locally_paused")
            state = _NativeState(
                source, paths, directories, writable, attempt, repository
            )
            with storage._changed:
                if source is not None and any(
                    s.source is source and s.uncertain for s in _states
                ):
                    raise bootstrap.RecoveryRequired(
                        "persona_visual_native_not_retired"
                    )
                _states[state] = state
                storage._raw_operations.add(state)
            for p in paths + directories:
                for ancestor in (p, *p.parents):
                    if ancestor not in state.expectations:
                        state.expectations[ancestor] = _identity(ancestor)
            for p in paths + directories:
                attempt.check()
                lease = storage.acquire_storage(p)
                state.leases.append(lease)
                state.holds.append(storage._holds.get(lease._key))
            for p, identity in state.expectations.items():
                if _identity(p) != identity:
                    raise bootstrap.RecoveryRequired("persona_visual_preflight_changed")
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
                if source is not None and source.native and all(state.holds):
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
            if not state.uncertain and not state.descriptors:
                for lease in reversed(state.leases):
                    lease.close()
                with storage._changed:
                    _states.pop(state, None)
                    storage._raw_operations.discard(state)
                    if state.candidate_cleanup is not None:
                        if state.result is True:
                            forget(state.candidate_cleanup)
                        elif state.source is not None and not any(
                            failure is state.candidate_cleanup
                            for failure in state.source.failures
                        ):
                            state.source.failures.append(state.candidate_cleanup)
                    if state.result is True and state.publication_cleanup is not None:
                        record = _publications.pop(id(state.publication_cleanup), None)
                        if (
                            state.source is not None
                            and record is not None
                            and record[0] is state.publication_cleanup
                            and record[1] is state.source
                        ):
                            original = record[3]
                            state.source.failures[:] = [
                                failure
                                for failure in state.source.failures
                                if failure is not original
                            ]
                    storage._changed.notify_all()
            if (
                state.source is not None
                and state.body_failed
                and state.owned
                and state.result is None
            ):
                with storage._changed:
                    state.source.failures.append(state)
            state.active = False
        _local.state = previous
        storage._operation_local.operation = core
        if acquired:
            source.lock.release()
        if state is not None and state.uncertain:
            raise PersonaVisualNativeError(state)


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
        raise bootstrap.RecoveryRequired("persona_visual_file_identity_changed")
    return fd


def _close(state, fd):
    if fd in state.failed_closes:
        raise PersonaVisualNativeError(state)
    try:
        _close_native(fd)
    except BaseException:
        state.failed_closes.add(fd)
        state.uncertain = True
        raise PersonaVisualNativeError(state) from None
    state.descriptors.pop(fd, None)


def native_close(fd):
    state = getattr(_local, "state", None)
    if state is None:
        return _close_native(fd)
    if fd not in state.descriptors:
        raise bootstrap.RecoveryRequired("persona_visual_descriptor_not_owned")
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
            raise bootstrap.RecoveryRequired("persona_visual_path_outside_scope")
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
        raise bootstrap.RecoveryRequired("persona_visual_helper_not_supported")
    _check_path(state, selected, writing=True)
    return state


def directory_created(path, fd):
    state = current()
    if state is not None:
        selected = profile_paths.lexical_path(path)
        info = os.fstat(fd)
        if selected not in state.writable:
            raise bootstrap.RecoveryRequired("persona_visual_path_outside_scope")
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
    return "source-idle"


def validate_publication_sources(repository, snapshot, source_root, profile):
    """Permit overlap only for this core graph or one actual issued generation."""
    source = repository_source(repository)
    if source is None or source.profile != profile:
        return False
    if source_root == profile and snapshot.expected_identity is not None:
        graph = repository.get_active_persona_pack(snapshot.persona_id)
        if graph is not None and graph.identity == snapshot.expected_identity:
            from ..Persona_Visual.assets import PersonaVisualAssetMetadata

            selected = {
                repository._get_active_asset_storage_key(
                    graph.identity, asset
                ): PersonaVisualAssetMetadata(
                    asset.asset_key,
                    asset.role,
                    asset.mime_type,
                    asset.byte_count,
                    asset.sha256,
                    asset.width,
                    asset.height,
                    asset.frame_count,
                    asset.duration_ms,
                )
                for asset in graph.assets
            }
            if all(
                selected.get(asset.source_storage_key) == asset.metadata
                for asset in snapshot.assets
            ):
                return True
    from ..Persona_Visual.authoring_workspace import PersonaVisualAuthoringWorkspace
    from ..Persona_Visual.importer import PersonaVisualImportReview

    with storage._changed:
        issued_candidates = tuple(_issued.values())
    for value, issuing_source, _shape_value, _pins in issued_candidates:
        if issuing_source is not source:
            continue
        if type(value) is PersonaVisualAuthoringWorkspace:
            if source_root != profile:
                continue
            selected = {
                profile / value.relative_root / "assets" / asset.name: asset.sha256
                for asset in value._assets
            }
        elif type(value) is PersonaVisualImportReview:
            candidate = profile / "persona_visual" / "imports" / value._candidate_name
            if source_root != candidate:
                continue
            selected = {
                candidate / asset.source_storage_key: asset.metadata.sha256
                for asset in value.draft.assets
            }
        else:
            continue
        if all(
            selected.get(source_root / asset.source_storage_key)
            == asset.metadata.sha256
            for asset in snapshot.assets
        ):
            candidate_source(value, profile)
            return True
    return False


_publications = {}


def remember_publication(capability, candidate):
    state = current()
    if state is not None:
        state.publication_candidate = capability
        selected = {
            p: identity
            for p, identity in state.expectations.items()
            if p == candidate or candidate in p.parents
        }
        with storage._changed:
            _publications[id(capability)] = (capability, state.source, selected, state)
    return capability


def publication_selection(repository, capability, profile, candidate):
    source = repository_source(repository)
    with storage._changed:
        record = _publications.get(id(capability))
    if record is not None and record[0] is not capability:
        record = None
    if source is not None:
        if source.profile != profile or record is None or record[1] is not source:
            raise bootstrap.RecoveryRequired("persona_visual_candidate_not_issued")
    if record is not None:
        for p, expected in record[2].items():
            if _identity(p) != expected:
                raise bootstrap.RecoveryRequired(
                    "persona_visual_candidate_identity_changed"
                )
        selected = tuple(
            p
            for p, ident in record[2].items()
            if ident is not None and ident[2] != stat.S_IFDIR
        )
    else:
        # An ordinary caller's older capability is not installed authority. Keep
        # its existing bounded cleanup contract, with a fresh ordinary scope.
        with storage.acquire_storage(candidate):
            selected = tuple(
                candidate / name
                for name in (".persona-visual-cleanup", "manifest.json")
            )
            try:
                with os.scandir(candidate / "assets") as entries:
                    names = [entry.name for entry in entries]
                if len(names) > 256:
                    raise ValueError("persona_visual_candidate_invalid")
                selected += tuple(candidate / "assets" / name for name in names)
            except FileNotFoundError:
                pass
    return source, selected


def _check_producer(frame):
    allowed = {
        "publication": {
            "publish_persona_visual",
            "cleanup_persona_visual_publication_candidate",
        },
        "authoring_workspace": {
            "create_persona_visual_authoring_workspace",
            "_write_workspace_asset",
            "cleanup_persona_visual_authoring_workspace",
        },
        "importer": {
            "import_persona_visual_pack",
            "persona_visual_import_source_root",
            "cleanup_persona_visual_import_review",
        },
        "assets": {"_read_profile_file"},
    }
    for module_name, names in allowed.items():
        module = sys.modules.get("tldw_chatbook.Persona_Visual." + module_name)
        for name in names:
            function = getattr(module, name, None)
            while function is not None:
                if (
                    getattr(function, "__code__", None) is frame.f_code
                    and frame.f_globals is module.__dict__
                ):
                    return module_name, name
                function = getattr(function, "__wrapped__", None)
    raise bootstrap.RecoveryRequired("persona_visual_producer_not_installed")
