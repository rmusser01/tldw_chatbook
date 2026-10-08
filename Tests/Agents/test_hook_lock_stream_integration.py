"""Config and hook locks keep actual stream, durability and ownership contracts."""

import errno
import inspect
import sys
from contextlib import closing, contextmanager

import portalocker
import pytest

from Tests.Backup_Recovery import test_raw_native_retirement as source_cases
from tldw_chatbook.Agents import hook_permissions
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import private_paths

configured_source = source_cases.configured_source
local_root = source_cases.local_root


@pytest.fixture
def lock_case(configured_source, monkeypatch, request):
    source = configured_source
    route = request.param
    if route == "config":
        path = source.get_cli_config_path()
        owner = source
    else:
        assert route == "hook"
        monkeypatch.setattr(hook_permissions, "config", source)
        owner = hook_permissions.HookPermissions()
        initial = owner.snapshot()
        assert initial.ready and not initial.rows
        path = initial.store_path
    lock = path.with_name(path.name + ".lock")
    try:
        yield source, owner, path, lock, route
    finally:
        if route == "hook":
            owner.close()


@contextmanager
def _acquire(case):
    source, owner, path, lock, route = case
    if route == "config":
        with source._config_interprocess_lock(path):
            yield None
    else:
        with source.locked_hooks_config_snapshot():
            parent = raw._local.operation
            assert raw._states[parent].source is source
            with owner._store_lock(path):
                yield parent
            assert raw._local.operation is parent


def _assert_closed(fd):
    with pytest.raises(OSError) as error:
        private_paths.os.fstat(fd)
    assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("lock_case", ["config", "hook"], indirect=True)
@pytest.mark.parametrize("warm", [False, True])
def test_original_lock_acquisition_preserves_durability_stream_and_exclusion(
    lock_case, monkeypatch, warm
):
    source, owner, path, lock, route = lock_case
    if warm:
        with _acquire(lock_case):
            pass
        assert lock.is_file()
        lock.write_bytes(b"preserved lock bytes")
        original_info = private_paths.os.stat(lock, follow_symlinks=False)
    else:
        if lock.exists():
            private_paths.os.unlink(lock)
        assert not lock.exists()
        original_info = None
    leases, operations = set(storage._live_leases), set(raw._states)
    real_lock, real_unlock = portalocker.lock, portalocker.unlock
    real_fsync = private_paths.os.fsync
    fsyncs, streams, events, helper_calls = [], [], [], []
    lock_code = inspect.unwrap(
        source._config_interprocess_lock if route == "config" else owner._store_lock
    ).__code__
    helper = getattr(private_paths, "open_private_lock_stream", None)
    helper_code = inspect.unwrap(helper).__code__ if helper is not None else None
    previous = sys.getprofile()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event != "call" or frame.f_code is not helper_code:
            return
        parent = frame.f_back
        while parent is not None and parent.f_code is not lock_code:
            parent = parent.f_back
        if parent is not None:
            assert private_paths.lexical_path(frame.f_locals["path"]) == lock
            helper_calls.append(frame.f_locals["application_owned_directory"])

    def fsync(fd):
        opened = private_paths.os.fstat(fd)
        try:
            named = private_paths.os.stat(lock, follow_symlinks=False)
        except FileNotFoundError:
            return real_fsync(fd)
        if (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino):
            fsyncs.append(fd)
            events.append("fsync")
        return real_fsync(fd)

    def locking(stream, flags):
        # Only the owning integration call is recorded; contender uses real_lock.
        if sys._getframe(1).f_code is lock_code:
            operation = raw._local.operation
            state = raw._states[operation]
            fd = stream.fileno()
            assert state.source is owner and state.active
            assert operation in storage._raw_operations
            assert fd in state.descriptors and stream._stream in state.files
            assert stream._operation is operation
            opened = private_paths.os.fstat(fd)
            named = private_paths.os.stat(lock, follow_symlinks=False)
            assert (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
            streams.append((stream, fd, operation))
            events.append("lock")
        return real_lock(stream, flags)

    def unlocking(stream):
        if streams and stream is streams[0][0]:
            assert not stream.closed
            events.append("unlock")
        return real_unlock(stream)

    monkeypatch.setattr(private_paths.os, "fsync", fsync)
    monkeypatch.setattr(portalocker, "lock", locking)
    monkeypatch.setattr(portalocker, "unlock", unlocking)
    sys.setprofile(observe)
    try:
        with _acquire(lock_case) as parent:
            assert len(streams) == 1
            stream, fd, operation = streams[0]
            assert raw._states[operation].source is owner
            if route == "hook":
                assert parent is not operation and parent in raw._states
            if warm:
                assert fsyncs == [] and events == ["lock"]
                opened = private_paths.os.fstat(fd)
                assert (opened.st_dev, opened.st_ino) == (
                    original_info.st_dev,
                    original_info.st_ino,
                )
            else:
                assert fsyncs == [fd] and events == ["fsync", "lock"]
            with closing(private_paths.open_private_text_append_stream(lock)) as other:
                assert other.fileno() != fd
                with pytest.raises(portalocker.exceptions.LockException):
                    real_lock(
                        other,
                        portalocker.LockFlags.EXCLUSIVE
                        | portalocker.LockFlags.NON_BLOCKING,
                    )
            assert not stream.closed and operation in raw._states
    finally:
        sys.setprofile(previous)
    stream, fd, operation = streams[0]
    assert stream.closed and stream._retired
    _assert_closed(fd)
    assert events[-1] == "unlock"
    assert set(raw._states) == operations and set(storage._live_leases) == leases
    assert lock.read_bytes() == (b"preserved lock bytes" if warm else b"")
    expected_directory = (
        source.application_owned_config_directory(path)
        if route == "config"
        else path.parent
    )
    assert helper_calls == [
        expected_directory
    ], "lock creation and stream were not one named operation"


@pytest.mark.parametrize("lock_case", ["config", "hook"], indirect=True)
def test_lock_body_failure_unlocks_and_retires_original_stream(lock_case, monkeypatch):
    source, owner, path, lock, route = lock_case
    real_lock, real_unlock = portalocker.lock, portalocker.unlock
    leases, operations = set(storage._live_leases), set(raw._states)
    acquired, unlocked = [], []
    lock_code = inspect.unwrap(
        source._config_interprocess_lock if route == "config" else owner._store_lock
    ).__code__

    def locking(stream, flags):
        result = real_lock(stream, flags)
        if sys._getframe(1).f_code is lock_code:
            operation = raw._local.operation
            state = raw._states[operation]
            assert state.source is owner and stream._stream in state.files
            assert stream.fileno() in state.descriptors
            acquired.append((stream, stream.fileno(), operation))
        return result

    def unlocking(stream):
        result = real_unlock(stream)
        if acquired and stream is acquired[0][0]:
            unlocked.append(stream)
        return result

    class BodyFailure(RuntimeError):
        pass

    monkeypatch.setattr(portalocker, "lock", locking)
    monkeypatch.setattr(portalocker, "unlock", unlocking)
    with pytest.raises(BodyFailure):
        with _acquire(lock_case):
            assert len(acquired) == 1
            raise BodyFailure("original owning lock body failed")
    stream, fd, operation = acquired[0]
    assert unlocked == [stream] and stream.closed and stream._retired
    _assert_closed(fd)
    assert set(raw._states) == operations and set(storage._live_leases) == leases
    assert lock.is_file()
