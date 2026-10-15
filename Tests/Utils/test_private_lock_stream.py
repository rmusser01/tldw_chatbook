"""Lock-stream durability uses the original native FD and admitted lifetime."""

import errno
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import stat
import sys

import pytest

from Tests.Backup_Recovery import test_raw_native_retirement as retirement_cases
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import private_paths

configured_source = retirement_cases.configured_source
local_root = retirement_cases.local_root


@pytest.fixture
def lock_path(configured_source):
    selected = configured_source._get_effective_config_path()
    target = selected.with_name(selected.name + ".lock")
    target.unlink(missing_ok=True)
    return target


def _record_actual_target_syncs(monkeypatch, target):
    synced = []
    original = private_paths.os.fsync

    def sync(fd):
        result = original(fd)
        opened = private_paths.os.fstat(fd)
        if stat.S_ISREG(opened.st_mode):
            named = private_paths.os.stat(target, follow_symlinks=False)
            if (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino):
                synced.append(fd)
        return result

    monkeypatch.setattr(private_paths.os, "fsync", sync)
    return synced


@pytest.mark.parametrize("cold", [True, False])
def test_lock_stream_syncs_only_actual_creation_and_keeps_that_fd(
    configured_source, lock_path, monkeypatch, cold
):
    if not cold:
        lock_path.write_bytes(b"existing")
        lock_path.chmod(0o600)
    synced = _record_actual_target_syncs(monkeypatch, lock_path)
    close_code = private_paths._native_close.__code__
    closed_before_return = []
    previous = sys.getprofile()
    before_leases = set(storage._live_leases)

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if frame.f_code is close_code and event == "call":
            if frame.f_locals["fd"] in synced:
                closed_before_return.append(frame.f_locals["fd"])

    with raw._scope(configured_source, "config", writing=True) as operation:
        state = raw._states[operation]
        sys.setprofile(observe)
        try:
            stream = private_paths.open_private_lock_stream(
                lock_path,
                application_owned_directory=configured_source.application_owned_config_directory(
                    configured_source._get_effective_config_path()
                ),
            )
        finally:
            sys.setprofile(previous)
        fd = stream.fileno()
        assert fd in state.descriptors
        assert synced == ([fd] if cold else [])
        assert closed_before_return == [], "created FD was closed and reopened"
        assert private_paths.os.fstat(fd).st_nlink == 1
        assert stat.S_IMODE(private_paths.os.fstat(fd).st_mode) == 0o600
        stream.close()
        assert fd not in state.descriptors
        with pytest.raises(OSError) as error:
            private_paths.os.fstat(fd)
        assert error.value.errno == errno.EBADF
        stream.close()  # Existing admitted close remains idempotent.
    assert operation not in raw._states
    assert set(storage._live_leases) == before_leases
    assert lock_path.read_bytes() == (b"" if cold else b"existing")


@pytest.mark.parametrize("cold", [True, False])
def test_original_append_api_does_not_gain_lock_fsync(
    configured_source, lock_path, monkeypatch, cold
):
    if not cold:
        lock_path.write_bytes(b"existing")
        lock_path.chmod(0o600)
    synced = _record_actual_target_syncs(monkeypatch, lock_path)
    with raw._scope(configured_source, "config", writing=True) as operation:
        stream = private_paths.open_private_text_append_stream(
            lock_path,
            application_owned_directory=configured_source.application_owned_config_directory(
                configured_source._get_effective_config_path()
            ),
        )
        fd = stream.fileno()
        assert fd in raw._states[operation].descriptors
        assert synced == []
        stream.close()
        assert fd not in raw._states[operation].descriptors
    assert operation not in raw._states
    assert lock_path.read_bytes() == (b"" if cold else b"existing")


@contextmanager
def _after_actual_exclusive_rejection(target, action):
    code = private_paths._native_open.__code__
    previous = sys.getprofile()
    reached = []

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if frame.f_code is not code or event != "return" or reached:
            return
        args = frame.f_locals.get("args", ())
        outcome = frame.f_locals.get("_outcome")
        if (
            len(args) > 1
            and args[0] == target.name
            and args[1] & private_paths.os.O_EXCL
            and outcome is not None
            and outcome.rejected
            and outcome.descriptor is None
        ):
            reached.append(True)
            action()

    sys.setprofile(observe)
    try:
        yield reached
    finally:
        sys.setprofile(previous)


def test_confirmed_existing_lock_that_disappears_is_not_recreated(
    configured_source, lock_path
):
    lock_path.write_bytes(b"original")
    lock_path.chmod(0o600)
    with raw._scope(configured_source, "config", writing=True):
        with _after_actual_exclusive_rejection(lock_path, lock_path.unlink) as reached:
            with pytest.raises((OSError, private_paths.PrivatePathError)):
                private_paths.open_private_lock_stream(lock_path)
        assert reached == [True]
    assert not lock_path.exists()


@pytest.mark.parametrize("unsafe", ["directory", "hardlink"])
def test_confirmed_existing_lock_replaced_with_unsafe_leaf_refuses(
    configured_source, lock_path, tmp_path, unsafe
):
    lock_path.write_bytes(b"original")
    lock_path.chmod(0o600)
    source = tmp_path / "separate-private-file"
    source.write_bytes(b"untouched")
    source.chmod(0o600)

    def replace_leaf():
        lock_path.unlink()
        if unsafe == "directory":
            lock_path.mkdir(mode=0o700)
        else:
            os.link(source, lock_path)

    with raw._scope(configured_source, "config", writing=True):
        with _after_actual_exclusive_rejection(lock_path, replace_leaf) as reached:
            with pytest.raises((OSError, private_paths.PrivatePathError)):
                private_paths.open_private_lock_stream(lock_path)
        assert reached == [True]
    assert source.read_bytes() == b"untouched"
    if unsafe == "directory":
        assert lock_path.is_dir()
    else:
        assert lock_path.read_bytes() == b"untouched"
        assert lock_path.stat().st_nlink == 2


def test_replacement_between_existing_lstat_and_actual_open_refuses(
    configured_source, lock_path, tmp_path
):
    lock_path.write_bytes(b"original")
    lock_path.chmod(0o600)
    successor = tmp_path / "replacement"
    successor.write_bytes(b"successor")
    successor.chmod(0o600)
    code = private_paths._native_open.__code__
    previous = sys.getprofile()
    changed = []

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if frame.f_code is not code or event != "call" or changed:
            return
        args = frame.f_locals.get("args", ())
        if (
            len(args) > 1
            and args[0] == lock_path.name
            and not args[1] & private_paths.os.O_EXCL
        ):
            os.replace(successor, lock_path)
            changed.append(True)

    with raw._scope(configured_source, "config", writing=True):
        sys.setprofile(observe)
        try:
            with pytest.raises((OSError, private_paths.PrivatePathError)):
                private_paths.open_private_lock_stream(lock_path)
        finally:
            sys.setprofile(previous)
    assert changed == [True]
    assert lock_path.read_bytes() == b"successor"


def test_ambiguous_allocator_creation_then_fileexists_does_not_retry(
    configured_source, lock_path, monkeypatch
):
    original = private_paths.os.open
    allocated, attempts = [], []

    def allocate_then_raise(path, flags, *args, **kwargs):
        if path == lock_path.name and flags & private_paths.os.O_EXCL:
            fd = original(path, flags, *args, **kwargs)
            allocated.append(fd)
            attempts.append(True)
            raise FileExistsError(errno.EEXIST, "allocator raised after real creation")
        if path == lock_path.name:
            attempts.append(False)
        return original(path, flags, *args, **kwargs)

    try:
        with raw._scope(configured_source, "config", writing=True):
            with monkeypatch.context() as changed:
                changed.setattr(private_paths.os, "open", allocate_then_raise)
                # This provider really delegates every directory-relative call;
                # retain that supported capability so the verified branch runs.
                if allocate_then_raise not in private_paths.os.supports_dir_fd:
                    changed.setattr(
                        private_paths.os,
                        "supports_dir_fd",
                        private_paths.os.supports_dir_fd | {allocate_then_raise},
                    )
                assert allocate_then_raise in private_paths.os.supports_dir_fd
                with pytest.raises((OSError, private_paths.PrivatePathError)):
                    private_paths.open_private_lock_stream(lock_path)
        assert attempts == [True]
        assert len(allocated) == 1
        assert private_paths.os.fstat(allocated[0]).st_nlink == 1
        assert lock_path.read_bytes() == b""
    finally:
        # The substituted allocator owns this observed FD. Its exception cannot
        # prove native rejection or authorize a second allocation.
        for fd in allocated:
            private_paths.os.close(fd)


def test_two_real_first_creators_keep_one_inode_and_retire_streams(
    configured_source, lock_path
):
    barrier = threading.Barrier(2)
    code = private_paths._native_open.__code__
    before_leases = set(storage._live_leases)

    def create():
        previous = sys.getprofile()
        entered = []

        def observe(frame, event, result):
            if previous is not None:
                previous(frame, event, result)
            if frame.f_code is not code or event != "call" or entered:
                return
            args = frame.f_locals.get("args", ())
            if (
                len(args) > 1
                and args[0] == lock_path.name
                and args[1] & private_paths.os.O_EXCL
            ):
                entered.append(True)
                barrier.wait(timeout=10)

        sys.setprofile(observe)
        try:
            stream = private_paths.open_private_lock_stream(lock_path)
            try:
                info = private_paths.os.fstat(stream.fileno())
                return info.st_dev, info.st_ino, entered
            finally:
                stream.close()
        finally:
            sys.setprofile(previous)

    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = workers.submit(create), workers.submit(create)
        results = first.result(timeout=15), second.result(timeout=15)
    assert results[0][:2] == results[1][:2]
    assert results[0][2] == results[1][2] == [True]
    assert lock_path.read_bytes() == b""
    assert set(storage._live_leases) == before_leases
