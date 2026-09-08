"""Real profile backup native ownership under injected close uncertainty."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


async def _backup_close_child(root, after):
    import asyncio
    import sqlite3
    import time

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    actual_connect = module.connect_private_sqlite
    calls = []
    allocated = []

    def connect(owner, path, **kwargs):
        native = actual_connect(owner, path, **kwargs)
        if (
            owner == "tts.profile_backup"
            and str(path).endswith(".backup")
            and not allocated
        ):
            allocated.append((native, path))
            actual_close = type(native).close

            def close(value):
                if value is native:
                    calls.append(value)
                    if after:
                        actual_close(value)
                    raise OSError("injected backup destination close uncertainty")
                return actual_close(value)

            type(native).close = close
        return native

    module.connect_private_sqlite = connect
    with pytest.raises(ProfileRepositoryError) as caught:
        await repo.backup_to(root / "snapshot.sqlite")
    assert caught.value.code == "backup_failed"
    native, temporary = allocated[0]
    assert len(calls) == 1, "an uncertain native close must never be retried"
    assert temporary.exists(), "uncertain destination must retain its exact namespace"

    def observe_native():
        if after:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                native.execute("SELECT 1")
        else:
            assert native.execute("SELECT 1").fetchone()[0] == 1

    await asyncio.wrap_future(repo._executor.submit(observe_native))
    retained = next(iter(repo._backup_native_operations))
    assert retained.connection is native
    assert retained.repository is repo and retained.lease in storage._live_leases
    assert retained.body_error is not None and not retained.published
    # An unrelated successful backup cannot retire an older failed operation.
    await repo.backup_to(root / "later.sqlite")
    assert repo._backup_native_operations == {retained}
    await repo.close()
    assert repo.terminal and repo._connection is None
    assert retained.connection is native and len(calls) == 1
    repo._maintenance_close_admission()
    pause = storage._begin_local_pause()
    assert not await repo._maintenance_drain(time.monotonic() + 2)
    assert not pause.drain(time.monotonic() + 0.03)
    assert len(calls) == 1
    pause.resume()
    # Failed resources remain in this private process until exit; no registry reset.


@pytest.mark.parametrize("after", [False, True])
def test_backup_destination_close_uncertainty_is_retained_without_retry(
    tmp_path, after
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_native_maintenance import _backup_close_child
asyncio.run(_backup_close_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _backup_foreign_cleanup_child(root):
    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    actual_backup = repo._worker_online_backup
    replaced = []

    def backup(source, destination, **kwargs):
        actual_backup(source, destination, **kwargs)
        temporary = next(root.glob(".snapshot.sqlite.*.backup"))
        temporary.rename(root / "owned-original.backup")
        temporary.write_bytes(b"foreign sentinel")
        replaced.append(temporary)
        raise OSError("injected body failure after foreign namespace substitution")

    repo._worker_online_backup = backup
    try:
        with pytest.raises(ProfileRepositoryError):
            await repo.backup_to(root / "snapshot.sqlite")
        assert replaced[0].read_bytes() == b"foreign sentinel"
    finally:
        await repo.close()


def test_backup_failure_never_removes_substituted_foreign_temporary(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_native_maintenance import _backup_foreign_cleanup_child
asyncio.run(_backup_foreign_cleanup_child(Path(sys.argv[1])))
""",
    )


async def _backup_descriptor_close_child(root, after):
    import asyncio
    import os
    import time
    import types

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    native_os = module.os
    actual_close = native_os.close
    calls = []
    allocated = []
    actual_mkstemp = module.tempfile.mkstemp

    def mkstemp(**kwargs):
        descriptor, name = actual_mkstemp(**kwargs)
        if str(name).endswith(".backup"):
            allocated.append((descriptor, name))
        return descriptor, name

    def close(descriptor):
        if allocated and descriptor == allocated[0][0]:
            calls.append(descriptor)
            if after:
                actual_close(descriptor)
            raise OSError("injected temporary descriptor close uncertainty")
        return actual_close(descriptor)

    module.tempfile = types.SimpleNamespace(mkstemp=mkstemp)
    module.os = types.SimpleNamespace(**vars(native_os))
    module.os.close = close
    with pytest.raises(ProfileRepositoryError):
        await repo.backup_to(root / "snapshot.sqlite")
    assert len(calls) == 1
    descriptor, name = allocated[0]
    assert os.path.exists(name), "uncertain descriptor cannot authorize unlink"
    if not after:
        assert os.fstat(descriptor).st_size == 0
    # Restore module functions only, never retry or close the uncertain descriptor.
    module.os = native_os
    repo._maintenance_close_admission()
    pause = storage._begin_local_pause()
    assert not await repo._maintenance_drain(time.monotonic() + 2)
    assert not pause.drain(time.monotonic() + 0.03)
    pause.resume()


@pytest.mark.parametrize("after", [False, True])
def test_backup_temporary_descriptor_uncertainty_keeps_namespace_and_drain_blocked(
    tmp_path, after
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_native_maintenance import _backup_descriptor_close_child
asyncio.run(_backup_descriptor_close_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _backup_fsync_child(root, role, after, body_failure=False):
    import os
    import time
    import types

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    native_os = module.os
    calls = []
    source_error = OSError("injected file fsync body failure")
    close_error = OSError("injected descriptor retirement failure")
    captured = []

    def selected(descriptor):
        for operation in repo._backup_native_operations:
            if operation.descriptors.get(role) == descriptor:
                return operation
        return None

    def close(descriptor):
        operation = selected(descriptor)
        if operation is not None:
            calls.append(descriptor)
            captured.append(operation)
            if after:
                native_os.close(descriptor)
            raise close_error
        native_os.close(descriptor)

    def fsync(descriptor):
        if body_failure and selected(descriptor) is not None:
            raise source_error
        native_os.fsync(descriptor)

    module.os = types.SimpleNamespace(**vars(native_os))
    module.os.close = close
    module.os.fsync = fsync
    destination = root / "snapshot.sqlite"
    with pytest.raises(ProfileRepositoryError) as caught:
        await repo.backup_to(destination)
    assert caught.value.code == "backup_failed"
    retained = captured[0]
    assert len(calls) == 1 and retained.uncertain
    assert retained.repository is repo and retained.generation == repo.generation
    assert retained.lease in storage._live_leases
    assert retained.receipt is not None
    assert retained.published is (role == "parent")
    assert destination.exists() is retained.published
    if body_failure:
        assert retained.body_error is source_error
        assert close_error in retained.cleanup_errors
    if not after:
        assert os.fstat(calls[0]).st_ino
    module.os = native_os
    await repo.close()
    assert retained in repo._backup_native_operations
    repo._maintenance_close_admission()
    pause = storage._begin_local_pause()
    assert not await repo._maintenance_drain(time.monotonic() + 1)
    assert not pause.drain(time.monotonic() + 0.03)
    pause.resume()


@pytest.mark.parametrize("role", ["file_sync", "parent"])
@pytest.mark.parametrize("after", [False, True])
def test_backup_fsync_descriptor_uncertainty_preserves_committed_result(
    tmp_path, role, after
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_native_maintenance import _backup_fsync_child
asyncio.run(_backup_fsync_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        role,
        str(after),
    )


def test_backup_preserves_both_fsync_body_and_close_errors(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_native_maintenance import _backup_fsync_child
asyncio.run(_backup_fsync_child(Path(sys.argv[1]), 'file_sync', False, True))
""",
    )


async def _backup_after_replace_child(root, failure):
    import types

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    native_os = module.os
    module.os = types.SimpleNamespace(**vars(native_os))

    def replace(source, destination):
        native_os.replace(source, destination)
        if failure == "rename":
            raise OSError("injected error after native rename")

    def fsync(descriptor):
        state = next(iter(repo._backup_native_operations))
        if state.published and descriptor == state.descriptors.get("parent"):
            raise OSError("injected published directory fsync failure")
        native_os.fsync(descriptor)

    module.os.replace = replace
    if failure == "fsync":
        module.os.fsync = fsync
    destination = root / "snapshot.sqlite"
    with pytest.raises(ProfileRepositoryError):
        await repo.backup_to(destination)
    state = next(iter(repo._backup_native_operations))
    assert state.publication_attempted and state.published and state.uncertain
    assert state.receipt.byte_count == destination.stat().st_size
    assert destination.read_bytes().startswith(b"SQLite format 3")
    assert not state.temporary_path.exists()
    module.os = native_os
    await repo.close()
    assert state in repo._backup_native_operations


@pytest.mark.parametrize("failure", ["rename", "fsync"])
def test_backup_published_destination_survives_post_publication_failure(
    tmp_path, failure
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_native_maintenance import _backup_after_replace_child
asyncio.run(_backup_after_replace_child(Path(sys.argv[1]), sys.argv[2]))
""",
        failure,
    )


@pytest.mark.asyncio
async def test_cancelled_running_and_queued_backups_keep_worker_native_ownership(
    tmp_path,
):
    import asyncio
    import threading

    import tldw_chatbook.TTS.profile_repository as module

    repo = module.TTSProfileRepository(tmp_path / "profiles.sqlite")
    await repo.open()
    actual_backup = repo._worker_online_backup
    entered, finish = threading.Event(), threading.Event()

    def backup(source, destination, **kwargs):
        actual_backup(source, destination, **kwargs)
        entered.set()
        assert finish.wait(5)

    repo._worker_online_backup = backup
    first = asyncio.create_task(repo.backup_to(tmp_path / "first.sqlite"))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        second = asyncio.create_task(repo.backup_to(tmp_path / "second.sqlite"))
        await asyncio.sleep(0)
        workers = tuple(repo._pending_futures)
        assert len(workers) == 2 and len(repo._backup_native_operations) == 1
        for task in (first, second):
            task.cancel()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert len(repo._backup_native_operations) == 1
        assert len(repo._pending_futures) == 2
        finish.set()
        receipts = await asyncio.gather(*(asyncio.wrap_future(f) for f in workers))
        assert all(receipt.byte_count > 0 for receipt in receipts)
        assert not repo._backup_native_operations
        assert not repo._pending_futures and not repo._publication_completions
        assert (tmp_path / "first.sqlite").exists()
        assert (tmp_path / "second.sqlite").exists()
    finally:
        finish.set()
        await repo.close()


@pytest.mark.asyncio
async def test_pause_refuses_delegated_sqlite_after_validation_file_effects(
    tmp_path, monkeypatch
):
    import asyncio
    import threading
    import time

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(tmp_path / "profiles.sqlite")
    await repo.open()
    import types
    import tldw_chatbook.TTS.profile_schema as schema

    effects = []
    actual_source_open = schema._open_candidate_source
    actual_tempfile = schema.tempfile

    def source_open(*args, **kwargs):
        result = actual_source_open(*args, **kwargs)
        effects.append(("source_fd", storage._pause is not None))
        return result

    def mkdtemp(*args, **kwargs):
        result = actual_tempfile.mkdtemp(*args, **kwargs)
        effects.append(("directory", storage._pause is not None))
        return result

    def mkstemp(*args, **kwargs):
        result = actual_tempfile.mkstemp(*args, **kwargs)
        effects.append(("snapshot_fd", storage._pause is not None))
        return result

    monkeypatch.setattr(schema, "_open_candidate_source", source_open)
    monkeypatch.setattr(
        schema, "tempfile", types.SimpleNamespace(**vars(actual_tempfile))
    )
    monkeypatch.setattr(schema.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(schema.tempfile, "mkstemp", mkstemp)
    actual_validation = repo._worker_validate_standalone_snapshot
    entered, finish = threading.Event(), threading.Event()

    def validate(path, **kwargs):
        entered.set()
        assert finish.wait(5)
        return actual_validation(path, **kwargs)

    repo._worker_validate_standalone_snapshot = validate
    pending = asyncio.create_task(repo.backup_to(tmp_path / "snapshot.sqlite"))
    pause = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        repo._maintenance_close_admission()
        pause = storage._begin_local_pause()
        assert not await repo._maintenance_drain(time.monotonic() + 0.03)
        finish.set()
        with pytest.raises(ProfileRepositoryError):
            await pending
        # This is eventual SQLite refusal, NOT admission before file effects.
        # The source FD and copy files are the mandatory next-phase native graph.
        assert {"source_fd", "directory", "snapshot_fd"} <= {
            role for role, paused in effects if paused
        }
        assert not (tmp_path / "snapshot.sqlite").exists()
        assert not tuple(tmp_path.glob(".snapshot.sqlite.*.backup"))
        assert await repo._maintenance_drain(time.monotonic() + 3)
    finally:
        finish.set()
        if pause is not None:
            pause.resume()
        await repo.close()


async def _backup_native_observer_child(root, failure):
    import asyncio
    import json
    import select
    import subprocess
    import sys
    import time

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    # Known minimal diagnostic process, not app startup retirement evidence.
    storage.admit_startup()
    if failure:
        await _backup_after_replace_child(root, "fsync")
        operation = next(
            state
            for state in storage._raw_operations
            if type(state) is module._BackupNativeState
        )
        assert operation.lease in storage._live_leases
        hold = storage._holds[operation.lease._key]
    else:
        repo = module.TTSProfileRepository(root / "profiles.sqlite")
        await repo.open()
        await repo.backup_to(root / "snapshot.sqlite")
        assert not repo._backup_native_operations
        await repo.close()
        hold = next(iter(storage._holds.values()))
    script = """
import json, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.admission import Admission
print('waiting', flush=True)
with Admission(Path(sys.argv[1])).maintenance(tuple(json.loads(sys.argv[2])), 8):
    print('entered', flush=True)
    input()
"""
    observer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(hold.authority.control_root),
            json.dumps(hold.names),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert observer.stdout.readline().strip() == "waiting"
        deadline = time.monotonic() + 3
        while not hold.authority.pause_requested(hold.names):
            assert time.monotonic() < deadline
            await asyncio.sleep(0.005)
        assert not select.select([observer.stdout], [], [], 0.03)[0]
        pause = storage._begin_local_pause()
        try:
            assert pause.drain(time.monotonic() + 0.03) is (not failure)
            if failure:
                assert operation.lease in storage._live_leases
                assert operation.descriptors["parent"] >= 0
                assert not select.select([observer.stdout], [], [], 0.03)[0]
            else:
                storage._shutdown()
                assert select.select([observer.stdout], [], [], 3)[0]
                assert observer.stdout.readline().strip() == "entered"
        finally:
            pause.resume()
    finally:
        if observer.poll() is None:
            if failure:
                observer.terminate()
            else:
                observer.stdin.write("\n")
                observer.stdin.flush()
        observer.wait(timeout=3)
        observer.stdin.close()
        observer.stdout.close()
        observer.stderr.close()


@pytest.mark.parametrize("failure", [False, True])
def test_independent_native_maintainer_observes_outer_backup_retirement(
    tmp_path, failure
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_native_maintenance import _backup_native_observer_child
asyncio.run(_backup_native_observer_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(failure),
    )


async def _backup_partial_allocation_child(root, role):
    import types

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    native_os = module.os
    captured = []

    def fstat(descriptor):
        for state in repo._backup_native_operations:
            if state.descriptors.get(role) == descriptor:
                captured.append((state, descriptor))
                raise OSError("injected failure immediately after allocation")
        return native_os.fstat(descriptor)

    module.os = types.SimpleNamespace(**vars(native_os))
    module.os.fstat = fstat
    with pytest.raises(ProfileRepositoryError):
        await repo.backup_to(root / "snapshot.sqlite")
    state, descriptor = captured[0]
    assert native_os.fstat(descriptor).st_ino
    assert state.descriptors[role] == descriptor
    assert state in repo._backup_native_operations and state.uncertain
    if role == "temporary":
        assert state.temporary_path.exists() and state.temporary_identity is None
    module.os = native_os
    await repo.close()
    assert state in repo._backup_native_operations


@pytest.mark.parametrize("role", ["parent", "temporary"])
def test_backup_retains_immediately_allocated_descriptor_before_identity_failure(
    tmp_path, role
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_native_maintenance import _backup_partial_allocation_child
asyncio.run(_backup_partial_allocation_child(Path(sys.argv[1]), sys.argv[2]))
""",
        role,
    )


async def _backup_remaining_sidecar_child(root):
    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    actual_connect = module.connect_private_sqlite
    sidecars = []

    def connect(owner, path, **kwargs):
        native = actual_connect(owner, path, **kwargs)
        if owner == "tts.profile_backup" and str(path).endswith(".backup"):
            actual_close = type(native).close

            def close(value):
                actual_close(value)
                if value is native:
                    sidecar = path.with_name(path.name + "-journal")
                    sidecar.write_bytes(b"foreign sidecar")
                    sidecars.append(sidecar)

            type(native).close = close
        return native

    module.connect_private_sqlite = connect
    with pytest.raises(ProfileRepositoryError):
        await repo.backup_to(root / "snapshot.sqlite")
    assert sidecars[0].read_bytes() == b"foreign sidecar"
    assert not (root / "snapshot.sqlite").exists()
    state = next(iter(repo._backup_native_operations))
    assert state.uncertain and state.connection is None
    await repo.close()
    assert sidecars[0].read_bytes() == b"foreign sidecar"


def test_backup_never_unlinks_unproven_sidecar_left_after_native_close(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_native_maintenance import _backup_remaining_sidecar_child
asyncio.run(_backup_remaining_sidecar_child(Path(sys.argv[1])))
""",
    )
