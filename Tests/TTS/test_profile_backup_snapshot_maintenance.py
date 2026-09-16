"""Outer snapshot ownership through real backup and recovery backup callers."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


async def _snapshot_child(root, after, recovery, cancelled=False):
    import asyncio
    import sqlite3
    import time

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    candidate = root / "candidate.sqlite"
    if recovery:
        await repo.backup_to(candidate)
    actual_connect = module.connect_private_sqlite
    allocated = []
    calls = []

    def connect(owner, path, **kwargs):
        native = actual_connect(owner, path, **kwargs)
        selected = str(path).endswith(".recovery.sqlite3" if recovery else ".backup")
        if owner == "tts.profile_snapshot" and selected and not allocated:
            allocated.append((native, path))
            actual_close = type(native).close

            def close(value):
                calls.append(value)
                if after:
                    actual_close(value)
                raise OSError("injected outer snapshot reader close uncertainty")

            type(native).close = close
        return native

    module.connect_private_sqlite = connect

    async def run():
        if recovery:
            return await repo.restore_from(candidate)
        return await repo.backup_to(root / "snapshot.sqlite")

    if cancelled:
        import threading

        entered, finish = threading.Event(), threading.Event()
        validation = repo._worker_validate_standalone_snapshot

        def blocked(*args, **kwargs):
            entered.set()
            assert finish.wait(4)
            return validation(*args, **kwargs)

        repo._worker_validate_standalone_snapshot = blocked
        pending = asyncio.create_task(run())
        assert await asyncio.to_thread(entered.wait, 3)
        pending.cancel()
        pending.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        # Backup callers may return cancellation before the actual worker. Join
        # that represented work on its owner executor before inspecting results.
        await asyncio.wrap_future(repo._executor.submit(lambda: None))
        await asyncio.sleep(0)
    else:
        with pytest.raises(ProfileRepositoryError):
            await run()
    assert len(allocated) == len(calls) == 1
    native, pathname = allocated[0]
    assert pathname.exists(), (
        "outer cleanup unlinked the uncertain native reader's file"
    )
    retained = next(iter(repo._backup_native_operations))
    assert retained.uncertain and retained.lease in storage._live_leases

    def observe():
        if after:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                native.execute("SELECT 1")
        else:
            assert native.execute("SELECT 1").fetchone()[0] == 1

    await asyncio.wrap_future(repo._executor.submit(observe))
    await repo.backup_to(root / "later.sqlite")
    assert repo._backup_native_operations == {retained}
    await repo.close()
    pause = storage._begin_local_pause()
    assert not pause.drain(time.monotonic() + 0.03)
    pause.resume()
    assert len(calls) == 1
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    hold = storage._holds[retained.lease._key]
    # Retire only this known private child's startup fixture. Actual uncertain
    # resources stay untouched; this is not installed app startup qualification.
    storage._shutdown()
    assert _probe(hold.authority.control_root, hold.names) == "blocked"


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("recovery", [False, True])
def test_snapshot_close_uncertainty_retains_outer_path(tmp_path, after, recovery):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_snapshot_maintenance import _snapshot_child
asyncio.run(_snapshot_child(Path(sys.argv[1]), sys.argv[2] == 'True', sys.argv[3] == 'True'))
""",
        str(after),
        str(recovery),
    )


async def _unreturned_child(root, role, recovery):
    import os
    import types

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    candidate = root / "candidate.sqlite"
    if recovery:
        await repo.backup_to(candidate)
    actual_os = module.os
    actual_tempfile = module.tempfile
    allocated = []
    module.os = types.SimpleNamespace(**vars(actual_os))

    def open(path, flags, *args, **kwargs):
        descriptor = actual_os.open(path, flags, *args, **kwargs)
        matches = (
            path == root
            if role == "parent"
            else str(path).endswith(".recovery.sqlite3" if recovery else ".backup")
        )
        if matches and not allocated:
            allocated.append((descriptor, path))
            raise OSError("allocated real outer descriptor before wrapper error")
        return descriptor

    def mkstemp(**kwargs):
        descriptor, path = actual_tempfile.mkstemp(**kwargs)
        allocated.append((descriptor, path))
        raise OSError("allocated real outer temporary before wrapper error")

    if role == "temporary":
        module.tempfile = types.SimpleNamespace(mkstemp=mkstemp)
    else:
        module.os.open = open
    with pytest.raises(ProfileRepositoryError):
        if recovery:
            await repo.restore_from(candidate)
        else:
            await repo.backup_to(root / "snapshot.sqlite")
    assert len(allocated) == 1
    assert os.fstat(allocated[0][0])
    assert repo._backup_native_operations, "unreturned allocation lost outer ownership"
    retained = next(iter(repo._backup_native_operations))
    assert retained.uncertain and retained.lease in storage._live_leases
    module.os = actual_os
    module.tempfile = actual_tempfile
    await repo.backup_to(root / "later.sqlite")
    assert repo._backup_native_operations == {retained}
    await repo.close()
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    hold = storage._holds[retained.lease._key]
    storage._shutdown()
    assert _probe(hold.authority.control_root, hold.names) == "blocked"


@pytest.mark.parametrize("role", ["parent", "temporary", "file_sync"])
@pytest.mark.parametrize("recovery", [False, True])
def test_outer_unreturned_allocation_retains_job(tmp_path, role, recovery):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_snapshot_maintenance import _unreturned_child
asyncio.run(_unreturned_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "True"))
""",
        role,
        str(recovery),
    )


async def _journal_child(root, substitute, recovery):
    import os
    from pathlib import Path

    import tldw_chatbook.DB.private_sqlite as private
    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    candidate = root / "candidate.sqlite"
    if recovery:
        await repo.backup_to(candidate)
    actual_pages = private._backup_pages
    observed = []

    def backup(source, destination, **kwargs):
        actual_pages(source, destination, **kwargs)
        temporary = Path(destination.execute("PRAGMA database_list").fetchone()[2])
        if not str(temporary).endswith(".recovery.sqlite3" if recovery else ".backup"):
            return
        journal = temporary.with_name(temporary.name + "-journal")
        assert journal.is_file(), "real native backup must expose the qualified journal"
        before = journal.stat()
        observed.append((journal, before.st_dev, before.st_ino))
        if substitute:
            journal.rename(root / "original-journal")
            journal.write_bytes(b"foreign journal sentinel")
            os.chmod(journal, 0o600)

    private._backup_pages = backup

    async def run():
        if recovery:
            return await repo.restore_from(candidate)
        return await repo.backup_to(root / "snapshot.sqlite")

    if substitute:
        with pytest.raises(ProfileRepositoryError):
            await run()
        assert observed[0][0].read_bytes() == b"foreign journal sentinel"
        assert repo._backup_native_operations
    else:
        await run()
        assert len(observed) == 1 and not observed[0][0].exists()
        assert not repo._backup_native_operations
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    hold = next(iter(storage._holds.values()))
    await repo.close()
    storage._shutdown()  # Known child startup only; no production startup release.
    assert _probe(hold.authority.control_root, hold.names) == (
        "blocked" if substitute else "entered"
    )


@pytest.mark.parametrize("substitute", [False, True])
@pytest.mark.parametrize("recovery", [False, True])
def test_real_native_destination_journal_retirement_and_substitution(
    tmp_path, substitute, recovery
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_snapshot_maintenance import _journal_child
asyncio.run(_journal_child(Path(sys.argv[1]), sys.argv[2] == 'True', sys.argv[3] == 'True'))
""",
        str(substitute),
        str(recovery),
    )


async def _recovery_destination_child(root, after):
    import tldw_chatbook.DB.private_sqlite as private
    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    candidate = root / "candidate.sqlite"
    await repo.backup_to(candidate)
    actual_connect = private._connect_registered_sqlite
    allocated = []
    calls = []

    def connect(owner, path, **kwargs):
        native = actual_connect(owner, path, **kwargs)
        if owner == "tts.profile_recovery" and not allocated:
            allocated.append((native, path))
            original_close = type(native).close

            def close(value):
                calls.append(value)
                if after:
                    original_close(value)
                raise OSError("injected delegated recovery destination close")

            type(native).close = close
        return native

    private._connect_registered_sqlite = connect
    with pytest.raises(ProfileRepositoryError):
        await repo.restore_from(candidate)
    assert len(allocated) == len(calls) == 1
    assert allocated[0][1].exists()
    assert repo._backup_native_operations


@pytest.mark.parametrize("after", [False, True])
def test_recovery_delegated_destination_close_is_not_success(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_snapshot_maintenance import _recovery_destination_child
asyncio.run(_recovery_destination_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _candidate_source_child(root, after, recovery):
    import os

    import tldw_chatbook.TTS.profile_repository as module
    import tldw_chatbook.TTS.profile_schema as schema
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    candidate = root / "candidate.sqlite"
    if recovery:
        await repo.backup_to(candidate)
    actual_open = schema._open_candidate_source
    actual_close = schema._close_candidate_fd
    allocated, calls = [], []

    def open(path, flags):
        fd = actual_open(path, flags)
        if str(path).endswith(".recovery.sqlite3" if recovery else ".backup"):
            allocated.append((fd, path))
        return fd

    def close(fd):
        if allocated and fd == allocated[0][0]:
            calls.append(fd)
            if after:
                actual_close(fd)
            raise OSError("candidate source descriptor close uncertainty")
        return actual_close(fd)

    schema._open_candidate_source = open
    schema._close_candidate_fd = close
    with pytest.raises(ProfileRepositoryError):
        if recovery:
            await repo.restore_from(candidate)
        else:
            await repo.backup_to(root / "snapshot.sqlite")
    assert len(calls) == 1
    if not after:
        assert os.fstat(allocated[0][0])
    assert allocated[0][1].exists(), "outer removed uncertain candidate source pathname"
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    retained = next(iter(repo._backup_native_operations))
    assert retained.candidate_job.uncertain
    assert retained.uncertain and retained.lease in storage._live_leases
    await repo.close()
    assert len(calls) == 1
    hold = storage._holds[retained.lease._key]
    # Retire only the known private-child startup fixture, leaving the actual
    # uncertain candidate and outer ownership untouched for the native observer.
    storage._shutdown()
    assert _probe(hold.authority.control_root, hold.names) == "blocked"


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("recovery", [False, True])
def test_candidate_source_uncertainty_retains_outer_path(tmp_path, after, recovery):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_snapshot_maintenance import _candidate_source_child
asyncio.run(_candidate_source_child(Path(sys.argv[1]), sys.argv[2] == 'True', sys.argv[3] == 'True'))
""",
        str(after),
        str(recovery),
    )


async def _callback_child(root, recovery):
    import tldw_chatbook.TTS.profile_repository as module

    class StopBackup(BaseException):
        pass

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    candidate = root / "candidate.sqlite"
    if recovery:
        await repo.backup_to(candidate)
    original_clock = module._monotonic
    signal = StopBackup("first actual callback control-flow error")
    changed = []

    def clock():
        for owner in repo._backup_native_operations:
            if owner.journal_identity is not None and not changed:
                journal = owner.temporary_path.with_name(
                    owner.temporary_path.name + "-journal"
                )
                journal.rename(root / "callback-original-journal")
                journal.write_bytes(b"foreign callback journal")
                journal.chmod(0o600)
                changed.append(journal)
                raise signal
        return original_clock()

    module._monotonic = clock
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    # Existing migration quarantine can supersede public control-flow redelivery;
    # the concrete backup still retains the original signal and cleanup evidence.
    expected_error = ProfileRepositoryError if recovery else StopBackup
    with pytest.raises(expected_error) as caught:
        if recovery:
            await repo.restore_from(candidate)
        else:
            await repo.backup_to(root / "snapshot.sqlite")
    if recovery:
        assert caught.value.code == "unavailable"
    else:
        assert caught.value is signal
    assert changed[0].read_bytes() == b"foreign callback journal"
    retained = next(iter(repo._backup_native_operations))
    assert retained.uncertain and retained.body_error is signal
    assert retained.cleanup_errors
    module._monotonic = original_clock
    await repo.close()


@pytest.mark.parametrize("recovery", [False, True])
def test_callback_error_precedes_observed_journal_substitution_error(
    tmp_path, recovery
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_snapshot_maintenance import _callback_child
asyncio.run(_callback_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(recovery),
    )


async def _recovery_fd_child(root, role, after):
    import os
    import types

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    candidate = root / "candidate.sqlite"
    await repo.backup_to(candidate)
    actual_os = module.os
    calls = []
    observed = []

    def close(fd):
        for owner in repo._backup_native_operations:
            if owner.descriptors.get(role) == fd:
                calls.append(fd)
                observed.append(owner)
                if after:
                    actual_os.close(fd)
                    with pytest.raises(OSError):
                        actual_os.fstat(fd)
                raise OSError("injected real recovery descriptor close uncertainty")
        actual_os.close(fd)

    module.os = types.SimpleNamespace(**vars(actual_os))
    module.os.close = close
    with pytest.raises(ProfileRepositoryError):
        await repo.restore_from(candidate)
    assert len(calls) == 1
    owner = observed[0]
    assert owner in repo._backup_native_operations and owner.uncertain
    assert owner.temporary_path.exists()
    if not after:
        assert os.fstat(calls[0])
    assert owner.published is (role == "parent")
    module.os = actual_os
    await repo.close()
    storage._shutdown()
    hold = storage._holds[owner.lease._key]
    assert _probe(hold.authority.control_root, hold.names) == "blocked"
    assert len(calls) == 1


@pytest.mark.parametrize("role", ["temporary", "file_sync", "parent"])
@pytest.mark.parametrize("after", [False, True])
def test_recovery_descriptor_close_uncertainty_keeps_native_boundary(
    tmp_path, role, after
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_snapshot_maintenance import _recovery_fd_child
asyncio.run(_recovery_fd_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        role,
        str(after),
    )


async def _nested_refusal_child(root, role, recovery):
    import time

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB import private_sqlite as private
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    candidate = root / "candidate.sqlite"
    if recovery:
        await repo.backup_to(candidate)
    connect = module.connect_private_sqlite
    attempted = []
    native_before = set(private._ordinary_connections)

    def paused_connect(owner, path, **kwargs):
        if owner == (
            "tts.profile_snapshot"
            if role == "snapshot"
            else ("tts.profile_recovery" if recovery else "tts.profile_backup")
        ):
            pause = storage._begin_local_pause()
            try:
                attempted.append(path)
                return connect(owner, path, **kwargs)
            finally:
                pause.resume()
        return connect(owner, path, **kwargs)

    module.connect_private_sqlite = paused_connect
    with pytest.raises(ProfileRepositoryError):
        if recovery:
            await repo.restore_from(candidate)
        else:
            await repo.backup_to(root / "snapshot.sqlite")
    assert len(attempted) == 1
    if not recovery:
        assert set(private._ordinary_connections) == native_before
    assert not any(
        lease.resource_path == attempted[0]
        for lease in private._ordinary_connections.values()
    )
    assert not repo._backup_native_operations, (
        "known nested admission refusal became unresolved allocation"
    )
    assert not tuple(root.glob(".snapshot.sqlite.*.backup"))
    await repo.close()
    pause = storage._begin_local_pause()
    assert pause.drain(time.monotonic() + 0.03)
    pause.resume()


@pytest.mark.parametrize("role", ["destination", "snapshot"])
@pytest.mark.parametrize("recovery", [False, True])
def test_known_nested_connect_refusal_retires_outer_operation(tmp_path, role, recovery):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_snapshot_maintenance import _nested_refusal_child
asyncio.run(_nested_refusal_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "True"))
""",
        role,
        str(recovery),
    )


async def _deadline_child(root, recovery):
    import time

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    candidate = root / "candidate.sqlite"
    if recovery:
        await repo.backup_to(candidate)
    original = module._monotonic
    fired = []

    def clock():
        for owner in repo._backup_native_operations:
            if owner.journal_identity is not None and not fired:
                fired.append(owner)
                return original() + 1000
        return original()

    module._monotonic = clock
    with pytest.raises(ProfileRepositoryError):
        if recovery:
            await repo.restore_from(candidate)
        else:
            await repo.backup_to(root / "snapshot.sqlite")
    assert len(fired) == 1
    assert not repo._backup_native_operations, (
        "ordinary native backup timeout failed to retire"
    )
    assert not tuple(root.glob("*.recovery.sqlite3"))
    assert not tuple(root.glob(".snapshot.sqlite.*.backup"))
    module._monotonic = original
    await repo.close()
    pause = storage._begin_local_pause()
    assert pause.drain(time.monotonic() + 0.03)
    pause.resume()


@pytest.mark.parametrize("recovery", [False, True])
def test_actual_backup_progress_timeout_positively_retires_outer_native(
    tmp_path, recovery
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_snapshot_maintenance import _deadline_child
asyncio.run(_deadline_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(recovery),
    )


@pytest.mark.parametrize("recovery", [False, True])
def test_repeated_cancellation_keeps_actual_snapshot_native_and_publication_work(
    tmp_path, recovery
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_snapshot_maintenance import _snapshot_child
asyncio.run(_snapshot_child(Path(sys.argv[1]), False, sys.argv[2] == 'True', cancelled=True))
""",
        str(recovery),
    )


async def _unknown_connect_child(root, recovery, snapshot):
    import asyncio

    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    candidate = root / "candidate.sqlite"
    if recovery:
        await repo.backup_to(candidate)
    connect = module.connect_private_sqlite
    allocated = []
    selected_owner = (
        "tts.profile_snapshot"
        if snapshot
        else ("tts.profile_recovery" if recovery else "tts.profile_backup")
    )

    def failing_connect(owner, path, **kwargs):
        native = connect(owner, path, **kwargs)
        if owner == selected_owner and not allocated:
            allocated.append((native, path))
            # Same class/code as a genuine refusal, after an actual allocation.
            raise RecoveryRequired("storage_locally_paused")
        return native

    module.connect_private_sqlite = failing_connect
    with pytest.raises(ProfileRepositoryError):
        if recovery:
            await repo.restore_from(candidate)
        else:
            await repo.backup_to(root / "snapshot.sqlite")
    assert len(allocated) == 1 and allocated[0][1].exists()
    assert repo._backup_native_operations
    retained = next(iter(repo._backup_native_operations))
    assert retained.uncertain
    await asyncio.wrap_future(
        repo._executor.submit(lambda: allocated[0][0].execute("SELECT 1").fetchone())
    )
    await repo.backup_to(root / "later.sqlite")
    assert repo._backup_native_operations == {retained}
    await repo.close()
    storage._shutdown()
    hold = storage._holds[retained.lease._key]
    assert _probe(hold.authority.control_root, hold.names) == "blocked"


@pytest.mark.parametrize(
    "recovery,snapshot", [(False, False), (False, True), (True, False), (True, True)]
)
def test_unreturned_native_connect_cannot_be_cleared_by_refusal_error_type(
    tmp_path, recovery, snapshot
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_snapshot_maintenance import _unknown_connect_child
asyncio.run(_unknown_connect_child(Path(sys.argv[1]), sys.argv[2] == 'True', sys.argv[3] == 'True'))
""",
        str(recovery),
        str(snapshot),
    )


async def _allocation_lease_child(root):
    import tldw_chatbook.TTS.profile_repository as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    repo = module.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    original_close = storage.StorageLease.close
    selected = []

    def close(lease):
        for owner in repo._backup_native_operations:
            if (
                owner.allocation_leases
                and lease is owner.allocation_leases[0]
                and not selected
            ):
                selected.append(owner)
                original_close(lease)
                raise OSError("retired allocation lease then wrapper error")
        original_close(lease)

    storage.StorageLease.close = close
    with pytest.raises(ProfileRepositoryError):
        await repo.backup_to(root / "snapshot.sqlite")
    owner = selected[0]
    hold = next(iter(storage._holds.values()))
    await repo.close()
    storage._shutdown()
    assert _probe(hold.authority.control_root, hold.names) == "blocked", (
        "outer lease was retired after a subordinate lease's uncertain close"
    )
    assert owner.lease in storage._live_leases


def test_allocation_lease_close_error_preserves_independent_outer_native_hold(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_backup_snapshot_maintenance import _allocation_lease_child
asyncio.run(_allocation_lease_child(Path(sys.argv[1])))
""",
    )


@pytest.mark.parametrize("after", [False, True])
def test_standalone_candidate_lease_uncertainty_keeps_independent_native_hold(
    tmp_path, after
):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
import pytest
import tldw_chatbook.TTS.profile_schema as schema
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from Tests.DB.test_sqlite_source_pin_lifetime import _probe
root = Path(sys.argv[1])
after = sys.argv[2] == 'True'
source = root / 'candidate.sqlite'
connection = schema.open_profile_store(source)
connection.close()
original_bytes = source.read_bytes()
original_close = storage.StorageLease.close
selected, calls = [], []
def close(lease):
    for job in tuple(storage._raw_operations):
        if isinstance(job, schema._CandidateValidationJob) and job.leases:
            if lease is job.leases[0] and not selected:
                hold = storage._holds[lease._key]
                selected.append((job, hold))
                calls.append(lease)
                if after:
                    original_close(lease)
                raise OSError('candidate allocation lease wrapper error')
    original_close(lease)
storage.StorageLease.close = close
with pytest.raises(ProfileRepositoryError):
    schema.validate_profile_candidate(source)
job, hold = selected[0]
assert job.uncertain and job in storage._raw_operations
assert len(calls) == 1 and source.read_bytes() == original_bytes
storage._shutdown()  # Only this private child's known startup fixture.
assert _probe(hold.authority.control_root, hold.names) == 'blocked', 'standalone candidate retired remaining holds after uncertain lease close'
assert any(lease in storage._live_leases for lease in job.leases)
schema.validate_profile_candidate(source)
assert job in storage._raw_operations and len(calls) == 1
assert _probe(hold.authority.control_root, hold.names) == 'blocked'
""",
        str(after),
    )
