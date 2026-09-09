"""Exact current-store native retirement through real repository lifecycles."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


async def _close_child(root, role, after):
    import os
    import time
    import types

    import tldw_chatbook.TTS.profile_schema as schema
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    owned = repo._connection
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    hold = next(iter(storage._holds.values()))
    calls = []
    if role in ("live", "evidence"):
        selected = owned._connection if role == "live" else owned._evidence_connection
        actual = selected.close

        def close(value):
            if value is selected:
                calls.append(value)
                if len(calls) == 1:
                    if after:
                        actual()
                    raise OSError("exact SQLite close outcome unknown")
            return actual()

        class Delegated:
            def __getattr__(self, name):
                return getattr(selected, name)

            def close(self):
                return close(selected)

        if role == "live":
            owned._connection = Delegated()
        else:
            owned._evidence_connection = Delegated()
    else:
        selected = {
            "parent": owned.parent_fd,
            "main": owned.file_fd,
            "wal": owned.sidecar_fds["-wal"],
            "shm": owned.sidecar_fds["-shm"],
        }[role]
        actual = os.close
        schema.os = types.SimpleNamespace(**vars(os))

        def close(value):
            if value == selected:
                calls.append(value)
                if len(calls) == 1:
                    if after:
                        actual(value)
                    raise OSError("exact descriptor close outcome unknown")
            return actual(value)

        schema.os.close = close
    repo._maintenance_close_admission()
    assert not await repo._maintenance_drain(time.monotonic() + 3)
    assert calls == [selected]
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    storage._shutdown()
    assert _probe(hold.authority.control_root, hold.names) == "blocked", (
        "independent maintenance entered after native uncertainty"
    )
    with pytest.raises((ProfileRepositoryError, OSError)):
        await repo.close()
    assert calls == [selected], "definitive cleanup retried uncertain native close"
    assert repo._connection is owned


@pytest.mark.parametrize("role", ["live", "evidence", "parent", "main", "wal", "shm"])
@pytest.mark.parametrize("after", [False, True])
def test_exact_native_close_is_never_replayed(tmp_path, role, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _close_child
asyncio.run(_close_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        role,
        str(after),
    )


async def _lock_child(root, after):
    import time
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    calls = []

    class Delegated:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            return getattr(self.native, name)

        def close(self):
            calls.append("close")
            if after:
                self.native.close()
            raise OSError("real lock handle close outcome unknown")

    from pathlib import Path

    actual_open = Path.open

    def open(path, *args, **kwargs):
        native = actual_open(path, *args, **kwargs)
        return (
            Delegated(native) if str(path).endswith("profiles.sqlite.lock") else native
        )

    Path.open = open
    repo = TTSProfileRepository(root / "profiles.sqlite")
    # Initialize the current store before faulting the shared lifetime only.
    Path.open = actual_open
    await repo.open()
    await repo.close()
    Path.open = open
    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    hold = next(iter(storage._holds.values()))
    repo._maintenance_close_admission()
    assert not await repo._maintenance_drain(time.monotonic() + 3)
    assert calls == ["close"]
    storage._shutdown()
    assert _probe(hold.authority.control_root, hold.names) == "blocked", (
        "native admission lost profile lock uncertainty"
    )


@pytest.mark.parametrize("after", [False, True])
def test_profile_lock_native_uncertainty_blocks_independent_maintenance(
    tmp_path, after
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _lock_child
asyncio.run(_lock_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _blob_child(root, after, writing, unreturned=False, parent_after=None):
    import asyncio
    import sqlite3
    import time
    from Tests.TTS.test_profile_reference_repository import (
        _canonical,
        _draft,
        _requirements_for_tests,
    )
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    profile = await repo.create_profile(_draft("reference"))
    profile_id = profile.value.profile_id
    canonical = _canonical()
    if not writing:
        profile = await repo.set_reference(
            profile_id,
            canonical,
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=1,
            expected_generation=repo.generation,
        )
    conn = repo._connection._connection
    real_open = type(conn).blobopen
    allocated = []

    class Delegated:
        def __init__(self, blob):
            self.blob = blob

        def __getattr__(self, name):
            return getattr(self.blob, name)

        def __len__(self):
            return len(self.blob)

        def close(self):
            if after:
                self.blob.close()
            raise OSError("real reference blob close uncertainty")

    def blobopen(value, *args, **kwargs):
        blob = real_open(value, *args, **kwargs)
        allocated.append(blob)
        if unreturned:
            raise OSError("real reference blob allocated before return failure")
        return Delegated(blob)

    type(conn).blobopen = blobopen
    with pytest.raises(ProfileRepositoryError):
        if writing:
            await repo.set_reference(
                profile_id,
                canonical,
                _requirements_for_tests(model_id="tts-1"),
                expected_revision=1,
                expected_generation=repo.generation,
            )
        else:
            await repo.get_reference(
                profile_id,
                expected_revision=profile.value.revision,
                expected_generation=repo.generation,
            )
    assert len(allocated) == 1
    native = allocated[0]

    def observe():
        if after and not unreturned:
            with pytest.raises(sqlite3.ProgrammingError):
                len(native)
        else:
            assert len(native) == len(canonical.wav_bytes)

    await asyncio.wrap_future(repo._executor.submit(observe))
    if parent_after is not None:

        class Parent:
            def __getattr__(self, name):
                return getattr(conn, name)

            def close(self):
                if parent_after:
                    conn.close()
                raise OSError("parent native close outcome unknown")

        repo._connection._connection = Parent()
    repo._maintenance_close_admission()
    if parent_after is not None:
        from tldw_chatbook.Backup_Recovery import storage_admission as storage
        from Tests.DB.test_sqlite_source_pin_lifetime import _probe

        hold = next(iter(storage._holds.values()))
        assert not await repo._maintenance_drain(time.monotonic() + 3)
        if parent_after:

            def parent_retired_blob():
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    len(native)

            await asyncio.wrap_future(repo._executor.submit(parent_retired_blob))
        else:
            await asyncio.wrap_future(repo._executor.submit(observe))
        storage._shutdown()
        assert _probe(hold.authority.control_root, hold.names) == "blocked"
        return
    assert await repo._maintenance_drain(time.monotonic() + 3)

    def prove_retired():
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            len(native)

    await asyncio.wrap_future(repo._executor.submit(prove_retired))
    await repo._maintenance_resume()
    if not writing:
        type(conn).blobopen = real_open
        exact = await repo.get_reference(
            profile_id,
            expected_revision=profile.value.revision,
            expected_generation=repo.generation,
        )
        assert exact.value.wav_bytes == canonical.wav_bytes
    await repo.close()


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("writing", [False, True])
def test_reference_blob_native_retirement_requires_actual_parent_close(
    tmp_path, after, writing
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _blob_child
asyncio.run(_blob_child(Path(sys.argv[1]), sys.argv[2] == 'True', sys.argv[3] == 'True'))
""",
        str(after),
        str(writing),
    )


async def _partial_child(root, role):
    import asyncio
    import os
    import types
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    import tldw_chatbook.TTS.profile_schema as schema
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    # A retained observer leaves the exact WAL/SHM present for the current opener.
    held = repo
    selected = repo._active_database_path
    native_os = schema.private_paths.os
    allocated = []

    def open(*args, **kwargs):
        fd = native_os.open(*args, **kwargs)
        match = (
            str(args[0]).endswith("-wal")
            if role == "optional_wal"
            else str(args[0]) == os.sep
            if role == "parent"
            else str(args[0]) == selected.name
        )
        if match and not allocated:
            allocated.append(fd)
            raise FileNotFoundError(
                "native allocated before substituted provider refusal"
            )
        return fd

    original_native_open = schema.private_paths._native_open
    if role == "parent":

        def parent_open(*args, **kwargs):
            if kwargs.get("_outcome") is not None and not allocated:
                kwargs.pop("_outcome")
                fd = native_os.open(*args, **kwargs)
                allocated.append(fd)
                raise FileNotFoundError("unknown exact parent provider outcome")
            return original_native_open(*args, **kwargs)

        schema.private_paths._native_open = parent_open
    else:
        schema.private_paths.os = types.SimpleNamespace(**vars(native_os))
        schema.private_paths.os.open = open
        schema.os = schema.private_paths.os
    hold = next(iter(storage._holds.values()))
    with pytest.raises(ProfileRepositoryError):
        await asyncio.wrap_future(
            repo._executor.submit(schema.open_exact_current_profile_store, selected)
        )
    assert len(allocated) == 1 and os.fstat(allocated[0])
    schema.private_paths.os = native_os
    schema.private_paths._native_open = original_native_open
    schema.os = native_os
    await held.close()
    storage._shutdown()
    assert _probe(hold.authority.control_root, hold.names) == "blocked", (
        "unreturned allocation was erased by later native result"
    )


@pytest.mark.parametrize("role", ["main", "optional_wal", "parent"])
def test_exact_unreturned_allocation_survives_later_results(tmp_path, role):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _partial_child
asyncio.run(_partial_child(Path(sys.argv[1]), sys.argv[2]))
""",
        role,
    )


async def _standalone_namespace_child(root):
    import asyncio
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    owned = repo._connection
    # The same public wrapper.close is returned to standalone current callers.
    wal = owned.selected.with_name(owned.selected.name + "-wal")
    old = wal.with_name("displaced-wal")
    wal.rename(old)
    wal.write_bytes(b"foreign native namespace sentinel")
    with pytest.raises(ProfileRepositoryError):
        await asyncio.wrap_future(repo._executor.submit(owned.close))
    assert wal.read_bytes() == b"foreign native namespace sentinel"


def test_standalone_exact_close_preserves_replaced_namespace(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _standalone_namespace_child
asyncio.run(_standalone_namespace_child(Path(sys.argv[1])))
""",
    )


def _lock_outcome_child(root, failure):
    import os
    import types
    from pathlib import Path
    import tldw_chatbook.TTS.profile_store_lock as module
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    lease = module.ProfileStoreLease(
        root / "profiles.sqlite", module.ProfileStoreLockMode.EXCLUSIVE
    )
    actual_open = Path.open
    allocated = []
    pause = None
    if failure == "admission_refusal":
        actual_admit = storage.acquire_storage
        count = 0
        admitted_hold = None

        def admit(*args, **kwargs):
            nonlocal count, pause, admitted_hold
            count += 1
            if count == 2:
                pause = storage._begin_local_pause()
            result = actual_admit(*args, **kwargs)
            admitted_hold = next(iter(storage._holds.values()))
            return result

        storage.acquire_storage = admit
        with pytest.raises(Exception):
            lease.acquire()
        assert count == 2 and not lease.lock_path.exists()
        assert not lease._native_outcomes
        pause.resume()
        storage._shutdown()
        assert (
            _probe(admitted_hold.authority.control_root, admitted_hold.names)
            == "entered"
        )
        return
    if failure == "open_unreturned":

        def open(path, *args, **kwargs):
            native = actual_open(path, *args, **kwargs)
            if path == lease.lock_path:
                allocated.append(native)
                raise OSError(
                    "native lock file allocated before provider return failure"
                )
            return native

        Path.open = open
        with pytest.raises(ProfileRepositoryError):
            lease.acquire()
        assert len(allocated) == 1 and os.fstat(allocated[0].fileno())
        Path.open = actual_open
    else:
        lease.acquire()
        positive_hold = next(iter(storage._holds.values()))
        native = lease._handle
        descriptor = native.fileno()
        actual_unlock = module.portalocker.unlock
        module.portalocker = types.SimpleNamespace(**vars(module.portalocker))

        def unlock(handle):
            if failure == "unlock_after":
                actual_unlock(handle)
            raise OSError("ordinary unlock error")

        module.portalocker.unlock = unlock
        with pytest.raises(ProfileRepositoryError):
            lease.release()
        with pytest.raises(OSError):
            os.fstat(descriptor)
    hold = next(iter(storage._holds.values())) if storage._holds else None
    if failure == "open_unreturned":
        assert hold is not None
        # Later ordinary success never erases the previous unknown allocation.
        lease.acquire()
        lease.release()
        storage._shutdown()
        assert _probe(hold.authority.control_root, hold.names) == "blocked"
    else:
        assert not lease._native_outcomes
        contender = module.ProfileStoreLease(
            root / "profiles.sqlite", module.ProfileStoreLockMode.EXCLUSIVE
        )
        contender.acquire()
        module.portalocker.unlock = actual_unlock
        contender.release()
        storage._shutdown()
        assert (
            _probe(positive_hold.authority.control_root, positive_hold.names)
            == "entered"
        )


@pytest.mark.parametrize(
    "failure", ["admission_refusal", "open_unreturned", "unlock_before", "unlock_after"]
)
def test_profile_lock_original_outcome_distinctions(tmp_path, failure):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _lock_outcome_child
_lock_outcome_child(Path(sys.argv[1]), sys.argv[2])
""",
        failure,
    )


@pytest.mark.parametrize("writing", [False, True])
def test_reference_blob_unreturned_handle_is_retired_by_original_parent(
    tmp_path, writing
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _blob_child
asyncio.run(_blob_child(Path(sys.argv[1]), False, sys.argv[2] == 'True', True))
""",
        str(writing),
    )


async def _connect_refusal_child(root):
    import asyncio
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    import tldw_chatbook.TTS.profile_schema as schema
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    before = set(storage._live_leases)
    actual_connect = schema.connect_private_sqlite
    pause = None

    def connect(owner, *args, **kwargs):
        nonlocal pause
        if owner == "tts.profile_store":
            pause = storage._begin_local_pause()
        return actual_connect(owner, *args, **kwargs)

    schema.connect_private_sqlite = connect
    with pytest.raises(ProfileRepositoryError):
        await asyncio.wrap_future(
            repo._executor.submit(
                schema.open_exact_current_profile_store, repo._active_database_path
            )
        )
    await asyncio.wrap_future(repo._executor.submit(pause.resume))
    assert set(storage._live_leases) == before, (
        "original pre-resource SQLite refusal leaked exact owner admission"
    )
    schema.connect_private_sqlite = actual_connect
    await repo.close()


def test_exact_original_connect_refusal_retires_partial_owner(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _connect_refusal_child
asyncio.run(_connect_refusal_child(Path(sys.argv[1])))
""",
    )


@pytest.mark.parametrize("parent_after", [False, True])
@pytest.mark.parametrize("writing", [False, True])
def test_reference_blob_failed_parent_keeps_native_exclusion(
    tmp_path, parent_after, writing
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _blob_child
asyncio.run(_blob_child(Path(sys.argv[1]), False, sys.argv[3] == 'True', True, sys.argv[2] == 'True'))
""",
        str(parent_after),
        str(writing),
    )


def _lock_control_child(root):
    import types
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    import tldw_chatbook.TTS.profile_store_lock as module
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    class Signal(BaseException):
        pass

    signal = Signal("original control flow")
    lease = module.ProfileStoreLease(
        root / "profiles.sqlite", module.ProfileStoreLockMode.EXCLUSIVE
    )
    lease.acquire()
    hold = next(iter(storage._holds.values()))
    actual_close = storage.StorageLease.close
    selected = lease._native_outcomes[0].leases[-1]

    def close(native):
        actual_close(native)
        if native is selected:
            raise OSError("ordinary lease close uncertainty after unlock control")

    storage.StorageLease.close = close
    module.portalocker = types.SimpleNamespace(**vars(module.portalocker))

    def unlock(native):
        raise signal

    module.portalocker.unlock = unlock
    with pytest.raises(Signal) as caught:
        lease.release()
    assert caught.value is signal
    storage._shutdown()
    assert _probe(hold.authority.control_root, hold.names) == "blocked"


def test_profile_lock_native_release_preserves_original_control_flow(tmp_path):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _lock_control_child
_lock_control_child(Path(sys.argv[1]))
""",
    )


async def _candidate_blob_child(root, after):
    import sqlite3
    from Tests.TTS.test_profile_reference_repository import (
        _canonical,
        _draft,
        _requirements_for_tests,
    )
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
    import tldw_chatbook.TTS.profile_repository as schema

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    profile = await repo.create_profile(_draft("candidate reference"))
    canonical = _canonical()
    await repo.set_reference(
        profile.value.profile_id,
        canonical,
        _requirements_for_tests(model_id="tts-1"),
        expected_revision=1,
        expected_generation=repo.generation,
    )
    snapshot = root / "snapshot.sqlite"
    await repo.backup_to(snapshot)
    await repo.close()
    before = snapshot.read_bytes()
    actual_connect = schema.connect_private_sqlite
    blobs = []

    class Blob:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            return getattr(self.native, name)

        def __len__(self):
            return len(self.native)

        def close(self):
            if after:
                self.native.close()
            raise OSError("real standalone validation BLOB close uncertainty")

    def connect(owner, *args, **kwargs):
        native = actual_connect(owner, *args, **kwargs)
        if owner == "tts.profile_snapshot":
            actual_blobopen = type(native).blobopen

            def blobopen(connection, *args, **kwargs):
                value = actual_blobopen(connection, *args, **kwargs)
                blobs.append(value)
                return Blob(value)

            type(native).blobopen = blobopen
        return native

    schema.connect_private_sqlite = connect
    with pytest.raises(ProfileRepositoryError):
        repo._worker_validate_standalone_snapshot(snapshot)
    assert blobs
    for blob in blobs:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            len(blob)
    assert snapshot.read_bytes() == before


@pytest.mark.parametrize("after", [False, True])
def test_standalone_candidate_original_parent_retires_real_blob(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _candidate_blob_child
asyncio.run(_candidate_blob_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _blob_cancel_child(root, writing):
    import asyncio
    import sqlite3
    import threading
    import time
    from Tests.TTS.test_profile_reference_repository import (
        _canonical,
        _draft,
        _requirements_for_tests,
    )
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    profile = await repo.create_profile(_draft("cancelled reference"))
    canonical = _canonical()
    if not writing:
        profile = await repo.set_reference(
            profile.value.profile_id,
            canonical,
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=1,
            expected_generation=repo.generation,
        )
    conn = repo._connection._connection
    original = type(conn).blobopen
    entered, finish = threading.Event(), threading.Event()
    blobs = []

    def blobopen(value, *args, **kwargs):
        blob = original(value, *args, **kwargs)
        blobs.append(blob)
        entered.set()
        assert finish.wait(4)
        return blob

    type(conn).blobopen = blobopen
    if writing:
        operation = repo.set_reference(
            profile.value.profile_id,
            canonical,
            _requirements_for_tests(model_id="tts-1"),
            expected_revision=1,
            expected_generation=repo.generation,
        )
    else:
        operation = repo.get_reference(
            profile.value.profile_id,
            expected_revision=profile.value.revision,
            expected_generation=repo.generation,
        )
    caller = asyncio.create_task(operation)
    assert await asyncio.to_thread(entered.wait, 3)
    queued = asyncio.create_task(repo.get_profile(profile.value.profile_id))
    await asyncio.sleep(0)
    caller.cancel()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    repo._maintenance_close_admission()
    assert not await repo._maintenance_drain(time.monotonic() + 0.02)
    finish.set()
    await queued
    assert await repo._maintenance_drain(time.monotonic() + 3)

    def observe():
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            len(blobs[0])

    await asyncio.wrap_future(repo._executor.submit(observe))
    type(conn).blobopen = original
    await repo._maintenance_resume()
    exact = await repo.get_reference(
        profile.value.profile_id,
        expected_revision=2,
        expected_generation=repo.generation,
    )
    assert exact.value.wav_bytes == canonical.wav_bytes
    await repo.close()


@pytest.mark.parametrize("writing", [False, True])
def test_cancelled_real_blob_and_queued_operation_retire_on_source_worker(
    tmp_path, writing
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _blob_cancel_child
asyncio.run(_blob_cancel_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(writing),
    )


async def _exact_lease_child(root, after):
    import time
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    owned = repo._connection
    selected = owned.leases[-1]
    hold = storage._holds[selected._key]
    original = storage.StorageLease.close
    calls = []

    def close(native):
        if native is selected:
            calls.append(native)
            if after:
                original(native)
            raise OSError("exact wrapper aggregate release unknown")
        return original(native)

    storage.StorageLease.close = close
    repo._maintenance_close_admission()
    assert not await repo._maintenance_drain(time.monotonic() + 3)
    assert calls == [selected]
    storage._shutdown()
    assert _probe(hold.authority.control_root, hold.names) == "blocked"


@pytest.mark.parametrize("after", [False, True])
def test_exact_aggregate_release_keeps_independent_native_hold(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_exact_native_maintenance import _exact_lease_child
asyncio.run(_exact_lease_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )
