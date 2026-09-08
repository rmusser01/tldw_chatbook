"""Candidate validation owns real file/native cleanup until positive retirement."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


def _candidate_child(root, mode, after=False):
    import os
    import time
    import types
    from pathlib import Path

    import tldw_chatbook.TTS.profile_schema as schema
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    source = root / "candidate.sqlite"
    connection = schema.open_profile_store(source)
    connection.close()
    actual_tempfile = schema.tempfile
    schema.tempfile = types.SimpleNamespace(**vars(actual_tempfile))
    effects = []
    source_fds = []
    snapshots = []
    actual_open = schema._open_candidate_source
    actual_close = schema._close_candidate_fd
    calls = []
    native_connections = []
    actual_connect = schema.connect_private_sqlite

    def connect(owner, *args, **kwargs):
        native = actual_connect(owner, *args, **kwargs)
        if (mode == "connection" and owner == "tts.profile_candidate_upgrade") or (
            mode == "read_connection" and owner == "tts.profile_candidate"
        ):
            native_connections.append(native)
            native_close = type(native).close

            def uncertain_close(value):
                if value is native:
                    calls.append(value)
                    if after:
                        native_close(value)
                    raise OSError("candidate connection close uncertainty")
                native_close(value)

            type(native).close = uncertain_close
        return native

    schema.connect_private_sqlite = connect

    def source_open(*args):
        fd = actual_open(*args)
        source_fds.append(fd)
        effects.append("source")
        return fd

    def mkdtemp(*args, **kwargs):
        kwargs["dir"] = root
        value = actual_tempfile.mkdtemp(*args, **kwargs)
        effects.append("directory")
        return value

    def mkstemp(*args, **kwargs):
        fd, name = actual_tempfile.mkstemp(*args, **kwargs)
        snapshots.append((fd, Path(name)))
        effects.append("snapshot")
        return fd, name

    def close(fd):
        calls.append(fd)
        if (mode == "fd" and snapshots and fd == snapshots[0][0]) or (
            mode == "source_fd" and source_fds and fd == source_fds[0]
        ):
            if after:
                actual_close(fd)
            raise OSError("private candidate close uncertainty")
        actual_close(fd)

    if mode == "foreign":
        actual_copy = schema._copy_source_to_snapshot

        def swap(*args, **kwargs):
            actual_copy(*args, **kwargs)
            snapshot = snapshots[0][1]
            snapshot.rename(snapshot.with_suffix(".owned"))
            snapshot.write_bytes(b"foreign candidate bytes")
            raise ValueError("candidate namespace changed")

        schema._copy_source_to_snapshot = swap

    schema._open_candidate_source = source_open
    schema._close_candidate_fd = close
    schema.tempfile.mkdtemp = mkdtemp
    schema.tempfile.mkstemp = mkstemp
    pause = storage._begin_local_pause() if mode == "pause" else None
    callback_calls = []
    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            schema.validate_profile_candidate(
                source, check_deadline=lambda: callback_calls.append(True)
            )
        assert caught.value.code == "schema_corrupt"
        if mode == "pause":
            assert effects == [], "paused standalone validation performed file effects"
            assert callback_calls == [], "paused entry invoked a caller callback"
        else:
            fd, snapshot = snapshots[0]
            if mode == "source_fd":
                fd = source_fds[0]
            assert snapshot.exists(), "uncertain snapshot close lost its namespace"
            if mode == "foreign":
                assert snapshot.read_bytes() == b"foreign candidate bytes"
            elif mode in ("connection", "read_connection"):
                import sqlite3

                native = native_connections[0]
                assert calls.count(native) == 1
                if after:
                    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                        native.execute("SELECT 1")
                else:
                    assert native.execute("SELECT 1").fetchone()[0] == 1
            elif after:
                with pytest.raises(OSError):
                    os.fstat(fd)
            else:
                assert (
                    os.fstat(fd).st_ino
                    == (source if mode == "source_fd" else snapshot).stat().st_ino
                )
            assert calls.count(fd) == 1
            assert storage._raw_operations, "native source job lost its strong owner"
            retained = set(storage._raw_operations)
            schema._close_candidate_fd = actual_close
            schema._open_candidate_source = actual_open
            schema.connect_private_sqlite = actual_connect
            schema.tempfile = actual_tempfile
            if mode == "foreign":
                schema._copy_source_to_snapshot = actual_copy
            schema.validate_profile_candidate(source)
            assert storage._raw_operations == retained
            pause = storage._begin_local_pause()
            assert not pause.drain(time.monotonic() + 0.02)
    finally:
        if pause is not None:
            pause.resume()


@pytest.mark.parametrize(
    "mode,after",
    [
        ("pause", False),
        ("fd", False),
        ("fd", True),
        ("source_fd", False),
        ("source_fd", True),
        ("connection", False),
        ("connection", True),
        ("read_connection", False),
        ("read_connection", True),
        ("foreign", False),
    ],
)
def test_candidate_first_io_and_uncertain_native_retirement(tmp_path, mode, after):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.TTS.test_profile_candidate_native_maintenance import _candidate_child
_candidate_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True')
""",
        mode,
        str(after),
    )


def test_callback_pause_refuses_later_native_allocation_and_unrelated_io(tmp_path):
    _run_private_child(
        tmp_path,
        """
import sys, time
from pathlib import Path
import pytest
import tldw_chatbook.TTS.profile_schema as schema
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
root = Path(sys.argv[1])
source = root / 'candidate.sqlite'
native = schema.open_profile_store(source)
native.close()
opened = []
actual_open = schema._open_candidate_source
pause = None
calls = []
def open_source(*args):
    fd = actual_open(*args)
    opened.append(fd)
    return fd
def callback():
    global pause
    calls.append(True)
    if opened and pause is None:
        pause = storage._begin_local_pause()
        with pytest.raises(RecoveryRequired):
            storage.acquire_storage(root / 'unrelated')
schema._open_candidate_source = open_source
try:
    with pytest.raises(ProfileRepositoryError):
        schema.validate_profile_candidate(source, check_deadline=callback)
    assert calls and len(opened) == 1
    assert not list(root.glob('tldw-tts-profile-candidate-*'))
    with pytest.raises(OSError):
        schema.os.fstat(opened[0])
    assert not storage._raw_operations
    assert pause.drain(time.monotonic() + 1)
finally:
    if pause is not None:
        pause.resume()
""",
    )


@pytest.mark.parametrize(
    "role", ["source", "snapshot", "directory_parent", "directory", "snapshot_parent"]
)
def test_candidate_retains_each_returned_fd_before_its_metadata_probe(tmp_path, role):
    _run_private_child(
        tmp_path,
        """
import sys, types
from pathlib import Path
import pytest
import tldw_chatbook.TTS.profile_schema as schema
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
source = Path(sys.argv[1]) / 'candidate.sqlite'
native = schema.open_profile_store(source)
native.close()
role = sys.argv[2]
actual_os = schema.os
schema.os = types.SimpleNamespace(**vars(actual_os))
observed = []
def fstat(fd):
    for job in storage._raw_operations:
        if type(job) is schema._CandidateValidationJob and job.descriptors.get(role) == fd:
            observed.append((job, fd))
            raise OSError('private metadata failure')
    return actual_os.fstat(fd)
schema.os.fstat = fstat
with pytest.raises(ProfileRepositoryError):
    schema.validate_profile_candidate(source)
assert observed
job, fd = observed[0]
assert job.source == source and job.resolved_source == source
if role == 'source':
    assert job not in storage._raw_operations
    with pytest.raises(OSError):
        actual_os.fstat(fd)
else:
    assert job in storage._raw_operations and job.uncertain
    assert any(lease in storage._live_leases for lease in job.leases)
""",
        role,
    )


def _native_observer_child(root, failure):
    import json
    import select
    import subprocess
    import sys
    import time

    import tldw_chatbook.TTS.profile_schema as schema
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    storage.admit_startup()
    if failure:
        _candidate_child(root, "fd", True)
        operation = next(
            job
            for job in storage._raw_operations
            if type(job) is schema._CandidateValidationJob
        )
        hold = storage._holds[operation.leases[0]._key]
        assert operation.descriptors["snapshot_parent"] >= 0
    else:
        source = root / "candidate.sqlite"
        native = schema.open_profile_store(source)
        native.close()
        connections = []
        actual_connect = schema.connect_private_sqlite

        def connect(*args, **kwargs):
            value = actual_connect(*args, **kwargs)
            connections.append(value)
            return value

        schema.connect_private_sqlite = connect
        schema.validate_profile_candidate(source)
        assert not storage._raw_operations
        import sqlite3

        assert len(connections) == 2
        for native in connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                native.execute("SELECT 1")
        hold = next(iter(storage._holds.values()))
    observer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import json, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.admission import Admission
print('waiting', flush=True)
with Admission(Path(sys.argv[1])).maintenance(tuple(json.loads(sys.argv[2])), 8):
    print('entered', flush=True)
    input()
""",
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
            time.sleep(0.005)
        assert not select.select([observer.stdout], [], [], 0.03)[0]
        pause = storage._begin_local_pause()
        try:
            assert pause.drain(time.monotonic() + 0.03) is (not failure)
            if failure:
                assert operation.leases[0] in storage._live_leases
                assert not select.select([observer.stdout], [], [], 0.03)[0]
            else:
                # Only this known minimal diagnostic process releases startup.
                # No production app/startup qualification follows from this test.
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
def test_candidate_native_maintainer_exclusion_and_positive_retirement(
    tmp_path, failure
):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.TTS.test_profile_candidate_native_maintenance import _native_observer_child
_native_observer_child(Path(sys.argv[1]), sys.argv[2] == 'True')
""",
        str(failure),
    )


@pytest.mark.parametrize("mode", ["success", "pause", "failure"])
@pytest.mark.parametrize("primitive", ["O_DIRECTORY", "fchmod"])
def test_missing_parent_pin_primitive_keeps_ordinary_candidate_semantics(
    tmp_path, mode, primitive
):
    _run_private_child(
        tmp_path,
        """
import sys, types, time
from pathlib import Path
import pytest
import tldw_chatbook.TTS.profile_schema as schema
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
root = Path(sys.argv[1])
source = root / 'candidate.sqlite'
native = schema.open_profile_store(source)
native.close()
mode = sys.argv[2]
actual_os = schema.os
schema.os = types.SimpleNamespace(**vars(actual_os))
setattr(schema.os, sys.argv[3], 0 if sys.argv[3] == "O_DIRECTORY" else None)
actual_open = schema.os.open
directory_attempts = []
def reject_directory_open(path, *args, **kwargs):
    if Path(path).is_dir():
        directory_attempts.append(path)
    assert not Path(path).is_dir(), 'missing directory primitive was used'
    return actual_open(path, *args, **kwargs)
schema.os.open = reject_directory_open
pause = storage._begin_local_pause() if mode == 'pause' else None
if mode == 'failure':
    actual_close = schema._close_candidate_fd
    def fail_close(fd):
        actual_close(fd)
        raise OSError('portable close uncertainty')
    schema._close_candidate_fd = fail_close
try:
    if mode == 'success':
        schema.validate_profile_candidate(source)
        assert not storage._raw_operations
    else:
        with pytest.raises(ProfileRepositoryError):
            schema.validate_profile_candidate(source)
        if mode == 'failure':
            job = next(iter(storage._raw_operations))
            assert job.uncertain
            assert any(lease._key is not None for lease in job.leases)
            assert all(lease in storage._live_leases for lease in job.leases)
    assert directory_attempts == []
    pause = pause or storage._begin_local_pause()
    with pytest.raises(RecoveryRequired):
        pause.require_runtime_coverage()
    assert pause.drain(time.monotonic() + .02) is (mode != 'failure')
finally:
    pause.resume()
""",
        mode,
        primitive,
    )


@pytest.mark.parametrize("mode", ["sidecar", "unlinked", "directory"])
def test_candidate_preserves_unproven_cleanup_namespaces(tmp_path, mode):
    _run_private_child(
        tmp_path,
        """
import sys, types, time
from pathlib import Path
import pytest
import tldw_chatbook.TTS.profile_schema as schema
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
source = Path(sys.argv[1]) / 'candidate.sqlite'
native = schema.open_profile_store(source)
native.close()
mode = sys.argv[2]
original_copy = schema._copy_source_to_snapshot
selected = []
def copy(*args, **kwargs):
    original_copy(*args, **kwargs)
    job = next(job for job in storage._raw_operations if type(job) is schema._CandidateValidationJob)
    snapshot = job.snapshot
    selected.append((job, snapshot))
    if mode == 'sidecar':
        Path(str(snapshot) + '-journal').write_bytes(b'foreign journal')
    elif mode == 'unlinked':
        snapshot.unlink()
    else:
        directory = job.directory
        directory.rename(directory.with_suffix('.owned'))
        directory.mkdir()
        (directory / 'foreign').write_bytes(b'foreign directory')
    raise ValueError('namespace changed')
schema._copy_source_to_snapshot = copy
with pytest.raises(ProfileRepositoryError):
    schema.validate_profile_candidate(source)
job, snapshot = selected[0]
assert job in storage._raw_operations and job.uncertain
if mode == 'sidecar':
    assert Path(str(snapshot) + '-journal').read_bytes() == b'foreign journal'
    assert snapshot.exists()
elif mode == 'unlinked':
    assert not snapshot.exists()
else:
    assert (job.directory / 'foreign').read_bytes() == b'foreign directory'
pause = storage._begin_local_pause()
try:
    assert not pause.drain(time.monotonic() + .02)
finally:
    pause.resume()
""",
        mode,
    )


def test_cancelled_backup_caller_keeps_actual_validation_job_until_worker_settles(
    tmp_path,
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys, threading, time
from pathlib import Path
import pytest
import tldw_chatbook.TTS.profile_schema as schema
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.Backup_Recovery import storage_admission as storage
async def run():
    root = Path(sys.argv[1])
    repo = TTSProfileRepository(root / 'profiles.sqlite')
    await repo.open()
    entered, finish = threading.Event(), threading.Event()
    actual_copy = schema._copy_source_to_snapshot
    def copy(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        actual_copy(*args, **kwargs)
    schema._copy_source_to_snapshot = copy
    pending = asyncio.create_task(repo.backup_to(root / 'snapshot.sqlite'))
    pause = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        job = next(job for job in storage._raw_operations if type(job) is schema._CandidateValidationJob)
        assert job.snapshot.exists()
        pending.cancel()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        repo._maintenance_close_admission()
        pause = storage._begin_local_pause()
        assert job in storage._raw_operations
        assert not await repo._maintenance_drain(time.monotonic() + .03)
        finish.set()
        assert await repo._maintenance_drain(time.monotonic() + 3)
        assert job not in storage._raw_operations
        assert not (root / 'snapshot.sqlite').exists()
        assert pause.drain(time.monotonic() + 1)
    finally:
        finish.set()
        if pause is not None:
            pause.resume()
        await repo.close()
asyncio.run(run())
""",
    )


def test_candidate_rechecks_pinned_directory_before_mode_or_child_creation(tmp_path):
    _run_private_child(
        tmp_path,
        """
import os, stat, sys, types
from pathlib import Path
import pytest
import tldw_chatbook.TTS.profile_schema as schema
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
root = Path(sys.argv[1])
source = root / 'candidate.sqlite'
native = schema.open_profile_store(source)
native.close()
original_pin = schema._CandidateValidationJob.pin_parent
original_tempfile = schema.tempfile
schema.tempfile = types.SimpleNamespace(**vars(original_tempfile))
foreign = []
children = []
def pin(job, role, path):
    result = original_pin(job, role, path)
    if role == 'directory':
        path.rename(path.with_suffix('.owned'))
        path.mkdir(mode=0o755)
        foreign.append((path, stat.S_IMODE(path.stat().st_mode)))
    return result
def mkstemp(*args, **kwargs):
    children.append(kwargs['dir'])
    return original_tempfile.mkstemp(*args, **kwargs)
schema._CandidateValidationJob.pin_parent = pin
schema.tempfile.mkstemp = mkstemp
with pytest.raises(ProfileRepositoryError):
    schema.validate_profile_candidate(source)
assert foreign
path, mode = foreign[0]
assert stat.S_IMODE(path.stat().st_mode) == mode, 'foreign directory was chmodded'
assert children == [], 'candidate allocated inside a replaced directory'
assert list(path.iterdir()) == []
assert storage._raw_operations
""",
    )


@pytest.mark.parametrize("role", ["directory_parent", "directory", "snapshot_parent"])
@pytest.mark.parametrize("after", [False, True])
def test_candidate_parent_close_uncertainty_keeps_exact_resource_and_lease(
    tmp_path, role, after
):
    _run_private_child(
        tmp_path,
        """
import os, sys, time
from pathlib import Path
import pytest
import tldw_chatbook.TTS.profile_schema as schema
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
root = Path(sys.argv[1])
source = root / 'candidate.sqlite'
native = schema.open_profile_store(source)
native.close()
role, after = sys.argv[2], sys.argv[3] == 'True'
actual_close = schema._close_candidate_fd
calls = []
def close(fd):
    job = next(job for job in storage._raw_operations if type(job) is schema._CandidateValidationJob)
    if job.descriptors.get(role) == fd:
        calls.append((job, fd))
        if after:
            actual_close(fd)
        raise OSError('parent native retirement uncertain')
    actual_close(fd)
schema._close_candidate_fd = close
with pytest.raises(ProfileRepositoryError):
    schema.validate_profile_candidate(source)
assert len(calls) == 1
job, fd = calls[0]
assert job.descriptors[role] == fd and role in job.attempted
assert job in storage._raw_operations
assert all(lease in storage._live_leases for lease in job.leases)
if after:
    with pytest.raises(OSError):
        os.fstat(fd)
else:
    assert os.fstat(fd).st_ino > 0
assert job.snapshot is None and job.directory is None
pause = storage._begin_local_pause()
try:
    assert not pause.drain(time.monotonic() + .02)
    assert not pause.drain(time.monotonic() + .02)
    assert len(calls) == 1
finally:
    pause.resume()
""",
        role,
        str(after),
    )
