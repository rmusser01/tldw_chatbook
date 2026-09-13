"""Native exact-current owners prove SQLite closure before retiring admission."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


def _native_admission_probe(control, names):
    """Distinguish observed lock contention from an expired traversal budget."""
    import json
    import subprocess  # nosec B404
    import sys

    # Fixed test interpreter/code and separate path arguments; no shell.
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", """
import json, sys, time
from pathlib import Path
from tldw_chatbook.Backup_Recovery import admission
native = admission.fcntl
class ObservedLocks:
    contended = False
    def __getattr__(self, name):
        return getattr(native, name)
    def flock(self, *args):
        try:
            return native.flock(*args)
        except BlockingIOError:
            self.contended = True
            raise
locks = ObservedLocks()
admission.fcntl = locks
started = time.monotonic()
try:
    with admission.Admission(Path(sys.argv[1])).maintenance(tuple(json.loads(sys.argv[2])), 2):
        status = "entered"
except admission.AdmissionTimeout:
    status = "contended" if locks.contended else "timeout_without_contention"
print(json.dumps({"status": status, "elapsed": time.monotonic() - started}))
""", str(control), json.dumps(names)],
        capture_output=True,
        text=True,
        timeout=6,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


async def _native_close_child(root, fault):
    import sqlite3

    from Tests.TTS.test_profile_schema import _build_candidate_version
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_repository as repository
    from tldw_chatbook.TTS import profile_schema as schema
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    active = root / "profiles.sqlite"
    fixture = _build_candidate_version(active, 4)
    assert fixture.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    fixture.execute("SELECT * FROM tts_generation_profiles").fetchall()
    active.chmod(0o600)
    # Select the actual Windows owner, preserving native SQLite and descriptors.
    repository.open_exact_current_profile_store = (
        schema._open_native_exact_current_profile_store
    )
    repo = repository.TTSProfileRepository(active)
    try:
        await repo.open()
    finally:
        fixture.close()
    owner = repo._connection
    assert isinstance(owner, schema._NativeExactCurrentProfileConnection)
    live, evidence = owner._connection, owner._evidence_connection
    hold = storage._holds[owner.leases[0]._key]
    signal = OSError("native close outcome unavailable")
    closes = []
    failures = []
    original_raise = repository._raise_cleanup_errors

    def raise_cleanup(*errors):
        failures.extend(error for error in errors if error is not None)
        original_raise(*errors)

    repository._raise_cleanup_errors = raise_cleanup

    class UncertainClose:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            return getattr(self.native, name)

        def close(self):
            closes.append(self.native)
            if fault == "live_after":
                self.native.close()
            raise signal

    if fault != "success":
        if fault == "evidence":
            owner._evidence_connection = UncertainClose(evidence)
        else:
            owner._connection = UncertainClose(live)
        with pytest.raises(ProfileRepositoryError):
            await repo.close()
        assert owner.cleanup_errors == [signal], [repr(error) for error in failures]
        assert owner._sqlite_closed is (fault == "evidence")
        assert owner.uncertain and owner.leases and not owner.pending
        assert repo._connection is owner and repo._lease.acquired
        assert len(closes) == 1
        with pytest.raises(schema.ExactProfileStoreCleanupError):
            owner.close()
        assert len(closes) == 1
        if fault in {"live_after", "evidence"}:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                _ = live.in_transaction
        else:
            assert not live.in_transaction
        storage._shutdown()
        probe = _native_admission_probe(hold.authority.control_root, hold.names)
        assert probe["status"] == "contended", probe
    else:
        held = _native_admission_probe(hold.authority.control_root, hold.names)
        assert held["status"] == "contended", held
        try:
            await repo.close()
        except ProfileRepositoryError:
            pytest.fail(repr([repr(error) for error in failures]))
        assert owner._sqlite_closed and not owner.uncertain
        assert not owner.leases and not owner.native_descriptors
        assert repo._connection is None and repo._lease is None
        for connection in (live, evidence):
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                _ = connection.in_transaction
        owner.close()
        released = _native_admission_probe(hold.authority.control_root, hold.names)
        assert released["status"] == "entered", released


@pytest.mark.parametrize("fault", ["success", "live_before", "live_after", "evidence"])
def test_native_repository_close_preserves_proven_state_and_exclusion(tmp_path, fault):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_native_close_contract import _native_close_child
asyncio.run(_native_close_child(Path(sys.argv[1]), sys.argv[2]))
""",
        fault,
    )
