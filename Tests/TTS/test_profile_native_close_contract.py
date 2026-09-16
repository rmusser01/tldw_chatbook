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
import ctypes, json, sys, time
from pathlib import Path
from tldw_chatbook.Backup_Recovery import admission
native = admission.fcntl
opened = {}
original_open = admission.Admission._open
def observe_open(parent, name, flags):
    descriptor = original_open(parent, name, flags)
    opened[descriptor] = "registry" if name == "registry.lock" else name.rsplit(".", 1)[-1]
    return descriptor
admission.Admission._open = staticmethod(observe_open)
class ObservedLocks:
    contended = False
    failures = set()
    def __getattr__(self, name):
        return getattr(native, name)
    def flock(self, *args):
        try:
            return native.flock(*args)
        except BlockingIOError:
            raw = ctypes.get_last_error() if sys.platform == "win32" else None
            self.contended = True
            self.failures.add((opened.get(args[0], "unknown"), raw))
            raise
locks = ObservedLocks()
admission.fcntl = locks
started = time.monotonic()
try:
    with admission.Admission(Path(sys.argv[1])).maintenance(tuple(json.loads(sys.argv[2])), 2):
        status = "entered"
except admission.AdmissionTimeout:
    status = "contended" if locks.contended else "timeout_without_contention"
print(json.dumps({"status": status, "elapsed": time.monotonic() - started, "locks": sorted(locks.failures)}))
""", str(control), json.dumps(names)],
        capture_output=True,
        text=True,
        timeout=6,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _native_hold_summary(storage, selected):
    """Bound failure receipts to owner roles and state; omit all user paths."""
    from collections import Counter

    def state(hold):
        return {
            "count": hold.count,
            "thread_alive": hold.thread.is_alive(),
            "stopped": hold.stop.is_set(),
            "error": type(hold.error).__name__ if hold.error else None,
        }

    with storage._lock:
        roles = Counter(
            (
                type(getattr(lease, "native_owner", None)).__name__,
                getattr(getattr(lease, "resource_policy", None), "production_module", None),
                lease._key == selected.key,
                lease.resource_close_failed,
            )
            for lease in storage._live_leases
        )
        return {
            "selected_hold": state(selected),
            "active_hold_count": len(storage._holds),
            "retiring_hold_count": len(storage._retiring_holds),
            "retiring": [state(hold) for hold in tuple(storage._retiring_holds)[:16]],
            "live_lease_count": len(storage._live_leases),
            "roles": [(*role, count) for role, count in roles.items()][:16],
        }


async def _native_close_child(root, fault):
    import sqlite3
    from pathlib import Path

    from Tests.TTS.test_profile_schema import _build_candidate_version
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_repository as repository
    from tldw_chatbook.TTS import profile_schema as schema
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    home_aligned = Path.home() == root / "home"
    authority_aligned = storage.bootstrap.default_bootstrap_root().is_relative_to(
        root / "home"
    )
    assert home_aligned and authority_aligned, {
        "home_aligned": home_aligned, "authority_aligned": authority_aligned
    }
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
        assert probe["status"] == "contended", {
            "probe": probe, "state": _native_hold_summary(storage, hold)
        }
    else:
        before = _native_hold_summary(storage, hold)
        held = _native_admission_probe(hold.authority.control_root, hold.names)
        assert held["status"] == "contended", {"probe": held, "state": before}
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
        after = _native_hold_summary(storage, hold)
        released = _native_admission_probe(hold.authority.control_root, hold.names)
        assert released["status"] == "entered", {
            "probe": released, "before": before, "after": after
        }


@pytest.mark.parametrize("fault", ["success", "live_before", "live_after", "evidence"])
def test_native_repository_close_preserves_proven_state_and_exclusion(
    tmp_path, monkeypatch, fault
):
    # The child helper selects HOME; Windows Path.home selects USERPROFILE.
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
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
