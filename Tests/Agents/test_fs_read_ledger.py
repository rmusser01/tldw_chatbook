"""TASK-28238 phase 1: read-ledger for the stale-write guard."""

import threading

from tldw_chatbook.Agents.fs_read_ledger import (
    ReadLedger,
    canonical_ledger_key,
)


def test_record_and_lookup_present():
    ledger = ReadLedger()
    ledger.record_present("run-a", "/x/f.txt", "ab" * 32, 10)
    stamp = ledger.stamp_for("run-a", "/x/f.txt")
    assert stamp is not None and stamp.sha256 == "ab" * 32 and stamp.size == 10


def test_runs_are_independent():
    ledger = ReadLedger()
    ledger.record_present("run-a", "/x/f.txt", "aa" * 32, 1)
    ledger.record_present("run-b", "/x/f.txt", "bb" * 32, 2)
    assert ledger.stamp_for("run-a", "/x/f.txt").sha256 == "aa" * 32
    assert ledger.stamp_for("run-b", "/x/f.txt").sha256 == "bb" * 32
    assert ledger.stamp_for("run-c", "/x/f.txt") is None


def test_absent_stamp():
    ledger = ReadLedger()
    ledger.record_absent("run-a", "/x/missing.txt")
    stamp = ledger.stamp_for("run-a", "/x/missing.txt")
    assert stamp is not None and stamp.is_absent


def test_update_written_replaces():
    ledger = ReadLedger()
    ledger.record_present("run-a", "/x/f.txt", "aa" * 32, 1)
    ledger.update_written("run-a", "/x/f.txt", "cc" * 32, 3)
    assert ledger.stamp_for("run-a", "/x/f.txt").sha256 == "cc" * 32


def test_per_run_cap_evicts_oldest():
    ledger = ReadLedger(max_paths_per_run=3)
    for i in range(5):
        ledger.record_present("run-a", f"/x/{i}.txt", "aa" * 32, i)
    assert ledger.stamp_for("run-a", "/x/0.txt") is None
    assert ledger.stamp_for("run-a", "/x/1.txt") is None
    assert ledger.stamp_for("run-a", "/x/4.txt") is not None
    # other runs unaffected by run-a's evictions
    ledger.record_present("run-b", "/x/0.txt", "bb" * 32, 0)
    assert ledger.stamp_for("run-b", "/x/0.txt") is not None


def test_thread_safety_no_lost_updates():
    ledger = ReadLedger(max_paths_per_run=10_000)

    def hammer(run_id):
        for i in range(500):
            ledger.record_present(run_id, f"/x/{i}.txt", "aa" * 32, i)

    threads = [threading.Thread(target=hammer, args=(f"run-{t}",)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for t in range(4):
        assert ledger.stamp_for(f"run-{t}", "/x/499.txt") is not None


def test_canonical_key_matches_cas_canonicalization(tmp_path):
    import os
    resolved = (tmp_path / "f.txt").resolve()
    assert canonical_ledger_key(resolved) == os.path.normcase(str(resolved.absolute()))
