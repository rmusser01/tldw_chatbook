"""Recheck watch evidence after its final native drive-root observation.

Apply alongside TASK-34601's test_windows_evidence_watch.py. These are native
regression controls, not timing measurements.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_windows_evidence_watch as oracle
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import windows_files

pytestmark = pytest.mark.skipif(os.name != "nt", reason="real Windows NTFS evidence")
native_scope = oracle.native_scope


@pytest.mark.parametrize("change", ["by-id-write", "backstop-expiry", "invalidated"])
def test_watch_evidence_is_rechecked_after_root_observation(
    native_scope, monkeypatch, change
):
    """A watch valid at lease acquisition can become stale before acceptance."""
    _, selector, _, _ = native_scope
    target, hold, evidence = oracle._watched(native_scope)
    oracle._assert_fast(target, monkeypatch)
    watch = oracle._verified(hold, evidence)
    assert watch is not None
    before = hold.count
    original = storage._anchors_unchanged
    crossed = []
    clock_offset = [0.0]
    original_clock = time.monotonic

    def change_after_root_observation(entries):
        unchanged = original(entries)
        assert unchanged
        if not crossed:
            assert hold.count == before + 1
            assert watch.users > 0, "the verified watch must already be claimed"
            assert watch.watch.quiet()
            crossed.append(True)
            if change == "by-id-write":
                generation = windows_files.native_mutation_generation()
                windows_files.WindowsOS().chmod(selector, 0o600)
                assert windows_files.native_mutation_generation() > generation
                assert watch.watch.quiet(), "this mutation must be notification-silent"
            elif change == "backstop-expiry":
                clock_offset[0] = storage._EVIDENCE_WATCH_BACKSTOP_S + 0.01
            else:
                storage._invalidate_watches(hold)
        return unchanged

    with monkeypatch.context() as patch:
        # Advance only the admission module's clock; do not sleep or alter any
        # global clock used by the fixture, OS facade, or native watch.
        patch.setattr(
            storage,
            "time",
            SimpleNamespace(
                monotonic=lambda: original_clock() + clock_offset[0],
                sleep=time.sleep,
                time_ns=time.time_ns,
            ),
        )
        patch.setattr(storage, "_anchors_unchanged", change_after_root_observation)
        observed, _ = oracle._count_observations(patch)
        result = oracle._verdict(target)

    assert crossed, "the existing fast path must reach its final observation"
    assert observed and max(observed) > oracle._root_count(
        target
    ), "stale watch evidence needs a full observation"
    assert result == oracle._derived(target, monkeypatch)
    assert result[0] == "allowed"
    assert hold.count == before, "the acquisition must retire its counted lease"
