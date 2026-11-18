"""Explicit Windows component measurement for one complete, counted warm lease."""

import json
from pathlib import Path
import time

from Tests.Backup_Recovery.test_agents_ordinary_activation_batch import _warm
from Tests.Backup_Recovery.test_agents_vector_admission import (
    native_scope as native_scope_fixture,
)
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import windows_files

native_scope = native_scope_fixture
_SETTLE_NS = storage._EVIDENCE_SETTLE_NS


def test_counted_warm_acquire_and_close(native_scope, monkeypatch, request):
    checkout = Path.cwd().resolve()
    for module in (storage, windows_files):
        assert (
            Path(module.__file__).resolve().is_relative_to(checkout / "tldw_chatbook")
        )
    _, selector, data, startup = native_scope
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", _SETTLE_NS)
    target = data / "child"
    windows_files.WindowsOS().mkdir(target, 0o700)
    time.sleep(_SETTLE_NS / 1_000_000_000 + 0.05)
    _warm((target,))
    key = startup._key
    hold = storage._holds[key]
    evidence = hold.evidence[str(selector)]
    assert (
        evidence.confirmed and hold.path_evidence[str(selector), str(target)].confirmed
    )
    before = set(storage._live_leases)
    assert hold.count == 1 and startup in before

    samples_ns = []
    for _ in range(21):
        start = time.perf_counter_ns()
        with storage.acquire_storage(target) as lease:
            observed = lease._warm_execution
            counted = hold.count
        samples_ns.append(time.perf_counter_ns() - start)
        assert observed is not None and observed[0] is hold and observed[1] is evidence
        assert counted == 2 and hold.count == 1
        assert lease._key is None and lease._warm_execution is None
        assert not lease.resource_close_failed and set(storage._live_leases) == before

    # Coordinator retirement is a separate control after the warm borrower timings.
    startup.close()
    assert key not in storage._holds and hold not in storage._retiring_holds
    assert not hold.thread.is_alive() and storage._live_leases == before - {startup}
    request.node.user_properties.append(
        ("complete_warm_lease_ns", json.dumps(samples_ns))
    )
