"""Deterministic representative-tree benchmark for Notes reconciliation."""

from __future__ import annotations

import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tldw_chatbook.Notes.notes_sync_models import NotesSyncDirection
from tldw_chatbook.Notes.notes_sync_reconciler import (
    BindingObservation,
    DELETION_GROUP_THRESHOLD,
    plan_reconciliation,
)

BASE_FILE = "1" * 64
BASE_NOTE = "2" * 64
IDENTITY = "3" * 64


def _binding(index: int, *, deletion_burst: bool) -> BindingObservation:
    case = index % 8
    file_digest = None if deletion_burst or case == 6 else BASE_FILE
    note_digest = BASE_NOTE
    if case == 1:
        file_digest = "4" * 64
    elif case == 2:
        note_digest = "5" * 64
    elif case == 3:
        file_digest, note_digest = "4" * 64, "5" * 64
    relative = f"folder-{index // 100:04d}/note-{index:06d}.md"
    if case == 4:
        relative = f"moved-{index // 100:04d}/note-{index:06d}.md"
    if deletion_burst:
        file_digest = None
        note_digest = BASE_NOTE
        relative = f"folder-{index // 100:04d}/note-{index:06d}.md"
    return BindingObservation(
        binding_id=f"binding-{index:06d}",
        baseline_file_digest=BASE_FILE,
        baseline_note_digest=BASE_NOTE,
        baseline_identity_digest=IDENTITY,
        baseline_relative_path=f"folder-{index // 100:04d}/note-{index:06d}.md",
        file_digest=file_digest,
        note_digest=note_digest,
        file_identity_digest=None if file_digest is None else IDENTITY,
        relative_path=relative,
        note_scope_id="scope-benchmark",
        note_id=f"note-{index:06d}",
        note_version=index,
    )


def _measure(count: int, *, deletion_burst: bool) -> tuple[float, int, int]:
    from tldw_chatbook.Notes.notes_sync_reconciler import ReconciliationInput

    request = ReconciliationInput(
        root_id="root-benchmark",
        direction=NotesSyncDirection.BIDIRECTIONAL,
        bindings=tuple(
            _binding(index, deletion_burst=deletion_burst) for index in range(count)
        ),
        observation_generation=1,
        expected_generation=1,
    )
    before = repr(request)
    plan_reconciliation(request)  # Warm caches outside the measurement.
    durations: list[float] = []
    peaks: list[int] = []
    classified = 0
    for _ in range(5):
        tracemalloc.start()
        started = time.perf_counter()
        plan = plan_reconciliation(request)
        durations.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
        classified = (
            len(plan.safe_actions)
            + len(plan.attention)
            + sum(len(group.binding_ids) for group in plan.deletion_groups)
        )
    assert repr(request) == before
    assert classified == count
    if deletion_burst:
        assert bool(plan.deletion_groups) is (count >= DELETION_GROUP_THRESHOLD)
    return statistics.median(durations), max(peaks), classified


def main() -> None:
    print("count\tscenario\tmedian_ms\tpeak_kib\tclassified")
    for count in (99, 100, 101, 1_000, 5_000, 10_000):
        for scenario, deletion_burst in (("mixed", False), ("deletion", True)):
            duration, peak, classified = _measure(
                count,
                deletion_burst=deletion_burst,
            )
            print(
                f"{count}\t{scenario}\t{duration * 1000:.3f}\t"
                f"{peak / 1024:.1f}\t{classified}"
            )


if __name__ == "__main__":
    main()
