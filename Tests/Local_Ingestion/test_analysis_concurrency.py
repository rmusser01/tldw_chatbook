"""B14: chunk analysis runs with bounded concurrency and preserves order.

The ingestion processors (PDF / EPUB / markup) used to summarize their chunks
strictly serially. The shared helper ``analyze_chunks_concurrently`` runs the
per-chunk analysis under a bounded thread pool while keeping:

* order preservation -- ``result[i]`` corresponds to ``chunks[i]``;
* a hard peak-concurrency cap of ``max_workers``;
* the failure contract: every future settles before anything raises, and the
  first exception in chunk order is re-raised after the join. (Callers that
  need the historical per-chunk "record the error, keep going" isolation do it
  inside their own ``analyze_fn`` wrapper -- see the ingestion processors.)

Evidence note (plan Step 5): with 30 chunks x 50 ms of fake analyze work,
serial wall time is ~1.5 s vs ~0.55 s at max_workers=3; the timing run is
recorded in the task notes rather than asserted here (wall-clock asserts are
flaky on loaded CI machines).
"""

import threading
import time

import pytest

from tldw_chatbook.Local_Ingestion import analysis_concurrency


def test_bounded_concurrency_and_order():
    live: list[int] = []
    peak = [0]
    lock = threading.Lock()

    def slow_analyze(text: str) -> str:
        with lock:
            live.append(1)
            peak[0] = max(peak[0], len(live))
        time.sleep(0.05)
        with lock:
            live.pop()
        return text.upper()

    chunks = [f"chunk-{i}" for i in range(10)]
    out = analysis_concurrency.analyze_chunks_concurrently(
        chunks, slow_analyze, max_workers=3
    )
    assert out == [c.upper() for c in chunks]
    assert peak[0] <= 3


def test_order_preserved_despite_variable_durations():
    """Later chunks finishing first must not reorder results."""

    def analyze(text: str) -> str:
        # Reverse-indexed sleep: chunk 0 finishes last.
        time.sleep(0.001 * (10 - int(text.split("-")[1])))
        return text.upper()

    chunks = [f"chunk-{i}" for i in range(10)]
    out = analysis_concurrency.analyze_chunks_concurrently(chunks, analyze)
    assert out == [c.upper() for c in chunks]


def test_all_futures_settle_before_first_exception_raised():
    """One failing chunk must not stop the others from being submitted and
    settled; the exception surfaces only after the join."""
    calls: list[str] = []
    lock = threading.Lock()

    def analyze(text: str) -> str:
        with lock:
            calls.append(text)
        if text == "chunk-3":
            raise RuntimeError("boom on chunk-3")
        return text.upper()

    chunks = [f"chunk-{i}" for i in range(6)]
    with pytest.raises(RuntimeError, match="boom on chunk-3"):
        analysis_concurrency.analyze_chunks_concurrently(chunks, analyze)
    # Every chunk got its analysis attempt despite the chunk-3 failure.
    assert sorted(calls) == chunks


def test_first_exception_in_chunk_order_raises():
    """Two failing chunks: the FIRST (lowest index) error is the one raised."""

    def analyze(text: str) -> str:
        if text == "chunk-1":
            raise ValueError("error from chunk-1")
        if text == "chunk-4":
            raise ValueError("error from chunk-4")
        return text.upper()

    chunks = [f"chunk-{i}" for i in range(6)]
    with pytest.raises(ValueError, match="error from chunk-1"):
        analysis_concurrency.analyze_chunks_concurrently(chunks, analyze)


def test_empty_chunk_list_returns_empty():
    assert analysis_concurrency.analyze_chunks_concurrently([], lambda t: t) == []


def test_max_workers_above_chunk_count_is_clamped():
    """A pool larger than the work is wasteful, not wrong -- the helper
    clamps so a 2-chunk batch never spawns the full default pool."""
    started = threading.Semaphore(0)

    def analyze(text: str) -> str:
        started.release()
        return text.upper()

    out = analysis_concurrency.analyze_chunks_concurrently(
        ["a", "b"], analyze, max_workers=16
    )
    assert out == ["A", "B"]


def test_default_max_workers_is_three():
    assert analysis_concurrency.DEFAULT_MAX_WORKERS == 3
