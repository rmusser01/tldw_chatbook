"""Bounded concurrency for the pairwise reranker's recursive sort (task 17, F16).

``PairwiseReranker`` merge-sorts via ``_tournament_rank``: split, recursively
sort the two halves, then ``_merge_with_comparisons`` interleaves them one LLM
comparison at a time. The two recursive HALVES are independent subproblems;
only the merge loop is order-dependent (each comparison decides the next), so
the merge stays sequential while the halves may run under an
``asyncio.Semaphore(max_concurrent_comparisons)`` created ONCE per ``rerank()``
call and threaded through the recursion -- bounding TOTAL concurrent
comparisons across the whole sort, not per level. With the default
``top_k_to_rerank = 20`` the sort makes ~87 comparisons; strictly sequential
at ~500 ms/call that is ~45 s of dead time between the halves.

What this module pins:

* **Equivalence.** For any comparator whose decision for a given pair is
  time-invariant, the output is IDENTICAL to the sequential implementation:
  gather returns its results in argument order, each half's result is
  deterministic by induction, and each merge still walks its inputs strictly
  in order. Pinned two ways against goldens captured from the PRE-change
  sequential code: a transitive comparator (by id) and a deliberately
  NON-transitive cyclic one, whose output is not a global sort of anything and
  therefore pins the exact merge structure. ``max_concurrent_comparisons=1``
  reproduces the old one-comparison-at-a-time behaviour and is the in-test
  serial baseline.
* **Bounding and reality of the parallelism.** A recording comparator asserts
  observed in-flight comparisons never exceed the cap and reach it.
* **Wall clock follows the critical path, not the sum.** 8 items at 50 ms per
  comparison: 12 comparisons serial (~600 ms) vs a 7-step critical path
  (~350 ms) under the default cap of 4.
* **Cap validation.** 0 / negative / non-int caps are rejected at construction.

Hermetic by construction: the fake replaces the instance's ``_call_llm`` (the
same seam ``test_reranker_construction`` uses), and BOTH prompt fields are
supplied explicitly so ``__init__`` never consults the prompt registry or the
config bootstrap -- no provider, no credential, no network.
"""

import asyncio
import re
import time
from typing import Callable, List, Tuple

import pytest

from tldw_chatbook.RAG_Search.reranker import (
    PairwiseReranker,
    RerankingConfig,
    RerankOutcome,
)
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult

#: Minimal template exercising every placeholder `_compare_pair` substitutes,
#: with the two titles on parseable lines of their own.
_TEMPLATE = "T1={title1}\nT2={title2}\nQ={query}\nC1={content1}\nC2={content2}\n{reasoning}"
_PAIR_RE = re.compile(r"^T1=(\S+)\nT2=(\S+)\n", re.MULTILINE)

#: Lower index wins -- a total order, so the golden output is the id sort.
_BY_ID: Callable[[int, int], bool] = lambda i1, i2: i1 < i2


def _cyclic(n: int) -> Callable[[int, int], bool]:
    """A NON-transitive preference: i beats j iff i is 1..n//2 steps ahead of
    j in the cycle (mod n). Deliberately not a total order, so the golden
    output depends on the exact merge structure rather than being "the sort"
    -- a much sharper pre/post equivalence pin."""

    def prefer(i1: int, i2: int) -> bool:
        return 1 <= ((i1 - i2) % n) <= n // 2

    return prefer


def _mk_results(indices: List[int]) -> List[SearchResult]:
    return [
        SearchResult(
            id=f"item-{i}",
            score=round(0.9 - 0.1 * i, 3),
            document=f"body of item number {i}",
            metadata={"doc_title": f"item-{i}"},
        )
        for i in indices
    ]


class _RecordingComparator:
    """Fake comparator: deterministic preference, controllable delay, and an
    in-flight gauge. Replaces the instance's ``_call_llm`` (the seam
    ``test_reranker_construction`` already uses), so the gauge measures
    exactly what the semaphore must bound: live ``_compare_pair`` executions.
    """

    def __init__(self, prefer: Callable[[int, int], bool], delay: float = 0.0):
        self._prefer = prefer
        self._delay = delay
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, prompt: str, system_prompt: str = None) -> str:
        m = _PAIR_RE.search(prompt)
        assert m, f"unparseable comparison prompt: {prompt!r}"
        i1 = int(m.group(1).split("-")[1])
        i2 = int(m.group(2).split("-")[1])
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            return '{"choice": 1}' if self._prefer(i1, i2) else '{"choice": 2}'
        finally:
            self.in_flight -= 1


async def _run(
    indices: List[int],
    prefer: Callable[[int, int], bool],
    max_concurrent: int,
    delay: float = 0.0,
) -> Tuple[_RecordingComparator, RerankOutcome, float]:
    """Rerank `indices` with a recording fake comparator; return the recorder,
    the outcome, and the wall-clock seconds the rerank took."""
    recorder = _RecordingComparator(prefer, delay=delay)
    config = RerankingConfig(
        strategy="pairwise",
        system_prompt="compare the two results",
        scoring_prompt_template=_TEMPLATE,
        retry_on_failure=False,
    )
    reranker = PairwiseReranker(config, max_concurrent_comparisons=max_concurrent)
    reranker._call_llm = recorder
    start = time.perf_counter()
    outcome = await reranker.rerank("concurrency probe query", _mk_results(indices))
    wall = time.perf_counter() - start
    return recorder, outcome, wall


# ---------------------------------------------------------------------------
# Cap validation and default
# ---------------------------------------------------------------------------


def _hermetic_config() -> RerankingConfig:
    return RerankingConfig(
        strategy="pairwise",
        system_prompt="compare the two results",
        scoring_prompt_template=_TEMPLATE,
    )


def test_default_cap_is_four():
    """The shipped default bounds a sort at 4 concurrent comparisons."""
    assert PairwiseReranker(_hermetic_config()).max_concurrent_comparisons == 4


@pytest.mark.parametrize("bad", [0, -1, -4])
def test_nonpositive_cap_is_rejected(bad):
    """A cap below 1 cannot bound anything and must fail at construction."""
    with pytest.raises(ValueError, match="max_concurrent_comparisons"):
        PairwiseReranker(_hermetic_config(), max_concurrent_comparisons=bad)


def test_non_integer_cap_is_rejected():
    """A fractional permit count is a configuration typo, not a rounding
    target; reject it rather than floor it."""
    with pytest.raises(TypeError, match="max_concurrent_comparisons"):
        PairwiseReranker(_hermetic_config(), max_concurrent_comparisons=2.5)


# ---------------------------------------------------------------------------
# Bounding: never above the cap, and the parallelism is real
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", [2, 4])
@pytest.mark.asyncio
async def test_comparator_concurrency_never_exceeds_the_cap_and_saturates_it(cap):
    """8 reversed items: the recursion offers 4 simultaneously-runnable leaf
    comparisons, so a working bound both LIMITS observed in-flight comparisons
    to the cap and lets them REACH it -- a semaphore that serialized
    everything would hold the <= cap half and fail the == cap half."""
    recorder, outcome, _ = await _run([7, 6, 5, 4, 3, 2, 1, 0], _BY_ID, cap, delay=0.025)

    assert recorder.max_in_flight <= cap, (
        f"{recorder.max_in_flight} comparisons ran at once under a cap of {cap}"
    )
    assert recorder.max_in_flight == cap, (
        f"parallelism is not real: only {recorder.max_in_flight} of {cap} "
        "permitted comparisons were ever in flight together"
    )
    assert recorder.calls > 0


# ---------------------------------------------------------------------------
# Wall clock: critical path, not the sum
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wall_clock_follows_the_critical_path_not_the_sum():
    """8 items at 50 ms/comparison. Serial (cap=1): 12 comparisons, ~600 ms.
    Concurrent (cap=4, default): every level's merges overlap, leaving a
    7-step critical path of ~350 ms. Assert the concurrent run beats 80% of
    the serial wall (theory says ~59%, so the threshold leaves CI-jitter
    headroom in both directions) while both runs still really sleep: the
    serial floor catches an accidentally-instant fake, the concurrent floor
    catches a fake that skipped its delays."""
    delay = 0.05

    serial_rec, serial_out, serial_wall = await _run(
        [7, 6, 5, 4, 3, 2, 1, 0], _BY_ID, max_concurrent=1, delay=delay
    )
    conc_rec, conc_out, conc_wall = await _run(
        [7, 6, 5, 4, 3, 2, 1, 0], _BY_ID, max_concurrent=4, delay=delay
    )

    # cap=1 reproduces the old behaviour exactly: one comparison at a time.
    assert serial_rec.max_in_flight == 1
    assert serial_wall >= 10 * delay, "serial baseline must really sleep (12 x 50ms)"
    assert conc_wall >= 4 * delay, "concurrent run must still sleep its critical path"
    assert conc_wall < 0.80 * serial_wall, (
        f"no meaningful speedup: concurrent {conc_wall:.3f}s vs serial "
        f"{serial_wall:.3f}s (ratio {conc_wall / serial_wall:.2f})"
    )
    assert [r.id for r in conc_out.results] == [r.id for r in serial_out.results]


# ---------------------------------------------------------------------------
# Equivalence: identical output to the sequential implementation
# ---------------------------------------------------------------------------

#: Goldens captured by running the PRE-change sequential implementation
#: (worktree a1cb45301e) with these exact inputs and comparators. The
#: non-transitive golden is the sharp one: its output is not a global sort of
#: anything, so it can only match if every merge walked exactly as before.
_GOLDENS = [
    pytest.param(
        [7, 6, 5, 4, 3, 2, 1, 0],
        _BY_ID,
        ["item-0", "item-1", "item-2", "item-3", "item-4", "item-5", "item-6", "item-7"],
        12,
        id="transitive-by-id",
    ),
    pytest.param(
        [4, 0, 8, 2, 6, 1, 5, 3, 7],
        _cyclic(9),
        [
            "item-4",
            "item-2",
            "item-1",
            "item-0",
            "item-8",
            "item-7",
            "item-6",
            "item-5",
            "item-3",
        ],
        17,
        id="non-transitive-cyclic",
    ),
]


@pytest.mark.parametrize("indices,prefer,expected_ids,expected_total", _GOLDENS)
@pytest.mark.asyncio
async def test_output_identical_to_prechange_golden_and_to_serial(
    indices, prefer, expected_ids, expected_total
):
    """Two cross-checks, per the task resolution: (1) the pinned order
    captured from the PRE-change sequential code, and (2) a post-change
    max_concurrent=1 run (which reproduces the sequential semantics exactly)
    -- the default-cap run must agree with both, comparison count included."""
    serial_rec, serial_out, _ = await _run(indices, prefer, max_concurrent=1)
    conc_rec, conc_out, _ = await _run(indices, prefer, max_concurrent=4)

    serial_ids = [r.id for r in serial_out.results]
    conc_ids = [r.id for r in conc_out.results]

    assert serial_ids == expected_ids, "serial run drifted from the pre-change golden"
    assert conc_ids == expected_ids, (
        "concurrent run reordered relative to the pre-change golden"
    )
    assert conc_ids == serial_ids
    assert (serial_out.total, conc_out.total) == (expected_total, expected_total), (
        "the comparison COUNT changed under concurrency"
    )
    assert serial_out.failed == conc_out.failed == 0
