"""Grid execution: order, progress, cancel, and preflight propagation."""

from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig, CellCapture, CellError, PreflightResult, Snippet, Target, TokenProb,
)
from tldw_chatbook.Evals.word_bench.runner import CancelToken, WordBenchRunner
from tldw_chatbook.Evals.word_bench.storage import load_grid, save_bench


# db, snippets, targets, config come from conftest.py. Target ids are real
# eval_models row ids, so tests key on target.name, never on a literal id.


def _cap(canary="pass"):
    return CellCapture(
        prompt_mode="raw", k_requested=5, k_returned=1, content_offset=0,
        top_k=(TokenProb(token=" a", logprob=-0.5, token_id=1),),
        canary=canary, captured_at="2026-07-26T00:00:00Z",
    )


class FakeClient:
    """Records (snippet_text, target_name) so assertions never need an id."""

    def __init__(self, order, *, canary="pass", fail_target=None):
        self._order = order
        self._canary = canary
        self._fail_target = fail_target

    async def preflight(self, target, mode, top_k):
        return PreflightResult(state="ok", k_returned=5, canary=self._canary)

    async def capture(self, snippet, target, mode, top_k):
        self._order.append((snippet, target.name))
        if target.name == self._fail_target:
            return CellError(reason="unreachable", detail="x")
        # The real client always returns "unchecked"; turning that into the
        # real verdict is _stamp_canary's job. Baking the answer in here
        # would let a deleted stamp pass unnoticed.
        return _cap("unchecked")


@pytest.mark.asyncio
async def test_runner_fills_the_grid_row_major(db, config, targets, snippets):
    """Complete comparable rows appear while the run is still going."""
    order = []
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient(order))
    await runner.run(config, targets, snippets, task_id)

    assert order == [
        ("The protestors were", "base"), ("The protestors were", "steered"),
        ("The rioters were", "base"), ("The rioters were", "steered"),
    ]


@pytest.mark.asyncio
async def test_every_cell_is_persisted(db, config, targets, snippets):
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([]))
    outcome = await runner.run(config, targets, snippets, task_id)

    grid = load_grid(db, outcome.group_id)
    assert len(grid["cells"]) == 4


@pytest.mark.asyncio
async def test_failed_cells_are_persisted_too(db, config, targets, snippets):
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([], fail_target="steered"))
    outcome = await runner.run(config, targets, snippets, task_id)

    base, steered = targets[0].id, targets[1].id
    grid = load_grid(db, outcome.group_id)
    assert isinstance(grid["cells"][("s1", steered)], CellError)
    assert isinstance(grid["cells"][("s1", base)], CellCapture)


@pytest.mark.asyncio
async def test_progress_reports_group_level_totals(db, config, targets, snippets):
    seen = []
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([]))
    await runner.run(config, targets, snippets, task_id,
                     progress=lambda done, total: seen.append((done, total)))

    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)], (
        "progress must be reported once per cell, not just once at the end"
    )


@pytest.mark.asyncio
async def test_a_raising_progress_callback_does_not_strand_the_run(
    db, config, targets, snippets
):
    """A broken UI-supplied progress callback is not a cancellation -- it
    must not propagate out of run() and leave every eval_runs row sitting
    at "running" forever. This is the same failure class the
    asyncio.CancelledError handler in sample_bench.create_and_run_sample_
    bench exists to close, just for a different exception type; the fix
    belongs at the call site in runner.py, not in that handler, so this
    test pins it directly against the runner."""
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([]))

    def broken_progress(done, total):
        raise RuntimeError("boom")

    outcome = await runner.run(
        config, targets, snippets, task_id, progress=broken_progress
    )

    grid = load_grid(db, outcome.group_id)
    assert len(grid["cells"]) == 4, "the run itself must still complete"

    runs = db.list_runs(run_group_id=outcome.group_id)
    assert len(runs) == len(targets)
    for run in runs:
        assert run["status"] == "completed", (
            "a throwing progress callback must not strand the run row at "
            "'running'"
        )


@pytest.mark.asyncio
async def test_cancel_stops_the_run_and_keeps_completed_cells(db, config, targets, snippets):
    token = CancelToken()
    order = []

    class CancellingClient(FakeClient):
        async def capture(self, snippet, target, mode, top_k):
            result = await super().capture(snippet, target, mode, top_k)
            if len(order) == 2:
                token.cancel()
            return result

    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: CancellingClient(order))
    outcome = await runner.run(config, targets, snippets, task_id, cancel_token=token)

    grid = load_grid(db, outcome.group_id)
    assert len(grid["cells"]) == 2, "a cancelled run is a real, partial measurement"


@pytest.mark.asyncio
async def test_cancelled_run_rows_read_cancelled_not_pending(db, config, targets, snippets):
    """Without an explicit status transition, every eval_runs row created by
    create_run_group sits at its 'pending' default forever -- indistinguishable
    from a run that hasn't started. A cancelled run group must read
    'cancelled' on every one of its run rows, with end_time set."""
    token = CancelToken()
    order = []

    class CancellingClient(FakeClient):
        async def capture(self, snippet, target, mode, top_k):
            result = await super().capture(snippet, target, mode, top_k)
            if len(order) == 1:
                token.cancel()
            return result

    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: CancellingClient(order))
    outcome = await runner.run(config, targets, snippets, task_id, cancel_token=token)

    runs = db.list_runs(run_group_id=outcome.group_id)
    assert len(runs) == len(targets)
    for run in runs:
        assert run["status"] == "cancelled"
        assert run["end_time"] is not None


@pytest.mark.asyncio
async def test_completed_run_rows_read_completed(db, config, targets, snippets):
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([]))
    outcome = await runner.run(config, targets, snippets, task_id)

    runs = db.list_runs(run_group_id=outcome.group_id)
    assert len(runs) == len(targets)
    for run in runs:
        assert run["status"] == "completed"
        assert run["end_time"] is not None


@pytest.mark.asyncio
async def test_degenerate_canary_propagates_onto_every_cell(db, config, targets, snippets):
    """The preflight warning must not be lost between preflight and grid."""
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([], canary="degenerate"))
    outcome = await runner.run(config, targets, snippets, task_id)

    grid = load_grid(db, outcome.group_id)
    assert all(c.canary == "degenerate" for c in grid["cells"].values())


@pytest.mark.asyncio
async def test_canary_pass_verdict_is_also_stamped_onto_every_cell(db, config, targets, snippets):
    """The stamp is unconditional: a verified target's cells must say 'pass',
    not the client's placeholder 'unchecked'."""
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([], canary="pass"))
    outcome = await runner.run(config, targets, snippets, task_id)

    grid = load_grid(db, outcome.group_id)
    assert all(c.canary == "pass" for c in grid["cells"].values())


@pytest.mark.asyncio
async def test_run_returns_preflight_results_per_target(db, config, targets, snippets):
    """PR 3 renders readiness from these; re-running preflight could disagree."""
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([], canary="degenerate"))
    outcome = await runner.run(config, targets, snippets, task_id)

    assert outcome.group_id
    assert set(outcome.preflight) == {t.id for t in targets}
    for result in outcome.preflight.values():
        assert result.state == "ok"
        assert result.canary == "degenerate"
        assert result.is_warned is True


@pytest.mark.asyncio
async def test_targets_invalid_for_the_mode_are_rejected_before_any_call(
    db, config, snippets
):
    bad = Target(id="unused", name="c", provider="p", model_id="m", system_prompt="x")
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([]))

    with pytest.raises(ValueError, match="raw"):
        await runner.run(config, [bad], snippets, task_id)


# --- TASK-707: concurrency ---------------------------------------------


@pytest.mark.asyncio
async def test_explicit_concurrency_of_one_still_produces_strict_row_major_order(
    db, dataset, targets, snippets
):
    """concurrency=1 must behave identically to the original sequential
    runner -- exact call order, not just eventual correctness. Pinned
    explicitly (rather than relying on BenchConfig's default) per
    TASK-707's acceptance criteria."""
    config1 = BenchConfig(
        name="loaded-nouns v1", prompt_mode="raw", top_k=20,
        dataset_id=dataset, target_ids=tuple(t.id for t in targets),
        concurrency=1,
    )
    order = []
    task_id = save_bench(db, config1)
    runner = WordBenchRunner(db, lambda t: FakeClient(order))
    await runner.run(config1, targets, snippets, task_id)

    assert order == [
        ("The protestors were", "base"), ("The protestors were", "steered"),
        ("The rioters were", "base"), ("The rioters were", "steered"),
    ]


@pytest.mark.asyncio
async def test_concurrency_above_one_runs_a_row_in_parallel_bounded_by_the_setting(
    db, dataset, targets, snippets
):
    """The whole point of TASK-707: concurrency > 1 must actually overlap
    in-flight requests (bounded by the configured value), and must never
    let two DIFFERENT rows be in flight at once -- a row is either fully
    captured or not yet started. If the runner silently ignores
    `concurrency` and stays sequential, `max_concurrent` never rises above
    1 and this test fails."""
    extra_id = db.create_model(name="extra", provider="llama_cpp", model_id="m")
    all_targets = list(targets) + [
        Target(id=extra_id, name="extra", provider="llama_cpp", model_id="m")
    ]
    config3 = BenchConfig(
        name="loaded-nouns v1", prompt_mode="raw", top_k=20,
        dataset_id=dataset, target_ids=tuple(t.id for t in all_targets),
        concurrency=2,
    )

    active: dict[str, set[str]] = {}
    max_concurrent = [0]
    cross_row_overlap: list[str] = []

    class RowTrackingClient:
        async def preflight(self, target, mode, top_k):
            return PreflightResult(state="ok", k_returned=5, canary="pass")

        async def capture(self, snippet, target, mode, top_k):
            active.setdefault(snippet, set()).add(target.name)
            if len(active) > 1:
                cross_row_overlap.append(f"rows in flight at once: {sorted(active)}")
            in_flight = sum(len(v) for v in active.values())
            max_concurrent[0] = max(max_concurrent[0], in_flight)
            await asyncio.sleep(0.01)
            active[snippet].discard(target.name)
            if not active[snippet]:
                del active[snippet]
            return CellCapture(
                prompt_mode="raw", k_requested=5, k_returned=1, content_offset=0,
                top_k=(TokenProb(token=" a", logprob=-0.5, token_id=1),),
                canary="unchecked", captured_at="2026-07-26T00:00:00Z",
            )

    task_id = save_bench(db, config3)
    runner = WordBenchRunner(db, lambda t: RowTrackingClient())
    outcome = await runner.run(config3, all_targets, snippets, task_id)

    assert cross_row_overlap == [], "two different rows were in flight at once"
    assert max_concurrent[0] == 2, (
        "the configured concurrency (2) was never actually reached -- "
        "captures ran sequentially despite concurrency > 1"
    )
    grid = load_grid(db, outcome.group_id)
    assert len(grid["cells"]) == len(snippets) * len(all_targets)


@pytest.mark.asyncio
async def test_concurrency_above_one_saves_a_row_in_target_order_regardless_of_completion_order(
    db, dataset, targets, snippets
):
    """asyncio.gather returns results in submission order, not completion
    order -- the slower target here finishes AFTER the faster one, but
    progress/save must still process the row in `targets` list order."""
    config2 = BenchConfig(
        name="loaded-nouns v1", prompt_mode="raw", top_k=20,
        dataset_id=dataset, target_ids=tuple(t.id for t in targets),
        concurrency=2,
    )

    class SkewedDelayClient:
        def __init__(self, delay: float) -> None:
            self._delay = delay

        async def preflight(self, target, mode, top_k):
            return PreflightResult(state="ok", k_returned=5, canary="pass")

        async def capture(self, snippet, target, mode, top_k):
            await asyncio.sleep(self._delay)
            return CellCapture(
                prompt_mode="raw", k_requested=5, k_returned=1, content_offset=0,
                top_k=(TokenProb(token=" a", logprob=-0.5, token_id=1),),
                canary="unchecked", captured_at="2026-07-26T00:00:00Z",
            )

    # targets[0] ("base") is deliberately the SLOWER of the two, so it
    # completes after targets[1] ("steered") despite being listed, and
    # dispatched, first.
    delays = {targets[0].id: 0.03, targets[1].id: 0.005}
    task_id = save_bench(db, config2)
    runner = WordBenchRunner(db, lambda t: SkewedDelayClient(delays[t.id]))
    progress_order: list[int] = []
    outcome = await runner.run(
        config2, targets, snippets, task_id,
        progress=lambda done, total: progress_order.append(done),
    )

    assert progress_order == [1, 2, 3, 4], (
        "progress must advance strictly 1..N even though the faster target "
        "completed its network call first"
    )
    grid = load_grid(db, outcome.group_id)
    assert len(grid["cells"]) == 4


@pytest.mark.asyncio
async def test_concurrency_field_must_be_at_least_one(targets, dataset):
    with pytest.raises(ValueError, match="concurrency"):
        BenchConfig(
            name="x", prompt_mode="raw", top_k=5,
            dataset_id=dataset, target_ids=tuple(t.id for t in targets),
            concurrency=0,
        )


@pytest.mark.asyncio
async def test_cancel_token_stops_a_concurrent_run_between_rows_without_stranding_rows(
    db, dataset, targets, snippets
):
    """The cooperative CancelToken must still work under concurrency > 1 --
    checked once per row rather than once per cell (a row already
    dispatched is allowed to finish), but no run row may be left at
    "running"."""
    config2 = BenchConfig(
        name="loaded-nouns v1", prompt_mode="raw", top_k=20,
        dataset_id=dataset, target_ids=tuple(t.id for t in targets),
        concurrency=2,
    )
    token = CancelToken()
    seen_snippets: list[str] = []

    class CancellingClient:
        async def preflight(self, target, mode, top_k):
            return PreflightResult(state="ok", k_returned=5, canary="pass")

        async def capture(self, snippet, target, mode, top_k):
            seen_snippets.append(snippet)
            if len(seen_snippets) == 2:
                token.cancel()
            return CellCapture(
                prompt_mode="raw", k_requested=5, k_returned=1, content_offset=0,
                top_k=(TokenProb(token=" a", logprob=-0.5, token_id=1),),
                canary="unchecked", captured_at="2026-07-26T00:00:00Z",
            )

    task_id = save_bench(db, config2)
    runner = WordBenchRunner(db, lambda t: CancellingClient())
    outcome = await runner.run(config2, targets, snippets, task_id, cancel_token=token)

    grid = load_grid(db, outcome.group_id)
    assert len(grid["cells"]) == 2, "row 1 completes; row 2 must not start"
    runs = db.list_runs(run_group_id=outcome.group_id)
    assert len(runs) == len(targets)
    for run in runs:
        assert run["status"] == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2])
async def test_external_task_cancellation_marks_rows_cancelled_and_reraises(
    db, dataset, targets, snippets, concurrency
):
    """A HARD cancellation -- the Task running .run() itself cancelled, e.g.
    by a Textual exclusive=True worker superseding an in-flight run, as
    opposed to the cooperative CancelToken -- must not strand any run row
    at "running" either, at any concurrency. asyncio.CancelledError is a
    BaseException; the runner must catch it, mark rows cancelled, and
    re-raise (never swallow it)."""
    config_n = BenchConfig(
        name="loaded-nouns v1", prompt_mode="raw", top_k=20,
        dataset_id=dataset, target_ids=tuple(t.id for t in targets),
        concurrency=concurrency,
    )
    gate = asyncio.Event()

    class BlockingClient:
        async def preflight(self, target, mode, top_k):
            return PreflightResult(state="ok", k_returned=5, canary="pass")

        async def capture(self, snippet, target, mode, top_k):
            gate.set()
            await asyncio.sleep(10)
            raise AssertionError("should have been cancelled before this returned")

    task_id = save_bench(db, config_n)
    runner = WordBenchRunner(db, lambda t: BlockingClient())

    run_task = asyncio.ensure_future(
        runner.run(config_n, targets, snippets, task_id)
    )
    await gate.wait()
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    runs = db.list_runs(task_id=task_id, limit=100)
    assert runs
    assert all(run["status"] == "cancelled" for run in runs)
    assert not any(run["status"] == "running" for run in runs)


# --- TASK-709: capture client lifecycle (runner-side wiring) ------------


class _ClosableFakeClient(FakeClient):
    """A FakeClient that also holds a resource, mirroring
    WordBenchCaptureClient's real aclose(). CaptureClientLike does not
    require aclose() -- every other fake in this file has none -- so the
    runner's cleanup must be duck-typed, not assume every client has one."""

    def __init__(self, order, closed, **kwargs):
        super().__init__(order, **kwargs)
        self._closed = closed

    async def aclose(self):
        self._closed.append(self)


@pytest.mark.asyncio
async def test_run_closes_every_client_it_created(db, config, targets, snippets):
    """WordBenchRunner.run must release any resources its clients hold
    (e.g. WordBenchCaptureClient's pooled httpx.AsyncClient) once the run
    is done."""
    closed: list = []
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: _ClosableFakeClient([], closed))
    await runner.run(config, targets, snippets, task_id)

    assert len(closed) == len(targets), "every client the run created must be closed exactly once"


@pytest.mark.asyncio
async def test_run_closes_clients_even_when_cooperatively_cancelled(
    db, config, targets, snippets
):
    token = CancelToken()
    closed: list = []
    order: list = []

    class CancellingClosableClient(_ClosableFakeClient):
        async def capture(self, snippet, target, mode, top_k):
            result = await super().capture(snippet, target, mode, top_k)
            if len(order) == 2:
                token.cancel()
            return result

    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: CancellingClosableClient(order, closed))
    await runner.run(config, targets, snippets, task_id, cancel_token=token)

    assert len(closed) == len(targets)


@pytest.mark.asyncio
async def test_run_closes_clients_even_when_hard_cancelled(
    db, dataset, targets, snippets
):
    config1 = BenchConfig(
        name="loaded-nouns v1", prompt_mode="raw", top_k=20,
        dataset_id=dataset, target_ids=tuple(t.id for t in targets),
        concurrency=1,
    )
    gate = asyncio.Event()
    closed: list = []

    class BlockingClosableClient:
        async def preflight(self, target, mode, top_k):
            return PreflightResult(state="ok", k_returned=5, canary="pass")

        async def capture(self, snippet, target, mode, top_k):
            gate.set()
            await asyncio.sleep(10)
            raise AssertionError("should have been cancelled before this returned")

        async def aclose(self):
            closed.append(self)

    task_id = save_bench(db, config1)
    runner = WordBenchRunner(db, lambda t: BlockingClosableClient())

    run_task = asyncio.ensure_future(
        runner.run(config1, targets, snippets, task_id)
    )
    await gate.wait()
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert len(closed) == len(targets)
