"""Grid execution: order, progress, cancel, and preflight propagation."""

from __future__ import annotations

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
    group_id = await runner.run(config, targets, snippets, task_id)

    grid = load_grid(db, group_id)
    assert len(grid["cells"]) == 4


@pytest.mark.asyncio
async def test_failed_cells_are_persisted_too(db, config, targets, snippets):
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([], fail_target="steered"))
    group_id = await runner.run(config, targets, snippets, task_id)

    base, steered = targets[0].id, targets[1].id
    grid = load_grid(db, group_id)
    assert isinstance(grid["cells"][("s1", steered)], CellError)
    assert isinstance(grid["cells"][("s1", base)], CellCapture)


@pytest.mark.asyncio
async def test_progress_reports_group_level_totals(db, config, targets, snippets):
    seen = []
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([]))
    await runner.run(config, targets, snippets, task_id,
                     progress=lambda done, total: seen.append((done, total)))

    assert seen[-1] == (4, 4), "progress is over the whole grid, not per run"


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
    group_id = await runner.run(config, targets, snippets, task_id, cancel_token=token)

    grid = load_grid(db, group_id)
    assert len(grid["cells"]) == 2, "a cancelled run is a real, partial measurement"


@pytest.mark.asyncio
async def test_degenerate_canary_propagates_onto_every_cell(db, config, targets, snippets):
    """The preflight warning must not be lost between preflight and grid."""
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([], canary="degenerate"))
    group_id = await runner.run(config, targets, snippets, task_id)

    grid = load_grid(db, group_id)
    assert all(c.canary == "degenerate" for c in grid["cells"].values())


@pytest.mark.asyncio
async def test_canary_pass_verdict_is_also_stamped_onto_every_cell(db, config, targets, snippets):
    """The stamp is unconditional: a verified target's cells must say 'pass',
    not the client's placeholder 'unchecked'."""
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([], canary="pass"))
    group_id = await runner.run(config, targets, snippets, task_id)

    grid = load_grid(db, group_id)
    assert all(c.canary == "pass" for c in grid["cells"].values())


@pytest.mark.asyncio
async def test_targets_invalid_for_the_mode_are_rejected_before_any_call(
    db, config, snippets
):
    bad = Target(id="unused", name="c", provider="p", model_id="m", system_prompt="x")
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([]))

    with pytest.raises(ValueError, match="raw"):
        await runner.run(config, [bad], snippets, task_id)
