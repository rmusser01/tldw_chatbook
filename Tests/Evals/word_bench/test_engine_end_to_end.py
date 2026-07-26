"""The engine's deliverable: a correct grid, end to end, with no UI."""

from __future__ import annotations

import math

import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.analysis import divergence, group_means, spread
from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig, CellCapture, PreflightResult, Snippet, Target, TokenProb,
)
from tldw_chatbook.Evals.word_bench.runner import WordBenchRunner
from tldw_chatbook.Evals.word_bench.storage import load_grid, save_bench


# db, snippets, targets, config come from conftest.py.

#: "steered" diverges from "base" only on the loaded snippet -- the shape a
#: real finding has. Keyed on target NAME: ids are database-assigned.
SCRIPT = {
    ("The protestors were", "base"):    [(" a", 0.7), (" the", 0.3)],
    ("The protestors were", "steered"): [(" a", 0.7), (" the", 0.3)],
    ("The rioters were", "base"):       [(" a", 0.7), (" the", 0.3)],
    ("The rioters were", "steered"):    [(" not", 0.8), (" a", 0.2)],
}


class ScriptedClient:
    def __init__(self, target):
        self._target = target

    async def preflight(self, target, mode, top_k):
        return PreflightResult(state="ok", k_returned=top_k, canary="pass")

    async def capture(self, snippet, target, mode, top_k):
        pairs = SCRIPT[(snippet, target.name)]
        return CellCapture(
            prompt_mode=mode, k_requested=top_k, k_returned=len(pairs),
            content_offset=0,
            top_k=tuple(
                TokenProb(token=t, logprob=math.log(p), token_id=i)
                for i, (t, p) in enumerate(pairs)
            ),
            canary="pass", captured_at="2026-07-26T00:00:00Z",
        )


@pytest.mark.asyncio
async def test_engine_produces_a_grid_whose_divergence_finds_the_steered_cell(
    db, config, targets, snippets
):
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, ScriptedClient)
    group_id = await runner.run(config, targets, snippets, task_id)

    grid = load_grid(db, group_id)
    assert len(grid["cells"]) == 4

    base, steered = targets[0].id, targets[1].id
    # Column baseline: base. Only the loaded snippet moved.
    neutral, _ = divergence(grid["cells"][("s1", base)], grid["cells"][("s1", steered)])
    loaded, _ = divergence(grid["cells"][("s2", base)], grid["cells"][("s2", steered)])

    assert neutral == pytest.approx(0.0, abs=1e-9)
    assert loaded > 0.3
    assert loaded > neutral

    # Group means are the headline number for a control/treatment set.
    by_group = {s.id: s.group for s in snippets}
    means = group_means([
        (by_group["s1"], neutral),
        (by_group["s2"], loaded),
    ])
    assert means["loaded"] > means["neutral"]


@pytest.mark.asyncio
async def test_spread_identifies_the_row_where_targets_disagree(
    db, config, targets, snippets
):
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, ScriptedClient)
    group_id = await runner.run(config, targets, snippets, task_id)
    grid = load_grid(db, group_id)

    s1 = spread([grid["cells"][("s1", t.id)] for t in targets])
    s2 = spread([grid["cells"][("s2", t.id)] for t in targets])
    assert s2 > s1


@pytest.mark.asyncio
async def test_grid_survives_the_bench_being_edited_afterwards(
    db, config, targets, snippets
):
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, ScriptedClient)
    group_id = await runner.run(config, targets, snippets, task_id)

    save_bench(db, BenchConfig(name="renamed", prompt_mode="chat", top_k=99,
                               dataset_id="d", target_ids=(targets[0].id,)),
               task_id=task_id)

    grid = load_grid(db, group_id)
    assert grid["snapshot"]["prompt_mode"] == "raw"
    assert grid["snapshot"]["top_k"] == 20
    assert len(grid["cells"]) == 4
