"""Results grid and lenses (PR 3b, Task 1).

Selecting a run group renders a pivoted grid -- rows are snippets, columns
are targets -- and a lens decides what a cell shows. Three renderings would
misrepresent the engine, and each has a dedicated test below: a bare Top-1
winner on a near-tie, a leading "≥" on a Δ baseline divergence, and entropy
computed without a shared K. See ``results_grid.py``'s own module docstring
for the underlying evidence (the observed rank-flip fixture, PR 2's
disproved lower-bound claim).

Mirrors ``test_evals_screen.py``/``test_evals_bench_editor.py``'s harness
(bundled CSS, a fake ``app_instance`` exposing
``evaluation_orchestrator.db``) rather than inventing a second one.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import DataTable, Select, Static

import tldw_chatbook
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench import analysis
from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig,
    CellCapture,
    CellError,
    PreflightResult,
    Snippet,
    Target,
    TokenProb,
)
from tldw_chatbook.Evals.word_bench.storage import create_run_group, save_bench, save_cell
from tldw_chatbook.UI.Evals.results_grid import (
    FAILED_MARK,
    ResultsGrid,
    degenerate_canary_text,
    render_probe_reading,
    render_token,
)
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus

_BUNDLED_CSS_PATH = str(
    Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
)


class _FakeOrchestrator:
    def __init__(self, db: EvalsDB) -> None:
        self.db = db


class _FakeAppInstance:
    def __init__(self, db: EvalsDB) -> None:
        self.evaluation_orchestrator = _FakeOrchestrator(db)
        self.notifications: list[tuple[str, str]] = []

    def notify(self, message: str, *, severity: str = "information", **kwargs) -> None:
        self.notifications.append((message, severity))


class EvalsHarness(App):
    CSS_PATH = _BUNDLED_CSS_PATH

    def __init__(self, app_instance: _FakeAppInstance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(EvalsScreen(self.app_instance))


@pytest.fixture
def evals_db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def evals_app(evals_db: EvalsDB) -> EvalsHarness:
    return EvalsHarness(_FakeAppInstance(evals_db))


def _cap(pairs: list[tuple[str, float]], *, k_returned: int | None = None) -> CellCapture:
    """Mirrors ``Tests/Evals/word_bench/test_analysis.py``'s own ``_cap``
    helper -- probabilities, converted to logprobs the same way the engine
    stores them."""
    top = tuple(
        TokenProb(token=t, logprob=math.log(p), bytes_=tuple(t.encode("utf-8")), token_id=i)
        for i, (t, p) in enumerate(pairs)
    )
    return CellCapture(
        prompt_mode="raw",
        k_requested=len(top),
        k_returned=k_returned if k_returned is not None else len(top),
        content_offset=0,
        top_k=top,
        canary="pass",
        captured_at="2026-07-26T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Fixture: one run group covering near-tie / clear-winner / failed / unrun /
# warned-column in a single small grid.
#
#   snippet \ target   base (t1)              steered (t2, WARNED)
#   s1 (neutral)        near-tie (.44/.43)     clear winner (.9/.05)
#   s2 (loaded)          FAILED (unreachable)   captured
#   s3 (loaded)          UNRUN                  captured
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_run_group(evals_db: EvalsDB) -> dict:
    base_id = evals_db.create_model(name="base", provider="llama_cpp", model_id="m")
    steered_id = evals_db.create_model(name="steered", provider="llama_cpp", model_id="m")
    dataset_id = evals_db.create_dataset(
        name="loaded-nouns", format="custom", source_path="inline:loaded-nouns"
    )
    config = BenchConfig(
        name="loaded-nouns v1",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(base_id, steered_id),
        probes=(" a",),
    )
    task_id = save_bench(evals_db, config)
    targets = [
        Target(id=base_id, name="base", provider="llama_cpp", model_id="m"),
        Target(id=steered_id, name="steered", provider="llama_cpp", model_id="m"),
    ]
    snippets = [
        Snippet(id="s1", text="The protestors were", group="neutral"),
        Snippet(id="s2", text="The rioters were", group="loaded"),
        Snippet(id="s3", text="The regime said", group="loaded"),
    ]
    preflight = {
        base_id: PreflightResult(state="ok", k_returned=20, canary="pass"),
        steered_id: PreflightResult(state="ok", k_returned=20, canary="degenerate"),
    }
    group_id, run_ids = create_run_group(
        evals_db, task_id, config, targets, snippets, preflight=preflight
    )

    # s1 x base: near-tie (gap << NEAR_TIE_LOGPROB_GAP_NATS).
    save_cell(
        evals_db, run_ids[base_id], snippets[0],
        _cap([(" a", 0.44), (" the", 0.43), (" an", 0.05)]),
    )
    # s1 x steered: clear winner (gap >> NEAR_TIE_LOGPROB_GAP_NATS).
    save_cell(
        evals_db, run_ids[steered_id], snippets[0],
        _cap([(" a", 0.9), (" the", 0.05)]),
    )
    # s2 x base: failed.
    save_cell(
        evals_db, run_ids[base_id], snippets[1],
        CellError(reason="unreachable", detail="connection refused"),
    )
    # s2 x steered: captured normally.
    save_cell(
        evals_db, run_ids[steered_id], snippets[1],
        _cap([(" a", 0.5), (" not", 0.3)]),
    )
    # s3 x base: deliberately never saved -- unrun.
    # s3 x steered: captured normally.
    save_cell(
        evals_db, run_ids[steered_id], snippets[2],
        _cap([(" it", 0.6), (" the", 0.2)]),
    )

    return {
        "group_id": group_id,
        "base_id": base_id,
        "steered_id": steered_id,
        "s1": "s1", "s2": "s2", "s3": "s3",
    }


# ---------------------------------------------------------------------------
# Fixture: a clean, all-captured 2-target x 3-snippet grid (no failures/
# unrun cells) dedicated to spread/group-mean and mixed-K entropy tests,
# where every cell must be present for the arithmetic to be predictable.
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_run_group(evals_db: EvalsDB) -> dict:
    base_id = evals_db.create_model(name="llama-3-8b", provider="llama_cpp", model_id="m")
    # k_returned=5 mirrors an OpenAI-legacy-style target capped below the
    # requested top_k, so effective_k must come out to 5, not 20.
    poor_id = evals_db.create_model(name="capped-target", provider="openai", model_id="m2")
    dataset_id = evals_db.create_dataset(
        name="clean-set", format="custom", source_path="inline:clean-set"
    )
    config = BenchConfig(
        name="clean bench", prompt_mode="raw", top_k=20,
        dataset_id=dataset_id, target_ids=(base_id, poor_id),
    )
    task_id = save_bench(evals_db, config)
    targets = [
        Target(id=base_id, name="llama-3-8b", provider="llama_cpp", model_id="m"),
        Target(id=poor_id, name="capped-target", provider="openai", model_id="m2"),
    ]
    snippets = [
        Snippet(id="s1", text="the protestors were", group="neutral"),
        Snippet(id="s2", text="the rioters were", group="loaded"),
        Snippet(id="s3", text="the regime said", group="loaded"),
        # Ungrouped deliberately -- keeps it out of the "loaded" group-mean
        # aggregate other tests assert on. Its only job is a combined
        # truncated mass (0.65) that clears TRUNCATION_WARN_THRESHOLD
        # (0.25), which no OTHER row in this fixture does (0.04/0.20/0.25/
        # 0.18 individually, none of the real pairs combine past 0.25
        # either) -- see test_delta_lens_flags_high_combined_truncation_
        # with_a_bang_marker.
        Snippet(id="s4", text="the report concluded", group=None),
    ]
    group_id, run_ids = create_run_group(evals_db, task_id, config, targets, snippets)

    # Same underlying distribution over the first 5 ranks at both K=20
    # (base) and K=5 (poor) for s1 -- but base's top_k has 15 MORE real
    # tokens beyond rank 5 (a genuinely richer native K), so entropy over
    # base's own full native top_k measurably DIFFERS from entropy at the
    # shared effective K (5). A version of this fixture where "rich" only
    # ever actually HAD 5 tokens (regardless of its claimed k_returned)
    # would pass this test even with `k=effective_k` deleted from
    # results_grid.py's entropy call -- caught in review: the engine
    # (analysis.entropy) confirmed 1.2712 in every mode (native, shared,
    # no-k) for that degenerate fixture, an inert assertion.
    same_dist = [(" a", 0.5), (" the", 0.3), (" an", 0.1), (" some", 0.05), (" one", 0.03)]
    rich_tail = [(f"_extra_{i}", 0.001) for i in range(15)]  # +15 real low-prob tokens
    save_cell(
        evals_db, run_ids[base_id], snippets[0],
        _cap(same_dist + rich_tail, k_returned=20),
    )
    save_cell(evals_db, run_ids[poor_id], snippets[0], _cap(same_dist, k_returned=5))

    # s2: base diverges a lot from poor; s3: base diverges a little.
    save_cell(evals_db, run_ids[base_id], snippets[1], _cap([(" a", 0.9)], k_returned=20))
    save_cell(evals_db, run_ids[poor_id], snippets[1], _cap([(" not", 0.9)], k_returned=5))
    save_cell(evals_db, run_ids[base_id], snippets[2], _cap([(" it", 0.6), (" the", 0.3)], k_returned=20))
    save_cell(evals_db, run_ids[poor_id], snippets[2], _cap([(" it", 0.55), (" the", 0.3)], k_returned=5))

    # s4: same rich/capped K convention as s1 (native 20 vs. native 5, so
    # the grid-wide effective K stays 5) -- but each cell's mass is
    # dominated by ONE high-but-not-total-probability token (0.7/0.65)
    # plus small filler tokens, giving individual truncated_mass 0.281/
    # 0.346 (own truncated_mass; the divergence()/combined_truncation()
    # figure, at the shared k=5, is 0.642 -- neither figure alone clears
    # TRUNCATION_WARN_THRESHOLD (0.25) by a trivial 1-token cap; both
    # individual masses here DO already exceed it on their own, which is
    # fine -- the point is the WIRING (grid -> combined_truncation), not
    # reproducing test_analysis.py's "neither alone, only combined" case.
    # Computed against the real engine: jsd=0.4716 (renders "0.47"),
    # is_bounded=True, combined_truncation=0.642 (renders "64.2%").
    base_pairs = [(" a", 0.7)] + [(f"_extra_{i}", 0.001) for i in range(19)]
    poor_pairs = [(" b", 0.65)] + [(f"_fill_{i}", 0.001) for i in range(4)]
    save_cell(evals_db, run_ids[base_id], snippets[3], _cap(base_pairs, k_returned=20))
    save_cell(evals_db, run_ids[poor_id], snippets[3], _cap(poor_pairs, k_returned=5))

    return {"group_id": group_id, "base_id": base_id, "poor_id": poor_id}


# ---------------------------------------------------------------------------
# Fixtures for TASK-1036: the run view's degenerate-canary callout.
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_warned_run_group(evals_db: EvalsDB) -> dict:
    """Three targets, one clean and TWO warned -- the callout must name
    both warned targets by name, not collapse them to "a target" (see
    degenerate_canary_text's docstring: an unnamed warning is not
    actionable on a bench with several targets) and must not name the
    clean one.
    """
    clean_id = evals_db.create_model(name="clean-target", provider="llama_cpp", model_id="m")
    steered_id = evals_db.create_model(name="steered", provider="llama_cpp", model_id="m")
    distilled_id = evals_db.create_model(name="distilled", provider="llama_cpp", model_id="m")
    dataset_id = evals_db.create_dataset(
        name="multi-warn-set", format="custom", source_path="inline:multi-warn-set"
    )
    config = BenchConfig(
        name="multi-warn bench", prompt_mode="raw", top_k=20,
        dataset_id=dataset_id, target_ids=(clean_id, steered_id, distilled_id),
    )
    task_id = save_bench(evals_db, config)
    targets = [
        Target(id=clean_id, name="clean-target", provider="llama_cpp", model_id="m"),
        Target(id=steered_id, name="steered", provider="llama_cpp", model_id="m"),
        Target(id=distilled_id, name="distilled", provider="llama_cpp", model_id="m"),
    ]
    snippets = [Snippet(id="s1", text="The protestors were", group=None)]
    preflight = {
        clean_id: PreflightResult(state="ok", k_returned=20, canary="pass"),
        steered_id: PreflightResult(state="ok", k_returned=20, canary="degenerate"),
        distilled_id: PreflightResult(state="ok", k_returned=20, canary="degenerate"),
    }
    group_id, run_ids = create_run_group(
        evals_db, task_id, config, targets, snippets, preflight=preflight
    )
    save_cell(evals_db, run_ids[clean_id], snippets[0], _cap([(" a", 0.9), (" the", 0.05)]))
    save_cell(
        evals_db, run_ids[steered_id], snippets[0],
        _cap([(" mente", 0.49), (" the", 0.2)]),
    )
    save_cell(
        evals_db, run_ids[distilled_id], snippets[0],
        _cap([(" xyzzy", 0.4), (" the", 0.3)]),
    )
    return {
        "group_id": group_id,
        "clean_id": clean_id,
        "steered_id": steered_id,
        "distilled_id": distilled_id,
    }


@pytest.fixture
def warned_markup_hazard_run_group(evals_db: EvalsDB) -> dict:
    """A single warned target whose NAME itself contains Rich markup
    syntax ("[...]") -- the same hazard ``markup_hazard_run_group`` above
    covers for snippet text, but for the canary callout's target-name
    interpolation specifically. The callout is a ``Static``, not a
    ``DataTable`` cell, so the guard under test is ``markup=False``
    (results_grid.py's ``compose()``), not ``_safe_cell``'s ``Text()``
    wrapping.
    """
    steered_id = evals_db.create_model(
        name="steered [redacted]", provider="llama_cpp", model_id="m"
    )
    dataset_id = evals_db.create_dataset(
        name="hazard-canary-set", format="custom", source_path="inline:hazard-canary-set"
    )
    config = BenchConfig(
        name="hazard canary bench", prompt_mode="raw", top_k=5,
        dataset_id=dataset_id, target_ids=(steered_id,),
    )
    task_id = save_bench(evals_db, config)
    targets = [
        Target(id=steered_id, name="steered [redacted]", provider="llama_cpp", model_id="m")
    ]
    snippets = [Snippet(id="s1", text="The protestors were", group=None)]
    preflight = {steered_id: PreflightResult(state="ok", k_returned=20, canary="degenerate")}
    group_id, run_ids = create_run_group(
        evals_db, task_id, config, targets, snippets, preflight=preflight
    )
    save_cell(
        evals_db, run_ids[steered_id], snippets[0],
        _cap([(" mente", 0.49), (" the", 0.2)]),
    )
    return {"group_id": group_id, "steered_id": steered_id}


# ---------------------------------------------------------------------------
# Fixtures for TASK-1477: the run-level failure callout.
# ---------------------------------------------------------------------------


def _make_two_by_two_run_group(
    evals_db: EvalsDB, name: str
) -> tuple[str, str, str, list, dict]:
    """Shared 2-target x 2-snippet scaffolding for the failure-callout
    fixtures below -- both need the identical 4-cell shape, differing only
    in which cells are ``CellError`` vs ``CellCapture``, so the run/target/
    snippet setup is factored out rather than duplicated twice."""
    base_id = evals_db.create_model(name="base", provider="llama_cpp", model_id="m")
    steered_id = evals_db.create_model(name="steered", provider="llama_cpp", model_id="m")
    dataset_id = evals_db.create_dataset(
        name=f"{name}-set", format="custom", source_path=f"inline:{name}-set"
    )
    config = BenchConfig(
        name=f"{name} bench", prompt_mode="raw", top_k=20,
        dataset_id=dataset_id, target_ids=(base_id, steered_id),
    )
    task_id = save_bench(evals_db, config)
    targets = [
        Target(id=base_id, name="base", provider="llama_cpp", model_id="m"),
        Target(id=steered_id, name="steered", provider="llama_cpp", model_id="m"),
    ]
    snippets = [
        Snippet(id="s1", text="The protestors were", group=None),
        Snippet(id="s2", text="The regime said", group=None),
    ]
    group_id, run_ids = create_run_group(evals_db, task_id, config, targets, snippets)
    return group_id, base_id, steered_id, snippets, run_ids


def _make_single_target_run_group(
    evals_db: EvalsDB, name: str, snippet_count: int
) -> tuple[str, str, list, str]:
    """A SINGLE target, ``snippet_count`` snippets, one run -- the
    narrowest shape that lets a test control exactly which order
    ``load_grid`` inserts cells into its ``cells`` dict (and therefore
    which ``CellError.reason`` ``_failure_summary`` sees "first"), used by
    the dominant-reason tie-break test below.

    Two DB-layer ordering facts make a MULTI-target fixture unsuitable for
    that test: ``_load_run_group_snapshot`` explicitly notes
    ``list_runs`` is newest-first, so ``load_grid``'s ``for run in runs``
    loop visits the LAST-created target's run FIRST -- the opposite of
    ``targets`` construction order; and ``EvalsDB.get_run_results`` orders
    each run's own rows ``ORDER BY created_at ASC``. With a single run,
    only the second fact applies, so ``cells`` insertion order is exactly
    this fixture's ``save_cell`` call order -- no target-interleaving to
    reason about.
    """
    target_id = evals_db.create_model(name=f"{name}-base", provider="llama_cpp", model_id="m")
    dataset_id = evals_db.create_dataset(
        name=f"{name}-set", format="custom", source_path=f"inline:{name}-set"
    )
    config = BenchConfig(
        name=f"{name} bench", prompt_mode="raw", top_k=20,
        dataset_id=dataset_id, target_ids=(target_id,),
    )
    task_id = save_bench(evals_db, config)
    targets = [Target(id=target_id, name=f"{name}-base", provider="llama_cpp", model_id="m")]
    snippets = [
        Snippet(id=f"s{i}", text=f"snippet {i}", group=None)
        for i in range(1, snippet_count + 1)
    ]
    group_id, run_ids = create_run_group(evals_db, task_id, config, targets, snippets)
    return group_id, target_id, snippets, run_ids[target_id]


@pytest.fixture
def all_cells_failed_run_group(evals_db: EvalsDB) -> dict:
    """2 targets x 2 snippets, EVERY cell failed with the same reason --
    the "run is otherwise unusable, name a concrete next step" case."""
    group_id, base_id, steered_id, snippets, run_ids = _make_two_by_two_run_group(
        evals_db, "all-failed"
    )
    for snippet in snippets:
        for target_id in (base_id, steered_id):
            save_cell(
                evals_db, run_ids[target_id], snippet,
                CellError(reason="unreachable", detail="connection refused"),
            )
    return {"group_id": group_id, "base_id": base_id, "steered_id": steered_id}


@pytest.fixture
def one_of_four_cells_failed_run_group(evals_db: EvalsDB) -> dict:
    """Same 2x2 shape as ``all_cells_failed_run_group``, but only ONE of
    the four cells failed -- the "run is still usable, state the fact with
    no next-step sentence" case."""
    group_id, base_id, steered_id, snippets, run_ids = _make_two_by_two_run_group(
        evals_db, "one-failed"
    )
    save_cell(
        evals_db, run_ids[base_id], snippets[0],
        CellError(reason="unreachable", detail="connection refused"),
    )
    save_cell(evals_db, run_ids[steered_id], snippets[0], _cap([(" a", 0.9), (" the", 0.05)]))
    save_cell(evals_db, run_ids[base_id], snippets[1], _cap([(" it", 0.6), (" the", 0.2)]))
    save_cell(evals_db, run_ids[steered_id], snippets[1], _cap([(" the", 0.5), (" a", 0.3)]))
    return {"group_id": group_id, "base_id": base_id, "steered_id": steered_id}


@pytest.fixture
def k_depth_matched_run_group(evals_db: EvalsDB) -> dict:
    """A single-snippet K=20-vs-K=5 grid where EVERY cell's ``top_k``
    length actually matches its claimed ``k_returned`` -- unlike
    ``clean_run_group``'s s2/s3 rows above, whose ``top_k`` is
    deliberately abbreviated below their claimed ``k_returned`` (a
    fixture-writing shorthand used throughout this file: list only the
    tokens whose probability matters and let the rest live in
    ``_distribution``'s implicit "other" bucket).

    ``analysis.effective_k`` (TASK-861) now clamps each cell's
    ``k_returned`` to ``len(top_k)`` before taking the cross-cell minimum
    -- a defensive guard against corrupted stored JSON, a no-op on a real
    capture where the two always agree by construction (see
    ``capture_client.py``). Mixing one of ``clean_run_group``'s
    abbreviated rows into a grid-wide ``effective_k`` computation would
    drag the WHOLE grid's effective K down to that row's tiny native
    length (1 or 2), which is exactly why this fixture exists rather than
    reusing ``clean_run_group`` for the two tests below: padding s2/s3
    out to their claimed K would change the divergence figures a dozen
    OTHER tests in this file hard-code (bang-marker flags, spread
    ordering, group means), while this fixture's single row is a direct
    copy of ``clean_run_group``'s own s1 -- already K-depth-matched by the
    original author, per its own comment -- so nothing about the asserted
    entropy/truncation numbers below changes.

    Args:
        evals_db: The EvalsDB database instance for creating models,
            datasets, and run groups.

    Returns:
        A dict with keys "group_id" (str), "base_id" (int), and "poor_id"
        (int) identifying the test run group and its two target models.
    """
    base_id = evals_db.create_model(name="llama-3-8b", provider="llama_cpp", model_id="m")
    poor_id = evals_db.create_model(name="capped-target", provider="openai", model_id="m2")
    dataset_id = evals_db.create_dataset(
        name="k-depth-set", format="custom", source_path="inline:k-depth-set"
    )
    config = BenchConfig(
        name="k-depth bench", prompt_mode="raw", top_k=20,
        dataset_id=dataset_id, target_ids=(base_id, poor_id),
    )
    task_id = save_bench(evals_db, config)
    targets = [
        Target(id=base_id, name="llama-3-8b", provider="llama_cpp", model_id="m"),
        Target(id=poor_id, name="capped-target", provider="openai", model_id="m2"),
    ]
    snippets = [Snippet(id="s1", text="the protestors were", group="neutral")]
    group_id, run_ids = create_run_group(evals_db, task_id, config, targets, snippets)

    same_dist = [(" a", 0.5), (" the", 0.3), (" an", 0.1), (" some", 0.05), (" one", 0.03)]
    rich_tail = [(f"_extra_{i}", 0.001) for i in range(15)]  # +15 real low-prob tokens
    save_cell(
        evals_db, run_ids[base_id], snippets[0],
        _cap(same_dist + rich_tail, k_returned=20),
    )
    save_cell(evals_db, run_ids[poor_id], snippets[0], _cap(same_dist, k_returned=5))

    return {"group_id": group_id, "base_id": base_id, "poor_id": poor_id}


# ---------------------------------------------------------------------------
# Fixture: a snippet whose text is itself Rich markup syntax -- the exact
# pattern confirmed (against this project's pinned Textual version) to
# raise textual.markup.MarkupError when passed as a plain str Select
# option label, distinct from the DataTable label-stripping defect
# _safe_cell already covers.
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_run_group(evals_db: EvalsDB) -> dict:
    """A run group with real targets but ZERO snippets -- reachable as soon
    as a bench is run against an empty dataset, and ``_create_new_dataset``
    (this PR) creates datasets with zero snippets."""
    base_id = evals_db.create_model(name="base", provider="llama_cpp", model_id="m")
    dataset_id = evals_db.create_dataset(
        name="empty-set", format="custom", source_path="inline:empty-set"
    )
    config = BenchConfig(
        name="empty bench", prompt_mode="raw", top_k=5,
        dataset_id=dataset_id, target_ids=(base_id,),
    )
    task_id = save_bench(evals_db, config)
    targets = [Target(id=base_id, name="base", provider="llama_cpp", model_id="m")]
    group_id, _run_ids = create_run_group(evals_db, task_id, config, targets, [])
    return {"group_id": group_id, "base_id": base_id}


@pytest.fixture
def multi_probe_run_group(evals_db: EvalsDB) -> dict:
    """Two configured probes -- the Probe lens can only show one at a time,
    so which one it is showing has to be nameable and switchable."""
    base_id = evals_db.create_model(name="base", provider="llama_cpp", model_id="m")
    dataset_id = evals_db.create_dataset(
        name="probe-set", format="custom", source_path="inline:probe-set"
    )
    config = BenchConfig(
        name="probe bench", prompt_mode="raw", top_k=5,
        dataset_id=dataset_id, target_ids=(base_id,),
        probes=(" Sure", " Cannot"),
    )
    task_id = save_bench(evals_db, config)
    targets = [Target(id=base_id, name="base", provider="llama_cpp", model_id="m")]
    snippets = [Snippet(id="s1", text="I will", group=None)]
    group_id, run_ids = create_run_group(evals_db, task_id, config, targets, snippets)
    # Both probes are genuinely present, at clearly different
    # probabilities, so the rendered cell says WHICH one is showing.
    save_cell(
        evals_db, run_ids[base_id], snippets[0],
        _cap([(" Sure", 0.6), (" Cannot", 0.2), (" maybe", 0.1)]),
    )
    return {"group_id": group_id, "base_id": base_id}


@pytest.fixture
def markup_hazard_run_group(evals_db: EvalsDB) -> dict:
    base_id = evals_db.create_model(name="base", provider="llama_cpp", model_id="m")
    dataset_id = evals_db.create_dataset(
        name="hazard-set", format="custom", source_path="inline:hazard-set"
    )
    config = BenchConfig(
        name="hazard bench", prompt_mode="raw", top_k=5,
        dataset_id=dataset_id, target_ids=(base_id,),
    )
    task_id = save_bench(evals_db, config)
    targets = [Target(id=base_id, name="base", provider="llama_cpp", model_id="m")]
    snippets = [
        # The reviewer's own confirmed crash case: a bare string
        # "row · a[/]b" raises MarkupError when handed to Select as a
        # plain str option label.
        Snippet(id="s1", text="a[/]b protest", group="loaded"),
    ]
    group_id, run_ids = create_run_group(evals_db, task_id, config, targets, snippets)
    save_cell(evals_db, run_ids[base_id], snippets[0], _cap([(" a", 0.9)]))
    return {"group_id": group_id, "base_id": base_id}


async def _select_run_group(pilot, group_id: str) -> ResultsGrid:
    screen: EvalsScreen = pilot.app.screen
    screen.select(kind="run_group", id=group_id)
    await pilot.pause()
    return screen.query_one("#evals-results-grid", ResultsGrid)


# ---------------------------------------------------------------------------
# Pure-function unit tests: no Textual app needed at all.
# ---------------------------------------------------------------------------


def test_ever_observed_helpers_share_one_scan_and_hold_the_right_axis_fixed():
    """TASK-861 item 1: ``_ever_observed_active_probe`` (one probe, every
    target) and ``_ever_observed_all_probes`` (one target, every probe)
    used to each carry their own copy of the identical nested scan (the
    second's own docstring admitted as much). Folded into a shared
    module-level ``_probe_observed_in_target(snippets, cells, target_id,
    probe)``.

    Monkeypatches that shared helper and asserts BOTH methods route
    through it -- proving neither kept a private copy of the scan -- and
    asserts the EXACT (target, probe) pairs each one calls it with, which
    pins that each call site still holds its own axis fixed rather than
    the fold accidentally making the single-active-probe call site pay for
    every configured probe (or vice versa)."""
    from tldw_chatbook.UI.Evals import results_grid as results_grid_module

    grid = ResultsGrid(view_model=None, run_group_id="g")
    calls: list[tuple[str, str]] = []

    def _fake_observed(snippets, cells, target_id, probe):
        calls.append((target_id, probe))
        return target_id == "t1"

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(results_grid_module, "_probe_observed_in_target", _fake_observed)
    try:
        targets = [{"id": "t1", "name": "a"}, {"id": "t2", "name": "b"}]
        snippets = [{"id": "s1", "text": "x"}]
        cells: dict = {}

        # One probe, every target -- probe held fixed.
        result = grid._ever_observed_active_probe(targets, snippets, cells, " a")
        assert result == {"t1": True, "t2": False}
        assert calls == [("t1", " a"), ("t2", " a")], (
            "must call the shared scan exactly once per target for the "
            f"single active probe, not once per (target, probe) pair: {calls!r}"
        )

        # One target, every probe -- target held fixed.
        calls.clear()
        grid._grid = {
            "snapshot": {"probes": (" a", " b"), "snippets": snippets},
            "cells": cells,
        }
        result2 = grid._ever_observed_all_probes("t1")
        assert result2 == {" a": True, " b": True}
        assert calls == [("t1", " a"), ("t1", " b")], (
            "must call the shared scan exactly once per configured probe "
            f"for the single fixed target, not once per (target, probe) pair: {calls!r}"
        )
    finally:
        monkeypatch.undo()


def test_degenerate_canary_text_uses_an_em_dash_not_ascii_double_dash():
    """TASK-1481 fix-round-1: the reviewer found this rendered sentence
    (shared verbatim by the grid callout and inspector.py's per-target
    callout, see the function's own docstring) still used ASCII ``--``
    where the rest of the Evals rail copy uses real em-dashes. Covers
    both the singular and plural grammar branches -- the dash sits after
    ``{be}``, which differs between them ("is"/"are")."""
    singular = degenerate_canary_text(["steered"])
    plural = degenerate_canary_text(["steered", "distilled"])
    assert " -- " not in singular
    assert " -- " not in plural
    assert "—" in singular
    assert "—" in plural


def test_near_tie_threshold_is_a_named_constant_not_a_magic_number():
    """The threshold and the ``near_tie()`` predicate itself live in
    ``analysis.py`` (moved there per review: a threshold comparison on raw
    logprobs is methodology, not view formatting, and belongs alongside
    ``TRUNCATION_WARN_THRESHOLD``/``divergence``'s own ``is_bounded``) --
    see its own docstring for the observed-instability rationale (a
    ~0.095-0.096 nat gap already produced a rank flip; this codebase's
    normalizer test independently drew the same 0.15 nat boundary for the
    identical fixture). This only pins that ``results_grid.py`` reads the
    value FROM the engine rather than carrying its own copy;
    ``Tests/Evals/word_bench/test_analysis.py`` pins ``near_tie()``'s own
    behaviour."""
    assert analysis.NEAR_TIE_LOGPROB_GAP_NATS == 0.15
    assert "NEAR_TIE_LOGPROB_GAP_NATS" not in dir(
        __import__("tldw_chatbook.UI.Evals.results_grid", fromlist=["x"])
    ), "the threshold must not also be re-defined in the view layer"


def test_render_token_makes_whitespace_visible():
    """`" a"` and `"a"` must not render identically -- a bare leading space
    is invisible in a terminal cell, especially with DataTable padding."""
    assert render_token(" a") == '" a"'
    assert render_token("a") == '"a"'
    assert render_token(" a") != render_token("a")


def test_render_probe_reading_covers_all_three_states_distinctly():
    matched = TokenProb(token=" Sure", logprob=math.log(0.6))
    observed = analysis.ProbeReading(
        probe=" Sure", state="observed", logprob=math.log(0.6), matched=matched
    )
    bounded = analysis.ProbeReading(probe=" Sure", state="bounded", logprob=math.log(0.9))
    never = analysis.ProbeReading(probe=" Sure", state="never_observed", logprob=None)

    observed_text = render_probe_reading(observed)
    bounded_text = render_probe_reading(bounded)
    never_text = render_probe_reading(never)

    assert observed_text == f"{math.log(0.6):.2f}  60.0%"
    assert bounded_text.startswith("< ")
    assert "≥" not in bounded_text
    assert never_text == "never observed"
    assert len({observed_text, bounded_text, never_text}) == 3


# ---------------------------------------------------------------------------
# Integration tests through the real EvalsScreen.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selecting_a_run_group_mounts_the_results_grid_with_the_right_shape(
    evals_app, mixed_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)
        # 1 "Snippet" column + 2 target columns (top1 lens: no Spread column).
        assert len(table.columns) == 3
        assert table.row_count == 3


@pytest.mark.asyncio
async def test_grid_meta_defines_its_own_jargon_via_tooltip(evals_app, mixed_run_group):
    """TASK-1076: the meta line ("<bench> * raw * K 20 * N cells * N
    failed") is a first-contact wall of undefined jargon for a new reader,
    with nothing else on the screen defining "raw", "K", or "cells". A
    tooltip keeps that definition reachable without permanently widening
    the header for every return visit."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        meta = grid.query_one("#evals-grid-meta")
        tooltip = str(meta.tooltip)
        assert "raw" in tooltip
        assert "K:" in tooltip
        assert "cells:" in tooltip


@pytest.mark.asyncio
async def test_grid_and_its_selectors_are_genuinely_visible_within_the_detail_pane(
    evals_app, mixed_run_group
):
    """Per the program's own trap: presence in the DOM does not prove a
    real, usable layout. Every descendant checked here must be fully
    contained within #evals-detail-pane's clip region, not merely
    `region.width > 0`."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        await _select_run_group(pilot, mixed_run_group["group_id"])
        screen = pilot.app.screen
        pane = screen.query_one("#evals-detail-pane")

        for widget_id in (
            "#evals-results-grid",
            "#evals-grid-meta",
            "#evals-grid-state",
            "#evals-lens-selector",
            "#evals-baseline-selector",
            "#evals-grid-table",
        ):
            widget = screen.query_one(widget_id)
            assert widget.region.width > 0, widget_id
            assert widget.region.height > 0, widget_id
            assert pane.region.contains_region(widget.region), (
                f"{widget_id} at {widget.region} escapes detail pane {pane.region}"
            )


@pytest.mark.asyncio
async def test_top1_lens_marks_a_near_tie_and_leaves_a_clear_winner_bare(
    evals_app, mixed_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)

        tie_cell = str(table.get_cell("s1", mixed_run_group["base_id"]))
        clear_cell = str(table.get_cell("s1", mixed_run_group["steered_id"]))

        assert "≈" in tie_cell, f"near-tie cell must show a tie marker, got {tie_cell!r}"
        assert '" a"' in tie_cell and '" the"' in tie_cell
        assert "≈" not in clear_cell, f"clear winner must not show a tie marker, got {clear_cell!r}"
        assert '" a"' in clear_cell


@pytest.mark.asyncio
async def test_failed_cell_renders_em_dash_and_its_reason_reaches_the_inspector(
    evals_app, mixed_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)

        failed_text = str(table.get_cell("s2", mixed_run_group["base_id"]))
        assert failed_text == FAILED_MARK
        assert failed_text != "0"

        row = table.get_row_index("s2")
        col = table.get_column_index(mixed_run_group["base_id"])
        table.focus()
        table.move_cursor(row=row, column=col)
        await pilot.pause()

        body = pilot.app.screen.query_one("#evals-cell-inspector-body")
        text = str(body.renderable)
        assert "unreachable" in text
        assert "connection refused" in text


@pytest.mark.asyncio
async def test_unrun_cell_renders_blank_never_zero(evals_app, mixed_run_group):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)

        unrun_text = str(table.get_cell("s3", mixed_run_group["base_id"]))
        assert unrun_text == "", f"unrun cell must render blank, got {unrun_text!r}"
        assert unrun_text != "0"
        assert unrun_text != FAILED_MARK

        row = table.get_row_index("s3")
        col = table.get_column_index(mixed_run_group["base_id"])
        table.focus()
        table.move_cursor(row=row, column=col)
        await pilot.pause()
        body = pilot.app.screen.query_one("#evals-cell-inspector-body")
        assert "Not yet run" in str(body.renderable)


@pytest.mark.asyncio
async def test_grid_content_survives_datatable_rendering_without_bracket_corruption(
    evals_app, mixed_run_group
):
    """Regression: DataTable's own ``default_cell_formatter`` (and
    ``add_column``'s label handling) run a plain ``str`` through
    ``Text.from_markup`` -- ``"[...]"`` is Rich markup syntax, so a bare
    string column label or cell value ending in e.g. ``" [warned]"`` or
    ``" [loaded]"`` is silently stripped to ``" "`` at render time (caught
    live while building this test: a first version of the warned-column
    label rendered as ``"steered "`` with no visible warning at all).
    ``get_cell()`` alone would NOT have caught this -- it reads the raw
    stored value, and only the COLUMN LABEL path transforms at storage
    time; cell values are only transformed lazily, at actual render time
    (``DataTable._get_row_renderables``) -- so this checks the real
    render path, not just storage.
    """
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)

        steered_col = table.columns[mixed_run_group["steered_id"]]
        assert "[warned]" in str(steered_col.label)

        row_index = table.get_row_index("s2")
        rendered = table._get_row_renderables(row_index)
        snippet_cell_text = (
            rendered.cells[0].plain
            if hasattr(rendered.cells[0], "plain")
            else str(rendered.cells[0])
        )
        assert "[loaded]" in snippet_cell_text, (
            f"snippet group annotation must survive DataTable rendering, "
            f"got {snippet_cell_text!r}"
        )


@pytest.mark.asyncio
async def test_baseline_selector_options_survive_markup_special_characters_in_snippet_text(
    evals_app, markup_hazard_run_group
):
    """Regression, one widget over from test_grid_content_survives_
    DataTable_rendering_without_bracket_corruption: `#evals-baseline-
    selector`'s "Row · <snippet text>" options embed raw snippet text as
    plain str Select option labels. Confirmed (this project's pinned
    Textual version) that ``"row · The rioters [loaded] were"`` silently
    renders with the bracketed span stripped, and ``"row · a[/]b"`` RAISES
    ``textual.markup.MarkupError`` outright -- selecting a run group with
    such a snippet would have crashed the whole screen on mount, not just
    looked wrong. This snippet is also the run group's ONLY snippet, so
    merely mounting the grid (which builds every baseline option up
    front, not just the selected one) already exercises the crash path.
    """
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        # Mounting alone must not raise MarkupError.
        grid = await _select_run_group(pilot, markup_hazard_run_group["group_id"])

        select = grid.query_one("#evals-baseline-selector", Select)
        option_texts = [str(label) for label, _value in select._options]
        row_options = [text for text in option_texts if text.startswith("Row ·")]
        assert any("a[/]b protest" in text for text in row_options), (
            f"snippet text must survive the Select option label unmangled, "
            f"got {row_options!r}"
        )


@pytest.mark.asyncio
async def test_warned_column_header_carries_the_warning_the_clean_one_does_not(
    evals_app, mixed_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)

        base_col = table.columns[mixed_run_group["base_id"]]
        steered_col = table.columns[mixed_run_group["steered_id"]]

        assert "warned" not in str(base_col.label).lower()
        assert "warned" in str(steered_col.label).lower()


def _header_row_text(app: App, table: DataTable) -> str:
    """Read the on-screen text of just ``table``'s header row.

    A whole-screen blob match (join every compositor strip -- see
    ``Tests/UI/test_lab_mode_strip.py``'s ``_rendered_text``) is too broad
    here: this module's fixtures name a target ``"base"``, and the lens/
    baseline ``Select`` controls above the grid render their own options
    as e.g. ``f"Column · {target['name']}"`` (``results_grid.py``), so
    ``"base"`` sits in the screen blob whether or not the table's header
    row exists at all -- confirmed by reverting the CSS fix this helper
    guards and observing the assertion still pass. Restricting the read
    to the table's own header LINE, not the screen, is what makes the
    check fail when the header is actually missing.

    ``table.region`` is stable across both the broken and fixed render
    paths (confirmed: same ``Region`` in each), so ``table.region.y`` is a
    safe, focus-state-independent way to find the header row -- a
    ``DataTable``'s first rendered row is always its column header, and
    that stays the top row of the table's own box whether it currently
    shows a border, an outline, or neither (the sole difference this bug
    produces). ``table.region.x``/``width`` then crop the strip to the
    table's own columns, excluding anything a sibling widget draws on the
    same screen row -- including the run view's degenerate-canary callout
    (task-1036), which can otherwise land on the same line as the grid.

    Args:
        app: The running Textual app whose screen was just rendered; its
            compositor holds the actual painted strips.
        table: The ``DataTable`` whose header row should be read.

    Returns:
        The header row's on-screen text, cropped to ``table``'s own
        region -- empty if ``table``'s top row is off-screen.
    """
    strips = app.screen._compositor.render_strips()
    header_y = table.region.y
    if not (0 <= header_y < len(strips)):
        return ""
    line = "".join(segment.text for segment in strips[header_y])
    x_start, x_end = table.region.x, table.region.x + table.region.width
    return line[x_start:x_end]


@pytest.mark.asyncio
async def test_focused_results_grid_keeps_its_header_and_the_warned_marker(
    evals_app, mixed_run_group
):
    """TASK-1034: a focused results grid used to lose its column header --
    and the "[warned]" canary marker riding inside it -- entirely.

    Root cause: the global fallback ``*:focus { outline: solid
    $ds-focus-accent; }`` (``core/_reset.tcss``) draws its outline INSIDE
    a widget's own box, painting over the widget's own first and last
    rendered rows rather than sitting outside them like ``border`` does. A
    ``DataTable``'s first rendered row is its column header, so as soon as
    the grid's ``DataTable`` took keyboard focus -- which ``ResultsGrid.
    on_mount`` (results_grid.py) does immediately on every fresh mount, so
    this is not a rare interaction but the DEFAULT state right after
    selecting a run group -- the outline's box-drawing top edge replaced
    the header outright.

    The tests above this one (``test_warned_column_header_carries_the_
    warning_the_clean_one_does_not`` and
    ``test_grid_content_survives_datatable_rendering_without_bracket_
    corruption``) all read ``table.columns[...].label`` -- the DataTable's
    stored data model, not what the compositor actually painted -- so none
    of them could have caught this: the column label was always correct
    in the data model, only its on-screen rendering vanished. This test
    reads the real compositor output instead (``_header_row_text``, built
    on the same ``screen._compositor.render_strips()`` primitive
    ``test_lab_mode_strip.py`` uses, but cropped to the table's own header
    LINE rather than joined across the whole screen -- an earlier version
    of this test used the whole-screen join and its "base" and
    "[warned]" assertions both still passed with the header completely
    absent, because unrelated widgets echo those same substrings
    elsewhere on screen; see the helper's own docstring) specifically
    WHILE the table holds focus, which ``_select_run_group`` already
    leaves it in via ``on_mount``'s auto-focus -- no extra ``table.
    focus()`` call needed, and none is made here, so this test fails
    exactly the way live UAT did: on the DEFAULT, no-extra-interaction
    path.

    Args:
        evals_app: The ``EvalsHarness`` app fixture (unstarted); pushes a
            real ``EvalsScreen`` on mount.
        mixed_run_group: Fixture run group with a clean ``"base"`` target
            and a ``"steered"`` target carrying the ``[warned]`` canary.
    """
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)

        assert table.has_focus, (
            "expected on_mount's auto-focus to have already put the grid "
            "in the exact state that dropped its header during UAT"
        )

        header_text = _header_row_text(pilot.app, table)
        assert "Snippet" in header_text, (
            f"column header missing from the focused grid's header row:\n"
            f"{header_text!r}"
        )
        assert "base" in header_text, (
            f"target column header missing from the focused grid's header "
            f"row:\n{header_text!r}"
        )
        assert "[warned]" in header_text, (
            f"the warned-target canary marker is missing from the focused "
            f"grid's header row:\n{header_text!r}"
        )


# ---------------------------------------------------------------------------
# TASK-1036: the degenerate-canary callout on the run view itself -- the
# grid's ONLY signal used to be the nine-character "[warned]" column
# suffix above, which TASK-1034 shows can vanish entirely along with the
# rest of the header in one of the grid's two render paths.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degenerate_canary_callout_names_the_target_and_states_the_consequence(
    evals_app, mixed_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])

        callout = grid.query_one("#evals-grid-canary-callout", Static)
        assert "ds-recovery-callout" in callout.classes
        # `.visual.plain`, not `.renderable`/`.content`: this codebase's own
        # Textual compatibility shim (tldw_chatbook/__init__.py) defines
        # `Static.renderable` as an alias for `.content` -- the RAW,
        # unparsed constructor argument -- so it reads back correctly
        # regardless of whether `markup=` actually got applied. `.visual`
        # is the actual `visualize(..., markup=self._render_markup)`
        # result Static.render() draws from (confirmed against this
        # module's own compatibility shim and Static.visual's source);
        # only that path can catch a lost `markup=False`. Mirrors
        # `Tests/UI/test_library_ingest_canvas.py::
        # test_error_and_warning_markup_is_escaped`'s identical pattern.
        text = callout.visual.plain

        # Named: the warned target ("steered"), not the clean one ("base").
        assert "steered" in text
        assert "base" not in text

        # Consequence, not just the bare state -- matching the bench
        # view's own wording (degenerate_canary_text is shared, see
        # inspector.py's _recovery_callout_text).
        assert "canary" in text.lower()
        assert "large divergence" in text
        assert "may reflect that, not the prompt" in text
        assert callout.region.width > 0
        assert callout.region.height > 0


@pytest.mark.asyncio
async def test_degenerate_canary_callout_absent_when_nothing_is_warned(
    evals_app, clean_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, clean_run_group["group_id"])

        # No callout, no reserved blank row, no empty container.
        assert not grid.query("#evals-grid-canary-callout")
        assert not grid.query(".ds-recovery-callout")


@pytest.mark.asyncio
async def test_degenerate_canary_callout_names_every_warned_target_when_several(
    evals_app, multi_warned_run_group
):
    """A bench can have several warned targets; the callout must name ALL
    of them, not settle for "a target was degenerate"."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, multi_warned_run_group["group_id"])

        callout = grid.query_one("#evals-grid-canary-callout", Static)
        text = callout.visual.plain  # see the .visual.plain rationale above

        assert "steered" in text
        assert "distilled" in text
        assert "clean-target" not in text
        # Plural grammar -- not two names glued onto a singular sentence.
        assert "These targets are still runnable" in text
        assert degenerate_canary_text(["steered", "distilled"]) == text


@pytest.mark.asyncio
async def test_degenerate_canary_callout_survives_a_target_name_containing_markup(
    evals_app, warned_markup_hazard_run_group
):
    """Regression, one widget over from test_grid_content_survives_
    DataTable_rendering_without_bracket_corruption: the callout's target
    name comes from the same free-text ``target["name"]`` a DataTable
    column label uses, which Rich's markup parser silently mangles (or
    raises on) when handed a plain ``str``. This pins that the callout's
    own guard (``markup=False`` on its ``Static``) is actually applied.
    """
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(
            pilot, warned_markup_hazard_run_group["group_id"]
        )

        callout = grid.query_one("#evals-grid-canary-callout", Static)
        text = callout.visual.plain  # see the .visual.plain rationale above
        assert "steered [redacted]" in text


# ---------------------------------------------------------------------------
# TASK-1477: the run-level failure callout.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_callout_names_the_next_step_when_every_cell_failed(
    evals_app, all_cells_failed_run_group
):
    """A fully-failed run used to read as an unexplained wall of
    ``FAILED_MARK`` em-dashes with only a buried "4 failed" in the meta
    line's jargon. Every cell failing means the run itself is otherwise
    unusable, so the callout must name a concrete next step (the bench
    Run action, wired earlier in this batch, makes "run the bench again"
    real)."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, all_cells_failed_run_group["group_id"])

        callout = grid.query_one("#evals-grid-failure-callout", Static)
        assert "ds-recovery-callout" in callout.classes
        # .visual.plain, not .renderable/.content -- see the canary
        # callout tests above for why (only .visual catches a lost
        # markup=False).
        text = callout.visual.plain
        assert text == (
            "All 4 cells failed — unreachable. Check that the target's "
            "server is running and reachable, then run the bench again."
        )
        assert callout.region.width > 0
        assert callout.region.height > 0


@pytest.mark.asyncio
async def test_failure_callout_states_the_fact_without_a_next_step_when_partial(
    evals_app, one_of_four_cells_failed_run_group
):
    """A partial failure still leaves a usable run -- the callout must
    state the count/reason but must NOT prescribe "run the bench again",
    which would misrepresent a run that already has real data in it."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(
            pilot, one_of_four_cells_failed_run_group["group_id"]
        )

        callout = grid.query_one("#evals-grid-failure-callout", Static)
        text = callout.visual.plain
        assert text == "1 of 4 cells failed — unreachable."
        assert "run the bench again" not in text


@pytest.mark.asyncio
async def test_failure_callout_absent_when_nothing_failed(evals_app, clean_run_group):
    """No reserved blank row, no empty container -- the callout must not
    exist in the DOM at all when every cell captured cleanly."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, clean_run_group["group_id"])

        assert not grid.query("#evals-grid-failure-callout")


@pytest.mark.asyncio
async def test_failure_callout_dominant_reason_ties_broken_by_first_seen_majority_otherwise_wins(
    evals_app, evals_db
):
    """``_failure_summary``'s docstring claims the dominant reason is "most
    frequent, ties broken by first-seen" -- the three tests above only ever
    exercise a SINGLE failure reason, so that claim was unpinned: a future
    refactor to e.g. ``collections.Counter.most_common()`` (whose tie order
    is an implementation detail, not a contract) could silently flip which
    reason a tied run's callout names, with nothing in the suite noticing.

    Two grids, one test, both built via ``_make_single_target_run_group``
    so ``save_cell`` call order is exactly the order ``load_grid`` inserts
    into ``cells`` (see that helper's docstring):

    - 2-vs-2 TIE: two ``timeout`` cells saved BEFORE two ``unreachable``
      cells -- the callout must name ``timeout``, the first-seen reason,
      not ``unreachable`` (which would win under alphabetical or
      most-recently-seen tie-breaking).
    - 1-vs-3 MAJORITY: one ``timeout`` cell saved BEFORE three
      ``unreachable`` cells -- the callout must name ``unreachable``, the
      real majority, proving the tie-break only applies on an actual tie
      rather than "first-seen always wins regardless of count".
    """
    tie_group_id, _tie_target_id, tie_snippets, tie_run_id = (
        _make_single_target_run_group(evals_db, "tie-break", 4)
    )
    for snippet, reason in zip(
        tie_snippets, ["timeout", "timeout", "unreachable", "unreachable"]
    ):
        save_cell(evals_db, tie_run_id, snippet, CellError(reason=reason, detail=""))

    majority_group_id, _majority_target_id, majority_snippets, majority_run_id = (
        _make_single_target_run_group(evals_db, "majority", 4)
    )
    for snippet, reason in zip(
        majority_snippets, ["timeout", "unreachable", "unreachable", "unreachable"]
    ):
        save_cell(evals_db, majority_run_id, snippet, CellError(reason=reason, detail=""))

    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()

        tie_grid = await _select_run_group(pilot, tie_group_id)
        tie_callout = tie_grid.query_one("#evals-grid-failure-callout", Static)
        # .visual.plain -- see the canary/failure callout tests above for
        # why (only .visual catches a lost markup=False).
        assert tie_callout.visual.plain == (
            "All 4 cells failed — timeout. Check that the target's "
            "server is running and reachable, then run the bench again."
        )

        majority_grid = await _select_run_group(pilot, majority_group_id)
        majority_callout = majority_grid.query_one("#evals-grid-failure-callout", Static)
        assert majority_callout.visual.plain == (
            "All 4 cells failed — unreachable. Check that the target's "
            "server is running and reachable, then run the bench again."
        )


@pytest.mark.asyncio
async def test_entropy_lens_states_effective_k_and_uses_it_for_every_cell(
    evals_app, k_depth_matched_run_group
):
    """A K=20 cell and a K=5 cell holding the SAME underlying distribution
    (over the first 5 ranks) must read identical entropy once both are
    read at the shared effective K -- otherwise the K=20 column would look
    "more confident" purely from its setting.

    Uses ``k_depth_matched_run_group`` rather than ``clean_run_group``: see
    that fixture's own docstring for why (``clean_run_group``'s s2/s3 rows
    would drag a clamped ``effective_k`` down to their own tiny native
    length)."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, k_depth_matched_run_group["group_id"])
        select = grid.query_one("#evals-lens-selector", Select)
        select.value = "entropy"
        await pilot.pause()

        meta = str(grid.query_one("#evals-grid-meta").renderable)
        # Delimited, not a bare substring: "K 5" alone also matches "K 50".
        assert "· K 5 ·" in meta, f"header must state the effective K (5), got {meta!r}"

        table = grid.query_one("#evals-grid-table", DataTable)
        rich_text = str(table.get_cell("s1", k_depth_matched_run_group["base_id"]))
        poor_text = str(table.get_cell("s1", k_depth_matched_run_group["poor_id"]))
        assert rich_text == poor_text, (
            f"same distribution at a shared effective K must produce equal "
            f"entropy: rich={rich_text!r} poor={poor_text!r}"
        )


@pytest.mark.asyncio
async def test_delta_lens_never_renders_a_leading_gte(evals_app, clean_run_group):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, clean_run_group["group_id"])
        select = grid.query_one("#evals-lens-selector", Select)
        select.value = "delta"
        await pilot.pause()

        table = grid.query_one("#evals-grid-table", DataTable)
        for row_index in range(table.row_count):
            for col_index in range(len(table.columns)):
                text = str(table.get_cell_at((row_index, col_index)))
                assert "≥" not in text, f"found a leading >= at ({row_index},{col_index}): {text!r}"


@pytest.mark.asyncio
async def test_delta_lens_flags_high_combined_truncation_with_a_bang_marker(
    evals_app, clean_run_group
):
    """The `!` marker is the grid's ENTIRE substitute for the leading "≥"
    PR 2's review disproved (see the module docstring). Before this test,
    no fixture in this suite actually reached TRUNCATION_WARN_THRESHOLD
    (0.25) -- s4's combined_truncation (0.642, computed at the shared
    effective K=5) clears it well past. Also checks the inspector
    explains the COMBINED mass, not just this cell's own (see
    test_focused_delta_cell_inspector_explains_the_bang_marker for the
    fuller inspector assertion)."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, clean_run_group["group_id"])
        grid.query_one("#evals-lens-selector", Select).value = "delta"
        await pilot.pause()

        table = grid.query_one("#evals-grid-table", DataTable)
        cell_text = str(table.get_cell("s4", clean_run_group["poor_id"]))
        assert cell_text == "0.47 !", f"expected the bang marker, got {cell_text!r}"
        assert "≥" not in cell_text

        # Every OTHER real comparison in this fixture stays unmarked --
        # proves the marker is cell-specific, not a lens-wide artifact.
        s2_text = str(table.get_cell("s2", clean_run_group["poor_id"]))
        s3_text = str(table.get_cell("s3", clean_run_group["poor_id"]))
        assert not s2_text.endswith("!"), s2_text
        assert not s3_text.endswith("!"), s3_text


@pytest.mark.asyncio
async def test_focused_delta_cell_inspector_explains_the_bang_marker_with_combined_mass(
    evals_app, clean_run_group
):
    """Closes the gap review found: the inspector used to show only the
    focused cell's OWN truncated mass, never the mass that actually
    triggered the `!` -- the combined mass across both compared cells."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, clean_run_group["group_id"])
        grid.query_one("#evals-lens-selector", Select).value = "delta"
        await pilot.pause()

        table = grid.query_one("#evals-grid-table", DataTable)
        row = table.get_row_index("s4")
        col = table.get_column_index(clean_run_group["poor_id"])
        table.focus()
        table.move_cursor(row=row, column=col)
        await pilot.pause()

        body = pilot.app.screen.query_one("#evals-cell-inspector-body")
        text = str(body.renderable)
        assert "Δ baseline: 0.47 !" in text
        assert "Combined truncated mass" in text
        assert "64.2%" in text, (
            f"combined mass must be the pair's 64.2% (at the shared K), "
            f"not this cell's own 34.6%: {text!r}"
        )
        # This cell's OWN truncated mass (34.6%, poor_pairs' native K=5
        # truncation) must also be present but is NOT what the marker
        # explanation cites -- both numbers coexist and are distinguishable.
        assert "Truncated mass: 34.6%" in text


@pytest.mark.asyncio
async def test_focused_delta_cell_inspector_says_nothing_extra_when_not_bounded(
    evals_app, clean_run_group
):
    """A Δ cell that is NOT flagged must not show a combined-mass
    explanation at all -- absence of the caveat is itself informative."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, clean_run_group["group_id"])
        grid.query_one("#evals-lens-selector", Select).value = "delta"
        await pilot.pause()

        table = grid.query_one("#evals-grid-table", DataTable)
        row = table.get_row_index("s3")
        col = table.get_column_index(clean_run_group["poor_id"])
        table.focus()
        table.move_cursor(row=row, column=col)
        await pilot.pause()

        body = pilot.app.screen.query_one("#evals-cell-inspector-body")
        text = str(body.renderable)
        assert "Δ baseline:" in text
        assert "!" not in text.split("Δ baseline:")[1].split("\n")[0]
        assert "Combined truncated mass" not in text


@pytest.mark.asyncio
async def test_delta_lens_baseline_column_shows_literal_baseline_text_or_blank_if_unrun(
    evals_app, mixed_run_group
):
    """Default baseline is column 0 (base). Its own captured cell (s1) must
    read the literal "baseline" text, never a number; its own UNRUN cell
    (s3) must still render blank, not "baseline" and not "0" -- the
    baseline position is not exempt from the unrun/failed rules."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        select = grid.query_one("#evals-lens-selector", Select)
        select.value = "delta"
        await pilot.pause()

        table = grid.query_one("#evals-grid-table", DataTable)
        base_id = mixed_run_group["base_id"]
        assert str(table.get_cell("s1", base_id)) == "baseline"
        assert str(table.get_cell("s3", base_id)) == ""

        state = str(grid.query_one("#evals-grid-state").renderable)
        assert "column · base" in state


@pytest.mark.asyncio
async def test_delta_lens_on_a_single_target_run_explains_itself_instead_of_faking_baseline(
    evals_app, evals_db
):
    """TASK-1481 (live UAT): a single-target run has no second TARGET for
    COLUMN-mode Δ baseline comparison (the default baseline mode). Before
    this fix, ``_delta_reading``'s "is_baseline_position" branch fired for
    EVERY cell (there being only one target, ``tid == baseline_id``
    always), so the whole column read as the literal word "baseline" --
    alongside an always-empty Spread column, since ``analysis.spread``
    needs at least two per-row captures across targets (see
    ``_compute_active_lens_rows``). The lens itself must stay selectable
    (this test switches to it the same way any other delta test does);
    only what it renders for this shape changes.

    TASK-1481 fix-round-1: this is deliberately scoped to COLUMN mode --
    see ``test_delta_lens_row_baseline_on_a_single_target_still_computes_
    real_divergence`` right below for why ROW mode's baseline (a snippet,
    not a target) is unaffected by a single-target run and must keep
    rendering real comparisons, not this same blank-and-explain
    treatment."""
    group_id, target_id, snippets, run_id = _make_single_target_run_group(
        evals_db, "lonely-target", 2
    )
    save_cell(evals_db, run_id, snippets[0], _cap([(" a", 0.9)]))
    save_cell(evals_db, run_id, snippets[1], _cap([(" b", 0.7)]))

    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, group_id)
        select = grid.query_one("#evals-lens-selector", Select)
        select.value = "delta"
        await pilot.pause()

        table = grid.query_one("#evals-grid-table", DataTable)
        assert str(table.get_cell("s1", target_id)) != "baseline"
        assert str(table.get_cell("s2", target_id)) != "baseline"
        assert str(table.get_cell("s1", target_id)) == ""
        assert str(table.get_cell("s2", target_id)) == ""

        state = str(grid.query_one("#evals-grid-state").renderable)
        assert "needs at least two targets" in state

        # The Δ lens is still a live, selectable option -- this fixture's
        # single-target shape only changes what it renders, never removes
        # it from the Select.
        assert select.value == "delta"


@pytest.mark.asyncio
async def test_delta_lens_row_baseline_on_a_single_target_still_computes_real_divergence(
    evals_app, evals_db
):
    """TASK-1481 fix-round-1: the reviewer traced ``_delta_reading`` and
    confirmed the first version of this fix's gate was too broad -- it
    keyed off target count ALONE, regardless of ``self._baseline_mode``,
    so it also blanked out ROW-mode baselines on a single-target run. Row
    mode's baseline is a SNIPPET, not a target: a cell there compares two
    DIFFERENT snippets' captures on the run's one (and only) target -- a
    real, independently reproducible divergence, never a degenerate
    comparison-with-itself. Row mode is fully reachable via
    ``#evals-baseline-selector`` even with one target (see
    ``_baseline_options``, which always lists every snippet as a "Row ·"
    option). This pins THREE things the broad gate got wrong: a real
    numeric divergence for a genuine comparison, the literal "baseline"
    text for the baseline row's own position, and ``FAILED_MARK`` (not
    blank) for a cell that itself failed."""
    group_id, target_id, snippets, run_id = _make_single_target_run_group(
        evals_db, "row-baseline-lonely", 3
    )
    baseline_cap = _cap([(" a", 0.9), (" b", 0.1)])
    real_cap = _cap([(" a", 0.2), (" b", 0.8)])
    save_cell(evals_db, run_id, snippets[0], baseline_cap)  # s1: the baseline row
    save_cell(evals_db, run_id, snippets[1], real_cap)  # s2: a real comparison
    save_cell(
        evals_db, run_id, snippets[2], CellError(reason="timeout", detail="")
    )  # s3: this cell itself failed

    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, group_id)
        grid.query_one("#evals-lens-selector", Select).value = "delta"
        await pilot.pause()
        grid.query_one("#evals-baseline-selector", Select).value = ("row", "s1")
        await pilot.pause()

        table = grid.query_one("#evals-grid-table", DataTable)
        # The baseline row's own position: still the literal word, same as
        # column mode -- comparing a cell to itself is not a finding.
        assert str(table.get_cell("s1", target_id)) == "baseline"

        # A real comparison: the SAME divergence analysis.divergence
        # itself produces for these two captures, computed independently
        # here rather than hand-picked -- mirrors this file's own
        # test_group_mean_rows_match_analysis_group_means_over_the_
        # rendered_divergences pattern.
        expected_jsd, _ = analysis.divergence(real_cap, baseline_cap)
        assert str(table.get_cell("s2", target_id)) == f"{expected_jsd:.2f}"

        # A cell that itself failed: FAILED_MARK, never blank -- the
        # broad gate silently turned this blank too (the reviewer's Minor).
        assert str(table.get_cell("s3", target_id)) == FAILED_MARK

        # The column-mode-only "needs at least two targets" explanation
        # must NOT show here -- row mode's baseline is genuinely usable.
        state = str(grid.query_one("#evals-grid-state").renderable)
        assert "needs at least two targets" not in state
        assert "row ·" in state


@pytest.mark.asyncio
async def test_baseline_cell_failing_makes_the_whole_comparison_unavailable_not_zero(
    evals_app, mixed_run_group
):
    """Baseline mode "row", reference row s2 -- whose `base` cell FAILED.
    Every other cell in the base COLUMN compares against that failed cell,
    so the whole column (except the baseline row itself) must read as
    unavailable (the FAILED_MARK), never a computed 0.00."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        grid.query_one("#evals-lens-selector", Select).value = "delta"
        await pilot.pause()
        grid.query_one("#evals-baseline-selector", Select).value = (
            "row", "s2",
        )
        await pilot.pause()

        table = grid.query_one("#evals-grid-table", DataTable)
        base_id = mixed_run_group["base_id"]
        # s1 x base compares against s2 x base, which failed.
        assert str(table.get_cell("s1", base_id)) == FAILED_MARK
        # s2 (the baseline row) x base: base's OWN cell at s2 failed, so
        # even the "baseline position" itself reads as failed, not "baseline".
        assert str(table.get_cell("s2", base_id)) == FAILED_MARK


@pytest.mark.asyncio
async def test_baseline_key_toggles_column_to_row_and_header_states_it(
    evals_app, mixed_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)
        table.focus()

        state_before = str(grid.query_one("#evals-grid-state").renderable)
        assert "Baseline: column" in state_before

        await pilot.press("b")
        await pilot.pause()

        state_after = str(grid.query_one("#evals-grid-state").renderable)
        assert "Baseline: row" in state_after
        select = grid.query_one("#evals-baseline-selector", Select)
        assert select.value[0] == "row"


@pytest.mark.asyncio
async def test_lens_key_cycles_the_selector_through_all_five_lenses(
    evals_app, mixed_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)
        table.focus()
        select = grid.query_one("#evals-lens-selector", Select)

        seen = [select.value]
        for _ in range(5):
            await pilot.press("l")
            await pilot.pause()
            seen.append(select.value)

        assert seen == ["top1", "entropy", "probe", "coverage", "delta", "top1"]


@pytest.mark.asyncio
async def test_sort_key_registers_and_reorders_by_spread(evals_app, clean_run_group):
    """Previously only asserted the header text changed and never checked
    row order -- and COULDN'T have, against ``mixed_run_group``, where
    only one row (s1) has >=2 captured cells at all (every other row's
    sort key is the same -1.0 sentinel, so a stable sort leaves them in
    their original positions regardless of desc/asc). ``clean_run_group``
    has four rows with distinct, real spread values -- computed here
    against the actual engine (analysis.spread) rather than hardcoded:
    s2 (~0.624) > s4 (~0.472) > s3 (~0.003) > s1 (0.0)."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, clean_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)
        table.focus()

        state_before = str(grid.query_one("#evals-grid-state").renderable)
        assert "Sort: dataset order" in state_before
        assert [
            table.get_row_index(sid) for sid in ("s1", "s2", "s3", "s4")
        ] == [0, 1, 2, 3], "unsorted must be dataset (authoring) order"

        await pilot.press("s")
        await pilot.pause()
        state_after = str(grid.query_one("#evals-grid-state").renderable)
        assert "Sort: spread" in state_after

        order_by_index = sorted(
            ("s1", "s2", "s3", "s4"), key=lambda sid: table.get_row_index(sid)
        )
        assert order_by_index == ["s2", "s4", "s3", "s1"], (
            f"expected descending-spread row order, got {order_by_index}"
        )


@pytest.mark.asyncio
async def test_ascending_sort_still_puts_undefined_spread_rows_last(
    evals_app, mixed_run_group
):
    """Spread is a Jensen-Shannon divergence, always >= 0, so a row with
    fewer than two captured cells (no defined spread) used to sort with a
    -1.0 sentinel -- below every real value by construction. That is
    correct for descending (see test_sort_key_registers_and_reorders_by_
    spread) but was backwards for ascending: the sentinel's ``reverse=
    False`` path put undefined rows FIRST, presenting "no measurement at
    all" as "the least disagreement". ``mixed_run_group`` has exactly one
    row with a real spread (s1: near-tie base vs. clear-winner steered,
    both captured) and two with fewer than two captured cells (s2: one
    CellError: only steered captured; s3: base never run: only steered
    captured) -- so this fails against the pre-fix sentinel/reverse
    scheme, which would put s2/s3 ahead of s1 in ascending order."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)
        table.focus()

        await pilot.press("s")  # none -> desc
        await pilot.pause()
        await pilot.press("s")  # desc -> asc
        await pilot.pause()

        state = str(grid.query_one("#evals-grid-state").renderable)
        assert "Sort: spread ▲" in state

        real_index = table.get_row_index("s1")
        undefined_indexes = [table.get_row_index(sid) for sid in ("s2", "s3")]
        assert all(real_index < idx for idx in undefined_indexes), (
            "row with a real spread (s1) must sort before every "
            f"undefined-spread row in ascending mode: real={real_index}, "
            f"undefined={undefined_indexes}"
        )


@pytest.mark.asyncio
async def test_grid_autofocuses_its_table_so_shortcuts_work_without_a_manual_focus(
    evals_app, mixed_run_group
):
    """The footer advertises `l`/`b`/`s`/`e` the instant a run group is
    selected (see evals_screen.py's _register_grid_shortcuts), but every
    OTHER test in this file calls `table.focus()` before pressing a
    shortcut key -- which would hide a missing auto-focus, since Textual
    key bindings only resolve against the focused widget's ancestor chain.
    This test deliberately does NOT call `.focus()`, so it fails if
    ResultsGrid.on_mount stops focusing its own DataTable."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        select = grid.query_one("#evals-lens-selector", Select)
        assert select.value == "top1"

        await pilot.press("l")
        await pilot.pause()

        assert select.value == "entropy", (
            "the `l` shortcut had no effect -- the grid's DataTable was "
            "not focused on mount"
        )


@pytest.mark.asyncio
async def test_grid_shortcuts_register_in_the_footer_including_export(
    evals_app, mixed_run_group
):
    """Task 1 deliberately left `e` unbound and unadvertised ("Task 2's
    job" -- see this module's own docstring history); PR 3b Task 2 claims
    it for export, so the footer must now advertise it too."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        await _select_run_group(pilot, mixed_run_group["group_id"])
        footer = pilot.app.screen.query_one(AppFooterStatus)

        assert "l lens" in footer.shortcut_text
        assert "b baseline" in footer.shortcut_text
        assert "s sort" in footer.shortcut_text
        assert "e export" in footer.shortcut_text
        assert "e" in {b.key for b in ResultsGrid.BINDINGS}


@pytest.mark.asyncio
async def test_grid_shortcuts_clear_when_selection_leaves_the_run_group(
    evals_app, mixed_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        await _select_run_group(pilot, mixed_run_group["group_id"])
        footer = pilot.app.screen.query_one(AppFooterStatus)
        assert "l lens" in footer.shortcut_text

        pilot.app.screen.select(kind="none")
        await pilot.pause()
        footer = pilot.app.screen.query_one(AppFooterStatus)
        assert "l lens" not in footer.shortcut_text


@pytest.mark.asyncio
async def test_arrow_keys_move_focus_and_update_the_inspector_without_a_recompose(
    evals_app, mixed_run_group
):
    """The trap this pins: a naive fix might recompose the whole screen on
    every focus change, which would tear down and rebuild the grid's own
    DataTable (losing cursor position) on every arrow-key press. The same
    ResultsGrid instance must survive the key presses below."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        grid_identity = id(grid)
        table = grid.query_one("#evals-grid-table", DataTable)
        table.focus()
        await pilot.pause()

        body = pilot.app.screen.query_one("#evals-cell-inspector-body")
        first_text = str(body.renderable)

        await pilot.press("right")
        await pilot.pause()

        second_text = str(body.renderable)
        assert second_text != first_text, "moving focus with arrow keys must update the inspector"

        still_same_grid = pilot.app.screen.query_one("#evals-results-grid", ResultsGrid)
        assert id(still_same_grid) == grid_identity, (
            "arrow-key focus movement must not recompose the screen/grid"
        )


@pytest.mark.asyncio
async def test_probe_lens_renders_an_observed_probe_with_a_percentage(
    evals_app, mixed_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        grid.query_one("#evals-lens-selector", Select).value = "probe"
        await pilot.pause()

        table = grid.query_one("#evals-grid-table", DataTable)
        # probes=(" a",) and s1 x base observed " a" at 44%. Asserted
        # exactly: `"%" in text` alone would pass on any percentage from
        # any probe -- including the wrong probe or the wrong cell.
        text = str(table.get_cell("s1", mixed_run_group["base_id"]))
        assert text == f"{math.log(0.44):.2f}  44.0%", (
            f"expected the observed reading for probe ' a' at 44%, got {text!r}"
        )
        assert text != FAILED_MARK
        assert text != ""

        # The state line names WHICH probe produced it -- with one probe
        # configured, no "n of m" position is needed.
        state = str(grid.query_one("#evals-grid-state").renderable)
        assert 'Lens: Probe (" a")' in state, state


@pytest.mark.asyncio
async def test_a_run_group_with_no_snippets_renders_an_empty_state_instead_of_crashing(
    evals_app, empty_run_group
):
    """Regression: ``compose()``'s empty-snapshot branch returned WITHOUT
    yielding the DataTable, but ``on_mount`` only guarded ``self._grid is
    None`` -- so it went straight on to ``_render_table``'s
    ``query_one("#evals-grid-table")`` and raised ``NoMatches`` out of
    ``on_mount``, taking the whole app down on the one input that branch
    exists to handle."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        # Selecting it at all is the crash: an unhandled on_mount
        # exception tears down the Textual app.
        grid = await _select_run_group(pilot, empty_run_group["group_id"])

        empty_state = grid.query_one("#evals-grid-empty")
        assert "no snippets" in str(empty_state.renderable)
        assert not grid.query("#evals-grid-table")
        # And the shortcuts stay inert rather than crashing on a table
        # that was never built.
        grid.action_cycle_lens()
        grid.action_cycle_baseline()
        grid.action_cycle_sort()
        grid.action_export()
        await pilot.pause()
        assert pilot.app.screen.query_one("#evals-results-grid", ResultsGrid) is grid


@pytest.mark.asyncio
async def test_unexpected_load_grid_failure_renders_error_state_without_crashing_the_app(
    evals_app, mixed_run_group, monkeypatch
):
    """Regression (TASK-861 item 2): ``compose()`` only wrapped
    ``load_grid(...)`` in ``except ValueError`` -- the deliberate
    "no runs found for run group" case ``storage.load_grid`` raises on
    purpose. Anything else (a locked database, a disk error, a schema
    mismatch from an older profile -- ``Evals_DB.list_runs``/
    ``get_run_results`` have no exception wrapping of their own) came
    through unconverted. Textual's ``Widget._compose`` wraps the entire
    ``compose()`` call in ``except Exception`` and hands it to
    ``App._handle_exception``, whose own docstring says this "always
    results in the app exiting" -- so an unconverted sqlite fault here
    does not just break this widget, it silently kills the whole process.
    Reproduces the escape directly (a stand-in for the author's own
    "build a valid run group, DROP TABLE eval_results" repro) by making
    ``load_grid`` raise ``sqlite3.OperationalError`` for an otherwise
    valid run group id.
    """
    import sqlite3

    from tldw_chatbook.UI.Evals import results_grid as results_grid_module

    def _raise_operational_error(db, run_group_id):
        raise sqlite3.OperationalError("no such table: eval_results")

    monkeypatch.setattr(results_grid_module, "load_grid", _raise_operational_error)

    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        await pilot.pause()

        # The direct signal that ``compose()`` did NOT let the exception
        # escape: Textual only ever populates ``App._exception`` via
        # ``_handle_exception``, and its docstring says that "always
        # results in the app exiting". No exception recorded here means
        # the app is still alive, not merely that the crash hasn't been
        # observed yet.
        assert pilot.app._exception is None
        assert pilot.app.is_running

        error_state = grid.query_one("#evals-grid-error")
        message = str(error_state.renderable)
        assert "unexpected error" in message
        # Must not impersonate the deliberate, distinctly-worded
        # "no runs found" ValueError copy -- that path has its own
        # meaning and its own test.
        assert "may have been deleted" not in message
        assert not grid.query("#evals-grid-table")

        # And the shortcuts stay inert rather than crashing on a table
        # that was never built (mirrors the empty-state regression test
        # above).
        grid.action_cycle_lens()
        grid.action_cycle_baseline()
        grid.action_cycle_sort()
        grid.action_export()
        await pilot.pause()
        assert pilot.app.screen.query_one("#evals-results-grid", ResultsGrid) is grid


@pytest.mark.asyncio
async def test_truncation_lens_uses_the_shared_k_the_header_states(
    evals_app, k_depth_matched_run_group
):
    """Same misrepresentation the shared-K entropy rule exists to prevent,
    one lens over. s1's two cells hold the SAME distribution over the first
    5 ranks; base additionally has 15 real low-probability tokens beyond
    rank 5 (native K=20) while the capped target stops at 5. Read at each
    cell's own native K they show 1% vs 2% -- a 2x "difference" produced
    purely by the requested K, under a header that says ``K 5``.

    Uses ``k_depth_matched_run_group`` rather than ``clean_run_group``: see
    that fixture's own docstring for why (``clean_run_group``'s s2/s3 rows
    would drag a clamped ``effective_k`` down to their own tiny native
    length)."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, k_depth_matched_run_group["group_id"])
        grid.query_one("#evals-lens-selector", Select).value = "coverage"
        await pilot.pause()

        meta = str(grid.query_one("#evals-grid-meta").renderable)
        assert "· K 5 ·" in meta, meta

        table = grid.query_one("#evals-grid-table", DataTable)
        rich_text = str(table.get_cell("s1", k_depth_matched_run_group["base_id"]))
        poor_text = str(table.get_cell("s1", k_depth_matched_run_group["poor_id"]))
        assert rich_text == poor_text, (
            f"same distribution at the shared K {5} must produce equal "
            f"truncation: rich={rich_text!r} poor={poor_text!r}"
        )
        # Not vacuously equal-because-both-blank: this is the real figure,
        # 1 - (0.5+0.3+0.1+0.05+0.03) = 0.02 -> "2%".
        assert rich_text == "2%", rich_text

        # The cells' OWN native truncated_mass genuinely differ -- proof
        # the fixture discriminates rather than the two happening to agree.
        from tldw_chatbook.Evals.word_bench.storage import load_grid

        db = pilot.app.app_instance.evaluation_orchestrator.db
        cells = load_grid(db, k_depth_matched_run_group["group_id"])["cells"]
        native_rich = cells[("s1", k_depth_matched_run_group["base_id"])].truncated_mass
        native_poor = cells[("s1", k_depth_matched_run_group["poor_id"])].truncated_mass
        assert round(native_rich, 4) != round(native_poor, 4)


@pytest.mark.asyncio
async def test_probe_lens_names_the_active_probe_and_lets_the_user_switch_it(
    evals_app, multi_probe_run_group
):
    """A bench with two probes rendered only probes[0], with no selector and
    a state line that said just "Lens: Probe" -- numbers a reader could not
    attribute to a probe, and a second probe unreachable from the grid."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, multi_probe_run_group["group_id"])
        grid.query_one("#evals-lens-selector", Select).value = "probe"
        await pilot.pause()

        table = grid.query_one("#evals-grid-table", DataTable)
        base_id = multi_probe_run_group["base_id"]

        state = str(grid.query_one("#evals-grid-state").renderable)
        assert 'Lens: Probe (" Sure" · 1 of 2)' in state, state
        assert str(table.get_cell("s1", base_id)) == f"{math.log(0.6):.2f}  60.0%"

        # The second probe is reachable, and switching to it changes both
        # the state line and the rendered number. The selector has to be
        # genuinely usable, not merely in the DOM -- three 1fr Selects now
        # share the control row.
        probe_select = grid.query_one("#evals-probe-selector", Select)
        pane = pilot.app.screen.query_one("#evals-detail-pane")
        assert probe_select.region.width > 0 and probe_select.region.height > 0
        assert pane.region.contains_region(probe_select.region), (
            f"probe selector at {probe_select.region} escapes {pane.region}"
        )
        probe_select.value = 1
        await pilot.pause()

        state = str(grid.query_one("#evals-grid-state").renderable)
        assert 'Lens: Probe (" Cannot" · 2 of 2)' in state, state
        assert str(table.get_cell("s1", base_id)) == f"{math.log(0.2):.2f}  20.0%"


@pytest.mark.asyncio
async def test_probe_percentage_follows_the_engines_matched_token_not_a_local_rematch(
    evals_app, mixed_run_group, monkeypatch
):
    """I5: the Probe lens and the focused-cell inspector each re-derived
    ``resolve_probe``'s ``token == probe`` match with their own copy of the
    predicate, so a change to the engine's matching rule (most obviously
    moving it to the bytes-based ``TokenProb.identity()`` the rest of
    ``analysis`` aligns on) would silently not reach either renderer.

    Simulated here by changing the engine's rule -- this fake matches rank
    2 (" the", 43%) rather than rank 1 (" a", 44%). Both renderers must
    follow the ``ProbeReading`` the engine returned; a local re-match would
    keep printing 44%.
    """
    real_resolve = analysis.resolve_probe

    def _rank_two_resolve(cap, probe, *, ever_observed):
        if len(cap.top_k) > 1:
            tok = cap.top_k[1]
            return analysis.ProbeReading(
                probe=probe, state="observed", logprob=tok.logprob, matched=tok
            )
        return real_resolve(cap, probe, ever_observed=ever_observed)

    monkeypatch.setattr(analysis, "resolve_probe", _rank_two_resolve)

    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        grid.query_one("#evals-lens-selector", Select).value = "probe"
        await pilot.pause()

        table = grid.query_one("#evals-grid-table", DataTable)
        base_id = mixed_run_group["base_id"]
        expected = f"{math.log(0.43):.2f}  43.0%"
        assert str(table.get_cell("s1", base_id)) == expected, (
            "the grid re-derived the match instead of using "
            "ProbeReading.matched"
        )

        row = table.get_row_index("s1")
        col = table.get_column_index(base_id)
        table.focus()
        table.move_cursor(row=row, column=col)
        await pilot.pause()

        body = pilot.app.screen.query_one("#evals-cell-inspector-body")
        text = str(body.renderable)
        assert f'" a": {expected}' in text, (
            f"the inspector re-derived the match instead of using "
            f"ProbeReading.matched: {text!r}"
        )


def test_resolve_probe_hands_back_the_token_it_matched():
    """The engine no longer discards the ``TokenProb`` it found -- the
    identity assertion is what makes a caller-side re-derivation
    unnecessary (and pins that ``matched`` is the engine's own object, not
    a reconstruction)."""
    cap = _cap([(" a", 0.44), (" the", 0.43)])
    reading = analysis.resolve_probe(cap, " the", ever_observed=True)
    assert reading.state == "observed"
    assert reading.matched is cap.top_k[1]

    bounded = analysis.resolve_probe(cap, " nope", ever_observed=True)
    assert bounded.state == "bounded"
    assert bounded.matched is None


@pytest.mark.asyncio
async def test_spread_column_only_appears_in_the_delta_lens(evals_app, clean_run_group):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, clean_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)

        labels_top1 = [str(col.label) for col in table.columns.values()]
        assert not any("Spread" in label for label in labels_top1)

        grid.query_one("#evals-lens-selector", Select).value = "delta"
        await pilot.pause()
        labels_delta = [str(col.label) for col in table.columns.values()]
        assert any("Spread" in label for label in labels_delta)


@pytest.mark.asyncio
async def test_group_mean_rows_match_analysis_group_means_over_the_rendered_divergences(
    evals_app, clean_run_group
):
    """Pins the WIRING, not analysis.group_means' own arithmetic (that is
    analysis.py's own test suite's job): the grid must feed exactly the
    divergence values it rendered, grouped by each snippet's `group`, into
    ``analysis.group_means`` -- computed here the same way, from the same
    engine calls, for comparison."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, clean_run_group["group_id"])
        grid.query_one("#evals-lens-selector", Select).value = "delta"
        await pilot.pause()

        table = grid.query_one("#evals-grid-table", DataTable)
        # "loaded" group is s2 and s3; baseline column is base (t1), so the
        # real divergence values live in the poor_id column.
        db = pilot.app.app_instance.evaluation_orchestrator.db
        from tldw_chatbook.Evals.word_bench.storage import load_grid

        raw = load_grid(db, clean_run_group["group_id"])
        cells = raw["cells"]
        base_id, poor_id = clean_run_group["base_id"], clean_run_group["poor_id"]
        loaded_values = []
        for sid in ("s2", "s3"):
            jsd, _ = analysis.divergence(cells[(sid, poor_id)], cells[(sid, base_id)])
            loaded_values.append(jsd)
        expected = sum(loaded_values) / len(loaded_values)

        row_key = None
        for row_index in range(table.row_count):
            snippet_cell = str(table.get_cell_at((row_index, 0)))
            if snippet_cell.startswith("group mean") and "loaded" in snippet_cell:
                row_key = row_index
                break
        assert row_key is not None, "expected a 'group mean [loaded]' row in the delta lens"
        col_index = table.get_column_index(poor_id)
        rendered = str(table.get_cell_at((row_key, col_index)))
        assert rendered == f"{expected:.2f}"
