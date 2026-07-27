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
from textual.widgets import DataTable, Select

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


@pytest.mark.asyncio
async def test_entropy_lens_states_effective_k_and_uses_it_for_every_cell(
    evals_app, clean_run_group
):
    """A K=20 cell and a K=5 cell holding the SAME underlying distribution
    (over the first 5 ranks) must read identical entropy once both are
    read at the shared effective K -- otherwise the K=20 column would look
    "more confident" purely from its setting."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, clean_run_group["group_id"])
        select = grid.query_one("#evals-lens-selector", Select)
        select.value = "entropy"
        await pilot.pause()

        meta = str(grid.query_one("#evals-grid-meta").renderable)
        # Delimited, not a bare substring: "K 5" alone also matches "K 50".
        assert "· K 5 ·" in meta, f"header must state the effective K (5), got {meta!r}"

        table = grid.query_one("#evals-grid-table", DataTable)
        rich_text = str(table.get_cell("s1", clean_run_group["base_id"]))
        poor_text = str(table.get_cell("s1", clean_run_group["poor_id"]))
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
    evals_app, clean_run_group
):
    """Same misrepresentation the shared-K entropy rule exists to prevent,
    one lens over. s1's two cells hold the SAME distribution over the first
    5 ranks; base additionally has 15 real low-probability tokens beyond
    rank 5 (native K=20) while the capped target stops at 5. Read at each
    cell's own native K they show 1% vs 2% -- a 2x "difference" produced
    purely by the requested K, under a header that says ``K 5``."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, clean_run_group["group_id"])
        grid.query_one("#evals-lens-selector", Select).value = "coverage"
        await pilot.pause()

        meta = str(grid.query_one("#evals-grid-meta").renderable)
        assert "· K 5 ·" in meta, meta

        table = grid.query_one("#evals-grid-table", DataTable)
        rich_text = str(table.get_cell("s1", clean_run_group["base_id"]))
        poor_text = str(table.get_cell("s1", clean_run_group["poor_id"]))
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
        cells = load_grid(db, clean_run_group["group_id"])["cells"]
        native_rich = cells[("s1", clean_run_group["base_id"])].truncated_mass
        native_poor = cells[("s1", clean_run_group["poor_id"])].truncated_mass
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
