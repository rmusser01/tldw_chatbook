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
    NEAR_TIE_LOGPROB_GAP_NATS,
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
    ]
    group_id, run_ids = create_run_group(evals_db, task_id, config, targets, snippets)

    # Same underlying distribution at both K=20 (base) and K=5 (poor) for
    # s1, so entropy at the shared effective K (5) must read identically.
    same_dist = [(" a", 0.5), (" the", 0.3), (" an", 0.1), (" some", 0.05), (" one", 0.03)]
    save_cell(evals_db, run_ids[base_id], snippets[0], _cap(same_dist, k_returned=20))
    save_cell(evals_db, run_ids[poor_id], snippets[0], _cap(same_dist[:5], k_returned=5))

    # s2: base diverges a lot from poor; s3: base diverges a little.
    save_cell(evals_db, run_ids[base_id], snippets[1], _cap([(" a", 0.9)], k_returned=20))
    save_cell(evals_db, run_ids[poor_id], snippets[1], _cap([(" not", 0.9)], k_returned=5))
    save_cell(evals_db, run_ids[base_id], snippets[2], _cap([(" it", 0.6), (" the", 0.3)], k_returned=20))
    save_cell(evals_db, run_ids[poor_id], snippets[2], _cap([(" it", 0.55), (" the", 0.3)], k_returned=5))

    return {"group_id": group_id, "base_id": base_id, "poor_id": poor_id}


async def _select_run_group(pilot, group_id: str) -> ResultsGrid:
    screen: EvalsScreen = pilot.app.screen
    screen.select(kind="run_group", id=group_id)
    await pilot.pause()
    return screen.query_one("#evals-results-grid", ResultsGrid)


# ---------------------------------------------------------------------------
# Pure-function unit tests: no Textual app needed at all.
# ---------------------------------------------------------------------------


def test_near_tie_threshold_is_a_named_constant_not_a_magic_number():
    """Pins the value itself, not just that a threshold exists -- see
    ``results_grid.NEAR_TIE_LOGPROB_GAP_NATS``'s own docstring for the
    observed-instability rationale (a ~0.095-0.096 nat gap already produced
    a rank flip; this codebase's normalizer test independently drew the
    same 0.15 nat boundary for the identical fixture)."""
    assert NEAR_TIE_LOGPROB_GAP_NATS == 0.15


def test_render_token_makes_whitespace_visible():
    """`" a"` and `"a"` must not render identically -- a bare leading space
    is invisible in a terminal cell, especially with DataTable padding."""
    assert render_token(" a") == '" a"'
    assert render_token("a") == '"a"'
    assert render_token(" a") != render_token("a")


def test_render_probe_reading_covers_all_three_states_distinctly():
    observed = analysis.ProbeReading(probe=" Sure", state="observed", logprob=math.log(0.6))
    bounded = analysis.ProbeReading(probe=" Sure", state="bounded", logprob=math.log(0.9))
    never = analysis.ProbeReading(probe=" Sure", state="never_observed", logprob=None)

    matched = TokenProb(token=" Sure", logprob=math.log(0.6))
    observed_text = render_probe_reading(observed, matched)
    bounded_text = render_probe_reading(bounded, None)
    never_text = render_probe_reading(never, None)

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
        assert "K 5" in meta, f"header must state the effective K (5), got {meta!r}"

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
async def test_sort_key_registers_and_reorders_by_spread(evals_app, mixed_run_group):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)
        table.focus()

        state_before = str(grid.query_one("#evals-grid-state").renderable)
        assert "Sort: dataset order" in state_before

        await pilot.press("s")
        await pilot.pause()
        state_after = str(grid.query_one("#evals-grid-state").renderable)
        assert "Sort: spread" in state_after


@pytest.mark.asyncio
async def test_grid_shortcuts_register_in_the_footer_and_export_key_is_left_free(
    evals_app, mixed_run_group
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        await _select_run_group(pilot, mixed_run_group["group_id"])
        footer = pilot.app.screen.query_one(AppFooterStatus)

        assert "l lens" in footer.shortcut_text
        assert "b baseline" in footer.shortcut_text
        assert "s sort" in footer.shortcut_text
        # Export (`e`) is Task 2's job -- the key must stay unadvertised and
        # unbound so a later PR can claim it without a collision.
        assert "e export" not in footer.shortcut_text
        assert "e" not in {b.key for b in ResultsGrid.BINDINGS}


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
        # probes=(" a",) and s1 x base observed " a" at 44%.
        text = str(table.get_cell("s1", mixed_run_group["base_id"]))
        assert "%" in text
        assert text != FAILED_MARK
        assert text != ""


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
