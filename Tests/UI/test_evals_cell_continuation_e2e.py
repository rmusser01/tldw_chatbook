"""End-to-end: an opt-in PER-CELL continuation (task-1710), captured,
persisted, and rendered end to end -- and its flag-off control.

task-1691 built a per-TARGET continuation of one fixed canary prompt,
captured once per target at preflight and shown in the READINESS pane
(``test_evals_continuation_e2e.py``). task-1710 is the per-CELL sibling:
an opt-in continuation of EACH SNIPPET actually measured, captured once
per (snippet, target) cell during the real run, and shown in the
focused-cell inspector (``EvalsCellInspector``) alongside that cell's own
top-K -- a different instrument, answering "what would the model say
after THIS snippet" rather than "what does this target do with a known
canary prompt".

Neither T1 (``capture_client.py``/``runner.py``/``storage.py``, engine +
persistence, tested against fakes/mocks only) nor T2 (``bench_editor.py``/
``inspector.py``, UI, tested against synthetic ``CellCapture``/
``BenchConfig`` fixtures built directly, never through a real run) drove a
REAL run through the screen's own worker (``EvalsScreen.
_sample_bench_client_factory`` -> ``WordBenchRunner.run`` ->
``create_run_group(...)``) and then read a captured continuation back off
a FRESH grid load (a genuine DB round-trip through ``word_bench.storage.
load_grid``, never the in-memory ``CellCapture`` the worker just built).
That joint -- and this task's own whole premise versus task-1691's
(a continuation captured PER CELL, so two different snippets against the
SAME target render two DIFFERENT continuations, never one continuation
reused for every row the way task-1691's per-target canary continuation
is) -- is this file's point.

Same harness and opening moves as ``test_evals_continuation_e2e.py``/
``test_evals_authoring_e2e.py``: the rail's Import flow (a real temp
file, bypassing the FileOpen modal), "+ New bench", the zero-
``llama_cpp``-models "Create target" button, Save, Run, then reading the
grid/cell inspector back.
"""

from __future__ import annotations

import pytest
from textual.widgets import Checkbox, DataTable

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import CellCapture, PreflightResult, Target, TokenProb
from tldw_chatbook.UI.Evals import inspector as inspector_module
from tldw_chatbook.UI.Evals.library_rail import LibraryRail
from tldw_chatbook.UI.Evals.results_grid import ResultsGrid
from tldw_chatbook.UI.Evals.snippet_editor import dataset_snippets
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen

from .test_evals_screen import EvalsHarness, _FakeAppInstance

_REALISTIC_SIZE = (160, 45)

#: Two snippets, two DIFFERENT continuations, against the SAME one
#: target -- the direct proof this is captured per CELL, not reused
#: across every row the way task-1691's per-target canary continuation
#: is (every row of a bench with one target would render the identical
#: string if this were a per-target capture reused per row instead).
#: Each continuation also exercises one of the two markers this pane's
#: rendering rules apply (task-1691's own convention, reused verbatim --
#: see ``inspector._cell_continuation_text``): leading anomalous
#: whitespace ("␣") for the "sky" continuation, an embedded newline
#: ("⏎") for the "grass" one.
_SKY_SNIPPET = "The sky is"
_GRASS_SNIPPET = "The grass is"
_SKY_CONTINUATION = "  scaffolding drifting toward blue"
_GRASS_CONTINUATION = "green\nvery green today"

_CONTINUATION_BY_SNIPPET = {
    _SKY_SNIPPET: _SKY_CONTINUATION,
    _GRASS_SNIPPET: _GRASS_CONTINUATION,
}


class _PerCellContinuationCaptureClient:
    """A fake capture client whose ``capture_with_continuation`` hands
    back a DIFFERENT continuation keyed off the snippet text actually
    passed in -- unlike every sibling fake client in this suite (e.g.
    ``test_evals_continuation_e2e.py``'s ``_ContinuationCaptureClient``,
    whose ``preflight()`` returns ONE fixed continuation no matter which
    target asks, or ``test_evals_authoring_e2e.py``'s
    ``_TwoTargetFakeCaptureClient``, keyed per-TARGET), because this
    task's own premise is per-CELL capture: a bench with several snippets
    against one target must be able to render several DIFFERENT
    continuations, not the same one reused for every row.

    ``capture()`` (the flag-off path, ``BenchConfig.capture_continuations
    = False``) returns the identical cell shape with NO continuation
    (``CellCapture.continuation``'s own ``""`` default) -- the flag-off
    control case renders nothing extra.
    """

    def __init__(self, target: Target) -> None:
        self._target = target

    async def preflight(self, target: Target, mode: str, top_k: int) -> PreflightResult:
        return PreflightResult(state="ok", k_returned=2, canary="pass")

    async def capture(
        self, snippet: str, target: Target, mode: str, top_k: int
    ) -> CellCapture:
        return self._cell()

    async def capture_with_continuation(
        self, snippet: str, target: Target, mode: str, top_k: int
    ) -> tuple[CellCapture, str]:
        return self._cell(), _CONTINUATION_BY_SNIPPET.get(snippet, "")

    @staticmethod
    def _cell() -> CellCapture:
        return CellCapture(
            prompt_mode="raw", k_requested=2, k_returned=2, content_offset=0,
            top_k=(
                TokenProb(token=" blue", logprob=-0.2, token_id=1),
                TokenProb(token=" grey", logprob=-1.5, token_id=2),
            ),
            canary="unchecked", captured_at="2026-07-31T00:00:00Z",
        )


@pytest.fixture
def evals_db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def cell_continuation_app(evals_db: EvalsDB) -> EvalsHarness:
    """A configured llama.cpp endpoint -- needed for the bench editor's
    zero-``llama_cpp``-models "Create target" mini-form (``evals_screen.
    py``'s own handler gates on ``sample_bench.configured_llama_cpp_url``
    before it will write a row). Mirrors ``test_evals_continuation_e2e.
    py``'s own ``continuation_app``."""
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    return EvalsHarness(_FakeAppInstance(evals_db, app_config=app_config))


async def _wait_until(pilot, predicate, *, tries: int = 300, interval: float = 0.02) -> None:
    """Mirrors the sibling authoring/steering/continuation E2E files' own
    helper -- polls until a background worker's completion becomes
    visible (a selection change), since ``run_worker`` schedules real
    async work that does not finish within a single ``pilot.pause()``."""
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause(interval)
    raise AssertionError("condition never became true")


async def _focus_cell(pilot, grid: ResultsGrid, snippet_id: str, target_id: str) -> None:
    """Mirrors ``test_evals_results_grid.py``'s own private helper of the
    identical name/shape (not imported from there -- that module's own
    comment scopes it to its own continuation-tests section)."""
    table = grid.query_one("#evals-grid-table", DataTable)
    row = table.get_row_index(snippet_id)
    col = table.get_column_index(target_id)
    table.focus()
    table.move_cursor(row=row, column=col)
    await pilot.pause()


@pytest.mark.asyncio
async def test_continuation_captured_per_cell_and_survives_reload(
    cell_continuation_app, evals_db, tmp_path
):
    """Import a 2-snippet dataset -> "+ New bench" -> create one target ->
    flip the "capture a continuation" checkbox on -> Save -> Run -> the
    focused-cell inspector for EACH snippet shows its OWN captured
    continuation, alongside that cell's own top-K -- then the continuation
    survives a fresh DB round-trip (select away to the bench and back to
    the run group, forcing ``results_grid.py``'s ``ResultsGrid.compose()``
    to call ``storage.load_grid`` again from scratch, never reusing the
    in-memory ``CellCapture`` the worker just built).
    """
    import_path = tmp_path / "imported.txt"
    import_path.write_text(f"{_SKY_SNIPPET}\n{_GRASS_SNIPPET}\n", encoding="utf-8")

    async with cell_continuation_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        screen: EvalsScreen = pilot.app.screen
        # Shared by the sample-bench AND bench-run workers (see
        # evals_screen.py's own field docstring) -- this loop only ever
        # exercises the bench-run path below, but the seam is the same
        # one either worker reads.
        screen._sample_bench_client_factory = lambda t: _PerCellContinuationCaptureClient(t)

        # -- Import a 2-snippet dataset via the rail's own Import flow
        # (bypasses the FileOpen modal -- established convention, see
        # library_rail.py's own `_handle_dataset_import_file_selected`
        # docstring).
        rail = screen.query_one(LibraryRail)
        rail._handle_dataset_import_file_selected(import_path)
        await pilot.pause()
        assert screen._selection.kind == "dataset"
        dataset_id = screen._selection.id
        snippets = dataset_snippets(evals_db.get_dataset(dataset_id))
        assert len(snippets) == 2
        sky_id = next(s["id"] for s in snippets if s["text"] == _SKY_SNIPPET)
        grass_id = next(s["id"] for s in snippets if s["text"] == _GRASS_SNIPPET)

        # -- "+ New bench" against the just-imported (and still selected)
        # dataset -- a draft bench with zero targets.
        await pilot.click("#evals-rail-new-bench")
        await pilot.pause()
        assert screen._selection.kind == "bench"
        bench_id = screen._selection.id

        # -- Target: the zero-`llama_cpp`-models "Create target" path. No
        # `llama_cpp` `eval_models` row exists anywhere yet, so the
        # button (never the Add picker) renders.
        assert evals_db.list_models(provider="llama_cpp") == []
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()
        created = evals_db.list_models(provider="llama_cpp")
        assert len(created) == 1, "Create target must mint exactly one row"
        target_id = created[0]["id"]

        # -- The opt-in itself: off by default (`BenchConfig.
        # capture_continuations`'s own dataclass default), flipped on via
        # the UI before Save -- the very thing that must be deliberately
        # chosen, per this task's own description.
        checkbox = screen.query_one("#evals-bench-capture-continuations", Checkbox)
        assert checkbox.value is False
        checkbox.value = True

        # -- Save persists the staged target AND the opt-in together.
        await pilot.click("#evals-bench-save")
        await pilot.pause()
        assert not screen.query_one("#evals-bench-form-error").display

        # -- Run. `_PerCellContinuationCaptureClient.capture_with_
        # continuation` above is what actually captures a continuation
        # per cell, exactly the way a real `WordBenchCaptureClient.
        # capture_with_continuation` would (T1) -- this test never calls
        # it directly, only through the screen's own worker.
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._selection.kind == "run_group")
        await pilot.pause()
        run_group_id = screen._selection.id

        grid = screen.query_one("#evals-results-grid", ResultsGrid)
        table = grid.query_one("#evals-grid-table", DataTable)
        assert table.row_count == 2

        # -- Focus the "sky" cell: its own continuation (leading
        # anomalous whitespace made visible via the ␣ marker) rendered
        # alongside its own top-K -- both instruments, in the same pane.
        await _focus_cell(pilot, grid, sky_id, target_id)
        continuation = screen.query_one("#evals-cell-inspector-continuation")
        assert continuation.display is True
        assert continuation.region.width > 0
        assert continuation.region.height > 0
        text = str(continuation.renderable)
        assert text.startswith(inspector_module._CELL_CONTINUATION_LABEL)
        assert "scaffolding drifting toward blue" in text
        assert "␣" in text
        assert "\n" not in text

        body = screen.query_one("#evals-cell-inspector-body")
        body_text = str(body.renderable)
        assert "Top-K:" in body_text
        assert " blue" in body_text

        # -- Focus the "grass" cell: a DIFFERENT continuation for the
        # SAME target -- the direct proof this is captured per CELL, not
        # reused across every row the way task-1691's per-target canary
        # continuation is.
        await _focus_cell(pilot, grid, grass_id, target_id)
        continuation = screen.query_one("#evals-cell-inspector-continuation")
        assert continuation.display is True
        text = str(continuation.renderable)
        assert text.startswith(inspector_module._CELL_CONTINUATION_LABEL)
        assert "very green today" in text
        assert "scaffolding" not in text
        assert "⏎" in text
        assert "\n" not in text

        # -- Reload: select away (the bench) and back (the run group) --
        # a fresh `storage.load_grid` read through a brand-new
        # `ResultsGrid` instance, never the in-memory `CellCapture` the
        # worker just built or any cached grid state.
        screen.select(kind="bench", id=bench_id)
        await pilot.pause()
        screen.select(kind="run_group", id=run_group_id)
        await pilot.pause()
        grid = screen.query_one("#evals-results-grid", ResultsGrid)
        await _focus_cell(pilot, grid, sky_id, target_id)
        continuation = screen.query_one("#evals-cell-inspector-continuation")
        assert continuation.display is True
        text = str(continuation.renderable)
        assert "scaffolding drifting toward blue" in text
        assert "␣" in text


@pytest.mark.asyncio
async def test_flag_off_run_renders_no_continuation_for_the_same_cell(
    cell_continuation_app, evals_db, tmp_path
):
    """The control case: a bench with `capture_continuations` left at its
    default (off) renders the SAME "sky" snippet/target cell shape
    WITHOUT any continuation line -- the rest of the cell inspector (the
    top-K) renders exactly as it always has. Proves the opt-in genuinely
    gates the behavior end to end, through the real UI/Save/Run loop, not
    only in the unit-level engine (T1) and UI (T2) tests that already
    cover this in isolation against synthetic fixtures.
    """
    import_path = tmp_path / "imported.txt"
    import_path.write_text(f"{_SKY_SNIPPET}\n", encoding="utf-8")

    async with cell_continuation_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        screen: EvalsScreen = pilot.app.screen
        screen._sample_bench_client_factory = lambda t: _PerCellContinuationCaptureClient(t)

        rail = screen.query_one(LibraryRail)
        rail._handle_dataset_import_file_selected(import_path)
        await pilot.pause()
        assert screen._selection.kind == "dataset"
        dataset_id = screen._selection.id
        snippets = dataset_snippets(evals_db.get_dataset(dataset_id))
        assert len(snippets) == 1
        sky_id = snippets[0]["id"]

        await pilot.click("#evals-rail-new-bench")
        await pilot.pause()
        assert screen._selection.kind == "bench"

        assert evals_db.list_models(provider="llama_cpp") == []
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()
        created = evals_db.list_models(provider="llama_cpp")
        assert len(created) == 1, "Create target must mint exactly one row"
        target_id = created[0]["id"]

        # -- The opt-in is left OFF -- the checkbox's own loaded-from-
        # draft default -- deliberately no interaction with
        # `#evals-bench-capture-continuations` beyond reading it.
        checkbox = screen.query_one("#evals-bench-capture-continuations", Checkbox)
        assert checkbox.value is False

        await pilot.click("#evals-bench-save")
        await pilot.pause()
        assert not screen.query_one("#evals-bench-form-error").display

        # -- Run. With the flag off, `WordBenchRunner._capture_cell`
        # dispatches to `client.capture()` only -- this fake's own
        # `capture_with_continuation` is never called on this path (see
        # `runner.py`'s own `_capture_cell` docstring: "the byte-
        # identical, single-request path this task's AC #2 pins").
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._selection.kind == "run_group")
        await pilot.pause()

        grid = screen.query_one("#evals-results-grid", ResultsGrid)
        table = grid.query_one("#evals-grid-table", DataTable)
        assert table.row_count == 1

        await _focus_cell(pilot, grid, sky_id, target_id)

        continuation = screen.query_one("#evals-cell-inspector-continuation")
        assert continuation.display is False
        assert str(continuation.renderable) == ""

        body = screen.query_one("#evals-cell-inspector-body")
        body_text = str(body.renderable)
        assert "Top-K:" in body_text
        assert " blue" in body_text
