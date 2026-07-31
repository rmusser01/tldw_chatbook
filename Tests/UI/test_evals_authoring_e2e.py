"""End-to-end: the whole bench-authoring loop, both cross-target lenses.

task-1482's final task. Every seam this program built gets driven together
here, through ONE authored bench: the rail's Import flow (a brand-new
dataset from a real file), "+ New bench" (a draft bench bound to it), the
bench editor's field form (rename/top-K/probes) AND its target editing
(the zero-models "Create target" button, then the Add picker for a second,
different target), Save -> the screen re-selecting and reloading the form
from what was actually persisted, Run, and the results grid's Probe and Δ
baseline lenses.

The fake capture client below is the one deliberate difference from every
other Evals worker test in this suite (see ``test_evals_empty_states.py``'s
``_FakeCaptureClient``/``_PausableFakeCaptureClient``): it hands its TWO
targets genuinely DIFFERENT top-K distributions, keyed off the target's own
name. Every other fake client in this codebase returns one shared
distribution for every target -- fine for a one-target sample bench, but it
would make this test's own two cross-target lenses vacuous: the Δ lens's
Spread column (a Jensen-Shannon divergence across targets) would read a
real but uninteresting ~0.0 for every row, and the Probe lens's per-target
hit/miss distinction would just show the SAME number twice, proving
nothing about whether the grid tells the columns apart at all.
"""

from __future__ import annotations

import math

import pytest
from textual.widgets import DataTable, Input, Select, TextArea

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import CellCapture, PreflightResult, Target, TokenProb
from tldw_chatbook.UI.Evals.library_rail import LibraryRail, _run_group_row_label
from tldw_chatbook.UI.Evals.results_grid import ResultsGrid
from tldw_chatbook.UI.Evals.snippet_editor import dataset_snippets
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen

from .test_evals_screen import EvalsHarness, _FakeAppInstance

_REALISTIC_SIZE = (160, 45)

#: The picker-added (second) target's own name -- the fake client below
#: keys its distribution off this exact, known name. The FIRST
#: (create-target-minted) target's name is whatever ``sample_bench.
#: _unique_name(BENCH_EDITOR_TARGET_NAME)`` mints (an opaque, suffixed
#: string this test never hardcodes), so the fake's "everything else"
#: branch covers it instead.
_SECOND_TARGET_NAME = "second-target"


class _TwoTargetFakeCaptureClient:
    """Mirrors this suite's established fake-capture-client shape
    (``client_factory=lambda t: ...``, per ``test_evals_empty_states.py``),
    except keyed per-target: the create-target-minted target's captures
    put the bench's own configured probes (``" Sure"``/``" I"``) at the top
    of the distribution; the picker-added target's captures never emit
    either. See the module docstring for why this test needs two
    genuinely different distributions rather than the shared-fake
    convention every other Evals worker test uses.
    """

    def __init__(self, target: Target) -> None:
        self._target = target

    async def preflight(self, target: Target, mode: str, top_k: int) -> PreflightResult:
        return PreflightResult(state="ok", k_returned=3, canary="pass")

    async def capture(
        self, snippet: str, target: Target, mode: str, top_k: int
    ) -> CellCapture:
        if target.name == _SECOND_TARGET_NAME:
            pairs = ((" the", 0.6), (" a", 0.25), (" an", 0.15))
        else:
            pairs = ((" Sure", 0.5), (" I", 0.3), (" a", 0.2))
        top = tuple(
            TokenProb(token=tok, logprob=math.log(p), token_id=index)
            for index, (tok, p) in enumerate(pairs)
        )
        return CellCapture(
            prompt_mode=mode,
            k_requested=top_k,
            k_returned=len(top),
            content_offset=0,
            top_k=top,
            canary="unchecked",
            captured_at="2026-07-26T00:00:00Z",
        )


@pytest.fixture
def evals_db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def authoring_app(evals_db: EvalsDB) -> EvalsHarness:
    """A configured llama.cpp endpoint (needed for the bench editor's
    zero-models "Create target" affordance -- ``sample_bench.
    resolve_sample_target(..., create=True)`` needs a real configured
    endpoint to mint a row from) and zero pre-existing benches/targets --
    mirrors ``test_evals_empty_states.py``'s own ``configured_app``."""
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    return EvalsHarness(_FakeAppInstance(evals_db, app_config=app_config))


async def _wait_until(pilot, predicate, *, tries: int = 300, interval: float = 0.02) -> None:
    """Mirrors ``test_evals_screen.py``'s own helper -- polls until a
    background worker's completion becomes visible (a selection change),
    since ``run_worker`` schedules real async work that does not finish
    within a single ``pilot.pause()``."""
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause(interval)
    raise AssertionError("condition never became true")


@pytest.mark.asyncio
async def test_authoring_loop_lights_up_both_cross_target_lenses(
    authoring_app, evals_db, tmp_path
):
    """Import a dataset -> "+ New bench" -> author it (rename, two targets
    via the create-target button THEN the Add picker, probes, top-K) ->
    Save -> Run -> the results grid's Probe and Δ baseline lenses both
    render real, non-degenerate, per-target-distinct content, and the rail
    marks the finished run with the plain "✓" glyph.
    """
    import_path = tmp_path / "imported.txt"
    import_path.write_text(
        "The protestors were\n"
        "The rioters were\n"
        "The government said\n"
        "The regime said\n",
        encoding="utf-8",
    )

    async with authoring_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        screen: EvalsScreen = pilot.app.screen
        # Shared by the sample-bench AND bench-run workers (see
        # evals_screen.py's own field docstring) -- this authoring loop
        # only ever exercises the bench-run path below, but the seam is
        # the same one either worker reads.
        screen._sample_bench_client_factory = lambda t: _TwoTargetFakeCaptureClient(t)

        # -- Import a 4-snippet dataset via the rail's own Import flow,
        # driving the picker's public-shaped callback directly with a real
        # temp file (bypasses the FileOpen modal -- established convention,
        # see library_rail.py's own `_handle_dataset_import_file_selected`
        # docstring and test_evals_empty_states.py's
        # test_import_dataset_file_selected_creates_a_dataset_from_the_file).
        rail = screen.query_one(LibraryRail)
        rail._handle_dataset_import_file_selected(import_path)
        await pilot.pause()
        assert screen._selection.kind == "dataset"
        dataset_id = screen._selection.id
        assert len(evals_db.list_datasets()) == 1
        snippets = dataset_snippets(evals_db.get_dataset(dataset_id))
        assert len(snippets) == 4

        # -- "+ New bench" against the just-imported (and still selected)
        # dataset -- a draft bench with zero targets.
        await pilot.click("#evals-rail-new-bench")
        await pilot.pause()
        assert screen._selection.kind == "bench"
        bench_id = screen._selection.id
        draft_bench = screen._view_model.bench_by_id(bench_id)
        assert draft_bench.get("dataset_id") == dataset_id

        # -- Target 1: the zero-`llama_cpp`-models "Create target" path.
        # No `llama_cpp` `eval_models` row exists anywhere yet, so the
        # button (never the Add picker) renders.
        assert evals_db.list_models(provider="llama_cpp") == []
        assert not screen.query("#evals-bench-add-target")
        assert screen.query_one("#evals-bench-create-target")
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()
        created = evals_db.list_models(provider="llama_cpp")
        assert len(created) == 1, "Create target must mint exactly one row"
        first_target_id = created[0]["id"]
        assert screen.query_one("#evals-bench-target-0")

        # A second, genuinely different `llama_cpp` target -- simulates
        # one already configured/created elsewhere (e.g. an earlier
        # session). This can only ever be a direct DB write in this test:
        # the "Create target" button reuses an existing row rather than
        # minting a second one the moment ANY `llama_cpp` row exists (see
        # `sample_bench.resolve_sample_target`'s own reuse-first docstring)
        # -- so a second click here would just re-stage `first_target_id`,
        # never reach the Add picker at all.
        second_target_id = evals_db.create_model(
            name=_SECOND_TARGET_NAME, provider="llama_cpp", model_id="m2"
        )

        # Save persists the create-target-staged first target and
        # re-selects the bench, recomposing the editor from storage --
        # this recompose is what makes the just-seeded second target
        # reachable at all: `_build_target_add_control` reads
        # `EvalsViewModel.llama_targets()` fresh on every compose, and the
        # picker only ever renders once that list is non-empty.
        await pilot.click("#evals-bench-save")
        await pilot.pause()
        assert not screen.query_one("#evals-bench-form-error").display

        # -- Target 2: the Add picker now offers the second, addable
        # target (not staged-filtered -- see bench_editor.py's own
        # docstring; the already-staged first target is also a listed
        # option, deliberately not picked here).
        picker = screen.query_one("#evals-bench-add-target", Select)
        picker.value = second_target_id
        await pilot.click("#evals-bench-add-target-button")
        await pilot.pause()
        assert not screen.query_one("#evals-bench-form-error").display
        assert screen.query_one("#evals-bench-target-1")

        # -- Rename, probes, top-K -- then the final Save.
        screen.query_one("#evals-bench-name", Input).value = "authored bench"
        screen.query_one("#evals-bench-top-k", Input).value = "20"
        screen.query_one("#evals-bench-probes", TextArea).text = " Sure\n I"

        await pilot.click("#evals-bench-save")
        await pilot.pause()
        assert not screen.query_one("#evals-bench-form-error").display
        assert screen._selection.kind == "bench"
        assert screen._selection.id == bench_id

        bench_row = screen._view_model.bench_by_id(bench_id)
        assert bench_row["name"] == "authored bench"
        config_data = bench_row.get("config_data") or {}
        assert set(config_data.get("target_ids") or ()) == {first_target_id, second_target_id}
        assert tuple(config_data.get("probes") or ()) == (" Sure", " I")

        # -- Run.
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._selection.kind == "run_group")
        await pilot.pause()
        run_group_id = screen._selection.id

        run_groups = screen._view_model.run_groups()
        run_row = next(row for row in run_groups if row["id"] == run_group_id)
        assert run_row["task_id"] == bench_id
        rail_label = _run_group_row_label(run_row)
        assert rail_label.startswith("✓ "), (
            f"an all-succeed run must render the plain check glyph in the "
            f"rail, got {rail_label!r}"
        )

        # -- The grid renders with both targets as columns.
        grid = screen.query_one("#evals-results-grid", ResultsGrid)
        table = grid.query_one("#evals-grid-table", DataTable)
        assert table.row_count == 4

        s1_id = next(s["id"] for s in snippets if s["text"] == "The protestors were")

        # -- Probe lens: real, per-target-distinct readings, never "n/a".
        grid.query_one("#evals-lens-selector", Select).value = "probe"
        await pilot.pause()
        first_probe_text = str(table.get_cell(s1_id, first_target_id))
        second_probe_text = str(table.get_cell(s1_id, second_target_id))
        assert first_probe_text == f"{math.log(0.5):.2f}  50.0%", first_probe_text
        assert first_probe_text not in ("", "n/a")
        assert second_probe_text not in ("", "n/a")
        # The two targets' fake distributions never overlap on " Sure" --
        # this is the whole point of using two DIFFERENT fakes rather than
        # one shared one (see the module docstring): the lens must tell
        # the columns apart, not print the same reading twice.
        assert second_probe_text != first_probe_text
        assert "never observed" in second_probe_text

        # -- Δ baseline lens: a real, nonzero Spread column -- only
        # possible because the two targets' distributions genuinely
        # differ (see the module docstring's own rationale for the fake).
        grid.query_one("#evals-lens-selector", Select).value = "delta"
        await pilot.pause()
        labels = [str(col.label) for col in table.columns.values()]
        assert any("Spread" in label for label in labels)
        spread_text = str(table.get_cell(s1_id, "__spread__"))
        assert spread_text not in ("", "0.00"), (
            f"two genuinely different target distributions must produce a "
            f"real, nonzero Spread value, got {spread_text!r}"
        )
        assert float(spread_text) > 0.0
