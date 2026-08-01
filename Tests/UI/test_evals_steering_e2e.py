"""End-to-end: two UI-authored, differently-steered targets light up the
Δ baseline lens's COLUMN mode with a real, nonzero Spread.

task-1611's final task. Tasks 1-2 built the two halves this test proves
close a real loop: T1 made ``eval_models.config`` carry a target's steering
(``storage.model_steering`` -> ``Target.prefix``/``system_prompt``, read at
run time by ``sample_bench._resolve_targets`` and persisted into the run
snapshot); T2 made ``bench_editor.py``'s "+ New target" mini-form render
ALWAYS (not only in the zero-``llama_cpp``-models state) so a bench author
can mint an ADDITIONAL, differently-steered target through the UI, breaking
the "the UI can create exactly one eval_models row ever" wall that made the
Δ lens's COLUMN mode ("compare targets against a baseline target")
permanently degenerate for a real user before this task.

``Tests/UI/test_evals_authoring_e2e.py`` (task-1482's own closing E2E) is
the established convention this test follows -- same ``EvalsHarness``/
``_FakeAppInstance`` harness, same rail-Import-then-"+ New bench" opening
moves, same ``_wait_until`` worker-completion poll -- but that test's own
docstring says outright it "seeds target 2 at the DB layer" (a direct
``evals_db.create_model`` call) precisely because, at task-1482 time, there
was no UI path to a second row at all. This test is a SEPARATE file, not an
extension of that one, for two reasons: (1) its whole point is the UI path
that test explicitly could not use, so folding it in would make the file's
own docstring describe two contradictory target-creation stories; (2) its
fake capture client is keyed on a different, more fundamental signal --
``target.prefix``/``target.system_prompt`` (whether steering is PRESENT at
all), not the target's name -- which only makes sense read on its own, not
diffed against a sibling fake one scroll up that keys on name for an
unrelated reason (telling two DB-seeded rows apart, not proving steering
itself changed anything).
"""

from __future__ import annotations

import math

import pytest
from textual.widgets import DataTable, Input, Select

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import CellCapture, PreflightResult, Target, TokenProb
from tldw_chatbook.UI.Evals.library_rail import LibraryRail
from tldw_chatbook.UI.Evals.results_grid import FAILED_MARK, ResultsGrid
from tldw_chatbook.UI.Evals.snippet_editor import dataset_snippets
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen

from .test_evals_screen import EvalsHarness, _FakeAppInstance

_REALISTIC_SIZE = (160, 45)

#: The prefix typed into target 2's mini-form -- raw mode (the draft
#: bench's own default prompt_mode, see library_rail.py's own
#: ``_create_new_bench``), so this exercises ``#evals-target-prefix``, the
#: SAME field ``bench_editor.py``'s own leading-whitespace-preservation
#: test drives. Deliberately holds a leading space -- steering.py's own
#: whitespace-preservation contract (task-1611 T1) means this is not
#: silently stripped anywhere between the Input and the run.
_STEERED_PREFIX = " Answer with an article: "


class _SteeringAwareFakeCaptureClient:
    """The crux of this test: a fake capture client keyed on whether the
    REQUEST it receives carries steering at all (``target.prefix`` or
    ``target.system_prompt``, truthy), not on the target's name or id.

    Every other fake capture client in this suite (see
    ``test_evals_empty_states.py``'s ``_FakeCaptureClient``/
    ``_PausableFakeCaptureClient``, and ``test_evals_authoring_e2e.py``'s
    ``_TwoTargetFakeCaptureClient``) returns one distribution per target
    NAME or a single shared one -- fine for proving the grid tells two
    columns apart at all, but useless for proving steering ITSELF is what
    moved the distribution. If this fake keyed off name/id instead, a
    passing spread/difference assertion below would be consistent with a
    bug that dropped ``target.prefix`` on the floor somewhere between the
    UI and the runner -- the two targets would still read differently
    (different names), for a reason that has nothing to do with task-1611.
    Keying on ``target.prefix``/``target.system_prompt`` instead makes a
    real Δ only possible when steering genuinely reached the ``Target``
    this client was constructed with.
    """

    def __init__(self, target: Target) -> None:
        self._target = target

    async def preflight(self, target: Target, mode: str, top_k: int) -> PreflightResult:
        return PreflightResult(state="ok", k_returned=3, canary="pass")

    async def capture(
        self, snippet: str, target: Target, mode: str, top_k: int
    ) -> CellCapture:
        if target.prefix or target.system_prompt:
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
            captured_at="2026-07-31T00:00:00Z",
        )


@pytest.fixture
def evals_db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def steering_app(evals_db: EvalsDB) -> EvalsHarness:
    """A configured llama.cpp endpoint -- needed for the bench editor's
    "+ New target" mini-form: ``evals_screen.py``'s own handler gates on
    ``sample_bench.configured_llama_cpp_url`` before it will write a row.
    Mirrors ``test_evals_authoring_e2e.py``'s own ``authoring_app``."""
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
async def test_two_ui_authored_targets_one_steered_light_up_column_mode_delta(
    steering_app, evals_db, tmp_path
):
    """Import a dataset -> "+ New bench" -> create target 1 (unsteered,
    blank steering field) -> create target 2 (raw-mode prefix) -> Save ->
    Run -> switch the grid's baseline selector to COLUMN mode with target 1
    as baseline -> the Spread column carries a real, nonzero value, and the
    two targets' own cells actually differ. Also: the run snapshot
    persisted target 2's prefix -- T1's snapshot seam, reached this time
    through a fully UI-authored path, not a direct DB write.

    No DB seeding of ANY target here -- unlike
    ``test_evals_authoring_e2e.py``'s own second target, both rows in this
    test are minted by pressing ``#evals-bench-create-target`` with the
    mini-form's Name/steering fields typed -- that is the whole point.
    """
    import_path = tmp_path / "imported.txt"
    import_path.write_text(
        "The weather today is\n"
        "The election results were\n",
        encoding="utf-8",
    )

    async with steering_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        screen: EvalsScreen = pilot.app.screen
        # Shared by the sample-bench AND bench-run workers (see
        # evals_screen.py's own field docstring) -- this loop only ever
        # exercises the bench-run path below, but the seam is the same
        # one either worker reads.
        screen._sample_bench_client_factory = lambda t: _SteeringAwareFakeCaptureClient(t)

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

        # -- "+ New bench" against the just-imported dataset -- a draft
        # bench with zero targets, prompt_mode "raw" (library_rail.py's
        # own `_create_new_bench` default).
        await pilot.click("#evals-rail-new-bench")
        await pilot.pause()
        assert screen._selection.kind == "bench"
        bench_id = screen._selection.id
        draft_bench = screen._view_model.bench_by_id(bench_id)
        assert draft_bench.get("dataset_id") == dataset_id

        # -- Target 1: the "+ New target" mini-form, Name and steering
        # BOTH left blank -- an auto-named, unsteered target. task-1611 T2
        # made this button render (and mint a fresh row) unconditionally,
        # not only in a zero-models state.
        assert evals_db.list_models(provider="llama_cpp") == []
        assert screen.query_one("#evals-bench-create-target")
        assert screen.query_one("#evals-target-prefix", Input).value == ""
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()
        created_after_first = evals_db.list_models(provider="llama_cpp")
        assert len(created_after_first) == 1, "first Create must mint exactly one row"
        first_target_id = created_after_first[0]["id"]
        assert created_after_first[0]["config"] in ({}, None), (
            "target 1's steering field was left blank -- its row must carry "
            "no prefix/system_prompt"
        )
        assert screen.query_one("#evals-bench-target-0")

        # -- Target 2: the SAME "+ New target" mini-form, pressed a SECOND
        # time -- task-1611 T2's whole point is that this mints ANOTHER
        # row rather than reusing target 1's. Name and prefix both typed.
        screen.query_one("#evals-target-name", Input).value = "steered target"
        screen.query_one("#evals-target-prefix", Input).value = _STEERED_PREFIX
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()
        created_after_second = evals_db.list_models(provider="llama_cpp")
        assert len(created_after_second) == 2, "second Create must mint an ADDITIONAL row"
        second_target_id = next(
            row["id"] for row in created_after_second if row["id"] != first_target_id
        )
        second_target_row = evals_db.get_model(second_target_id)
        assert second_target_row["name"] == "steered target"
        assert second_target_row["config"]["prefix"] == _STEERED_PREFIX, (
            "target 2's typed prefix must reach its eval_models.config row byte-exact"
        )
        assert screen.query_one("#evals-bench-target-1")

        # -- Save. Both UI-authored targets persist on the bench.
        await pilot.click("#evals-bench-save")
        await pilot.pause()
        assert not screen.query_one("#evals-bench-form-error").display
        bench_row = screen._view_model.bench_by_id(bench_id)
        config_data = bench_row.get("config_data") or {}
        assert set(config_data.get("target_ids") or ()) == {first_target_id, second_target_id}

        # -- Run.
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._selection.kind == "run_group")
        await pilot.pause()
        run_group_id = screen._selection.id

        # -- The run snapshot persisted target 2's prefix -- T1's snapshot
        # seam (storage._snapshot -> create_run_group), reached this time
        # through a fully UI-authored path (mirrors
        # test_run_existing_bench.py::
        # test_run_existing_bench_persists_a_steered_targets_prefix_in_the_snapshot,
        # which drove the identical assertion against a directly-DB-seeded
        # target).
        run_row = evals_db.list_runs(run_group_id=run_group_id, limit=10)[0]
        snapshot_targets = run_row["config_overrides"]["snapshot"]["targets"]
        second_snapshot_target = next(
            t for t in snapshot_targets if t["id"] == second_target_id
        )
        assert second_snapshot_target["prefix"] == _STEERED_PREFIX
        first_snapshot_target = next(
            t for t in snapshot_targets if t["id"] == first_target_id
        )
        assert first_snapshot_target["prefix"] is None
        assert first_snapshot_target["system_prompt"] is None

        # -- The grid renders with both targets as columns.
        grid = screen.query_one("#evals-results-grid", ResultsGrid)
        table = grid.query_one("#evals-grid-table", DataTable)
        assert table.row_count == 2
        s1_id = next(s["id"] for s in snippets if s["text"] == "The weather today is")

        # -- Top-1 lens (the grid's default): the two targets' cells for
        # the same snippet actually differ -- steering genuinely changed
        # what the fake client returned, not merely which column it's in.
        first_cell = str(table.get_cell(s1_id, first_target_id))
        second_cell = str(table.get_cell(s1_id, second_target_id))
        assert first_cell not in ("", "n/a")
        assert second_cell not in ("", "n/a")
        assert first_cell != second_cell, (
            f"an unsteered and a steered target must read differently, got "
            f"{first_cell!r} == {second_cell!r}"
        )

        # -- Switch to COLUMN mode with target 1 (unsteered) as baseline --
        # explicitly, via the same `#evals-baseline-selector` a real user
        # drives, rather than relying on the grid's own column/index-0
        # default. Before task-1611, a real user could never reach this
        # state authoring entirely through the UI: there was no second
        # target to pick as a non-baseline column at all.
        grid.query_one("#evals-lens-selector", Select).value = "delta"
        await pilot.pause()
        grid.query_one("#evals-baseline-selector", Select).value = ("column", first_target_id)
        await pilot.pause()

        labels = [str(col.label) for col in table.columns.values()]
        assert any("Spread" in label for label in labels)
        baseline_col_label = str(table.columns[first_target_id].label)
        assert "baseline" in baseline_col_label

        # -- Δ baseline lens: a real, nonzero Spread -- only possible
        # because target 2's prefix genuinely moved its distribution away
        # from target 1's (baseline). This is the assertion the authoring
        # program's own E2E could only make with a DB-seeded second
        # target; here both targets are fully UI-authored.
        spread_text = str(table.get_cell(s1_id, "__spread__"))
        assert spread_text not in ("", "0.00"), (
            f"a steered and an unsteered UI-authored target must produce a "
            f"real, nonzero Spread value, got {spread_text!r}"
        )
        assert float(spread_text) > 0.0

        # -- The non-baseline (steered) target's own Δ cell is a real
        # divergence reading too, not the "baseline" literal.
        second_delta_cell = str(table.get_cell(s1_id, second_target_id))
        assert second_delta_cell != "baseline"
        assert second_delta_cell not in ("", FAILED_MARK)
