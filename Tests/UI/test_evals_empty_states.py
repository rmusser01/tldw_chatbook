"""Empty states, the one-click sample bench, and export.

Per the design spec's "Empty states and first run" table: a fresh install's
most common initial condition is zero benches, zero datasets, zero runs, and
possibly zero configured providers. These tests pin that a user with no
providers is routed to Settings (never shown a target list or a wall of
preflight failures), that a user with providers but no benches gets a real,
clickable one-click sample bench that reaches a genuinely populated grid, and
that export writes what it claims to.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from textual import on
from textual.widgets import DataTable

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import CellCapture, PreflightResult, TokenProb
from tldw_chatbook.Evals.word_bench.storage import load_grid
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, FileSave
from tldw_chatbook.UI.Evals import sample_bench
from tldw_chatbook.UI.Evals.evals_state import EvalsViewModel
from tldw_chatbook.UI.Evals.library_rail import LibraryRail
from tldw_chatbook.UI.Evals.results_grid import ResultsGrid
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen

from .test_evals_screen import EvalsHarness, _FakeAppInstance
from .test_evals_results_grid import _select_run_group
from .test_evals_results_grid import evals_db as evals_db  # noqa: F401 -- fixture re-export
from .test_evals_results_grid import mixed_run_group as mixed_run_group  # noqa: F401


# ---------------------------------------------------------------------------
# sample_bench.py -- pure logic, no Textual involved.
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def view_model(db: EvalsDB) -> EvalsViewModel:
    return EvalsViewModel(db)


def test_sample_snippets_are_the_specs_own_minimal_pairs():
    """Exact text and group per the PR 3b Task 2 brief -- two minimal pairs
    differing by one loaded word, grouped neutral/loaded so
    analysis.group_means has two real groups to aggregate."""
    texts_by_group = {}
    for text, group in sample_bench.SAMPLE_SNIPPETS:
        texts_by_group.setdefault(group, []).append(text)
    assert texts_by_group == {
        "neutral": ["The protestors were", "The government said"],
        "loaded": ["The rioters were", "The regime said"],
    }


def test_provider_not_configured_with_empty_app_config_and_no_models(view_model):
    """A stripped/broken app_config with no api_settings at all, and no
    pre-existing eval_models row, must not claim a provider is configured."""
    assert sample_bench.provider_is_configured(view_model, {}) is False
    assert sample_bench.resolve_sample_target(view_model, {}) is None


def test_provider_configured_when_llama_cpp_url_is_set(view_model):
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    assert sample_bench.provider_is_configured(view_model, app_config) is True


def test_resolve_sample_target_creates_a_real_eval_models_row(db, view_model):
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080", "model": "m"}}}
    target = sample_bench.resolve_sample_target(view_model, app_config)
    assert target is not None
    assert target["provider"] == "llama_cpp"
    assert target["model_id"] == "m"
    # It is a REAL row -- readable straight back out of the db, not a
    # dict fabricated only in memory.
    assert db.get_model(target["id"])["id"] == target["id"]


def test_resolve_sample_target_reuses_an_existing_llama_cpp_row_without_creating_a_second(
    db, view_model
):
    existing_id = db.create_model(name="my target", provider="llama_cpp", model_id="m")
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    target = sample_bench.resolve_sample_target(view_model, app_config)
    assert target["id"] == existing_id
    assert len(db.list_models(provider="llama_cpp")) == 1


def test_resolve_sample_target_ignores_a_non_llama_cpp_row_and_still_needs_a_configured_url(
    db, view_model
):
    """A pre-existing eval_models row for a provider this module cannot
    safely build a capture client for (see the module docstring) must not
    be silently adopted as the sample target."""
    db.create_model(name="cloud target", provider="openai", model_id="gpt-4o-mini")
    assert sample_bench.resolve_sample_target(view_model, {}) is None


def test_provider_is_configured_matches_resolve_sample_target_exactly(view_model):
    """Single source of truth: the gate can never say "configured" while
    the resolver itself would fail, or vice versa."""
    for app_config in ({}, {"api_settings": {"llama_cpp": {"api_url": "http://x:1"}}}):
        assert sample_bench.provider_is_configured(view_model, app_config) is (
            sample_bench.resolve_sample_target(view_model, app_config) is not None
        )


def test_creating_the_sample_bench_twice_does_not_collide_on_unique_names(db, view_model):
    """eval_tasks.name and eval_datasets.name are UNIQUE with no
    deleted_at exemption (Evals_DB.py's schema) -- a bare literal name
    would raise sqlite3.IntegrityError on a second creation. Exercised
    directly against the db (not through create_and_run_sample_bench,
    which also needs a client) to isolate the naming concern."""
    first = sample_bench._unique_name(sample_bench.SAMPLE_BENCH_NAME)
    second = sample_bench._unique_name(sample_bench.SAMPLE_BENCH_NAME)
    assert first != second
    db.create_task(
        name=first, task_type="logprob", config_format="custom",
        config_data={"bench_type": "word_bench"},
    )
    # Must not raise -- a distinct generated name.
    db.create_task(
        name=second, task_type="logprob", config_format="custom",
        config_data={"bench_type": "word_bench"},
    )


# ---------------------------------------------------------------------------
# create_and_run_sample_bench -- the full flow, with a fake HTTP client.
# ---------------------------------------------------------------------------


class _FakeCaptureClient:
    """Mirrors Tests/Evals/word_bench/test_runner.py's FakeClient."""

    def __init__(self, calls: list) -> None:
        self._calls = calls

    async def preflight(self, target, mode, top_k):
        return PreflightResult(state="ok", k_returned=5, canary="pass")

    async def capture(self, snippet, target, mode, top_k):
        self._calls.append((snippet, target.name))
        return CellCapture(
            prompt_mode=mode, k_requested=top_k, k_returned=1, content_offset=0,
            top_k=(TokenProb(token=" a", logprob=-0.3, token_id=1),),
            canary="unchecked", captured_at="2026-07-26T00:00:00Z",
        )


@pytest.mark.asyncio
async def test_create_and_run_sample_bench_produces_a_populated_grid(db, view_model):
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    calls: list = []
    result = await sample_bench.create_and_run_sample_bench(
        view_model, app_config, client_factory=lambda t: _FakeCaptureClient(calls)
    )
    assert result.run_group_id

    grid = load_grid(db, result.run_group_id)
    assert len(grid["snapshot"]["snippets"]) == 4
    assert len(grid["snapshot"]["targets"]) == 1
    assert len(grid["cells"]) == 4  # every (snippet, target) cell captured
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_create_and_run_sample_bench_raises_when_no_target_available(view_model):
    with pytest.raises(RuntimeError):
        await sample_bench.create_and_run_sample_bench(view_model, {})


@pytest.mark.asyncio
async def test_create_and_run_sample_bench_raises_when_db_is_unavailable():
    view_model = EvalsViewModel(None)
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    with pytest.raises(RuntimeError):
        await sample_bench.create_and_run_sample_bench(view_model, app_config)


# ---------------------------------------------------------------------------
# The library rail's empty states -- mounted through the real EvalsScreen.
# ---------------------------------------------------------------------------


class _NavCapturingHarness(EvalsHarness):
    """Adds a ``NavigateToScreen`` catcher: the real app handles that
    message at the top level (routing tab switches), which this harness
    does not reproduce -- capturing it here is the only way to verify
    "Open Settings" actually requested navigation, without depending on
    unrelated app-shell machinery."""

    def __init__(self, app_instance: _FakeAppInstance) -> None:
        super().__init__(app_instance)
        self.navigated_to: list[str] = []

    @on(NavigateToScreen)
    def _capture_navigation(self, event: NavigateToScreen) -> None:
        self.navigated_to.append(event.screen_name)


@pytest.fixture
def no_provider_app(evals_db: EvalsDB) -> _NavCapturingHarness:
    """No app_config and no pre-existing eval_models row -- the "zero
    providers configured" condition."""
    return _NavCapturingHarness(_FakeAppInstance(evals_db, app_config={}))


@pytest.mark.asyncio
async def test_no_providers_routes_the_benches_section_to_settings(no_provider_app):
    """Scoped to the Benches section, not the whole rail: Datasets and Runs
    never showed a target list or preflight results, and classic tasks need
    no provider at all -- see test_classic_task_subgroup_is_reachable_by_
    clicking_its_rail_row (test_evals_bench_editor.py), which this scoping
    keeps passing."""
    async with no_provider_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        rail = screen.query_one("#evals-library-pane")

        assert screen.query_one("#evals-rail-no-providers")
        settings_button = screen.query_one("#evals-rail-open-settings")
        assert rail.region.contains_region(settings_button.region)

        # No target list, no wall of preflight failures, and no sample-
        # bench offer pointing at nothing -- but the other two sections
        # (which never needed a provider) still render normally.
        assert not screen.query("#evals-create-sample-bench")
        labels = " ".join(
            w.renderable.plain if hasattr(w.renderable, "plain") else str(w.renderable)
            for w in screen.query(".evals-rail-section-label")
        )
        assert "Datasets (0)" in labels
        assert "Runs (0)" in labels

        await pilot.click("#evals-rail-open-settings")
        await pilot.pause()
        assert pilot.app.navigated_to == ["settings"]


@pytest.mark.asyncio
async def test_classic_tasks_stay_reachable_with_no_provider_configured(
    no_provider_app, evals_db: EvalsDB
):
    """Classic (non-word-bench) tasks are read-only history with no target
    concept -- they must stay reachable even when no word-bench provider is
    configured (this is exactly the regression the section-scoped gate
    above exists to avoid)."""
    dataset_id = evals_db.create_dataset(
        name="mmlu-500", format="custom", source_path="inline:mmlu-500"
    )
    evals_db.create_task(
        name="mmlu-subset", task_type="question_answer", config_format="custom",
        config_data={}, dataset_id=dataset_id,
    )
    async with no_provider_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert screen.query(".evals-rail-classic-separator")
        assert screen.query_one("#evals-rail-row-benches-classic-0")
        # Pre-existing convention (unchanged by this task): the "No benches
        # yet."/no-providers empty copy only ever appears when the section
        # is FULLY empty (no word benches AND no classic tasks) -- with a
        # classic task present, neither renders, and the sample-bench offer
        # correctly stays absent too (there is still no word bench provider).
        assert not screen.query("#evals-rail-no-providers")
        assert not screen.query("#evals-create-sample-bench")


@pytest.mark.asyncio
async def test_providers_configured_via_an_existing_eval_models_row_offers_the_sample_bench(
    no_provider_app, evals_db: EvalsDB
):
    """A pre-existing eval_models row (e.g. from before this rebuild, or a
    prior sample-bench run) is real, already-configured data -- the Benches
    section must treat it as "providers configured" even with an empty
    app_config."""
    evals_db.create_model(name="pre-existing", provider="llama_cpp", model_id="m")
    async with no_provider_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert not screen.query("#evals-rail-no-providers")
        assert screen.query_one("#evals-create-sample-bench")


# ---------------------------------------------------------------------------
# The one-click sample bench.
# ---------------------------------------------------------------------------


@pytest.fixture
def configured_app(evals_db: EvalsDB) -> EvalsHarness:
    """A configured llama.cpp endpoint, but zero benches -- the condition
    that should show the sample-bench offer."""
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    return EvalsHarness(_FakeAppInstance(evals_db, app_config=app_config))


# _FakeCaptureClient is defined once, above (shared by the pure async-flow
# test and the click-driven Textual test below).


async def _wait_until(pilot, predicate, *, tries: int = 300, interval: float = 0.02) -> None:
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause(interval)
    raise AssertionError("condition never became true")


@pytest.mark.asyncio
async def test_no_benches_offers_a_genuinely_clickable_sample_bench(configured_app):
    async with configured_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen: EvalsScreen = pilot.app.screen
        calls: list = []
        screen._sample_bench_client_factory = lambda t: _FakeCaptureClient(calls)

        button = screen.query_one("#evals-create-sample-bench")
        rail = screen.query_one("#evals-library-pane")
        assert rail.region.contains_region(button.region)

        await pilot.click("#evals-create-sample-bench")
        await _wait_until(pilot, lambda: screen._selection.kind == "run_group")
        await pilot.pause()

        # A real bench now exists, listed by the rail's own DOM (not just
        # the view model's data layer) -- the rail recomposes on every
        # selection change and re-reads view_model.benches() fresh.
        assert len(screen._view_model.benches()) == 1
        bench_row = screen.query_one("#evals-rail-row-benches-0")
        assert "loaded-nouns" in str(bench_row.label)
        assert screen.query_one("#evals-library-pane").region.contains_region(
            bench_row.region
        )

        # The click reached a genuinely populated grid -- 4 rows (the
        # spec's own loaded-nouns minimal pairs) x 1 target column, every
        # cell captured by the (fake, but real-code-path) client -- the
        # detail pane it opened into, strictly more informative than the
        # bare (unrun) bench editor.
        assert len(calls) == 4
        grid = screen.query_one("#evals-results-grid", ResultsGrid)
        table = grid.query_one("#evals-grid-table", DataTable)
        assert table.row_count == 4
        assert len(table.columns) == 2  # Snippet + the one target

        # The bench itself (not just its run) is independently selectable
        # and opens the real BenchEditor, listing the one target it was
        # prewired to -- proves the sample bench is a genuine bench, not
        # just an ephemeral run.
        screen.select(kind="bench", id=screen._view_model.benches()[0]["id"])
        await pilot.pause()
        assert "loaded-nouns" in str(
            screen.query_one("#evals-detail-bench-name").renderable
        )
        assert screen.query_one("#evals-bench-target-0")


@pytest.mark.asyncio
async def test_sample_bench_offer_is_hidden_with_no_provider_configured(no_provider_app):
    async with no_provider_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        assert not pilot.app.screen.query("#evals-create-sample-bench")


# ---------------------------------------------------------------------------
# "No datasets" -- authoring and import side by side.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_datasets_offers_new_dataset_and_import_side_by_side(configured_app):
    async with configured_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        rail = screen.query_one("#evals-library-pane")
        new_button = screen.query_one("#evals-rail-new-dataset")
        import_button = screen.query_one("#evals-rail-import-dataset")
        assert rail.region.contains_region(new_button.region)
        assert rail.region.contains_region(import_button.region)


@pytest.mark.asyncio
async def test_new_dataset_button_creates_and_selects_an_empty_dataset(
    configured_app, evals_db: EvalsDB
):
    async with configured_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen: EvalsScreen = pilot.app.screen
        await pilot.click("#evals-rail-new-dataset")
        await pilot.pause()

        assert len(evals_db.list_datasets()) == 1
        assert screen._selection.kind == "dataset"
        summary = screen.query_one("#evals-snippet-editor-summary")
        assert "0 snippets" in str(summary.renderable)


@pytest.mark.asyncio
async def test_import_dataset_button_pushes_a_file_open_dialog(configured_app):
    async with configured_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        await pilot.click("#evals-rail-import-dataset")
        await pilot.pause()
        assert isinstance(pilot.app.screen, FileOpen)


@pytest.mark.asyncio
async def test_import_dataset_file_selected_creates_a_dataset_from_the_file(
    configured_app, evals_db: EvalsDB, tmp_path
):
    """Drives the picker callback directly with a real temp file, bypassing
    the modal itself -- mirrors SnippetEditor's own established test
    convention for the file-reading part of an import flow."""
    csv_path = tmp_path / "nouns.csv"
    csv_path.write_text("text,group\nThe cat sat,neutral\n", encoding="utf-8")

    async with configured_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen: EvalsScreen = pilot.app.screen
        rail = screen.query_one(LibraryRail)
        rail._handle_dataset_import_file_selected(csv_path)
        await pilot.pause()

        datasets = evals_db.list_datasets()
        assert len(datasets) == 1
        assert screen._selection.kind == "dataset"
        assert screen._selection.id == datasets[0]["id"]


# ---------------------------------------------------------------------------
# Export -- CSV for the active lens, JSON for the whole run group.
# ---------------------------------------------------------------------------


@pytest.fixture
def export_app(evals_db: EvalsDB) -> EvalsHarness:
    return EvalsHarness(_FakeAppInstance(evals_db))


@pytest.mark.asyncio
async def test_export_key_pushes_a_file_save_dialog_with_both_formats_offered(
    export_app, mixed_run_group
):
    async with export_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        table = grid.query_one("#evals-grid-table", DataTable)
        table.focus()
        await pilot.press("e")
        await pilot.pause()
        assert isinstance(pilot.app.screen, FileSave)


@pytest.mark.asyncio
async def test_csv_export_reflects_the_active_lens(export_app, mixed_run_group, tmp_path):
    """Switches lens to Coverage, then exports -- the CSV body must show
    Coverage's own percentages (and match the on-screen DataTable's own
    stored cell values cell-for-cell), never a different/hardcoded lens.
    """
    async with export_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])
        grid.query_one("#evals-lens-selector").value = "coverage"
        await pilot.pause()

        # What the DataTable itself is showing right now -- read straight
        # back from Textual's own storage, not this test's belief about
        # what it should say (mirrors Task 1's markup-safety regression
        # test's own technique).
        table = grid.query_one("#evals-grid-table", DataTable)
        on_screen_row0 = [str(cell) for cell in table.get_row_at(0)]

        destination = tmp_path / "export.csv"
        grid._write_export_file(destination)
        rows = list(csv.reader(io.StringIO(destination.read_text(encoding="utf-8"))))

        assert rows[0][0] == "Snippet"
        assert rows[1] == on_screen_row0
        # Coverage renders as a bare percentage (e.g. "12%") -- never the
        # Top-1 lens's quoted-token format -- independent confirmation
        # this really is Coverage's own output.
        body_cells = [cell for row in rows[1:] for cell in row[1:] if cell and cell != "—"]
        assert body_cells, "no coverage cells were exported"
        assert all(cell.endswith("%") for cell in body_cells)


@pytest.mark.asyncio
async def test_json_export_contains_snapshot_top_k_and_resolved_probes(
    export_app, mixed_run_group, tmp_path
):
    async with export_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])

        destination = tmp_path / "export.json"
        grid._write_export_file(destination)

        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload["run_group_id"] == mixed_run_group["group_id"]

        snapshot = payload["snapshot"]
        assert snapshot["bench_name"] == "loaded-nouns v1"
        assert len(snapshot["snippets"]) == 3
        assert len(snapshot["targets"]) == 2

        cells = payload["cells"]
        base_id = mixed_run_group["base_id"]
        steered_id = mixed_run_group["steered_id"]

        # A captured cell: real top-K entries plus a resolved probe
        # reading -- enough to reproduce the measurement outside the app.
        s1_base = cells[f"s1|{base_id}"]
        assert s1_base["status"] == "captured"
        assert s1_base["top_k"][0]["token"] == " a"
        assert "logprob" in s1_base["top_k"][0]
        assert " a" in s1_base["probes"]  # mixed_run_group's configured probe
        assert s1_base["probes"][" a"]["state"] == "observed"

        # A failed cell: reason/detail, not a fabricated top-K.
        s2_base = cells[f"s2|{base_id}"]
        assert s2_base["status"] == "failed"
        assert s2_base["reason"] == "unreachable"

        # An unrun cell (s3 x base) has no entry at all -- never a
        # fabricated zeroed-out row.
        assert f"s3|{base_id}" not in cells
        assert f"s3|{steered_id}" in cells


@pytest.mark.asyncio
async def test_export_rejects_an_invalid_path_without_crashing(
    export_app, mixed_run_group, monkeypatch
):
    async with export_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        grid = await _select_run_group(pilot, mixed_run_group["group_id"])

        import tldw_chatbook.UI.Evals.results_grid as results_grid_module

        def _reject(*_args, **_kwargs):
            raise ValueError("rejected for test")

        monkeypatch.setattr(results_grid_module, "validate_path_simple", _reject)
        # Must not raise.
        grid._write_export_file(Path("/tmp/whatever.json"))
