"""Results-grid commands follow real Evals keyboard scope (TASK-35869)."""

import pytest
from textual.binding import Binding
from textual.widgets import Input, Select

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_destination_shells import DestinationHarness, SCREEN_BY_ROUTE
from Tests.UI.test_evals_results_grid import (  # noqa: F401 -- real SQLite fixtures
    mixed_run_group as _mixed_run_group,
)
from Tests.UI.test_evals_screen import _FakeAppInstance
from tldw_chatbook.app_lifecycle import LifecycleMixin
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import Target
from tldw_chatbook.Evals.word_bench.storage import create_run_group, load_bench
from tldw_chatbook import config as config_module
from tldw_chatbook.UI.Evals.library_rail import LibraryRail
from tldw_chatbook.Third_Party.textual_fspicker import FileSave
from tldw_chatbook.UI.Evals.results_grid import ResultsGrid
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


mixed_run_group = _mixed_run_group


class GridDestinationHarness(DestinationHarness):
    CSS_PATH = [str(path) for path in APP_STYLESHEETS]
    BINDINGS = [
        Binding("f1", "show_workbench_help"),
        Binding("f6", "focus_next_workbench_pane"),
    ]
    action_show_workbench_help = LifecycleMixin.action_show_workbench_help
    action_focus_next_workbench_pane = LifecycleMixin.action_focus_next_workbench_pane


@pytest.fixture
def evals_db():
    db = EvalsDB(db_path=":memory:", client_id="test")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def empty_run_group(evals_db, mixed_run_group):
    row = evals_db.list_runs(run_group_id=mixed_run_group["group_id"], limit=1)[0]
    config = load_bench(evals_db, row["task_id"])
    group_id, _ = create_run_group(
        evals_db,
        row["task_id"],
        config,
        [
            Target(
                id=mixed_run_group["base_id"],
                name="base",
                provider="llama_cpp",
                model_id="m",
            )
        ],
        [],
    )
    return {"group_id": group_id}


def _app(evals_db, monkeypatch):
    monkeypatch.setitem(SCREEN_BY_ROUTE, "evals", EvalsScreen)
    return GridDestinationHarness(_FakeAppInstance(evals_db), "evals")


async def _settle_selection(screen, pilot):
    # Drain the exact region owner after the rail's bubbling message, never all workers.
    await pilot.pause()
    for worker in list(screen.workers):
        if worker.node is screen and worker.group in {
            "evals-selection-regions",
            "evals-store-skills-probe",
        }:
            await worker.wait()
    await pilot.pause()


async def _select_run(app, pilot, group_id):
    screen = app.screen
    row_id = next(
        row_id
        for row_id, target in screen.query_one(LibraryRail)._row_targets.items()
        if target.kind == "run_group" and target.id == group_id
    )
    screen.query_one(f"#{row_id}").focus()
    await pilot.press("enter")
    await _settle_selection(screen, pilot)
    assert screen._selection.kind == "run_group"
    assert screen._selection.id == group_id
    return screen


def _grid_hints(screen):
    footer = screen.query_one(AppFooterStatus)
    hints = {
        action.key
        for action in footer._context_actions
        if action.key in {"l", "b", "s", "e"}
    }
    for key, label in (
        ("l", "lens"),
        ("b", "baseline"),
        ("s", "sort"),
        ("e", "export"),
    ):
        if key not in hints:
            assert f"{key} {label}" not in footer.shortcut_text
    return hints


def _in_pane(app, pane_id):
    return any(node.id == pane_id for node in app.focused.ancestors_with_self)


@pytest.mark.parametrize("size", [(80, 35), (220, 50)], ids=["compact", "wide"])
@private_profile_test
async def test_grid_footer_and_f1_follow_keyboard_panes(
    request, evals_db, mixed_run_group, monkeypatch, size
):
    """A rail/inspector focus must remove unreachable commands; F1 teaches their scope."""
    app = _app(evals_db, monkeypatch)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = await _select_run(app, pilot, mixed_run_group["group_id"])
        grid = screen.query_one(ResultsGrid)
        assert app.focused.id == "evals-grid-table"
        assert _grid_hints(screen) == {"l", "b", "s", "e"}
        if size[0] == 220:
            assert "l lens" in screen.query_one(AppFooterStatus).shortcut_text
        screen.query_one("#evals-rail-row-runs-0").focus()
        await pilot.pause()
        assert _grid_hints(screen) == set()
        grid.query_one("#evals-grid-table").focus()
        await pilot.pause()
        await pilot.press("f1")
        assert isinstance(app.screen, WorkbenchHelpPanel)
        text = str(app.screen.query_one("#workbench-help-body").renderable)
        for key, label in (
            ("l", "lens"),
            ("b", "baseline"),
            ("s", "sort"),
            ("e", "export"),
        ):
            assert f"{key}: {label}" in text
        assert "results grid" in text.lower()
        await pilot.press("f1")
        # The real global delegate must move out of the detail into Inspector.
        await pilot.press("f6")
        assert _in_pane(app, "lab-inspector")
        assert _grid_hints(screen) == set()
        await pilot.press("f6")
        assert _in_pane(app, "lab-rail")
        assert _grid_hints(screen) == set()
        lens = grid._lens
        await pilot.press("l", "b", "s", "e")
        assert app.screen is screen
        assert grid._lens == lens
        assert grid._baseline_mode == "column"
        assert grid._sort_mode == "none"
        await pilot.press("f1")
        text = str(app.screen.query_one("#workbench-help-body").renderable)
        assert "focus the results grid" in text.lower()
        assert "l: lens" in text and "e: export" in text
        await pilot.press("f1", "f6")
        assert _in_pane(app, "lab-body")
        assert _grid_hints(screen) == {"l", "b", "s", "e"}
        # Tab between grid controls retains reachability; open Select consumes letters.
        grid.query_one("#evals-lens-selector", Select).focus()
        await pilot.press("tab")
        assert _in_pane(app, "lab-body")
        assert _grid_hints(screen) == {"l", "b", "s", "e"}
        await pilot.press("enter")
        assert _grid_hints(screen) == set()
        await pilot.press("l")
        assert grid._lens == "top1"
        await pilot.press("escape")
        assert _grid_hints(screen) == {"l", "b", "s", "e"}
        # Collapsed panes are skipped by the same existing focus helper.
        screen.query_one("#lab-inspector-collapse").press()
        await pilot.pause()
        grid.query_one("#evals-grid-table").focus()
        await pilot.press("f6")
        assert _in_pane(app, "lab-rail")
        assert _grid_hints(screen) == set()


@private_profile_test
async def test_grid_commands_export_and_editor_typing_without_focus_io(
    request, evals_db, mixed_run_group, monkeypatch, tmp_path
):
    """Real grid actions and the picker export callback survive scope projection."""
    app = _app(evals_db, monkeypatch)
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        screen = await _select_run(app, pilot, mixed_run_group["group_id"])
        grid = screen.query_one(ResultsGrid)

        # Any new DB/config probe on focus/help/action dispatch is a regression.
        def unexpected_read(*args, **kwargs):
            raise AssertionError("focus or grid shortcut performed fresh I/O")

        with monkeypatch.context() as guard:
            guard.setattr(evals_db, "_get_connection", unexpected_read)
            guard.setattr(config_module, "get_cli_setting", unexpected_read)
            guard.setattr(screen, "_current_app_config", unexpected_read)
            await pilot.press("l")
            assert grid._lens == "entropy"
            await pilot.press("b")
            assert grid._baseline_mode == "row"
            await pilot.press("s")
            assert grid._sort_mode == "desc"
            await pilot.press("f6", "f6", "f1", "f1", "f6")
            assert app.focused.id == "evals-grid-table"
        await pilot.press("e")
        assert isinstance(app.screen, FileSave)
        destination = tmp_path / "grid.csv"
        app.screen.dismiss(destination)
        await pilot.pause()
        exported = destination.read_text(encoding="utf-8")
        assert "The protestors were" in exported
        assert "base" in exported and "steered" in exported
        # Select the actual bench editor from the surviving Catalog row.
        screen.query_one("#evals-rail-row-benches-0").focus()
        await pilot.press("enter")
        await _settle_selection(screen, pilot)
        field = screen.query_one("#evals-bench-name", Input)
        assert _in_pane(app, "lab-rail")
        await pilot.press("f6")
        assert app.focused is field
        await pilot.press("tab", "shift+tab")
        assert app.focused is field
        await pilot.press("home", "shift+end", "backspace", "l", "b", "s", "e")
        assert field.value == "lbse"
        assert _grid_hints(screen) == set()
        assert not screen.query(ResultsGrid)


@private_profile_test
async def test_unready_run_never_advertises_grid_actions(
    request, evals_db, mixed_run_group, empty_run_group, monkeypatch
):
    """A selected run with an empty snapshot has no usable grid commands."""
    app = _app(evals_db, monkeypatch)
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        screen = await _select_run(app, pilot, mixed_run_group["group_id"])
        assert _grid_hints(screen) == {"l", "b", "s", "e"}
        screen = await _select_run(app, pilot, empty_run_group["group_id"])
        assert "no snippets" in str(screen.query_one("#evals-grid-empty").renderable)
        assert _grid_hints(screen) == set()
        await pilot.press("f1")
        text = str(app.screen.query_one("#workbench-help-body").renderable)
        assert "l: lens" not in text and "e: export" not in text
        await pilot.press("f1")
        screen.select(kind="none")
        await _settle_selection(screen, pilot)
        assert _grid_hints(screen) == set()
