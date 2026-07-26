"""Evals screen shell. The old hub rendered an empty body because it mounted
Screen objects inside a Container; these tests pin that the replacement
actually puts widgets on screen."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App

import tldw_chatbook
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import BenchConfig
from tldw_chatbook.Evals.word_bench.storage import save_bench
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen

#: The real bundled stylesheet (mirrors HomeHarness in test_home_screen.py
#: and CanvasAppWithBundledCSS in test_mcp_servers_mode.py). This is not what
#: makes the Screen-in-Container defect this PR fixes detectable -- an
#: isolated repro during implementation showed that bug reproduces under
#: Textual's own bare default CSS too (a nested Screen's descendants get
#: `region=Region(0, 0, 0, 0)` regardless of any stylesheet; DOM presence
#: checks like `query_one`/`children` pass either way). It is loaded so the
#: real design-system widgets this screen composes (DestinationHeader,
#: LabModeStrip, the .ds-panel/.ds-inspector pane borders, the workbench's
#: 1fr pane split) resolve exactly as a user would see them, rather than
#: Textual's bare fallback layout.
_BUNDLED_CSS_PATH = str(
    Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
)


class _FakeOrchestrator:
    def __init__(self, db: EvalsDB) -> None:
        self.db = db


class _FakeAppInstance:
    """The minimal app_instance surface EvalsScreen (and its BaseAppScreen
    chrome -- MainNavigationBar, AppFooterStatus) actually reads.

    ``evaluation_orchestrator.db`` mirrors the real attribute app.py's
    ``_wire_evaluation_services`` sets on the live ``TldwCli`` app
    (``self.evaluation_orchestrator = EvaluationOrchestrator(...)``, which
    wraps a real ``EvalsDB`` as ``.db``) -- EvalsScreen._resolve_db reads
    that exact path, so this fake reproduces the real wiring shape rather
    than inventing a parallel attribute name a real app would never
    populate.
    """

    def __init__(self, db: EvalsDB) -> None:
        self.evaluation_orchestrator = _FakeOrchestrator(db)
        self.notifications: list[tuple[str, str]] = []

    def notify(self, message: str, *, severity: str = "information", **kwargs) -> None:
        self.notifications.append((message, severity))


class EvalsHarness(App):
    """Minimal Textual App hosting the real EvalsScreen against a real
    (``:memory:``) EvalsDB -- pushed via ``push_screen`` exactly like the
    real app's master shell routing pushes a destination screen, so
    EvalsScreen runs through the identical ``BaseAppScreen.compose()`` ->
    ``compose_content()`` chain (MainNavigationBar, the screen-content
    Container, AppFooterStatus) it runs through in production. See the
    module-level ``_BUNDLED_CSS_PATH`` comment for why the real stylesheet
    is also loaded.
    """

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


@pytest.fixture
def seeded_bench(evals_db: EvalsDB) -> str:
    """One real word bench, named to match the program's own naming
    convention (Tests/Evals/word_bench/conftest.py's "loaded-nouns v1")
    so #evals-primary-action can name it."""
    base_model_id = evals_db.create_model(
        name="base", provider="llama_cpp", model_id="m"
    )
    dataset_id = evals_db.create_dataset(
        name="loaded-nouns", format="custom", source_path="inline:loaded-nouns"
    )
    config = BenchConfig(
        name="loaded-nouns v1",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(base_model_id,),
        probes=(" Sure", " I"),
    )
    return save_bench(evals_db, config)


@pytest.mark.asyncio
async def test_screen_mounts_the_three_pane_workbench(evals_app):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        assert screen.query_one("#evals-workbench")
        assert screen.query_one("#evals-library-pane")
        assert screen.query_one("#evals-detail-pane")
        assert screen.query_one("#evals-inspector-pane")


@pytest.mark.asyncio
async def test_workbench_body_is_not_empty(evals_app):
    """The regression that motivated this PR: the hub rendered header and
    strip with nothing beneath."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        pane = evals_app.screen.query_one("#evals-library-pane")
        assert list(pane.children), "library pane rendered no children"


@pytest.mark.asyncio
async def test_workbench_pane_has_a_real_rendered_region(evals_app):
    """Closes a gap found empirically while building this harness (see the
    Task 3 report): a Screen mounted inside a Container mounts
    STRUCTURALLY -- `query_one`/`children` checks (the two tests above)
    both pass against it -- but the compositor never gives it a laid-out
    region; a throwaway repro showed its descendants report
    `region=Region(0, 0, 0, 0)` even though they exist in the DOM. A bare
    presence check cannot tell that apart from a genuinely rendered pane.
    This screen uses zero Screen subclasses, so its panes get a real
    region from ordinary widget layout; asserting that directly is what
    would have caught the retired hub's actual (visual, PR-1-screenshot)
    defect, which the brief's own DOM-presence tests structurally cannot.
    """
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        pane = evals_app.screen.query_one("#evals-library-pane")
        assert pane.region.width > 0
        assert pane.region.height > 0


@pytest.mark.asyncio
async def test_library_rail_shows_three_sections_with_counts(evals_app):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        labels = [
            w.renderable.plain if hasattr(w.renderable, "plain") else str(w.renderable)
            for w in evals_app.screen.query(".evals-rail-section-label")
        ]
        joined = " ".join(labels)
        for section in ("Benches", "Datasets", "Runs"):
            assert section in joined


@pytest.mark.asyncio
async def test_primary_action_names_its_object_when_a_bench_is_selected(
    evals_app, seeded_bench
):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()
        action = evals_app.screen.query_one("#evals-primary-action")
        assert "loaded-nouns" in str(action.label)


@pytest.mark.asyncio
async def test_primary_action_is_disabled_with_a_reason_when_nothing_is_selected(
    evals_app,
):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        action = evals_app.screen.query_one("#evals-primary-action")
        assert action.disabled is True
        assert action.tooltip, "a disabled primary action must say why"


@pytest.mark.asyncio
async def test_screen_does_not_push_a_child_screen_on_mount(evals_app):
    """Ported from the retired test_evals_screen_shell.py.

    The destination used to push a Screen over itself on mount, hiding the
    shell chrome and stranding users on a placeholder after Escape.
    """
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(evals_app.screen, EvalsScreen)
        assert len(evals_app.screen_stack) == 2


@pytest.mark.asyncio
async def test_escape_does_not_pop_the_shell_screen(evals_app):
    """Ported from the retired test_evals_screen_shell.py."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(evals_app.screen, EvalsScreen)


@pytest.mark.asyncio
async def test_escape_and_bare_digits_are_no_longer_bound(evals_app):
    """Both existed only for the retired card hub."""
    bound = {b.key for b in EvalsScreen.BINDINGS}
    assert "escape" not in bound
    assert not bound & {"1", "2", "3", "4", "5", "6"}
