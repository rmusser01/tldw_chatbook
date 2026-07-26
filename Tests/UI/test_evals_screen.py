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
async def test_workbench_panes_and_their_descendants_have_a_real_rendered_region(
    evals_app,
):
    """Closes a gap found empirically while building this harness (see the
    Task 3 report): a Screen mounted inside a Container mounts
    STRUCTURALLY -- `query_one`/`children` checks (the two tests above)
    both pass against it -- but the compositor never gives it a laid-out
    region.

    An earlier version of this test asserted region on `#evals-library-pane`
    alone, which review caught as too weak: a repro of the actual
    historical shape (a Screen mounted *inside* a workbench pane, not as
    the pane itself) showed the WRAPPING pane still reports a real region
    -- it's an ordinary Container, unaffected -- while only the nested
    Screen and its descendants collapse to `Region(0, 0, 0, 0)`. A pane-only
    assertion would still pass if someone reintroduced a Screen one level
    deeper (e.g. inside `#evals-detail-pane`), and so would every other
    test in this file: the pane is still present, `#evals-primary-action`
    is still findable *through* the nested Screen with its label and
    disabled state intact (`query_one` walks the whole DOM regardless of
    layout), and `test_screen_does_not_push_a_child_screen_on_mount` only
    checks `screen_stack`, which mounting a Screen never touches.

    So this asserts region on all three panes AND on a descendant inside
    each of the two swappable ones (`#evals-detail-empty`,
    `#evals-primary-action`) -- a descendant is what actually distinguishes
    "real widgets" from "a Screen wrapping real widgets." Verified against
    a reconstructed nested-Screen-in-detail-pane shape during review: this
    amended test fails on the descendant assertion (region 0x0) while the
    pane-only assertions above it still pass -- see the Task 3 fix report.
    """
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        for pane_id in (
            "#evals-library-pane",
            "#evals-detail-pane",
            "#evals-inspector-pane",
        ):
            pane = screen.query_one(pane_id)
            assert pane.region.width > 0, pane_id
            assert pane.region.height > 0, pane_id

        detail_descendant = screen.query_one("#evals-detail-empty")
        assert detail_descendant.region.width > 0
        assert detail_descendant.region.height > 0

        inspector_descendant = screen.query_one("#evals-primary-action")
        assert inspector_descendant.region.width > 0
        assert inspector_descendant.region.height > 0


@pytest.mark.asyncio
async def test_library_rail_shows_three_sections_with_counts(evals_app, seeded_bench):
    """`seeded_bench` also creates the one dataset it needs (see the
    fixture), and creates no runs -- so the live counts should read
    Benches (1), Datasets (1), Runs (0). The original version of this test
    (from the brief) only grepped for the section words and never actually
    asserted a count despite its own name; review caught that gap."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        labels = [
            w.renderable.plain if hasattr(w.renderable, "plain") else str(w.renderable)
            for w in evals_app.screen.query(".evals-rail-section-label")
        ]
        joined = " ".join(labels)
        assert "Benches (1)" in joined
        assert "Datasets (1)" in joined
        assert "Runs (0)" in joined


@pytest.mark.asyncio
async def test_selecting_a_bench_row_in_the_rail_updates_the_detail_pane(
    evals_app, seeded_bench
):
    """Every other selection test in this file drives selection by calling
    `screen.select()` directly, which leaves the actual user path --
    LibraryRail's row Button -> `on_button_pressed` ->
    `post_message(EvalsSelectionChanged)` -> EvalsScreen's
    `_on_library_selection_changed` handler -> `select()` -- completely
    unexercised. That path is this screen's primary interaction, and the
    one Tasks 4/5 mount their editors on; drive it end to end here instead
    of stubbing it out."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#evals-rail-row-benches-0")
        await pilot.pause()
        name = evals_app.screen.query_one("#evals-detail-bench-name")
        assert "loaded-nouns v1" in str(name.renderable)


@pytest.mark.asyncio
async def test_collapsing_a_rail_section_hides_its_rows(evals_app, seeded_bench):
    """The collapse/expand toggle is the other rail interaction no test
    exercised; verify a press actually hides the section body rather than
    just flipping an internal flag nothing reads."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        assert evals_app.screen.query_one("#evals-rail-row-benches-0")
        await pilot.click("#evals-rail-toggle-benches")
        await pilot.pause()
        body = evals_app.screen.query_one("#evals-rail-section-body-benches")
        assert body.styles.display == "none"


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
