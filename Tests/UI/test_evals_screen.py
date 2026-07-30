"""Evals screen shell. The old hub rendered an empty body because it mounted
Screen objects inside a Container; these tests pin that the replacement
actually puts widgets on screen."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from loguru import logger as loguru_logger
from textual.app import App
from textual.widget import Widget
from textual.widgets import Button

import tldw_chatbook
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig,
    CellCapture,
    PreflightResult,
    Target,
    TokenProb,
)
from tldw_chatbook.Evals.word_bench.storage import create_run_group, save_bench
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen
from tldw_chatbook.UI.Evals.snippet_editor import import_snippets_into_dataset
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

    def __init__(self, db: EvalsDB, app_config: dict | None = None) -> None:
        self.evaluation_orchestrator = _FakeOrchestrator(db)
        self.notifications: list[tuple[str, str]] = []
        #: Read by EvalsScreen._current_app_config for
        #: sample_bench.provider_is_configured's gate -- see
        #: test_evals_empty_states.py for scenarios that set this.
        self.app_config: dict = app_config or {}

    def notify(
        self, message: str, *, severity: str = "information", markup: bool = True,
        **kwargs,
    ) -> None:
        """A pure recorder, except for one deliberate exception: when
        ``markup`` is left at its (Textual-matching) default of ``True``,
        this parses ``message`` through the SAME ``Content.from_markup``
        call the real ``Toast.render()`` uses (``textual/widgets/_toast.py``)
        -- raising the same ``textual.markup.MarkupError`` on unbalanced
        markup (e.g. a bare ``[/]``) that crashes the real app. TASK-1476's
        review found this reachable: ``EvalsScreen`` interpolates exception
        text (which can carry a user-controlled dataset/bench name) into
        `notify()` calls, and a bare recorder that never looks at the text
        would let a regression here pass silently. ``markup=False`` (what
        every real call site now passes) skips this entirely, matching
        ``Toast.render()``'s own ``else`` branch.
        """
        if markup:
            from textual.content import Content  # noqa: PLC0415 -- narrow, mirrors _toast.py's own import

            Content.from_markup(message)
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
        assert screen.query_one("#lab-workbench")
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
async def test_workbench_height_matches_available_body_height(evals_app):
    """C1 regression: `#evals-workbench` carries classes `"ds-panel
    destination-workbench"`. `.destination-workbench { height: 1fr }`
    (layout/_panes.tcss) and `.ds-panel { height: auto; min-height: 3 }`
    (components/_agentic_terminal.tcss) are equal specificity (one class
    each), and `_agentic_terminal.tcss` sits later in the build manifest --
    so `auto` used to win, collapsing the workbench to a fixed ~11 rows
    regardless of terminal size (measured 11 of 35 available rows at
    160x45 during the PR 3a review). Fixed by an id-level `#evals-workbench`
    rule (mirroring the eight sibling destinations that already carry one)
    at the SAME position in the cascade as those siblings, so it outranks
    both classes by specificity rather than by manifest-order luck.

    Asserted against the space actually available below the workbench
    (screen height minus everything already laid out above it), not a
    hardcoded row count, so this survives a header/mode-strip height
    change without going stale.
    """
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = evals_app.screen
        workbench = screen.query_one("#lab-workbench")
        available = screen.size.height - workbench.region.y
        unused = available - workbench.region.height
        assert 0 <= unused <= 2, (
            f"workbench height {workbench.region.height} leaves {unused} "
            f"rows unused out of {available} available at the bottom of "
            f"the screen"
        )
        # The historical collapse was a small FIXED height regardless of
        # terminal size -- guard against a regression back to that shape,
        # not just against this one screen size happening to work out.
        assert workbench.region.height > 20, (
            f"workbench height {workbench.region.height} looks like the "
            "historical fixed small collapse, not a real 1fr height"
        )


@pytest.mark.asyncio
async def test_every_pane_descendant_stays_within_its_pane(evals_app, seeded_bench):
    """The upgrade over a bare `region.width > 0` check (which proves "laid
    out", not "visible"): C1's collapse laid every workbench child out at a
    real, non-zero region that nonetheless sat OUTSIDE its pane's clip
    rectangle -- e.g. `#evals-import-snippets` at y=18 against a detail
    pane clipped to y<18, where `pilot.click` never reached its handler
    (see `test_import_button_stays_in_its_pane_and_pilot_click_opens_the_
    picker` below for that specific case). This asserts full containment
    for every rendered descendant of all three panes with a bench
    selected, which exercises the richest content (library rail rows, the
    bench editor's target table, the inspector's readiness list and
    primary action button)."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = evals_app.screen
        screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()

        checked = 0
        for pane_id in (
            "#evals-library-pane",
            "#evals-detail-pane",
            "#evals-inspector-pane",
        ):
            pane = screen.query_one(pane_id)
            for descendant in pane.walk_children(Widget):
                if descendant.region.width == 0 or descendant.region.height == 0:
                    continue  # not actually rendered (e.g. a collapsed rail section)
                assert pane.region.contains_region(descendant.region), (
                    f"{descendant!r} at {descendant.region} escapes "
                    f"{pane_id}'s clip region {pane.region}"
                )
                checked += 1
        assert checked > 0, "no descendants were actually checked"

        # Library-pane descendant, named explicitly: the other two panes
        # already had one before this fix (`#evals-detail-empty`,
        # `#evals-primary-action`, in the test above) -- the rail's own
        # containment was only ever incidental, never separately pinned.
        rail_row = screen.query_one("#evals-rail-row-benches-0")
        library_pane = screen.query_one("#evals-library-pane")
        assert library_pane.region.contains_region(rail_row.region)


@pytest.mark.asyncio
async def test_import_button_stays_in_its_pane_and_pilot_click_opens_the_picker(
    evals_app, evals_db
):
    """C1's most concrete symptom: `#evals-import-snippets` -- the snippet
    editor's only control -- used to lay out at a real, non-zero region
    that sat outside `#evals-detail-pane`'s clip rectangle, so `pilot.click`
    never fired its handler. Driven through `pilot.click` deliberately,
    never by calling `_open_import_dialog`/`_handle_import_file_selected`
    directly -- calling the callback would still pass even with the pane
    collapsed, which is exactly how this defect shipped undetected."""
    dataset_id = evals_db.create_dataset(
        name="click-target", format="custom", source_path="inline:click-target"
    )
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        screen = evals_app.screen

        pane = screen.query_one("#evals-detail-pane")
        button = screen.query_one("#evals-import-snippets")
        assert pane.region.contains_region(button.region), (
            f"import button {button.region} escapes detail pane {pane.region}"
        )

        stack_depth_before = len(evals_app.screen_stack)
        await pilot.click("#evals-import-snippets")
        await pilot.pause()

        assert len(evals_app.screen_stack) == stack_depth_before + 1
        assert isinstance(evals_app.screen, FileOpen)


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
async def test_primary_action_reason_is_visible_without_hovering(evals_app):
    """TASK-1076: a disabled Textual ``Button`` never emits ``Pressed`` --
    the previous test proves the button IS disabled with a tooltip, but a
    tooltip only reaches a user who hovers with a mouse. The reason must
    also be reachable as plain, always-rendered text -- mirroring
    ``EvalsInspector``'s own readiness convention (a status badge naming
    what is blocked, plus a callout stating why), not a second,
    invented vocabulary.
    """
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        status = screen.query_one("#evals-primary-action-status")
        assert str(status.renderable) == "Run Bench: Blocked"
        reason = screen.query_one("#evals-primary-action-reason")
        assert "Select a bench in the library rail to run it." in str(
            reason.renderable
        )
        # Both sit in the inspector pane, ahead of the button itself --
        # never silently mounted somewhere the user would not see them
        # alongside the control they explain.
        inspector_pane = screen.query_one("#evals-inspector-pane")
        assert inspector_pane.region.contains_region(status.region)
        assert inspector_pane.region.contains_region(reason.region)


@pytest.mark.asyncio
async def test_primary_action_is_enabled_and_names_the_bench_once_one_is_selected(
    evals_app, seeded_bench
):
    """TASK-1476: a found, selected bench now ENABLES the primary action --
    the Blocked badge/callout (``#evals-primary-action-status`` /
    ``#evals-primary-action-reason``) only render while ``disabled`` (see
    ``_compose_inspector_pane``), so once the button is enabled they must
    not render at all; the ready reason lives on the button's own tooltip
    instead, tracking the SAME per-selection explanation
    ``_primary_action_state`` produces."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()
        screen = evals_app.screen
        action = screen.query_one("#evals-primary-action", Button)
        assert "loaded-nouns" in str(action.label)
        assert action.disabled is False
        assert "loaded-nouns v1" in str(action.tooltip)
        assert not screen.query("#evals-primary-action-status")
        assert not screen.query("#evals-primary-action-reason")


@pytest.mark.asyncio
async def test_primary_action_state_stays_disabled_for_an_unresolvable_bench(
    evals_app,
):
    """A ``kind="bench"`` selection naming an id with no matching row (e.g.
    deleted between the rail rendering it and this being read) must stay
    disabled with its own reason -- unchanged by wiring the found-bench
    branch."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        screen.select(kind="bench", id="does-not-exist")
        await pilot.pause()
        label, disabled, tooltip = screen._primary_action_state()
        assert label == "Run Bench"
        assert disabled is True
        assert "no longer exists" in tooltip


@pytest.mark.asyncio
async def test_primary_action_state_stays_disabled_for_a_dataset_selection(
    evals_app, evals_db
):
    dataset_id = evals_db.create_dataset(
        name="ds", format="custom", source_path="inline:ds"
    )
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        label, disabled, tooltip = screen._primary_action_state()
        assert label == "Run Bench"
        assert disabled is True
        assert "Datasets are run from within a bench" in tooltip


@pytest.mark.asyncio
async def test_primary_action_state_stays_disabled_for_a_completed_run_group(
    evals_app, evals_db
):
    base_id = evals_db.create_model(name="base", provider="llama_cpp", model_id="m")
    dataset_id = evals_db.create_dataset(
        name="rg-ds", format="custom", source_path="inline:rg-ds"
    )
    config = BenchConfig(
        name="rg bench", prompt_mode="raw", top_k=5,
        dataset_id=dataset_id, target_ids=(base_id,),
    )
    task_id = save_bench(evals_db, config)
    target = Target(id=base_id, name="base", provider="llama_cpp", model_id="m")
    group_id, _run_ids = create_run_group(evals_db, task_id, config, [target], [])

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        screen.select(kind="run_group", id=group_id)
        await pilot.pause()
        label, disabled, tooltip = screen._primary_action_state()
        assert label == "Run Bench"
        assert disabled is True
        assert "This run has already completed" in tooltip


def test_the_not_wired_up_copy_no_longer_exists_anywhere_in_the_app():
    """TASK-1476: the primary action is wired up now -- the old deferral
    copy (and the docstring paragraph mirroring it) must be gone entirely,
    not merely unreachable from a live selection."""
    package_root = Path(tldw_chatbook.__file__).parent
    offenders = [
        path
        for path in package_root.rglob("*.py")
        if "isn't wired up yet" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# Pressing the primary action -- the wiring itself (TASK-1476).
# ---------------------------------------------------------------------------


class _FakeCaptureClient:
    """Mirrors ``test_evals_empty_states.py``'s own fake -- duplicated per
    that module's convention (fakes are not shared/imported across test
    modules here)."""

    def __init__(self, calls: list) -> None:
        self._calls = calls

    async def preflight(self, target, mode, top_k):
        return PreflightResult(state="ok", k_returned=5, canary="pass")

    async def capture(self, snippet, target, mode, top_k):
        self._calls.append((snippet, target.name))
        return CellCapture(
            prompt_mode=mode, k_requested=top_k, k_returned=1, content_offset=0,
            top_k=(TokenProb(token=" a", logprob=-0.3, token_id=1),),
            canary="unchecked", captured_at="2026-07-30T00:00:00Z",
        )


class _PausableFakeCaptureClient:
    """Blocks on ``release_event`` inside ``capture`` (never ``preflight``)
    -- gives a test a controllable window in which a run is genuinely in
    flight, mirroring ``test_evals_empty_states.py``'s own
    ``_PausableFakeCaptureClient``."""

    def __init__(self, calls: list, release_event: "asyncio.Event") -> None:
        self._calls = calls
        self._release_event = release_event

    async def preflight(self, target, mode, top_k):
        return PreflightResult(state="ok", k_returned=5, canary="pass")

    async def capture(self, snippet, target, mode, top_k):
        await self._release_event.wait()
        self._calls.append((snippet, target.name))
        return CellCapture(
            prompt_mode=mode, k_requested=top_k, k_returned=1, content_offset=0,
            top_k=(TokenProb(token=" a", logprob=-0.3, token_id=1),),
            canary="unchecked", captured_at="2026-07-30T00:00:00Z",
        )


async def _wait_until(pilot, predicate, *, tries: int = 300, interval: float = 0.02) -> None:
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause(interval)
    raise AssertionError("condition never became true")


@pytest.fixture
def runnable_bench(evals_db: EvalsDB) -> str:
    """A bench whose dataset carries a real snippet -- unlike
    ``seeded_bench`` (whose empty dataset only ever needs to exist for
    naming/count tests), this is the minimum
    ``sample_bench.run_existing_bench`` needs to actually complete a run
    instead of raising "has no snippets to run"."""
    base_model_id = evals_db.create_model(
        name="base", provider="llama_cpp", model_id="m"
    )
    dataset_id = evals_db.create_dataset(
        name="loaded-nouns", format="custom", source_path="inline:loaded-nouns"
    )
    import_snippets_into_dataset(
        evals_db,
        dataset_id,
        [{"id": "s1", "text": "The protestors were", "group": "neutral", "note": None}],
    )
    config = BenchConfig(
        name="loaded-nouns v1", prompt_mode="raw", top_k=20,
        dataset_id=dataset_id, target_ids=(base_model_id,),
        probes=(" Sure", " I"),
    )
    return save_bench(evals_db, config)


@pytest.mark.asyncio
async def test_pressing_the_primary_action_runs_the_bench_and_selects_its_run_group(
    evals_app, runnable_bench
):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=runnable_bench)
        await pilot.pause()
        calls: list = []
        screen._sample_bench_client_factory = lambda t: _FakeCaptureClient(calls)

        assert screen._view_model.run_groups() == []
        # `#evals-primary-action` sits at the bottom of `#evals-inspector-
        # pane`, inside `#lab-inspector` (a VerticalScroll) -- a pre-
        # existing virtual-size/container-size off-by-one in that pane
        # (same shape as the rail's own, already fixed in _lab.tcss; see
        # its comment there) leaves the button's own row scrolled one cell
        # past the viewport with a fresh selection, painted-over by the
        # footer rather than the button -- `pilot.click` would silently hit
        # the footer instead. `scroll_visible` is exactly what a real user
        # would need to do first; it is not a workaround for anything this
        # task changed.
        screen.query_one("#evals-primary-action").scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._selection.kind == "run_group")
        await pilot.pause()

        assert len(calls) == 1
        run_groups = screen._view_model.run_groups()
        assert len(run_groups) == 1
        assert screen._selection.id == run_groups[0]["id"]
        assert run_groups[0]["task_id"] == runnable_bench


class _RaisingCaptureClient:
    """A client whose ``preflight`` raises with markup-hazard text baked
    into the message -- stands in for the real hazard
    (``sample_bench._load_snippets``'s ``RuntimeError(f"Dataset {name!r}
    has no snippets to run.")``, where an imported dataset's name defaults
    to the imported filename's stem, so a file named ``notes[/].txt``
    would carry it) without needing a REAL dataset/bench/target actually
    named with a bare ``[/]`` -- that would ALSO crash
    ``LibraryRail``'s own rail-row ``Button(label=...)`` the moment the
    rail composes (a separate, pre-existing, out-of-scope hazard the
    controller is tracking on its own), which would fail this test for
    the wrong reason before the click under test even happens."""

    async def preflight(self, target, mode, top_k):
        raise RuntimeError("Target 'notes[/].txt' could not be reached.")

    async def capture(self, snippet, target, mode, top_k):
        raise AssertionError("capture must not be reached -- preflight fails first")


@pytest.mark.asyncio
async def test_bench_run_failure_toast_with_markup_hazard_text_does_not_crash_the_app(
    evals_app, runnable_bench
):
    """TASK-1476 review Critical: ``_run_bench_worker``'s error ``notify()``
    interpolates the caught exception's message, which can carry user-
    controlled text (see ``_RaisingCaptureClient``'s own docstring for the
    real-world shape) -- a bare ``[/]`` is unbalanced Rich/Textual markup
    (``textual.markup.MarkupError``). That path was unreachable before
    this task wired up the button (it was always disabled); this pins
    that a bench run failing with hazard text in its exception message
    produces an error toast WITHOUT crashing the app, and that the toast
    carries the raw, unmangled text. ``_FakeAppInstance.notify`` parses
    real markup exactly like ``Toast.render()`` does when ``markup=True``
    -- see its own docstring -- so this fails loudly (a real
    ``MarkupError``, propagating out of the worker as an unhandled
    exception) if a future edit ever drops ``markup=False`` from that call
    again.
    """
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=runnable_bench)
        await pilot.pause()
        screen._sample_bench_client_factory = lambda t: _RaisingCaptureClient()

        button = screen.query_one("#evals-primary-action")
        button.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#evals-primary-action")  # must not crash the app
        await _wait_until(pilot, lambda: not screen._bench_run_running)
        await pilot.pause()

        assert pilot.app.is_running, "the app must survive the failure toast"
        notifications = evals_app.app_instance.notifications
        assert notifications, "the failure must still produce a toast"
        message, severity = notifications[-1]
        assert severity == "error"
        assert "[/]" in message, message
        assert "could not be reached" in message


@pytest.mark.asyncio
async def test_a_second_press_while_a_bench_run_is_in_flight_is_a_no_op(
    evals_app, runnable_bench
):
    """Mirrors ``test_a_second_click_while_running_does_not_start_a_second_
    run`` (test_evals_empty_states.py) for the run-existing-bench worker:
    posts the Pressed event directly (simulating whatever might get past a
    disabled-but-not-yet-rerendered button) and proves it is a genuine
    no-op -- exactly one run group, never two."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=runnable_bench)
        await pilot.pause()
        release = asyncio.Event()
        calls: list = []
        screen._sample_bench_client_factory = lambda t: _PausableFakeCaptureClient(
            calls, release
        )

        button = screen.query_one("#evals-primary-action", Button)
        # See the sibling press test's comment above -- scroll the button
        # into its scroll container's viewport before clicking.
        button.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._bench_run_running)
        await pilot.pause()
        assert button.disabled is True

        screen.post_message(Button.Pressed(button))
        await pilot.pause()

        release.set()
        await _wait_until(pilot, lambda: not screen._bench_run_running)
        await pilot.pause()

        assert len(calls) == 1  # sanity check -- holds even without the guard
        # The assertion that actually discriminates: dropping the
        # `if self._bench_run_running: return` guard (while keeping
        # `exclusive=True`) would produce 2 here, because the second
        # `run_worker` call cancels the already-running worker via the
        # shared exclusive group AFTER it created its own run group, then a
        # fresh worker creates a second one.
        assert len(screen._view_model.run_groups()) == 1


@pytest.mark.asyncio
async def test_inspector_reports_an_unexpected_load_bench_failure_instead_of_going_blank(
    evals_app, seeded_bench, monkeypatch
):
    """Regression (TASK-861 item 5): ``EvalsInspector.compose`` caught
    ``load_bench``'s failure with a bare ``except Exception: return`` --
    since ``compose()`` is a generator, that yielded ZERO widgets, leaving a
    blank inspector pane with no message and no log line: nothing to
    diagnose from.

    ``_compose_inspector_pane`` only mounts ``EvalsInspector`` once
    ``EvalsViewModel.bench_by_id`` has already found the bench (see
    ``evals_screen.py``), so reaching this branch for real means either a
    race (deleted between that read and this one) or an unexpected failure
    below it -- simulated here the same way
    ``test_unexpected_load_grid_failure_renders_error_state_without_crashing_the_app``
    (``test_evals_results_grid.py``) simulates the analogous failure for
    ``ResultsGrid``: monkeypatching ``load_bench`` in the ``inspector``
    module's own namespace to raise a DB-level fault for an otherwise valid
    bench id.
    """
    import sqlite3

    from tldw_chatbook.UI.Evals import inspector as inspector_module

    def _raise_operational_error(db, task_id):
        raise sqlite3.OperationalError("no such table: eval_tasks")

    monkeypatch.setattr(inspector_module, "load_bench", _raise_operational_error)

    records: list[dict] = []
    sink_id = loguru_logger.add(lambda message: records.append(message.record), level="ERROR")
    try:
        async with evals_app.run_test(size=(160, 45)) as pilot:
            await pilot.pause()
            evals_app.screen.select(kind="bench", id=seeded_bench)
            await pilot.pause()

            error_state = evals_app.screen.query_one("#evals-inspector-error")
            message = str(error_state.renderable)
            assert "unexpected error" in message

            # A failed load must not also render stale/partial readiness
            # rows -- compose() returned before yielding any of them.
            # Filtered in Python by id prefix (not the bare
            # ".ds-status-badge" class, which the shell chrome's own
            # workbench-header-status Static also carries for an unrelated
            # purpose, and not a CSS attribute selector -- Textual's own
            # selector grammar doesn't support "^=" prefix matching).
            target_badges = [
                w
                for w in evals_app.screen.query(Widget)
                if w.id and w.id.startswith("evals-inspector-target-")
            ]
            assert not target_badges, target_badges
    finally:
        loguru_logger.remove(sink_id)

    assert records, "the failure must be logged, not just rendered"
    assert any(seeded_bench in r["message"] for r in records), (
        f"the log line must name which bench failed: {[r['message'] for r in records]}"
    )


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


@pytest.mark.asyncio
async def test_detail_empty_text_points_at_the_sample_bench_when_the_library_is_empty(
    evals_app,
):
    """TASK-1076: the old, single wording ("Select a bench, dataset, or
    run...") is unactionable exactly when it is guaranteed to show -- a
    first launch, where the rail has nothing to select. ``evals_app`` seeds
    nothing (no benches, datasets, or runs), so this is that condition."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        empty = evals_app.screen.query_one("#evals-detail-empty")
        text = str(empty.renderable)
        assert "Nothing here yet" in text
        assert "sample bench" in text


@pytest.mark.asyncio
async def test_detail_empty_text_stays_generic_when_the_library_has_real_rows(
    evals_app, seeded_bench
):
    """The genuinely-empty-library wording must NOT show once real rows
    exist -- a user who has a bench (and a dataset, via ``seeded_bench``)
    but has not clicked anything yet still gets pointed at the rail, not
    told there is nothing there."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        assert evals_app.screen._selection.kind == "none"
        empty = evals_app.screen.query_one("#evals-detail-empty")
        text = str(empty.renderable)
        assert "Nothing here yet" not in text
        assert "Select a bench, dataset, or run" in text


@pytest.mark.asyncio
async def test_selection_change_recompose_reads_tasks_once_for_the_empty_state_check(
    evals_app, evals_db, monkeypatch
):
    """TASK-1076-QODO: ``_empty_detail_text()`` used to call
    ``view_model.benches()`` and ``view_model.classic_tasks()`` back to
    back on every selection-change recompose -- each independently
    re-running ``EvalsDB.list_tasks(limit=500)`` -- on top of
    ``LibraryRail``'s own two reads of the exact same table for the same
    recompose (out of this fix's scope; it needs the full rows to render
    rail sections, not just an emptiness check). Pins the fixed shape: the
    rail still costs two ``list_tasks`` calls, but the detail pane's
    emptiness check (now ``EvalsViewModel.library_is_empty()``) costs
    exactly one more, not two -- three calls per recompose, never the old
    four. A test that only checked the copy (see the two tests above)
    would pass against the inefficient version just as well; this is the
    assertion that actually protects the fix.
    """
    calls: list[int] = []
    original_list_tasks = EvalsDB.list_tasks

    def counting_list_tasks(self, *args, **kwargs):
        calls.append(1)
        return original_list_tasks(self, *args, **kwargs)

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(EvalsDB, "list_tasks", counting_list_tasks)
        calls.clear()
        evals_app.screen.select(kind="none", id=None)
        await pilot.pause()
        assert len(calls) == 3


@pytest.mark.asyncio
async def test_inspector_pane_widens_at_a_wide_terminal_instead_of_staying_fixed(
    evals_app, seeded_bench
):
    """TASK-1076: ``#lab-inspector`` was a hard 30 cells at every terminal
    width (the same anti-pattern ``#lab-rail`` was already fixed for) --
    at 200 columns that left the Evals inspector's own content (e.g. the
    focused-cell detail's "K requested 20 * K returned 20 * canary ..."
    line) wrapping mid-phrase with most of a wide terminal sitting unused
    beside it. Selecting a bench (rather than leaving the selection empty)
    exercises the pane with its richest real content, mirroring
    ``test_every_pane_descendant_stays_within_its_pane`` above.
    """
    widths: dict[int, int] = {}
    for columns in (80, 200):
        app = EvalsHarness(_FakeAppInstance(EvalsDB(db_path=":memory:", client_id="t")))
        # Reseed a bench per app instance -- a Textual App is not re-runnable.
        base_model_id = app.app_instance.evaluation_orchestrator.db.create_model(
            name="base", provider="llama_cpp", model_id="m"
        )
        dataset_id = app.app_instance.evaluation_orchestrator.db.create_dataset(
            name="loaded-nouns", format="custom", source_path="inline:loaded-nouns"
        )
        bench_id = save_bench(
            app.app_instance.evaluation_orchestrator.db,
            BenchConfig(
                name="loaded-nouns v1", prompt_mode="raw", top_k=20,
                dataset_id=dataset_id, target_ids=(base_model_id,),
                probes=(" Sure", " I"),
            ),
        )
        async with app.run_test(size=(columns, 45)) as pilot:
            await pilot.pause()
            app.screen.select(kind="bench", id=bench_id)
            await pilot.pause()
            widths[columns] = app.screen.query_one("#lab-inspector").region.width

    assert 30 <= widths[80] <= 50
    assert 30 <= widths[200] <= 50
    assert widths[80] < widths[200], (
        f"inspector did not scale with the terminal: {widths}"
    )
