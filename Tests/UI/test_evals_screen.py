"""Evals screen shell. The old hub rendered an empty body because it mounted
Screen objects inside a Container; these tests pin that the replacement
actually puts widgets on screen."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
from loguru import logger as loguru_logger
from rich.markup import escape as escape_markup
from textual.app import App
from textual.widget import Widget
from textual.widgets import Button, Input, Select

import tldw_chatbook
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import (
    BenchConfig,
    CellCapture,
    PreflightResult,
    Snippet,
    Target,
    TokenProb,
)
from tldw_chatbook.Evals.word_bench.storage import create_run_group, save_bench
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen
from tldw_chatbook.UI.Evals.bench_editor import BenchEditor
from tldw_chatbook.UI.Evals.library_rail import LibraryRail
from tldw_chatbook.UI.Evals.snippet_editor import import_snippets_into_dataset
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

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
def sample_bench_app(evals_db: EvalsDB) -> EvalsHarness:
    """Mirrors ``test_evals_empty_states.py``'s own ``configured_app``: a
    configured llama.cpp endpoint, zero benches -- what a bare
    ``evals_app`` (no configured provider, no pre-existing ``eval_models``
    row) does not give ``#evals-create-sample-bench`` to render at all
    (see ``library_rail.py``'s "no configured provider" branch)."""
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    return EvalsHarness(_FakeAppInstance(evals_db, app_config=app_config))


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

        # task-1710: the per-cell continuation checkbox added one row to
        # `BenchEditor`'s own form -- confirmed live (a worktree diff
        # against the parent commit) that this is the ONE thing that
        # changed: `#evals-bench-targets-section` (the Add picker plus
        # the always-rendered "+ New target" mini-form) no longer fits
        # below `#evals-bench-name`/`#evals-bench-description`/prompt
        # mode/Top-K/probes/the checkbox itself even for `seeded_bench`'s
        # single target, where every prior row DID fit. This is the
        # KNOWN, already-documented trade-off `#evals-bench-editor`'s own
        # `overflow-y: auto` exists for (see that id's CSS comment: "this
        # targets section has a small, FIXED budget") -- reachable by
        # scrolling, not lost, and already covered by its own dedicated
        # reachability test (`test_target_rows_stay_reachable_at_4_and_8_
        # targets`, below), which explicitly accepts scrolling for this
        # exact section rather than asserting it always fits unscrolled.
        # Excluded here so THIS test keeps catching every OTHER
        # accidental-clipping regression (name/description/prompt mode/
        # top-K/probes/the checkbox itself, the rail, and the inspector
        # pane) without re-asserting a claim its own sibling test already
        # disproves on purpose.
        try:
            targets_section = screen.query_one("#evals-bench-targets-section")
        except Exception:
            targets_section = None

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
                if targets_section is not None and targets_section in descendant.ancestors_with_self:
                    continue
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


#: The number of target rows a realistic 160x45 viewport genuinely shows
#: without scrolling, once there are that many or more -- confirmed live
#: (see `test_target_rows_stay_reachable_at_4_and_8_targets`'s own
#: docstring). A LITERAL constant, not derived from any live-measured
#: region -- whole-branch review, Minor: an earlier version of this test
#: computed the equivalent count as `min(body.region.height, n_targets)`,
#: which made the assertion self-adjust to whatever a regression produced
#: (e.g. a collapse back to a literal 1-row floor would have shrunk the
#: expected count to 1 and kept passing) instead of failing against a
#: fixed, independently-chosen expectation.
#:
#: task-1710: lowered from 3 to 2. The per-cell continuation opt-in added
#: one more field (a `Checkbox` plus its own `margin-bottom: 1`, the same
#: two-row cost -- content plus trailing margin -- every other field in
#: this form already pays) above the targets section, shifting it down by
#: two rows. Confirmed live (a throwaway `git worktree` at this task's
#: own parent commit, compared row-by-row against the current tree): the
#: true unscrolled limit at 160x45 went from 4 rows to 2; this constant
#: already undercounted the true baseline (4) as a deliberately
#: conservative floor, so it drops to the new true floor rather than to
#: some fraction of the old one.
_TARGET_ROWS_VISIBLE_WITHOUT_SCROLLING = 2


@pytest.mark.asyncio
async def test_target_rows_stay_reachable_at_4_and_8_targets(
    evals_db,
):
    """task-1611 T2 fix round 1, then superseded by the whole-branch review
    fix round below. The "+ New target" mini-form plus the Add picker
    used to be TWO SEPARATE fixed siblings of the row table, each
    independently subtracted from the targets section's own tiny `1fr`
    share at a realistic viewport -- confirmed live that the table's OWN
    box collapsed all the way down to a literal 1-row floor once there
    were enough targets, at which point a 4-target bench's 4th row
    escaped `#evals-detail-pane`'s own clip rectangle at the DEFAULT,
    unscrolled position -- the exact signature
    `test_every_pane_descendant_stays_within_its_pane` above catches.

    Fix round 1's own fix (wrapping the row table, the Add picker, and
    the mini-form in ONE shared scrollable `#evals-bench-targets-body`)
    was ITSELF superseded by the whole-branch review fix round: that
    local scroll level could not also fix a SEPARATE, worse failure (a
    tall `#evals-bench-form-error` callout pushing remedies off the
    SCREEN with no scrolling anywhere in `BenchEditor` at all -- see
    `test_blocked_save_remedies_stay_reachable_on_short_terminals`
    below), and layering a second scroll level on top of the first one
    demonstrably did not work (`BenchEditor`'s own virtual-size
    computation could not see past the body's own separately-scrolling,
    still-small box). `#evals-bench-editor` is now the ONE scrollable
    region for the whole form; see its own CSS comment in `_evals.tcss`
    for the full story, and `#evals-bench-targets-section`'s for why its
    own local scroll was removed rather than kept alongside the outer
    one.

    This test's OWN claims changed to match: (1) the targets heading is
    pane-contained at the default (unscrolled) position, regardless of
    target count; (2) `_TARGET_ROWS_VISIBLE_WITHOUT_SCROLLING` rows (a
    FIXED constant, not read back from live geometry -- see its own
    comment) are ALSO pane-contained at the default position whenever
    there are that many or more targets; and (3) scrolling the OUTER
    editor all the way to its own end genuinely brings the "+ New
    target" button into the pane's clip rectangle -- not merely that
    `max_scroll_y` is non-zero, which is the claim the original T2
    report made WITHOUT live-verifying it.
    """
    for n_targets in (4, 8):
        ids = [
            evals_db.create_model(
                name=f"scale-target-{n_targets}-{i}", provider="llama_cpp", model_id="m"
            )
            for i in range(n_targets)
        ]
        dataset_id = evals_db.create_dataset(
            name=f"scale-set-{n_targets}",
            format="custom",
            source_path=f"inline:scale-set-{n_targets}",
            metadata={"sample_count": 4},
        )
        config = BenchConfig(
            name=f"scale bench {n_targets}",
            prompt_mode="raw",
            top_k=20,
            dataset_id=dataset_id,
            target_ids=tuple(ids),
        )
        task_id = save_bench(evals_db, config)

        app = EvalsHarness(_FakeAppInstance(evals_db))
        async with app.run_test(size=(160, 45)) as pilot:
            await pilot.pause()
            screen = app.screen
            screen.select(kind="bench", id=task_id)
            await pilot.pause()

            editor = screen.query_one(BenchEditor)
            pane = screen.query_one("#evals-detail-pane")
            heading = screen.query_one("#evals-bench-targets-heading")
            assert pane.region.contains_region(heading.region), (
                f"the targets heading at {heading.region} escapes the pane's "
                f"clip region {pane.region} with {n_targets} targets"
            )
            assert editor.max_scroll_y > 0, (
                f"expected {n_targets} targets plus the Add picker and the "
                "create-target form to overflow the editor's own box -- "
                "this test's whole premise needs genuine overflow to prove "
                "anything"
            )

            for index in range(min(_TARGET_ROWS_VISIBLE_WITHOUT_SCROLLING, n_targets)):
                row = screen.query_one(f"#evals-bench-target-{index}")
                assert pane.region.contains_region(row.region), (
                    f"row {index} (of the "
                    f"{_TARGET_ROWS_VISIBLE_WITHOUT_SCROLLING} expected to "
                    f"fit without scrolling) escapes the pane with "
                    f"{n_targets} targets"
                )

            # And scrolling the OUTER editor all the way to its own end
            # genuinely brings the "+ New target" button -- the lowest-
            # priority, always-present affordance in this shared flow --
            # into view.
            editor.scroll_to(y=editor.max_scroll_y, animate=False)
            await pilot.pause()
            create_button = screen.query_one("#evals-bench-create-target")
            assert pane.region.contains_region(create_button.region), (
                "scrolling to the end never brought the create-target "
                f"button into the pane's clip region ({n_targets} targets)"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (120, 40)], ids=["160x45", "120x40"])
async def test_blocked_save_remedies_stay_reachable_on_short_terminals(
    evals_db, size
):
    """Whole-branch review, IMPORTANT. A tall `#evals-bench-form-error`
    callout (this task's own reworded mode-revalidation copy, which wraps
    to 5+ lines at a realistic 160x45/120x40 terminal) is composed ABOVE
    the targets section -- pushing BOTH remedies the error text itself
    names (the offending target row's own `Remove` button, and the "+ New
    target" mini-form) off the bottom of the SCREEN entirely, with
    neither `#evals-detail-pane` nor `BenchEditor` scrollable (confirmed
    live before this fix: `overflow_y` was `hidden`, `allow_vertical_
    scroll` was `False`, and `editor.scroll_to()` moved nothing). A user
    hitting this was told to do two things they could neither see nor
    reach; only flipping the mode `Select` back or resizing the terminal
    escaped. Pre-existing (the Top-K error callout already did this at
    160x45, unrelated to this task), but this task's own copy -- which
    explicitly instructs "remove it" -- walks a user straight into it.

    `#evals-bench-editor` becoming the sole scrollable region for the
    whole form (see its own `_evals.tcss` comment) fixes this: scrolls to
    reveal `#evals-bench-save` first (120x40 needs this even in the
    CLEAN, no-error state -- confirmed live as a second, independently
    real pre-existing gap this same fix happens to close too), clicks it
    to trigger the error, then scrolls again to reach the offending row's
    `Remove` button and the create-target control -- asserting both are
    pane-CONTAINED and that `Screen.get_widget_at` resolves to the actual
    button, not to `footer-spacer` or anything else painted on top of a
    geometrically-plausible-but-actually-obscured position (the exact
    failure mode a bare containment check alone would miss -- see
    `test_every_pane_descendant_stays_within_its_pane`'s own docstring
    for why that distinction matters here too). Red-first against the
    code before this fix round: both `contained` assertions failed
    outright (`OutOfBounds`-adjacent, off the SCREEN's own bottom edge),
    confirmed live before writing this test.
    """
    id1 = evals_db.create_model(name="target-1", provider="llama_cpp", model_id="m")
    id2 = evals_db.create_model(
        name="target-2",
        provider="llama_cpp",
        model_id="m",
        config={"prefix": "Continue: "},
    )
    dataset_id = evals_db.create_dataset(
        name="loaded-nouns",
        format="custom",
        source_path="inline:loaded-nouns",
        metadata={"sample_count": 12},
    )
    config = BenchConfig(
        name="loaded-nouns v1",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(id1, id2),
        probes=(" Sure", " I"),
    )
    task_id = save_bench(evals_db, config)

    app = EvalsHarness(_FakeAppInstance(evals_db))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = app.screen
        screen.select(kind="bench", id=task_id)
        await pilot.pause()

        editor = screen.query_one(BenchEditor)
        pane = screen.query_one("#evals-detail-pane")

        screen.query_one("#evals-bench-prompt-mode", Select).value = "chat"
        await pilot.pause()

        # Reach Save -- itself already off-screen at 120x40 even before
        # any error renders (see the docstring above).
        editor.scroll_to(y=editor.max_scroll_y, animate=False)
        await pilot.pause()
        await pilot.click("#evals-bench-save")
        await pilot.pause()

        callout = screen.query_one("#evals-bench-form-error")
        assert callout.display
        assert "target-2" in str(callout.renderable)

        # The error callout just grew the form's total content height --
        # scroll again to reach its new bottom.
        editor.scroll_to(y=editor.max_scroll_y, animate=False)
        await pilot.pause()
        await pilot.pause()

        # target-2 is index 1 (config.target_ids == (id1, id2)) -- its own
        # Remove button, not any remove button (`.evals-bench-target-
        # remove` alone would match target-1's too, with two staged).
        remove_btn = screen.query_one("#evals-bench-target-remove-1")
        create_btn = screen.query_one("#evals-bench-create-target")

        assert pane.region.contains_region(remove_btn.region), (
            f"target-2's Remove button at {remove_btn.region} escapes the "
            f"pane's clip region {pane.region} at {size[0]}x{size[1]} -- "
            "the error's own remedy is unreachable"
        )
        assert pane.region.contains_region(create_btn.region), (
            f"the create-target button at {create_btn.region} escapes the "
            f"pane's clip region {pane.region} at {size[0]}x{size[1]} -- "
            "the error's own remedy is unreachable"
        )

        resolved_remove, _ = screen.get_widget_at(
            remove_btn.region.x + 1, remove_btn.region.y
        )
        assert resolved_remove is remove_btn, (
            f"the Remove button's own screen position resolves to "
            f"{resolved_remove!r}, not the button itself, at "
            f"{size[0]}x{size[1]} -- painted underneath something else"
        )
        resolved_create, _ = screen.get_widget_at(
            create_btn.region.x + 1, create_btn.region.y
        )
        assert resolved_create is create_btn, (
            f"the create-target button's own screen position resolves to "
            f"{resolved_create!r}, not the button itself, at "
            f"{size[0]}x{size[1]} -- painted underneath something else"
        )


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
        name = evals_app.screen.query_one("#evals-bench-name", Input)
        assert name.value == "loaded-nouns v1"


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
        assert "Select a bench in the Catalog rail to run it." in str(
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
async def test_primary_action_state_stays_disabled_for_a_target_less_bench(
    evals_app, evals_db
):
    """task-1482 fix round 1: a draft bench created via "+ New bench" has
    ``target_ids=()`` until the bench editor (Task 6) wires one on.
    Without this guard, pressing "Run" reached ``run_existing_bench`` with
    zero targets, which "completed" an EMPTY run group -- the exact
    dead-end pattern (a success toast followed by "this run could not be
    found") ``_primary_action_state``'s own naming rule exists to
    prevent, just one step further downstream."""
    dataset_id = evals_db.create_dataset(
        name="ds", format="custom", source_path="inline:ds"
    )
    config = BenchConfig(
        name="draft bench", prompt_mode="raw", top_k=20,
        dataset_id=dataset_id, target_ids=(),
    )
    task_id = save_bench(evals_db, config)

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        screen.select(kind="bench", id=task_id)
        await pilot.pause()
        label, disabled, tooltip = screen._primary_action_state()
        assert label == "Run draft bench"
        assert disabled is True
        # Exact-match. task-1612: appends "and Save" -- staging a target
        # in the bench editor's Add picker does not touch this row's
        # persisted `target_ids` (only Save does), so a user who has just
        # staged one but not yet saved would otherwise read this tooltip
        # as stale/wrong while it still claims "no targets yet".
        assert tooltip == (
            "This bench has no targets yet; add one in the bench editor "
            "and Save."
        )


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
        # Exact-match, deliberately (task-1482): the copy now points at the
        # concrete fix -- "+ New bench" in the Catalog rail -- rather than
        # the old, more general "select a bench that uses this dataset
        # instead" (which presupposed one already existed).
        assert tooltip == (
            "Datasets are run from within a bench; use + New bench in "
            "the Catalog rail to create one against this dataset."
        )


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
        # No `scroll_visible()` needed here (unlike an earlier version of
        # this test): `#evals-inspector-bench { height: auto; }` in
        # _evals.tcss (whole-branch review clipping fix) stopped
        # `EvalsInspector` from claiming the whole `#lab-inspector`
        # viewport as an unstyled `1fr` child, which used to push
        # `#evals-primary-action` out of the visible/painted area --
        # see `test_primary_action_paints_inside_the_inspector_viewport_
        # without_scrolling` for the dedicated regression test.
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

        # No `scroll_visible()` needed -- see the whole-branch review
        # clipping fix's own comment above
        # `test_pressing_the_primary_action_runs_the_bench_and_selects_
        # its_run_group`.
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
        # No `scroll_visible()` needed -- see the whole-branch review
        # clipping fix's own comment on the sibling press test above.
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
async def test_action_buttons_stay_disabled_across_a_mid_run_recompose(
    evals_app, evals_db, runnable_bench
):
    """Whole-branch review Important finding: ``_primary_action_state()``
    never consulted ``_bench_run_running``/``_sample_bench_running``, so
    any rail click while a run is genuinely in flight
    (``EvalsScreen.select()`` always schedules ``refresh(recompose=True)``,
    even for a same-bench reselection) recomposed the inspector into a
    FRESH, ENABLED "Run <name>" button -- a press there would hit
    ``_on_primary_action_pressed``'s own ``_bench_run_running`` guard and
    silently no-op, the exact dead-end anti-pattern
    ``_primary_action_state``'s own docstring forbids. Task 4's persistent
    "Create sample bench" rail button (no longer empty-only, see
    ``library_rail.py``'s module docstring) opened the identical seam for
    itself. Mirrors ``test_rail_run_row_shows_the_running_glyph_while_the_
    run_is_in_flight``'s own mid-flight-recompose technique, just below.
    """
    other_dataset_id = evals_db.create_dataset(
        name="other-dataset", format="custom", source_path="inline:other-dataset"
    )
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

        # No `scroll_visible()` needed -- see the whole-branch review
        # clipping fix's own comment on the earlier press tests above.
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._bench_run_running)

        # Force a mid-flight recompose: select a different row, then
        # reselect the running bench -- exactly the seam the finding
        # describes (a rail click during an in-flight run).
        screen.select(kind="dataset", id=other_dataset_id)
        await pilot.pause()
        screen.select(kind="bench", id=runnable_bench)
        await pilot.pause()

        label, disabled, tooltip = screen._primary_action_state()
        assert disabled is True, (label, disabled, tooltip)
        assert tooltip == "A bench run is already in flight."
        action_button = screen.query_one("#evals-primary-action", Button)
        assert action_button.disabled is True

        sample_bench_button = screen.query_one(
            "#evals-create-sample-bench", Button
        )
        assert sample_bench_button.disabled is True

        release.set()
        await _wait_until(pilot, lambda: not screen._bench_run_running)
        await pilot.pause()


@pytest.mark.asyncio
async def test_rail_run_row_shows_the_running_glyph_while_the_run_is_in_flight(
    evals_app, runnable_bench
):
    """TASK-1480: ``46d56f371``/``da4967a7a`` wired the primary action to a
    real ``WordBenchRunner`` pass, which sets every run in a group to
    "running" in the DB before capturing a single cell (``runner.py``,
    right after ``create_run_group``) -- a group CAN genuinely be
    "running" while the rail composes. This pins that a rail recompose
    landing in that window (mirrors a user clicking a rail row again
    while a run is in flight -- ``EvalsScreen.select()`` always schedules
    ``refresh(recompose=True)``, even for a no-op reselection; the
    progress callback itself only touches the primary-action button, see
    ``_on_bench_run_progress``) renders the run row with the ``●`` glyph,
    and that it flips to ``✓`` once the run completes -- i.e. that the
    view-model's roll-up (TASK-1480) actually reaches the mounted rail
    row, not just ``run_groups()``'s own return value.
    """
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

        # No `scroll_visible()` needed -- see the whole-branch review
        # clipping fix's own comment on the earlier press tests above.
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._bench_run_running)

        # Force a fresh rail recompose while the run is still in flight.
        screen.select(kind="bench", id=runnable_bench)
        await pilot.pause()

        run_row = screen.query_one("#evals-rail-row-runs-0", Button)
        assert str(run_row.label).startswith("● "), str(run_row.label)

        release.set()
        await _wait_until(pilot, lambda: not screen._bench_run_running)
        await pilot.pause()

        run_row = screen.query_one("#evals-rail-row-runs-0", Button)
        assert str(run_row.label).startswith("✓ "), str(run_row.label)


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
async def test_no_rendered_copy_says_library_rail_or_uses_ascii_double_dash(
    evals_app, seeded_bench
):
    """TASK-1481 (live UAT): the rail's actual painted header is "Catalog"
    (see ``LabWorkbench``'s ``label="Catalog"``/``title="Catalog"``), never
    "library rail" -- and the rail's own copy uses real em-dashes, not
    ASCII ``--``. Sweeps every rendered copy string these branches can
    produce, not the source file: this module's docstrings legitimately
    keep both "library rail" (module/class/method docstrings) and " -- "
    (this file's comment convention throughout) -- only user-facing text
    changes.

    Both "none"-selection detail-empty branches are exercised: the
    genuinely-empty-library wording (fresh ``evals_app``, no selection
    yet) and the real-rows-exist wording (after selecting away from
    ``seeded_bench`` back to "none"). The disabled primary action's
    reason string is the third known site.
    """
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen

        empty_library_text = str(
            screen.query_one("#evals-detail-empty").renderable
        )
        assert "library rail" not in empty_library_text
        assert " -- " not in empty_library_text

        _label, _disabled, reason = screen._primary_action_state()
        assert "library rail" not in reason
        assert " -- " not in reason

        screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()
        screen.select(kind="none", id=None)
        await pilot.pause()
        real_rows_text = str(screen.query_one("#evals-detail-empty").renderable)
        assert "library rail" not in real_rows_text
        assert " -- " not in real_rows_text


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


@pytest.mark.asyncio
async def test_primary_action_paints_inside_the_inspector_viewport_without_scrolling(
    evals_app, runnable_bench
):
    """Clipping blocker found during live verification at 235x52 (real
    terminal, not this harness's default). ``EvalsInspector``
    (``#evals-inspector-bench``) is an unstyled ``Vertical``, and
    Textual's ``Vertical``/``Container`` DEFAULT_CSS is ``height: 1fr`` --
    not ``auto``. Inside ``#evals-inspector-pane`` (``_lab.tcss``:
    ``height: auto``, itself inside ``#lab-inspector``, a
    ``VerticalScroll``), an unstyled ``1fr`` child has no real denominator
    during the parent's ``auto`` measurement pass, so ``EvalsInspector``
    claimed nearly the WHOLE ``#lab-inspector`` viewport regardless of its
    own (much shorter) actual content -- pushing its sibling
    ``#evals-primary-action`` button to a row ``#lab-inspector``'s own
    ``virtual_size`` never grew to cover. The button was unreachable even
    by scrolling to ``max_scroll_y`` in the live repro (this harness's own
    single-target repro happened to be reachable at ``max_scroll_y``, but
    the ROOT CAUSE -- and the fix -- are identical either way: see
    ``_evals.tcss``'s ``#evals-inspector-bench { height: auto; }`` comment
    for the full mechanism and why ``#evals-classic-detail`` was verified
    NOT to share it).

    Exactly the off-by-one/starved-sibling family ``_lab.tcss`` already
    fixed one level up, for the PANE itself.
    """
    async with evals_app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=runnable_bench)
        await pilot.pause()

        lab_inspector = screen.query_one("#lab-inspector")
        button = screen.query_one("#evals-primary-action", Button)

        assert lab_inspector.max_scroll_y == 0, (
            "the inspector pane still needs to scroll to fit its content "
            f"(max_scroll_y={lab_inspector.max_scroll_y}) -- EvalsInspector "
            "is claiming more height than its own content needs"
        )
        assert button.region.width > 0 and button.region.height > 0, button.region
        assert lab_inspector.region.contains_region(button.region), (
            f"button {button.region} escapes the inspector's own visible "
            f"viewport {lab_inspector.region}"
        )

        # Belt: a click with NO scroll_visible() call first must reach the
        # real handler -- proves the button is genuinely PAINTED there, not
        # merely LAID OUT there (a region can be geometrically correct on
        # paper while still sitting under another widget's paint layer --
        # see Task 2's own now-removed scroll_visible() workaround, which
        # existed only because of this exact mismatch).
        calls: list = []
        screen._sample_bench_client_factory = lambda t: _FakeCaptureClient(calls)
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._selection.kind == "run_group")
        await pilot.pause()
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_blocked_primary_action_badge_and_callout_paint_inside_the_inspector_viewport(
    evals_app, runnable_bench
):
    """The DISABLED variant of the primary action -- badge + callout +
    button, three widgets instead of one -- must ALSO stay fully inside
    ``#lab-inspector``'s viewport after the ``height: auto`` fix; a fix
    that only accounted for the (shorter) enabled-state content would be
    an incomplete fix. The only selection state that mounts
    ``EvalsInspector`` (i.e. actually exercises ``#evals-inspector-bench``)
    AND renders a disabled button is a bench run genuinely in flight (see
    ``_primary_action_state``'s in-flight branch, whole-branch review
    Finding 1) -- reuses the same pausable-fake-client technique
    ``test_action_buttons_stay_disabled_across_a_mid_run_recompose`` uses.
    """
    async with evals_app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=runnable_bench)
        await pilot.pause()
        release = asyncio.Event()
        calls: list = []
        screen._sample_bench_client_factory = lambda t: _PausableFakeCaptureClient(
            calls, release
        )

        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._bench_run_running)
        # Force a fresh recompose while the run is still in flight, so the
        # rendered badge/callout/button are a genuinely FRESH instance from
        # the fixed CSS, not the live-mutated original widget.
        screen.select(kind="bench", id=runnable_bench)
        await pilot.pause()

        lab_inspector = screen.query_one("#lab-inspector")
        status = screen.query_one("#evals-primary-action-status")
        reason = screen.query_one("#evals-primary-action-reason")
        button = screen.query_one("#evals-primary-action", Button)

        assert lab_inspector.max_scroll_y == 0, lab_inspector.max_scroll_y
        for widget in (status, reason, button):
            assert widget.region.width > 0 and widget.region.height > 0, widget.region
            assert lab_inspector.region.contains_region(widget.region), (
                f"{widget!r} at {widget.region} escapes the inspector's "
                f"visible viewport {lab_inspector.region}"
            )

        release.set()
        await _wait_until(pilot, lambda: not screen._bench_run_running)
        await pilot.pause()


@pytest.mark.asyncio
async def test_pressing_the_primary_action_while_a_sample_bench_run_is_in_flight_is_a_no_op(
    evals_app, runnable_bench
):
    """PR #1113 review (Qodo, seconding whole-branch review Note 6): the
    two workers were only guarded against THEMSELVES --
    ``_on_sample_bench_requested`` checked only ``_sample_bench_running``,
    ``_on_primary_action_pressed`` checked only ``_bench_run_running`` --
    so a press of one while the OTHER was in flight started two genuinely
    overlapping workers (separate ``exclusive`` groups, so neither
    cancelled the other), producing interleaved toasts and a last-wins
    completion ``select()``. The recompose-time UI already disables BOTH
    controls while EITHER flag is set (whole-branch review Finding 1), so
    this posts ``Button.Pressed`` directly against
    ``#evals-primary-action`` -- simulating whatever might reach the
    handler despite that disabled state (a stale render, a race), exactly
    like the existing same-flag "second press" tests already do -- rather
    than relying on the disabled attribute alone to prove the handler
    itself now cross-checks.
    """
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

        await pilot.click("#evals-create-sample-bench")
        await _wait_until(pilot, lambda: screen._sample_bench_running)
        await pilot.pause()
        # WordBenchRunner.run() creates its run group (status "running")
        # BEFORE capturing a single cell -- see runner.py, right after
        # `create_run_group` -- so one already exists here, from the
        # sample-bench worker itself.
        assert len(screen._view_model.run_groups()) == 1

        button = screen.query_one("#evals-primary-action", Button)
        screen.post_message(Button.Pressed(button))
        await pilot.pause()

        # The assertions that discriminate: dropping the cross-guard (while
        # keeping each flag's own-worker guard) would leave both of these
        # true -- a SECOND, overlapping worker genuinely started.
        assert screen._bench_run_running is False
        assert screen._bench_run_task_id is None

        release.set()
        await _wait_until(pilot, lambda: not screen._sample_bench_running)
        await pilot.pause()

        # Only the sample-bench worker's own run group exists -- never a
        # second one from a bench-run worker that should never have
        # started.
        assert len(screen._view_model.run_groups()) == 1


@pytest.mark.asyncio
async def test_sample_bench_request_while_a_bench_run_is_in_flight_is_a_no_op(
    evals_app, runnable_bench
):
    """Mirror of the test above: a bench run started from the primary
    action, then a directly-posted ``SampleBenchRequested`` (mirroring
    ``test_a_second_click_while_running_does_not_start_a_second_run``'s own
    technique in ``test_evals_empty_states.py``) must not start an
    overlapping sample-bench worker."""
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

        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._bench_run_running)
        await pilot.pause()
        assert len(screen._view_model.benches()) == 1

        screen.post_message(LibraryRail.SampleBenchRequested())
        await pilot.pause()

        # Discriminates: dropping the cross-guard would leave this True --
        # a second, overlapping sample-bench worker genuinely started (and
        # would go on to mint its own "loaded-nouns" bench).
        assert screen._sample_bench_running is False

        release.set()
        await _wait_until(pilot, lambda: not screen._bench_run_running)
        await pilot.pause()

        # Only the pre-existing `runnable_bench` -- never a second,
        # sample-bench-minted one.
        assert len(screen._view_model.benches()) == 1
        assert len(screen._view_model.run_groups()) == 1


# ---------------------------------------------------------------------------
# Task 2 (task-1482 prep): a completing worker must not yank a selection the
# user has since moved away from -- see `_selection_unmoved_since_launch`.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bench_run_completion_does_not_yank_a_selection_moved_away_mid_flight(
    evals_app, evals_db, runnable_bench
):
    """Once the bench editor holds unsaved form state, a completing
    background run recomposing the whole screen would destroy it. Presses
    Run on `runnable_bench`, navigates to an unrelated dataset WHILE the
    run is still genuinely in flight (a real async suspension inside
    `_PausableFakeCaptureClient.capture`, not a completed-before-the-test-
    could-look race), then releases -- the completion toast must still
    fire (the run is real, not silently dropped), but `self._selection`
    must remain exactly where the user left it, never yanked to the new
    run group.
    """
    other_dataset_id = evals_db.create_dataset(
        name="other-dataset", format="custom", source_path="inline:other-dataset"
    )
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

        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._bench_run_running)
        await pilot.pause()

        # Navigate away while the run is genuinely in flight.
        screen.select(kind="dataset", id=other_dataset_id)
        await pilot.pause()

        release.set()
        await _wait_until(pilot, lambda: not screen._bench_run_running)
        await pilot.pause()

        assert screen._selection.kind == "dataset"
        assert screen._selection.id == other_dataset_id
        message, severity = evals_app.app_instance.notifications[-1]
        assert severity == "information"
        assert message == "Bench run finished — see the Runs section."


@pytest.mark.asyncio
async def test_bench_run_completion_selects_the_run_group_when_selection_is_unchanged(
    evals_app, runnable_bench
):
    """The paired happy-path case: press Run and do not navigate away
    while it is in flight -- release still moves the selection to the new
    run group, exactly like before Task 2's guard existed."""
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

        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._bench_run_running)
        await pilot.pause()

        release.set()
        await _wait_until(pilot, lambda: not screen._bench_run_running)
        await pilot.pause()

        assert screen._selection.kind == "run_group"
        run_groups = screen._view_model.run_groups()
        assert screen._selection.id == run_groups[0]["id"]
        message, severity = evals_app.app_instance.notifications[-1]
        assert message == "Bench run finished."


@pytest.mark.asyncio
async def test_bench_run_completion_treats_drilling_into_its_own_run_group_as_unmoved(
    evals_app, runnable_bench
):
    """The guard's second branch (`_selection_unmoved_since_launch`):
    navigating INTO the launched bench's own (still-"running") run group
    mid-flight -- e.g. clicking the rail row for the run in progress --
    counts as "still watching this run", not a yank, once the run
    finishes; this is deliberately a DIFFERENT branch than the "selection
    == launch_selection" one above, since the selection here is
    `kind="run_group"`, never `kind="bench"`."""
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

        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._bench_run_running)
        await pilot.pause()

        # WordBenchRunner.run() creates its run group (status "running")
        # before capturing a single cell -- see runner.py, right after
        # `create_run_group` -- so it already exists here.
        running_group = screen._view_model.run_groups()[0]
        screen.select(kind="run_group", id=running_group["id"])
        await pilot.pause()

        release.set()
        await _wait_until(pilot, lambda: not screen._bench_run_running)
        await pilot.pause()

        assert screen._selection.kind == "run_group"
        assert screen._selection.id == running_group["id"]
        message, severity = evals_app.app_instance.notifications[-1]
        assert message == "Bench run finished."


@pytest.mark.asyncio
async def test_sample_bench_completion_does_not_yank_a_selection_moved_away_mid_flight(
    sample_bench_app, evals_db
):
    """Mirrors `test_bench_run_completion_does_not_yank_a_selection_moved_
    away_mid_flight` for the sample-bench worker: a sample bench does not
    exist yet at press time, so there is no pre-existing bench selection
    to pin against -- `_sample_bench_launch_selection` (captured in
    `_on_sample_bench_requested` at press time) stands in for it."""
    other_dataset_id = evals_db.create_dataset(
        name="other-dataset", format="custom", source_path="inline:other-dataset"
    )
    async with sample_bench_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = sample_bench_app.screen
        release = asyncio.Event()
        calls: list = []
        screen._sample_bench_client_factory = lambda t: _PausableFakeCaptureClient(
            calls, release
        )

        await pilot.click("#evals-create-sample-bench")
        await _wait_until(pilot, lambda: screen._sample_bench_running)
        await pilot.pause()

        # Navigate away while the run is genuinely in flight.
        screen.select(kind="dataset", id=other_dataset_id)
        await pilot.pause()

        release.set()
        await _wait_until(pilot, lambda: not screen._sample_bench_running)
        await pilot.pause()

        assert screen._selection.kind == "dataset"
        assert screen._selection.id == other_dataset_id
        message, severity = sample_bench_app.app_instance.notifications[-1]
        assert severity == "information"
        assert message == "Sample bench created and run — see the Runs section."


@pytest.mark.asyncio
async def test_sample_bench_completion_selects_the_run_group_when_selection_is_unchanged(
    sample_bench_app,
):
    """The paired happy-path case for the sample-bench worker: press
    Create sample bench and do not navigate away while it is in flight --
    release still moves the selection to the new run group."""
    async with sample_bench_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = sample_bench_app.screen
        release = asyncio.Event()
        calls: list = []
        screen._sample_bench_client_factory = lambda t: _PausableFakeCaptureClient(
            calls, release
        )

        await pilot.click("#evals-create-sample-bench")
        await _wait_until(pilot, lambda: screen._sample_bench_running)
        await pilot.pause()

        release.set()
        await _wait_until(pilot, lambda: not screen._sample_bench_running)
        await pilot.pause()

        assert screen._selection.kind == "run_group"
        run_groups = screen._view_model.run_groups()
        assert screen._selection.id == run_groups[0]["id"]
        message, severity = sample_bench_app.app_instance.notifications[-1]
        assert message == "Sample bench created and run."


# ---------------------------------------------------------------------------
# task-1610: a run completing must not destroy a DIRTY bench editor, even
# when the selection never moved -- `_selection_unmoved_since_launch`'s own
# selection-identity branches alone would call this case "safe" and
# recompose, discarding unsaved form state. See `BenchEditor.is_dirty()`.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bench_run_completion_does_not_yank_a_dirty_bench_editor(
    evals_app, runnable_bench
):
    """Unlike ``test_bench_run_completion_does_not_yank_a_selection_moved_
    away_mid_flight`` above, the user here never moves the selection at
    all -- they stay parked on ``runnable_bench``'s own editor and type an
    unsaved name edit WHILE the run they started is still in flight.
    Without a dirty check, the guard's "selection == launch_selection"
    branch alone would call this safe and ``select()`` would recompose the
    whole screen, discarding the typed value. This pins the fix: the
    completion degrades to the same toast the moved-away case gets, and
    the typed ``Input`` (the same widget instance) survives untouched --
    proof this is a skipped recompose, not merely a value that happens to
    match after a rebuild.
    """
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

        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._bench_run_running)
        await pilot.pause()

        name_input = screen.query_one("#evals-bench-name", Input)
        name_input.value = "typed-while-running"

        release.set()
        await _wait_until(pilot, lambda: not screen._bench_run_running)
        await pilot.pause()

        assert screen._selection.kind == "bench"
        assert screen._selection.id == runnable_bench
        assert screen.query_one("#evals-bench-name", Input) is name_input
        assert screen.query_one("#evals-bench-name", Input).value == "typed-while-running"
        message, severity = evals_app.app_instance.notifications[-1]
        assert severity == "information"
        assert message == "Bench run finished — see the Runs section."
        # The run itself is real -- the DB write is not lost, only the
        # auto-navigate is skipped.
        assert len(screen._view_model.run_groups()) == 1


@pytest.mark.asyncio
async def test_bench_run_completion_does_not_yank_a_bench_editor_with_unsaved_mini_form_text(
    evals_app, runnable_bench
):
    """task-1611 T2 fix round 1: the SAME protection as
    ``test_bench_run_completion_does_not_yank_a_dirty_bench_editor`` above,
    but the unsaved edit lives in the "+ New target" mini-form
    (``#evals-target-name``) rather than the bench's own Name field.
    ``BenchEditor.is_dirty()`` had to learn about that mini-form for this
    to hold -- before that fix, this exact scenario recomposed right over
    the typed-but-never-created target name."""
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

        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._bench_run_running)
        await pilot.pause()

        name_input = screen.query_one("#evals-target-name", Input)
        name_input.value = "typed-while-running"

        release.set()
        await _wait_until(pilot, lambda: not screen._bench_run_running)
        await pilot.pause()

        assert screen._selection.kind == "bench"
        assert screen._selection.id == runnable_bench
        assert screen.query_one("#evals-target-name", Input) is name_input
        assert screen.query_one("#evals-target-name", Input).value == "typed-while-running"
        message, severity = evals_app.app_instance.notifications[-1]
        assert severity == "information"
        assert message == "Bench run finished — see the Runs section."
        # The run itself is real -- the DB write is not lost, only the
        # auto-navigate is skipped.
        assert len(screen._view_model.run_groups()) == 1


@pytest.mark.asyncio
async def test_sample_bench_completion_does_not_yank_a_dirty_bench_editor(
    sample_bench_app, runnable_bench
):
    """task-1610's sharpest case (per the task description): "Create
    sample bench" is a persistent rail affordance, always available
    regardless of the current selection -- its completion yanks to a run
    group belonging to a bench entirely DIFFERENT from whatever the user
    happens to be editing. Parks on ``runnable_bench``'s own editor (never
    navigating away at all, so the plain selection-identity branch alone
    would call this "unmoved" and safe), types an unsaved edit, then
    presses Create sample bench -- the sample run's completion must not
    yank this dirty, unrelated bench editor either.
    """
    async with sample_bench_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = sample_bench_app.screen
        screen.select(kind="bench", id=runnable_bench)
        await pilot.pause()
        release = asyncio.Event()
        calls: list = []
        screen._sample_bench_client_factory = lambda t: _PausableFakeCaptureClient(
            calls, release
        )

        name_input = screen.query_one("#evals-bench-name", Input)
        name_input.value = "typed-while-sample-running"

        await pilot.click("#evals-create-sample-bench")
        await _wait_until(pilot, lambda: screen._sample_bench_running)
        await pilot.pause()

        release.set()
        await _wait_until(pilot, lambda: not screen._sample_bench_running)
        await pilot.pause()

        assert screen._selection.kind == "bench"
        assert screen._selection.id == runnable_bench
        assert screen.query_one("#evals-bench-name", Input) is name_input
        assert (
            screen.query_one("#evals-bench-name", Input).value
            == "typed-while-sample-running"
        )
        message, severity = sample_bench_app.app_instance.notifications[-1]
        assert severity == "information"
        assert message == "Sample bench created and run — see the Runs section."
        # The sample bench itself is real -- a second, distinct bench now
        # exists alongside `runnable_bench`.
        assert len(screen._view_model.benches()) == 2


@pytest.mark.asyncio
async def test_sample_bench_completion_selects_the_run_group_when_parked_on_a_clean_unrelated_bench_editor(
    sample_bench_app, runnable_bench
):
    """Pins the unchanged-behavior half of task-1610's fix: parked on
    ``runnable_bench``'s own editor with NO edits (clean), the sample
    worker's completion still auto-navigates away to the freshly created
    run group -- exactly the pre-existing "selection unmoved" behavior,
    proving the new dirty check only ever blocks on genuine unsaved state,
    never merely on a `BenchEditor` being mounted at all. Complements
    (does not duplicate) ``test_sample_bench_completion_selects_the_run_
    group_when_selection_is_unchanged`` above, which starts from no
    selection at all rather than a clean, unrelated bench's editor."""
    async with sample_bench_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = sample_bench_app.screen
        screen.select(kind="bench", id=runnable_bench)
        await pilot.pause()
        release = asyncio.Event()
        calls: list = []
        screen._sample_bench_client_factory = lambda t: _PausableFakeCaptureClient(
            calls, release
        )

        await pilot.click("#evals-create-sample-bench")
        await _wait_until(pilot, lambda: screen._sample_bench_running)
        await pilot.pause()

        release.set()
        await _wait_until(pilot, lambda: not screen._sample_bench_running)
        await pilot.pause()

        assert screen._selection.kind == "run_group"
        run_groups = screen._view_model.run_groups()
        assert screen._selection.id == run_groups[0]["id"]
        message, severity = sample_bench_app.app_instance.notifications[-1]
        assert message == "Sample bench created and run."


# ---------------------------------------------------------------------------
# Task 7 (task-1482): Duplicate and Delete.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_and_delete_buttons_render_only_for_a_resolved_bench(
    evals_app, evals_db, seeded_bench
):
    """``#evals-duplicate-bench``/``#evals-delete-bench`` are composed only
    inside the resolved-bench branch of ``_compose_inspector_pane`` --
    absent for every other selection kind (including a ``kind="bench"``
    id that no longer resolves), present and enabled once a real bench is
    selected."""
    dataset_id = evals_db.create_dataset(
        name="other-ds", format="custom", source_path="inline:other-ds"
    )
    classic_id = evals_db.create_task(
        name="classic task",
        task_type="question_answer",
        config_format="custom",
        config_data={},
    )
    base_id = evals_db.create_model(name="rg-base", provider="llama_cpp", model_id="m")
    rg_config = BenchConfig(
        name="rg bench", prompt_mode="raw", top_k=5,
        dataset_id=dataset_id, target_ids=(base_id,),
    )
    rg_task_id = save_bench(evals_db, rg_config)
    target = Target(id=base_id, name="base", provider="llama_cpp", model_id="m")
    group_id, _run_ids = create_run_group(evals_db, rg_task_id, rg_config, [target], [])

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen

        for kind, sel_id in (
            ("none", None),
            ("dataset", dataset_id),
            ("classic", classic_id),
            ("run_group", group_id),
            ("bench", "does-not-exist"),
        ):
            screen.select(kind=kind, id=sel_id)
            await pilot.pause()
            assert not screen.query("#evals-duplicate-bench"), kind
            assert not screen.query("#evals-delete-bench"), kind

        screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()
        duplicate = screen.query_one("#evals-duplicate-bench", Button)
        delete = screen.query_one("#evals-delete-bench", Button)
        assert duplicate.disabled is False
        assert delete.disabled is False
        # The blocked badge/callout only render while disabled -- see the
        # in-flight test below.
        assert not screen.query("#evals-delete-bench-status")
        assert not screen.query("#evals-delete-bench-reason")


@pytest.mark.asyncio
async def test_pressing_duplicate_creates_and_selects_a_copy(evals_app, seeded_bench):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()

        before_ids = {bench["id"] for bench in screen._view_model.benches()}
        await pilot.click("#evals-duplicate-bench")
        await pilot.pause()

        new_ids = {bench["id"] for bench in screen._view_model.benches()} - before_ids
        assert len(new_ids) == 1, new_ids
        new_id = new_ids.pop()
        assert screen._selection.kind == "bench"
        assert screen._selection.id == new_id

        new_bench = screen._view_model.bench_by_id(new_id)
        assert re.fullmatch(r"loaded-nouns v1 copy [0-9a-f]{8}", new_bench["name"]), (
            new_bench["name"]
        )

        notifications = evals_app.app_instance.notifications
        assert notifications
        message, severity = notifications[-1]
        assert severity == "information"
        assert message == f"Duplicated as {new_bench['name']}."


@pytest.mark.asyncio
async def test_duplicate_of_a_corrupt_legacy_bench_toasts_instead_of_crashing(
    evals_app, evals_db
):
    """Controller ruling (Task 3's review): ``_on_duplicate_bench_pressed``
    catches broad ``Exception``, not ``duplicate_bench``'s own narrower
    ``RuntimeError`` (which it raises only for a missing/soft-deleted
    source). A corrupt legacy bench -- ``config_data.target_ids`` carrying
    a malformed (non-string) entry -- makes ``duplicate_bench``'s own
    ``load_bench`` call raise a plain ``ValueError`` instead (see
    ``test_load_bench_rejects_a_malformed_stored_target_id``,
    Tests/Evals/word_bench/test_storage.py), which a narrow
    ``except RuntimeError`` would not catch."""
    dataset_id = evals_db.create_dataset(
        name="corrupt-ds", format="custom", source_path="inline:corrupt-ds"
    )
    target_id = evals_db.create_model(name="t", provider="llama_cpp", model_id="m")
    task_id = evals_db.create_task(
        name="corrupted bench",
        task_type="logprob",
        config_format="custom",
        config_data={
            "bench_type": "word_bench",
            "prompt_mode": "raw",
            "top_k": 20,
            "probes": [],
            "target_ids": [target_id, 123],
            "concurrency": 1,
        },
        dataset_id=dataset_id,
    )

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=task_id)
        await pilot.pause()

        await pilot.click("#evals-duplicate-bench")  # must not crash the app
        await pilot.pause()

        assert pilot.app.is_running, "the app must survive the failure toast"
        assert screen._selection.id == task_id  # unchanged -- duplication failed
        notifications = evals_app.app_instance.notifications
        assert notifications
        message, severity = notifications[-1]
        assert severity == "error"
        assert "Could not duplicate the bench" in message


@pytest.mark.asyncio
async def test_delete_confirmed_removes_from_rail_selection_none_runs_remain(
    evals_app, evals_db
):
    """Deleting a bench soft-deletes its ``eval_tasks`` row -- it
    disappears from the rail's Benches section and ``screen._selection``
    moves to ``kind="none"`` -- but its run history is NOT cascaded: the
    Runs section keeps listing the run group, and reselecting it still
    renders the grid (see ``_apply_bench_deletion``'s own comment:
    ``list_runs``/``get_run``'s ``JOIN eval_tasks`` is unfiltered on
    ``t.deleted_at``).

    Uses ``_apply_bench_deletion`` directly (``confirmed=True``), bypassing
    the modal per this task's own public-shaped-callback convention (see
    ``snippet_editor.py``'s ``_handle_import_file_selected``) -- the
    dialog's own message content is pinned separately, below."""
    base_id = evals_db.create_model(name="base", provider="llama_cpp", model_id="m")
    dataset_id = evals_db.create_dataset(
        name="del-ds", format="custom", source_path="inline:del-ds"
    )
    config = BenchConfig(
        name="to delete", prompt_mode="raw", top_k=5,
        dataset_id=dataset_id, target_ids=(base_id,),
    )
    task_id = save_bench(evals_db, config)
    target = Target(id=base_id, name="base", provider="llama_cpp", model_id="m")
    group_id, _run_ids = create_run_group(evals_db, task_id, config, [target], [])

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=task_id)
        await pilot.pause()

        screen._apply_bench_deletion(True, task_id)
        await pilot.pause()

        assert screen._selection.kind == "none"
        assert screen._view_model.bench_by_id(task_id) is None
        assert not screen.query("#evals-rail-row-benches-0")

        notifications = evals_app.app_instance.notifications
        assert notifications
        message, severity = notifications[-1]
        assert severity == "information"
        assert message == "Bench deleted. Its runs remain in the Runs section."

        run_groups = screen._view_model.run_groups()
        assert len(run_groups) == 1
        assert run_groups[0]["id"] == group_id

        screen.select(kind="run_group", id=group_id)
        await pilot.pause()
        assert screen.query_one("#evals-results-grid")
        assert not screen.query("#evals-detail-missing")


@pytest.mark.asyncio
async def test_delete_cancelled_is_a_no_op(evals_app, seeded_bench):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()

        screen._apply_bench_deletion(False, seeded_bench)
        await pilot.pause()

        assert screen._selection.kind == "bench"
        assert screen._selection.id == seeded_bench
        assert screen._view_model.bench_by_id(seeded_bench) is not None
        assert evals_app.app_instance.notifications == []


@pytest.mark.asyncio
async def test_delete_confirm_dialog_message_contains_the_escaped_bench_name(
    evals_app, evals_db
):
    """Drives the real button + real ``push_screen_wait`` (unlike the
    confirmed/cancelled tests above, which bypass the modal via
    ``_apply_bench_deletion`` directly) -- this is the one test proving
    the pushed ``ConfirmationDialog`` itself carries the escaped name, per
    the Watchlists convention (``escape_markup(name)`` in ``message``,
    watchlists_collections_screen.py:2117-2135). The name carries a bare
    ``[/]`` -- unbalanced Rich/Textual markup -- so an unescaped ``message``
    would raise inside the dialog's own ``Label`` render.

    A real target (``target_ids=[target_id]``), not the empty tuple this
    test used before task-1482 Task 7 fix round 1's reorder: with no
    targets, ``_primary_action_state`` returns its "blocked" branch, which
    renders TWO extra ``Static`` rows (status + reason) ahead of
    ``#evals-primary-action`` -- and, since Duplicate/Delete now compose
    AFTER that button (the fix round's own reorder), those two rows push
    Delete's painted position past ``#lab-inspector``'s fold at this
    harness's cramped default (80x24) test size, so a bare
    ``pilot.click(\"#evals-delete-bench\")`` lands on nothing. Irrelevant
    to what this test actually verifies (the confirm dialog's escaped
    message) -- a real target keeps the primary action in its one-widget
    enabled form instead, matching every other click-driven delete/
    duplicate test in this file (``seeded_bench``/``runnable_bench``, both
    real-targeted)."""
    dataset_id = evals_db.create_dataset(
        name="markup-ds", format="custom", source_path="inline:markup-ds"
    )
    target_id = evals_db.create_model(name="t", provider="llama_cpp", model_id="m")
    task_id = evals_db.create_task(
        name="notes[/].txt bench",
        task_type="logprob",
        config_format="custom",
        config_data={
            "bench_type": "word_bench", "prompt_mode": "raw", "top_k": 5,
            "probes": [], "target_ids": [target_id], "concurrency": 1,
        },
        dataset_id=dataset_id,
    )

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=task_id)
        await pilot.pause()

        stack_depth_before = len(evals_app.screen_stack)
        await pilot.click("#evals-delete-bench")
        await pilot.pause()

        assert len(evals_app.screen_stack) == stack_depth_before + 1
        dialog = evals_app.screen
        assert isinstance(dialog, ConfirmationDialog)
        assert dialog.title == "Delete bench?"
        assert escape_markup("notes[/].txt bench") in dialog.message
        assert "notes[/].txt bench" not in dialog.message

        # Cancel (the dialog's own primary button, per its DEFAULT_CSS/
        # variant) rather than deleting -- confirms the bench survives a
        # dismissed dialog reached through the real UI path, not just
        # through `_apply_bench_deletion(False, ...)` directly.
        await pilot.click("#cancel-button")
        await _wait_until(pilot, lambda: len(evals_app.screen_stack) == stack_depth_before)
        await pilot.pause()

        assert screen._selection.kind == "bench"
        assert screen._view_model.bench_by_id(task_id) is not None


@pytest.mark.asyncio
async def test_pressing_delete_then_confirming_in_the_real_dialog_deletes_the_bench(
    evals_app, seeded_bench
):
    """End-to-end wiring check: ``_on_delete_bench_pressed`` ->
    ``_delete_bench_flow`` -> the real ``ConfirmationDialog`` -> Confirm ->
    ``_apply_bench_deletion``. The confirmed/cancelled BEHAVIOR itself is
    pinned against ``_apply_bench_deletion`` directly elsewhere in this
    file (bypassing the modal, per this task's own convention) -- this
    proves the pieces are actually wired together end to end."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()

        await pilot.click("#evals-delete-bench")
        await pilot.pause()
        assert isinstance(evals_app.screen, ConfirmationDialog)

        await pilot.click("#confirm-button")
        await _wait_until(pilot, lambda: screen._selection.kind == "none")
        await pilot.pause()

        assert screen._view_model.bench_by_id(seeded_bench) is None
        message, severity = evals_app.app_instance.notifications[-1]
        assert message == "Bench deleted. Its runs remain in the Runs section."
        assert severity == "information"


@pytest.mark.asyncio
async def test_delete_is_disabled_with_a_reason_while_this_benchs_run_is_in_flight(
    evals_app, runnable_bench
):
    """Mirrors ``test_action_buttons_stay_disabled_across_a_mid_run_
    recompose``'s technique: starts a real (paused) bench run, forces a
    recompose while it is still in flight, then checks Delete -- not the
    primary action -- picks up the SAME in-flight state. Duplicate must
    stay enabled throughout: nothing about duplicating this bench's
    CONFIG conflicts with a run reading its already-loaded snapshot."""
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

        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._bench_run_running)
        # Force a recompose while the run is still in flight -- mirrors
        # `test_action_buttons_stay_disabled_across_a_mid_run_recompose`.
        screen.select(kind="bench", id=runnable_bench)
        await pilot.pause()

        delete_button = screen.query_one("#evals-delete-bench", Button)
        assert delete_button.disabled is True
        assert delete_button.tooltip == "A run of this bench is in flight."
        status = screen.query_one("#evals-delete-bench-status")
        assert str(status.renderable) == "Delete: Blocked"
        reason = screen.query_one("#evals-delete-bench-reason")
        assert "A run of this bench is in flight." in str(reason.renderable)

        duplicate_button = screen.query_one("#evals-duplicate-bench", Button)
        assert duplicate_button.disabled is False

        # A press that somehow reaches the handler anyway (e.g. a stale,
        # not-yet-rerendered button) must still be a genuine no-op, not
        # just visually disabled -- mirrors
        # `test_a_second_press_while_a_bench_run_is_in_flight_is_a_no_op`.
        screen.post_message(Button.Pressed(delete_button))
        await pilot.pause()
        assert screen._selection.kind == "bench"
        assert screen._selection.id == runnable_bench
        assert screen._view_model.bench_by_id(runnable_bench) is not None

        release.set()
        await _wait_until(pilot, lambda: not screen._bench_run_running)
        await pilot.pause()


@pytest.mark.asyncio
async def test_delete_stays_enabled_while_an_unrelated_sample_bench_run_is_in_flight(
    evals_app, seeded_bench
):
    """The sample-bench worker pins its own not-yet-existing bench, never
    an already-selected one -- unlike the primary action
    (``_primary_action_state``), Delete must not treat a running SAMPLE
    bench as a reason to block deleting some other, unrelated, already-
    selected bench (task-7 brief: "decide whether sample runs block
    Delete of an UNRELATED bench")."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()

        screen._sample_bench_running = True
        screen.refresh(recompose=True)
        await pilot.pause()

        delete_button = screen.query_one("#evals-delete-bench", Button)
        assert delete_button.disabled is False
        assert not screen.query("#evals-delete-bench-status")


# ---------------------------------------------------------------------------
# Task 7 fix round 1 (task-1482): reviewer-found reentrancy + spec ordering.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_queued_delete_presses_push_exactly_one_confirmation_dialog(
    evals_app, seeded_bench
):
    """Reviewer-reproduced race: ``_on_delete_bench_pressed`` is a plain
    (non-async) handler that only calls ``run_worker(self._delete_bench_
    flow(...), group="evals-delete-bench")`` -- deliberately NOT
    ``exclusive=True`` (see that handler's own docstring for why
    ``exclusive=True`` is wrong here: ``push_screen_wait`` awaits
    ``asyncio.shield(future)``, which shields the WAIT from cancellation,
    not the widget it already pushed -- cancelling a superseded worker via
    an exclusive group would still orphan its already-mounted
    ``ConfirmationDialog`` on the screen stack, whose Confirm/Cancel click
    would then silently do nothing).

    Posting two ``Button.Pressed`` messages back to back, with no
    intervening ``await``, queues both before either handler invocation
    can run its worker's first line -- exactly the shape a rapid real
    double-click (or two events already queued in the message pump)
    produces. Without a synchronous pending-flag guard, each handler
    invocation independently passes the (unrelated) in-flight-run disabled
    check and calls ``run_worker`` a second time, mounting a SECOND
    ``ConfirmationDialog`` on top of the first."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()

        delete_button = screen.query_one("#evals-delete-bench", Button)
        stack_depth_before = len(evals_app.screen_stack)

        screen.post_message(Button.Pressed(delete_button))
        screen.post_message(Button.Pressed(delete_button))
        await pilot.pause()
        await pilot.pause()

        dialogs = [s for s in evals_app.screen_stack if isinstance(s, ConfirmationDialog)]
        assert len(dialogs) == 1, (
            f"expected exactly one ConfirmationDialog, found "
            f"{len(dialogs)} (screen stack depth "
            f"{stack_depth_before} -> {len(evals_app.screen_stack)})"
        )


@pytest.mark.asyncio
async def test_inspector_pane_buttons_compose_in_the_spec_order(
    evals_app, seeded_bench
):
    """Design-spec ordering (inspector mock): ``[ Run bench ]`` then
    ``[ Duplicate ]`` then ``[ Delete ]`` -- asserted as a real DOM-order
    check (not three separate presence checks, which pass regardless of
    order) so a future edit that reintroduces Duplicate/Delete BEFORE the
    primary action fails here."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()

        inspector_pane = screen.query_one("#evals-inspector-pane")
        button_ids = [
            button.id for button in inspector_pane.query(Button) if button.id
        ]
        assert button_ids == [
            "evals-primary-action",
            "evals-duplicate-bench",
            "evals-delete-bench",
        ], button_ids


@pytest.mark.asyncio
async def test_duplicate_and_delete_buttons_paint_full_width_like_the_primary_action(
    evals_app, seeded_bench
):
    """``#evals-duplicate-bench``/``#evals-delete-bench`` had no width rule
    of their own and rendered auto-width (Textual's ``Button`` DEFAULT_CSS
    floors every button at ``min-width: 16``, far short of a real pane at
    a realistic terminal size) -- inconsistent with ``#evals-primary-
    action`` directly above, and this pane has a documented history of
    geometry defects (see ``_evals.tcss``'s own ``#evals-inspector-bench``
    comment). A painted-geometry check, at the same 235x52 realistic size
    the sibling primary-action geometry tests in this file use, not a bare
    ``width == 100%`` CSS-source grep -- proves the fix actually PAINTS
    wide, not just declares a rule some later cascade entry could still
    lose to."""
    async with evals_app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()

        inspector_pane = screen.query_one("#evals-inspector-pane")
        half_width = inspector_pane.region.width / 2
        duplicate = screen.query_one("#evals-duplicate-bench", Button)
        delete = screen.query_one("#evals-delete-bench", Button)
        for button in (duplicate, delete):
            assert button.region.width > half_width, (
                f"{button.id} region.width={button.region.width} is not "
                f"wider than half the inspector pane's width "
                f"({half_width}) -- {button.region}"
            )


# ---------------------------------------------------------------------------
# task-1691 Task 3 (review Minor, T2 reviewer): the readiness pane's
# documented collapse/clipping risk class, now WITH multiple continuation
# sub-lines present at once.
# ---------------------------------------------------------------------------


@pytest.fixture
def bench_with_multiple_continuations(evals_db: EvalsDB) -> str:
    """Three local targets covering task-1691's own mix, chosen to stress
    this pane's documented collapse/clipping history (see ``_evals.tcss``'s
    ``#evals-inspector-bench`` comment and the sibling 235x52 geometry
    tests above) with SEVERAL continuation sub-lines stacked at once,
    something none of task-1691's own Task 1/2 tests exercised together:

    - a warned target: the TALLEST possible per-target shape -- badge +
      continuation sub-line + a degenerate-canary recovery callout, three
      stacked widgets for one row.
    - an unreachable target: Blocked, with NO continuation at all --
      ``PreflightResult.continuation`` is only ever resolved for a clean
      canary capture (see ``capture_client.preflight``'s own docstring), so
      a Blocked/Unavailable row never gets a continuation sub-line no
      matter what this fixture sets.
    - a clean Ready target whose continuation is longer than the
      inspector's own preview cap (``inspector._CONTINUATION_PREVIEW_MAX_
      LEN``, 100) and gets truncated.
    """
    warned_id = evals_db.create_model(
        name="warned-target", provider="llama_cpp", model_id="m"
    )
    unreachable_id = evals_db.create_model(
        name="unreachable-target", provider="llama_cpp", model_id="m"
    )
    long_id = evals_db.create_model(
        name="long-target", provider="llama_cpp", model_id="m"
    )
    dataset_id = evals_db.create_dataset(
        name="loaded-nouns", format="custom", source_path="inline:loaded-nouns",
        metadata={"sample_count": 6},
    )
    config = BenchConfig(
        name="multi-continuation v1",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(warned_id, unreachable_id, long_id),
        probes=(" Sure", " I"),
    )
    task_id = save_bench(evals_db, config)
    targets = [
        Target(id=warned_id, name="warned-target", provider="llama_cpp", model_id="m"),
        Target(id=unreachable_id, name="unreachable-target", provider="llama_cpp", model_id="m"),
        Target(id=long_id, name="long-target", provider="llama_cpp", model_id="m"),
    ]
    snippets = [Snippet(id="s1", text="The protestors were", group="neutral")]
    preflight = {
        warned_id: PreflightResult(
            state="ok", k_returned=20, canary="degenerate",
            continuation="<|channel><|channel>thought\n<channel|>The sky is **blue",
        ),
        unreachable_id: PreflightResult(
            state="unreachable", k_returned=None, canary="unchecked",
            detail="connection refused",
        ),
        long_id: PreflightResult(
            state="ok", k_returned=20, canary="pass",
            continuation="x" * 150,
        ),
    }
    create_run_group(evals_db, task_id, config, targets, snippets, preflight=preflight)
    return task_id


@pytest.mark.asyncio
async def test_readiness_rows_with_several_continuations_paint_inside_the_inspector_viewport(
    evals_app, bench_with_multiple_continuations
):
    """A captured continuation (task-1691) adds a NEW sub-line under every
    target row that has one -- this pane's own documented collapse/
    clipping risk class (see the sibling 235x52 geometry tests above),
    never exercised before with MULTIPLE continuation sub-lines present at
    once, including the tallest possible per-target shape (a warned badge
    + a continuation sub-line + a recovery callout, three widgets stacked
    for one target). Asserts every readiness row genuinely paints, and
    that ``#evals-primary-action`` and the Estimate section stay
    reachable -- scrolled into view where needed, mirroring
    ``test_primary_action_paints_inside_the_inspector_viewport_without_
    scrolling``'s own convention at this file's other 235x52 tests, rather
    than assuming a single screenful covers three stacked target rows
    (unlike those single-target fixtures, this one deliberately does not
    assert ``max_scroll_y == 0``)."""
    async with evals_app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen: EvalsScreen = evals_app.screen
        screen.select(kind="bench", id=bench_with_multiple_continuations)
        await pilot.pause()

        lab_inspector = screen.query_one("#lab-inspector")

        # Every target's own badge paints.
        for index in range(3):
            badge = screen.query_one(f"#evals-inspector-target-{index}")
            assert badge.region.width > 0 and badge.region.height > 0, badge.region

        # Target 0 (warned): continuation sub-line AND recovery callout
        # both paint -- the tallest stacked per-target shape.
        warned_continuation = screen.query_one(
            "#evals-inspector-target-continuation-0"
        )
        assert warned_continuation.region.width > 0
        assert warned_continuation.region.height > 0
        warned_callout = screen.query_one("#evals-inspector-target-callout-0")
        assert warned_callout.region.width > 0
        assert warned_callout.region.height > 0

        # Target 1 (unreachable): no continuation sub-line at all.
        assert not screen.query("#evals-inspector-target-continuation-1")
        unreachable_callout = screen.query_one(
            "#evals-inspector-target-callout-1"
        )
        assert unreachable_callout.region.width > 0
        assert unreachable_callout.region.height > 0

        # Target 2 (long continuation): paints, truncated with an
        # ellipsis.
        long_continuation = screen.query_one(
            "#evals-inspector-target-continuation-2"
        )
        assert long_continuation.region.width > 0
        assert long_continuation.region.height > 0
        assert long_continuation.visual.plain.endswith("…")

        # `#evals-primary-action` and the Estimate section stay reachable
        # -- scrolled into view, since three stacked target rows
        # (including one three widgets tall) plausibly do not fit in one
        # screenful; each widget's own region is genuinely non-zero once
        # scrolled to, i.e. actually painted, not merely present in the
        # DOM.
        button = screen.query_one("#evals-primary-action", Button)
        button.scroll_visible(animate=False)
        await pilot.pause()
        assert button.region.width > 0 and button.region.height > 0, button.region
        assert lab_inspector.region.contains_region(button.region), (
            f"button {button.region} escapes the inspector's own visible "
            f"viewport {lab_inspector.region}"
        )

        estimate_calls = screen.query_one("#evals-inspector-estimate-calls")
        estimate_calls.scroll_visible(animate=False)
        await pilot.pause()
        assert estimate_calls.region.width > 0 and estimate_calls.region.height > 0
        assert lab_inspector.region.contains_region(estimate_calls.region), (
            f"estimate {estimate_calls.region} escapes the inspector's "
            f"own visible viewport {lab_inspector.region}"
        )
