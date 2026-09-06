"""TASK-23025: resize/focus cheap-state gates and per-visit deferrals.

The Library screen's resize handler ran its three query-heavy layout legs on
EVERY resize frame (measured 71.6 DOM queries/frame on the landing route);
the width-crossing early return sat AFTER the query work. It now decides
from cheap state -- cached widget references and width-bucket arithmetic --
whether a frame can change anything, before any query work. Same treatment
for ``research_workspace_screen.py``'s ``_apply_pane_layout``, which was
completely ungated. The focus path (25.4 queries per Tab) is bounded by
route-gating the scroll-observer probes and resolving invariant chrome
through the ref cache. The model-install progress pair -- invisible on the
default route -- now grows on demand (the TASK-23024 slot-pool pattern);
the rail Details body deferral was considered and reverted because its
children are a queried-while-closed contract (see the test pinning it).

Query-count assertions attribute each ``query``/``query_one`` call to the
nearest non-framework repo frame, exactly like the measurement probe that
produced the before/after numbers, so a reverted gate fails them by an
order of magnitude rather than by noise.

TASK-23151 extended the same contract to the stage-visibility leg, which the
narrow-emergency work had re-added ABOVE the crossing return: it is now gated
on its own cheap signature rather than moved, because the 64-cell emergency
band is a different band from the 120-cell compact one and only the leg above
the return can cross it. The crossing tests here pin the COST -- exactly one
application per crossing -- so neither wrong shape (unconditional, or
relocated below the return) can pass. The gate's own branches (skip on an
unchanged signature, apply on a changed one, fail open on an unavailable one,
plus the three inputs the signature shapes deliberately) are pinned by focused
unit tests against a fake screen, below the harness tests.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.dom import DOMNode
from textual.widgets import Button, Static

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_INGEST_MEDIA,
)
from tldw_chatbook.UI.Screens.library_screen import (
    LIBRARY_NOTES_SOURCE_DATABASE,
    LibraryScreen,
)
from tldw_chatbook.Widgets.Library.library_rail import LibraryRail
from tldw_chatbook.Widgets.ModelArtifacts.install_progress import (
    InstallProgressed,
    ModelInstallProgress,
)

_SITE_PACKAGES = "/site-packages/"


def _nearest_repo_frame() -> str:
    """Return the filename of the nearest tldw_chatbook frame, mirroring the
    measurement probe's attribution."""
    frame = sys._getframe(2)
    for _ in range(40):
        if frame is None:
            return ""
        filename = frame.f_code.co_filename
        if "tldw_chatbook/" in filename and _SITE_PACKAGES not in filename:
            return filename
        frame = frame.f_back
    return ""


@contextmanager
def _counting_queries(module_fragment: str, counts: dict[str, int]):
    """Count DOM queries whose nearest repo frame is in ``module_fragment``."""
    real_query = DOMNode.query
    real_query_one = DOMNode.query_one

    def counting_query(self, *args, **kwargs):
        if counts.get("enabled") and module_fragment in _nearest_repo_frame():
            counts["n"] += 1
        return real_query(self, *args, **kwargs)

    def counting_query_one(self, *args, **kwargs):
        if counts.get("enabled") and module_fragment in _nearest_repo_frame():
            counts["n"] += 1
        return real_query_one(self, *args, **kwargs)

    DOMNode.query = counting_query
    DOMNode.query_one = counting_query_one
    try:
        yield counts
    finally:
        DOMNode.query = real_query
        DOMNode.query_one = real_query_one


async def _settled_library(host, pilot):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await pilot.pause()
    return screen


@pytest.mark.asyncio
async def test_resize_gate_skips_library_query_work_on_non_crossing_frames():
    """Steady non-crossing resize frames issue ZERO library_screen queries.

    Born-red against the pre-gate implementation: the same three frames
    issued ~60 library-attributed queries each (the ingest/contract/stage
    legs ran unconditionally, the crossing check sat after them).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _settled_library(host, pilot)
        counts = {"n": 0, "enabled": False}
        with _counting_queries("UI/Screens/library_screen.py", counts):
            # Warm-up frame: seeds the ref cache and the applied signature.
            await pilot.resize_terminal(169, 48)
            await pilot.pause()
            await pilot.pause()
            assert screen._library_resize_applied_signature is not None
            counts["enabled"] = True
            for width in (168, 167, 166):
                await pilot.resize_terminal(width, 48)
                await pilot.pause()
                await pilot.pause()
            counts["enabled"] = False
        assert counts["n"] == 0, (
            f"{counts['n']} library_screen-attributed DOM queries across 3 "
            "non-crossing resize frames; the cheap-state gate should have "
            "returned before any query work"
        )


@pytest.mark.asyncio
async def test_resize_compact_crossing_still_transitions_both_ways_twice():
    """The 120-column compact crossing keeps working through the gate.

    TASK-23151 additionally pins the COST of each crossing: exactly one
    stage-visibility application per crossing.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _settled_library(host, pilot)
        assert screen._library_notes_compact is False
        applied = Mock(wraps=screen._apply_library_notes_stage_visibility)
        screen._apply_library_notes_stage_visibility = applied

        for _ in range(2):  # twice: a stale applied-signature would eat round 2
            applied.reset_mock()
            await pilot.resize_terminal(110, 48)
            await _wait_for_condition(
                pilot,
                lambda: screen._library_notes_compact,
                message="narrow resize never crossed into compact",
            )
            await pilot.pause()
            grid = screen.query_one("#library-shell-grid")
            assert grid.has_class("library-notes-compact")
            assert applied.call_count == 1, (
                f"{applied.call_count} stage-visibility applications for one "
                "crossing into compact"
            )

            applied.reset_mock()
            await pilot.resize_terminal(170, 48)
            await _wait_for_condition(
                pilot,
                lambda: not screen._library_notes_compact,
                message="wide resize never crossed back out of compact",
            )
            await pilot.pause()
            grid = screen.query_one("#library-shell-grid")
            assert not grid.has_class("library-notes-compact")
            assert applied.call_count == 1, (
                f"{applied.call_count} stage-visibility applications for one "
                "crossing out of compact"
            )


@pytest.mark.asyncio
async def test_resize_emergency_crossing_still_engages_and_restores():
    """The <64-column emergency takeover engages and restores through the gate."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _settled_library(host, pilot)
        assert screen._library_emergency_stage is None

        await pilot.resize_terminal(60, 48)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_emergency_stage is not None,
            message="60-column resize never engaged the emergency stage",
        )
        rail = screen.query_one("#library-rail")
        canvas = screen.query_one("#library-canvas")
        assert rail.display != canvas.display, (
            "emergency stage should show exactly one of rail/canvas"
        )

        await pilot.resize_terminal(170, 48)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_emergency_stage is None,
            message="wide resize never restored the ordinary stage",
        )
        assert screen.query_one("#library-rail").display
        assert screen.query_one("#library-canvas").display


@pytest.mark.asyncio
async def test_stage_visibility_runs_once_per_emergency_band_crossing():
    """TASK-23151: the stage leg is gated, not moved, above the crossing return.

    The narrow emergency band (64 cells) is a DIFFERENT band from the compact
    breakpoint (120), so an 80 <-> 63 resize crosses only the former. The
    stage-visibility leg therefore has to sit above ``on_resize``'s
    compact-crossing early return -- which is why TASK-23151's regression put
    unconditional per-frame Notes work back on every same-side frame.

    This pins both halves at once, and fails under either wrong shape: a
    same-side frame that does stage work (the regression) fails the ``== 0``
    legs, and a fix that simply deletes or relocates the call below the
    crossing return fails the ``== 1`` legs with a stranded emergency stage.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=(110, 30)) as pilot:
        screen = await _settled_library(host, pilot)
        assert screen._library_emergency_stage is None
        assert screen._library_notes_compact is True

        applied = Mock(wraps=screen._apply_library_notes_stage_visibility)
        screen._apply_library_notes_stage_visibility = applied

        # These frames DO change the wider resize signature -- they cross the
        # 100-cell Ingest auto-collapse bucket, so ``on_resize``'s outer gate
        # cannot skip them and the legs really run -- but they cross neither
        # band the stage leg reads. This is the shape the regression turned
        # into per-frame Notes work.
        for width in (105, 98, 96, 102):
            await pilot.resize_terminal(width, 30)
            await pilot.pause()
            await pilot.pause()
        assert applied.call_count == 0, (
            f"{applied.call_count} stage-visibility applications across 4 "
            "resize frames that cross no band the stage leg reads"
        )
        assert screen._library_emergency_stage is None

        # Crossing INTO the emergency band. No compact crossing happens here
        # (102 and 63 are both below 120), so this gate is the only thing
        # that can engage the takeover.
        applied.reset_mock()
        await pilot.resize_terminal(63, 30)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_emergency_stage is not None,
            message="63-column resize never engaged the emergency stage",
        )
        assert applied.call_count == 1, (
            f"{applied.call_count} stage-visibility applications for one "
            "emergency-band crossing"
        )
        rail = screen.query_one("#library-rail")
        canvas = screen.query_one("#library-canvas")
        assert rail.display != canvas.display

        # Crossing back OUT restores the ordinary two-pane stage, again with
        # no compact crossing to fall back on.
        applied.reset_mock()
        await pilot.resize_terminal(64, 30)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_emergency_stage is None,
            message="64-column resize never released the emergency stage",
        )
        assert applied.call_count == 1, (
            f"{applied.call_count} stage-visibility applications for one "
            "emergency-band release"
        )
        assert screen.query_one("#library-rail").display
        assert screen.query_one("#library-canvas").display

        # And the gate re-arms: further same-side frames are free again.
        applied.reset_mock()
        for width in (105, 98, 96, 102):
            await pilot.resize_terminal(width, 30)
            await pilot.pause()
            await pilot.pause()
        assert applied.call_count == 0, (
            f"{applied.call_count} stage-visibility applications after the "
            "crossing settled; the gate did not re-arm"
        )


class _FakeStageWidget:
    """Attribute-only stand-in for a widget the stage signature reads.

    The signature carries widget references by identity and reads only
    ``region.width`` and ``display`` off them, so nothing here has to be a
    real Textual widget.
    """

    def __init__(
        self,
        *,
        widget_id: str | None = None,
        width: int = 0,
        display: bool = True,
    ) -> None:
        self.id = widget_id
        self.region = SimpleNamespace(width=width)
        self.display = display


class _StageGateScreen:
    """Fake screen carrying exactly the state the stage-visibility gate reads.

    TASK-23151 unit coverage: the gate, the signature it compares and the seam
    that records it are taken UNBOUND from ``LibraryScreen``, so these tests
    exercise the shipped functions -- not a re-implementation -- without a
    Textual harness. Only the leg itself is replaced, by a counter, because
    "did the leg run" is the whole contract under test. The cheap route/flag
    predicates the signature calls into are real too; a stub of
    ``_library_ordinary_route_active`` would hide the effective-emergency
    subtlety these tests exist to pin.
    """

    _library_notes_stage_signature = LibraryScreen._library_notes_stage_signature
    _apply_library_notes_stage_visibility_for_resize = (
        LibraryScreen._apply_library_notes_stage_visibility_for_resize
    )
    _apply_library_notes_stage_visibility = (
        LibraryScreen._apply_library_notes_stage_visibility
    )
    _library_ordinary_route_active = LibraryScreen._library_ordinary_route_active
    _library_notes_compact_stage_applies = (
        LibraryScreen._library_notes_compact_stage_applies
    )
    _library_notes_compact_workflow_active = (
        LibraryScreen._library_notes_compact_workflow_active
    )
    _library_notes_workflow_active = LibraryScreen._library_notes_workflow_active
    _file_notes_active = LibraryScreen._file_notes_active
    _library_notes_focused_task_active = (
        LibraryScreen._library_notes_focused_task_active
    )

    def __init__(
        self,
        *,
        width: int = 100,
        row_id: str = LIBRARY_ROW_INGEST_MEDIA,
        reader_active: bool = False,
    ) -> None:
        self.is_mounted = True
        self.size = SimpleNamespace(width=width)
        self.shell = _FakeStageWidget(widget_id="library-shell-grid", width=width)
        self.rail = _FakeStageWidget(widget_id="library-rail")
        self.canvas = _FakeStageWidget(widget_id="library-canvas")
        self.refs: dict[str, _FakeStageWidget | None] = {
            "#library-shell-grid": self.shell,
            "#library-rail": self.rail,
            "#library-canvas": self.canvas,
        }
        self.reader_active = reader_active
        self.leg_calls = 0
        self.leg_failure: Exception | None = None
        # Cheap state the signature reads directly.
        self._library_compose_generation = 0
        self._library_reader_shell_ref = (
            _FakeStageWidget(widget_id="library-notes-reader-shell")
            if reader_active
            else None
        )
        self._library_selected_row_id = row_id
        self._library_notes_source = LIBRARY_NOTES_SOURCE_DATABASE
        self._library_notes_view = "list"
        self._library_notes_stage = "rail"
        self._library_notes_compact = False
        self._library_rail_collapsed = False
        self._library_emergency_stage: str | None = None
        self._library_emergency_restore_receipt = None
        self._library_reader_shared_preferences = SimpleNamespace(
            library_open=True, custom_widths_enabled=False, library_width=34
        )
        self._library_notes_stage_applied_signature: tuple | None = None

    def set_width(self, width: int) -> None:
        """Resize both the viewport and the shell grid, as a real frame does."""
        self.size = SimpleNamespace(width=width)
        self.shell.region = SimpleNamespace(width=width)

    def _library_layout_ref(self, selector: str) -> _FakeStageWidget | None:
        return self.refs.get(selector)

    def _library_adaptive_reader_shell_active(self) -> bool:
        return self.reader_active

    def _apply_library_notes_stage_legs(self) -> None:
        """Stand in for the query-heavy leg: count it, optionally fail it."""
        self.leg_calls += 1
        if self.leg_failure is not None:
            raise self.leg_failure


def test_stage_gate_skips_resize_frames_that_cross_no_band():
    """TASK-23151 unit: an unchanged signature skips the leg entirely.

    The band-free same-side resize is the exact shape that measured 201 and
    100 applications against a ratchet demanding 0.
    """
    screen = _StageGateScreen(width=100)
    screen._apply_library_notes_stage_visibility_for_resize()
    assert screen.leg_calls == 1, "the cold gate must apply once to arm itself"

    for width in (96, 92, 88):
        screen.set_width(width)
        screen._apply_library_notes_stage_visibility_for_resize()
    assert screen.leg_calls == 1, (
        f"{screen.leg_calls} stage-visibility applications across 3 resize "
        "frames that cross no band the stage leg reads; the gate should have "
        "skipped all three"
    )


def test_stage_gate_applies_exactly_once_per_emergency_band_crossing():
    """TASK-23151 unit: a changed signature applies the leg, once."""
    screen = _StageGateScreen(width=80)
    screen._apply_library_notes_stage_visibility_for_resize()  # arm
    screen.leg_calls = 0

    screen.set_width(63)  # below LIBRARY_EMERGENCY_WIDTH (64)
    screen._apply_library_notes_stage_visibility_for_resize()
    assert screen.leg_calls == 1, (
        f"{screen.leg_calls} applications for one crossing into the emergency "
        "band"
    )

    # Re-armed: repeating the same frame is free again.
    for _ in range(3):
        screen._apply_library_notes_stage_visibility_for_resize()
    assert screen.leg_calls == 1, (
        f"{screen.leg_calls} applications after the crossing settled; the "
        "gate did not re-arm"
    )

    screen.set_width(64)  # back out of the band
    screen._apply_library_notes_stage_visibility_for_resize()
    assert screen.leg_calls == 2, (
        f"{screen.leg_calls - 1} applications for one emergency-band release"
    )


def test_stage_gate_fails_open_when_the_signature_cannot_be_computed():
    """TASK-23151 unit: an unavailable signature must never suppress the leg.

    ``_library_notes_stage_signature`` returns ``None`` when it cannot decide
    cheaply (unmounted, or a shell/rail/canvas reference missing mid-recompose).
    Both sides of the comparison are then ``None``, so a gate that compared
    them for equality would silently skip the work forever.
    """
    screen = _StageGateScreen(width=100)
    screen._apply_library_notes_stage_visibility_for_resize()  # arm
    assert screen._library_notes_stage_applied_signature is not None

    screen.refs["#library-rail"] = None  # mid-recompose: reference gone
    assert screen._library_notes_stage_signature() is None
    for _ in range(3):
        screen._apply_library_notes_stage_visibility_for_resize()
    assert screen.leg_calls == 4, (
        f"{screen.leg_calls - 1} of 3 frames applied the leg while the "
        "signature was unavailable; the gate must fail open"
    )
    assert screen._library_notes_stage_applied_signature is None


def test_stage_signature_carries_the_effective_not_raw_emergency_decision():
    """TASK-23151 unit: the width bucket only counts on an ordinary route.

    ``_apply_library_emergency_geometry`` gates its takeover on
    ``_library_ordinary_route_active()``, so on a browse route the same
    63-cell frame changes nothing -- carrying the RAW bucket would make every
    such crossing look like a change and re-run the leg for nothing.
    """
    inert = _StageGateScreen(width=80, row_id=LIBRARY_ROW_BROWSE_MEDIA)
    inert._apply_library_notes_stage_visibility_for_resize()  # arm
    inert.leg_calls = 0
    inert.set_width(63)
    inert._apply_library_notes_stage_visibility_for_resize()
    assert inert.leg_calls == 0, (
        "the emergency width bucket flipped the signature on a route where "
        "the emergency geometry is inert"
    )

    # Control: the identical crossing, differing only in the route id.
    ordinary = _StageGateScreen(width=80, row_id=LIBRARY_ROW_INGEST_MEDIA)
    ordinary._apply_library_notes_stage_visibility_for_resize()  # arm
    ordinary.leg_calls = 0
    ordinary.set_width(63)
    ordinary._apply_library_notes_stage_visibility_for_resize()
    assert ordinary.leg_calls == 1, (
        "the ordinary route's emergency-band crossing was skipped"
    )


def test_stage_gate_applies_when_legacy_rail_display_changes_outside_the_leg():
    """TASK-23151 unit: while the legacy path owns rail/canvas, they count.

    The stage toggles WRITE ``rail.display``/``canvas.display``, so an outside
    mutation of either has to re-run the leg rather than be gated away.
    """
    screen = _StageGateScreen(width=100, reader_active=False)
    screen._apply_library_notes_stage_visibility_for_resize()  # arm
    screen.leg_calls = 0

    screen.rail.display = False  # something else hid the rail
    screen._apply_library_notes_stage_visibility_for_resize()
    assert screen.leg_calls == 1, (
        "an outside rail.display change was gated away while the legacy path "
        "owned the toggles"
    )

    screen.canvas.display = False
    screen._apply_library_notes_stage_visibility_for_resize()
    assert screen.leg_calls == 2, (
        "an outside canvas.display change was gated away while the legacy "
        "path owned the toggles"
    )


def test_stage_gate_ignores_reader_owned_rail_display_under_adaptive_shell():
    """TASK-23151 unit: under an adaptive reader shell the rail is not ours.

    The leg returns before the legacy toggles when a reader shell is mounted;
    the reader's own ``sync_layout`` then hides the rail purely as a function
    of width. Carrying ``rail.display`` there made every same-side resize look
    like a change and held the wide case at 100 applications.
    """
    screen = _StageGateScreen(width=100, reader_active=True)
    screen._apply_library_notes_stage_visibility_for_resize()  # arm
    screen.leg_calls = 0

    # A narrower same-band frame; the reader's sync_layout hides the rail.
    screen.set_width(90)
    screen.rail.display = False
    screen._apply_library_notes_stage_visibility_for_resize()
    assert screen.leg_calls == 0, (
        f"{screen.leg_calls} stage-visibility applications for a same-band "
        "frame whose rail was hidden by the reader shell that owns it"
    )


def test_every_stage_visibility_seam_arms_the_gate_not_only_the_resize_one():
    """TASK-23151 unit: the applied signature is recorded inside the leg seam.

    The screen has ~20 seams calling ``_apply_library_notes_stage_visibility``
    directly. Recording only inside the resize gate left the record stale
    after any of them, so each resize burst paid exactly one needless
    application.
    """
    screen = _StageGateScreen(width=100)
    screen._apply_library_notes_stage_visibility()  # a non-resize seam
    assert screen.leg_calls == 1

    screen._apply_library_notes_stage_visibility_for_resize()
    assert screen.leg_calls == 1, (
        "the resize gate re-applied a leg a non-resize seam had just settled; "
        "the applied signature is not being recorded inside the seam"
    )
    assert screen._library_notes_stage_applied_signature is not None


def test_a_raising_stage_leg_leaves_the_gate_re_armed_not_stale():
    """TASK-23151 unit: the record is cleared BEFORE the legs run.

    A leg that raises has settled nothing, so leaving the previous signature
    in place would gate away the retry.
    """
    screen = _StageGateScreen(width=100)
    screen._apply_library_notes_stage_visibility_for_resize()  # arm
    assert screen._library_notes_stage_applied_signature is not None

    screen.leg_failure = RuntimeError("stage leg blew up")
    with pytest.raises(RuntimeError):
        screen._apply_library_notes_stage_visibility()
    assert screen._library_notes_stage_applied_signature is None, (
        "a raising leg left a stale applied signature behind"
    )

    screen.leg_failure = None
    screen.leg_calls = 0
    screen._apply_library_notes_stage_visibility_for_resize()
    assert screen.leg_calls == 1, (
        "the retry after a raising leg was gated away by a stale signature"
    )


@pytest.mark.asyncio
async def test_resize_ingest_rail_autocollapse_crossing_still_works():
    """The Ingest route's 100-column rail auto-collapse crosses both ways."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _settled_library(host, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await pilot.pause()
        assert screen._library_rail_collapsed is False

        await pilot.resize_terminal(90, 48)
        await _wait_for_condition(
            pilot,
            lambda: screen._ingest_state.auto_collapsed_rail,
            message="narrow Ingest resize never auto-collapsed the rail",
        )
        assert screen._library_rail_collapsed is True
        assert screen.query_one("#library-rail").display is False

        await pilot.resize_terminal(170, 48)
        await _wait_for_condition(
            pilot,
            lambda: not screen._ingest_state.auto_collapsed_rail,
            message="wide Ingest resize never restored the rail",
        )
        assert screen._library_rail_collapsed is False
        assert screen.query_one("#library-rail").display is True


@pytest.mark.asyncio
async def test_tab_focus_path_library_query_volume_is_bounded():
    """A Tab on the landing route costs at most 1 library_screen query.

    Born-red against the ungated observer installer: each press probed all
    eight scroll-owner regions (seven guaranteed-failing whole-tree walks)
    plus per-call rail/canvas/footer lookups -- 9+ library-attributed
    queries per press.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await _settled_library(host, pilot)
        counts = {"n": 0, "enabled": False}
        with _counting_queries("UI/Screens/library_screen.py", counts):
            await pilot.press("tab")  # warm-up: seeds the ref caches
            await pilot.pause()
            for _ in range(3):
                counts["n"] = 0
                counts["enabled"] = True
                await pilot.press("tab")
                await pilot.pause()
                counts["enabled"] = False
                assert counts["n"] <= 1, (
                    f"{counts['n']} library_screen-attributed DOM queries "
                    "for one Tab press on the landing route"
                )


@pytest.mark.asyncio
async def test_details_disclosure_children_stay_queryable_while_closed():
    """The closed Details body keeps its queried-while-closed contract.

    TASK-23025 considered growing the Details body on demand and reverted
    it: the counts line and the workspace Console-handoff state are read
    while the disclosure is closed (test_destination_shells,
    test_post_release_workspaces_library_depth, the landing-hub counts
    test). This pins that contract so a future deferral has to face it
    deliberately.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _settled_library(host, pilot)
        body = screen.query_one("#library-rail-section-body-details")
        assert body.display is False
        assert screen.query_one("#library-details-group-status", Static)
        assert screen.query_one("#library-details-body", Static)
        assert screen.query_one("#library-workspaces-depth-panel")

        screen.query_one(
            "#console-rail-section-toggle-library-details", Button
        ).press()
        await pilot.pause()
        await pilot.pause()
        assert body.display is True
        assert screen.query_one("#library-details-group-status", Static).display


@pytest.mark.asyncio
async def test_install_progress_grows_on_demand_renders_and_unmounts_cleanly():
    """The install label+bar pair mounts on first progress, then updates.

    Born-red against the eager implementation (pair mounted display=False on
    every visit) AND against a deferral whose handlers forget to mount.
    Ends by popping the screen with the deferred pair mounted -- the
    unmount/quit walk for the grown subtree.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _settled_library(host, pilot)
        assert not list(screen.query("#library-model-install-progress")), (
            "idle visit should not construct the install progress pair"
        )
        assert not list(screen.query("#library-model-install-progress-label"))

        mib = 1024 * 1024
        screen._library_model_install_progress_label = "Parakeet v2"
        first = SimpleNamespace(
            phase="fetch",
            ref=None,
            file="model.bin",
            bytes_done=10 * mib,
            bytes_total=100 * mib,
        )
        screen.post_message(InstallProgressed(first))
        await pilot.pause()
        await pilot.pause()

        progress = screen.query_one(
            "#library-model-install-progress", ModelInstallProgress
        )
        label = screen.query_one("#library-model-install-progress-label", Static)
        assert progress.display and label.display
        assert str(label.render()) == "Parakeet v2"
        phase = progress.query_one("#model-install-progress-phase", Static)
        assert "Downloading model" in str(phase.render())
        detail = progress.query_one("#model-install-progress-detail", Static)
        first_detail = str(detail.render())
        assert "model.bin" in first_detail

        # Second event takes the update path on the now-mounted pair; the
        # byte counter must actually advance.
        second = SimpleNamespace(
            phase="fetch",
            ref=None,
            file="model.bin",
            bytes_done=90 * mib,
            bytes_total=100 * mib,
        )
        screen.post_message(InstallProgressed(second))
        await pilot.pause()
        second_detail = str(detail.render())
        assert "model.bin" in second_detail
        assert second_detail != first_detail, (
            "second progress event did not update the mounted pair"
        )

        # Unmount walk: pop the screen with the grown subtree mounted.
        host.pop_screen()
        await pilot.pause()
        await pilot.pause()


@pytest.mark.asyncio
async def test_research_pane_layout_gate_skips_queries_and_keeps_crossings():
    """_apply_pane_layout returns before query work when nothing changed.

    Born-red against the ungated implementation: every resize frame paid
    ~11 query_one calls regardless of whether the derived layout changed.
    The band crossings (wide/medium/narrow) must keep working through the
    gate.
    """
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )
    from Tests.UI.consolidated_css import ConsolidatedCSSApp

    class _Harness(ConsolidatedCSSApp):
        async def on_mount(self) -> None:
            await self.push_screen(ResearchWorkspaceScreen(SimpleNamespace()))

    host = _Harness()
    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]
        assert screen._pane_layout is not None and screen._pane_layout.mode == "wide"

        counts = {"n": 0, "enabled": False}
        with _counting_queries(
            "UI/Screens/research_workspace_screen.py", counts
        ):
            await pilot.resize_terminal(158, 40)  # warm-up, still wide
            await pilot.pause()
            counts["enabled"] = True
            for width in (157, 156, 155):
                await pilot.resize_terminal(width, 40)
                await pilot.pause()
            counts["enabled"] = False
        assert counts["n"] == 0, (
            f"{counts['n']} research_workspace_screen-attributed DOM queries "
            "across 3 same-band resize frames"
        )

        # Crossings still apply through the gate.
        await pilot.resize_terminal(120, 40)
        await pilot.pause()
        assert screen._pane_layout.mode == "medium"
        grid = screen.query_one("#research-workspace-grid")
        assert grid.has_class("layout-medium")
        assert screen.query_one("#research-pane-mode-strip").display

        await pilot.resize_terminal(84, 24)
        await pilot.pause()
        assert screen._pane_layout.mode == "narrow"
        grid = screen.query_one("#research-workspace-grid")
        assert grid.has_class("layout-narrow")
        shell = screen.query_one("#research-workspace-shell")
        assert shell.has_class("height-compact")  # height bucket applied too

        await pilot.resize_terminal(160, 40)
        await pilot.pause()
        assert screen._pane_layout.mode == "wide"
        grid = screen.query_one("#research-workspace-grid")
        assert grid.has_class("layout-wide")
        shell = screen.query_one("#research-workspace-shell")
        assert not shell.has_class("height-compact")
