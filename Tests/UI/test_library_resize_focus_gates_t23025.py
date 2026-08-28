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
relocated below the return) can pass.
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
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
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
            lambda: screen._library_ingest_auto_collapsed_rail,
            message="narrow Ingest resize never auto-collapsed the rail",
        )
        assert screen._library_rail_collapsed is True
        assert screen.query_one("#library-rail").display is False

        await pilot.resize_terminal(170, 48)
        await _wait_for_condition(
            pilot,
            lambda: not screen._library_ingest_auto_collapsed_rail,
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
