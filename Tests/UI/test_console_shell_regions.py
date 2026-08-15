"""Painted-geometry contract for the Console shell's production regions.

The expectation table pins the approved shell policy at each size, including
regions that are legitimately hidden in compact mode. It mounts the production
hierarchy with the shipped stylesheet so simplified widget geometry cannot
stand in for application behavior.

Three sizes are pinned:

- 160x45 and 235x52 both sit ABOVE the shell's own ``-console-compact``
  height threshold (``CONSOLE_COMPACT_HEIGHT_ROWS = 35``, see
  ``chat_screen.py``) and above the right-rail width-forced-collapse
  threshold (``CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS = 150``, see
  ``Chat/console_rail_state.py``). Against a freshly built harness (no
  stored rail preferences, no active run, right rail closed by default),
  every region is hidden/hittable identically at both -- an observed fact
  about this id list and these two sizes, not an assumption.
- 120x30 crosses BOTH thresholds (30 < 35 rows, 120 < 150 columns) AND
  lands inside a third, narrower, deliberate rule:
  ``_should_open_standard_width_inspector`` auto-opens the Inspector rail
  whenever available columns fall in ``118..128`` and the Inspector already
  has a "Run recipe" row plus a companion row (Blocked impact / Next
  action / Sources / Tools / Approvals / Artifacts) -- the fresh harness's
  setup-blocked state supplies exactly that, so at 120x30 the Inspector is
  OPEN by default where it is closed at the other two sizes. Inspector-first
  compact priority then hides the Context rail, exposes its reveal handle,
  and grants the Inspector compact-override authority. The Transcript's
  minimum-width waiver keeps all displayed workspace-grid children inside
  both the grid and viewport. The Inspector reveal handle stays hidden, and
  ``#console-run-inspector`` is newly present in the DOM with ``display=True``
  -- but see the "clipped" state below before assuming that means visible.

``#console-mode-bar`` is hidden unconditionally at any size: it is a legacy
compatibility seam retained only for older selectors and is composed via
``_hidden_static`` regardless of geometry (see the comment directly above
its ``compose()`` yield in chat_screen.py).

A third expectation state, ``"clipped"``, exists alongside "hittable" and
"hidden": at 120x30 narrow-layout overflow can leave a mounted region with
positive virtual geometry whose reported center is either outside the screen
or painted by an unrelated widget. The auto-opened Inspector's
scrollable body (``#console-inspector-rail-body``) has a real viewport only
3 rows tall against ~28 rows of virtual content. Textual still reports a
non-empty, ``display=True`` ``.region`` for ``#console-run-inspector`` (a
child scrolled below that viewport), but its *unclipped* center is either
outside the screen or resolves to an unrelated painted widget. So this region
is neither cleanly "hidden" (``display`` is True) nor cleanly "hittable" (its
own reported center never resolves to itself or a descendant) -- pinning it
as a fabricated "hittable" or "hidden" would misrepresent what the shell
does today. "clipped" asserts both halves of that observed reality.
"""

from contextlib import asynccontextmanager

import pytest
from textual.errors import NoWidget
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.Console.console_rail_handle import ConsoleRailHandle

# (id, expected_at_160x45, expected_at_235x52, expected_at_120x30) where
# expected is "hittable" | "hidden" | "clipped" -- pinned against the
# production hierarchy and shipped stylesheet.
_REGIONS: list[tuple[str, str, str, str]] = [
    ("#console-shell", "hittable", "hittable", "hittable"),
    ("#console-left-rail", "hittable", "hittable", "hidden"),
    ("#console-left-rail-body", "hittable", "hittable", "hidden"),
    ("#console-main-column", "hittable", "hittable", "hittable"),
    ("#console-context-rail-handle", "hidden", "hidden", "hittable"),
    ("#console-inspector-rail-handle", "hittable", "hittable", "hidden"),
    ("#console-control-bar", "hittable", "hittable", "hittable"),
    ("#console-mode-bar", "hidden", "hidden", "hidden"),
    ("#console-native-composer", "hittable", "hittable", "hittable"),
    ("#console-run-inspector", "hidden", "hidden", "clipped"),
]

_EXPECTED_BY_SIZE = {
    (160, 45): 0,
    (235, 52): 1,
    (120, 30): 2,
}


class ProductionCSSConsoleHarness(ConsoleHarness):
    """Console harness with the exact production stylesheet stack and order."""

    CSS_PATH = TldwCli.CSS_PATH


@asynccontextmanager
async def make_console_pilot(*, size):
    """Mount a fresh Console (ChatScreen) at ``size`` via the production harness.

    Build a fresh ``TldwCli`` with every real I/O seam faked out
    (``_build_test_app``), push its real ``ChatScreen`` onto a
    ``ConsoleHarness`` carrying the exact production CSS stack, and wait for
    the composer -- the same "the shell is up" signal used elsewhere -- before
    handing control to the caller.
    """
    app = _build_test_app()
    host = ProductionCSSConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        # The setup-blocked state supplies the Inspector rows that trigger its
        # production auto-open. Hide only its covering overlay afterward so
        # hit-tests can inspect the underlying shell geometry.
        console.query_one("#console-setup-modal").display = False
        await pilot.pause()
        yield pilot


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stacked", "left_width", "right_width", "left_label", "right_label"),
    [
        (False, 13, 11, "Context->", "<-Inspect"),
        (True, 3, 3, "C\no\nn\nt\ne\nx\nt", "I\nn\ns\np\ne\nc\nt\no\nr"),
    ],
)
async def test_fresh_console_composes_saved_rail_label_style(
    stacked, left_width, right_width, left_label, right_label
):
    """A fresh Console reads the saved style for both collapsed handles."""
    app = _build_test_app()
    app.app_config.setdefault("console", {})["stack_collapsed_rail_labels"] = stacked
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        left = console.query_one("#console-context-rail-handle", ConsoleRailHandle)
        right = console.query_one("#console-inspector-rail-handle", ConsoleRailHandle)
        left_button = console.query_one("#console-context-rail-open", Button)
        right_button = console.query_one("#console-inspector-rail-open", Button)
        right_badge = console.query_one("#console-inspector-rail-badge", Static)

        assert left.styles.width.value == left_width
        assert right.styles.width.value == right_width
        assert left._display_label() == left_label
        assert right._display_label() == right_label
        assert left_button.tooltip == "Open Context rail"
        assert right_button.tooltip == "Open Inspector rail"
        assert str(right_badge.renderable) == right._display_badge()
        assert right_badge.tooltip == right.badge

        await pilot.click("#console-context-rail-collapse")
        await pilot.pause()
        assert left.display is True
        assert console.query_one("#console-left-rail").display is False
        await pilot.click("#console-context-rail-open")
        await pilot.pause()
        assert left.display is False
        assert console.query_one("#console-left-rail").display is True

        assert right.display is True
        await pilot.click("#console-inspector-rail-open")
        await pilot.pause()
        assert right.display is False
        assert console.query_one("#console-right-rail").display is True
        await pilot.click("#console-inspector-rail-collapse")
        await pilot.pause()
        assert right.display is True
        assert console.query_one("#console-right-rail").display is False


@pytest.mark.asyncio
async def test_console_pilot_uses_the_exact_production_css_stack() -> None:
    """Geometry evidence loads every production stylesheet in production order."""
    async with make_console_pilot(size=(120, 30)) as pilot:
        assert pilot.app.CSS_PATH == pilot.app.app_instance.CSS_PATH


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (235, 52), (120, 30)])
@pytest.mark.parametrize(
    "region_id,expect_160x45,expect_235x52,expect_120x30", _REGIONS
)
async def test_region_geometry_is_stable(
    region_id, expect_160x45, expect_235x52, expect_120x30, size
):
    expected = (expect_160x45, expect_235x52, expect_120x30)[_EXPECTED_BY_SIZE[size]]
    async with make_console_pilot(size=size) as pilot:
        nodes = pilot.app.screen.query(region_id)
        if expected == "hidden":
            # Tightened per review: a query returning zero nodes is NOT the
            # same fact as a mounted-but-display:none node, and Tasks 3/4
            # extract exactly the blocks holding these ids -- a silently
            # dropped/renamed id must fail this baseline, not sail through
            # the same branch as "legitimately hidden".
            assert len(nodes) == 1 and not nodes[0].display
            return
        if expected == "clipped":
            # See the module docstring's "clipped" paragraph: mounted and
            # displayed with a purported region, but scrolled below its
            # container's real viewport so nothing is actually painted at
            # its own reported center -- pin both halves of that fact.
            assert len(nodes) == 1
            node = nodes[0]
            assert node.display and node.region.width > 0
            center = node.region.center
            if not pilot.app.screen.region.contains(*center):
                return
            try:
                hit = pilot.app.screen.get_widget_at(*center)[0]
            except NoWidget:
                return
            assert not (
                hit is node or node in hit.ancestors or hit in node.walk_children()
            )
            return
        node = nodes[0]
        assert node.display and node.region.width > 0
        hit = pilot.app.screen.get_widget_at(*node.region.center)[0]
        assert hit is node or node in hit.ancestors or hit in node.walk_children()


@pytest.mark.asyncio
async def test_compact_workspace_grid_children_are_contained() -> None:
    """The real 120x30 workspace keeps every displayed pane horizontally in bounds."""
    async with make_console_pilot(size=(120, 30)) as pilot:
        screen = pilot.app.screen
        grid = screen.query_one("#console-workspace-grid")
        children = tuple(grid.children)

        assert {child.id for child in children} == {
            "console-context-rail-handle",
            "console-left-rail",
            "console-main-column",
            "console-right-rail",
            "console-inspector-rail-handle",
        }
        displayed = tuple(child for child in children if child.display)
        assert len(displayed) == 3

        for child in displayed:
            child_id = child.id
            assert child.region.width > 0 and child.region.height > 0, (
                f"{child_id} has no painted geometry: child={child.region}"
            )
            assert grid.content_region.x <= child.region.x, (
                f"{child_id} starts before workspace grid content: "
                f"child={child.region}, grid={grid.content_region}"
            )
            assert child.region.right <= grid.content_region.right, (
                f"{child_id} ends after workspace grid content: "
                f"child={child.region}, grid={grid.content_region}"
            )
            assert screen.region.x <= child.region.x, (
                f"{child_id} starts before the 120x30 viewport: "
                f"child={child.region}, screen={screen.region}"
            )
            assert child.region.right <= screen.region.right, (
                f"{child_id} ends after the 120x30 viewport: "
                f"child={child.region}, screen={screen.region}"
            )

        assert {child.id for child in displayed} == {
            "console-context-rail-handle",
            "console-main-column",
            "console-right-rail",
        }
