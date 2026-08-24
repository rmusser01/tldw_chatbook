"""Focused behavior tests for the Watchlists pane grip."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Watchlists_Modules.pane_grip import (
    RegionToggled,
    WatchlistsPaneGrip,
)
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region


class PaneGripApp(App[None]):
    """Minimal host which records pane-toggle messages."""

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def __init__(self, region: Region, *, expanded: bool) -> None:
        super().__init__()
        self.grip = WatchlistsPaneGrip(region, expanded=expanded, id="pane-grip")
        self.toggles: list[Region] = []

    def compose(self) -> ComposeResult:
        yield self.grip

    def on_region_toggled(self, message: RegionToggled) -> None:
        self.toggles.append(message.region)


@pytest.mark.parametrize(
    ("region", "expanded", "expected"),
    [
        (Region.LEFT_RAIL, False, "--->"),
        (Region.LEFT_RAIL, True, "<---"),
        (Region.ITEMS, False, "--->"),
        (Region.ITEMS, True, "<---"),
        (Region.RIGHT_RAIL, False, "<---"),
        (Region.RIGHT_RAIL, True, "--->"),
    ],
)
def test_grip_direction_matches_pane_side_and_state(
    region: Region, expanded: bool, expected: str
) -> None:
    grip = WatchlistsPaneGrip(region, expanded=expanded)

    assert str(grip.label) == expected


@pytest.mark.parametrize(
    ("region", "pane_name"),
    [
        (Region.LEFT_RAIL, "Navigation"),
        (Region.ITEMS, "Feed Items"),
        (Region.RIGHT_RAIL, "Inspector"),
    ],
)
def test_grip_exposes_actionable_copy_and_compact_behavior(
    region: Region, pane_name: str
) -> None:
    grip = WatchlistsPaneGrip(region, expanded=False)

    assert grip.can_focus
    assert grip.tooltip == f"Expand {pane_name}"
    assert grip.name == f"Expand {pane_name}"
    assert grip.compact
    assert grip.styles.line_pad == 0
    assert grip.has_class("watchlists-pane-grip")


@pytest.mark.asyncio
async def test_click_posts_exactly_one_region_toggled_message() -> None:
    app = PaneGripApp(Region.LEFT_RAIL, expanded=False)

    async with app.run_test() as pilot:
        await pilot.click("#pane-grip")
        await pilot.pause()

    assert app.toggles == [Region.LEFT_RAIL]


@pytest.mark.asyncio
async def test_enter_posts_exactly_one_region_toggled_message() -> None:
    app = PaneGripApp(Region.RIGHT_RAIL, expanded=False)

    async with app.run_test() as pilot:
        app.grip.focus()
        await pilot.press("enter")
        await pilot.pause()

    assert app.toggles == [Region.RIGHT_RAIL]


@pytest.mark.asyncio
async def test_expanded_update_relabels_the_same_widget_in_place() -> None:
    app = PaneGripApp(Region.ITEMS, expanded=False)

    async with app.run_test() as pilot:
        original = app.grip
        original.expanded = True
        await pilot.pause()

        assert app.query_one("#pane-grip") is original
        assert str(original.label) == "<---"
        assert original.tooltip == "Collapse Feed Items"
        assert original.name == "Collapse Feed Items"


def _painted_centre_row(app: App[None], grip: WatchlistsPaneGrip) -> str:
    """Return the five compositor cells occupied by ``grip``."""
    strips = list(app.screen._compositor.render_strips())
    y = grip.region.y + (grip.region.height - 1) // 2
    return strips[y].crop(grip.region.x, grip.region.right).text


@pytest.mark.parametrize(
    ("region", "expanded", "arrow"),
    [
        (Region.LEFT_RAIL, False, "--->"),
        (Region.LEFT_RAIL, True, "<---"),
        (Region.ITEMS, False, "--->"),
        (Region.ITEMS, True, "<---"),
        (Region.RIGHT_RAIL, False, "<---"),
        (Region.RIGHT_RAIL, True, "--->"),
    ],
)
@pytest.mark.asyncio
async def test_real_bundle_paints_full_arrow_beside_one_fixed_divider(
    region: Region, expanded: bool, arrow: str
) -> None:
    app = PaneGripApp(region, expanded=expanded)

    async with app.run_test(size=(5, 9)) as pilot:
        await pilot.pause()
        grip = app.query_one("#pane-grip", WatchlistsPaneGrip)
        resting_region = grip.region
        resting_content = grip.content_region

        assert grip.styles.width is not None
        assert grip.styles.width.value == 5
        assert grip.styles.min_width is not None
        assert grip.styles.min_width.value == 5
        assert grip.styles.max_width is not None
        assert grip.styles.max_width.value == 5
        assert grip.outer_size.width == grip.region.width == 5
        assert grip.content_region.width == 4
        assert _painted_centre_row(app, grip)[1:] == arrow

        grip.focus()
        await pilot.pause()

        assert grip.region == resting_region
        assert grip.content_region == resting_content
        assert _painted_centre_row(app, grip)[1:] == arrow
