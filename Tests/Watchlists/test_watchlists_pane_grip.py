"""Focused behavior tests for the Watchlists pane grip."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Watchlists_Modules.pane_grip import (
    RegionToggled,
    WatchlistsPaneGrip,
)
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region


class PaneGripApp(App[None]):
    """Minimal host which records pane-toggle messages."""

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
def test_grip_exposes_actionable_copy_and_compact_geometry(
    region: Region, pane_name: str
) -> None:
    grip = WatchlistsPaneGrip(region, expanded=False)

    assert grip.can_focus
    assert grip.tooltip == f"Expand {pane_name}"
    assert grip.name == f"Expand {pane_name}"
    assert grip.compact
    assert grip.styles.width is not None
    assert grip.styles.width.value == 5
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
