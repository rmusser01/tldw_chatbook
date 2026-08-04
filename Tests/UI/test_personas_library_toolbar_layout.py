# Tests/UI/test_personas_library_toolbar_layout.py
"""Rendered-layout regression tests for the Roleplay library toolbar (F-030).

At supported terminal sizes every library toolbar action (New, Import,
Duplicate, Sort, Tag) must stay reachable: at 100x30 the old fixed-width
buttons clipped Import/Duplicate/Tag off the pane's right edge while the
empty-state copy still told users to "use New or Import". The pane now slims
its buttons to their labels and stacks the bars vertically when the pane is
too narrow for one row.
"""

import pytest
from textual.widgets import Button

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.UI.test_personas_dictionaries import patch_character_paging
from Tests.UI.test_personas_workbench import CHARACTERS, StyledPersonasTestApp

pytestmark = pytest.mark.asyncio


@pytest.fixture
def stub_characters(monkeypatch):
    """Same character stubs as the workbench suite (kept local for lint)."""
    monkeypatch.setattr(
        character_handler_module,
        "fetch_all_characters",
        lambda: [dict(c) for c in CHARACTERS],
    )
    monkeypatch.setattr(
        character_handler_module,
        "fetch_character_by_id",
        lambda character_id: next(
            dict(c) for c in CHARACTERS if str(c["id"]) == str(character_id)
        ),
    )
    patch_character_paging(monkeypatch)

_TOOLBAR_BUTTON_IDS = (
    "#personas-library-new",
    "#personas-library-import",
    "#personas-library-duplicate",
    "#personas-library-sort",
    "#personas-library-tag",
)


def _assert_toolbar_buttons_inside_pane(screen) -> None:
    """Every visible toolbar button renders fully inside the library pane."""
    pane = screen.query_one("#personas-library-pane")
    pane_right = pane.region.x + pane.region.width
    pane_bottom = pane.region.y + pane.region.height
    for button_id in _TOOLBAR_BUTTON_IDS:
        button = screen.query_one(button_id, Button)
        assert button.display is True, f"{button_id} not rendered in characters mode"
        region = button.region
        assert region.width > 0 and region.height > 0, (
            f"{button_id} has no rendered area: {region}"
        )
        assert region.x >= pane.region.x, (
            f"{button_id} starts left of the pane: {region}"
        )
        assert region.x + region.width <= pane_right, (
            f"{button_id} clips past the pane's right edge: "
            f"{region} vs pane right {pane_right}"
        )
        assert region.y + region.height <= pane_bottom, (
            f"{button_id} clips past the pane's bottom edge: "
            f"{region} vs pane bottom {pane_bottom}"
        )


async def _mounted_screen(pilot):
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return pilot.app.screen


async def test_toolbar_actions_reachable_at_100x30(
    mock_app_instance, stub_characters
):
    """F-030 evidence size: Import/Duplicate/Tag used to clip here."""
    app = StyledPersonasTestApp(mock_app_instance)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await _mounted_screen(pilot)
        pane = screen.query_one("#personas-library-pane")
        # Too narrow for one row: the bars must wrap, not clip.
        assert pane.has_class("personas-library-stacked-controls")
        _assert_toolbar_buttons_inside_pane(screen)


async def test_toolbar_actions_reachable_at_80x24(
    mock_app_instance, stub_characters
):
    """Smallest supported size: all actions still reachable."""
    app = StyledPersonasTestApp(mock_app_instance)
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _mounted_screen(pilot)
        pane = screen.query_one("#personas-library-pane")
        assert pane.has_class("personas-library-stacked-controls")
        _assert_toolbar_buttons_inside_pane(screen)


async def test_toolbar_single_row_at_wide_terminal(
    mock_app_instance, stub_characters
):
    """Counter-case: wide terminals keep the one-row toolbar (no stacking)."""
    app = StyledPersonasTestApp(mock_app_instance)
    async with app.run_test(size=(170, 50)) as pilot:
        screen = await _mounted_screen(pilot)
        pane = screen.query_one("#personas-library-pane")
        assert not pane.has_class("personas-library-stacked-controls")
        _assert_toolbar_buttons_inside_pane(screen)
        # Both bars stay on one row each when there is room.
        toolbar = screen.query_one("#personas-library-toolbar")
        new_button = screen.query_one("#personas-library-new", Button)
        import_button = screen.query_one("#personas-library-import", Button)
        assert new_button.region.y == toolbar.region.y
        assert import_button.region.y == toolbar.region.y


async def test_preview_pane_anchors_top_of_center_canvas(
    mock_app_instance, stub_characters
):
    """F-039: the preview affordance is attached to the canvas it previews -
    the toggle sits immediately above the center detail stack, not stranded
    at the bottom of the work area."""
    app = StyledPersonasTestApp(mock_app_instance)
    async with app.run_test(size=(170, 50)) as pilot:
        screen = await _mounted_screen(pilot)
        work_area = screen.query_one("#personas-work-area")
        preview = screen.query_one("#personas-preview-pane")
        stack = screen.query_one("#personas-detail-stack")
        # Inside the center column...
        assert preview.region.x >= work_area.region.x
        assert preview.region.x + preview.region.width <= (
            work_area.region.x + work_area.region.width
        )
        # ...and flush against the top of the detail stack (adjacent edges).
        assert preview.region.y + preview.region.height == stack.region.y
        # The toggle reads as an expand/collapse section header, and
        # (task-2234) its label states the payoff, not the feature name.
        toggle = screen.query_one("#personas-preview-toggle", Button)
        assert str(toggle.label) == "▸ Try a test chat (nothing saved)"
        await pilot.click("#personas-preview-toggle")
        await pilot.pause()
        assert str(toggle.label) == "▾ Try a test chat (nothing saved)"
        assert screen.query_one("#personas-preview-body").display is True
