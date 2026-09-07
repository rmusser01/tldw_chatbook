"""The Scope Inspector's "Focused setting" line must name the focused control.

TASK-23192: the line read "Appearance defaults" while "Reduce motion"
demonstrably held focus. The cause was two hand-maintained lists of the
same thing -- the ``DescendantFocus`` handler's set of ids it will record,
and the guidance branches that name them -- which drifted. This suite
mounts the real screen and pins the line against the control that actually
holds focus, by both routes users reach a control: search landing and
plain Tab traversal.
"""

import pytest
from textual.widgets import Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import NO_FOCUSED_SETTING_COPY


def _guide_line(screen, prefix: str) -> str:
    """The rendered "Focused setting" row of a category's field guide."""
    return str(screen.query_one(f"#settings-{prefix}-field-guide-0", Static).renderable)


async def _tab_to(pilot, widget_id: str, limit: int = 120) -> bool:
    """Walk focus with real Tab presses until ``widget_id`` holds it."""
    for _ in range(limit):
        if str(getattr(pilot.app.focused, "id", "") or "") == widget_id:
            return True
        await pilot.press("tab")
    return str(getattr(pilot.app.focused, "id", "") or "") == widget_id


@pytest.mark.asyncio
async def test_focused_setting_line_names_the_control_that_holds_focus():
    """AC1/AC3: two distinct settings, one reached each way."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.APPEARANCE.value)
        await _wait_for_selector(screen, pilot, "#settings-appearance-reduce-motion")
        await pilot.pause()

        # Route 1: search landing (the guarantee TASK-23109 depends on).
        screen._land_search_focus_on_field(
            "settings-appearance-reduce-motion", "Reduce motion"
        )
        await pilot.pause()
        assert (
            str(getattr(pilot.app.focused, "id", "") or "")
            == "settings-appearance-reduce-motion"
        ), "precondition: search landing put focus on Reduce motion"
        assert _guide_line(screen, "appearance") == "Focused setting: Reduce motion"
        assert (
            screen._active_settings_field_id == "settings-appearance-reduce-motion"
        )

        # Route 2: plain Tab traversal onto a different setting.
        assert await _tab_to(pilot, "settings-appearance-density"), (
            "precondition: Tab reaches the Density select"
        )
        await pilot.pause()
        assert _guide_line(screen, "appearance") == "Focused setting: Density"


@pytest.mark.asyncio
async def test_focused_setting_line_says_so_when_focus_is_not_on_a_setting():
    """AC2: a container or non-setting control must not borrow a name."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.APPEARANCE.value)
        await _wait_for_selector(screen, pilot, "#settings-appearance-reduce-motion")
        await pilot.pause()

        screen._land_search_focus_on_field(
            "settings-appearance-reduce-motion", "Reduce motion"
        )
        await pilot.pause()
        assert _guide_line(screen, "appearance") == "Focused setting: Reduce motion"

        # The category rail button is focusable and is not a setting.
        rail_button = screen.query_one("#settings-category-appearance")
        rail_button.focus()
        await pilot.pause()
        assert (
            _guide_line(screen, "appearance")
            == f"Focused setting: {NO_FOCUSED_SETTING_COPY}"
        )
        # ...and the screen must not RECORD it as the focused setting either:
        # every downstream reader of this id treats it as a named setting.
        assert screen._active_settings_field_id is None


@pytest.mark.asyncio
async def test_focused_setting_line_names_the_model_context_window():
    """AC1 on the second category that renders the line.

    ``settings-model-context-window`` has guidance rows of its own but was
    absent from the Providers focus list, so it reported "Provider setup".
    """
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await _wait_for_selector(screen, pilot, "#settings-model-context-window")
        await pilot.pause()

        screen._land_search_focus_on_field(
            "settings-model-context-window", "Model context window tokens"
        )
        await pilot.pause()
        assert (
            str(getattr(pilot.app.focused, "id", "") or "")
            == "settings-model-context-window"
        ), "precondition: focus landed on the context-window input"
        assert (
            _guide_line(screen, "provider") == "Focused setting: Model context window"
        )
