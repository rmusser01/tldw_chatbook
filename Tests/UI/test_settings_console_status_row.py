"""Settings contracts for the Console status-row placement toggle (task-17652)."""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
)
from Tests.UI.test_settings_configuration_hub import _open_settings_category
import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId


POSITION_TOGGLE = "#settings-console-status-row-position-toggle"


@pytest.mark.asyncio
async def test_status_row_position_toggle_flips_and_pokes_live_config():
    """The toggle carries the state in its label and applies immediately.

    ADR-020-style immediate write (remote-images precedent): pressing the
    button flips ``[console] status_chips_position`` in the live config —
    no category draft — and the label always reads the current placement.
    """
    app = _build_test_app()
    app.app_config["console"] = {}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        toggle = screen.query_one(POSITION_TOGGLE, Button)

        assert str(toggle.label) == "Above composer"

        toggle.press()
        await pilot.pause()
        assert app.app_config["console"]["status_chips_position"] == "below"
        assert str(toggle.label) == "Below composer"

        toggle.press()
        await pilot.pause()
        assert app.app_config["console"]["status_chips_position"] == "above"
        assert str(toggle.label) == "Above composer"


def test_status_row_position_is_field_search_indexed():
    """"/" search must land on the placement control by its label."""
    settings_screen_module._build_field_search_index()
    entries = settings_screen_module.FIELD_SEARCH_INDEX[
        SettingsCategoryId.CONSOLE_BEHAVIOR
    ]
    assert (
        "settings-console-status-row-position-toggle",
        "Status row placement",
    ) in entries
