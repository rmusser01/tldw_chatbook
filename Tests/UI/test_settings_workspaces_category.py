"""Settings ▸ Workspaces category registration (spec §4)."""

from __future__ import annotations

import pytest

from Tests.UI.test_settings_configuration_hub import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _open_settings_category,
    _visible_text,
)


@pytest.mark.asyncio
async def test_workspaces_category_registered_and_immediate() -> None:
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        text = _visible_text(screen)

        assert "Workspace management" in text
        # Immediate-apply category: the guided Save/Revert buttons are
        # suppressed exactly like Theme/Splash.
        assert not screen.query("#settings-save-category")
