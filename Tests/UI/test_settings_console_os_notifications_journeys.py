"""Console OS notification settings: staged, savable, durable (TASK-18925)."""

import tomllib
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_settings_overview_search_journeys import _category
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness


@private_profile_test
@pytest.mark.timeout(240)
async def test_os_notifications_toggle_journey_saves_and_persists(request):
    """Toggling the master and one per-type box saves the nested
    ``[console] os_notifications`` table durably, and re-entering the
    category shows the saved values (not the opt-in defaults)."""
    from textual.widgets import Checkbox

    from tldw_chatbook import config

    app = _build_test_app()
    host = _StyledDestinationHarness(app, "settings")
    path = Path(config.get_cli_config_path())

    async with host.run_test(size=(170, 48)) as pilot:
        await _category(host, pilot, "Console Behavior")
        screen = host.screen
        master = screen.query_one(
            "#settings-console-os-notifications-enabled", Checkbox
        )
        run_failed = screen.query_one(
            "#settings-console-os-notifications-run-failed", Checkbox
        )
        # Opt-in defaults: master OFF, per-type ON (TASK-18925 AC#1).
        assert master.value is False
        assert run_failed.value is True

        master.focus()
        await _settle(host, pilot)
        await pilot.press("space")
        await _settle(host, pilot)
        assert master.value is True

        run_failed.focus()
        await _settle(host, pilot)
        await pilot.press("space")
        await _settle(host, pilot)
        assert run_failed.value is False

        await pilot.press("escape", "s")
        await _settle(host, pilot)

        data = tomllib.loads(path.read_text())
        assert data["console"]["os_notifications"] == {
            "enabled": True,
            "run_finished": True,
            "run_failed": False,
            "needs_approval": True,
        }

        # Leaving and re-entering the category must show the saved values,
        # proving the post-mount sync reads config, not compose-time state.
        await _category(host, pilot, "Overview")
        await _category(host, pilot, "Console Behavior")
        assert (
            screen.query_one(
                "#settings-console-os-notifications-enabled", Checkbox
            ).value
            is True
        )
        assert (
            screen.query_one(
                "#settings-console-os-notifications-run-failed", Checkbox
            ).value
            is False
        )
