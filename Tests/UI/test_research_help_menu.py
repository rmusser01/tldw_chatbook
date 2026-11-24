import pytest
from textual.widgets import Button, Static

from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from Tests.app_module_patches import patch_app_global

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_research_help_opens_real_bindings_and_qualified_status_at_narrow_width(
    tmp_path, monkeypatch
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(60, 20)) as pilot:
            screen, _ = await open_chat(app, pilot)
            help_button = screen.query_one("#research-workspace-help", Button)
            help_button.focus()
            await pilot.press("f1")
            await _until(
                pilot,
                lambda: bool(list(app.screen.query("#research-help-keyboard"))),
                "Help commands",
            )
            assert not list(app.screen.query("#research-help-telemetry"))
            app.screen.query_one("#research-help-keyboard", Button).press()
            await _until(
                pilot,
                lambda: bool(list(app.screen.query("#research-help-detail-body"))),
                "keyboard help",
            )
            shortcuts = str(
                app.screen.query_one("#research-help-detail-body", Static).render()
            )
            assert (
                "/" in shortcuts
                and "Search workspaces" in shortcuts
                and "F1" in shortcuts
            )
            await pilot.press("escape")
            await _until(pilot, lambda: app.screen is screen, "returned from help")
            await _until(pilot, lambda: not help_button.disabled, "Help ready")
            help_button.press()
            await _until(
                pilot,
                lambda: bool(list(app.screen.query("#research-help-status"))),
                "status in Help",
            )
            app.screen.query_one("#research-help-status", Button).press()
            await _until(
                pilot,
                lambda: bool(list(app.screen.query("#research-help-detail-body"))),
                "workspace status",
            )
            status = str(
                app.screen.query_one("#research-help-detail-body", Static).render()
            )
            assert (
                "Local" in status
                and "Study group" in status
                and "No active operations" in status
            )
