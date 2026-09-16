"""Deferred Workflows styles reach the real app, including its first dialog."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_screen_navigation import _wait_for_initial_screen
from Tests.UI.test_workflows_editor import assert_hit, painted_text
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.workflows_screen import WorkflowsScreen
from tldw_chatbook.UI.Workflows_Modules.library import ChoiceModal


@pytest.mark.parametrize("initial_route", ["home", "workflows"])
async def test_first_workflows_entry_loads_styles_and_paints_dialog(
    tmp_path, monkeypatch, initial_route
):
    """Both route entry paths style compact controls without a boot tax on Home."""
    save_setting_to_cli_config("splash_screen", "enabled", False)
    monkeypatch.setattr(
        "tldw_chatbook.config.get_workflows_db_path",
        lambda: tmp_path / "workflows.sqlite3",
    )
    app = _build_test_app(
        initial_route, config_overrides={"splash_screen": {"enabled": False}}
    )
    monkeypatch.setattr(app, "_refresh_model_catalogs", AsyncMock())
    sheet = str(BUNDLED_STYLESHEET.parent / "screen_feature_workflows.tcss")
    async with app.run_test(size=(110, 36)) as pilot:
        await _wait_for_initial_screen(pilot)
        if initial_route == "home":
            assert not app.stylesheet.has_source(sheet, "")
            app.post_message(NavigateToScreen("workflows"))
            await pilot.pause()
        await asyncio.gather(
            *(w.wait() for w in app.workers if w.group.startswith("workflows-"))
        )
        await pilot.pause()
        assert isinstance(app.screen, WorkflowsScreen)
        assert app.stylesheet.has_source(sheet, "")
        screen = app.screen
        more = screen.query_one("#workflow-more", Button)
        assert more.region.height == 1
        assert_hit(screen, more)
        context = screen.query_one("#workflows-console-unavailable")
        assert context.region.height == 1
        assert_hit(screen, context)
        assert "No active workflow run" in painted_text(screen)

        assert await pilot.click("#workflow-more")
        await pilot.pause()
        assert isinstance(app.screen, ChoiceModal)
        assert "Workflow actions" in painted_text(app.screen)
        cancel = app.screen.query_one("#workflow-dialog-cancel", Button)
        assert cancel.region.height == 1
        assert_hit(app.screen, cancel)
        assert await pilot.click("#workflow-dialog-cancel")
        await pilot.pause()
        assert app.screen is screen
        assert app.focused is more
