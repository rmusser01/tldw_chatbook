import pytest
from textual.widgets import Button, Input, Select

from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from Tests.app_module_patches import patch_app_global

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_slash_search_reuses_authority_from_inline_and_narrow_layouts(
    tmp_path, monkeypatch
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    app.workspace_registry_service.create_workspace(
        workspace_id="observatory", name="Field observatory"
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, _ = await open_chat(app, pilot)
            for size in ((160, 40), (60, 20)):
                await pilot.resize_terminal(*size)
                screen.query_one("#research-workspace-actions", Button).focus()
                await pilot.press("slash")
                await _until(
                    pilot,
                    lambda: (
                        isinstance(app.focused, Input)
                        and app.focused.id == "research-global-search-field"
                    ),
                    "search field focus",
                )
                assert (app.screen is screen) == (size[0] >= 150)
                app.focused.value = "workspace: observatory"
                await pilot.press("enter")
                go = app.screen.query_one("#research-global-search-open", Button)
                await _until(pilot, lambda go=go: not go.disabled, "search result ready")
                picked = app.screen.query_one(
                    "#research-global-search-results", Select
                ).value
                assert picked.workspace.data_source.value == "local"
                go.press()
                await _until(
                    pilot,
                    lambda: (
                        app.screen is screen
                        and screen.controller.selected_ref.workspace_id == "observatory"
                    ),
                    "canonical workspace opened",
                )
