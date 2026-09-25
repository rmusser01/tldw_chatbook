"""TASK-32948: Settings > Theme opens on the picker; Clone/New swap in the editor."""

import pytest
from textual.widgets import ContentSwitcher

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_settings_overview_search_journeys import _category
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness
from tldw_chatbook.css.Themes.themes import ALL_THEMES


def _host():
    host = _StyledDestinationHarness(_build_test_app(), "settings")
    for theme in ALL_THEMES:
        host.register_theme(theme)
    return host


@pytest.mark.asyncio
@private_profile_test
async def test_theme_category_opens_on_the_picker_without_state_banner(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-picker"
        assert not host.screen.query("#settings-category-state-banner")
        # R6: picker and editor buttons no longer share ids.
        for button_id in ("clone", "new"):
            assert len(host.screen.query(f"#settings-theme-{button_id}")) == 1
            assert len(host.screen.query(f"#settings-theme-picker-{button_id}")) == 1


@pytest.mark.asyncio
@private_profile_test
async def test_clone_opens_editor_and_back_returns(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        host.screen.query_one("#settings-theme-list").focus()
        await pilot.press("c")
        await pilot.pause(0.2)
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-editor-view"
        editor = host.screen.query_one("#settings-theme-editor")
        assert editor.current_theme_name.endswith("_copy")
        assert not host.screen.query("#settings-theme-set-default")
        editor.is_modified = False
        await pilot.click("#settings-theme-back")
        await pilot.pause(0.2)
        assert pane.current == "settings-theme-picker"


@pytest.mark.asyncio
@private_profile_test
async def test_back_with_unsaved_edits_asks_first(request):
    from tldw_chatbook.Widgets.settings_theme_editor import ThemeLeaveModal

    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        host.screen.query_one("#settings-theme-list").focus()
        await pilot.press("c")
        await pilot.pause(0.2)
        host.screen.query_one("#settings-theme-editor").is_modified = True
        await pilot.click("#settings-theme-back")
        await pilot.pause(0.2)
        assert isinstance(host.screen, ThemeLeaveModal)
        await pilot.press("escape")  # Stay
        await pilot.pause(0.2)
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-editor-view"


@pytest.mark.asyncio
@private_profile_test
async def test_appearance_shows_read_only_theme_row_and_opens_picker(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Appearance")
        assert not host.screen.query("#settings-appearance-theme")
        summary = str(host.screen.query_one("#settings-appearance-theme-summary").render())
        assert "launch default" in summary and "active" in summary
        host.theme = "nord"  # e.g. from the palette, while Appearance is open
        await pilot.pause(0.2)
        assert "active: Nord" in str(host.screen.query_one("#settings-appearance-theme-summary").render())
        await pilot.click("#settings-appearance-open-theme")
        await pilot.pause(0.3)
        assert host.screen.query_one("#settings-theme-pane").current == "settings-theme-picker"


def test_appearance_save_never_writes_default_theme():
    from dataclasses import replace

    from tldw_chatbook.UI.Screens import settings_appearance_defaults as sad

    values = replace(sad.SettingsAppearanceDefaults(), default_theme="textual-light")
    sections = sad.build_appearance_save_sections({"general": {"default_theme": "nord"}}, values)
    # The existing launch default passes through untouched; the draft value never lands.
    assert sections["general"]["default_theme"] == "nord"


def test_appearance_validation_ignores_theme():
    from dataclasses import replace

    from tldw_chatbook.UI.Screens import settings_appearance_defaults as sad

    values = replace(sad.SettingsAppearanceDefaults(), default_theme="")
    assert sad.validate_appearance_defaults(values).valid
