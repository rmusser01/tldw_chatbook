import pytest
from textual.widgets import Input

from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.Widgets.settings_theme_editor import SettingsThemeEditor


@pytest.mark.asyncio
async def test_settings_theme_editor_can_compose():
    app = _build_test_app()
    editor = SettingsThemeEditor()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.mount(editor)
        await pilot.pause()
        assert editor.is_mounted


@pytest.mark.asyncio
async def test_settings_theme_editor_has_color_inputs_for_all_base_colors():
    app = _build_test_app()
    editor = SettingsThemeEditor()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.mount(editor)
        await pilot.pause()
        inputs = editor.query(Input)
        color_inputs = [
            inp for inp in inputs if inp.id and inp.id.startswith("settings-theme-color-")
        ]
        assert len(color_inputs) == len(editor.BASE_COLORS)


@pytest.mark.asyncio
async def test_settings_theme_editor_tracks_modified_state():
    app = _build_test_app()
    editor = SettingsThemeEditor()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.mount(editor)
        await pilot.pause()
        editor.is_modified = True
        await pilot.pause()
        assert editor.is_modified is True


@pytest.mark.asyncio
async def test_settings_theme_editor_mounts_without_error():
    app = _build_test_app()
    editor = SettingsThemeEditor()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.mount(editor)
        await pilot.pause()
        assert editor.is_mounted
