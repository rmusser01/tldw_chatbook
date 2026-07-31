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


@pytest.mark.asyncio
async def test_settings_theme_editor_mount_does_not_mark_modified():
    """Mounting and initializing the editor must leave is_modified False.

    load_theme writes Input values programmatically; the resulting
    Input.Changed events must not count as user edits. A spurious True here
    recomposes SettingsScreen, which mounts a fresh editor, which posts
    again -- the infinite recompose storm that froze the whole app
    (rescore P1: Theme traps keyboard navigation).
    """
    app = _build_test_app()
    editor = SettingsThemeEditor()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.mount(editor)
        for _ in range(8):
            await pilot.pause()
        assert editor.is_modified is False


@pytest.mark.asyncio
async def test_settings_theme_editor_mount_posts_no_modified_status():
    """A clean mount must not emit ThemeModifiedStatus at all.

    Textual calls watch methods with the initial value during mount when the
    reactive keeps init=True; that ThemeModifiedStatus(False) is one half of
    the False/True oscillation that drove the recompose storm.
    """
    app = _build_test_app()
    editor = SettingsThemeEditor()
    posted = []
    original = editor.post_message

    def record(message):
        if isinstance(message, SettingsThemeEditor.ThemeModifiedStatus):
            posted.append(message.is_modified)
        return original(message)

    editor.post_message = record
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.mount(editor)
        for _ in range(8):
            await pilot.pause()
        assert posted == []


@pytest.mark.asyncio
async def test_settings_theme_editor_user_edit_still_marks_modified():
    """A real user edit (a color value differing from the loaded theme)
    still flips is_modified True -- the no-op guard must not eat real edits."""
    app = _build_test_app()
    editor = SettingsThemeEditor()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.mount(editor)
        for _ in range(8):
            await pilot.pause()
        target = editor.query_one("#settings-theme-color-primary", Input)
        target.value = "#123456"
        await pilot.pause()
        assert editor.is_modified is True
