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


@pytest.mark.asyncio
async def test_theme_tree_has_empty_state_guidance():
    """The collapsed Themes tree left a large blank region (rescore P3);
    a hint under it explains what the tree is for and how to start."""
    app = _build_test_app()
    editor = SettingsThemeEditor()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.mount(editor)
        await pilot.pause()
        from textual.widgets import Static

        hint = editor.query_one("#settings-theme-tree-hint", Static)
        text = str(hint.renderable)
        assert "New" in text and "theme" in text.lower()


@pytest.mark.asyncio
async def test_dark_mode_switch_has_text_state():
    """The Dark theme switch was an empty rectangle with no text state --
    the only control-without-text-state in the critique evidence. A state
    word beside it mirrors the Splash Screen switch pattern."""
    from textual.widgets import Static, Switch

    app = _build_test_app()
    editor = SettingsThemeEditor()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.mount(editor)
        for _ in range(6):
            await pilot.pause()
        switch = editor.query_one("#settings-theme-dark-mode", Switch)
        state = editor.query_one("#settings-theme-dark-mode-state", Static)
        assert str(state.renderable) == ("On" if switch.value else "Off")

        switch.value = not switch.value
        await pilot.pause()
        state = editor.query_one("#settings-theme-dark-mode-state", Static)
        assert str(state.renderable) == ("On" if switch.value else "Off")
