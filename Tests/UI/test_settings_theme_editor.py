import pytest

import types

from textual.app import App, ComposeResult
from textual.widgets import Checkbox, Input, Tree

from Tests.UI.test_destination_shells import _build_test_app
from Tests.textual_test_harness import IsolatedWidgetTestApp
from tldw_chatbook.Widgets.settings_theme_editor import SettingsThemeEditor
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.css.Themes.themes import ALL_THEMES


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
async def test_settings_theme_editor_dark_mode_checkbox_tracks_real_changes_only():
    app = _build_test_app()
    editor = SettingsThemeEditor()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.mount(editor)
        await pilot.pause()

        checkbox = editor.query_one("#settings-theme-dark-mode", Checkbox)
        initial = editor.is_dark_theme
        assert checkbox.value is initial

        # A real toggle flips the flag and marks the editor modified.
        editor.is_modified = False
        checkbox.value = not initial
        await pilot.pause()
        assert editor.is_dark_theme is (not initial)
        assert editor.is_modified is True

        # Programmatic sync firing Changed with event.value == is_dark_theme
        # must not re-mark modified (guard at on_dark_mode_changed).
        editor.is_dark_theme = initial
        editor.is_modified = False
        editor._update_dark_mode_checkbox()
        await pilot.pause()
        assert checkbox.value is initial
        assert editor.is_modified is False


def _isolated_editor_app(editor: SettingsThemeEditor) -> IsolatedWidgetTestApp:
    """Mount the editor in a minimal app, not a real TldwCli.

    A real ``TldwCli`` keeps its startup machinery (initial-screen push with a
    DestinationHeader, background workers) alive during run_test; the delete
    path's extra message-loop pumps can then race that ambient screen compose
    and fail with unrelated ``NoMatches`` errors depending on which test files
    ran before.  The isolated app has no such machinery, so these tests only
    ever touch their own app instance.
    """

    def compose() -> ComposeResult:
        yield editor

    return IsolatedWidgetTestApp(compose)


def _isolated_editor_app_with_real_screens(
    editor: SettingsThemeEditor,
) -> IsolatedWidgetTestApp:
    """Isolated app with the REAL screen stack restored.

    ``IsolatedWidgetTestApp`` mocks ``push_screen``/``pop_screen``, which
    would swallow the delete-confirmation dialog (task-1367); these tests
    need the dialog to really mount so it can be clicked.
    """
    app = _isolated_editor_app(editor)
    app.push_screen = types.MethodType(App.push_screen, app)
    app.pop_screen = types.MethodType(App.pop_screen, app)
    return app


@pytest.mark.asyncio
async def test_settings_theme_editor_delete_blocks_builtin_themes(tmp_path):
    """Built-in themes (Textual defaults) are not deletable."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        for theme_name in ("textual-dark", "textual-light"):
            app.notify.reset_mock()
            editor.current_theme_name = theme_name
            editor.on_delete_theme()
            await pilot.pause()

            message = app.notify.call_args.args[0]
            assert "built-in theme" in message
            assert theme_name in message
            assert not (tmp_path / f"{theme_name}.toml").exists()


@pytest.mark.asyncio
async def test_settings_theme_editor_delete_blocks_shipped_themes(tmp_path):
    """Shipped catalog themes are not deletable and say "shipped", matching
    the tree's own grouping (Built-in / Custom catalog / User Themes)."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        shipped_name = next(t.name for t in ALL_THEMES if hasattr(t, "name"))
        editor.current_theme_name = shipped_name
        editor.on_delete_theme()
        await pilot.pause()

        message = app.notify.call_args.args[0]
        assert "shipped theme" in message
        assert shipped_name in message
        assert not (tmp_path / f"{shipped_name}.toml").exists()


def _write_user_theme(themes_dir, theme_name: str):
    theme_file = themes_dir / f"{theme_name}.toml"
    theme_file.write_text(
        f'[theme]\nname = "{theme_name}"\ndark = true\n'
        '[colors]\nprimary = "#0099FF"\n',
        encoding="utf-8",
    )
    return theme_file


def _user_theme_labels(editor: SettingsThemeEditor) -> set[str]:
    tree = editor.query_one("#settings-theme-tree", Tree)
    for node in tree.root.children:
        if str(node.label) == "User Themes":
            return {str(child.label) for child in node.children}
    return set()


@pytest.mark.asyncio
async def test_settings_theme_editor_delete_removes_custom_theme(tmp_path):
    """A saved custom (user-created) theme file is deletable, but only after
    confirming the dialog (task-1367: irreversible unlink needs a guard)."""
    theme_file = _write_user_theme(tmp_path, "my_custom_theme")
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app_with_real_screens(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        # The tree picked the file up on mount, so the delete also exercises
        # the tree-removal branch.
        assert "user:my_custom_theme" in _user_theme_labels(editor)

        editor.current_theme_name = "my_custom_theme"
        editor.on_delete_theme()
        await pilot.pause()

        # Confirmation dialog is up and NOTHING was deleted yet.
        assert isinstance(app.screen, ConfirmationDialog)
        assert "my_custom_theme" in app.screen.message
        assert theme_file.exists()

        # Cancel keeps the file.
        await pilot.click("#cancel-button")
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmationDialog)
        assert theme_file.exists()
        assert "user:my_custom_theme" in _user_theme_labels(editor)

        # Re-invoke and confirm: only now is the file unlinked.
        editor.on_delete_theme()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationDialog)
        await pilot.click("#confirm-button")
        await pilot.pause()

        assert not theme_file.exists()
        assert "user:my_custom_theme" not in _user_theme_labels(editor)
        assert editor.current_theme_name == "textual-dark"
        message = app.notify.call_args.args[0]
        assert message == "Deleted theme 'my_custom_theme'"


@pytest.mark.asyncio
async def test_settings_theme_editor_delete_user_file_shadowing_shipped_name(tmp_path):
    """A user theme file whose name shadows a shipped catalog theme is still
    deletable: the save guard allows the shadowing save, so the delete guard
    must not strand the file."""
    shipped_name = next(t.name for t in ALL_THEMES if hasattr(t, "name"))
    theme_file = _write_user_theme(tmp_path, shipped_name)
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app_with_real_screens(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        assert f"user:{shipped_name}" in _user_theme_labels(editor)

        editor.current_theme_name = shipped_name
        editor.on_delete_theme()
        await pilot.pause()

        # Same confirmation guard as any user file: nothing is deleted
        # until the dialog is confirmed.
        assert isinstance(app.screen, ConfirmationDialog)
        assert theme_file.exists()
        await pilot.click("#confirm-button")
        await pilot.pause()

        assert not theme_file.exists()
        assert f"user:{shipped_name}" not in _user_theme_labels(editor)
        assert editor.current_theme_name == "textual-dark"
        message = app.notify.call_args.args[0]
        assert message == f"Deleted theme '{shipped_name}'"


@pytest.mark.asyncio
async def test_settings_theme_editor_delete_missing_custom_theme_warns(tmp_path):
    """Deleting a name with no saved custom theme file says so honestly."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        editor.current_theme_name = "never_saved_theme"
        editor.on_delete_theme()
        await pilot.pause()

        message = app.notify.call_args.args[0]
        assert message == "No saved custom theme named 'never_saved_theme'"


@pytest.mark.asyncio
async def test_settings_theme_editor_apply_hint_announces_instant_apply():
    """task-1369/1371: Apply re-themes the whole app instantly; the hint uses
    the Settings screen's instant-apply phrasing
    (INSTANT_APPLY_BEHAVIOR_COPY)."""
    app = _build_test_app()
    editor = SettingsThemeEditor()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.mount(editor)
        await pilot.pause()

        from textual.widgets import Static

        hint = editor.query_one("#settings-theme-apply-hint", Static)
        assert "applies immediately - no Save needed" in str(hint.renderable)


@pytest.mark.asyncio
async def test_settings_theme_editor_preset_swatches_are_keyboard_activatable():
    """task-1369: preset swatches must be focusable and apply on Enter/Space,
    not just on mouse click."""
    app = _build_test_app()
    editor = SettingsThemeEditor()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.app.mount(editor)
        await pilot.pause()

        swatch = editor.query_one("#settings-theme-preset-Blues-0")
        assert swatch.can_focus is True

        editor.is_modified = False
        swatch.focus()
        await pilot.pause()
        assert swatch.has_focus

        await pilot.press("enter")
        await pilot.pause()

        expected = editor.COLOR_PRESETS["Blues"][0]
        assert editor.color_inputs["primary"].value == expected
        assert editor.current_theme_data["primary"] == expected
        assert editor.is_modified is True

        # Space applies too (Greens row, second swatch).
        other = editor.query_one("#settings-theme-preset-Greens-1")
        editor.is_modified = False
        other.focus()
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        assert editor.color_inputs["primary"].value == editor.COLOR_PRESETS["Greens"][1]
        assert editor.is_modified is True


@pytest.mark.asyncio
async def test_settings_theme_editor_reset_without_edits_skips_confirmation(tmp_path):
    """task-1371: Reset with no unsaved edits is lossless, so it runs without
    a confirmation dialog."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app_with_real_screens(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        editor.is_modified = False
        editor.on_reset_theme()
        await pilot.pause()

        assert not isinstance(app.screen, ConfirmationDialog)
        message = app.notify.call_args.args[0]
        assert message == "Theme reset to original values"


@pytest.mark.asyncio
async def test_settings_theme_editor_reset_confirms_before_discarding_edits(tmp_path):
    """task-1371: Reset discards unapplied edits, so it follows the Settings
    screen's revert rule (ADR-031): confirm first, cancel keeps the edits."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app_with_real_screens(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        editor.current_theme_data["primary"] = "#123456"
        editor.is_modified = True
        editor.on_reset_theme()
        await pilot.pause()

        # Dialog is up; edits untouched.
        assert isinstance(app.screen, ConfirmationDialog)
        assert app.screen.confirm_label == "Discard changes"
        assert editor.current_theme_data["primary"] == "#123456"
        assert editor.is_modified is True

        # Cancel keeps the edits.
        await pilot.click("#cancel-button")
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmationDialog)
        assert editor.current_theme_data["primary"] == "#123456"
        assert editor.is_modified is True

        # Confirm discards: palette reloads and modified clears.
        editor.current_theme_data["primary"] = "#123456"
        editor.is_modified = True
        editor.on_reset_theme()
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()

        assert editor.current_theme_data["primary"] != "#123456"
        assert editor.is_modified is False
        message = app.notify.call_args.args[0]
        assert message == "Theme reset to original values"


@pytest.mark.asyncio
async def test_settings_theme_editor_new_confirms_before_discarding_edits(tmp_path):
    """task-1371: New replaces the working palette, so it follows the same
    discard confirmation rule as Reset; unmodified editors skip the dialog."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app_with_real_screens(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        # Unmodified: no dialog, straight to the new theme.
        editor.is_modified = False
        editor.on_new_theme()
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmationDialog)
        assert editor.current_theme_name == "new_theme"

        # Modified: confirmation guard first. Pause after load_theme so the
        # async Input.Changed events from the reload flush before editing.
        editor.load_theme("textual-dark")
        await pilot.pause()
        editor.current_theme_data["primary"] = "#123456"
        editor.is_modified = True
        editor.on_new_theme()
        await pilot.pause()

        assert isinstance(app.screen, ConfirmationDialog)
        assert editor.current_theme_data["primary"] == "#123456"

        await pilot.click("#cancel-button")
        await pilot.pause()
        assert editor.current_theme_name == "textual-dark"
        assert editor.current_theme_data["primary"] == "#123456"

        editor.on_new_theme()
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()
        assert editor.current_theme_name == "new_theme"
        message = app.notify.call_args.args[0]
        assert message == "Creating new theme"


@pytest.mark.asyncio
async def test_settings_theme_editor_name_box_drives_apply_save_reset_delete(tmp_path):
    """TASK-31251: New -> rename -> Apply/Save/Reset/Delete all use the typed name."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app_with_real_screens(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.on_new_theme()
        await pilot.pause()
        name_input = editor.query_one("#settings-theme-name", Input)
        name_input.value = "ocean"
        await pilot.pause()
        assert editor.current_theme_name == "ocean"

        editor.on_apply_theme()
        await pilot.pause()
        assert app.notify.call_args.args[0] == "Theme 'ocean' applied"

        editor.on_save_theme()
        await pilot.pause()
        assert (tmp_path / "ocean.toml").exists()
        assert editor.current_theme_name == "ocean"

        editor.on_reset_theme()
        await pilot.pause()
        assert name_input.value == "ocean"

        editor.on_delete_theme()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationDialog)
        assert "ocean" in app.screen.message


@pytest.mark.asyncio
async def test_settings_theme_editor_selecting_builtin_leaf_does_not_retheme_app(tmp_path):
    """TASK-31255: browsing the tree is read-only for the running app, and the
    palette comes from the real registered Theme, not a hardcoded table."""
    from textual.color import Color

    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.theme = "textual-light"
        await pilot.pause()
        editor.load_theme("textual-dark")
        await pilot.pause()
        assert app.theme == "textual-light"
        resolved = app.available_themes["textual-dark"].to_color_system().generate()
        for key in ("background", "secondary", "panel"):
            assert editor.color_inputs[key].value.upper() == Color.parse(
                resolved[key]
            ).hex.upper(), key


@pytest.mark.asyncio
async def test_settings_theme_editor_delete_keeps_app_theme(tmp_path):
    """TASK-31255: deleting a saved theme resets the editor, not the app theme."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    _write_user_theme(tmp_path, "my_custom_theme")
    app = _isolated_editor_app_with_real_screens(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.theme = "textual-light"
        editor.load_user_theme("my_custom_theme")
        await pilot.pause()
        editor.on_delete_theme()
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()
        assert app.theme == "textual-light"
        assert editor.current_theme_name == "textual-dark"


@pytest.mark.asyncio
async def test_settings_theme_editor_save_registers_theme_with_app(tmp_path):
    """TASK-31250: Save registers the theme so Appearance/palette can offer it."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.on_new_theme()
        await pilot.pause()
        editor.query_one("#settings-theme-name", Input).value = "ocean"
        await pilot.pause()
        editor.on_save_theme()
        await pilot.pause()
        assert "ocean" in app.available_themes


@pytest.mark.asyncio
async def test_settings_theme_editor_set_launch_default_requires_saved_theme(
    tmp_path, monkeypatch
):
    """TASK-31250: unsaved -> warning; saved -> general.default_theme written."""
    import tldw_chatbook.config as config_module

    written: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        config_module,
        "save_setting_to_cli_config",
        lambda section, key, value: written.append((section, key, value)) or True,
    )
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.on_new_theme()
        await pilot.pause()
        editor.query_one("#settings-theme-name", Input).value = "ocean"
        await pilot.pause()
        editor.on_set_launch_default()
        await pilot.pause()
        assert "Save the theme first" in app.notify.call_args.args[0]
        assert written == []

        editor.on_save_theme()
        await pilot.pause()
        editor.on_set_launch_default()
        await pilot.pause()
        assert written == [("general", "default_theme", "ocean")]


@pytest.mark.asyncio
async def test_settings_theme_editor_remount_after_apply_restores_palette(tmp_path):
    """TASK-31252: app.theme == 'custom_<name>' must load, not blank the editor."""
    editor = SettingsThemeEditor()
    editor.custom_themes_path = tmp_path
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.on_new_theme()
        await pilot.pause()
        editor.color_inputs["primary"].value = "#123456"
        await pilot.pause()
        editor.on_apply_theme()
        await pilot.pause()
        assert app.theme == "custom_new_theme"

        editor._initialize_editor()  # what a remount does
        await pilot.pause()
        assert editor.current_theme_name == "new_theme"
        assert editor.query_one("#settings-theme-name", Input).value == "new_theme"
        assert editor.color_inputs["primary"].value.upper() == "#123456"
        assert editor.color_inputs["background"].value != ""
