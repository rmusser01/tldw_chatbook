"""TASK-36457: a completed Save must not conceal a newer live theme draft."""

import asyncio
import threading

import pytest
import toml
from textual.widgets import Checkbox, Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_theme_file_actions_off_thread import _settle, _themes_dir
from Tests.UI.test_settings_theme_picker_screen import _highlight, _host, _saved_theme
from Tests.UI.theme_editor_helpers import FULL_SCREEN_SIZE
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.settings_theme_editor import SettingsThemeEditor


async def _until(predicate):
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(0.01)


async def _open(host, pilot, name):
    await _highlight(host, pilot, name)
    await _settle(host, pilot)
    pane = host.screen.query_one("#settings-theme-pane")
    edit = pane.query_one("#settings-theme-picker-edit")
    assert not edit.disabled and edit.display
    edit.press()
    await _until(lambda: pane.current == "settings-theme-editor-view")
    await _settle(host, pilot)
    return pane, pane._editor()


async def _type(pilot, field, value):
    assert not field.disabled and field.display
    field.focus()
    await pilot.pause()
    await pilot.press("home", "shift+end")
    await pilot.press(*value)
    assert field.value == value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "later", ["none", "palette", "name", "dark", "invalid", "empty_name"]
)
@private_profile_test
async def test_pending_save_preserves_later_draft_and_next_save(
    request, monkeypatch, later
):
    host = _host()
    path = _saved_theme(host, "pending")
    async with host.run_test(size=FULL_SCREEN_SIZE) as pilot:
        pane, editor = await _open(host, pilot, "pending")
        color = editor.query_one("#settings-theme-color-primary", Input)
        name = editor.query_one("#settings-theme-name", Input)
        dark = editor.query_one("#settings-theme-dark-mode", Checkbox)
        await _type(pilot, color, "#112233")
        await _settle(host, pilot)
        entered, release = threading.Event(), threading.Event()
        real_write = SettingsThemeEditor._write_toml

        def held(self, target, data):
            entered.set()
            assert release.wait(30), "write gate not released"
            return real_write(self, target, data)

        monkeypatch.setattr(SettingsThemeEditor, "_write_toml", held)
        editor.on_save_theme()
        try:
            await _until(entered.is_set)
            if later in ("palette", "invalid"):
                await _type(pilot, color, "#AABBCC" if later == "palette" else "#bad")
            elif later in ("name", "empty_name"):
                await _type(pilot, name, "renamed" if later == "name" else " ")
            elif later == "dark":
                assert not dark.disabled
                dark.focus()
                await pilot.press("space")
                assert dark.value is False
            before = (name.value, dark.value, color.value)
        finally:
            release.set()
        await _settle(host, pilot)
        saved = toml.loads(path.read_text(encoding="utf-8"))
        assert saved["colors"]["primary"] == "#112233"
        assert saved["theme"] == {"name": "pending", "dark": True}
        assert editor._loaded_user_theme == "pending"
        if later == "none":
            assert pane.current == "settings-theme-picker" and not editor.is_modified
            await _open(host, pilot, "pending")
            assert color.value == "#112233" and not editor.is_modified
            return
        assert pane.current == "settings-theme-editor-view"
        assert (name.value, dark.value, color.value) == before
        assert editor.is_modified
        if later in ("invalid", "empty_name"):
            return  # Invalid raw text remains available to correct.
        monkeypatch.setattr(SettingsThemeEditor, "_write_toml", real_write)
        editor.on_save_theme()
        await _settle(host, pilot)
        target = path.parent / ("renamed.toml" if later == "name" else "pending.toml")
        saved = toml.loads(target.read_text(encoding="utf-8"))
        assert saved["theme"]["name"] == ("renamed" if later == "name" else "pending")
        assert saved["theme"]["dark"] is (later != "dark")
        assert saved["colors"]["primary"] == (
            "#AABBCC" if later == "palette" else "#112233"
        )
        assert pane.current == "settings-theme-picker" and not editor.is_modified
        await _open(host, pilot, saved["theme"]["name"])
        assert color.value == saved["colors"]["primary"] and not editor.is_modified


@pytest.mark.asyncio
@pytest.mark.parametrize("later", ["palette", "name", "dark"])
@private_profile_test
async def test_pending_save_observes_live_controls_before_queued_changed(
    request, monkeypatch, later
):
    host = _host()
    path = _saved_theme(host, "queued")
    async with host.run_test(size=FULL_SCREEN_SIZE) as pilot:
        pane, editor = await _open(host, pilot, "queued")
        color = editor.query_one("#settings-theme-color-primary", Input)
        color.value = "#112233"
        await _settle(host, pilot)
        entered, release = threading.Event(), threading.Event()
        real_write = SettingsThemeEditor._write_toml

        def held(self, target, data):
            entered.set()
            assert release.wait(30)
            return real_write(self, target, data)

        monkeypatch.setattr(SettingsThemeEditor, "_write_toml", held)
        editor.on_save_theme()
        await _until(entered.is_set)
        pump_entered, pump_release = asyncio.Event(), asyncio.Event()

        async def hold_editor_pump():
            pump_entered.set()
            await pump_release.wait()

        editor.call_later(hold_editor_pump)
        await _until(pump_entered.is_set)
        try:
            field = editor.query_one(
                "#settings-theme-dark-mode"
                if later == "dark"
                else "#settings-theme-name"
                if later == "name"
                else "#settings-theme-color-primary"
            )
            assert not field.disabled
            if later == "dark":
                field.value = False
                assert field.value is False and editor.is_dark_theme is True
            else:
                value = "queued_new" if later == "name" else "#AABBCC"
                field.replace(value, 0, len(field.value))
                assert field.value == value
                assert (
                    editor.current_theme_name
                    if later == "name"
                    else editor.current_theme_data["primary"]
                ) != value
            release.set()
            await editor.save_settled()
            assert path.exists()
        finally:
            release.set()
            pump_release.set()
        await _settle(host, pilot)
        assert pane.current == "settings-theme-editor-view"
        assert editor.is_modified
        assert (
            toml.loads(path.read_text(encoding="utf-8"))["colors"]["primary"]
            == "#112233"
        )
        assert field.value == (
            False if later == "dark" else "queued_new" if later == "name" else "#AABBCC"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("later_name", [False, True])
@private_profile_test
async def test_pending_save_as_retains_newer_draft_and_saved_identity(
    request, monkeypatch, later_name
):
    host = _host()
    source = _saved_theme(host, "source")
    other = _saved_theme(host, "occupied")
    other = other.rename(other.with_name("existing-target.toml"))
    original, occupied = source.read_bytes(), other.read_bytes()
    async with host.run_test(size=FULL_SCREEN_SIZE) as pilot:
        pane, editor = await _open(host, pilot, "source")
        color = editor.query_one("#settings-theme-color-primary", Input)
        color.value = "#112233"
        await _settle(host, pilot)
        entered, release = threading.Event(), threading.Event()
        real_write = SettingsThemeEditor._write_toml

        def held(self, target, data):
            entered.set()
            assert release.wait(30)
            return real_write(self, target, data)

        monkeypatch.setattr(SettingsThemeEditor, "_write_toml", held)
        host.screen._handle_theme_save_as_result("saved_copy")
        try:
            await _until(entered.is_set)
            await _type(pilot, color, "#AABBCC")
            if later_name:
                await _type(
                    pilot, editor.query_one("#settings-theme-name", Input), "occupied"
                )
        finally:
            release.set()
        await _settle(host, pilot)
        assert pane.current == "settings-theme-editor-view" and editor.is_modified
        assert editor._loaded_user_theme == "saved_copy"
        assert editor.current_theme_name == ("occupied" if later_name else "saved_copy")
        assert source.read_bytes() == original and other.read_bytes() == occupied
        assert (
            toml.loads((_themes_dir() / "saved_copy.toml").read_text(encoding="utf-8"))[
                "colors"
            ]["primary"]
            == "#112233"
        )
        monkeypatch.setattr(SettingsThemeEditor, "_write_toml", real_write)
        editor.on_save_theme()
        await _settle(host, pilot)
        if later_name:
            assert isinstance(host.screen, ConfirmationDialog)
            assert other.read_bytes() == occupied
            await pilot.click("#confirm-button")
            await _settle(host, pilot)
        target = other if later_name else _themes_dir() / "saved_copy.toml"
        assert (
            toml.loads(target.read_text(encoding="utf-8"))["colors"]["primary"]
            == "#AABBCC"
        )
        assert pane.current == "settings-theme-picker" and not editor.is_modified
        await _open(host, pilot, "occupied" if later_name else "saved_copy")
        assert color.value == "#AABBCC" and not editor.is_modified


@pytest.mark.asyncio
@private_profile_test
async def test_failed_pending_save_retains_newer_draft_and_original_file(
    request, monkeypatch
):
    host = _host()
    path = _saved_theme(host, "failure")
    original = path.read_bytes()
    async with host.run_test(size=FULL_SCREEN_SIZE) as pilot:
        pane, editor = await _open(host, pilot, "failure")
        color = editor.query_one("#settings-theme-color-primary", Input)
        await _type(pilot, color, "#112233")
        entered, release = threading.Event(), threading.Event()

        def failed(self, target, data):
            entered.set()
            assert release.wait(30)
            raise OSError("definite test write failure")

        monkeypatch.setattr(SettingsThemeEditor, "_write_toml", failed)
        editor.on_save_theme()
        try:
            await _until(entered.is_set)
            await _type(pilot, color, "#AABBCC")
        finally:
            release.set()
        await _settle(host, pilot)
        assert pane.current == "settings-theme-editor-view" and editor.is_modified
        assert color.value == "#AABBCC" and path.read_bytes() == original
        assert editor._loaded_user_theme == "failure"


@pytest.mark.asyncio
@private_profile_test
async def test_saved_publication_preserves_raw_edit_before_changed_is_delivered(
    request, monkeypatch
):
    from tldw_chatbook.Widgets.settings_theme_picker import ThemePane

    host = _host()
    path = _saved_theme(host, "handoff")
    async with host.run_test(size=FULL_SCREEN_SIZE) as pilot:
        pane, editor = await _open(host, pilot, "handoff")
        color = editor.query_one("#settings-theme-color-primary", Input)
        await _type(pilot, color, "#112233")
        await _settle(host, pilot)
        held = []
        published = asyncio.Event()
        real_post = ThemePane.post_message

        def hold_saved(self, message):
            if isinstance(message, SettingsThemeEditor.Saved):
                held.append(message)
                published.set()
                return True
            return real_post(self, message)

        monkeypatch.setattr(ThemePane, "post_message", hold_saved)
        editor.on_save_theme()
        await _until(published.is_set)
        await _settle(host, pilot)
        assert not editor.is_modified
        assert (
            toml.loads(path.read_text(encoding="utf-8"))["colors"]["primary"]
            == "#112233"
        )
        pump_entered, pump_release = asyncio.Event(), asyncio.Event()

        async def hold_editor_pump():
            pump_entered.set()
            await pump_release.wait()

        editor.call_later(hold_editor_pump)
        await _until(pump_entered.is_set)
        try:
            assert not color.disabled and color.display
            color.replace("#AABBCC", 0, len(color.value))
            assert color.value == "#AABBCC"
            assert editor.current_theme_data["primary"] == "#112233"
            assert not editor.is_modified
            consumed = asyncio.Event()
            real_post(pane, held.pop())
            pane.call_later(consumed.set)
            await _until(consumed.is_set)
            assert pane.current == "settings-theme-editor-view"
            assert editor.is_modified
        finally:
            pump_release.set()
        await _settle(host, pilot)
        assert color.value == "#AABBCC" and editor.is_modified
