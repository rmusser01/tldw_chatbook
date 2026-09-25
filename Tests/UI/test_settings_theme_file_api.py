"""TASK-32948 PR 2 Task 1: the theme editor's public file API.

list / delete-with-fallback / rename / export-by-name, which the picker calls
through ThemePane. Every test runs in a private profile (the editor writes the
profile's themes/ directory) and mocks the config write.
"""

import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml
from textual import on
from textual.app import App, ComposeResult

from Tests.private_profile import is_private_profile_child, private_profile_test
from Tests.textual_test_harness import IsolatedWidgetTestApp
from tldw_chatbook.Backup_Recovery import raw_participants
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.css.Themes import theme_catalog
from tldw_chatbook.css.Themes.themes import ALL_THEMES, create_theme_from_dict
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.settings_theme_editor import SettingsThemeEditor

MINE = {
    "primary": "#112233",
    "secondary": "#223344",
    "accent": "#334455",
    "background": "#0A0A0A",
    "surface": "#141414",
    "panel": "#1E1E1E",
    "foreground": "#F0F0F0",
    "success": "#00AA00",
    "warning": "#AAAA00",
    "error": "#AA0000",
}


@pytest.fixture
def tmp_path(tmp_path_factory, request):
    """Use the profile selected before imports for this exact test process."""
    if not is_private_profile_child(request):
        return tmp_path_factory.mktemp("theme-profile-launcher")
    from tldw_chatbook import config

    themes = config._get_effective_config_path().parent / "themes"
    themes.mkdir(exist_ok=True)
    return themes


class _App(IsolatedWidgetTestApp):
    def __init__(self, compose_func):
        super().__init__(compose_func)
        self.themes_changed = 0
        # The real screen stack, so the confirmation dialog mounts.
        self.push_screen = types.MethodType(App.push_screen, self)
        self.pop_screen = types.MethodType(App.pop_screen, self)

    @on(SettingsThemeEditor.ThemesChanged)
    def _count(self) -> None:
        self.themes_changed += 1


def _app(editor: SettingsThemeEditor) -> _App:
    def compose() -> ComposeResult:
        yield editor

    return _App(compose)


def _write(themes_dir: Path, stem: str, name: str | None = None, colors=MINE) -> Path:
    path = themes_dir / f"{stem}.toml"
    with path.open("w", encoding="utf-8") as f:
        toml.dump({"theme": {"name": name or stem, "dark": True}, "colors": dict(colors)}, f)
    return path


@pytest.fixture
def config_writes(monkeypatch):
    """Record config writes instead of touching any config file."""
    writes: list[dict] = []

    def record(mutation):
        writes.append(mutation)
        return SimpleNamespace(file_replaced=True, caches_reloaded=True)

    monkeypatch.setattr(theme_catalog, "_apply_config_mutation", record)
    return writes


def _launch_default(monkeypatch, name: str) -> None:
    monkeypatch.setattr(theme_catalog, "current_launch_default", lambda: name)


async def _mounted(pilot, app, editor, tmp_path):
    await pilot.pause()
    editor.custom_themes_path = tmp_path
    for theme in ALL_THEMES:
        app.register_theme(theme)


@pytest.mark.asyncio
@private_profile_test
async def test_list_user_theme_names_reads_theme_name_not_stem(request, tmp_path):
    _write(tmp_path, "a", name="b")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        assert editor.list_user_theme_names() == {"b"}


@pytest.mark.asyncio
@private_profile_test
async def test_list_user_themes_raises_through_on_pause(request, tmp_path, monkeypatch):
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)

        def paused(*_a, **_k):
            raise RecoveryRequired("x")

        monkeypatch.setattr(raw_participants, "_scope", paused)
        with pytest.raises(RecoveryRequired):
            editor.list_user_theme_names()


async def _confirm_delete(pilot, app, editor, name):
    editor.request_delete(name)
    await pilot.pause()
    assert isinstance(app.screen, ConfirmationDialog)
    assert app.screen.confirm_label == "Delete theme"
    await pilot.click("#confirm-button")
    await pilot.pause()


@pytest.mark.asyncio
@private_profile_test
async def test_delete_active_launch_default_falls_back_to_textual_dark(
    request, tmp_path, monkeypatch, config_writes
):
    path = _write(tmp_path, "mine")
    _launch_default(monkeypatch, "mine")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        app.register_theme(create_theme_from_dict("mine", {**MINE, "dark": True}))
        app.theme = "mine"
        await _confirm_delete(pilot, app, editor, "mine")

        assert not path.exists()
        assert "mine" not in app.available_themes
        assert app.theme == "textual-dark"
        assert config_writes[-1] == {"general": {"default_theme": "textual-dark"}}
        assert app.themes_changed >= 1


@pytest.mark.asyncio
@private_profile_test
async def test_delete_active_non_default_switches_to_launch_default(
    request, tmp_path, monkeypatch, config_writes
):
    _write(tmp_path, "mine")
    _launch_default(monkeypatch, "nord")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        app.register_theme(create_theme_from_dict("mine", {**MINE, "dark": True}))
        app.theme = "mine"
        await _confirm_delete(pilot, app, editor, "mine")

        assert app.theme == "nord"
        assert config_writes == []


@pytest.mark.asyncio
@private_profile_test
async def test_rename_moves_file_registration_active_and_launch_default(
    request, tmp_path, monkeypatch, config_writes
):
    _write(tmp_path, "mine")
    _launch_default(monkeypatch, "mine")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        app.register_theme(create_theme_from_dict("mine", {**MINE, "dark": True}))
        app.theme = "mine"

        assert editor.rename_user_theme("mine", "ours") is True
        await pilot.pause()

        assert (tmp_path / "ours.toml").exists()
        assert not (tmp_path / "mine.toml").exists()
        assert toml.load(tmp_path / "ours.toml")["theme"]["name"] == "ours"
        assert "ours" in app.available_themes
        assert "mine" not in app.available_themes
        assert app.theme == "ours"
        assert config_writes[-1] == {"general": {"default_theme": "ours"}}
        assert app.themes_changed >= 1


@pytest.mark.asyncio
@private_profile_test
async def test_rename_to_taken_name_changes_nothing(request, tmp_path, config_writes):
    mine, ours = _write(tmp_path, "mine"), _write(tmp_path, "ours")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        app.notify.reset_mock()
        assert editor.rename_user_theme("mine", "ours") is False
        assert "Name taken" in app.notify.call_args.args[0]
        assert mine.exists() and ours.exists()
        assert toml.load(ours)["theme"]["name"] == "ours"


@pytest.mark.asyncio
@private_profile_test
async def test_rename_rejects_invalid_name(request, tmp_path, config_writes):
    mine = _write(tmp_path, "mine")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        app.notify.reset_mock()
        before = sorted(p.name for p in tmp_path.iterdir())
        assert editor.rename_user_theme("mine", "../evil") is False
        assert app.notify.called
        assert mine.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == before
        assert not (tmp_path.parent / "evil.toml").exists()


@pytest.mark.asyncio
@private_profile_test
async def test_rename_override_restores_catalog_theme(
    request, tmp_path, monkeypatch, config_writes
):
    _write(tmp_path, "apricot")
    _launch_default(monkeypatch, "textual-dark")
    shipped = next(t for t in ALL_THEMES if t.name == "apricot")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        app.register_theme(create_theme_from_dict("apricot", {**MINE, "dark": True}))
        assert editor.rename_user_theme("apricot", "apricot_mine") is True
        assert app.available_themes["apricot"] is shipped
        assert "apricot_mine" in app.available_themes


@pytest.mark.asyncio
@private_profile_test
async def test_export_theme_writes_saved_file_data(request, tmp_path, monkeypatch):
    _write(tmp_path, "mine")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        editor.load_theme("nord")  # a different palette in the editor
        await pilot.pause()
        editor.export_theme("mine")
        await pilot.pause()
        exported = toml.load(tmp_path / "Downloads" / "mine_theme.toml")
        assert exported["colors"] == MINE
        assert exported["theme"]["name"] == "mine"
