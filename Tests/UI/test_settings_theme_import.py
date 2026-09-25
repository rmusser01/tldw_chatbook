"""TASK-32948 PR 3 Task 2: Import a theme ``.toml`` from a pasted/dropped path.

Source files live in a NON-profile temp dir (a user's file anywhere on disk);
the import target is the private profile's themes dir (the ``tmp_path``
fixture below). Hostile or broken files are refused with a specific,
path-free reason and nothing is written.
"""

import types
from pathlib import Path

import pytest
import toml
from textual import on
from textual.app import App, ComposeResult

from Tests.private_profile import is_private_profile_child, private_profile_test
from Tests.textual_test_harness import IsolatedWidgetTestApp
from tldw_chatbook.Backup_Recovery import raw_participants
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.css.Themes.themes import ALL_THEMES
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.settings_theme_editor import (
    THEMES_UNAVAILABLE_LABEL,
    SettingsThemeEditor,
)

GOOD = '[theme]\nname = "sunny"\ndark = false\n[colors]\nprimary = "#FFAA00"\nbackground = "#FFFFFF"\n'


@pytest.fixture
def tmp_path(tmp_path_factory, request):
    """The private profile's themes dir (the import target)."""
    if not is_private_profile_child(request):
        return tmp_path_factory.mktemp("theme-profile-launcher")
    from tldw_chatbook import config

    themes = config._get_effective_config_path().parent / "themes"
    themes.mkdir(exist_ok=True)
    return themes


@pytest.fixture
def src(tmp_path_factory) -> Path:
    """A non-profile directory holding the files being imported."""
    return tmp_path_factory.mktemp("import-src")


class _App(IsolatedWidgetTestApp):
    def __init__(self, compose_func):
        super().__init__(compose_func)
        self.themes_changed = 0
        self.push_screen = types.MethodType(App.push_screen, self)
        self.pop_screen = types.MethodType(App.pop_screen, self)

    @on(SettingsThemeEditor.ThemesChanged)
    def _count(self) -> None:
        self.themes_changed += 1


def _app(editor: SettingsThemeEditor) -> _App:
    def compose() -> ComposeResult:
        yield editor

    return _App(compose)


async def _mounted(pilot, app, editor, themes_dir):
    await pilot.pause()
    editor.custom_themes_path = themes_dir
    for theme in ALL_THEMES:
        app.register_theme(theme)


def _snapshot(themes_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in themes_dir.iterdir()}


@pytest.mark.asyncio
@private_profile_test
async def test_valid_import_lands_registers_and_announces(request, tmp_path, src):
    source = src / "whatever.toml"
    source.write_text(GOOD, encoding="utf-8")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        assert editor.import_theme(str(source)) == "sunny"
        await pilot.pause()
        written = toml.loads((tmp_path / "sunny.toml").read_text(encoding="utf-8"))
        assert written["theme"] == {"name": "sunny", "dark": False}
        assert written["colors"]["primary"] == "#FFAA00"
        assert "sunny" in app.available_themes
        assert "sunny" in editor.list_user_theme_names()
        assert app.themes_changed == 1
        message = app.notify.call_args.args[0]
        assert message == "Imported 'sunny'"
        assert str(src) not in message


@pytest.mark.asyncio
@private_profile_test
async def test_import_sanitises_variables(request, tmp_path, src):
    source = src / "vars.toml"
    source.write_text(
        '[theme]\nname = "vars"\n[colors]\nprimary = "#112233"\n'
        '[variables]\ntext-muted = "#AABBCC"\n'
        'evil = "red; } Screen { display: none"\n',
        encoding="utf-8",
    )
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        assert editor.import_theme(str(source)) == "vars"
        written = toml.loads((tmp_path / "vars.toml").read_text(encoding="utf-8"))
        assert written["variables"] == {"text-muted": "#AABBCC"}


HOSTILE = [
    ("notes.txt", GOOD, "Import needs a .toml file"),
    ("big.toml", GOOD + "# " + "x" * (64 * 1024) + "\n", "Theme file is larger than 64 KB"),
    ("broken.toml", "[[[ not toml", "File is not valid TOML"),
    ("noprimary.toml", '[colors]\nbackground = "#000000"\n', "Missing [colors].primary"),
    ("nocolors.toml", '[theme]\nname = "x"\n', "Missing [colors].primary"),
    (
        "badcolour.toml",
        '[colors]\nprimary = "#112233"\nbackground = "notacolour"\n',
        "background: 'notacolour' is not a colour",
    ),
    (
        "markupcolour.toml",
        '[colors]\nprimary = "#112233"\nbackground = "[b]x[/b]"\n',
        "background: '[b]x[/b]' is not a colour",
    ),
    (
        "numbercolour.toml",
        '[colors]\nprimary = "#112233"\nbackground = 7\n',
        "background: '7' is not a colour",
    ),
    ("sep.toml", '[theme]\nname = "../evil"\n[colors]\nprimary = "#112233"\n', "Invalid theme name"),
    ("markup.toml", '[theme]\nname = "x[/]"\n[colors]\nprimary = "#112233"\n', "Invalid theme name"),
    (
        "bold.toml",
        '[theme]\nname = "[bold red]boom"\n[colors]\nprimary = "#112233"\n',
        "Invalid theme name",
    ),
    ("theme-not-table.toml", 'theme = 3\n[colors]\nprimary = "#112233"\n', "[theme] must be a table"),
    # R32: real Theme kwargs that are not colours (crashed generate()).
    (
        "variables-key.toml",
        '[colors]\nprimary = "#112233"\nvariables = "#ffffff"\n',
        "variables is not a theme colour",
    ),
    (
        "dark-key.toml",
        '[colors]\nprimary = "#112233"\ndark = "#000000"\n',
        "dark is not a theme colour",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("filename", "content", "reason"), HOSTILE, ids=[h[0] for h in HOSTILE])
@private_profile_test
async def test_hostile_or_broken_file_is_refused_and_writes_nothing(
    request, tmp_path, src, filename, content, reason
):
    source = src / filename
    source.write_text(content, encoding="utf-8")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        before = _snapshot(tmp_path)
        themes_before = set(app.available_themes)
        assert editor.import_theme(str(source)) is None
        await pilot.pause()
        assert _snapshot(tmp_path) == before
        assert set(app.available_themes) == themes_before
        assert app.themes_changed == 0
        call = app.notify.call_args
        assert call.kwargs["severity"] == "error"
        # notify parses markup: file-derived text arrives escaped.
        from textual.markup import to_content

        rendered = to_content(call.args[0]).plain
        assert reason in rendered
        assert str(src) not in rendered  # R16: never the source path


@pytest.mark.asyncio
@private_profile_test
async def test_missing_and_directory_sources_are_refused_without_path(request, tmp_path, src):
    folder = src / "folder.toml"
    folder.mkdir()
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        before = _snapshot(tmp_path)
        for source in (src / "gone.toml", folder):
            app.notify.reset_mock()
            assert editor.import_theme(str(source)) is None
            message = app.notify.call_args.args[0]
            assert app.notify.call_args.kwargs["severity"] == "error"
            assert str(src) not in message
        assert _snapshot(tmp_path) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("quote", ["'", '"'])
@private_profile_test
async def test_quoted_pasted_path_imports(request, tmp_path, src, quote):
    source = src / "drop.toml"
    source.write_text(GOOD, encoding="utf-8")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        assert editor.import_theme(f"  {quote}{source}{quote}\n") == "sunny"
        assert (tmp_path / "sunny.toml").exists()


@pytest.mark.asyncio
@private_profile_test
async def test_tilde_path_expands(request, tmp_path, src, monkeypatch):
    (src / "home.toml").write_text(GOOD, encoding="utf-8")
    monkeypatch.setenv("HOME", str(src))
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        assert editor.import_theme("~/home.toml") == "sunny"
        assert (tmp_path / "sunny.toml").exists()


@pytest.mark.asyncio
@private_profile_test
async def test_relative_path_is_refused(request, tmp_path, src, monkeypatch):
    (src / "rel.toml").write_text(GOOD, encoding="utf-8")
    monkeypatch.chdir(src)
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        before = _snapshot(tmp_path)
        assert editor.import_theme("rel.toml") is None
        assert app.notify.call_args.kwargs["severity"] == "error"
        assert "full path" in app.notify.call_args.args[0]
        assert _snapshot(tmp_path) == before


@pytest.mark.asyncio
@private_profile_test
async def test_import_onto_existing_name_confirms_and_cancel_keeps_file(request, tmp_path, src):
    existing = tmp_path / "sunny.toml"
    existing.write_text('[theme]\nname = "sunny"\n[colors]\nprimary = "#010203"\n', encoding="utf-8")
    original = existing.read_bytes()
    source = src / "new.toml"
    source.write_text(GOOD, encoding="utf-8")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        assert editor.import_theme(str(source)) is None  # pending confirmation
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationDialog)
        assert app.screen.message == "Replace the saved theme 'sunny'?"
        assert app.screen.confirm_label == "Replace"
        await pilot.click("#cancel-button")
        await pilot.pause()
        assert existing.read_bytes() == original
        assert app.themes_changed == 0

        editor.import_theme(str(source))
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()
        assert toml.loads(existing.read_text(encoding="utf-8"))["colors"]["primary"] == "#FFAA00"
        assert app.themes_changed == 1


@pytest.mark.asyncio
@private_profile_test
async def test_import_replaces_the_file_that_claims_the_name(request, tmp_path, src):
    """R12: a.toml holding name "sunny" is the saved "sunny"; import writes
    back to it rather than creating a second file claiming the same name."""
    claimed = tmp_path / "a.toml"
    claimed.write_text('[theme]\nname = "sunny"\n[colors]\nprimary = "#010203"\n', encoding="utf-8")
    source = src / "new.toml"
    source.write_text(GOOD, encoding="utf-8")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        editor.import_theme(str(source))
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationDialog)
        await pilot.click("#confirm-button")
        await pilot.pause()
        assert not (tmp_path / "sunny.toml").exists()
        assert toml.loads(claimed.read_text(encoding="utf-8"))["colors"]["primary"] == "#FFAA00"


@pytest.mark.asyncio
@private_profile_test
async def test_import_during_pause_writes_nothing(request, tmp_path, src, monkeypatch):
    source = src / "paused.toml"
    source.write_text(GOOD, encoding="utf-8")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        before = _snapshot(tmp_path)

        def paused(*_a, **_k):
            raise RecoveryRequired("x")

        monkeypatch.setattr(raw_participants, "_scope", paused)
        assert editor.import_theme(str(source)) is None
        await pilot.pause()
        assert app.notify.call_args.args[0] == THEMES_UNAVAILABLE_LABEL
        assert _snapshot(tmp_path) == before
        assert "sunny" not in app.available_themes


# -- Fix round 1 (R33-R36) ---------------------------------------------------


@pytest.mark.asyncio
@private_profile_test
async def test_replace_dialog_and_toast_escape_the_name(request, tmp_path, src, monkeypatch):
    """R33: the name goes through escape_markup in the dialog and the toast."""
    from tldw_chatbook.Widgets import settings_theme_editor as module

    monkeypatch.setattr(module, "escape_markup", lambda value: f"<{value}>")
    (tmp_path / "sunny.toml").write_text('[colors]\nprimary = "#010203"\n', encoding="utf-8")
    source = src / "new.toml"
    source.write_text(GOOD, encoding="utf-8")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        editor.import_theme(str(source))
        await pilot.pause()
        assert app.screen.message == "Replace the saved theme '<sunny>'?"
        await pilot.click("#confirm-button")
        await pilot.pause()
        assert app.notify.call_args.args[0] == "Imported '<sunny>'"


@pytest.mark.asyncio
@private_profile_test
async def test_source_is_checked_and_read_through_one_nonblocking_descriptor(
    request, tmp_path, src, monkeypatch
):
    """R34: open O_NONBLOCK, fstat that fd, read that fd -- no window between
    a path check and the read for a FIFO to be swapped in."""
    import os

    from tldw_chatbook.Widgets import settings_theme_editor as module

    source = src / "fd.toml"
    source.write_text(GOOD, encoding="utf-8")
    opened: list[int] = []  # flags of each open of the SOURCE file
    real_open = os.open

    def spy(path, flags, *args, **kwargs):
        if str(path) == str(source):
            opened.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", spy)
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        assert editor.import_theme(str(source)) == "sunny"
    assert opened and all(
        flags & os.O_NONBLOCK and not flags & (os.O_WRONLY | os.O_RDWR) for flags in opened
    ), opened


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["My\\ Theme.toml", "My\\ \\(copy\\).toml"])
@private_profile_test
async def test_backslash_escaped_drop_path_imports(request, tmp_path, src, typed):
    """R35: macOS Terminal/iTerm drop `/a/My\\ Theme.toml` (unquoted)."""
    real_name = typed.replace("\\", "")
    (src / real_name).write_text(GOOD, encoding="utf-8")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        assert editor.import_theme(f"{src}/{typed}") == "sunny"


def _special_sources(src: Path) -> list[tuple[str, str]]:
    """R36: (typed path, expected reason) for special-file sources."""
    import os

    big = src / "big-target.toml"
    big.write_text(GOOD + "# " + "x" * (64 * 1024) + "\n", encoding="utf-8")
    (src / "to-big.toml").symlink_to(big)
    (src / "to-devnull.toml").symlink_to(os.devnull)
    return [
        (str(src / "to-devnull.toml"), "Import needs a regular file"),
        (str(src / "to-big.toml"), "Theme file is larger than 64 KB"),
        (f"{src}/nul\x00.toml", "Import needs the full path to a .toml file"),
    ]


@pytest.mark.asyncio
@private_profile_test
async def test_special_file_sources_are_refused(request, tmp_path, src):
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        before = _snapshot(tmp_path)
        for typed, reason in _special_sources(src):
            app.notify.reset_mock()
            assert editor.import_theme(typed) is None, typed
            assert reason in app.notify.call_args.args[0], typed
            assert str(src) not in app.notify.call_args.args[0]
        assert _snapshot(tmp_path) == before


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="no FIFOs here")
@private_profile_test
async def test_fifo_source_is_refused_without_blocking(request, tmp_path, src):
    import os

    fifo = src / "pipe.toml"
    os.mkfifo(fifo)
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        before = _snapshot(tmp_path)
        assert editor.import_theme(str(fifo)) is None
        assert "Import needs a regular file" in app.notify.call_args.args[0]
        assert _snapshot(tmp_path) == before


@pytest.mark.asyncio
@private_profile_test
async def test_uppercase_suffix_imports(request, tmp_path, src):
    (src / "THEME.TOML").write_text(GOOD, encoding="utf-8")
    editor = SettingsThemeEditor()
    app = _app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await _mounted(pilot, app, editor, tmp_path)
        assert editor.import_theme(str(src / "THEME.TOML")) == "sunny"
