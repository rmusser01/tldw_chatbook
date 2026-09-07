"""Regression test: file picker per-context start directory (task-431 AC#3).

``EnhancedFileDialog.__init__`` computes
``self._last_directory = self._get_last_directory()`` and, when a saved
value exists for the picker's ``context``, overrides the caller-supplied
``location=`` with it (see ``_get_last_directory``/``_save_last_directory``,
config key ``filepicker.last_dir_{context}``). Each ``context`` is keyed
independently -- there is no shared/global "last directory".

These tests prove:

1. Two different contexts keep independent last-dirs (a saved
   ``last_dir_character_import`` does not leak into a ``chat_images``
   picker, and vice versa).
2. A saved ``filepicker.last_dir_{context}`` value is used as the resolved
   start ``_location`` when a picker of that context opens.
3. A new character import starts at Documents when available, then home,
   and never inherits the process working directory.
4. An explicit caller location remains authoritative even when that context
   has a remembered directory.
"""

import pytest
from textual.app import App

from tldw_chatbook.Widgets import enhanced_file_picker as efp
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen


@pytest.fixture
def config_store(monkeypatch):
    """In-memory (section, key) -> value store standing in for the TOML config.

    Mirrors the monkeypatch pattern used in
    ``tests/UI/test_file_picker_bookmarks_lazy.py``: patch the
    ``get_cli_setting``/``save_setting_to_cli_config`` names as imported into
    the ``enhanced_file_picker`` module (not the ``config`` module itself),
    so the dialog's calls are intercepted without touching the real
    ``~/.config/tldw_cli/config.toml``.
    """
    store: dict[tuple[str, str], object] = {}

    def fake_get(section, key, default=None):
        return store.get((section, key), default)

    def fake_save(section, key, value):
        store[(section, key)] = value

    def fake_save_settings(section_values):
        for section, values in section_values.items():
            for key, value in values.items():
                store[(section, key)] = value

    monkeypatch.setattr(efp, "get_cli_setting", fake_get)
    monkeypatch.setattr(efp, "save_setting_to_cli_config", fake_save)
    monkeypatch.setattr(efp, "save_settings_to_cli_config", fake_save_settings)
    return store


def test_last_dir_is_per_context_and_independent(config_store, tmp_path):
    """A saved dir for one context must not leak into another context."""
    dir_character_import = tmp_path / "character_import_dir"
    dir_chat_images = tmp_path / "chat_images_dir"
    dir_character_import.mkdir()
    dir_chat_images.mkdir()

    config_store[("filepicker", "last_dir_character_import")] = str(
        dir_character_import
    )
    config_store[("filepicker", "last_dir_chat_images")] = str(dir_chat_images)

    character_picker = EnhancedFileOpen(context="character_import")
    images_picker = EnhancedFileOpen(context="chat_images")

    assert character_picker._location == dir_character_import
    assert images_picker._location == dir_chat_images
    assert character_picker._location != images_picker._location, (
        "the character_import last-dir leaked into the chat_images picker"
    )


def test_saved_last_dir_is_used_as_start_location(config_store, tmp_path):
    """A saved last-dir for a context is used as the picker's start _location."""
    saved_dir = tmp_path / "saved_start_dir"
    saved_dir.mkdir()
    config_store[("filepicker", "last_dir_notes_export")] = str(saved_dir)

    picker = EnhancedFileOpen(context="notes_export")

    assert picker._location == saved_dir


def test_context_without_saved_dir_keeps_caller_location(config_store):
    """No saved value for the context -> the caller-passed location= wins."""
    picker = EnhancedFileOpen(location="/tmp", context="brand_new_context")

    assert picker._location == "/tmp"


def test_character_import_start_precedence(tmp_path):
    home = tmp_path / "home"
    documents = home / "Documents"
    remembered = home / "cards"
    documents.mkdir(parents=True)
    remembered.mkdir()

    assert (
        efp.resolve_file_picker_start(
            "character_import", remembered, home=home
        )
        == remembered
    )
    assert (
        efp.resolve_file_picker_start("character_import", None, home=home)
        == documents
    )

    documents.rmdir()
    assert efp.resolve_file_picker_start("character_import", None, home=home) == home


def test_character_import_ignores_invalid_remembered_directory(tmp_path):
    home = tmp_path / "home"
    documents = home / "Documents"
    documents.mkdir(parents=True)

    assert (
        efp.resolve_file_picker_start(
            "character_import", home / "missing-cards", home=home
        )
        == documents
    )


def test_explicit_location_overrides_remembered_directory(config_store, tmp_path):
    explicit = tmp_path / "explicit"
    remembered = tmp_path / "remembered"
    explicit.mkdir()
    remembered.mkdir()
    config_store[("filepicker", "last_dir_character_import")] = str(remembered)

    picker = EnhancedFileOpen(location=explicit, context="character_import")

    assert picker._location == explicit


@pytest.mark.asyncio
async def test_character_import_selection_saves_only_selected_parent(
    config_store, tmp_path
):
    selected_parent = tmp_path / "selected-parent"
    selected_parent.mkdir()
    selected_file = selected_parent / "character.json"
    selected_file.write_text("{}", encoding="utf-8")
    other_directory = tmp_path / "other-directory"
    other_directory.mkdir()
    picker = EnhancedFileOpen(location=other_directory, context="character_import")
    app = App()

    async with app.run_test() as pilot:
        await app.push_screen(picker)
        await pilot.pause()
        picker.dismiss(selected_file)
        await pilot.pause()

    assert config_store[("filepicker", "last_dir_character_import")] == str(
        selected_parent
    )


@pytest.mark.asyncio
async def test_character_import_cancel_keeps_remembered_directory(
    config_store, tmp_path
):
    remembered = tmp_path / "remembered"
    current = tmp_path / "current"
    remembered.mkdir()
    current.mkdir()
    config_store[("filepicker", "last_dir_character_import")] = str(remembered)
    picker = EnhancedFileOpen(location=current, context="character_import")
    app = App()

    async with app.run_test() as pilot:
        await app.push_screen(picker)
        await pilot.pause()
        picker.dismiss(None)
        await pilot.pause()

    assert config_store[("filepicker", "last_dir_character_import")] == str(
        remembered
    )


@pytest.mark.asyncio
async def test_character_import_directory_result_keeps_remembered_directory(
    config_store, tmp_path
):
    remembered = tmp_path / "remembered"
    selected_directory = tmp_path / "selected-directory"
    remembered.mkdir()
    selected_directory.mkdir()
    config_store[("filepicker", "last_dir_character_import")] = str(remembered)
    picker = EnhancedFileOpen(
        location=selected_directory,
        context="character_import",
    )
    app = App()

    async with app.run_test() as pilot:
        await app.push_screen(picker)
        await pilot.pause()
        picker.dismiss(selected_directory)
        await pilot.pause()

    assert config_store[("filepicker", "last_dir_character_import")] == str(
        remembered
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("result_kind", ["empty_list", "missing_file"])
async def test_character_import_invalid_result_keeps_remembered_directory(
    config_store, tmp_path, result_kind
):
    remembered = tmp_path / "remembered"
    current = tmp_path / "current"
    remembered.mkdir()
    current.mkdir()
    config_store[("filepicker", "last_dir_character_import")] = str(remembered)
    picker = EnhancedFileOpen(location=current, context="character_import")
    result = [] if result_kind == "empty_list" else current / "missing.json"
    app = App()

    async with app.run_test() as pilot:
        await app.push_screen(picker)
        await pilot.pause()
        picker.dismiss(result)
        await pilot.pause()

    assert config_store[("filepicker", "last_dir_character_import")] == str(
        remembered
    )
