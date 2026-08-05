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
3. A context with nothing saved leaves the caller-passed ``location=``
   untouched (i.e. the override only fires when a saved value exists).

Ordering caveat (see ``EnhancedFileDialog.__init__``): a saved last-dir
OVERRIDES a caller-passed ``location=``. The tests below therefore never
pass a conflicting ``location=`` for a context that also has a saved value
-- that would only prove the override still happens, not test anything new.
"""

from pathlib import Path

import pytest

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

    monkeypatch.setattr(efp, "get_cli_setting", fake_get)
    monkeypatch.setattr(efp, "save_setting_to_cli_config", fake_save)
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
