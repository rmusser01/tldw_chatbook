"""Dictation exports must follow the active user's data directory.

These tests call the production export methods directly. ``Widget.__init__``
initializes Textual's real reactive state after ``__new__``; a running
``App`` is neither needed nor permitted here. The inherited ``app`` property
cannot be assigned outside an active Textual application, so the narrow
property seam below supplies only the external notification endpoint. The
export directory, file creation, and file contents remain production code.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.widget import Widget

pytestmark = pytest.mark.unit


def _window_with_transcript(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Mock]:
    """Build a real widget state without constructing a Textual App."""
    window_class = module.ImprovedDictationWindow
    window = window_class.__new__(window_class)
    Widget.__init__(window)
    window.transcript_text = "A short active-user dictation transcript."
    window.duration = 12.5
    window.word_count = 6

    notify = Mock()
    app = SimpleNamespace(notify=notify)
    monkeypatch.setattr(window_class, "app", property(lambda _self: app))
    return window, notify


def test_dictation_export_directory_retargets_each_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Changing the active user changes the next export-directory result."""
    import tldw_chatbook.UI.Dictation_Window_Improved as module

    alice = tmp_path / "data" / "alice"
    bob = tmp_path / "data" / "bob"
    monkeypatch.setattr(module, "get_user_data_dir", lambda: alice, raising=False)
    assert module.dictation_export_directory() == alice / "exports" / "dictation"
    monkeypatch.setattr(module, "get_user_data_dir", lambda: bob)
    assert module.dictation_export_directory() == bob / "exports" / "dictation"


def test_export_as_text_creates_one_txt_file_under_active_user_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A text export belongs to the selected user, with its plain content."""
    import tldw_chatbook.UI.Dictation_Window_Improved as module

    active_user = tmp_path / "data" / "alice"
    monkeypatch.setattr(module, "get_user_data_dir", lambda: active_user, raising=False)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path / "legacy-home")
    window, notify = _window_with_transcript(module, monkeypatch)

    window._export_as_text()

    export_dir = active_user / "exports" / "dictation"
    files = list(export_dir.glob("*.txt"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == "A short active-user dictation transcript."
    assert list(export_dir.glob("*")) == files
    notify.assert_called_once()


def test_export_as_markdown_creates_one_md_file_under_active_user_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A Markdown export belongs to the selected user and retains metadata."""
    import tldw_chatbook.UI.Dictation_Window_Improved as module

    active_user = tmp_path / "data" / "bob"
    monkeypatch.setattr(module, "get_user_data_dir", lambda: active_user, raising=False)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path / "legacy-home")
    window, notify = _window_with_transcript(module, monkeypatch)

    window._export_as_markdown()

    export_dir = active_user / "exports" / "dictation"
    files = list(export_dir.glob("*.md"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "# Dictation Transcript" in content
    assert "**Duration:** 12.5 seconds" in content
    assert "**Words:** 6" in content
    assert content.endswith("A short active-user dictation transcript.\n")
    assert list(export_dir.glob("*")) == files
    notify.assert_called_once()
