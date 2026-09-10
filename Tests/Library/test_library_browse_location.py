"""Contracts for the shared last-used picker directory (task-32174).

PR #2554 review: the remembered directory is persisted user state read back
from ``config.toml``, so it is a trust boundary -- it must be validated
through ``Utils/path_validation.py`` before any filesystem probe or picker
use, and the write for one key must be latest-wins rather than
last-worker-to-finish.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Library import library_browse_location as module
from tldw_chatbook.Library.library_browse_location import (
    claim_browse_directory,
    remember_browse_directory,
    validated_browse_directory,
)


def test_a_real_directory_comes_back_normalized(tmp_path):
    """The happy path: an existing directory survives, fully resolved."""
    assert validated_browse_directory(str(tmp_path)) == tmp_path.resolve()


@pytest.mark.parametrize(
    "remembered",
    [
        None,
        "",
        "../../etc",  # traversal
        "relative/dir",  # not absolute
        "/definitely/not/here/32174",  # gone since it was remembered
        "/tmp/nul\x00byte",  # malformed
        12345,  # not even a path
    ],
)
def test_an_unusable_remembered_value_is_refused(remembered):
    """Every malformed spelling falls back to the caller's own default."""
    assert validated_browse_directory(remembered) is None


def test_a_remembered_file_is_not_offered_as_a_directory(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("# hi", encoding="utf-8")
    assert validated_browse_directory(str(note)) is None


def test_a_superseded_selection_cannot_overwrite_a_newer_one(tmp_path, monkeypatch):
    """Two picks in flight: the newer directory is the one that survives."""
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    older.mkdir()
    newer.mkdir()
    saved: list[tuple] = []
    monkeypatch.setattr(
        module,
        "save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )

    first = claim_browse_directory("library.test_32174", "last_directory")
    second = claim_browse_directory("library.test_32174", "last_directory")
    # The newer worker wins the race, then the older one finishes late.
    remember_browse_directory("library.test_32174", "last_directory", newer, second)
    remember_browse_directory("library.test_32174", "last_directory", older, first)

    assert saved == [("library.test_32174", "last_directory", str(newer))]


def test_a_picked_file_remembers_its_parent_directory(tmp_path, monkeypatch):
    picked = tmp_path / "note.md"
    picked.write_text("# hi", encoding="utf-8")
    saved: list[tuple] = []
    monkeypatch.setattr(
        module,
        "save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )

    generation = claim_browse_directory("library.test_32174_parent", "last_directory")
    remember_browse_directory(
        "library.test_32174_parent", "last_directory", picked, generation
    )

    assert saved == [
        ("library.test_32174_parent", "last_directory", str(tmp_path))
    ]


def test_a_refused_config_write_is_reported_not_swallowed(tmp_path, monkeypatch, caplog):
    """A failed save used to leave no trace at all -- the picker just
    silently reopened somewhere else next time."""
    monkeypatch.setattr(
        module, "save_setting_to_cli_config", lambda *args, **kwargs: False
    )
    logged: list[str] = []
    sink_id = module.logger.add(lambda message: logged.append(str(message)), level="ERROR")
    try:
        generation = claim_browse_directory("library.test_32174_fail", "last_directory")
        remember_browse_directory(
            "library.test_32174_fail", "last_directory", tmp_path, generation
        )
    finally:
        module.logger.remove(sink_id)

    assert any("library.test_32174_fail" in line for line in logged)


def test_a_raising_config_write_is_reported_not_swallowed(tmp_path, monkeypatch):
    """The Folder files worker used to ``except Exception: pass``."""

    def _boom(*args, **kwargs):
        raise RuntimeError("config is on fire")

    monkeypatch.setattr(module, "save_setting_to_cli_config", _boom)
    logged: list[str] = []
    sink_id = module.logger.add(lambda message: logged.append(str(message)), level="ERROR")
    try:
        generation = claim_browse_directory("library.test_32174_boom", "last_directory")
        # Must not propagate: this runs inside a Textual worker.
        remember_browse_directory(
            "library.test_32174_boom", "last_directory", tmp_path, generation
        )
    finally:
        module.logger.remove(sink_id)

    assert any("library.test_32174_boom" in line for line in logged)


def test_home_is_never_used_as_a_silent_stand_in_for_a_real_directory():
    """``validated_browse_directory`` reports refusal; the fallback choice
    stays with each caller (all three use ``Path.home()``)."""
    assert validated_browse_directory(str(Path.home())) == Path.home().resolve()
