"""The editor's "file written" time only ever reads a validated path.

PR #3021 review (TASK-34000.2): ``note_file_written_label`` handed the path
the sync runtime answered with straight to ``os.stat``. Every other Library
boundary that touches the filesystem goes through
``Utils/path_validation.py`` first; this one did not, and a path ``os.stat``
rejects outright (an embedded NUL raises ``ValueError``, which the helper did
not catch) escaped into the worker that paints the editor's location row.

The label is a status line: a path the shared validator refuses yields "no
label", and nothing here may raise into the UI.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tldw_chatbook.UI.Library_Modules import library_notes_sync_attention as attention

#: ``bootstrap_profile`` for the same reason as the sibling listener tests: a
#: Tests/UI file run under the per-test profile redirect trips
#: ``RecoveryRequired: raw_source_selection_changed`` at setup.
pytestmark = [pytest.mark.unit, pytest.mark.bootstrap_profile]

WRITTEN_AT = datetime(2026, 9, 15, 8, 6, tzinfo=UTC).timestamp()

#: Local time in the codebase's absolute-timestamp spelling (task-32640).
WRITTEN_LABEL = (
    datetime.fromtimestamp(WRITTEN_AT, tz=UTC).astimezone().strftime("%Y-%m-%d %H:%M")
)


def _note_file(folder: Path, name: str = "Sam.md") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    note = folder / name
    note.write_text("# Sam\n", encoding="utf-8")
    os.utime(note, (WRITTEN_AT, WRITTEN_AT))
    return note


def test_a_synced_notes_file_is_labelled_with_its_own_write_time(tmp_path) -> None:
    note = _note_file(tmp_path / "vault" / "People")

    assert attention.note_file_written_label(str(note)) == WRITTEN_LABEL


def test_a_file_name_with_shell_punctuation_is_still_labelled(tmp_path) -> None:
    """``;``, ``&&``, ``|`` and ``$(`` are legal in a file name, and a vault
    can hold such a note. This boundary only ever stats the path, so the
    validator's command-injection guards do not apply to it."""
    note = _note_file(tmp_path / "vault", "Q3 P&L; plan && notes | $(draft).md")

    assert attention.note_file_written_label(str(note)) == WRITTEN_LABEL


def test_a_path_the_shared_validator_refuses_gets_no_label(tmp_path) -> None:
    note = _note_file(tmp_path / "vault")
    (tmp_path / "vault" / "a" / "b").mkdir(parents=True)
    traversing = f"{tmp_path}/vault/a/b/../../{note.name}"
    assert os.stat(traversing).st_mtime == WRITTEN_AT, "the fixture must be statable"

    assert attention.note_file_written_label(traversing) == ""


def test_a_path_with_a_nul_byte_gets_no_label_and_never_raises(tmp_path) -> None:
    note = _note_file(tmp_path / "vault")

    assert attention.note_file_written_label(f"{note}\x00.md") == ""


def test_a_relative_path_gets_no_label(tmp_path, monkeypatch) -> None:
    """The runtime's contract is an absolute path. A relative one would be
    read against the process's working directory -- some other file."""
    _note_file(tmp_path / "vault")
    monkeypatch.chdir(tmp_path)
    assert os.stat("vault/Sam.md").st_mtime == WRITTEN_AT

    assert attention.note_file_written_label("vault/Sam.md") == ""


def test_a_missing_file_and_an_empty_path_get_no_label(tmp_path) -> None:
    assert attention.note_file_written_label(str(tmp_path / "vault" / "gone.md")) == ""
    assert attention.note_file_written_label("") == ""


def test_the_path_is_validated_before_it_is_ever_statted(tmp_path, monkeypatch) -> None:
    """Whatever the shared validator refuses is never handed to ``os.stat``."""
    note = _note_file(tmp_path / "vault")
    validated: list[tuple[str, dict]] = []

    def refuse(user_path, *args, **kwargs):
        validated.append((str(user_path), kwargs))
        raise ValueError("refused by the shared validator")

    statted: list[str] = []
    real_stat = os.stat

    def spy_stat(path, *args, **kwargs):
        if not isinstance(path, int):
            statted.append(os.fspath(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(attention, "validate_path_simple", refuse, raising=False)
    monkeypatch.setattr(os, "stat", spy_stat)

    assert attention.note_file_written_label(str(note)) == ""
    assert [path for path, _ in validated] == [str(note)], (
        "the path never went through Utils/path_validation.py"
    )
    assert str(note) not in statted, "a refused path was statted anyway"
