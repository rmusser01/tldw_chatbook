"""File Notes delete-safety completion (task-34383).

Delete/restore landed minimal: restore refused an occupied path with a bare
``exists`` and a missing parent with a raw ENOENT ``error``. These tests pin
both refusals as typed results with clear reasons, and pin that the existing
exact-export read-path (task-34381) serves as the fallback action for both
shapes -- the deleted bytes can still be written to a new, absent path.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# Avoid importing the unrelated optional MLX stack during focused Notes tests.
sys.modules.setdefault("parakeet_mlx", types.ModuleType("parakeet_mlx"))

from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica  # noqa: E402
from tldw_chatbook.Notes.file_notes_service import FileNotesService  # noqa: E402

DELETED_BYTES = b"exact deleted payload\r\nwith crlf\r\n"


@pytest.fixture
def replica() -> FileNotesReplica:
    value = FileNotesReplica(":memory:")
    yield value
    value.close()


def _deleted_note(tmp_path: Path, replica: FileNotesReplica) -> Path:
    """One tombstoned note whose exact bytes survive in the delete revision."""
    root = tmp_path / "safety-notes"
    root.mkdir()
    (root / "note.md").write_bytes(DELETED_BYTES)
    service = FileNotesService(root, replica)
    opened = service.open_file("note.md")
    assert (
        service.delete_file("note.md", expected_hash=opened.content_hash).status
        == "ok"
    )
    return root


def test_restore_refuses_an_occupied_path_with_a_clear_reason(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = _deleted_note(tmp_path, replica)
    (root / "note.md").write_bytes(b"occupied by something new\n")
    service = FileNotesService(root, replica)

    result = service.restore_file("note.md")

    assert result.status == "exists"
    assert "never replaces" in (result.message or "")
    assert (root / "note.md").read_bytes() == b"occupied by something new\n"


def test_restore_refuses_a_missing_parent_with_a_clear_reason(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    root = tmp_path / "safety-notes"
    root.mkdir()
    nested = root / "gone"
    nested.mkdir()
    (nested / "note.md").write_bytes(DELETED_BYTES)
    service = FileNotesService(root, replica)
    opened = service.open_file("gone/note.md")
    assert (
        service.delete_file(
            "gone/note.md",
            expected_hash=opened.content_hash,
        ).status
        == "ok"
    )
    nested.rmdir()

    result = service.restore_file("gone/note.md")

    assert result.status == "missing"
    assert "parent" in (result.message or "").lower()
    assert not nested.exists()


def test_both_refusals_fall_back_to_exact_export_of_the_deleted_bytes(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> None:
    # Shape 1: occupied destination.
    root = _deleted_note(tmp_path, replica)
    (root / "note.md").write_bytes(b"occupied by something new\n")
    service = FileNotesService(root, replica)
    occupied = service.restore_file("note.md")
    assert occupied.status == "exists"

    exported = service.export_revision_file(
        "note.md",
        "recovered-occupied.md",
        kind="delete",
        session_key=None,
    )
    assert exported.status == "ok"
    assert (root / "recovered-occupied.md").read_bytes() == DELETED_BYTES

    # Shape 2: missing parent.
    nested = root / "gone"
    nested.mkdir()
    (nested / "note.md").write_bytes(DELETED_BYTES)
    opened = service.open_file("gone/note.md")
    assert (
        service.delete_file(
            "gone/note.md",
            expected_hash=opened.content_hash,
        ).status
        == "ok"
    )
    nested.rmdir()
    missing = service.restore_file("gone/note.md")
    assert missing.status == "missing"

    exported_missing = service.export_revision_file(
        "gone/note.md",
        "recovered-missing.md",
        kind="delete",
        session_key=None,
    )
    assert exported_missing.status == "ok"
    assert (root / "recovered-missing.md").read_bytes() == DELETED_BYTES
