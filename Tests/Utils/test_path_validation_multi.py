"""Multi-root path validation (spec 2026-07-26 settings-workspaces §3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Utils.path_validation import validate_path_multi


def test_accepts_path_inside_any_root(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    (root_b / "sub").mkdir(parents=True)
    root_a.mkdir()
    target = root_b / "sub" / "file.txt"
    target.write_text("x")

    assert validate_path_multi(target, [root_a, root_b]) == target.resolve()


def test_relative_paths_resolve_against_first_root(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "note.txt").write_text("x")

    resolved = validate_path_multi("note.txt", [sandbox, tmp_path / "other"])
    assert resolved == (sandbox / "note.txt").resolve()


def test_rejection_names_all_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")

    with pytest.raises(ValueError) as excinfo:
        validate_path_multi(outside, [root_a, root_b])
    message = str(excinfo.value)
    assert str(root_a) in message and str(root_b) in message


def test_empty_roots_rejected() -> None:
    with pytest.raises(ValueError):
        validate_path_multi("anything", [])
