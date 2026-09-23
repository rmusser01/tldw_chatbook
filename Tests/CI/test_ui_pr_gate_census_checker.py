"""Exercise the PR-gate census checker's FAILURE paths, not just its happy one.

The committed census passing proves the census is currently valid; it proves
nothing about the checker. A guard whose failure branches have never been
observed to fire is the exact shape this whole review is about -- a check that
cannot fail reads identically to a check that passes.

Each test below drives `main()` through one rejection with a temporary census
and asserts both the exit status and the diagnostic, so a future edit that
silently stops rejecting is caught.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_ui_pr_gate_census.py"


def _checker(tmp_path: Path, lines: list[str], *, floor: int | None = None,
             root: Path | None = None):
    """Load a fresh checker module pointed at a throwaway census.

    Args:
        tmp_path: pytest's temporary directory.
        lines: Raw census lines to write.
        floor: Override MINIMUM_FILES; defaults to the module's own value.
        root: Override REPO_ROOT, so `is_file()` resolves against a fixture tree.

    Returns:
        The imported module, ready for `main()`.
    """
    spec = importlib.util.spec_from_file_location(f"cen_{tmp_path.name}", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # the census must sit UNDER the repo root: the checker reports paths with
    # `relative_to(REPO_ROOT)`, so a census outside it raises ValueError.
    base = root if root is not None else tmp_path
    base.mkdir(parents=True, exist_ok=True)
    census = base / "census.txt"
    census.write_text("\n".join(lines) + "\n", encoding="utf-8")
    module.CENSUS_PATH = census
    if root is not None:
        module.REPO_ROOT = root
    if floor is not None:
        module.MINIMUM_FILES = floor
    return module


def _tree(tmp_path: Path, *rel: str) -> Path:
    root = tmp_path / "repo"
    for r in rel:
        p = root / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
    return root


def test_a_valid_census_passes(tmp_path, capsys):
    """The positive control: without it a checker that always fails looks correct."""
    root = _tree(tmp_path, "Tests/UI/test_a.py", "Tests/UI/test_b.py")
    m = _checker(tmp_path, ["# comment", "", "Tests/UI/test_a.py", "Tests/UI/test_b.py"],
                 floor=2, root=root)
    assert m.main() == 0
    assert "OK:" in capsys.readouterr().out


def test_comments_and_blank_lines_are_not_counted_as_entries(tmp_path):
    """A census padded with comments must not satisfy the floor."""
    root = _tree(tmp_path, "Tests/UI/test_a.py")
    m = _checker(tmp_path, ["# one", "", "  ", "# two", "Tests/UI/test_a.py"],
                 floor=2, root=root)
    assert m.main() == 1


def test_a_duplicate_entry_is_rejected(tmp_path, capsys):
    root = _tree(tmp_path, "Tests/UI/test_a.py")
    m = _checker(tmp_path, ["Tests/UI/test_a.py", "Tests/UI/test_a.py"], floor=1, root=root)
    assert m.main() == 1
    assert "duplicate entry" in capsys.readouterr().err


def test_a_path_outside_tests_ui_is_rejected(tmp_path, capsys):
    root = _tree(tmp_path, "Tests/Chat/test_x.py")
    m = _checker(tmp_path, ["Tests/Chat/test_x.py"], floor=1, root=root)
    assert m.main() == 1
    assert "not a Tests/UI path" in capsys.readouterr().err


def test_a_censused_file_that_no_longer_exists_is_rejected(tmp_path, capsys):
    """The case the gate exists for: a rename silently shrinks what runs."""
    root = _tree(tmp_path, "Tests/UI/test_a.py")
    m = _checker(tmp_path, ["Tests/UI/test_a.py", "Tests/UI/test_renamed_away.py"],
                 floor=1, root=root)
    assert m.main() == 1
    assert "listed file does not exist" in capsys.readouterr().err


def test_a_shrunk_census_is_rejected(tmp_path, capsys):
    root = _tree(tmp_path, "Tests/UI/test_a.py")
    m = _checker(tmp_path, ["Tests/UI/test_a.py"], floor=5, root=root)
    assert m.main() == 1
    assert "census has shrunk" in capsys.readouterr().err


def test_a_missing_census_file_is_rejected(tmp_path, capsys):
    m = _checker(tmp_path, ["Tests/UI/test_a.py"], floor=1, root=_tree(tmp_path))
    m.CENSUS_PATH = m.REPO_ROOT / "gone.txt"
    assert m.main() == 1
    assert "census file is missing" in capsys.readouterr().err


def test_the_committed_census_is_currently_valid():
    """Separately from the above: the real census in this repo passes today."""
    spec = importlib.util.spec_from_file_location("cen_real", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
