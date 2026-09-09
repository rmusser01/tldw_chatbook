"""Direct unit tests for the shared picker-path resolvers in ``base_dialog.py``.

task-32122 PR #2542 review (Qodo finding 3, "Picker rules lack isolated
tests"): ``resolve_typed_directory``/``resolve_default_location`` were only
reachable through three full Textual dialogs' Enter/Select handlers.
These call the functions directly so a regression in trimming, relative-path
bases, exception conversion, or unchanged-path identity fails here instead
of only showing up as a dialog-level integration failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import (
    FileSystemPickerScreen,
    resolve_default_location,
    resolve_typed_directory,
)


# ---------------------------------------------------------------------------
# resolve_typed_directory
# ---------------------------------------------------------------------------


def test_empty_value_returns_current_directory_object(tmp_path):
    result = resolve_typed_directory("", tmp_path)
    assert result is tmp_path


def test_unchanged_value_returns_current_directory_object_not_a_reresolve(tmp_path):
    # Regression guard (task-2 self-review): a plain, no-typing Select must
    # return the exact `current` object, not `Path(str(current)).resolve()`
    # -- on macOS "/tmp" resolves to "/private/tmp", which would otherwise
    # silently change a no-op Select's result.
    result = resolve_typed_directory(str(tmp_path), tmp_path)
    assert result is tmp_path


def test_absolute_typed_path_resolves_to_that_directory(tmp_path):
    target = tmp_path / "vault"
    target.mkdir()
    result = resolve_typed_directory(str(target), tmp_path)
    assert result == target.resolve()


def test_relative_typed_path_resolves_against_current_directory(tmp_path):
    (tmp_path / "sub").mkdir()
    result = resolve_typed_directory("sub", tmp_path)
    assert result == (tmp_path / "sub").resolve()


def test_home_expansion_is_preserved(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    # ``Path.expanduser()`` reads ``$HOME`` (via ``os.path.expanduser``), not
    # ``Path.home()`` -- patch the environment variable, not the classmethod.
    monkeypatch.setenv("HOME", str(fake_home))
    result = resolve_typed_directory("~", tmp_path)
    assert result == fake_home.resolve()


def test_nul_byte_is_rejected_with_a_friendly_error(tmp_path):
    result = resolve_typed_directory("bad\x00path", tmp_path)
    assert isinstance(result, str)
    assert "null" in result.lower()


def test_missing_path_reports_not_found(tmp_path):
    result = resolve_typed_directory(str(tmp_path / "nope"), tmp_path)
    assert isinstance(result, str)
    assert "not found" in result.lower()


def test_existing_file_is_reported_as_not_a_directory(tmp_path):
    a_file = tmp_path / "notes.md"
    a_file.write_text("hi")
    result = resolve_typed_directory(str(a_file), tmp_path)
    assert isinstance(result, str)
    assert "not a directory" in result.lower()


def test_permission_error_is_reported_without_a_raw_errno(tmp_path, monkeypatch):
    def _boom(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "resolve", _boom)
    result = resolve_typed_directory(str(tmp_path / "locked"), tmp_path)
    assert result == FileSystemPickerScreen.ERROR_PERMISSION_ERROR
    assert "errno" not in result.lower()
    assert "13" not in result


def test_folder_name_with_significant_edge_spaces_is_preserved(tmp_path):
    # Qodo finding 5 (review round 4): a POSIX folder can legitimately be
    # named with leading/trailing spaces. Stripping the typed value before
    # comparing/resolving it silently changes what gets selected.
    padded = tmp_path / " padded name "
    padded.mkdir()
    result = resolve_typed_directory(str(padded), tmp_path)
    assert result == padded.resolve()


def test_whitespace_only_value_is_treated_as_untyped(tmp_path):
    # A field the user only put spaces into is still "nothing typed",
    # matching the always-empty-string case above -- just via a different
    # spelling of "empty".
    result = resolve_typed_directory("   ", tmp_path)
    assert result is tmp_path


# ---------------------------------------------------------------------------
# resolve_default_location
# ---------------------------------------------------------------------------


def test_bare_dot_default_becomes_home():
    assert resolve_default_location(".") == Path.home()


def test_explicit_location_passes_through_unchanged(tmp_path):
    assert resolve_default_location(tmp_path) == tmp_path


def test_explicit_string_location_passes_through_unchanged():
    assert resolve_default_location("/var/tmp") == "/var/tmp"
