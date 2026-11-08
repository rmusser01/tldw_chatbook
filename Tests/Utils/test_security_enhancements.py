"""Tests for security enhancement utilities."""

import pytest
from pathlib import Path
from tldw_chatbook.Utils.path_validation import (
    validate_existing_absolute_directory,
    validate_path_simple,
)


class TestValidatePathSimple:
    """Test the simple path validation function."""

    def test_valid_paths(self):
        """Test that valid paths are accepted."""
        # Relative path
        result = validate_path_simple("test.txt")
        assert isinstance(result, Path)
        assert str(result) == "test.txt"

        # Absolute path
        result = validate_path_simple("/tmp/test.txt")
        assert isinstance(result, Path)
        assert str(result) == "/tmp/test.txt"

    def test_dangerous_patterns_rejected(self):
        """Test that dangerous patterns are rejected.

        TASK-38227: dev's list also held seven shell metacharacter spellings
        (``;``, ``&&``, ``||``, a backtick, ``$(``, ``${``, ``|``). They are
        legal file-name characters and no caller hands the path to a shell,
        so this PR admits them by default; they moved to
        ``test_legal_names_with_shell_metacharacters_pass`` below.
        """
        dangerous_paths = [
            "../../etc/passwd",  # Path traversal
            "../..",  # Multiple parent refs
            "..\\..\\etc\\passwd",  # Multiple parent refs (Windows)
            "test\x00file",  # Null byte
            "~/sensitive",  # Home directory
            "~\\sensitive",  # Windows home
        ]

        for path in dangerous_paths:
            with pytest.raises(ValueError, match="dangerous pattern|null byte"):
                validate_path_simple(path)

    @pytest.mark.parametrize(
        "name",
        [
            "Q3 P&L; card.json",
            "notes `draft`.md",
            "budget ${2024}.csv",
            "a|b.txt",
            "sales && marketing.txt",
            "this || that.txt",
            "$(report).json",
            "it's \"quoted\" <here> *.?",
            "with space.json",
        ],
    )
    def test_legal_names_with_shell_metacharacters_pass(self, name):
        """A legal file name passes the default validator (TASK-38227).

        Dev refused ``Q3 P&L; card.json`` with ``dangerous pattern: ;`` in
        every importer that calls ``validate_path_simple``, because its
        default rejected shell metacharacters in names that never reach a
        shell. This PR flips that default; traversal, NUL and ``~`` are
        still refused (``test_kept_refusals_still_refuse``).
        """
        result = validate_path_simple(name)
        assert isinstance(result, Path)
        assert result == Path(name)

    def test_shell_metacharacter_opt_in_still_refuses(self):
        """``reject_shell_metacharacters=True`` stays available as an opt-in.

        No caller needs it today (TASK-38227 audit: 148 calls, none reaching
        a shell), but a future boundary that does hand a path to a shell can
        ask for the strict list explicitly.
        """
        for spelling in ("a;b.txt", "a|b.txt", "a && b", "a || b", "x`y`", "$(x)", "${x}"):
            with pytest.raises(ValueError, match="dangerous pattern"):
                validate_path_simple(spelling, reject_shell_metacharacters=True)

    @pytest.mark.parametrize(
        ("spelling", "message"),
        [
            pytest.param("test\x00file", "null bytes", id="nul"),
            pytest.param("../../etc/passwd", "dangerous pattern: ../..", id="posix-double-parent"),
            pytest.param("..\\..\\etc\\passwd", r"dangerous pattern: \.\.\\\.\.", id="windows-double-parent"),
            pytest.param("~/secrets.txt", "dangerous pattern: ~/", id="posix-home"),
            pytest.param("~\\secrets.txt", r"dangerous pattern: ~\\", id="windows-home"),
        ],
    )
    def test_kept_refusals_still_refuse(self, spelling, message):
        """Each non-shell refusal survives the TASK-38227 flip, one case each."""
        with pytest.raises(ValueError, match=message):
            validate_path_simple(spelling)

    def test_require_exists_refusals_survive_the_flip(self, tmp_path):
        """``require_exists`` still refuses a missing path, and still needs
        ``probe_existing`` (TASK-38227 kept both)."""
        missing = tmp_path / "Q3 P&L; missing.json"
        with pytest.raises(ValueError, match="does not exist"):
            validate_path_simple(missing, require_exists=True)
        with pytest.raises(ValueError, match="requires probe_existing"):
            validate_path_simple(missing, require_exists=True, probe_existing=False)
        present = tmp_path / "Q3 P&L; present.json"
        present.write_text("{}", encoding="utf-8")
        assert validate_path_simple(present, require_exists=True) == present

    def test_require_exists_option(self):
        """Test the require_exists option."""
        # Non-existent file should fail when require_exists=True
        with pytest.raises(ValueError, match="does not exist"):
            validate_path_simple(
                "/tmp/definitely_does_not_exist_12345.txt", require_exists=True
            )

        # Should pass when require_exists=False
        result = validate_path_simple("/tmp/new_file.txt", require_exists=False)
        assert isinstance(result, Path)

    def test_single_parent_ref_accepted_both_separator_conventions(self):
        """A legitimate single parent-dir segment must be treated the same
        regardless of which separator convention the string uses.

        Regression test for task-838: the raw-substring scan used to look
        for POSIX "../.." (two consecutive parent refs) but Windows "..\\"
        (a *single* parent ref), so the same logical path --
        ``nested/../locks`` -- was accepted with forward slashes and
        rejected with backslashes. This is exercised directly (not via
        ``os.path`` helpers) so it is meaningful on POSIX CI too: the whole
        point is that the pattern list must not depend on the host platform.
        """
        # POSIX-style: single parent ref, unresolved.
        result = validate_path_simple("/tmp/xyz/nested/../locks")
        assert isinstance(result, Path)

        # Windows-style: the same logical path, single parent ref.
        result = validate_path_simple("C:\\Temp\\xyz\\nested\\..\\locks")
        assert isinstance(result, Path)

    def test_multi_level_parent_ref_still_rejected_both_conventions(self):
        """A genuine multi-level traversal attempt must still be rejected
        for both separator conventions -- the parity fix must not weaken
        the check, only stop over-rejecting single, legitimate parent refs.
        """
        with pytest.raises(ValueError, match="dangerous pattern"):
            validate_path_simple("../../etc/passwd")

        with pytest.raises(ValueError, match="dangerous pattern"):
            validate_path_simple("..\\..\\etc\\passwd")

        # The second parent reference may terminate the path. Windows CI
        # constructs this exact shape for a lock root beneath ``tmp_path``.
        with pytest.raises(ValueError, match="dangerous pattern"):
            validate_path_simple("C:\\Temp\\cache\\..\\..")


def test_existing_absolute_directory_returns_normalized_path(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()

    result = validate_existing_absolute_directory(nested / "..")

    assert result == tmp_path.resolve()


def test_existing_absolute_directory_rejects_relative_missing_and_files(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    relative = Path("relative")
    relative.mkdir()
    regular_file = tmp_path / "file.txt"
    regular_file.write_text("not a directory", encoding="utf-8")

    for candidate in (relative, tmp_path / "missing", regular_file):
        with pytest.raises(ValueError, match="absolute existing directory"):
            validate_existing_absolute_directory(candidate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
