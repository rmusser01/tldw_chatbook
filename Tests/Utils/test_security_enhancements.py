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
        """Test that dangerous patterns are rejected."""
        dangerous_paths = [
            "../../etc/passwd",  # Path traversal
            "../..",  # Multiple parent refs
            "test;rm -rf /",  # Command injection
            "test && cat /etc/passwd",  # Command chaining
            "test || whoami",  # Command chaining
            "test`whoami`",  # Command substitution
            "test$(whoami)",  # Command substitution
            "test${PATH}",  # Variable expansion
            "test\x00file",  # Null byte
            "test|cat",  # Pipe
            "~/sensitive",  # Home directory
        ]

        for path in dangerous_paths:
            with pytest.raises(ValueError, match="dangerous pattern|null byte"):
                validate_path_simple(path)

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
