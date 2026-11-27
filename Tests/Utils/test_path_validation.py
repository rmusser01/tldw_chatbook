"""
Unit tests for path validation utilities.
"""

import os
import tempfile
from pathlib import Path
import pytest

from tldw_chatbook.Utils.path_validation import (
    validate_path,
    validate_filename,
    safe_join_path,
    is_safe_path,
    get_safe_relative_path,
)


class TestValidatePath:
    """Test cases for validate_path function."""

    def test_valid_path_within_base_directory(self):
        """Test that valid paths within base directory are accepted."""
        with tempfile.TemporaryDirectory() as base_dir:
            # Create a subdirectory
            sub_dir = Path(base_dir) / "subdir"
            sub_dir.mkdir()

            # Test absolute path
            result = validate_path(str(sub_dir), base_dir)
            assert result.resolve() == sub_dir.resolve()

            # Test relative path
            result = validate_path("subdir", base_dir)
            assert result.resolve() == sub_dir.resolve()

    def test_path_traversal_attempt_raises_error(self):
        """Test that path traversal attempts are blocked."""
        with tempfile.TemporaryDirectory() as base_dir:
            # Test parent directory access
            with pytest.raises(ValueError, match="outside the allowed directory"):
                validate_path("../", base_dir)

            # Test absolute path outside base
            with pytest.raises(ValueError, match="outside the allowed directory"):
                validate_path("/etc/passwd", base_dir)

            # Test complex traversal
            with pytest.raises(ValueError, match="outside the allowed directory"):
                validate_path("subdir/../../..", base_dir)

    def test_redacted_traversal_error_does_not_echo_the_private_path(self):
        """Redacted validation keeps traversal exception text value-free."""
        with tempfile.TemporaryDirectory() as base_dir:
            private_path = Path(base_dir).parent / "private-character-export.json"

            with pytest.raises(ValueError) as caught:
                validate_path(private_path, base_dir, redact_paths=True)

            assert str(caught.value) == "Path is outside the allowed directory"
            assert private_path.name not in str(caught.value)

    def test_hidden_file_access_blocked(self):
        """Test that hidden files/directories are blocked."""
        with tempfile.TemporaryDirectory() as base_dir:
            with pytest.raises(ValueError, match="hidden files"):
                validate_path(".hidden", base_dir)

            with pytest.raises(ValueError, match="hidden files"):
                validate_path("subdir/.config", base_dir)

    def test_hidden_base_directory_blocked(self):
        """A base_directory whose OWN final component is dotted is rejected.

        task-564: callers that validate an arbitrary destination against its
        own immediate parent (``base_directory=target.parent``) fold a hidden
        final directory into the base itself, where the relative-parts hidden
        check never sees it. Pin the dedicated base-name check so the branch
        cannot regress silently.
        """
        with tempfile.TemporaryDirectory() as base_dir:
            hidden_base = Path(base_dir) / ".hidden_base"
            hidden_base.mkdir()
            with pytest.raises(ValueError, match="hidden files"):
                validate_path("export.json", hidden_base)

    def test_base_directory_under_hidden_ancestor_allowed(self):
        """A base dir merely LIVING UNDER a dotted ancestor stays valid.

        The base-name check deliberately does not walk ancestors — app-owned
        roots like ``~/.local/share/tldw_cli`` must keep working.
        """
        with tempfile.TemporaryDirectory() as base_dir:
            nested = Path(base_dir) / ".config" / "app_data"
            nested.mkdir(parents=True)
            result = validate_path("export.json", nested)
            assert result.name == "export.json"

    def test_symlink_resolution(self):
        """Test that symlinks are resolved correctly."""
        with tempfile.TemporaryDirectory() as base_dir:
            base_path = Path(base_dir)

            # Create a file inside base directory
            safe_file = base_path / "safe.txt"
            safe_file.write_text("safe content")

            # Create a symlink inside base directory
            link = base_path / "link_to_safe"
            link.symlink_to(safe_file)

            # This should be allowed
            result = validate_path(str(link), base_dir)
            assert (
                result.resolve() == safe_file.resolve()
            )  # Should resolve to the actual file

            # Create a file outside base directory
            with tempfile.NamedTemporaryFile(delete=False) as outside_file:
                outside_file.write(b"outside content")
                outside_path = outside_file.name

            try:
                # Create a symlink pointing outside
                bad_link = base_path / "link_to_outside"
                bad_link.symlink_to(outside_path)

                # This should be blocked
                with pytest.raises(ValueError, match="outside the allowed directory"):
                    validate_path(str(bad_link), base_dir)
            finally:
                os.unlink(outside_path)


class TestValidateFilename:
    """Test cases for validate_filename function."""

    def test_valid_filenames(self):
        """Test that valid filenames are accepted."""
        valid_names = [
            "file.txt",
            "document.pdf",
            "image.png",
            "my-file_123.doc",
            "файл.txt",  # Unicode filename
            "文档.pdf",  # Chinese characters
        ]

        for name in valid_names:
            assert validate_filename(name) == name

    def test_invalid_filenames(self):
        """Test that invalid filenames are rejected."""
        # Empty filename
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_filename("")

        # Path separators
        with pytest.raises(ValueError, match="path separators"):
            validate_filename("path/to/file.txt")

        with pytest.raises(ValueError, match="path separators"):
            validate_filename("path\\to\\file.txt")

        # Parent directory references
        with pytest.raises(ValueError, match="parent directory"):
            validate_filename("..file.txt")

        with pytest.raises(ValueError, match="parent directory"):
            validate_filename("file..txt")

        # Null bytes
        with pytest.raises(ValueError, match="null bytes"):
            validate_filename("file\x00.txt")

        # Reserved Windows names
        reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
        for name in reserved_names:
            with pytest.raises(ValueError, match="reserved filename"):
                validate_filename(name)

            # Test with extensions too
            with pytest.raises(ValueError, match="reserved filename"):
                validate_filename(f"{name}.txt")


class TestSafeJoinPath:
    """Test cases for safe_join_path function."""

    def test_safe_path_joining(self):
        """Test that paths are joined safely."""
        with tempfile.TemporaryDirectory() as base_dir:
            base_path = Path(base_dir)

            # Simple join
            result = safe_join_path(base_dir, "subdir", "file.txt")
            expected = base_path / "subdir" / "file.txt"
            assert result.resolve() == expected.resolve()

            # Validate each component
            result = safe_join_path(base_dir, "dir1", "dir2", "file.txt")
            expected = base_path / "dir1" / "dir2" / "file.txt"
            assert result.resolve() == expected.resolve()

    def test_unsafe_components_rejected(self):
        """Test that unsafe path components are rejected."""
        with tempfile.TemporaryDirectory() as base_dir:
            # Path separator in component
            with pytest.raises(ValueError, match="path separators"):
                safe_join_path(base_dir, "dir/subdir", "file.txt")

            # Parent directory reference
            with pytest.raises(ValueError, match="parent directory"):
                safe_join_path(base_dir, "..", "file.txt")

            # Null byte
            with pytest.raises(ValueError, match="null bytes"):
                safe_join_path(base_dir, "dir\x00", "file.txt")


class TestIsSafePath:
    """Test cases for is_safe_path function."""

    def test_safe_paths_return_true(self):
        """Test that safe paths return True."""
        with tempfile.TemporaryDirectory() as base_dir:
            assert is_safe_path(base_dir, base_dir) is True
            assert is_safe_path("subdir", base_dir) is True

            # Create actual subdirectory
            sub_dir = Path(base_dir) / "subdir"
            sub_dir.mkdir()
            assert is_safe_path(str(sub_dir), base_dir) is True

    def test_unsafe_paths_return_false(self):
        """Test that unsafe paths return False."""
        with tempfile.TemporaryDirectory() as base_dir:
            assert is_safe_path("../", base_dir) is False
            assert is_safe_path("/etc/passwd", base_dir) is False
            assert is_safe_path(".hidden", base_dir) is False


class TestGetSafeRelativePath:
    """Test cases for get_safe_relative_path function."""

    def test_valid_relative_paths(self):
        """Test that valid relative paths are returned."""
        with tempfile.TemporaryDirectory() as base_dir:
            base_path = Path(base_dir)

            # Create subdirectory structure
            sub_path = base_path / "sub1" / "sub2"
            sub_path.mkdir(parents=True)

            # Get relative path
            result = get_safe_relative_path(str(sub_path), base_dir)
            assert result == Path("sub1/sub2")

            # Test with file
            file_path = sub_path / "file.txt"
            file_path.touch()
            result = get_safe_relative_path(str(file_path), base_dir)
            assert result == Path("sub1/sub2/file.txt")

    def test_unsafe_paths_return_none(self):
        """Test that unsafe paths return None."""
        with tempfile.TemporaryDirectory() as base_dir:
            # Path outside base directory
            assert get_safe_relative_path("/etc/passwd", base_dir) is None

            # Parent directory
            parent = Path(base_dir).parent
            assert get_safe_relative_path(str(parent), base_dir) is None


class TestValidatePathAllowHidden:
    """Test cases for the allow_hidden opt-out on validate_path."""

    def test_validate_path_allow_hidden_permits_dotdirs_under_base(self, tmp_path):
        from tldw_chatbook.Utils.path_validation import validate_path

        hidden = tmp_path / ".github" / "workflows"
        hidden.mkdir(parents=True)
        result = validate_path(".github/workflows", tmp_path, allow_hidden=True)
        assert result == hidden.resolve()

    def test_validate_path_default_still_rejects_hidden(self, tmp_path):
        from tldw_chatbook.Utils.path_validation import validate_path

        (tmp_path / ".hidden").mkdir()
        with pytest.raises(ValueError, match="hidden"):
            validate_path(".hidden", tmp_path)

    def test_validate_path_allow_hidden_still_blocks_traversal(self, tmp_path):
        from tldw_chatbook.Utils.path_validation import validate_path

        with pytest.raises(ValueError, match="outside"):
            validate_path("../escape", tmp_path, allow_hidden=True)


class TestCanonicalDirectory:
    def test_accepts_existing_hidden_canonical_directory(self, tmp_path):
        from tldw_chatbook.Utils.path_validation import validate_canonical_directory

        root = tmp_path.resolve() / ".private"
        root.mkdir()
        assert validate_canonical_directory(root) == root

    @pytest.mark.parametrize(
        "kind",
        ["relative", "dotdot", "symlink", "file", "missing", "null", "trailing-slash"],
    )
    def test_rejects_noncanonical_or_nondirectory_roots_without_path_content(
        self, tmp_path, kind
    ):
        from tldw_chatbook.Utils.path_validation import validate_canonical_directory

        root = tmp_path.resolve()
        target = root / "private-token"
        target.mkdir()
        alias = root / "alias"
        alias.symlink_to(target, target_is_directory=True)
        file = root / "file"
        file.write_text("fixture")
        value = {
            "relative": "private-token",
            "dotdot": str(target) + "/../private-token",
            "symlink": str(alias),
            "file": str(file),
            "missing": str(root / "missing-private-token"),
            "null": "private-token\x00",
            "trailing-slash": str(target) + "/",
        }[kind]
        with pytest.raises(ValueError) as error:
            validate_canonical_directory(value)
        assert "private-token" not in str(error.value)


def test_validate_path_simple_still_refuses_an_unexpanded_tilde() -> None:
    """TASK-34000.20 AC#4 / TASK-32927: expansion is the CALLER's job.

    The Library pre-check now expands ``~`` itself (``ingest_preflight.
    expand_user_path``); this pins that the validator's own guard did not
    move, so every other caller still refuses an unexpanded ``~/``.
    """
    from tldw_chatbook.Utils.path_validation import validate_path_simple

    with pytest.raises(ValueError, match="dangerous pattern: ~/"):
        validate_path_simple("~/Documents/x.txt")


class TestQuietSuccessPath:
    """TASK-40608: a successful validation fetches no logger, and the one-shot worker records no metrics."""

    @pytest.fixture
    def base(self, tmp_path):
        # Resolved: the simple validator warns when a path resolves differently.
        folder = tmp_path.resolve()
        (folder / "inside.txt").write_text("x", encoding="utf-8")
        return folder

    def test_a_successful_validation_never_fetches_the_logger(self, base, monkeypatch):
        from tldw_chatbook.Utils import path_validation

        def fetched():
            raise AssertionError("the logger was fetched for a successful validation")

        monkeypatch.setattr(path_validation, "_logging_logger", fetched)
        assert path_validation.validate_path("inside.txt", base) == (base / "inside.txt").resolve()
        assert path_validation.validate_path_simple(str(base / "inside.txt")).name == "inside.txt"

    def test_a_refusal_is_still_logged(self, base, monkeypatch):
        from tldw_chatbook.Utils import path_validation

        messages = []

        class Recorder:
            def warning(self, message, *args):
                messages.append(("warning", message))

            def error(self, message, *args):
                messages.append(("error", message))

        monkeypatch.setattr(path_validation, "_logging_logger", Recorder)
        with pytest.raises(ValueError):
            path_validation.validate_path("../outside.txt", base)
        assert [level for level, _ in messages] == ["warning", "error"]
        assert "traversal" in messages[0][1].lower()

    @staticmethod
    def _recorded(monkeypatch):
        from tldw_chatbook.Metrics import metrics_logger

        seen = []
        monkeypatch.setattr(metrics_logger, "log_counter", lambda name, *rest: seen.append(name))
        monkeypatch.setattr(metrics_logger, "log_histogram", lambda name, *rest: seen.append(name))
        return seen

    def test_an_ordinary_process_still_records_validation_metrics(self, base, monkeypatch):
        seen = self._recorded(monkeypatch)
        validate_path("inside.txt", base)
        assert seen == [
            "path_validation_validate_path_attempt",
            "path_validation_validate_path_duration",
            "path_validation_validate_path_success",
        ]

    def test_a_process_without_the_metrics_module_records_nothing_and_never_loads_it(
        self, base, monkeypatch
    ):
        """The one-shot tool worker and the remote bundle: nothing reads a metric there."""
        import sys

        from tldw_chatbook.Utils import path_validation

        seen = self._recorded(monkeypatch)
        monkeypatch.delitem(sys.modules, path_validation._METRICS_MODULE)
        assert validate_path("inside.txt", base) == (base / "inside.txt").resolve()
        with pytest.raises(ValueError):
            validate_path("../outside.txt", base)
        assert path_validation.validate_path_simple(str(base / "inside.txt")).name == "inside.txt"
        assert seen == []
        assert path_validation._METRICS_MODULE not in sys.modules
