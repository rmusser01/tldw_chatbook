"""
Unit tests for path validation utilities.
"""

import os
import tempfile
from pathlib import Path
import pytest

from tldw_chatbook.Utils.path_validation import (
    validate_path, validate_filename, safe_join_path,
    is_safe_path, get_safe_relative_path
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
    
    def test_hidden_file_access_blocked(self):
        """Test that hidden files/directories are blocked."""
        with tempfile.TemporaryDirectory() as base_dir:
            with pytest.raises(ValueError, match="hidden files"):
                validate_path(".hidden", base_dir)
            
            with pytest.raises(ValueError, match="hidden files"):
                validate_path("subdir/.config", base_dir)
    
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
            assert result.resolve() == safe_file.resolve()  # Should resolve to the actual file
            
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
            "文档.pdf",   # Chinese characters
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