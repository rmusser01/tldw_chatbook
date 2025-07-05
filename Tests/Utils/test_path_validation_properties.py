"""
Property-based tests for path validation using hypothesis.
"""

import os
import tempfile
from pathlib import Path
import string

import pytest
from hypothesis import given, strategies as st, assume

from tldw_chatbook.Utils.path_validation import (
    validate_path, validate_filename, safe_join_path,
    is_safe_path
)


# Custom strategies for generating test data
def safe_filename_strategy():
    """Generate potentially safe filenames."""
    return st.text(
        alphabet=string.ascii_letters + string.digits + "-_.() ",
        min_size=1,
        max_size=255
    ).filter(lambda x: x.strip() != "" and '..' not in x)


def unsafe_filename_strategy():
    """Generate filenames with potentially unsafe characters."""
    return st.one_of(
        st.just(""),  # Empty
        st.just(".."),  # Parent directory
        st.sampled_from(["/", "\\", "\x00"]).flatmap(  # Bad characters
            lambda char: st.text(min_size=1).map(lambda s: s[:len(s)//2] + char + s[len(s)//2:])
        ),
        st.sampled_from(["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]),  # Reserved names
    )


def path_component_strategy():
    """Generate path components that might be safe or unsafe."""
    return st.one_of(
        safe_filename_strategy(),
        st.text(alphabet=string.printable, min_size=1, max_size=100),
        st.sampled_from([".", "..", "~", ".hidden", "...", "...."])
    )


class TestPathValidationProperties:
    """Property-based tests for path validation."""
    
    @given(st.lists(safe_filename_strategy(), min_size=1, max_size=5))
    def test_safe_paths_always_validate(self, components):
        """Property: Safe path components joined together should always validate."""
        with tempfile.TemporaryDirectory() as base_dir:
            # Create the directory structure
            current_path = Path(base_dir)
            for component in components[:-1]:  # All but last are directories
                current_path = current_path / component
                current_path.mkdir(exist_ok=True)
            
            # Last component is the file
            full_path = current_path / components[-1]
            
            # Should validate successfully
            try:
                result = validate_path(str(full_path), base_dir)
                # Use resolve() to handle path resolution differences on macOS
                assert str(result.resolve()).startswith(str(Path(base_dir).resolve()))
            except ValueError as e:
                # If it fails, it should only be due to hidden files
                assert "hidden files" in str(e)
    
    @given(path_component_strategy())
    def test_path_outside_base_never_validates(self, component):
        """Property: Paths outside base directory should never validate."""
        with tempfile.TemporaryDirectory() as base_dir:
            with tempfile.TemporaryDirectory() as other_dir:
                # Try to access something in other_dir from base_dir
                target = Path(other_dir) / component
                
                # Should never validate
                assert is_safe_path(str(target), base_dir) is False
    
    @given(st.lists(path_component_strategy(), min_size=1, max_size=3))
    def test_traversal_always_caught(self, components):
        """Property: Any path that resolves outside base should be caught."""
        assume(any(".." in str(c) for c in components))  # Ensure at least one traversal
        
        with tempfile.TemporaryDirectory() as base_dir:
            try:
                path = Path(base_dir)
                for component in components:
                    path = path / component
                
                # If the final path is outside base_dir, validation should fail
                resolved = path.resolve()
                base_resolved = Path(base_dir).resolve()
                
                if not str(resolved).startswith(str(base_resolved)):
                    with pytest.raises(ValueError, match="outside the allowed directory"):
                        validate_path(str(path), base_dir)
            except (OSError, ValueError):
                # Some paths might be invalid at OS level
                pass


class TestFilenameValidationProperties:
    """Property-based tests for filename validation."""
    
    @given(safe_filename_strategy())
    def test_safe_filenames_preserve_content(self, filename):
        """Property: Safe filenames should be returned unchanged."""
        # Filter out hidden files and reserved names
        assume(not filename.startswith('.'))
        assume(filename.split('.')[0].upper() not in {
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
            'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
            'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        })
        
        result = validate_filename(filename)
        assert result == filename
    
    @given(unsafe_filename_strategy())
    def test_unsafe_filenames_always_rejected(self, filename):
        """Property: Unsafe filenames should always be rejected."""
        with pytest.raises(ValueError):
            validate_filename(filename)
    
    @given(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=50))
    def test_alphanumeric_always_safe(self, filename):
        """Property: Pure alphanumeric filenames are always safe."""
        # Skip reserved Windows filenames
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
            'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
            'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        assume(filename.upper() not in reserved_names)
        
        result = validate_filename(filename)
        assert result == filename
    
    @given(
        st.text(alphabet=string.ascii_letters, min_size=1, max_size=10),
        st.sampled_from(["/", "\\", "\x00", ".."])
    )
    def test_dangerous_chars_always_caught(self, safe_part, dangerous_char):
        """Property: Dangerous characters are always caught regardless of position."""
        filename = safe_part[:len(safe_part)//2] + dangerous_char + safe_part[len(safe_part)//2:]
        
        with pytest.raises(ValueError):
            validate_filename(filename)


class TestSafeJoinPathProperties:
    """Property-based tests for safe path joining."""
    
    @given(st.lists(safe_filename_strategy(), min_size=1, max_size=5))
    def test_safe_components_join_successfully(self, components):
        """Property: Safe components should join without errors."""
        with tempfile.TemporaryDirectory() as base_dir:
            # Filter out hidden files and reserved names
            reserved_names = {
                'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
                'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
            }
            components = [c for c in components if not c.startswith('.')]
            components = [c for c in components if c.split('.')[0].upper() not in reserved_names]
            
            if components:  # Only test if we have valid components
                result = safe_join_path(base_dir, *components)
                # Use resolve() to handle path resolution differences on macOS
                assert str(result.resolve()).startswith(str(Path(base_dir).resolve()))
                
                # Verify all components are in the result
                for component in components:
                    assert component in str(result)
    
    @given(
        st.lists(safe_filename_strategy(), min_size=0, max_size=2),
        unsafe_filename_strategy(),
        st.lists(safe_filename_strategy(), min_size=0, max_size=2)
    )
    def test_unsafe_component_fails_join(self, safe_before, unsafe, safe_after):
        """Property: Any unsafe component should cause join to fail."""
        with tempfile.TemporaryDirectory() as base_dir:
            components = list(safe_before) + [unsafe] + list(safe_after)
            
            if any(components):  # Don't test empty list
                with pytest.raises(ValueError):
                    safe_join_path(base_dir, *components)


class TestPathValidationInvariants:
    """Test invariants that should always hold."""
    
    @given(st.text(min_size=1, max_size=100))
    def test_validated_path_always_in_base(self, user_input):
        """Invariant: Successfully validated paths are always within base directory."""
        with tempfile.TemporaryDirectory() as base_dir:
            try:
                result = validate_path(user_input, base_dir)
                # If validation succeeds, path must be within base
                # Use resolve() to handle path resolution differences on macOS
                assert str(result.resolve()).startswith(str(Path(base_dir).resolve()))
                assert result.resolve().is_relative_to(Path(base_dir).resolve())
            except ValueError:
                # Validation can fail, that's expected
                pass
    
    @given(st.text(min_size=1, max_size=255))
    def test_is_safe_path_consistent_with_validate_path(self, user_input):
        """Invariant: is_safe_path and validate_path should be consistent."""
        with tempfile.TemporaryDirectory() as base_dir:
            is_safe = is_safe_path(user_input, base_dir)
            
            try:
                validate_path(user_input, base_dir)
                validates = True
            except ValueError:
                validates = False
            
            # If path is safe, it should validate
            # If it validates, it should be safe
            assert is_safe == validates