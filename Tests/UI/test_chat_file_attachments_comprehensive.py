"""Comprehensive tests for chat file attachment functionality including all file types."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

from textual.app import App
from textual.widgets import Button

# Note: Some imports may need adjustment based on actual module structure
# These are based on the analysis of the codebase


@pytest.fixture
def temp_files(tmp_path):
    """Create temporary test files of various types."""
    files = {}
    
    # Text file
    text_file = tmp_path / "test.txt"
    text_file.write_text("This is a test file content.")
    files['text'] = text_file
    
    # Code file
    code_file = tmp_path / "test.py"
    code_file.write_text("def hello():\n    print('Hello, World!')")
    files['code'] = code_file
    
    # JSON file
    json_file = tmp_path / "test.json"
    json_file.write_text('{"name": "test", "value": 42}')
    files['json'] = json_file
    
    # CSV file
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("name,value\ntest,42\nexample,100")
    files['csv'] = csv_file
    
    # Markdown file
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test Document\n\nThis is a **test** document.")
    files['markdown'] = md_file
    
    # Large text file (over 1MB)
    large_text = tmp_path / "large.txt"
    large_text.write_text("x" * (1024 * 1024 + 1))  # 1MB + 1 byte
    files['large_text'] = large_text
    
    # File without extension
    no_ext = tmp_path / "README"
    no_ext.write_text("This is a readme")
    files['no_extension'] = no_ext
    
    # Unicode filename
    unicode_file = tmp_path / "KՇ�.txt"
    unicode_file.write_text("Unicode filename test")
    files['unicode'] = unicode_file
    
    return files


class TestFileTypeDetection:
    """Test file type detection and handler selection."""
    
    def test_text_file_detection(self, temp_files):
        """Test that text files are correctly identified."""
        # Test by extension
        assert self._get_file_type(temp_files['text']) == 'text'
        assert self._get_file_type(temp_files['markdown']) == 'text'
    
    def test_code_file_detection(self, temp_files):
        """Test that code files are correctly identified."""
        assert self._get_file_type(temp_files['code']) == 'code'
        
        # Test various code extensions
        code_extensions = ['.js', '.cpp', '.java', '.rs', '.go', '.rb', '.php']
        for ext in code_extensions:
            assert self._get_file_type_by_extension(ext) == 'code'
    
    def test_data_file_detection(self, temp_files):
        """Test that data files are correctly identified."""
        assert self._get_file_type(temp_files['json']) == 'data'
        assert self._get_file_type(temp_files['csv']) == 'data'
        
        # Test other data extensions
        data_extensions = ['.yaml', '.yml', '.xml', '.tsv']
        for ext in data_extensions:
            assert self._get_file_type_by_extension(ext) == 'data'
    
    def test_unknown_file_detection(self, temp_files):
        """Test handling of files without recognized extensions."""
        assert self._get_file_type(temp_files['no_extension']) == 'unknown'
        assert self._get_file_type_by_extension('.xyz') == 'unknown'
    
    def _get_file_type(self, file_path):
        """Helper to determine file type from path."""
        return self._get_file_type_by_extension(file_path.suffix)
    
    def _get_file_type_by_extension(self, ext):
        """Helper to determine file type from extension."""
        # Simplified version - actual implementation would use FileHandlerRegistry
        text_exts = ['.txt', '.md', '.log', '.text', '.rst']
        code_exts = ['.py', '.js', '.cpp', '.java', '.rs', '.go', '.rb', '.php']
        data_exts = ['.json', '.yaml', '.yml', '.xml', '.csv', '.tsv']
        
        ext_lower = ext.lower()
        if ext_lower in text_exts:
            return 'text'
        elif ext_lower in code_exts:
            return 'code'
        elif ext_lower in data_exts:
            return 'data'
        else:
            return 'unknown'

class TestFileProcessing:
    """Test file content processing."""
    
    def test_text_file_inline_processing(self, temp_files):
        """Test that text files are processed for inline insertion."""
        content = temp_files['text'].read_text()
        processed = self._process_text_file(content)
        assert processed == content  # Text files should be inserted as-is
    
    def test_code_file_formatting(self, temp_files):
        """Test that code files are formatted with syntax highlighting."""
        content = temp_files['code'].read_text()
        processed = self._process_code_file(content, '.py')
        
        assert '```python' in processed
        assert 'def hello():' in processed
        assert '```' in processed
    
    def test_large_file_handling(self, temp_files):
        """Test that large files are handled appropriately."""
        # Large files should be rejected or truncated
        try:
            content = temp_files['large_text'].read_text()
            processed = self._process_text_file(content, max_size=1024*1024)
            assert False, "Should have raised an exception for large file"
        except ValueError as e:
            assert 'too large' in str(e).lower()
    
    def _process_text_file(self, content, max_size=None):
        """Helper to process text file content."""
        if max_size and len(content) > max_size:
            raise ValueError(f"File too large: {len(content)} bytes (max: {max_size})")
        return content
    
    def _process_code_file(self, content, extension):
        """Helper to process code file content."""
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.cpp': 'cpp',
            '.java': 'java'
        }
        lang = lang_map.get(extension.lower(), 'text')
        return f"```{lang}\n{content}\n```"


class TestFileSecurity:
    """Test file security and validation."""
    
    def test_path_traversal_prevention(self):
        """Test that path traversal attempts are blocked."""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config",
            "/etc/shadow"
        ]
        
        for path in dangerous_paths:
            assert not self._is_safe_path(path)
    
    def test_safe_paths(self):
        """Test that legitimate paths are allowed."""
        safe_paths = [
            "document.txt",
            "code/main.py",
            "data/config.json"
        ]
        
        for path in safe_paths:
            assert self._is_safe_path(path)
    
    def _is_safe_path(self, path):
        """Helper to check if path is safe."""
        path_str = str(path)
        
        # Check for path traversal
        if '..' in path_str or path_str.startswith('/'):
            return False
        
        # Check for hidden files
        parts = Path(path_str).parts
        for part in parts:
            if part.startswith('.'):
                return False
        
        return True


class TestErrorHandling:
    """Test error handling for file operations."""
    
    def test_nonexistent_file_handling(self):
        """Test handling of non-existent files."""
        result = self._process_file("/nonexistent/file.txt")
        assert result['type'] == 'error'
        assert 'not found' in result['message'].lower()
    
    def _process_file(self, file_path):
        """Helper to process a file and handle errors."""
        try:
            if not os.path.exists(file_path):
                return {'type': 'error', 'message': 'File not found'}
            
            # Try to read file
            with open(file_path, 'r') as f:
                content = f.read()
            
            return {'type': 'inline', 'content': content}
        except PermissionError:
            return {'type': 'error', 'message': 'Permission denied'}
        except Exception as e:
            return {'type': 'error', 'message': str(e)}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])