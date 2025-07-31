"""
Simplified integration tests for file extraction workflow.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import asyncio

from tldw_chatbook.Utils.file_extraction import FileExtractor, ExtractedFile
from tldw_chatbook.Widgets.file_extraction_dialog import FileExtractionDialog


class TestFileExtractionIntegrationSimple:
    """Simplified integration tests for the file extraction workflow."""
    
    @pytest.fixture
    def sample_files(self):
        """Create sample extracted files for testing."""
        return [
            ExtractedFile(
                filename="script.py",
                content="print('Hello, World!')",
                language="python",
                start_pos=0,
                end_pos=100
            ),
            ExtractedFile(
                filename="data.json",
                content='{"name": "test", "value": 42}',
                language="json",
                start_pos=101,
                end_pos=200
            ),
        ]
    
    def test_file_extractor_basic_workflow(self):
        """Test basic file extraction workflow."""
        extractor = FileExtractor()
        
        text = '''
        Here's a Python script:
        ```python
        def hello():
            print("Hello!")
        ```
        
        And some JSON data:
        ```json
        {
            "key": "value"
        }
        ```
        '''
        
        files = extractor.extract_files(text)
        
        assert len(files) == 2
        assert files[0].language == "python"
        assert files[0].filename == "extracted_1.py"
        assert "def hello():" in files[0].content
        
        assert files[1].language == "json"
        assert files[1].filename == "extracted_2.json"
        assert '"key": "value"' in files[1].content
    
    def test_file_extraction_with_special_types(self):
        """Test extraction of special file types."""
        extractor = FileExtractor()
        
        test_cases = [
            ('dockerfile', 'FROM python:3.9', 'Dockerfile'),
            ('requirements', 'numpy==1.21.0', 'requirements.txt'),
            ('docker-compose', 'version: "3"', 'docker-compose.yml'),
            ('gitignore', '*.pyc', '.gitignore'),
        ]
        
        for lang, content, expected_filename in test_cases:
            text = f'```{lang}\n{content}\n```'
            files = extractor.extract_files(text)
            
            assert len(files) == 1
            assert files[0].filename == expected_filename
            assert files[0].content == content
    
    def test_markdown_table_extraction(self):
        """Test extraction of markdown tables as CSV."""
        extractor = FileExtractor()
        
        text = '''
        Here's a data table:
        
        | Name | Age | City |
        |------|-----|------|
        | John | 30  | NYC  |
        | Jane | 25  | LA   |
        '''
        
        files = extractor.extract_files(text)
        
        assert len(files) == 1
        assert files[0].filename == "table_1.csv"
        assert files[0].language == "csv"
        
        # Check CSV content (handle possible \r\n line endings)
        csv_lines = files[0].content.strip().replace('\r', '').split('\n')
        assert csv_lines[0] == "Name,Age,City"
        assert csv_lines[1] == "John,30,NYC"
        assert csv_lines[2] == "Jane,25,LA"
    
    def test_file_validation(self):
        """Test file validation logic."""
        extractor = FileExtractor()
        
        # Valid JSON
        valid_json = ExtractedFile(
            filename="test.json",
            content='{"valid": true}',
            language="json",
            start_pos=0,
            end_pos=0
        )
        assert extractor.validate_content(valid_json) is None
        
        # Invalid JSON
        invalid_json = ExtractedFile(
            filename="test.json",
            content='{"invalid": syntax}',
            language="json",
            start_pos=0,
            end_pos=0
        )
        error = extractor.validate_content(invalid_json)
        assert error is not None
        assert "Invalid JSON" in error
        
        # Valid package.json
        valid_package = ExtractedFile(
            filename="package.json",
            content='{"name": "test", "version": "1.0.0"}',
            language="json",
            start_pos=0,
            end_pos=0
        )
        assert extractor.validate_content(valid_package) is None
        
        # Invalid package.json (missing name)
        invalid_package = ExtractedFile(
            filename="package.json",
            content='{"version": "1.0.0"}',
            language="json",
            start_pos=0,
            end_pos=0
        )
        error = extractor.validate_content(invalid_package)
        assert error is not None
        assert "Missing 'name'" in error
    
    def test_file_dialog_creation(self, sample_files):
        """Test that FileExtractionDialog can be created with files."""
        dialog = FileExtractionDialog(sample_files)
        
        assert dialog.extracted_files == sample_files
        assert len(dialog._selected_files) == 2  # All selected by default
    
    def test_save_files_manually(self, sample_files):
        """Test saving files to disk manually without dialog."""
        with tempfile.TemporaryDirectory() as temp_dir:
            downloads_path = Path(temp_dir) / "Downloads"
            downloads_path.mkdir()
            
            # Save files manually
            saved_files = []
            for file in sample_files:
                save_path = downloads_path / file.filename
                save_path.write_text(file.content, encoding='utf-8')
                saved_files.append({
                    'filename': file.filename,
                    'path': str(save_path),
                    'size': len(file.content),
                    'language': file.language
                })
            
            # Verify files were saved
            assert len(saved_files) == 2
            
            # Check files exist on disk
            for saved in saved_files:
                file_path = Path(saved['path'])
                assert file_path.exists()
                assert file_path.parent == downloads_path
                
                # Read back and verify content
                if saved['filename'] == 'script.py':
                    content = file_path.read_text()
                    assert content == "print('Hello, World!')"
    
    def test_multiple_file_types_extraction(self):
        """Test extracting multiple different file types."""
        extractor = FileExtractor()
        
        text = '''
        ```python
        print("Python code")
        ```
        
        ```javascript
        console.log("JavaScript code");
        ```
        
        ```dockerfile
        FROM node:14
        ```
        
        ```yaml
        version: "3.8"
        ```
        
        | Data | Value |
        |------|-------|
        | A    | 1     |
        | B    | 2     |
        '''
        
        files = extractor.extract_files(text)
        
        # Should extract 4 code blocks + 1 table
        assert len(files) == 5
        
        # Check file types
        filenames = [f.filename for f in files]
        # Table comes first, then code blocks
        assert "table_1.csv" in filenames
        assert any("py" in f for f in filenames)  # Python file
        assert any("js" in f for f in filenames)  # JavaScript file
        assert "Dockerfile" in filenames
        assert any("yaml" in f for f in filenames)  # YAML file
    
    def test_file_size_validation(self):
        """Test file size limits."""
        extractor = FileExtractor()
        
        # Create a file that's too large (>10MB)
        large_content = "x" * (11 * 1024 * 1024)
        large_file = ExtractedFile(
            filename="large.txt",
            content=large_content,
            language="text",
            start_pos=0,
            end_pos=0
        )
        
        error = extractor.validate_content(large_file)
        assert error is not None
        assert "too large" in error
    
    def test_filename_sanitization(self):
        """Test filename sanitization in extraction."""
        extractor = FileExtractor()
        
        # Test internal sanitization method directly
        assert extractor._sanitize_filename("my<>file:name|test") == "myfilenametest"
        assert extractor._sanitize_filename("file with spaces") == "file_with_spaces"
        assert extractor._sanitize_filename("file/with/path") == "filewithpath"
        
        # Test with a simple code block
        text = '''
        ```python
        print("test")
        ```
        '''
        
        files = extractor.extract_files(text)
        assert len(files) == 1
        # Should get a default filename
        assert files[0].filename == "extracted_1.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])