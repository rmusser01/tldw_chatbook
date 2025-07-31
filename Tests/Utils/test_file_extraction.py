"""
Unit tests for file extraction functionality.
"""
import pytest
from unittest.mock import Mock, patch
import json
import yaml

from tldw_chatbook.Utils.file_extraction import FileExtractor, ExtractedFile


class TestFileExtractor:
    """Test cases for the FileExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a FileExtractor instance for testing."""
        return FileExtractor()
    
    def test_extract_simple_code_block(self, extractor):
        """Test extraction of a simple code block."""
        text = '''Here's a Python script:
```python
print("Hello, World!")
```
'''
        
        files = extractor.extract_files(text)
        assert len(files) == 1
        assert files[0].filename == "extracted_1.py"
        assert files[0].content == 'print("Hello, World!")'
        assert files[0].language == "python"
    
    def test_extract_multiple_code_blocks(self, extractor):
        """Test extraction of multiple code blocks."""
        text = '''
        First file:
        ```javascript
        console.log("Hello");
        ```
        
        Second file:
        ```python
        print("World")
        ```
        '''
        
        files = extractor.extract_files(text)
        assert len(files) == 2
        assert files[0].filename == "extracted_1.js"
        assert files[0].language == "javascript"
        assert files[1].filename == "extracted_2.py"
        assert files[1].language == "python"
    
    def test_extract_with_filename_hint(self, extractor):
        """Test extraction with filename hints in comments."""
        text = '''
        # main.py
        ```python
        def main():
            pass
        ```
        '''
        
        files = extractor.extract_files(text)
        assert len(files) == 1
        assert files[0].filename == "main.py"
    
    def test_extract_special_filenames(self, extractor):
        """Test extraction of files with special naming conventions."""
        test_cases = [
            ("dockerfile", "Dockerfile"),
            ("makefile", "Makefile"),
            ("jenkinsfile", "Jenkinsfile"),
            ("gitignore", ".gitignore"),
            ("htaccess", ".htaccess"),
            ("requirements", "requirements.txt"),
            ("docker-compose", "docker-compose.yml"),
            ("nginx", "nginx.conf"),
        ]
        
        for lang, expected_filename in test_cases:
            text = f'''```{lang}
test content
```'''
            files = extractor.extract_files(text)
            assert len(files) == 1, f"Expected 1 file for {lang}, got {len(files)}"
            assert files[0].filename == expected_filename, f"Expected {expected_filename} for {lang}, got {files[0].filename if files else 'None'}"
    
    def test_extract_markdown_table(self, extractor):
        """Test extraction of markdown tables as CSV."""
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
        assert "Name,Age,City" in files[0].content
        assert "John,30,NYC" in files[0].content
    
    def test_extract_markdown_table_with_title(self, extractor):
        """Test extraction of markdown table with title."""
        text = '''
        ## User Data Summary
        
        | Name | Age |
        |------|-----|
        | Bob  | 35  |
        '''
        
        files = extractor.extract_files(text)
        assert len(files) == 1
        assert files[0].filename == "User_Data_Summary.csv"
    
    def test_validate_json_content(self, extractor):
        """Test JSON validation."""
        valid_file = ExtractedFile(
            filename="test.json",
            content='{"name": "test", "value": 123}',
            language="json",
            start_pos=0,
            end_pos=0
        )
        assert extractor.validate_content(valid_file) is None
        
        invalid_file = ExtractedFile(
            filename="test.json",
            content='{"name": "test" invalid json}',
            language="json",
            start_pos=0,
            end_pos=0
        )
        error = extractor.validate_content(invalid_file)
        assert error is not None
        assert "Invalid JSON" in error
    
    def test_validate_yaml_content(self, extractor):
        """Test YAML validation."""
        valid_file = ExtractedFile(
            filename="test.yaml",
            content='name: test\nvalue: 123',
            language="yaml",
            start_pos=0,
            end_pos=0
        )
        assert extractor.validate_content(valid_file) is None
        
        invalid_file = ExtractedFile(
            filename="test.yaml",
            content='name: test\n  invalid: yaml:',
            language="yaml",
            start_pos=0,
            end_pos=0
        )
        error = extractor.validate_content(invalid_file)
        assert error is not None
        assert "Invalid YAML" in error
    
    def test_validate_csv_content(self, extractor):
        """Test CSV validation."""
        valid_file = ExtractedFile(
            filename="test.csv",
            content='name,age\nJohn,30\nJane,25',
            language="csv",
            start_pos=0,
            end_pos=0
        )
        assert extractor.validate_content(valid_file) is None
        
        invalid_file = ExtractedFile(
            filename="test.csv",
            content='name,age\nJohn,30,extra\nJane,25',
            language="csv",
            start_pos=0,
            end_pos=0
        )
        error = extractor.validate_content(invalid_file)
        assert error is not None
        assert "columns" in error
    
    def test_validate_package_json(self, extractor):
        """Test package.json validation."""
        valid_file = ExtractedFile(
            filename="package.json",
            content='{"name": "test-package", "version": "1.0.0"}',
            language="json",
            start_pos=0,
            end_pos=0
        )
        assert extractor.validate_content(valid_file) is None
        
        # Test missing name field
        invalid_file = ExtractedFile(
            filename="package.json",
            content='{"version": "1.0.0", "description": "missing name field"}',
            language="json",
            start_pos=0,
            end_pos=0
        )
        error = extractor.validate_content(invalid_file)
        assert error is not None
        assert "Missing 'name'" in error
        
        # Test missing version field
        invalid_file2 = ExtractedFile(
            filename="package.json",
            content='{"name": "test-package", "description": "missing version field"}',
            language="json",
            start_pos=0,
            end_pos=0
        )
        error2 = extractor.validate_content(invalid_file2)
        assert error2 is not None
        assert "Missing 'version'" in error2
    
    def test_validate_terraform_file(self, extractor):
        """Test Terraform file validation."""
        valid_file = ExtractedFile(
            filename="main.tf",
            content='resource "aws_instance" "example" {\n  ami = "ami-123"\n}',
            language="terraform",
            start_pos=0,
            end_pos=0
        )
        assert extractor.validate_content(valid_file) is None
        
        invalid_file = ExtractedFile(
            filename="main.tf",
            content='# Just a comment, no terraform content',
            language="terraform",
            start_pos=0,
            end_pos=0
        )
        error = extractor.validate_content(invalid_file)
        assert error is not None
        assert "Terraform configuration blocks" in error
    
    def test_empty_code_block(self, extractor):
        """Test that empty code blocks are skipped."""
        text = '''
        ```python
        
        ```
        '''
        files = extractor.extract_files(text)
        assert len(files) == 0
    
    def test_file_size_limit(self, extractor):
        """Test file size validation."""
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
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
    
    def test_language_extension_mapping(self, extractor):
        """Test various language to extension mappings."""
        mappings = {
            'javascript': '.js',
            'typescript': '.ts',
            'python': '.py',
            'rust': '.rs',
            'go': '.go',
            'dart': '.dart',
            'scala': '.scala',
            'clj': '.clj',  # Use 'clj' instead of 'clojure' since that's what's in LANGUAGE_EXTENSIONS
        }
        
        for lang, ext in mappings.items():
            text = f'''```{lang}
content
```'''
            files = extractor.extract_files(text)
            assert len(files) == 1
            assert files[0].filename.endswith(ext), f"Expected {ext} for {lang}"
    
    def test_csv_content_cleaning(self, extractor):
        """Test CSV content is properly cleaned."""
        text = '''
        ```csv
          name,age  
          John,30  
          Jane,25  
        ```
        '''
        files = extractor.extract_files(text)
        assert len(files) == 1
        # Check that lines are stripped but structure preserved
        assert files[0].content == "name,age\nJohn,30\nJane,25"
    
    def test_filename_sanitization(self, extractor):
        """Test filename sanitization."""
        # Test internal method
        assert extractor._sanitize_filename("file<>name.txt") == "filename.txt"
        assert extractor._sanitize_filename("file:name|test.txt") == "filenametest.txt"
        assert extractor._sanitize_filename("file name.txt") == "file_name.txt"
        assert extractor._sanitize_filename("a" * 200 + ".txt")[:100] == "a" * 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])