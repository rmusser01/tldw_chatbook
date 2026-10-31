"""
Unit tests for file extraction functionality.
"""

import re

import pytest

from tldw_chatbook.Utils.file_extraction import FileExtractor, ExtractedFile


class TestFileExtractor:
    """Test cases for the FileExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a FileExtractor instance for testing."""
        return FileExtractor()

    def test_extract_simple_code_block(self, extractor):
        """Test extraction of a simple code block."""
        text = """Here's a Python script:
```python
print("Hello, World!")
```
"""

        files = extractor.extract_files(text)
        assert len(files) == 1
        assert files[0].filename == "extracted_1.py"
        assert files[0].content == 'print("Hello, World!")'
        assert files[0].language == "python"

    def test_extract_multiple_code_blocks(self, extractor):
        """Test extraction of multiple code blocks."""
        text = """
        First file:
        ```javascript
        console.log("Hello");
        ```
        
        Second file:
        ```python
        print("World")
        ```
        """

        files = extractor.extract_files(text)
        assert len(files) == 2
        assert files[0].filename == "extracted_1.js"
        assert files[0].language == "javascript"
        assert files[1].filename == "extracted_2.py"
        assert files[1].language == "python"

    def test_extract_with_filename_hint(self, extractor):
        """Test extraction with filename hints in comments."""
        text = """
        # main.py
        ```python
        def main():
            pass
        ```
        """

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
            text = f"""```{lang}
test content
```"""
            files = extractor.extract_files(text)
            assert len(files) == 1, f"Expected 1 file for {lang}, got {len(files)}"
            assert files[0].filename == expected_filename, (
                f"Expected {expected_filename} for {lang}, got {files[0].filename if files else 'None'}"
            )

    def test_extract_markdown_table(self, extractor):
        """Test extraction of markdown tables as CSV."""
        text = """
        Here's a data table:
        
        | Name | Age | City |
        |------|-----|------|
        | John | 30  | NYC  |
        | Jane | 25  | LA   |
        """

        files = extractor.extract_files(text)
        assert len(files) == 1
        assert files[0].filename == "table_1.csv"
        assert files[0].language == "csv"
        assert "Name,Age,City" in files[0].content
        assert "John,30,NYC" in files[0].content

    def test_extract_markdown_table_with_title(self, extractor):
        """Test extraction of markdown table with title."""
        text = """
        ## User Data Summary
        
        | Name | Age |
        |------|-----|
        | Bob  | 35  |
        """

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
            end_pos=0,
        )
        assert extractor.validate_content(valid_file) is None

        invalid_file = ExtractedFile(
            filename="test.json",
            content='{"name": "test" invalid json}',
            language="json",
            start_pos=0,
            end_pos=0,
        )
        error = extractor.validate_content(invalid_file)
        assert error is not None
        assert "Invalid JSON" in error

    def test_validate_yaml_content(self, extractor):
        """Test YAML validation."""
        valid_file = ExtractedFile(
            filename="test.yaml",
            content="name: test\nvalue: 123",
            language="yaml",
            start_pos=0,
            end_pos=0,
        )
        assert extractor.validate_content(valid_file) is None

        invalid_file = ExtractedFile(
            filename="test.yaml",
            content="name: test\n  invalid: yaml:",
            language="yaml",
            start_pos=0,
            end_pos=0,
        )
        error = extractor.validate_content(invalid_file)
        assert error is not None
        assert "Invalid YAML" in error

    def test_validate_csv_content(self, extractor):
        """Test CSV validation."""
        valid_file = ExtractedFile(
            filename="test.csv",
            content="name,age\nJohn,30\nJane,25",
            language="csv",
            start_pos=0,
            end_pos=0,
        )
        assert extractor.validate_content(valid_file) is None

        invalid_file = ExtractedFile(
            filename="test.csv",
            content="name,age\nJohn,30,extra\nJane,25",
            language="csv",
            start_pos=0,
            end_pos=0,
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
            end_pos=0,
        )
        assert extractor.validate_content(valid_file) is None

        # Test missing name field
        invalid_file = ExtractedFile(
            filename="package.json",
            content='{"version": "1.0.0", "description": "missing name field"}',
            language="json",
            start_pos=0,
            end_pos=0,
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
            end_pos=0,
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
            end_pos=0,
        )
        assert extractor.validate_content(valid_file) is None

        invalid_file = ExtractedFile(
            filename="main.tf",
            content="# Just a comment, no terraform content",
            language="terraform",
            start_pos=0,
            end_pos=0,
        )
        error = extractor.validate_content(invalid_file)
        assert error is not None
        assert "Terraform configuration blocks" in error

    def test_empty_code_block(self, extractor):
        """Test that empty code blocks are skipped."""
        text = """
        ```python
        
        ```
        """
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
            end_pos=0,
        )
        error = extractor.validate_content(large_file)
        assert error is not None
        assert "too large" in error

    def test_language_extension_mapping(self, extractor):
        """Test various language to extension mappings."""
        mappings = {
            "javascript": ".js",
            "typescript": ".ts",
            "python": ".py",
            "rust": ".rs",
            "go": ".go",
            "dart": ".dart",
            "scala": ".scala",
            "clj": ".clj",  # Use 'clj' instead of 'clojure' since that's what's in LANGUAGE_EXTENSIONS
        }

        for lang, ext in mappings.items():
            text = f"""```{lang}
content
```"""
            files = extractor.extract_files(text)
            assert len(files) == 1
            assert files[0].filename.endswith(ext), f"Expected {ext} for {lang}"

    def test_csv_content_preservation(self, extractor):
        """Test fenced CSV content preserves file-significant whitespace."""
        text = """
        ```csv
          name,age  
          John,30  
          Jane,25  
        ```
        """
        files = extractor.extract_files(text)
        assert len(files) == 1
        assert (
            files[0].content
            == "          name,age  \n          John,30  \n          Jane,25  \n        "
        )

    def test_filename_sanitization(self, extractor):
        """Test filename sanitization."""
        # Test internal method
        assert extractor._sanitize_filename("file<>name.txt") == "filename.txt"
        assert extractor._sanitize_filename("file:name|test.txt") == "filenametest.txt"
        assert extractor._sanitize_filename("file name.txt") == "file_name.txt"
        assert extractor._sanitize_filename("a" * 200 + ".txt")[:100] == "a" * 100


class TestFindFilenameHintWindow:
    """TASK-34394: `_find_filename_hint` scans only the last few lines
    before a code block (windowed backward `rfind` walk) instead of
    slicing + splitting the whole text prefix."""

    CODE_BLOCK_START = len("prefix")

    @pytest.fixture
    def extractor(self):
        """Create a FileExtractor instance for testing."""
        return FileExtractor()

    def _text_with_block_at(self, prefix: str) -> "tuple[str, int]":
        """Return (text, code_block_start) for a code block directly
        after `prefix`."""
        return prefix + "```python\n", len(prefix)

    def test_hint_on_immediately_preceding_line(self, extractor):
        """A hint on the line right before the block is found."""
        prefix = "some prose\nfilename: app.py\n"
        text, start = self._text_with_block_at(prefix)
        assert extractor._find_filename_hint(text, start) == "app.py"

    def test_hint_within_window_lines_above(self, extractor):
        """A hint several lines above the block (but inside the 5-line
        window) is still found."""
        prefix = "line1\nfilename: helper.py\nline3\nline4\nline5\n"
        text, start = self._text_with_block_at(prefix)
        assert extractor._find_filename_hint(text, start) == "helper.py"

    def test_hint_on_first_line_of_exact_window(self, extractor):
        """With exactly `search_lines` newline-separated segments of
        prefix, the first segment is inside the window."""
        prefix = "# top.py\nb\nc\nd\n"
        text, start = self._text_with_block_at(prefix)
        assert extractor._find_filename_hint(text, start) == "top.py"

    def test_no_hint_returns_none(self, extractor):
        """No hint anywhere in the window returns None."""
        prefix = "just some prose\nwith no filenames at all\n"
        text, start = self._text_with_block_at(prefix)
        assert extractor._find_filename_hint(text, start) is None

    def test_hint_outside_window_is_not_found(self, extractor):
        """A hint more than `search_lines` lines above the block falls
        outside the window and is not found (same bound as the old
        `lines[-search_lines:]` slice)."""
        prefix = "filename: old.py\nl2\nl3\nl4\nl5\nl6\nl7\n"
        text, start = self._text_with_block_at(prefix)
        assert extractor._find_filename_hint(text, start) is None

    def test_empty_prefix_returns_none(self, extractor):
        """A code block at position 0 has no preceding text."""
        text = "```python\npass\n```\n"
        assert extractor._find_filename_hint(text, 0) is None

    @pytest.mark.parametrize(
        "prefix",
        [
            "filename: a.py\n",
            "x\nfilename: b.py\n",
            "a\nb\nc\n",
            "a\nb\nc\nd\ne\nf\nfilename: g.py\n",
            "no hint here\n",
            "",
            "l1\nl2\nl3\nl4\n# tail.py\n",
        ],
    )
    def test_window_matches_legacy_full_prefix_slice(self, extractor, prefix):
        """The windowed scan must match the legacy whole-prefix behavior:
        the last `search_lines` newline-separated segments of the prefix.
        Computes the legacy slice independently and compares verdicts."""
        text, start = self._text_with_block_at(prefix)

        # Legacy reference: slice + split + join of the last N lines.
        before_text = text[:start]
        lines = before_text.split("\n")
        legacy_search_text = (
            "\n".join(lines[-5:]) if len(lines) >= 5 else before_text
        )
        legacy_hint = None
        for pattern in FileExtractor.FILENAME_HINTS:
            matches = re.findall(pattern, legacy_search_text)
            if matches:
                candidate = matches[-1].strip()
                if extractor._is_valid_filename(candidate):
                    legacy_hint = candidate
                    break

        assert extractor._find_filename_hint(text, start) == legacy_hint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
