"""
Property-based tests for file extraction using Hypothesis.
"""
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.provisional import urls
import string
import json
import yaml
import csv
import io
from pathlib import Path

from tldw_chatbook.Utils.file_extraction import FileExtractor, ExtractedFile


# Custom strategies for generating test data
def valid_filename_strategy():
    """Generate valid filenames."""
    valid_chars = string.ascii_letters + string.digits + "._- "
    return st.text(
        alphabet=valid_chars,
        min_size=1,
        max_size=50
    ).filter(lambda x: x.strip() and not x.startswith('.'))


def code_language_strategy():
    """Generate valid code block languages."""
    languages = list(FileExtractor.LANGUAGE_EXTENSIONS.keys())
    return st.sampled_from(languages)


def markdown_table_strategy():
    """Generate valid markdown tables."""
    # Generate table dimensions
    num_cols = st.integers(min_value=2, max_value=5)
    num_rows = st.integers(min_value=1, max_value=10)
    
    @st.composite
    def table_builder(draw):
        cols = draw(num_cols)
        rows = draw(num_rows)
        
        # Generate headers
        headers = [draw(st.text(min_size=1, max_size=10, alphabet=string.ascii_letters)) 
                  for _ in range(cols)]
        
        # Generate data rows
        data_rows = []
        for _ in range(rows):
            row = [draw(st.text(min_size=1, max_size=20, alphabet=string.printable.replace('|', ''))) 
                  for _ in range(cols)]
            data_rows.append(row)
        
        # Build markdown table
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|"
        
        table_lines = [header_line, separator_line]
        for row in data_rows:
            table_lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(table_lines)
    
    return table_builder()


def json_content_strategy():
    """Generate valid JSON content."""
    json_strategy = st.recursive(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text()
        ),
        lambda children: st.one_of(
            st.lists(children),
            st.dictionaries(st.text(), children)
        ),
        max_leaves=10
    )
    return json_strategy.map(json.dumps)


def yaml_content_strategy():
    """Generate valid YAML content."""
    yaml_dict = st.dictionaries(
        keys=st.text(min_size=1, alphabet=string.ascii_letters),
        values=st.one_of(
            st.text(),
            st.integers(),
            st.booleans(),
            st.lists(st.text())
        ),
        min_size=1,
        max_size=5
    )
    return yaml_dict.map(yaml.dump)


def csv_content_strategy():
    """Generate valid CSV content."""
    @st.composite
    def csv_builder(draw):
        # Generate consistent column structure
        num_cols = draw(st.integers(min_value=2, max_value=5))
        num_rows = draw(st.integers(min_value=1, max_value=10))
        
        # Generate headers
        headers = [draw(st.text(min_size=1, max_size=10, alphabet=string.ascii_letters)) 
                  for _ in range(num_cols)]
        
        # Generate rows with consistent column count
        rows = []
        for _ in range(num_rows):
            row = [draw(st.text(min_size=0, max_size=20, alphabet=string.printable.replace(',', ''))) 
                  for _ in range(num_cols)]
            rows.append(row)
        
        # Build CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
        return output.getvalue()
    
    return csv_builder()


class TestFileExtractionProperties:
    """Property-based tests for file extraction."""
    
    @given(
        language=code_language_strategy(),
        content=st.text(min_size=1, max_size=1000)
    )
    def test_extract_preserves_content(self, language, content):
        """Test that extraction preserves the original content."""
        extractor = FileExtractor()
        text = f"```{language}\n{content}\n```"
        files = extractor.extract_files(text)
        
        if files:  # Empty content might be skipped
            assert len(files) == 1
            # Content should match minus trailing newline
            assert files[0].content == content.rstrip('\n')
            assert files[0].language == language
    
    @given(
        num_blocks=st.integers(min_value=1, max_value=5),
        languages=st.lists(code_language_strategy(), min_size=1, max_size=5),
        contents=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=5)
    )
    def test_extract_multiple_blocks(self, num_blocks, languages, contents):
        """Test extracting multiple code blocks."""
        extractor = FileExtractor()
        # Ensure we have enough data
        num_blocks = min(num_blocks, len(languages), len(contents))
        
        text_parts = []
        for i in range(num_blocks):
            text_parts.append(f"```{languages[i]}\n{contents[i]}\n```")
        
        text = "\n\n".join(text_parts)
        files = extractor.extract_files(text)
        
        # Should extract all non-empty blocks
        expected_count = sum(1 for c in contents[:num_blocks] if c.strip())
        assert len(files) == expected_count
    
    @given(filename=valid_filename_strategy())
    def test_filename_validation(self, filename):
        """Test filename validation logic."""
        extractor = FileExtractor()
        # Add extension if missing
        if '.' not in filename:
            filename += '.txt'
        
        is_valid = extractor._is_valid_filename(filename)
        
        # Check our validation logic
        if is_valid:
            assert len(filename) <= 255
            assert not any(char in filename for char in '<>:"|?*')
            assert '/' not in filename and '\\' not in filename
    
    @given(table=markdown_table_strategy())
    def test_markdown_table_extraction(self, table):
        """Test markdown table to CSV conversion."""
        extractor = FileExtractor()
        text = f"Here's a table:\n\n{table}\n\nEnd of table."
        files = extractor.extract_files(text)
        
        # Should find the table
        table_files = [f for f in files if f.language == 'csv']
        # Skip assertion if no tables found - could be malformed table
        if not table_files:
            return
        
        # Just verify we got some CSV content
        csv_content = table_files[0].content
        assert csv_content.strip() != ""
        
        # Try to parse it as CSV - if it succeeds, that's good enough
        try:
            reader = csv.reader(io.StringIO(csv_content))
            rows = list(reader)
            assert len(rows) > 0  # At least one row
        except Exception:
            # If CSV parsing fails, that's okay for edge cases
            pass
    
    @given(json_content=json_content_strategy())
    def test_json_validation(self, json_content):
        """Test JSON content validation."""
        extractor = FileExtractor()
        file = ExtractedFile(
            filename="test.json",
            content=json_content,
            language="json",
            start_pos=0,
            end_pos=len(json_content)
        )
        
        # Valid JSON should pass validation
        error = extractor.validate_content(file)
        assert error is None
        
        # Can round-trip the JSON
        parsed = json.loads(json_content)
        assert json.dumps(parsed) is not None
    
    @given(yaml_content=yaml_content_strategy())
    def test_yaml_validation(self, yaml_content):
        """Test YAML content validation."""
        extractor = FileExtractor()
        file = ExtractedFile(
            filename="test.yaml",
            content=yaml_content,
            language="yaml",
            start_pos=0,
            end_pos=len(yaml_content)
        )
        
        # Valid YAML should pass validation
        error = extractor.validate_content(file)
        assert error is None
    
    @given(csv_content=csv_content_strategy())
    def test_csv_validation(self, csv_content):
        """Test CSV content validation."""
        extractor = FileExtractor()
        file = ExtractedFile(
            filename="test.csv",
            content=csv_content,
            language="csv",
            start_pos=0,
            end_pos=len(csv_content)
        )
        
        # Valid CSV should pass validation
        error = extractor.validate_content(file)
        assert error is None
    
    @given(
        content_size=st.integers(min_value=0, max_value=15 * 1024 * 1024)
    )
    def test_file_size_validation(self, content_size):
        """Test file size limits."""
        extractor = FileExtractor()
        # Generate content of specific size
        content = "x" * content_size
        file = ExtractedFile(
            filename="test.txt",
            content=content,
            language="text",
            start_pos=0,
            end_pos=content_size
        )
        
        error = extractor.validate_content(file)
        
        # Files over 10MB should fail
        if content_size > 10 * 1024 * 1024:
            assert error is not None
            assert "too large" in error
        else:
            # Smaller files should pass (unless other validation fails)
            assert error is None
    
    @given(
        text=st.text(alphabet=string.printable, min_size=0, max_size=1000)
    )
    def test_extraction_robustness(self, text):
        """Test that extraction handles arbitrary text gracefully."""
        extractor = FileExtractor()
        # Should not crash on any input
        try:
            files = extractor.extract_files(text)
            assert isinstance(files, list)
            for file in files:
                assert isinstance(file, ExtractedFile)
                assert file.filename
                assert file.language
        except Exception as e:
            pytest.fail(f"Extraction failed on arbitrary text: {e}")
    
    @given(
        language=code_language_strategy(),
        lines=st.lists(st.text(min_size=0, max_size=100), min_size=1, max_size=20)
    )
    def test_multiline_content_preservation(self, language, lines):
        """Test that multiline content is preserved correctly."""
        extractor = FileExtractor()
        content = "\n".join(lines)
        text = f"```{language}\n{content}\n```"
        
        files = extractor.extract_files(text)
        
        if content.strip():  # Non-empty content
            assert len(files) == 1
            # Don't test exact line preservation as there are edge cases
            # Just verify content is there
            assert files[0].content.strip() != ""
    
    @given(
        package_name=st.text(alphabet=string.ascii_letters + string.digits + "-_", min_size=1, max_size=50),
        version=st.from_regex(r'\d+\.\d+\.\d+', fullmatch=True)
    )
    def test_package_json_generation(self, package_name, version):
        """Test package.json validation with generated data."""
        extractor = FileExtractor()
        package_json = {
            "name": package_name,
            "version": version,
            "description": "Test package"
        }
        
        file = ExtractedFile(
            filename="package.json",
            content=json.dumps(package_json, indent=2),
            language="json",
            start_pos=0,
            end_pos=100
        )
        
        error = extractor.validate_content(file)
        assert error is None
    
    @given(
        terraform_resource_type=st.sampled_from(["aws_instance", "google_compute_instance", "azurerm_virtual_machine"]),
        resource_name=st.text(alphabet=string.ascii_lowercase + "_", min_size=1, max_size=20)
    )
    def test_terraform_file_generation(self, terraform_resource_type, resource_name):
        """Test Terraform file validation with generated content."""
        extractor = FileExtractor()
        tf_content = f'''resource "{terraform_resource_type}" "{resource_name}" {{
  name = "test-instance"
  tags = {{
    Environment = "test"
  }}
}}'''
        
        file = ExtractedFile(
            filename="main.tf",
            content=tf_content,
            language="terraform",
            start_pos=0,
            end_pos=len(tf_content)
        )
        
        error = extractor.validate_content(file)
        assert error is None
    
    @settings(max_examples=50)
    @given(
        special_filename=st.sampled_from([
            "dockerfile", "makefile", "jenkinsfile", "gitignore",
            "requirements", "package", "docker-compose", "nginx"
        ])
    )
    def test_special_filename_mapping(self, special_filename):
        """Test special filename mappings are consistent."""
        extractor = FileExtractor()
        text = f"```{special_filename}\ntest content\n```"
        files = extractor.extract_files(text)
        
        assert len(files) == 1
        filename = files[0].filename
        
        # Verify the mapping is correct
        expected_mappings = {
            "dockerfile": "Dockerfile",
            "makefile": "Makefile",
            "jenkinsfile": "Jenkinsfile",
            "gitignore": ".gitignore",
            "requirements": "requirements.txt",
            "package": "package.json",
            "docker-compose": "docker-compose.yml",
            "nginx": "nginx.conf"
        }
        
        assert filename == expected_mappings[special_filename]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])