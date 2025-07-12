# Tests/tldw_api/test_plaintext_schema.py
"""
Tests for plaintext schema and request models.
"""

import pytest
from pydantic import ValidationError

from tldw_chatbook.tldw_api.schemas import ProcessPlaintextRequest, MediaType


class TestPlaintextSchema:
    """Test plaintext schema definitions."""
    
    def test_media_type_includes_plaintext(self):
        """Test that MediaType literal includes 'plaintext'."""
        # This will raise an error if 'plaintext' is not a valid MediaType
        assert "plaintext" in MediaType.__args__
    
    def test_process_plaintext_request_defaults(self):
        """Test ProcessPlaintextRequest with default values."""
        request = ProcessPlaintextRequest()
        
        assert request.encoding == "utf-8"
        assert request.line_ending == "auto"
        assert request.remove_extra_whitespace is True
        assert request.convert_to_paragraphs is False
        assert request.split_pattern is None
        assert request.chunk_method == "paragraphs"  # Default for plaintext
    
    def test_process_plaintext_request_with_values(self):
        """Test ProcessPlaintextRequest with custom values."""
        request = ProcessPlaintextRequest(
            urls=["http://example.com/test.txt"],
            title="Test Document",
            author="Test Author",
            keywords=["test", "plaintext"],
            encoding="latin-1",
            line_ending="crlf",
            remove_extra_whitespace=False,
            convert_to_paragraphs=True,
            split_pattern=r"\n\n+",
            chunk_method="sentences",
            chunk_size=1000,
            chunk_overlap=100
        )
        
        assert request.encoding == "latin-1"
        assert request.line_ending == "crlf"
        assert request.remove_extra_whitespace is False
        assert request.convert_to_paragraphs is True
        assert request.split_pattern == r"\n\n+"
        assert request.chunk_method == "sentences"
        assert request.chunk_size == 1000
        assert request.chunk_overlap == 100
    
    def test_process_plaintext_request_validation(self):
        """Test ProcessPlaintextRequest validation."""
        # Test invalid encoding
        with pytest.raises(ValidationError) as exc_info:
            ProcessPlaintextRequest(encoding="invalid-encoding")
        assert "encoding" in str(exc_info.value)
        
        # Test invalid line ending
        with pytest.raises(ValidationError) as exc_info:
            ProcessPlaintextRequest(line_ending="invalid")
        assert "line_ending" in str(exc_info.value)
        
        # Test invalid chunk method
        with pytest.raises(ValidationError) as exc_info:
            ProcessPlaintextRequest(chunk_method="invalid-method")
        assert "chunk_method" in str(exc_info.value)
    
    def test_process_plaintext_inherits_base_fields(self):
        """Test that ProcessPlaintextRequest inherits from BaseMediaRequest."""
        request = ProcessPlaintextRequest(
            custom_prompt="Summarize this text",
            system_prompt="You are a helpful assistant",
            perform_analysis=True,
            api_name="openai",
            api_key="test-key"
        )
        
        # These fields come from BaseMediaRequest
        assert request.custom_prompt == "Summarize this text"
        assert request.system_prompt == "You are a helpful assistant"
        assert request.perform_analysis is True
        assert request.api_name == "openai"
        assert request.api_key == "test-key"