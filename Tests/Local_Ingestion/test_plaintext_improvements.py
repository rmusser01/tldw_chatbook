# test_plaintext_improvements.py
"""
Test suite for plaintext ingestion improvements.
Tests new features like async operations, caching, and new format support.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
import pytest
import yaml

from tldw_chatbook.Local_Ingestion.Plaintext_Files import (
    process_document_content,
    process_document_with_improvements,
    convert_document_to_text,
    SUPPORTED_FORMATS
)

# Import improvements to test directly
try:
    from tldw_chatbook.Local_Ingestion.async_file_utils import (
        detect_encoding_async,
        read_file_async,
        stream_file_content
    )
    from tldw_chatbook.Local_Ingestion.content_cache import ContentCache
    from tldw_chatbook.Local_Ingestion.format_converters import (
        convert_file,
        get_supported_formats,
        can_convert_file
    )
    IMPROVEMENTS_AVAILABLE = True
except ImportError:
    IMPROVEMENTS_AVAILABLE = False
    pytest.skip("Improvements not available", allow_module_level=True)


class TestAsyncFileUtils:
    """Test async file operations."""
    
    @pytest.mark.asyncio
    async def test_detect_encoding(self, tmp_path):
        """Test encoding detection."""
        # UTF-8 file
        utf8_file = tmp_path / "utf8.txt"
        utf8_file.write_text("Hello, 世界! 🌍", encoding='utf-8')
        
        encoding = await detect_encoding_async(utf8_file)
        assert encoding == 'utf-8'
        
        # Latin-1 file
        latin1_file = tmp_path / "latin1.txt"
        latin1_file.write_bytes("Café".encode('latin-1'))
        
        encoding = await detect_encoding_async(latin1_file)
        assert encoding in ['latin-1', 'ISO-8859-1', 'windows-1252']
    
    @pytest.mark.asyncio
    async def test_read_file_async(self, tmp_path):
        """Test async file reading."""
        test_file = tmp_path / "test.txt"
        test_content = "Test content\nLine 2\nLine 3"
        test_file.write_text(test_content)
        
        content = await read_file_async(test_file)
        assert content == test_content
    
    @pytest.mark.asyncio
    async def test_stream_file_content(self, tmp_path):
        """Test file streaming."""
        test_file = tmp_path / "large.txt"
        test_content = "x" * 10000  # 10KB of content
        test_file.write_text(test_content)
        
        chunks = []
        async for chunk in stream_file_content(test_file, chunk_size=1000):
            chunks.append(chunk)
        
        assert len(chunks) == 10
        assert ''.join(chunks) == test_content


class TestContentCache:
    """Test content caching functionality."""
    
    def test_cache_basic_operations(self, tmp_path):
        """Test basic cache operations."""
        cache = ContentCache(cache_dir=tmp_path / "cache", ttl_hours=1)
        
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        options = {'chunk_size': 100}
        result = {'status': 'Success', 'content': 'Test content'}
        
        # Test cache miss
        assert cache.get(test_file, options) is None
        
        # Test cache set
        cache.set(test_file, options, result)
        
        # Test cache hit
        cached = cache.get(test_file, options)
        assert cached == result
        
        # Test different options give different cache
        different_options = {'chunk_size': 200}
        assert cache.get(test_file, different_options) is None
    
    def test_cache_expiration(self, tmp_path):
        """Test cache TTL."""
        # Create cache with very short TTL
        cache = ContentCache(cache_dir=tmp_path / "cache", ttl_hours=0.0001)  # ~0.36 seconds
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        cache.set(test_file, {}, {'result': 'data'})
        
        # Should be cached immediately
        assert cache.get(test_file, {}) is not None
        
        # Wait for expiration
        time.sleep(0.5)
        
        # Should be expired
        assert cache.get(test_file, {}) is None
    
    def test_cache_size_limit(self, tmp_path):
        """Test cache size limiting."""
        # Create cache with small size limit
        cache = ContentCache(cache_dir=tmp_path / "cache", max_cache_size_mb=0.001)  # 1KB
        
        # Add multiple items that exceed limit
        for i in range(10):
            test_file = tmp_path / f"test{i}.txt"
            test_file.write_text(f"Content {i}")
            
            # Large result to exceed cache size
            large_result = {'data': 'x' * 1000}
            cache.set(test_file, {}, large_result)
        
        # Check that old entries were removed
        stats = cache.get_stats()
        assert stats['total_entries'] < 10


class TestFormatConverters:
    """Test new format conversion capabilities."""
    
    def test_json_conversion(self, tmp_path):
        """Test JSON file conversion."""
        json_file = tmp_path / "test.json"
        test_data = {"name": "Test", "value": 42, "items": [1, 2, 3]}
        json_file.write_text(json.dumps(test_data))
        
        text, metadata = convert_file(json_file)
        
        assert '"name": "Test"' in text
        assert metadata['format'] == 'json'
        assert metadata['keys'] == ['name', 'value', 'items']
    
    def test_csv_conversion(self, tmp_path):
        """Test CSV file conversion."""
        csv_file = tmp_path / "test.csv"
        csv_content = "Name,Age,City\nAlice,30,NYC\nBob,25,LA"
        csv_file.write_text(csv_content)
        
        text, metadata = convert_file(csv_file)
        
        # Check markdown table format
        assert '| Name | Age | City |' in text
        assert '|---|---|---|' in text
        assert '| Alice | 30 | NYC |' in text
        
        assert metadata['format'] == 'csv'
        assert metadata['columns'] == ['Name', 'Age', 'City']
        assert metadata['row_count'] == 2
    
    def test_yaml_conversion(self, tmp_path):
        """Test YAML file conversion."""
        yaml_file = tmp_path / "test.yaml"
        test_data = {"server": {"host": "localhost", "port": 8080}}
        yaml_file.write_text(yaml.dump(test_data))
        
        text, metadata = convert_file(yaml_file)
        
        assert 'server:' in text
        assert 'host: localhost' in text
        assert metadata['format'] == 'yaml'
    
    def test_supported_formats(self):
        """Test format support detection."""
        formats = get_supported_formats()
        
        # Should have at least these converters
        assert 'JSONConverter' in formats
        assert 'CSVConverter' in formats
        assert 'YAMLConverter' in formats
        
        # Check extensions
        assert '.json' in formats['JSONConverter']
        assert '.csv' in formats['CSVConverter']
        assert '.yaml' in formats['YAMLConverter']


class TestPlaintextImprovements:
    """Test the integrated improvements in Plaintext_Files.py."""
    
    def test_process_with_improvements_json(self, tmp_path):
        """Test processing JSON file with improvements."""
        json_file = tmp_path / "data.json"
        json_file.write_text('{"title": "Test Doc", "content": "Some content"}')
        
        result = process_document_with_improvements(
            json_file,
            perform_chunking=False,
            use_cache=False
        )
        
        assert result['status'] == 'Success'
        assert result['content'] is not None
        assert 'title' in result['content']
    
    def test_process_with_cache(self, tmp_path):
        """Test that caching works."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for caching")
        
        # First call - should process
        start = time.time()
        result1 = process_document_with_improvements(
            test_file,
            perform_chunking=False,
            use_cache=True
        )
        first_duration = time.time() - start
        
        # Second call - should use cache
        start = time.time()
        result2 = process_document_with_improvements(
            test_file,
            perform_chunking=False,
            use_cache=True
        )
        cached_duration = time.time() - start
        
        assert result1 == result2
        # Cached call should be much faster (at least 50% faster)
        # Note: This might be flaky on slow systems
        assert cached_duration < first_duration * 0.5
    
    def test_backwards_compatibility(self, tmp_path):
        """Test that old function still works."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Backwards compatibility test")
        
        # Old function should still work
        result = process_document_content(
            test_file,
            perform_chunking=False,
            chunk_options=None,
            perform_analysis=False,
            summarize_recursively=False,
            api_name=None,
            api_key=None,
            custom_prompt=None,
            system_prompt=None
        )
        
        assert result['status'] == 'Success'
        assert result['content'] == "Backwards compatibility test"
    
    def test_supported_formats_list(self):
        """Test the supported formats constant."""
        assert '.txt' in SUPPORTED_FORMATS['standard']
        assert '.json' in SUPPORTED_FORMATS['enhanced']
        assert '.csv' in SUPPORTED_FORMATS['enhanced']


@pytest.mark.asyncio
async def test_async_improvements(tmp_path):
    """Test async processing directly."""
    from tldw_chatbook.Local_Ingestion.Plaintext_Files import process_document_content_async
    
    test_file = tmp_path / "async_test.txt"
    test_file.write_text("Async test content")
    
    result = await process_document_content_async(
        test_file,
        perform_chunking=False,
        chunk_options=None,
        perform_analysis=False,
        summarize_recursively=False,
        api_name=None,
        api_key=None,
        custom_prompt=None,
        system_prompt=None,
        use_cache=False
    )
    
    assert result['status'] == 'Success'
    assert result['content'] == "Async test content"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])