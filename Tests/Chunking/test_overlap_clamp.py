import pytest

from tldw_chatbook.Chunking.engine import Chunker
import tldw_chatbook.Chunking._shims.config as _cfg_mod


# Keep regex operations snappy via config patch
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


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_ebook_chapters_overlap_ge_maxsize_does_not_hang():
    """Ensure ebook_chapters chunking makes forward progress when overlap >= max_size.

    This guards against infinite/negative-step loops and verifies that we return chunks.
    """
    chunker = Chunker()
    text = " ".join(["word"] * 1000)  # No chapter markers; triggers size-based split

    # overlap greater than max_size previously caused non-progressing loop
    chunks = chunker.chunk_text(text, method="ebook_chapters", max_size=50, overlap=200)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    # Ensure chunks are not degenerate
    assert all(isinstance(c, str) and c for c in chunks)


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_ebook_chapters_with_metadata_overlap_ge_maxsize_does_not_hang():
    """Same safety check for the with_metadata path."""
    chunker = Chunker()
    text = " ".join(["word"] * 1000)

    results = chunker.chunk_text_with_metadata(text, method="ebook_chapters", max_size=50, overlap=200)

    assert isinstance(results, list)
    assert len(results) > 0
    # Basic shape check
    first = results[0]
    assert hasattr(first, "text") and isinstance(first.text, str)
    assert hasattr(first, "metadata") and hasattr(first.metadata, "word_count")
