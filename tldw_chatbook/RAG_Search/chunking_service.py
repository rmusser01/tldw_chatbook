"""
Simple chunking service wrapper for the simplified RAG implementation.

This provides a minimal interface to the existing chunking functionality
to satisfy the import requirements of the simplified RAG service.

Phase B (chunking-engine-parity, task 7): the module's independent regex
splitter (``_chunk_text_in_process``) is DELETED. Every method -- including
``ebook_chapters`` and the other structure-aware methods the old wrapper's
five-method whitelist rejected (spec §7.2) -- now routes through the vendored
engine via the ``Chunk_Lib`` compatibility shim. The wrapper keeps only what
the legacy contract needs on top of the engine: the three legacy validation
messages (the engine clamps degenerate size/overlap instead of raising) and
the flat output shape with a top-level 0-based ``chunk_index``.
"""

import json
import logging
from typing import List, Dict, Any

# Phase B: the exception classes are ALIASES of the engine's (re-exported by
# the Chunk_Lib shim), so ``except chunking_service.ChunkingError`` and
# ``except Chunk_Lib.ChunkingError`` catch the same objects.
from tldw_chatbook.Chunking.Chunk_Lib import ChunkingError, InvalidChunkingMethodError
from tldw_chatbook.Chunking.Chunk_Lib import (
    improved_chunking_process as _shim_improved_chunking_process,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkingError",
    "InvalidChunkingMethodError",
    "ChunkingService",
    "improved_chunking_process",
]


def _chunk_to_text(chunk: Any) -> str:
    """Return the text of a chunk, whatever shape the chunker produced.

    ``Chunker.chunk_text`` is not uniform: the text methods yield plain strings,
    while the structure-aware ones (``json``, ``xml``, ``ebook_chapters``) yield
    dicts carrying their text alongside metadata. Callers that assumed one shape
    crashed on the other -- see task-840 for the audio path and task-841 for this
    one.

    Retained for its importers (``Tests/RAG/test_chunking_service.py`` imports
    it directly); since task 7 the chunking path itself runs through
    ``Chunk_Lib.improved_chunking_process``, which normalizes chunk shapes
    itself, so this helper is no longer on the hot path.

    Args:
        chunk: A chunk as returned by the underlying chunker.

    Returns:
        The chunk's text, or an empty string when it carries none.
    """
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        for key in ("text", "content", "chunk"):
            value = chunk.get(key)
            if isinstance(value, str):
                return value
        # A structured payload with no text field: serialise rather than drop it,
        # so the content still reaches the index.
        return json.dumps(chunk, ensure_ascii=False, default=str)
    return str(chunk)


def _validate_legacy_chunk_params(chunk_size: Any, chunk_overlap: Any) -> None:
    """Enforce the legacy size/overlap contract before delegating to the engine.

    The engine's semantics differ: it clamps ``overlap >= max_size`` and raises
    a plain ``ValueError`` for ``max_size <= 0`` -- neither matches the messages
    callers of this module have depended on (pinned by
    ``Tests/RAG/test_chunking_service.py`` and
    ``Tests/RAG/simplified/test_chunking_algorithms.py``), so the wrapper
    enforces the legacy contract itself, BEFORE delegation.

    Args:
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.

    Raises:
        ChunkingError: With the legacy message for each violated invariant.
    """
    if chunk_size <= 0:
        raise ChunkingError("max_words must be positive")
    if chunk_overlap < 0:
        raise ChunkingError("Overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ChunkingError("Overlap must be less than max_words")


def _with_flat_chunk_index(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add the legacy top-level 0-based ``chunk_index`` to shim output.

    The shim emits the flat contract (top-level ``text``/``start_char``/
    ``end_char``/``word_count``) but carries the chunk index only inside
    ``metadata`` (1-based). The legacy wrapper emitted a top-level 0-based
    ``chunk_index``, and callers index on it (``parallel_processor``, the
    simplified RAG tests), so restore it.

    Args:
        chunks: Chunks as returned by ``Chunk_Lib.improved_chunking_process``.

    Returns:
        The same chunk dicts with a top-level 0-based ``chunk_index`` added.
    """
    return [{**chunk, "chunk_index": i} for i, chunk in enumerate(chunks)]


class ChunkingService:
    """
    Minimal chunking service that routes all methods through the Chunk_Lib shim.

    This is a compatibility layer for the simplified RAG implementation.
    """

    def __init__(self):
        """Initialize the chunking service."""
        # Initialize with default options
        self.default_options = {"method": "words", "max_size": 400, "overlap": 200}
        logger.info("Initialized ChunkingService wrapper")

    def chunk_text(
        self,
        content: str,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        method: str = "words",
    ) -> List[Dict[str, Any]]:
        """
        Chunk text using the specified method.

        Delegates to ``Chunk_Lib.improved_chunking_process`` (the engine shim)
        for ALL methods -- the module's own regex splitter was deleted
        (chunking-engine-parity task 7) -- and returns the flat contract
        (top-level text/start_char/end_char/word_count/chunk_index plus the
        shim's rich ``metadata`` dict).

        Args:
            content: Text to chunk
            chunk_size: Target size of chunks
            chunk_overlap: Overlap between chunks
            method: Chunking method ("words", "sentences", "paragraphs",
                "tokens", "semantic", "json", "xml", "ebook_chapters", ...)

        Returns:
            List of chunk dictionaries with text and metadata

        Raises:
            ChunkingError: If chunk_size/chunk_overlap violate the legacy
                contract, or the engine fails to chunk the text.
            InvalidChunkingMethodError: If the method is not supported.
        """
        try:
            # Legacy validation contract first: the engine clamps degenerate
            # size/overlap values instead of raising (spec §6.3.1).
            _validate_legacy_chunk_params(chunk_size, chunk_overlap)

            raw_chunks = _shim_improved_chunking_process(
                content,
                {"method": method, "max_size": chunk_size, "overlap": chunk_overlap},
            )
        except ChunkingError:
            # Engine exceptions (including InvalidChunkingMethodError) keep
            # their type and message so ``except`` blocks match precisely.
            raise
        except Exception as e:
            logger.error(
                f"Error chunking text with method '{method}': {e}", exc_info=True
            )
            # Re-raise with more context instead of hiding the error
            raise ChunkingError(
                f"Failed to chunk text using method '{method}': {str(e)}"
            ) from e

        chunks = _with_flat_chunk_index(raw_chunks)
        logger.debug(
            f"Chunked text into {len(chunks)} chunks using method '{method}'"
        )
        return chunks


def improved_chunking_process(
    text: str, options: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Wrapper function to provide compatibility with the server's chunking interface.

    Thin delegate to ``Chunk_Lib.improved_chunking_process`` (the engine shim).
    The legacy five-method whitelist is GONE (spec §7.2): every method the
    engine implements -- including ``ebook_chapters`` -- now works, and
    genuinely unknown methods raise the (aliased) engine
    ``InvalidChunkingMethodError``.

    Args:
        text: The text to chunk.
        options: Dictionary containing chunking options:
            - method: The chunking method to use
            - max_size: Maximum size of each chunk
            - overlap: Overlap between chunks

    Returns:
        List of chunk dictionaries (flat contract with chunk_index)

    Raises:
        InvalidChunkingMethodError: If the chunking method is not supported
        ChunkingError: For other chunking-related errors
    """
    try:
        # Legacy validation contract first (same rationale as chunk_text):
        # the engine clamps degenerate values instead of raising.
        _validate_legacy_chunk_params(
            options.get("max_size", 400), options.get("overlap", 100)
        )
        raw_chunks = _shim_improved_chunking_process(text, options)
    except ChunkingError:
        raise
    except Exception as e:
        raise ChunkingError(f"Error during chunking: {str(e)}") from e
    return _with_flat_chunk_index(raw_chunks)
