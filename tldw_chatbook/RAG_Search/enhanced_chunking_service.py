"""
Enhanced chunking service (Phase B: retired to the vendored engine).

Phase B (chunking-engine-parity, task 8, Q5 ruling): this module's home-grown
structure-aware chunking — ``DocumentStructureParser``,
``_hierarchical_chunking``, ``_structural_chunking``, ``_sub_chunk_element``
and the bespoke table/PDF-artifact preprocessing — is DELETED. The vendored
engine's ``structure_aware`` strategy driven through
``Chunker.chunk_text_hierarchical_flat`` is the only structure-aware
implementation, and :mod:`tldw_chatbook.RAG_Search.parent_child_adapter` is
the seam that derives the legacy parent/child retrieval shape from the
engine's hierarchical output.

``EnhancedChunkingService`` is retained as a thin delegating class (same
class name and method signatures) so existing consumers —
``RAG_Search/simplified/enhanced_indexing_helpers.py``,
``RAG_Search/simplified/enhanced_rag_service.py`` and
``Widgets/chunk_preview_modal.py`` — keep importing it unchanged.
``StructuredChunk``/``ChunkType`` are re-exported from the adapter for any
straggler imports; ``create_enhanced_chunking_service()`` keeps returning
the service.
"""

from typing import Any, Dict, List

from .chunking_service import ChunkingService
from .parent_child_adapter import (  # noqa: F401  (re-export)
    ChunkType,
    StructuredChunk,
    chunk_text_with_structure as _pca_chunk_text_with_structure,
    chunk_with_parent_retrieval as _pca_chunk_with_parent_retrieval,
)

__all__ = [
    "ChunkType",
    "StructuredChunk",
    "EnhancedChunkingService",
    "create_enhanced_chunking_service",
]


class EnhancedChunkingService(ChunkingService):
    """
    Enhanced chunking service backed by the engine's hierarchical path.

    Delegates ``chunk_with_parent_retrieval`` and
    ``chunk_text_with_structure`` to
    :mod:`tldw_chatbook.RAG_Search.parent_child_adapter`, which calls the
    vendored engine (``Chunker.chunk_text_hierarchical_flat`` with the
    ``structure_aware`` strategy) and preserves the legacy return shapes.
    """

    def chunk_text_with_structure(
        self,
        content: str,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        method: str = "hierarchical",
        preserve_structure: bool = True,
        clean_artifacts: bool = True,
        serialize_tables: bool = True,
    ) -> List[StructuredChunk]:
        """
        Enhanced chunking with structure preservation (legacy signature).

        Args:
            content: Text to chunk
            chunk_size: Element budget per chunk (engine grouping semantics)
            chunk_overlap: Element overlap between chunks
            method: Legacy method name ("hierarchical", "structural",
                "contextual") — accepted for compatibility; the engine's
                ``structure_aware`` strategy is the only structure-aware
                implementation
            preserve_structure: Ignored (always structure-preserving)
            clean_artifacts: Ignored (the engine sanitizes input internally)
            serialize_tables: Ignored (tables are chunked as structural
                blocks by the engine)

        Returns:
            List of StructuredChunk objects
        """
        return _pca_chunk_text_with_structure(
            content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            method=method,
            parent_size_multiplier=3,
            preserve_structure=preserve_structure,
            clean_artifacts=clean_artifacts,
            serialize_tables=serialize_tables,
        )

    def chunk_with_parent_retrieval(
        self,
        content: str,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        parent_size_multiplier: int = 3,
    ) -> Dict[str, Any]:
        """
        Chunk text with parent document retrieval support (legacy signature).

        Args:
            content: Text to chunk
            chunk_size: Element budget per retrieval chunk
            chunk_overlap: Element overlap between retrieval chunks
            parent_size_multiplier: Parent chunks hold this many times more
                grouped elements than retrieval chunks

        Returns:
            Dictionary with 'chunks', 'parent_chunks' and 'metadata'
        """
        return _pca_chunk_with_parent_retrieval(
            content,
            max_size=chunk_size,
            overlap=chunk_overlap,
            parent_size_multiplier=parent_size_multiplier,
        )


# Convenience function to maintain API compatibility
def create_enhanced_chunking_service() -> EnhancedChunkingService:
    """Create an instance of the enhanced chunking service."""
    return EnhancedChunkingService()
