"""
Fixed-size chunking strategy.

Splits text into fixed-length character windows with optional overlap.
"""

from loguru import logger

from ..base import BaseChunkingStrategy
from ..exceptions import ChunkingError


class FixedSizeChunkingStrategy(BaseChunkingStrategy):
    """Chunk text into fixed-size character windows."""

    def chunk(
        self,
        text: str,
        max_size: int,
        overlap: int = 0,
        **options,
    ) -> list[str]:
        """
        Chunk text using fixed-size windows measured in characters.

        Args:
            text: Text to chunk
            max_size: Maximum characters per chunk
            overlap: Number of characters to overlap between consecutive chunks
            **options: Unused but accepted for interface compatibility

        Returns:
            List of text chunks
        """
        if max_size is None or max_size <= 0:
            raise ChunkingError(f"Fixed-size chunking requires positive max_size, got {max_size}")
        if overlap < 0:
            raise ChunkingError(f"Overlap cannot be negative, got {overlap}")
        if overlap >= max_size:
            logger.warning("Overlap >= max_size; adjusting overlap to max_size - 1")
            overlap = max_size - 1
        if not text:
            return []

        step = max_size - overlap
        chunks: list[str] = []
        idx = 0
        text_length = len(text)

        while idx < text_length:
            end = min(idx + max_size, text_length)
            chunk = text[idx:end]
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            idx += step if step > 0 else max_size
            if step <= 0:
                # Should not happen with normalized overlap, but guard against infinite loop
                idx = min(idx + 1, text_length)

        return chunks
