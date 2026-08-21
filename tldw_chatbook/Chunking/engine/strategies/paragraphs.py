# paragraphs.py
"""
Paragraph-based chunking strategy.
Splits text into chunks based on paragraph boundaries.
"""

import re
from typing import Any

from loguru import logger

from ..base import BaseChunkingStrategy, ChunkMetadata, ChunkResult
from ..exceptions import InvalidInputError, ProcessingError

# Maximum text size for paragraph chunking (50MB) - prevents DoS with ReDoS patterns
MAX_PARAGRAPH_TEXT_SIZE = 50 * 1024 * 1024


class ParagraphChunkingStrategy(BaseChunkingStrategy):
    """
    Strategy for chunking text by paragraphs.
    """

    def __init__(self, language: str = 'en'):
        """
        Initialize the paragraph chunking strategy.

        Args:
            language: Language code for text processing
        """
        super().__init__(language)
        logger.debug(f"ParagraphChunkingStrategy initialized for language: {language}")

    def chunk(self,
              text: str,
              max_size: int = 2,
              overlap: int = 0,
              **options) -> list[str]:
        """
        Chunk text by paragraphs.

        Args:
            text: Text to chunk
            max_size: Maximum number of paragraphs per chunk
            overlap: Number of paragraphs to overlap between chunks
            **options: Additional options

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []  # Consistent with other strategies

        # Input size validation to prevent DoS
        if len(text) > MAX_PARAGRAPH_TEXT_SIZE:
            raise InvalidInputError(
                f"Text size ({len(text)} bytes) exceeds maximum allowed "
                f"({MAX_PARAGRAPH_TEXT_SIZE} bytes) for paragraph chunking"
            )

        if max_size < 1:
            raise InvalidInputError(f"max_size must be at least 1, got {max_size}")

        if overlap < 0:
            raise InvalidInputError(f"overlap must be non-negative, got {overlap}")

        if overlap >= max_size:
            # Align with other strategies: clamp to ensure forward progress
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), setting to max_size - 1")
            overlap = max_size - 1

        try:
            # Split text into paragraphs (handling various paragraph separators)
            # Use simple linear-time pattern: two or more newlines (avoids ReDoS with \s*)
            paragraphs = re.split(r'\n{2,}', text.strip())

            # Filter out empty paragraphs
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

            if not paragraphs:
                # If no paragraphs found, treat entire text as one paragraph
                paragraphs = [text.strip()]

            logger.debug(f"Split text into {len(paragraphs)} paragraphs")

            chunks = []
            i = 0
            chunk_index = 0

            while i < len(paragraphs):
                # Determine the end index for this chunk
                end_idx = min(i + max_size, len(paragraphs))

                # Extract paragraphs for this chunk
                chunk_paragraphs = paragraphs[i:end_idx]

                # Join paragraphs with double newline
                chunk_text = '\n\n'.join(chunk_paragraphs)
                chunks.append(chunk_text)

                chunk_index += 1

                # Move to next chunk with overlap
                i += max_size - overlap if overlap > 0 else max_size

            logger.debug(f"Created {len(chunks)} paragraph-based chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error during paragraph chunking: {e}")
            raise ProcessingError(f"Failed to chunk by paragraphs: {str(e)}") from e

    def chunk_with_metadata(self,
                           text: str,
                           max_size: int = 2,
                           overlap: int = 0,
                           **options) -> list[ChunkResult]:
        """
        Chunk text by paragraphs and return with metadata using accurate source offsets.

        This implementation preserves the original character spans from the input
        text when computing start/end offsets. Paragraph boundaries are two or more
        newlines (optionally with whitespace). By default, chunk text is sliced from
        the original source span to retain spacing; pass align_text_to_source=False
        to return a normalized join of trimmed paragraph content instead.
        """
        if not text or not text.strip():
            return []  # Consistent with other strategies

        # Input size validation to prevent DoS
        if len(text) > MAX_PARAGRAPH_TEXT_SIZE:
            raise InvalidInputError(
                f"Text size ({len(text)} bytes) exceeds maximum allowed "
                f"({MAX_PARAGRAPH_TEXT_SIZE} bytes) for paragraph chunking"
            )

        if max_size < 1:
            raise InvalidInputError(f"max_size must be at least 1, got {max_size}")

        if overlap < 0:
            raise InvalidInputError(f"overlap must be non-negative, got {overlap}")

        if overlap >= max_size:
            # Align with other strategies: clamp to ensure forward progress
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), setting to max_size - 1")
            overlap = max_size - 1

        try:
            # Build paragraph spans directly from the original text
            # Use simple linear-time pattern: two or more newlines (avoids ReDoS with \s*)
            sep = re.compile(r"\n{2,}")
            spans: list[tuple[int, int]] = []
            pos = 0
            n = len(text)

            for m in sep.finditer(text):
                seg_start = pos
                seg_end = m.start()
                if seg_end > seg_start:
                    raw_segment = text[seg_start:seg_end]
                    # Trim leading/trailing whitespace for paragraph content offsets
                    ltrim = len(raw_segment) - len(raw_segment.lstrip())
                    rtrim = len(raw_segment) - len(raw_segment.rstrip())
                    p_start = seg_start + ltrim
                    p_end = seg_end - rtrim if rtrim else seg_end
                    if p_end > p_start:
                        spans.append((p_start, p_end))
                pos = m.end()

            # Tail segment
            if pos < n:
                raw_segment = text[pos:n]
                ltrim = len(raw_segment) - len(raw_segment.lstrip())
                rtrim = len(raw_segment) - len(raw_segment.rstrip())
                p_start = pos + ltrim
                p_end = n - rtrim if rtrim else n
                if p_end > p_start:
                    spans.append((p_start, p_end))

            # If no non-empty paragraphs detected, treat entire text (trimmed) as one paragraph
            if not spans:
                ltrim = len(text) - len(text.lstrip())
                rtrim = len(text) - len(text.rstrip())
                p_start = ltrim
                p_end = n - rtrim if rtrim else n
                if p_end > p_start:
                    spans.append((p_start, p_end))

            logger.debug(f"Detected {len(spans)} paragraph spans")

            # Window the paragraph spans according to max_size/overlap
            results: list[ChunkResult] = []
            align_text_to_source = bool(options.get("align_text_to_source", True))
            text_len = len(text)
            step = max(1, max_size - overlap)
            chunk_index = 0
            for i in range(0, len(spans), step):
                window = spans[i:i + max_size]
                if not window:
                    continue
                start_char = window[0][0]
                end_char = window[-1][1]
                try:
                    end_char = self._expand_end_to_grapheme_boundary(text, end_char, options=options)
                except (IndexError, ValueError) as e:
                    logger.debug(f"Grapheme expansion failed for paragraph chunk {chunk_index}: {e}")
                if align_text_to_source:
                    start_char = max(0, min(int(start_char), text_len))
                    end_char = max(start_char, min(int(end_char), text_len))
                    chunk_text = text[start_char:end_char]
                else:
                    # Build display text by joining trimmed paragraph content with double newlines
                    parts = [text[s:e] for (s, e) in window]
                    chunk_text = "\n\n".join(p.strip() for p in parts)
                metadata = ChunkMetadata(
                    index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    word_count=len(chunk_text.split()) if chunk_text else 0,
                    language=self.language,
                    method='paragraphs',
                    options={
                        'max_paragraphs': max_size,
                        'overlap': overlap,
                        'paragraph_count': len(window),
                    },
                )
                results.append(ChunkResult(text=chunk_text, metadata=metadata))
                chunk_index += 1

            logger.debug(f"Created {len(results)} paragraph-based chunks with metadata")
            return results

        except Exception as e:
            logger.error(f"Error during paragraph chunking: {e}")
            raise ProcessingError(f"Failed to chunk by paragraphs: {str(e)}") from e

    def validate_options(self, options: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize options for paragraph chunking.

        Args:
            options: Options dictionary

        Returns:
            Validated options
        """
        validated = super().validate_options(options)

        # Ensure max_size is reasonable for paragraphs
        if 'max_size' in validated and validated['max_size'] > 100:
            logger.warning(f"Very large max_size for paragraphs: {validated['max_size']}")

        return validated
