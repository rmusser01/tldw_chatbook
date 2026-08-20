# base.py
"""
Base classes and protocols for the chunking system.
Provides abstract interfaces and common functionality for all chunking strategies.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

from loguru import logger

from tldw_chatbook.Chunking._shims.testing import is_truthy


class ChunkingMethod(Enum):
    """Enumeration of available chunking methods."""
    WORDS = "words"
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"
    PROPOSITIONS = "propositions"
    TOKENS = "tokens"
    SEMANTIC = "semantic"
    JSON = "json"
    XML = "xml"
    EBOOK_CHAPTERS = "ebook_chapters"
    ROLLING_SUMMARIZE = "rolling_summarize"
    FIXED_SIZE = "fixed_size"
    CODE = "code"
    STRUCTURE_AWARE = "structure_aware"


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    index: int
    start_char: int
    end_char: int
    word_count: int
    token_count: Optional[int] = None
    section: Optional[str] = None
    language: Optional[str] = None
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    char_count: Optional[int] = None  # Add for compatibility
    sentence_count: Optional[int] = None  # Add for compatibility
    method: Optional[str] = None  # Add chunking method used
    options: Optional[dict[str, Any]] = field(default=None)  # Add options used

    def __post_init__(self):
        """Calculate derived fields if not provided."""
        if self.char_count is None and self.end_char is not None and self.start_char is not None:
            self.char_count = self.end_char - self.start_char


@dataclass
class ChunkResult:
    """Result of a chunking operation."""
    text: str
    metadata: ChunkMetadata


class ChunkingStrategy(Protocol):
    """Protocol for chunking strategies."""

    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> list[str]:
        """
        Chunk text according to the strategy.

        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk
            overlap: Overlap between chunks
            **options: Strategy-specific options

        Returns:
            List of text chunks
        """
        ...

    def chunk_with_metadata(self,
                           text: str,
                           max_size: int,
                           overlap: int = 0,
                           **options) -> list[ChunkResult]:
        """
        Chunk text and return with metadata.

        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk
            overlap: Overlap between chunks
            **options: Strategy-specific options

        Returns:
            List of ChunkResult objects
        """
        ...


class BaseChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    def __init__(self, language: str = 'en'):
        """
        Initialize the chunking strategy.

        Args:
            language: Language code for text processing
        """
        self.language = language
        self._cache = {}
        logger.debug(f"Initialized {self.__class__.__name__} for language: {language}")

    def set_language(self, language: str) -> None:
        """Update strategy language (override in subclasses needing reinit)."""
        self.language = language

    # ---------------- Common helpers: config + grapheme boundaries ----------------
    @staticmethod
    def _get_chunking_bool(key: str, default: bool) -> bool:
        try:
            import os
            from tldw_chatbook.Chunking._shims.testing import is_truthy
            v = os.getenv(key.upper())
            if v is None:
                from tldw_chatbook.Chunking._shims.config import load_comprehensive_config
                cp = load_comprehensive_config()
                if hasattr(cp, 'has_section') and cp.has_section('Chunking'):
                    v = cp.get('Chunking', key, fallback=str(default))
            s = str(v).strip().lower() if v is not None else str(default).lower()
            return is_truthy(s)
        except (ImportError, AttributeError, KeyError) as e:
            logger.debug(f"_get_chunking_bool: config lookup failed for '{key}', using default={default}: {e}")
            return default
        except ValueError as e:
            logger.debug(f"_get_chunking_bool: invalid value for '{key}', using default={default}: {e}")
            return default

    def _strict_grapheme_mode(self, options: dict | None = None) -> bool:
        if options and 'strict_grapheme_end_expansion' in options:
            try:
                return bool(options.get('strict_grapheme_end_expansion'))
            except (TypeError, ValueError) as e:
                logger.debug(f"_strict_grapheme_mode: invalid option value, falling back to config: {e}")
        return self._get_chunking_bool('strict_grapheme_end_expansion', False)

    def _expand_end_to_grapheme_boundary(self, text: str, end: int, *, options: dict | None = None) -> int:
        """Expand end index so that we don't split grapheme clusters.

        Default mode (strict=False):
        - Includes trailing combining marks (Mn/Me), variation selectors, and
          other zero-width non-joiners (Cf except ZWJ).
        Strict mode (strict=True):
        - Additionally includes Zero Width Joiner (ZWJ) sequences (ZWJ + next
          base and trailing marks) and emoji skin tone modifiers.
        """
        import unicodedata as _ud

        strict = self._strict_grapheme_mode(options)
        n = len(text)
        i = min(max(0, end), n)

        def _is_vs(cp: int) -> bool:
            return (0xFE00 <= cp <= 0xFE0F) or (0xE0100 <= cp <= 0xE01EF)

        def _is_combining(ch: str) -> bool:
            cat = _ud.category(ch)
            return cat in ("Mn", "Me")

        def _is_skin_tone(cp: int) -> bool:
            return 0x1F3FB <= cp <= 0x1F3FF

        while i < n:
            ch = text[i]
            cp = ord(ch)
            cat = _ud.category(ch)
            if _is_combining(ch) or _is_vs(cp) or (cat == 'Cf' and (cp != 0x200D)):
                i += 1
                continue
            if strict and _is_skin_tone(cp):
                i += 1
                # include any following marks after modifier
                while i < n:
                    ch2 = text[i]
                    cp2 = ord(ch2)
                    cat2 = _ud.category(ch2)
                    if _is_combining(ch2) or _is_vs(cp2) or (cat2 == 'Cf' and cp2 != 0x200D):
                        i += 1
                    else:
                        break
                continue
            if (cp == 0x200D) and strict:
                # Include ZWJ and next codepoint + trailing marks
                i += 1
                if i < n:
                    i += 1
                    while i < n:
                        ch2 = text[i]
                        cp2 = ord(ch2)
                        cat2 = _ud.category(ch2)
                        if _is_combining(ch2) or _is_vs(cp2) or (cat2 == 'Cf' and cp2 != 0x200D):
                            i += 1
                        else:
                            break
                continue
            break
        # Final bounds check (defensive - should already be in range)
        return min(i, n)

    @abstractmethod
    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> list[str]:
        """
        Chunk text according to the strategy.

        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk
            overlap: Overlap between chunks
            **options: Strategy-specific options

        Returns:
            List of text chunks
        """
        pass

    def chunk_with_metadata(self,
                           text: str,
                           max_size: int,
                           overlap: int = 0,
                           **options) -> list[ChunkResult]:
        """
        Chunk text and return with metadata.

        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk
            overlap: Overlap between chunks
            **options: Strategy-specific options

        Returns:
            List of ChunkResult objects
        """
        chunks = self.chunk(text, max_size, overlap, **options)
        results = []
        current_pos = 0

        for i, chunk in enumerate(chunks):
            # Find chunk position in original text
            chunk_start = text.find(chunk, current_pos)
            used_backtrack = False
            if chunk_start == -1:
                # Backtrack up to one chunk length to handle non-char overlap units
                try:
                    backtrack = min(current_pos, len(chunk)) if chunk else 0
                except Exception:
                    backtrack = 0
                if backtrack > 0:
                    chunk_start = text.find(chunk, max(0, current_pos - backtrack))
                    used_backtrack = chunk_start != -1
            if chunk_start == -1:
                chunk_start = current_pos
            chunk_end = chunk_start + len(chunk)
            # Expand to avoid splitting grapheme clusters in metadata
            try:
                chunk_end = self._expand_end_to_grapheme_boundary(text, chunk_end, options=options)
            except (IndexError, ValueError) as e:
                logger.debug(f"chunk_with_metadata: grapheme expansion failed for chunk {i}, using original end: {e}")

            metadata = ChunkMetadata(
                index=i,
                start_char=chunk_start,
                end_char=chunk_end,
                word_count=len(chunk.split()),
                language=self.language,
                overlap_with_previous=overlap if i > 0 else 0,
                overlap_with_next=overlap if i < len(chunks) - 1 else 0
            )

            results.append(ChunkResult(text=chunk, metadata=metadata))
            if overlap > 0 and not used_backtrack and chunk_start >= current_pos:
                try:
                    current_pos = max(chunk_start, chunk_end - min(overlap, len(chunk)))
                except Exception:
                    current_pos = chunk_end
            else:
                current_pos = chunk_end

        return results

    def validate_options(self, options: Optional[dict[str, Any]]) -> dict[str, Any]:
        """
        Validate and normalize options for chunking strategies.

        Args:
            options: Options dictionary (may be None)

        Returns:
            A shallow-copied options dict
        """
        if options is None:
            return {}
        if not isinstance(options, dict):
            raise ValueError(f"options must be a dict, got {type(options).__name__}")
        return dict(options)

    def validate_parameters(self, text: str, max_size: int, overlap: int):
        """
        Validate chunking parameters.

        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk
            overlap: Overlap between chunks

        Raises:
            ValueError: If parameters are invalid

        Notes:
            This method performs validation only. Implementations must re-clamp
            `overlap` at the strategy level to ensure forward progress, i.e.,
            when `overlap >= max_size` set `overlap = max_size - 1` before
            windowing. Do not rely on this helper to mutate caller state.
        """
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, got {type(text).__name__}")

        if not text.strip():
            logger.debug("Empty text provided for chunking")
            return False

        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        if overlap < 0:
            raise ValueError(f"overlap cannot be negative, got {overlap}")

        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), adjusting to max_size - 1")
            overlap = max_size - 1

        return True

    def chunk_generator(self,
                       text: str,
                       max_size: int,
                       overlap: int = 0,
                       **options) -> Generator[str, None, None]:
        """
        Generator version of chunk for memory efficiency.

        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk
            overlap: Overlap between chunks
            **options: Strategy-specific options

        Yields:
            Individual text chunks
        """
        chunks = self.chunk(text, max_size, overlap, **options)
        yield from chunks


class ChunkerConfig:
    """Configuration for the chunking system."""

    def __init__(self,
                 default_method: ChunkingMethod = ChunkingMethod.WORDS,
                 default_max_size: int = 400,
                 default_overlap: int = 200,
                 language: str = 'en',
                 enable_cache: bool = True,
                 cache_size: int = 100,
                 cache_copy_on_access: bool = True,
                 cache_max_text_length: int = 2_000_000,
                 min_text_length_to_cache: int = 0,
                 max_text_length_to_cache: int = 2_000_000,
                 max_text_size: int = 100_000_000,  # 100MB
                 enable_metrics: bool = True,
                 verbose_logging: bool = False,
                 strategy_cache_mode: str = "shared",
                 # Execution/concurrency knobs (used by AsyncChunker; optional here)
                 max_workers: int = 4,
                 max_concurrent: int = 10):
        """
        Initialize chunker configuration.

        Args:
            default_method: Default chunking method
            default_max_size: Default maximum chunk size
            default_overlap: Default overlap between chunks
            language: Default language for text processing
            enable_cache: Whether to enable caching
            cache_size: Maximum cache size
            max_text_size: Maximum text size to process
            enable_metrics: Whether to collect metrics
            max_workers: Default worker threads for async execution helpers
            max_concurrent: Default concurrency semaphore for async helpers
            strategy_cache_mode: Strategy reuse policy ('shared', 'thread', 'call')
        """
        # Convert string to enum if needed
        if isinstance(default_method, str):
            try:
                default_method = ChunkingMethod(default_method)
            except ValueError:
                logger.warning(f"Unknown chunking method '{default_method}', using default")
                default_method = ChunkingMethod.WORDS

        # Basic validations
        if not isinstance(default_max_size, int) or default_max_size <= 0:
            raise ValueError(f"default_max_size must be a positive integer, got {default_max_size}")
        if not isinstance(default_overlap, int) or default_overlap < 0:
            raise ValueError(f"default_overlap must be a non-negative integer, got {default_overlap}")
        if not isinstance(cache_size, int) or cache_size <= 0:
            raise ValueError(f"cache_size must be a positive integer, got {cache_size}")
        if not isinstance(max_text_size, int) or max_text_size <= 0:
            raise ValueError(f"max_text_size must be a positive integer, got {max_text_size}")
        if not isinstance(cache_max_text_length, int) or cache_max_text_length <= 0:
            raise ValueError(f"cache_max_text_length must be a positive integer, got {cache_max_text_length}")
        if not isinstance(min_text_length_to_cache, int) or min_text_length_to_cache < 0:
            raise ValueError(f"min_text_length_to_cache must be a non-negative integer, got {min_text_length_to_cache}")
        if not isinstance(max_text_length_to_cache, int) or max_text_length_to_cache <= 0:
            raise ValueError(f"max_text_length_to_cache must be a positive integer, got {max_text_length_to_cache}")
        if not isinstance(max_workers, int) or max_workers <= 0:
            raise ValueError(f"max_workers must be a positive integer, got {max_workers}")
        if not isinstance(max_concurrent, int) or max_concurrent <= 0:
            raise ValueError(f"max_concurrent must be a positive integer, got {max_concurrent}")
        if not isinstance(strategy_cache_mode, str):
            raise ValueError(f"strategy_cache_mode must be a string, got {type(strategy_cache_mode).__name__}")
        strategy_cache_mode = strategy_cache_mode.strip().lower()
        if strategy_cache_mode not in {"shared", "thread", "call"}:
            raise ValueError(
                "strategy_cache_mode must be one of 'shared', 'thread', or 'call', "
                f"got {strategy_cache_mode!r}"
            )

        self.default_method = default_method
        self.default_max_size = default_max_size
        self.default_overlap = default_overlap
        self.language = language
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.cache_copy_on_access = bool(cache_copy_on_access)
        self.cache_max_text_length = cache_max_text_length
        # New cache policy thresholds (preferred)
        self.min_text_length_to_cache = min_text_length_to_cache
        self.max_text_length_to_cache = max_text_length_to_cache
        self.max_text_size = max_text_size
        self.enable_metrics = enable_metrics
        self.verbose_logging = bool(verbose_logging)
        self.strategy_cache_mode = strategy_cache_mode
        # Execution/concurrency knobs (used by AsyncChunker)
        self.max_workers = max_workers
        self.max_concurrent = max_concurrent

        # Allow config.txt overrides for selected settings (no ENV toggles)
        try:
            from tldw_chatbook.Chunking._shims.config import load_comprehensive_config
            cp = load_comprehensive_config()
            if hasattr(cp, 'has_section') and cp.has_section('Chunking'):
                try:
                    v = cp.get('Chunking', 'cache_copy_on_access', fallback=None)
                    if v is not None:
                        self.cache_copy_on_access = is_truthy(str(v))
                except (AttributeError, KeyError, TypeError) as e:
                    logger.debug(f"ChunkerConfig: failed to read 'cache_copy_on_access' from config: {e}")
                try:
                    v = cp.get('Chunking', 'verbose_logging', fallback=None)
                    if v is not None:
                        self.verbose_logging = is_truthy(str(v))
                except (AttributeError, KeyError, TypeError) as e:
                    logger.debug(f"ChunkerConfig: failed to read 'verbose_logging' from config: {e}")
        except ImportError as e:
            logger.debug(f"ChunkerConfig: config module not available, using defaults: {e}")
        except (AttributeError, TypeError) as e:
            logger.debug(f"ChunkerConfig: config loading failed, using defaults: {e}")

        logger.info(f"ChunkerConfig initialized with method={self.default_method.value if hasattr(self.default_method, 'value') else self.default_method}, "
                   f"max_size={default_max_size}, overlap={default_overlap}")


    # Note: Exceptions for the chunking module live in
    # tldw_chatbook.Chunking.engine.exceptions. This file intentionally
    # avoids duplicating those definitions.
