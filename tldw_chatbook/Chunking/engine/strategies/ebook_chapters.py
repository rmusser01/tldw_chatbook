# ebook_chapters.py
"""
eBook chapter-based chunking strategy.
Splits text into chunks based on chapter markers.
"""

import multiprocessing as mp
import re
import threading
from contextlib import contextmanager, suppress
from queue import Queue
from typing import Any, Optional

from loguru import logger
from tldw_chatbook.Chunking._shims.testing import is_truthy

from ..base import BaseChunkingStrategy, ChunkMetadata, ChunkResult
from ..exceptions import InvalidInputError, ProcessingError

_EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    InvalidInputError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    ProcessingError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    ZeroDivisionError,
    re.error,
)


class EbookChapterChunkingStrategy(BaseChunkingStrategy):
    """
    Strategy for chunking text by eBook chapters.
    """

    # Default chapter patterns for various languages
    CHAPTER_PATTERNS = {
        'en': r'(?:Chapter|CHAPTER|Section|SECTION|Part|PART)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)',
        'es': r'(?:Capítulo|CAPÍTULO|Sección|SECCIÓN|Parte|PARTE)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)',
        'fr': r'(?:Chapitre|CHAPITRE|Section|SECTION|Partie|PARTIE)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)',
        'de': r'(?:Kapitel|KAPITEL|Abschnitt|ABSCHNITT|Teil|TEIL)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)',
        'it': r'(?:Capitolo|CAPITOLO|Sezione|SEZIONE|Parte|PARTE)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)',
        'pt': r'(?:Capítulo|CAPÍTULO|Seção|SEÇÃO|Parte|PARTE)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)',
        'default': r'(?:Chapter|CHAPTER|Section|SECTION|Part|PART)\s+(?:[0-9]+|[IVXLCDM]+)(?:\.|:|\s|$)'
    }

    # Regex complexity limits for security
    MAX_REGEX_LENGTH = 500  # Maximum length of custom regex pattern
    REGEX_TIMEOUT = 2  # Seconds before regex timeout
    MAX_CHAPTER_MARKERS = 10000  # Hard cap on detected markers
    # Heuristic patterns to catch nested-quantifier risks in Python's 're'.
    # Note: Python re does not support PCRE features like recursion or DEFINE;
    # we therefore focus on constructs that can cause catastrophic backtracking.
    DANGEROUS_PATTERNS = [
        r'{\d{4,}}',                  # Very large bounded repetitions
        r'[*+]{2,}',                   # Repeated quantifiers like '++' or '**'
        r'\([^)]*[+*]\)[+*?]',       # (a+)+, (a*)*, (a+)?
        r'\([^)]*[+*]\){',           # (a+){n,m}
        r'\(\w+\+\)\+',           # (word+)+
        r'\(\w+\*\)\*',           # (word*)*
        r'\(\w+\?\)\?',           # (word?)?
        r'\([^)]*\|[^)]*[+*]\)[+*]',  # (a|b+)+
        r'\(\([^)]+\)[+*]\)[+*]',  # ((a)+)+
    ]

    def __init__(self, language: str = 'en'):
        """
        Initialize the ebook chapter chunking strategy.

        Args:
            language: Language code for text processing
        """
        super().__init__(language)
        # Policy toggles via config.txt (no env toggles)
        self._force_simple_only = False
        # Prefer process-based isolation by default to enable hard timeouts.
        self._disable_mp = False
        try:
            from tldw_chatbook.Chunking._shims.config import load_comprehensive_config
            _cp = load_comprehensive_config()
            if hasattr(_cp, 'has_section') and _cp.has_section('Chunking'):
                try:
                    v = _cp.get('Chunking', 'regex_simple_only', fallback=None)
                    if v is not None:
                        self._force_simple_only = is_truthy(str(v).strip())
                except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS:
                    pass
                try:
                    v = _cp.get('Chunking', 'regex_disable_multiprocessing', fallback=None)
                    if v is not None:
                        self._disable_mp = is_truthy(str(v).strip())
                except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS:
                    pass
                try:
                    v = _cp.get('Chunking', 'regex_timeout_seconds', fallback=None)
                    if v is not None:
                        vt = float(str(v))
                        if vt > 0:
                            self.REGEX_TIMEOUT = vt
                except (TypeError, ValueError):
                    pass
        except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS:
            # Default values above remain in effect if config unavailable
            pass
        logger.debug(f"EbookChapterChunkingStrategy initialized for language: {language}")

    @contextmanager
    def _regex_timeout(self, seconds):
        """
        Cross-platform context manager for regex timeout to prevent ReDoS attacks.
        Uses threading instead of signals for Windows compatibility.
        """
        result = {'completed': False, 'error': None}

        def target_func(func, args, kwargs):
            try:
                result['value'] = func(*args, **kwargs)
                result['completed'] = True
            except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS as e:
                result['error'] = e
                result['completed'] = True

        # For cross-platform compatibility, we'll use a different approach
        # This wraps the regex operation in a way that can be interrupted
        class RegexTimeout:
            def __init__(self, timeout_seconds):
                self.timeout_seconds = timeout_seconds
                self.timed_out = False

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def run_with_timeout(self, func, *args, **kwargs):
                """Run a function with a timeout."""
                result_container = {'value': None, 'error': None}

                def wrapper():
                    try:
                        result_container['value'] = func(*args, **kwargs)
                    except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS as e:
                        result_container['error'] = e

                thread = threading.Thread(target=wrapper, daemon=True)
                thread.start()
                thread.join(timeout=self.timeout_seconds)

                if thread.is_alive():
                    # Thread is still running after timeout
                    self.timed_out = True
                    raise ProcessingError("Regex operation timed out - possible ReDoS attack")

                if result_container['error']:
                    raise result_container['error']

                return result_container['value']

        yield RegexTimeout(seconds)

    @staticmethod
    def _is_simple_safe_pattern(pattern: str) -> bool:
        r"""Allow only a restricted subset of regex constructs considered safe.

        Allowed:
        - Literals (letters, digits, spaces)
        - Anchors ^ and $
        - Character classes like [A-Z], [IVX]
        - Escapes: \d, \w
        - Quantifier + following a literal, escape (\d/\w), or a character class

        Disallowed: grouping (), alternation |, wildcard ., ?, *, nested classes, backrefs.
        """
        # Quick rejects (keep '.' disallowed to avoid wildcard); allow common literals like ':'
        for bad in ("(", ")", "|", ".", "?", "*"):
            if bad in pattern:
                return False
        # Validate escapes and '+' placement
        i = 0
        n = len(pattern)
        while i < n:
            ch = pattern[i]
            if ch == "\\":
                if i + 1 >= n:
                    return False
                nxt = pattern[i + 1]
                if nxt not in ("d", "w"):
                    return False
                i += 2
                continue
            if ch == "[":
                j = pattern.find("]", i + 1)
                if j == -1:
                    return False
                # rudimentary check: disallow nested '[' inside
                if "[" in pattern[i + 1:j]:
                    return False
                i = j + 1
                continue
            # Anchors and common literals allowed
            if ch in "^$ :," or ch.isalnum():
                i += 1
                continue
            if ch == "+":
                if i == 0:
                    return False
                prev = pattern[i - 1]
                # plus must follow a charclass close ']' or alnum or 'd'/'w' from an escape
                if prev == "]" or prev.isalnum():
                    i += 1
                    continue
                # handle case of plus after escape like '\\d+'
                if i >= 2 and pattern[i - 2] == "\\" and pattern[i - 1] in ("d", "w"):
                    i += 1
                    continue
                return False
            # any other symbol is disallowed
            return False
        return True

    @staticmethod
    def _finditer_worker(pattern: str, text: str, flags: int, out_queue: "mp.Queue") -> None:
        """Worker process to run regex finditer safely and return spans.

        Sends a list of (start, end, group0) tuples via the queue.
        """
        try:
            compiled = re.compile(pattern, flags)
            results: list[tuple[int, int, str]] = []
            for m in compiled.finditer(text):
                # Limit group length to avoid huge IPC payloads
                g = m.group(0)
                if len(g) > 1024:
                    g = g[:1024]
                results.append((m.start(), m.end(), g))
            out_queue.put(("ok", results))
        except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS as e:
            out_queue.put(("err", str(e)))

    def _run_finditer_process(self, pattern: str, text: str, flags: int, timeout_s: float) -> list[tuple[int, int, str]]:
        """Run regex finditer in a separate process and enforce a hard timeout."""
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
        q: mp.Queue = ctx.Queue()
        p = ctx.Process(target=self._finditer_worker, args=(pattern, text, flags, q))
        p.daemon = True
        p.start()
        p.join(timeout_s)
        if p.is_alive():
            try:
                p.terminate()
            finally:
                p.join(1)
            raise ProcessingError("Regex operation timed out - possible ReDoS attack (process)")
        if not q.empty():
            st, pl = q.get_nowait()
            if st == "ok":
                return pl  # type: ignore[return-value]
            raise ProcessingError(f"Regex execution failed: {pl}")
        raise ProcessingError("Regex execution failed without result (process)")

    def _safe_finditer(self, pattern: str, text: str, flags: int, timeout_s: float) -> list[tuple[int, int, str]]:
        """Run regex finditer in a separate process with a deadline.

        Returns list of (start, end, group0) tuples or raises ProcessingError on timeout/error.
        """
        if not getattr(self, "_disable_mp", True):
            try:
                return self._run_finditer_process(pattern, text, flags, timeout_s)
            except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Process-based regex execution failed; falling back to thread: {e}")
        # Fallback to thread-based execution when multiprocessing is disabled/unavailable
        done_q: Queue = Queue(maxsize=1)

        def _runner():
            try:
                compiled = re.compile(pattern, flags)
                results: list[tuple[int, int, str]] = []
                for m in compiled.finditer(text):
                    g = m.group(0)
                    if len(g) > 1024:
                        g = g[:1024]
                    results.append((m.start(), m.end(), g))
                done_q.put(("ok", results))
            except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover - defensive
                done_q.put(("err", str(e)))

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            # Abandon the work; enforce hard limit by failing fast.
            raise ProcessingError("Regex operation timed out - possible ReDoS attack (thread)")
        try:
            status, payload = done_q.get_nowait()
        except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS:
            raise ProcessingError("Regex execution failed without result (thread)") from None
        if status == "ok":
            return payload  # type: ignore[return-value]
        raise ProcessingError(f"Regex execution failed: {payload}")

    def _timed_finditer(self, pattern: str, text: str, flags: int, timeout_s: float) -> list[tuple[int, int, str]]:
        """Run a simple finditer in a daemon thread with a timeout.

        Even for simple patterns, guard execution to avoid unexpected hangs.
        Returns list of (start, end, group0) tuples.
        """
        if not getattr(self, "_disable_mp", True):
            try:
                return self._run_finditer_process(pattern, text, flags, timeout_s)
            except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Process-based regex execution failed; falling back to thread: {e}")
        done_q: Queue = Queue(maxsize=1)

        def _runner():
            try:
                compiled = re.compile(pattern, flags)
                results: list[tuple[int, int, str]] = []
                for m in compiled.finditer(text):
                    g = m.group(0)
                    if len(g) > 1024:
                        g = g[:1024]
                    results.append((m.start(), m.end(), g))
                done_q.put(("ok", results))
            except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS as e:
                done_q.put(("err", str(e)))

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            raise ProcessingError("Regex operation timed out - possible ReDoS attack (timed)")
        status, payload = done_q.get()
        if status == "ok":
            return payload  # type: ignore[return-value]
        raise ProcessingError(f"Regex execution failed: {payload}")

    def _validate_regex_pattern(self, pattern: str) -> bool:
        """
        Validate a regex pattern for security issues.

        Args:
            pattern: Regex pattern to validate

        Returns:
            True if pattern is safe, False otherwise

        Raises:
            InvalidInputError: If pattern is dangerous
        """
        # Check pattern length
        if len(pattern) > self.MAX_REGEX_LENGTH:
            raise InvalidInputError(
                f"Regex pattern too long ({len(pattern)} chars). "
                f"Maximum allowed: {self.MAX_REGEX_LENGTH}"
            )

        # Check for dangerous patterns
        for dangerous in self.DANGEROUS_PATTERNS:
            if re.search(dangerous, pattern):
                raise InvalidInputError(
                    f"Regex pattern contains potentially dangerous construct: {dangerous}"
                )

        # Enforce simple-only mode if configured
        if getattr(self, "_force_simple_only", False) and not self._is_simple_safe_pattern(pattern):
            raise InvalidInputError("Only simple regex patterns are allowed by policy")

        # Test pattern compilation
        try:
            re.compile(pattern)
        except re.error as e:
            raise InvalidInputError(f"Invalid regex pattern: {e}") from e

        # Don't actually test the regex execution - the DANGEROUS_PATTERNS check above
        # should catch problematic patterns. Testing them would cause the very problem
        # we're trying to prevent (ReDoS during validation).

        return True

    def chunk(self,
              text: str,
              max_size: int = 5000,
              overlap: int = 0,
              **options) -> list[str]:
        """
        Chunk text by chapters.

        Args:
            text: Text to chunk
            max_size: Maximum words per chunk (if chapter exceeds this, it will be split)
            overlap: Number of words to overlap (only applies when splitting large chapters)
            **options: Additional options including:
                - custom_chapter_pattern: Custom regex pattern for chapter detection

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []  # Consistent with other strategies

        try:
            # Get custom pattern or use language-specific default
            custom_pattern = options.get('custom_chapter_pattern')
            if custom_pattern:
                # Validate custom pattern for security
                self._validate_regex_pattern(custom_pattern)
                chapter_pattern = custom_pattern
                logger.debug(f"Using validated custom chapter pattern: {custom_pattern}")
            else:
                chapter_pattern = self.CHAPTER_PATTERNS.get(
                    self.language,
                    self.CHAPTER_PATTERNS['default']
                )
                logger.debug(f"Using {self.language} chapter pattern")

            # Find all chapter markers; prefer direct search for simple safe patterns
            try:
                if self._is_simple_safe_pattern(chapter_pattern):
                    # Still guard with a timeout, even for simple patterns
                    chapter_markers = self._timed_finditer(chapter_pattern, text, re.MULTILINE, self.REGEX_TIMEOUT)
                else:
                    spans = self._safe_finditer(chapter_pattern, text, re.MULTILINE, self.REGEX_TIMEOUT)
                    # Reconstruct minimal marker info from spans (we need only starts)
                    chapter_markers = spans
            except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Chapter marker detection timed out or failed ({e}); falling back to size-based split")
                return self._split_by_size(text, max_size, overlap)

            if not chapter_markers:
                logger.info("No chapter markers found, treating entire text as single chapter")
                # No chapters found, split by size if needed
                return self._split_by_size(text, max_size, overlap)

            # Hard cap number of markers to prevent pathological cases
            if isinstance(chapter_markers, list) and len(chapter_markers) > self.MAX_CHAPTER_MARKERS:
                logger.warning("Too many chapter markers detected; applying size-based split to avoid overload")
                return self._split_by_size(text, max_size, overlap)

            logger.debug(f"Found {len(chapter_markers)} chapter markers")

            chunks = []

            # Process each chapter
            for i, marker in enumerate(chapter_markers):
                # Determine chapter boundaries
                chapter_start = marker[0] if isinstance(marker, tuple) else marker.start()  # support both formats
                if i < len(chapter_markers) - 1:
                    next_start = chapter_markers[i + 1][0] if isinstance(chapter_markers[i + 1], tuple) else chapter_markers[i + 1].start()
                    chapter_end = next_start
                else:
                    chapter_end = len(text)

                chapter_text = text[chapter_start:chapter_end].strip()

                # Count words in chapter
                word_count = len(chapter_text.split())

                # Check if chapter needs to be split due to size
                if word_count > max_size:
                    logger.debug(f"Chapter has {word_count} words, splitting...")
                    # Split large chapter into smaller chunks
                    chapter_chunks = self._split_by_size(chapter_text, max_size, overlap)
                    chunks.extend(chapter_chunks)
                else:
                    chunks.append(chapter_text)

            logger.debug(f"Created {len(chunks)} chapter-based chunks")
            return chunks

        except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error during chapter chunking: {e}")
            raise ProcessingError(f"Failed to chunk by chapters: {str(e)}") from e

    def _split_by_size(self, text: str, max_size: int, overlap: int) -> list[str]:
        """
        Split text by word count when no chapters or chapter is too large.

        Args:
            text: Text to split
            max_size: Maximum words per chunk
            overlap: Number of words to overlap

        Returns:
            List of text chunks
        """
        if max_size <= 0:
            raise InvalidInputError("max_size must be positive")
        if overlap < 0:
            overlap = 0
        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}); adjusting to max_size - 1")
            overlap = max_size - 1

        words = text.split()
        chunks: list[str] = []

        # Ensure positive progress per iteration
        step = max(1, (max_size - overlap) if overlap > 0 else max_size)
        for i in range(0, len(words), step):
            end_idx = min(i + max_size, len(words))
            chunk_words = words[i:end_idx]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)

        return chunks

    def chunk_with_metadata(self,
                           text: str,
                           max_size: int = 5000,
                           overlap: int = 0,
                           **options) -> list[ChunkResult]:
        """
        Chunk text by chapters and return with metadata.

        Args:
            text: Text to chunk
            max_size: Maximum words per chunk
            overlap: Number of words to overlap
            **options: Additional options

        Returns:
            List of ChunkResult objects with metadata
        """
        if not text or not text.strip():
            return []  # Consistent with other strategies

        if max_size <= 0:
            raise InvalidInputError("max_size must be positive")
        if overlap < 0:
            overlap = 0
        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}); adjusting to max_size - 1")
            overlap = max_size - 1

        try:
            chunks: list[ChunkResult] = []
            chunk_index = 0

            def _trim_bounds(start: int, end: int) -> Optional[tuple[int, int]]:
                segment = text[start:end]
                ltrim = len(segment) - len(segment.lstrip())
                rtrim = len(segment) - len(segment.rstrip())
                new_start = start + ltrim
                new_end = end - rtrim if rtrim else end
                if new_end <= new_start:
                    return None
                return new_start, new_end

            def _word_spans(segment_text: str) -> list[tuple[int, int]]:
                return [(m.start(), m.end()) for m in re.finditer(r'\S+', segment_text)]

            def _append_chunk(seg_start: int, spans: list[tuple[int, int]],
                              start_idx: int, end_idx: int, meta_opts: dict[str, Any]) -> None:
                nonlocal chunk_index
                if end_idx <= start_idx:
                    return
                chunk_start = seg_start + spans[start_idx][0]
                chunk_end = seg_start + spans[end_idx - 1][1]
                with suppress(_EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS):
                    chunk_end = self._expand_end_to_grapheme_boundary(text, chunk_end)
                if chunk_end < chunk_start:
                    chunk_end = chunk_start
                chunk_text = text[chunk_start:chunk_end]
                metadata = ChunkMetadata(
                    index=chunk_index,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    word_count=end_idx - start_idx,
                    language=self.language,
                    method='ebook_chapters',
                    options=dict(meta_opts),
                )
                chunks.append(ChunkResult(text=chunk_text, metadata=metadata))
                chunk_index += 1

            # Get custom pattern or use language-specific default
            custom_pattern = options.get('custom_chapter_pattern')
            if custom_pattern:
                # Validate custom pattern for security
                self._validate_regex_pattern(custom_pattern)
                chapter_pattern = custom_pattern
            else:
                chapter_pattern = self.CHAPTER_PATTERNS.get(
                    self.language,
                    self.CHAPTER_PATTERNS['default']
                )

            # Find all chapter markers; prefer direct search for simple safe patterns
            try:
                if self._is_simple_safe_pattern(chapter_pattern):
                    chapter_markers = list(re.finditer(chapter_pattern, text, re.MULTILINE))
                else:
                    spans = self._safe_finditer(chapter_pattern, text, re.MULTILINE, self.REGEX_TIMEOUT)
                    chapter_markers = spans
            except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Chapter marker detection timed out or failed ({e}); falling back to size-based split (metadata)")
                bounds = _trim_bounds(0, len(text))
                if not bounds:
                    return []
                seg_start, seg_end = bounds
                spans = _word_spans(text[seg_start:seg_end])
                if not spans:
                    return []
                step = max(1, (max_size - overlap) if overlap > 0 else max_size)
                for i in range(0, len(spans), step):
                    end_idx = min(i + max_size, len(spans))
                    _append_chunk(seg_start, spans, i, end_idx, {'fallback': 'size_split'})
                return chunks

            if not chapter_markers:
                # No chapters found, treat as single chunk
                bounds = _trim_bounds(0, len(text))
                if not bounds:
                    return []
                seg_start, seg_end = bounds
                spans = _word_spans(text[seg_start:seg_end])
                if not spans:
                    return []
                step = max(1, (max_size - overlap) if overlap > 0 else max_size)
                for i in range(0, len(spans), step):
                    end_idx = min(i + max_size, len(spans))
                    _append_chunk(seg_start, spans, i, end_idx, {'no_chapters': True})
                return chunks

            # Process each chapter
            for i, marker in enumerate(chapter_markers):
                chapter_start = marker[0] if isinstance(marker, tuple) else marker.start()
                if i < len(chapter_markers) - 1:
                    next_start = chapter_markers[i + 1][0] if isinstance(chapter_markers[i + 1], tuple) else chapter_markers[i + 1].start()
                    chapter_end = next_start
                else:
                    chapter_end = len(text)

                bounds = _trim_bounds(chapter_start, chapter_end)
                if not bounds:
                    continue
                seg_start, seg_end = bounds
                segment = text[seg_start:seg_end]
                spans = _word_spans(segment)
                if not spans:
                    continue
                chapter_title = (marker[2] if isinstance(marker, tuple) else marker.group()).strip()
                word_count = len(spans)

                # Check if chapter needs to be split
                if word_count > max_size:
                    # Split large chapter
                    step = max(1, (max_size - overlap) if overlap > 0 else max_size)
                    part = 1
                    for j in range(0, len(spans), step):
                        end_idx = min(j + max_size, len(spans))
                        _append_chunk(
                            seg_start,
                            spans,
                            j,
                            end_idx,
                            {
                                'chapter_title': f"{chapter_title} (Part {part})",
                                'chapter_number': i + 1,
                                'is_split': True
                            }
                        )
                        part += 1
                else:
                    # Keep chapter as single chunk
                    _append_chunk(
                        seg_start,
                        spans,
                        0,
                        len(spans),
                        {
                            'chapter_title': chapter_title,
                            'chapter_number': i + 1,
                            'total_chapters': len(chapter_markers)
                        }
                    )

            logger.debug(f"Created {len(chunks)} chapter-based chunks with metadata")
            return chunks

        except _EBOOK_CHUNKING_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error during chapter chunking: {e}")
            raise ProcessingError(f"Failed to chunk by chapters: {str(e)}") from e

    def validate_options(self, options: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize options for chapter chunking.

        Args:
            options: Options dictionary

        Returns:
            Validated options
        """
        validated = super().validate_options(options)

        # Validate custom pattern if provided
        if 'custom_chapter_pattern' in validated:
            pattern = validated['custom_chapter_pattern']
            try:
                re.compile(pattern)
            except re.error as e:
                raise InvalidInputError(f"Invalid chapter pattern: {e}") from e

        return validated
