# strategies/semantic.py
"""
Semantic chunking strategy.
Groups text based on semantic similarity using TF-IDF vectors and cosine similarity.
"""

import importlib
import threading
from collections.abc import Generator

from loguru import logger

from ..base import BaseChunkingStrategy, ChunkMetadata, ChunkResult
from ..exceptions import ChunkingError


class SemanticChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks text based on semantic similarity between sentences.
    Uses TF-IDF vectorization and cosine similarity to identify
    natural semantic boundaries in the text.
    """

    def __init__(self,
                 language: str = 'en',
                 similarity_threshold: float = 0.3):
        """
        Initialize semantic chunking strategy.

        Args:
            language: Language code for text processing
            similarity_threshold: Threshold for semantic similarity (0-1)
        """
        super().__init__(language)
        self.similarity_threshold = similarity_threshold
        self._vectorizer = None
        self._sklearn_available = None
        self._nltk_available = None
        self._tokenizer = None
        self._tokenizer_name = "gpt2"
        self._tokenizer_type = None
        self._tokenizer_failed_names = set()
        self._tokenizer_lock = threading.Lock()

        # Check dependencies
        self._check_dependencies()

        logger.debug(f"SemanticChunkingStrategy initialized with threshold: {similarity_threshold}")

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        # Check scikit-learn
        try:
            importlib.import_module("sklearn.feature_extraction.text")
            importlib.import_module("sklearn.metrics.pairwise")
            self._sklearn_available = True
        except ImportError:
            self._sklearn_available = False
            logger.warning(
                "scikit-learn not available. Semantic chunking will not be available. "
                "Install with: pip install scikit-learn"
            )

        # Check NLTK
        try:
            import nltk
            self._nltk_available = True
            # Verify punkt presence but do not auto-download; fall back later if missing
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info(
                    "NLTK punkt tokenizer not found; will use simple fallback sentence splitting."
                )
        except ImportError:
            self._nltk_available = False
            logger.warning(
                "NLTK not available. Will use simple sentence splitting. "
                "Install with: pip install nltk"
            )

    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> list[str]:
        """
        Chunk text based on semantic similarity.

        Args:
            text: Text to chunk
            max_size: Maximum size per chunk (in units specified by 'unit' option)
            overlap: Number of sentences to overlap between chunks
            **options: Additional options:
                - unit: Unit for max_size ('words', 'tokens', 'characters')
                - similarity_threshold: Override default similarity threshold
                - min_chunk_size: Minimum chunk size before breaking on similarity

        Returns:
            List of text chunks grouped by semantic similarity
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []

        # Check dependencies
        if not self._sklearn_available:
            raise ChunkingError(
                "scikit-learn not installed. Cannot use semantic chunking. "
                "Install with: pip install scikit-learn"
            )

        # Import here to avoid import errors if not available

        # Get options
        options.get('unit', 'words')
        tokenizer_name = options.get('tokenizer_name_or_path') or options.get('tokenizer_name')
        if isinstance(tokenizer_name, str) and tokenizer_name.strip():
            self._tokenizer_name = tokenizer_name.strip()
        options.get('similarity_threshold', self.similarity_threshold)
        options.get('min_chunk_size', max_size // 2)

        chunk_spans = self._chunk_text_with_spans(text, max_size, overlap, **options)
        return [chunk for chunk, _start, _end in chunk_spans]

    def _chunk_text_with_spans(
        self,
        text: str,
        max_size: int,
        overlap: int = 0,
        **options,
    ) -> list[tuple[str, int, int]]:
        """Return chunks with exact source spans for semantic splitting."""
        if not self.validate_parameters(text, max_size, overlap):
            return []

        # Check dependencies
        if not self._sklearn_available:
            raise ChunkingError(
                "scikit-learn not installed. Cannot use semantic chunking. "
                "Install with: pip install scikit-learn"
            )

        # Import here to avoid import errors if not available
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Get options
        unit = options.get('unit', 'words')
        tokenizer_name = options.get('tokenizer_name_or_path') or options.get('tokenizer_name')
        if isinstance(tokenizer_name, str) and tokenizer_name.strip():
            self._tokenizer_name = tokenizer_name.strip()
        similarity_threshold = options.get('similarity_threshold', self.similarity_threshold)
        min_chunk_size = options.get('min_chunk_size', max_size // 2)

        # Split into sentences with source spans
        sentences_with_spans = self._split_sentences_with_spans(text)
        if not sentences_with_spans:
            return []

        # Filter out empty sentences while preserving spans
        valid = [(s, start, end) for s, start, end in sentences_with_spans if s and s.strip()]
        if not valid:
            return []

        # Single sentence edge case
        if len(valid) == 1:
            s, start, end = valid[0]
            return [(text[start:end], start, end)]

        # Vectorize sentences
        try:
            vectorizer = TfidfVectorizer()
            valid_sentences = [s for s, _start, _end in valid]
            sentence_vectors = vectorizer.fit_transform(valid_sentences)
        except ValueError as e:
            # Can happen if all words are stop words or text is too short
            logger.warning(
                f"TF-IDF vectorization failed (possibly all stop words): {e}. "
                "Returning single chunk."
            )
            chunk_text = text.strip()
            if not chunk_text:
                return []
            start = text.find(chunk_text)
            if start == -1:
                start = 0
            end = start + len(chunk_text)
            return [(chunk_text, start, end)]

        # Build chunks based on semantic similarity with exact offsets
        chunks: list[tuple[str, int, int]] = []
        current_chunk: list[tuple[str, int, int]] = []
        current_size = 0

        def _emit_chunk(block: list[tuple[str, int, int]]) -> None:
            if not block:
                return
            start = block[0][1]
            end = block[-1][2]
            if end < start:
                end = start
            chunk_text = text[start:end]
            chunks.append((chunk_text, start, end))

        for i, (sentence, start, end) in enumerate(valid):
            sentence_size = self._count_units(sentence, unit)

            # Check if adding sentence exceeds max size
            if current_size + sentence_size > max_size and current_chunk:
                _emit_chunk(current_chunk)

                # Handle overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:]
                    current_size = sum(self._count_units(s, unit) for s, _s0, _e0 in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0

            # Add sentence to current chunk
            current_chunk.append((sentence, start, end))
            current_size += sentence_size

            # Check semantic similarity with next sentence
            if i + 1 < len(valid):
                # Calculate cosine similarity
                current_vector = sentence_vectors[i:i + 1]
                next_vector = sentence_vectors[i + 1:i + 2]

                try:
                    similarity = cosine_similarity(current_vector, next_vector)[0, 0]
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not compute similarity at index {i}: {e}")
                    similarity = 1.0  # Assume high similarity on error

                # Break if similarity is low and chunk is large enough
                if (similarity < similarity_threshold and
                        current_size >= min_chunk_size and
                        current_chunk):

                    _emit_chunk(current_chunk)

                    # Handle overlap
                    if overlap > 0 and len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:]
                        current_size = sum(self._count_units(s, unit) for s, _s0, _e0 in current_chunk)
                    else:
                        current_chunk = []
                        current_size = 0

        # Add remaining sentences
        if current_chunk:
            _emit_chunk(current_chunk)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        spans = self._split_sentences_with_spans(text)
        if spans:
            return [s for s, _start, _end in spans]
        return []

    def _split_sentences_with_spans(self, text: str) -> list[tuple[str, int, int]]:
        """Split sentences and return (sentence, start, end) spans."""
        if not text:
            return []

        if self._nltk_available:
            import nltk

            # Language mapping for NLTK
            nltk_lang_map = {
                'en': 'english',
                'es': 'spanish',
                'fr': 'french',
                'de': 'german',
                'it': 'italian',
                'pt': 'portuguese',
                'nl': 'dutch',
                'pl': 'polish',
                'ru': 'russian',
                'tr': 'turkish'
            }

            nltk_language = nltk_lang_map.get(self.language, 'english')

            try:
                tokenizer = nltk.data.load(f'tokenizers/punkt/{nltk_language}.pickle')
                spans = list(tokenizer.span_tokenize(text))
                results = []
                for start, end in spans:
                    if end <= start:
                        continue
                    sent = text[start:end]
                    if sent.strip():
                        results.append((sent, start, end))
                if results:
                    return results
            except LookupError:
                logger.warning(
                    f"NLTK punkt tokenizer for '{nltk_language}' not found. "
                    "Using fallback sentence splitting."
                )
            except Exception as e:
                logger.error(f"NLTK sentence tokenization failed: {e}")

        # Fallback: regex-based sentence splitting with spans
        import re

        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        spans: list[tuple[str, int, int]] = []
        pos = 0
        for match in re.finditer(sentence_pattern, text):
            end = match.start()
            segment = text[pos:end]
            if segment and segment.strip():
                ltrim = len(segment) - len(segment.lstrip())
                rtrim = len(segment) - len(segment.rstrip())
                start_idx = pos + ltrim
                end_idx = end - rtrim if rtrim else end
                if end_idx > start_idx:
                    spans.append((text[start_idx:end_idx], start_idx, end_idx))
            pos = match.end()

        tail = text[pos:]
        if tail and tail.strip():
            ltrim = len(tail) - len(tail.lstrip())
            rtrim = len(tail) - len(tail.rstrip())
            start_idx = pos + ltrim
            end_idx = len(text) - rtrim if rtrim else len(text)
            if end_idx > start_idx:
                spans.append((text[start_idx:end_idx], start_idx, end_idx))

        if spans:
            return spans

        # Final fallback: split on newlines to retain some spans
        spans = []
        cursor = 0
        for line in text.splitlines(keepends=True):
            line_start = cursor
            line_end = cursor + len(line)
            cursor = line_end
            if not line.strip():
                continue
            ltrim = len(line) - len(line.lstrip())
            rtrim = len(line) - len(line.rstrip())
            start_idx = line_start + ltrim
            end_idx = line_end - rtrim if rtrim else line_end
            if end_idx > start_idx:
                spans.append((text[start_idx:end_idx], start_idx, end_idx))
        return spans

    def _count_units(self, text: str, unit: str) -> int:
        """
        Count units in text.

        Args:
            text: Text to count units in
            unit: Unit type ('words', 'tokens', 'characters')

        Returns:
            Unit count
        """
        if unit == 'words':
            return len(text.split())
        elif unit == 'characters':
            return len(text)
        elif unit == 'tokens':
            # Try to use a tokenizer if available
            try:
                tokenizer_name = self._tokenizer_name or "gpt2"
                tokenizer = self._get_tokenizer(tokenizer_name)
                if tokenizer is None:
                    raise RuntimeError("Tokenizer unavailable")
                return len(tokenizer.encode(text))
            except Exception:
                # Fallback to word count * 1.3 (approximate)
                return int(len(text.split()) * 1.3)
        else:
            logger.warning(f"Unknown unit type '{unit}'. Using word count.")
            return len(text.split())

    def _get_tokenizer(self, tokenizer_name: str):
        """Load and cache a tokenizer for token unit counting.

        Prefers tiktoken when available, otherwise falls back to transformers.
        Uses local-only loading to avoid network calls in restricted environments.
        """
        name = str(tokenizer_name or "gpt2")
        if name in self._tokenizer_failed_names:
            return None
        if self._tokenizer is not None and self._tokenizer_name == name:
            return self._tokenizer
        with self._tokenizer_lock:
            if self._tokenizer is not None and self._tokenizer_name == name:
                return self._tokenizer
            if name in self._tokenizer_failed_names:
                return None
            # Reset cache if switching names
            if self._tokenizer_name != name:
                self._tokenizer = None
                self._tokenizer_type = None
            self._tokenizer_name = name
            # Prefer tiktoken when available
            try:
                import tiktoken  # type: ignore
                try:
                    enc = tiktoken.encoding_for_model(name)
                except Exception:
                    enc = tiktoken.get_encoding("cl100k_base")
                self._tokenizer = enc
                self._tokenizer_type = "tiktoken"
                return self._tokenizer
            except Exception as tiktoken_error:
                logger.debug("Semantic chunker tiktoken initialization failed; trying transformers fallback", exc_info=tiktoken_error)
            # Fallback to transformers (local-only)
            try:
                from transformers import AutoTokenizer  # type: ignore
                tok = AutoTokenizer.from_pretrained(name, local_files_only=True)  # nosec B615
                self._tokenizer = tok
                self._tokenizer_type = "transformers"
                return self._tokenizer
            except Exception as e:
                logger.debug(f"Tokenizer load failed for '{name}', using fallback approximation: {e}")
                self._tokenizer_failed_names.add(name)
                self._tokenizer = None
                self._tokenizer_type = None
                return None

    def chunk_generator(self,
                       text: str,
                       max_size: int,
                       overlap: int = 0,
                       **options) -> Generator[str, None, None]:
        """
        Generator version of chunk method.

        Note: Semantic chunking requires analyzing the entire text
        to compute similarities, so this just yields pre-computed chunks.

        Args:
            text: Text to chunk
            max_size: Maximum size per chunk
            overlap: Number of sentences to overlap
            **options: Additional options

        Yields:
            Individual text chunks
        """
        # Semantic chunking needs full text analysis, so compute all chunks first
        chunks = self.chunk(text, max_size, overlap, **options)
        yield from chunks

    def chunk_with_metadata(self,
                           text: str,
                           max_size: int,
                           overlap: int = 0,
                           **options) -> list[ChunkResult]:
        """
        Chunk text and return results with metadata.

        Args:
            text: Text to chunk
            max_size: Maximum size per chunk
            overlap: Number of sentences to overlap
            **options: Additional options

        Returns:
            List of ChunkResult objects with metadata
        """
        chunk_spans = self._chunk_text_with_spans(text, max_size, overlap, **options)
        results = []

        for i, (chunk_text, start_pos, end_pos) in enumerate(chunk_spans):
            metadata = ChunkMetadata(
                index=i,
                start_char=start_pos,
                end_char=end_pos,
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
                sentence_count=len(self._split_sentences(chunk_text)),
                language=self.language,
                method='semantic',
                options={
                    'similarity_threshold': options.get('similarity_threshold', self.similarity_threshold),
                    'unit': options.get('unit', 'words'),
                    'overlap': overlap
                }
            )

            results.append(ChunkResult(
                text=chunk_text,
                metadata=metadata
            ))

        return results
