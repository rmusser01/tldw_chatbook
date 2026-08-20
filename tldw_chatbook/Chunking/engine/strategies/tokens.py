# strategies/tokens.py
"""
Token-based chunking strategy.
Splits text into chunks based on token count using transformers or fallback methods.
"""

import threading
from collections.abc import Generator
from typing import Optional, Protocol

from loguru import logger

from ..base import BaseChunkingStrategy, ChunkMetadata, ChunkResult
from ..exceptions import TokenizerError

_TOKENS_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ImportError,
    UnicodeError,
)

_TOKENS_TIKTOKEN_RESOLUTION_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
)

_TOKENS_TOKENIZATION_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
)

_TOKENS_DECODE_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    NotImplementedError,
)


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer interface."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        ...


class TransformersTokenizer:
    """Tokenizer wrapper using transformers library."""

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize transformers tokenizer.

        Args:
            model_name: Name of the tokenizer model
        """
        self.model_name = model_name
        self._tokenizer = None
        self._transformers = None

        # Try to import transformers
        try:
            import transformers
            self._transformers = transformers
            self.available = True
            logger.debug("transformers library available for tokenization")
        except ImportError:
            self.available = False
            logger.debug("transformers not available, will use fallback tokenization")

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer when first accessed."""
        if self._tokenizer is None:
            if not self.available:
                raise ImportError(
                    "transformers library not found. Please install it for token-based chunking: "
                    "pip install transformers"
                )

            try:
                AutoTokenizer = self._transformers.AutoTokenizer
                logger.info(f"Loading tokenizer: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.debug(f"Tokenizer {self.model_name} loaded successfully")
            except _TOKENS_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Failed to load tokenizer '{self.model_name}': {e}")
                raise TokenizerError(f"Failed to load tokenizer: {e}") from e

        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encode(text))


class TiktokenTokenizer:
    """Tokenizer wrapper using tiktoken library (OpenAI-style encodings)."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize tiktoken tokenizer.

        Args:
            model_name: Model name to select appropriate encoding.
        """
        self.model_name = model_name
        self.available = False
        self._enc = None

        try:
            import tiktoken  # type: ignore
            self._tiktoken = tiktoken
            # Prefer model-specific encoding; fallback to cl100k_base
            try:
                self._enc = tiktoken.encoding_for_model(model_name)
            except _TOKENS_TIKTOKEN_RESOLUTION_EXCEPTIONS:
                self._enc = tiktoken.get_encoding("cl100k_base")
            self.available = True
            logger.debug(f"tiktoken available for tokenization (model={model_name})")
        except ImportError:
            self._tiktoken = None
            logger.debug("tiktoken not available, will try transformers or fallback")

    def encode(self, text: str) -> list[int]:
        if not self.available or self._enc is None:
            raise ImportError("tiktoken not available")
        return list(self._enc.encode(text))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        if not self.available or self._enc is None:
            raise ImportError("tiktoken not available")
        # tiktoken decode ignores skip_special_tokens; it's fine for chunking
        return self._enc.decode(token_ids)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))


class FallbackTokenizer:
    """Fallback tokenizer using simple word/character splitting."""

    def __init__(self, model_name: str = "fallback"):
        """
        Initialize fallback tokenizer.

        Args:
            model_name: Name identifier (for compatibility)
        """
        self.model_name = model_name
        self.available = True
        # Average tokens per word for different models
        self.tokens_per_word = {
            'gpt2': 1.3,
            'gpt-3.5-turbo': 1.3,
            'gpt-4': 1.3,
            'claude': 1.2,
            'llama': 1.5,
            'default': 1.3
        }

        logger.warning(
            "Using fallback tokenizer (word-based approximation). "
            "Install transformers for accurate token-based chunking: pip install transformers"
        )

    def encode(self, text: str) -> list[int]:
        """
        Simulate encoding by splitting into words and creating fake token IDs.

        Args:
            text: Text to encode

        Returns:
            List of fake token IDs
        """
        # Split into words and subwords
        import hashlib
        import re

        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)

        # Create consistent fake token IDs using MD5 for better distribution
        # Using a larger modulus (2^31) to minimize collision probability
        token_ids = []
        for token in tokens:
            # Use MD5 hash for better distribution and consistency across runs
            h = hashlib.md5(token.encode('utf-8', errors='replace'), usedforsecurity=False).hexdigest()
            token_id = int(h[:8], 16) % 2147483647  # 2^31 - 1
            token_ids.append(token_id)

            # Simulate subword tokenization for longer words
            if len(token) > 7:
                # Add extra tokens for long words
                extra_tokens = (len(token) - 7) // 3
                for i in range(extra_tokens):
                    h = hashlib.md5(
                        f"{token}_{i}".encode('utf-8', errors='replace'),
                        usedforsecurity=False,
                    ).hexdigest()
                    token_ids.append(int(h[:8], 16) % 2147483647)

        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Cannot accurately decode from fake token IDs.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Placeholder text

        Raises:
            NotImplementedError: Always (cannot reverse fake tokenization)
        """
        raise NotImplementedError(
            "Fallback tokenizer cannot decode token IDs back to text. "
            "This should not be called in normal chunking operations."
        )

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count using word count.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate token count
        """
        word_count = len(text.split())
        # Use model-specific ratio if available
        ratio = self.tokens_per_word.get(self.model_name, self.tokens_per_word['default'])
        return int(word_count * ratio)


class TokenChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks text by token count.
    Uses transformers library when available, falls back to word-based approximation.
    """

    # Class-level cache for failed tokenizer names to avoid repeated import attempts
    # Protected by _failed_tokenizers_lock for thread-safe access
    _failed_tokenizers: set = set()
    _failed_tokenizers_lock = threading.Lock()

    def __init__(self,
                 language: str = 'en',
                 tokenizer_name: str = 'gpt2'):
        """
        Initialize token chunking strategy.

        Args:
            language: Language code for text processing
            tokenizer_name: Name of the tokenizer to use
        """
        super().__init__(language)
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
        self._tokenizer_init_attempted = False

        logger.debug(f"TokenChunkingStrategy initialized with tokenizer: {tokenizer_name}")

    @property
    def tokenizer(self) -> TokenizerProtocol:
        """Get or create tokenizer instance with caching of failed attempts."""
        if self._tokenizer is None:
            # Check if this tokenizer previously failed (thread-safe read)
            with TokenChunkingStrategy._failed_tokenizers_lock:
                previously_failed = self.tokenizer_name in TokenChunkingStrategy._failed_tokenizers

            # If we've already tried and failed for this tokenizer name, go straight to fallback
            if self._tokenizer_init_attempted or previously_failed:
                if not self._tokenizer_init_attempted:
                    logger.debug(f"Tokenizer '{self.tokenizer_name}' previously failed, using fallback")
                self._tokenizer = FallbackTokenizer(self.tokenizer_name)
                self._tokenizer_init_attempted = True
                return self._tokenizer

            self._tokenizer_init_attempted = True

            # Prefer tiktoken when available (fast, consistent for OpenAI-family models)
            try:
                tk = TiktokenTokenizer(self.tokenizer_name)
                if tk.available:
                    _ = tk.encode("test")
                    self._tokenizer = tk
                    logger.info(f"Using tiktoken tokenizer: {self.tokenizer_name}")
                    return self._tokenizer
                else:
                    raise ImportError("tiktoken not available")
            except ImportError as te:
                logger.debug(f"tiktoken not available: {te}")
            except (RuntimeError, ValueError, OSError) as te:
                logger.debug(f"tiktoken initialization failed: {te}")

            # Try transformers next
            try:
                self._tokenizer = TransformersTokenizer(self.tokenizer_name)
                _ = self._tokenizer.encode("test")
                logger.info(f"Using transformers tokenizer: {self.tokenizer_name}")
                return self._tokenizer
            except ImportError as e:
                logger.debug(f"transformers not available: {e}")
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning(f"Could not initialize transformers tokenizer '{self.tokenizer_name}': {e}")

            # Cache this tokenizer name as failed to avoid repeated attempts (thread-safe write)
            with TokenChunkingStrategy._failed_tokenizers_lock:
                TokenChunkingStrategy._failed_tokenizers.add(self.tokenizer_name)
            logger.info("Falling back to word-based token approximation")
            self._tokenizer = FallbackTokenizer(self.tokenizer_name)

        return self._tokenizer

    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> list[str]:
        """
        Chunk text by token count.

        Args:
            text: Text to chunk
            max_size: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            **options: Additional options:
                - preserve_words: Try not to break words (when possible)
                - add_special_tokens: Include special tokens in encoding

        Returns:
            List of text chunks
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []

        # Validate token-specific parameters
        MAX_CHUNK_SIZE_TOKENS = 10000  # Safety limit
        if max_size > MAX_CHUNK_SIZE_TOKENS:
            raise ValueError(
                f"max_size {max_size} exceeds maximum allowed {MAX_CHUNK_SIZE_TOKENS} tokens"
            )

        # Adjust overlap if needed
        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), setting to max_size - 1")
            overlap = max_size - 1

        # For fallback tokenizer, use different approach
        if isinstance(self.tokenizer, FallbackTokenizer):
            return self._chunk_with_fallback(text, max_size, overlap, **options)

        # Encode text to tokens (and capture offsets when available)
        token_ids: list[int]
        offsets: Optional[list[tuple[int, int]]] = None
        try:
            add_special = options.get('add_special_tokens', False)
            if hasattr(self.tokenizer, 'tokenizer'):
                # For TransformersTokenizer
                tok = self.tokenizer.tokenizer
                try:
                    enc = tok(
                        text,
                        add_special_tokens=add_special,
                        return_offsets_mapping=True,
                        return_attention_mask=False,
                        return_special_tokens_mask=False,
                    )
                    token_ids = list(enc.get('input_ids') or enc['input_ids'])
                    offsets = list(enc.get('offset_mapping') or enc['offset_mapping'])
                except _TOKENS_TOKENIZATION_EXCEPTIONS:
                    token_ids = tok.encode(text, add_special_tokens=add_special)
                    offsets = None
            else:
                token_ids = self.tokenizer.encode(text)

        except _TOKENS_TOKENIZATION_EXCEPTIONS as e:
            logger.error(f"Tokenization failed: {e}")
            raise TokenizerError(f"Failed to tokenize text: {e}") from e

        if not token_ids:
            return []

        if offsets is not None and len(offsets) != len(token_ids):
            offsets = None

        logger.debug(f"Tokenized text into {len(token_ids)} tokens")

        # Create chunks
        chunks = []
        step = max(1, max_size - overlap)

        for i in range(0, len(token_ids), step):
            chunk_tokens = token_ids[i:i + max_size]
            if not chunk_tokens:
                continue

            if offsets is not None:
                start_idx = i
                end_idx = min(i + len(chunk_tokens) - 1, len(offsets) - 1)
                # Skip zero-width tokens at the bounds
                while start_idx < len(offsets) and (offsets[start_idx][1] - offsets[start_idx][0]) == 0:
                    start_idx += 1
                while end_idx > start_idx and (offsets[end_idx][1] - offsets[end_idx][0]) == 0:
                    end_idx -= 1
                if start_idx < len(offsets) and end_idx >= start_idx:
                    start_char = int(offsets[start_idx][0])
                    end_char = int(offsets[end_idx][1])
                    if end_char < start_char:
                        end_char = start_char
                    chunk_text = text[start_char:end_char]
                else:
                    chunk_text = ""
            else:
                # Decode tokens back to text
                try:
                    if hasattr(self.tokenizer, 'decode'):
                        try:
                            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                        except TypeError:
                            chunk_text = self.tokenizer.decode(chunk_tokens)
                    elif hasattr(self.tokenizer, 'tokenizer') and hasattr(self.tokenizer.tokenizer, 'decode'):
                        try:
                            chunk_text = self.tokenizer.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                        except TypeError:
                            chunk_text = self.tokenizer.tokenizer.decode(chunk_tokens)
                    else:
                        raise AttributeError('No decode() available on tokenizer or underlying implementation')
                except _TOKENS_DECODE_EXCEPTIONS as e:
                    logger.warning(f"Failed to decode chunk at position {i}: {e}; falling back to word approximation")
                    return self._chunk_with_fallback(text, max_size, overlap, **options)

            if chunk_text:
                chunks.append(chunk_text)

        logger.debug(f"Created {len(chunks)} token-based chunks")
        return chunks

    def _chunk_with_fallback(self,
                            text: str,
                            max_size: int,
                            overlap: int,
                            **options) -> list[str]:
        """
        Chunk using fallback tokenizer (word-based approximation).

        Args:
            text: Text to chunk
            max_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
            **options: Additional options

        Returns:
            List of text chunks
        """
        # Convert token counts to approximate word counts
        if hasattr(self.tokenizer, 'tokens_per_word'):
            ratio = self.tokenizer.tokens_per_word.get(
                self.tokenizer.model_name,
                1.3
            )
        else:
            ratio = 1.3

        # Calculate word counts
        max_words = max(1, int(max_size / ratio))
        raw_overlap_words = int(overlap / ratio)
        overlap_words = 0 if max_words == 1 else max(0, min(max_words - 1, raw_overlap_words))

        logger.debug(f"Using fallback: {max_size} tokens ≈ {max_words} words")

        # Split into words
        words = text.split()

        if not words:
            return []

        # Create chunks
        chunks = []
        step = max(1, max_words - overlap_words)

        for i in range(0, len(words), step):
            chunk_words = words[i:i + max_words]
            chunk_text = ' '.join(chunk_words)

            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        if hasattr(self.tokenizer, 'count_tokens'):
            return self.tokenizer.count_tokens(text)
        else:
            return len(self.tokenizer.encode(text))

    def chunk_with_metadata(self,
                            text: str,
                            max_size: int,
                            overlap: int = 0,
                            **options) -> list[ChunkResult]:
        """Chunk text and return metadata with best-possible char offsets.

        - Transformers (fast) tokenizers: use offset mapping to compute spans.
        - tiktoken: rebuild per-token spans by decoding tokens sequentially.
        - Fallback tokenizer: approximate via word windows; precise char spans,
          approximate token counts.
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []

        MAX_CHUNK_SIZE_TOKENS = 10000
        if max_size > MAX_CHUNK_SIZE_TOKENS:
            raise ValueError(
                f"max_size {max_size} exceeds maximum allowed {MAX_CHUNK_SIZE_TOKENS} tokens"
            )

        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), setting to max_size - 1")
            overlap = max_size - 1

        # Fallback path: approximate by words for reliable character spans
        if isinstance(self.tokenizer, FallbackTokenizer):
            return self._chunk_with_metadata_fallback(text, max_size, overlap, **options)

        add_special = bool(options.get('add_special_tokens', False))

        token_ids: list[int]
        offsets: Optional[list[tuple[int, int]]] = None
        decode_fn = None

        try:
            if hasattr(self.tokenizer, 'tokenizer'):
                tok = self.tokenizer.tokenizer
                # Choose an available decode: prefer wrapper.decode if present,
                # otherwise fall back to the underlying tokenizer's decode.
                if hasattr(self.tokenizer, 'decode'):
                    def decode_fn(ids):
                        return self.tokenizer.decode(ids, skip_special_tokens=True)
                elif hasattr(tok, 'decode'):
                    def decode_fn(ids):
                        return tok.decode(ids)
                else:
                    decode_fn = None
                try:
                    enc = tok(
                        text,
                        add_special_tokens=add_special,
                        return_offsets_mapping=True,
                        return_attention_mask=False,
                        return_special_tokens_mask=False,
                    )
                    token_ids = list(enc.get('input_ids') or enc['input_ids'])
                    offsets = list(enc.get('offset_mapping') or enc['offset_mapping'])
                except _TOKENS_TOKENIZATION_EXCEPTIONS:
                    token_ids = tok.encode(text, add_special_tokens=add_special)
                    offsets = None
            else:
                token_ids = self.tokenizer.encode(text)
                # Generic tokenizer path (e.g., tiktoken or simple mocks): use plain decode(ids)
                decode_fn = (lambda ids: self.tokenizer.decode(ids)) if hasattr(self.tokenizer, 'decode') else None
                offsets = self._reconstruct_offsets_by_decoding(token_ids, text)
        except _TOKENS_TOKENIZATION_EXCEPTIONS as e:
            logger.error(f"Tokenization failed: {e}")
            raise TokenizerError(f"Failed to tokenize text: {e}") from e

        if not token_ids:
            return []

        # Ensure offsets align with token ids length when present
        if offsets is not None and len(offsets) != len(token_ids):
            try:
                offsets = self._reconstruct_offsets_by_decoding(token_ids, text)
            except _TOKENS_TOKENIZATION_EXCEPTIONS:
                offsets = None

        results: list[ChunkResult] = []
        step = max(1, max_size - overlap)
        # Rolling pointer for fallback bounds mapping when offsets are unavailable
        rolling_pos = 0

        for i in range(0, len(token_ids), step):
            ids_window = token_ids[i:i + max_size]
            if not ids_window:
                continue
            # Decode without trimming to preserve exact mapping
            try:
                if decode_fn is not None:
                    chunk_text = decode_fn(ids_window)
                else:
                    chunk_text = self.tokenizer.decode(ids_window, skip_special_tokens=True)
            except _TOKENS_DECODE_EXCEPTIONS as e:
                logger.warning(f"Failed to decode token window at {i}: {e}")
                continue

            if offsets is not None:
                start_idx = i
                end_idx = i + len(ids_window) - 1
                # Clamp end_idx to valid range before iterating
                end_idx = min(end_idx, len(offsets) - 1)
                # Skip zero-width tokens at the start
                while start_idx < len(offsets) and (offsets[start_idx][1] - offsets[start_idx][0]) == 0:
                    start_idx += 1
                # Skip zero-width tokens at the end, but don't go below start_idx
                while end_idx > start_idx and (offsets[end_idx][1] - offsets[end_idx][0]) == 0:
                    end_idx -= 1
                if start_idx < len(offsets) and end_idx >= start_idx and end_idx < len(offsets):
                    start_char = int(offsets[start_idx][0])
                    end_char = int(offsets[end_idx][1])
                    # Expand end bound to avoid slicing mid-grapheme when safe
                    try:
                        expanded_end = self._expand_end_to_grapheme_boundary(text, end_char)
                        if expanded_end != end_char:
                            # Only adopt expansion if it doesn't contradict the decoded chunk materially
                            import unicodedata as _ud
                            def _strip_cf(s: str) -> str:
                                return ''.join(ch for ch in s if _ud.category(ch) != 'Cf')
                            a = _strip_cf(text[start_char:expanded_end])
                            b = _strip_cf(chunk_text)
                            # Accept if either equals or one is a prefix of the other (ZWJ/vs16 tolerated)
                            if a == b or a.startswith(b) or b.startswith(a):
                                end_char = expanded_end
                    except _TOKENS_NONCRITICAL_EXCEPTIONS:
                        pass
                else:
                    start_char, end_char = self._bounds_via_rolling_pointer(text, chunk_text, start_from=rolling_pos)
                    rolling_pos = end_char
            else:
                start_char, end_char = self._bounds_via_rolling_pointer(text, chunk_text, start_from=rolling_pos)
                rolling_pos = end_char

            # Skip zero-length or unmapped spans to maintain monotonic progress
            if end_char <= start_char:
                continue

            md = ChunkMetadata(
                index=len(results),
                start_char=start_char,
                end_char=end_char,
                word_count=len(chunk_text.split()) if chunk_text else 0,
                token_count=len(ids_window),
                language=self.language,
                overlap_with_previous=overlap if i > 0 else 0,
                overlap_with_next=overlap if (i + step) < len(token_ids) else 0,
                method='tokens',
                options={'add_special_tokens': add_special},
            )
            results.append(ChunkResult(text=chunk_text, metadata=md))

        logger.debug(f"Created {len(results)} token-based chunks with metadata")
        return results

    # ------------------------ helpers for offsets ------------------------
    def _expand_end_to_grapheme_boundary(self, text: str, end: int) -> int:
        # Delegate to BaseChunkingStrategy implementation (config-aware)
        return super()._expand_end_to_grapheme_boundary(text, end)

    def _reconstruct_offsets_by_decoding(self, token_ids: list[int], text: str) -> list[tuple[int, int]]:
        """Rebuild per-token char spans using sequential decode with grapheme-safe clamping.

        Strategy:
        - Prefer direct sequential mapping: decode all tokens; if it matches the
          original `text` exactly, compute spans by cumulative lengths without
          substring search (robust for ZWJ/modifiers and repeated substrings).
        - Fallback: attempt localized search from a rolling pointer with a small
          Cf-agnostic (zero-width) match, and clamp indices to [0, len(text)].
        """
        # Select decode function for single token and full sequence
        def _decode_one(tid: int) -> str:
            try:
                # Prefer wrapper decode to ensure skip_special_tokens=True
                return self.tokenizer.decode([tid], skip_special_tokens=True)
            except _TOKENS_DECODE_EXCEPTIONS:
                try:
                    if hasattr(self.tokenizer, 'tokenizer'):
                        return self.tokenizer.tokenizer.decode([tid], skip_special_tokens=True)
                except _TOKENS_DECODE_EXCEPTIONS:
                    pass
                return ''

        def _decode_all(tids: list[int]) -> str:
            try:
                return self.tokenizer.decode(tids, skip_special_tokens=True)
            except _TOKENS_DECODE_EXCEPTIONS:
                try:
                    if hasattr(self.tokenizer, 'tokenizer'):
                        return self.tokenizer.tokenizer.decode(tids, skip_special_tokens=True)
                except _TOKENS_DECODE_EXCEPTIONS:
                    pass
                return ''

        decoded_all = _decode_all(token_ids)

        offsets: list[tuple[int, int]] = []
        n = len(text)

        if decoded_all == text:
            # Fast path: cumulative lengths map 1:1 to original text
            pos = 0
            for tid in token_ids:
                piece = _decode_one(tid)
                start = pos
                end = start + len(piece)
                if end > n:
                    end = n
                if start < 0:
                    start = 0
                offsets.append((start, end))
                pos = end
            return offsets

        # Fallback path: tolerant forward scan with Cf-insensitive lookups
        import unicodedata as _ud

        def _strip_cf(s: str) -> str:
            return ''.join(ch for ch in s if _ud.category(ch) != 'Cf')

        pos = 0
        for tid in token_ids:
            piece = _decode_one(tid)
            if not piece:
                offsets.append((pos, pos))
                continue
            idx = text.find(piece, pos)
            if idx == -1:
                # Try a small sliding window search ignoring Cf to better align
                window_end = min(n, pos + max(64, len(piece) * 4))
                window = text[pos:window_end]
                sp = _strip_cf(piece)
                sw = _strip_cf(window)
                rel = sw.find(sp) if sp else -1
                if rel != -1:
                    idx = pos + rel
            if idx == -1:
                # Fallback: token piece not found in text - use current position
                # This may produce incorrect offsets for unusual tokenization
                logger.debug(
                    f"Token piece not found in text during offset reconstruction: "
                    f"piece={repr(piece[:50] + '...' if len(piece) > 50 else piece)}, pos={pos}"
                )
                idx = pos
            start = max(0, min(idx, n))
            end = min(n, start + len(piece))
            if end < start:
                end = start
            offsets.append((start, end))
            pos = end
        return offsets

    def _bounds_via_rolling_pointer(self, text: str, chunk_text: str, start_from: int = 0) -> tuple[int, int]:
        """Compute approximate bounds by scanning forward for the chunk text.

        Uses a rolling start to avoid mapping every repeated substring to its
        first occurrence in the source.
        """
        n = len(text)
        sf = max(0, min(start_from, n))
        idx = text.find(chunk_text, sf)
        if idx == -1:
            idx = sf
        end = min(n, idx + len(chunk_text))
        if end < idx:
            end = idx
        return idx, end

    def _chunk_with_metadata_fallback(self, text: str, max_size: int, overlap: int, **options) -> list[ChunkResult]:
        """Approximate token windows using words; precise char spans, approximate token counts."""
        ratio = getattr(self.tokenizer, 'tokens_per_word', {}).get(getattr(self.tokenizer, 'model_name', 'default'), 1.3)
        max_words = max(1, int(max_size / ratio))
        raw_overlap_words = int(overlap / ratio)
        overlap_words = 0 if max_words == 1 else max(0, min(max_words - 1, raw_overlap_words))

        # Build word spans
        spans: list[tuple[int, int]] = []
        words: list[str] = []
        pos = 0
        for part in text.split():
            idx = text.find(part, pos)
            if idx == -1:
                idx = pos
            start = idx
            end = idx + len(part)
            spans.append((start, end))
            words.append(part)
            pos = end

        if not words:
            return []

        results: list[ChunkResult] = []
        step = max(1, max_words - overlap_words)
        wi = 0
        while wi < len(words):
            j = min(len(words), wi + max_words)
            if j <= wi:
                break
            start_char = spans[wi][0]
            end_char = spans[j - 1][1]
            chunk_text = ' '.join(words[wi:j])
            md = ChunkMetadata(
                index=len(results),
                start_char=start_char,
                end_char=end_char,
                word_count=j - wi,
                token_count=int(round((j - wi) * ratio)),
                language=self.language,
                overlap_with_previous=overlap_words if wi > 0 else 0,
                overlap_with_next=overlap_words if j < len(words) else 0,
                method='tokens',
                options={'approximate': True},
            )
            results.append(ChunkResult(text=chunk_text, metadata=md))
            wi += step
        return results
    def chunk_generator(self,
                       text: str,
                       max_size: int,
                       overlap: int = 0,
                       **options) -> Generator[str, None, None]:
        """
        Memory-efficient generator version of chunk.

        Args:
            text: Text to chunk
            max_size: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            **options: Additional options

        Yields:
            Individual text chunks
        """
        # For token-based chunking, we need to encode the full text
        # So we just use the regular chunk method
        chunks = self.chunk(text, max_size, overlap, **options)
        yield from chunks
