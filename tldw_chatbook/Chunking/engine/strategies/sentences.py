# strategies/sentences.py
"""
Sentence-based chunking strategy.
Splits text into chunks based on sentence count with optional overlap.
"""

import contextlib
import re
from collections.abc import Generator
from typing import Any, Optional

from loguru import logger

from ..base import BaseChunkingStrategy, ChunkMetadata, ChunkResult

_SENTENCE_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    re.error,
)


class SentenceChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks text by sentence count.
    Uses language-specific sentence boundary detection.
    """

    def __init__(self, language: str = 'en'):
        """
        Initialize sentence chunking strategy.

        Args:
            language: Language code for text processing
        """
        super().__init__(language)

        # Try to import pysbd for better sentence splitting
        self.pysbd_available = False
        try:
            import pysbd
            self.pysbd = pysbd
            self.pysbd_available = True
            logger.debug("pysbd available for sentence segmentation")
        except ImportError:
            logger.debug("pysbd not available, using fallback sentence splitting")

        # Language-specific sentence delimiters
        self.sentence_delimiters = {
            'zh': ['。', '！', '？', '；'],
            'zh-cn': ['。', '！', '？', '；'],
            'zh-tw': ['。', '！', '？', '；'],
            'ja': ['。', '！', '？'],
            'ko': ['.', '!', '?', '。', '！', '？'],
            'ar': ['.', '!', '?', '؟', '۔'],
            'hi': ['।', '|', '.', '!', '?'],
            # Thai has no explicit spaces between sentences; avoid space as a delimiter.
            # Prefer PyThaiNLP when available; fallback uses a conservative set of marks.
            # Include '…' (ellipsis) and 'ฯ' (paiyannoi) commonly seen at sentence/clause ends.
            'th': ['!', '?', '…', 'ฯ'],
            'default': ['.', '!', '?']
        }

        logger.debug(f"SentenceChunkingStrategy initialized for language: {language}")

        # Optional Thai sentence tokenizer (PyThaiNLP)
        self.pythainlp_available = False
        self._th_sent_tokenize = None
        if self.language == 'th':
            try:
                from pythainlp.tokenize import sent_tokenize as _th_sent_tokenize  # type: ignore
                self._th_sent_tokenize = _th_sent_tokenize
                self.pythainlp_available = True
                logger.debug("PyThaiNLP available for Thai sentence segmentation")
            except ImportError:
                logger.debug("PyThaiNLP not available; using regex fallback for Thai")

    def set_language(self, language: str) -> None:
        if language == self.language:
            return
        self.language = language
        self.pythainlp_available = False
        self._th_sent_tokenize = None
        if self.language == 'th':
            try:
                from pythainlp.tokenize import sent_tokenize as _th_sent_tokenize  # type: ignore
                self._th_sent_tokenize = _th_sent_tokenize
                self.pythainlp_available = True
                logger.debug("PyThaiNLP available for Thai sentence segmentation")
            except ImportError:
                logger.debug("PyThaiNLP not available; using regex fallback for Thai")

    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> list[str]:
        """
        Chunk text by sentence count.

        Args:
            text: Text to chunk
            max_size: Maximum sentences per chunk
            overlap: Number of sentences to overlap between chunks
            **options: Additional options:
                - combine_short: Combine short sentences
                - min_sentence_length: Minimum characters per sentence

        Returns:
            List of text chunks
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []

        # Adjust overlap if needed
        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), setting to max_size - 1")
            overlap = max_size - 1

        records, combined_sentences = self._prepare_chunk_records(text, max_size, overlap, **options)
        if not combined_sentences or not records:
            return []
        chunks = [record['text'] for record in records]
        logger.debug(f"Created {len(chunks)} chunks from {len(combined_sentences)} sentences")
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences based on language.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Thai first: prefer PyThaiNLP when available
        if self.language == 'th' and self.pythainlp_available and callable(self._th_sent_tokenize):
            try:
                sents = [s for s in self._th_sent_tokenize(text) if s and s.strip()]
                if sents:
                    return sents
            except _SENTENCE_NONCRITICAL_EXCEPTIONS:
                logger.debug("PyThaiNLP sentence splitting failed; falling back")

        # Try pysbd next if available
        if self.pysbd_available:
            sentences = self._split_with_pysbd(text)
            if sentences:
                return sentences

        # Fallback to language-specific splitting
        return self._split_with_regex(text)

    def _split_sentences_with_spans(self, text: str) -> list[tuple[str, int, int]]:
        """Split sentences and return (sentence, start, end) spans.

        Uses the same underlying splitting as _split_sentences to ensure
        parity, but carries start/end offsets robustly (avoids naive find()).
        """
        # Thai first: prefer PyThaiNLP when available; recover spans via rolling pointer
        if self.language == 'th' and self.pythainlp_available and callable(self._th_sent_tokenize):
            try:
                sentences = [s for s in self._th_sent_tokenize(text) if s and s.strip()]
                spans = []
                pos = 0
                for s in sentences:
                    idx = text.find(s, pos)
                    if idx == -1:
                        idx = pos
                    end = min(idx + len(s), len(text))  # Bounds check
                    spans.append((s, idx, end))
                    pos = end
                if spans:
                    return spans
            except _SENTENCE_NONCRITICAL_EXCEPTIONS:
                logger.debug("PyThaiNLP sentence splitting (spans) failed; falling back")

        # Try pysbd next if available; if used, recover spans via rolling pointer
        if self.pysbd_available:
            try:
                sentences = self._split_with_pysbd(text)
                spans = []
                pos = 0
                for s in sentences:
                    # Use simple forward scan to locate the next sentence occurrence
                    # This is safe because segment order is preserved and we advance pos
                    idx = text.find(s, pos)
                    if idx == -1:
                        idx = pos
                    end = min(idx + len(s), len(text))  # Bounds check
                    spans.append((s, idx, end))
                    pos = end
                return spans
            except _SENTENCE_NONCRITICAL_EXCEPTIONS:
                pass

        # Regex path: compute spans directly during reconstruction
        delimiters = self.sentence_delimiters.get(
            self.language,
            self.sentence_delimiters['default']
        )
        delimiter_pattern = '|'.join(re.escape(d) for d in delimiters)
        pattern = f'([{delimiter_pattern}])'

        parts = re.split(pattern, text)

        spans: list[tuple[str, int, int]] = []
        cur_txt = ""
        cur_start = 0
        pos = 0
        for part in parts:
            if part in delimiters:
                if cur_txt:
                    sent = cur_txt + part
                    start = cur_start
                    end = min(pos + len(part), len(text))  # Bounds check
                    spans.append((sent.strip(), start, end))
                    cur_txt = ""
                    cur_start = end
                pos += len(part)
            else:
                if not cur_txt:
                    # Trim leading whitespace for the new sentence and adjust start
                    # Use Unicode-aware whitespace stripping for proper offset calculation
                    lstripped = part.lstrip()  # Unicode-aware whitespace strip
                    ltrim = len(part) - len(lstripped)
                    cur_start = pos + ltrim
                    cur_txt += lstripped
                else:
                    cur_txt += part
                pos += len(part)
        if cur_txt.strip():
            stripped = cur_txt.strip()
            # Adjust start to match stripped content if leading whitespace remained
            # Use Unicode-aware whitespace stripping for proper offset calculation
            adjust = len(cur_txt) - len(cur_txt.lstrip())
            start = cur_start + adjust
            end = min(start + len(stripped), len(text))  # Bounds check
            spans.append((stripped, start, end))
        return spans

    def _split_with_pysbd(self, text: str) -> list[str]:
        """
        Split sentences using pysbd library.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        try:
            # Map our language codes to pysbd language codes
            pysbd_lang_map = {
                'en': 'en',
                'de': 'de',
                'fr': 'fr',
                'it': 'it',
                'es': 'es',
                'pt': 'pt',
                'nl': 'nl',
                'pl': 'pl',
                'zh': 'zh',
                'zh-cn': 'zh',
                'zh-tw': 'zh',
                'ja': 'ja',
                'ar': 'ar',
                'hi': 'hi',
                'ru': 'ru',
                'da': 'da',
                'sv': 'sv',
                'no': 'no'
            }

            lang_code = pysbd_lang_map.get(self.language, 'en')
            segmenter = self.pysbd.Segmenter(language=lang_code, clean=False)
            sentences = segmenter.segment(text)

            logger.debug(f"pysbd split text into {len(sentences)} sentences")
            return sentences

        except _SENTENCE_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"pysbd sentence splitting failed: {e}, falling back to regex")
            return []

    def _split_with_regex(self, text: str) -> list[str]:
        """
        Split sentences using regex patterns.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Get delimiters for language
        delimiters = self.sentence_delimiters.get(
            self.language,
            self.sentence_delimiters['default']
        )

        # Build regex pattern
        delimiter_pattern = '|'.join(re.escape(d) for d in delimiters)
        pattern = f'([{delimiter_pattern}])'

        # Split on delimiters but keep them
        parts = re.split(pattern, text)

        # Reconstruct sentences with their delimiters
        sentences = []
        current_sentence = ""

        for _i, part in enumerate(parts):
            if part in delimiters:
                if current_sentence:
                    current_sentence += part
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
            else:
                current_sentence += part

        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        # Filter out empty sentences
        sentences = [s for s in sentences if s.strip()]

        logger.debug(f"Regex split text into {len(sentences)} sentences")
        return sentences

    def _combine_short_sentences_with_spans(
        self,
        sentences_with_spans: list[tuple[str, int, int]],
        min_length: int,
    ) -> list[tuple[str, int, int]]:
        """Combine short sentences while preserving spans."""
        combined: list[tuple[str, int, int]] = []
        current_text = ""
        current_start: Optional[int] = None
        current_end: Optional[int] = None
        min_length = max(0, int(min_length))

        no_space_languages = {'zh', 'zh-cn', 'zh-tw', 'ja', 'th'}

        for sentence, start, end in sentences_with_spans:
            if current_start is None:
                current_text = sentence
                current_start = start
                current_end = end
                continue

            if len(current_text) + len(sentence) < min_length:
                if self.language in no_space_languages:
                    current_text += sentence
                else:
                    current_text = (current_text + " " + sentence).strip()
                current_end = end
            else:
                combined.append((current_text, current_start, current_end if current_end is not None else end))
                current_text = sentence
                current_start = start
                current_end = end

        if current_start is not None:
            combined.append((current_text, current_start, current_end if current_end is not None else current_start))

        return combined

    def _combine_short_sentences(self, sentences: list[str], min_length: int) -> list[str]:
        """
        Combine short sentences to meet minimum length.

        Args:
            sentences: List of sentences
            min_length: Minimum characters per sentence

        Returns:
            List of combined sentences
        """
        combined = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) < min_length:
                if self.language in ['zh', 'zh-cn', 'zh-tw', 'ja', 'th']:
                    current += sentence
                else:
                    current = (current + " " + sentence).strip()
            else:
                if current:
                    combined.append(current)
                current = sentence

        if current:
            combined.append(current)

        logger.debug(f"Combined {len(sentences)} sentences into {len(combined)} sentences")
        return combined

    def chunk_generator(self,
                       text: str,
                       max_size: int,
                       overlap: int = 0,
                       **options) -> Generator[str, None, None]:
        """
        Memory-efficient generator version of chunk.

        Args:
            text: Text to chunk
            max_size: Maximum sentences per chunk
            overlap: Number of sentences to overlap between chunks
            **options: Additional options

        Yields:
            Individual text chunks
        """
        chunks = self.chunk(text, max_size, overlap, **options)
        yield from chunks

    def _prepare_chunk_records(
        self,
        text: str,
        max_size: int,
        overlap: int,
        **options,
    ) -> tuple[list[dict[str, Any]], list[tuple[str, int, int]]]:
        """Prepare chunk records with accurate offsets."""
        sentences_with_spans = self._split_sentences_with_spans(text)
        if not sentences_with_spans:
            return [], []

        logger.debug(f"Split text into {len(sentences_with_spans)} sentences")

        combined = sentences_with_spans
        if options.get('combine_short', False):
            try:
                min_length = int(options.get('min_sentence_length', 10))
            except (TypeError, ValueError):
                min_length = 10
            combined = self._combine_short_sentences_with_spans(sentences_with_spans, min_length)
        # Ensure we operate on a copy to avoid mutating shared state
        combined = list(combined)

        records: list[dict[str, Any]] = []
        step = max(1, max_size - overlap)
        no_space_languages = {'zh', 'zh-cn', 'zh-tw', 'ja', 'th'}

        for i in range(0, len(combined), step):
            window = combined[i:i + max_size]
            if not window:
                continue
            start_char = window[0][1]
            end_char = window[-1][2]
            with contextlib.suppress(_SENTENCE_NONCRITICAL_EXCEPTIONS):
                end_char = self._expand_end_to_grapheme_boundary(text, end_char, options=options)
            sentences_only = [item[0] for item in window]
            if self.language in no_space_languages:
                chunk_text = ''.join(sentences_only).strip()
            else:
                chunk_text = ' '.join(sentences_only).strip()
            records.append({
                'text': chunk_text,
                'start_char': start_char,
                'end_char': end_char,
                'sentence_count': len(window),
            })

        return records, combined

    def chunk_with_metadata(self,
                            text: str,
                            max_size: int,
                            overlap: int = 0,
                            **options) -> list[ChunkResult]:
        """Chunk text and include metadata with reliable offsets.

        By default, chunk text is sliced from the original source span to
        preserve spacing for multilingual inputs. Pass
        align_text_to_source=False to keep the normalized sentence-joined text.
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []

        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), setting to max_size - 1")
            overlap = max_size - 1

        records, combined = self._prepare_chunk_records(text, max_size, overlap, **options)
        if not records:
            return []

        try:
            min_length_opt = int(options.get('min_sentence_length', 10))
        except (TypeError, ValueError):
            min_length_opt = 10

        results: list[ChunkResult] = []
        align_text_to_source = bool(options.get("align_text_to_source", True))
        text_len = len(text)
        total = len(records)
        for idx, record in enumerate(records):
            chunk_text = record['text']
            start_char = record['start_char']
            end_char = record['end_char']
            sentence_count = record['sentence_count']
            if align_text_to_source:
                start_char = max(0, min(int(start_char), text_len))
                end_char = max(start_char, min(int(end_char), text_len))
                chunk_text = text[start_char:end_char]
            word_count = len(chunk_text.split()) if chunk_text else 0
            metadata = ChunkMetadata(
                index=idx,
                start_char=start_char,
                end_char=end_char,
                word_count=word_count,
                sentence_count=sentence_count,
                language=self.language,
                overlap_with_previous=overlap if idx > 0 else 0,
                overlap_with_next=overlap if idx < total - 1 else 0,
                method='sentences',
                options={
                    'combine_short': bool(options.get('combine_short', False)),
                    'min_sentence_length': min_length_opt,
                }
            )
            results.append(ChunkResult(text=chunk_text, metadata=metadata))

        logger.debug(f"Created {len(results)} chunks with metadata from {len(combined)} sentences")
        return results
