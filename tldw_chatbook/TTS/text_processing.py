# text_processing.py
# Description: Text processing utilities for TTS including chunking, normalization, and language detection
#
# Imports
import re
from typing import List, Tuple, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import unicodedata
from loguru import logger

#######################################################################################################################
#
# Text Processing Classes

@dataclass
class TextChunk:
    """Represents a chunk of text for TTS processing"""
    text: str
    token_count: int
    is_sentence_end: bool = True
    metadata: Optional[Dict[str, Any]] = None


class TextNormalizer:
    """Text normalization for TTS processing"""
    
    # Regex patterns for various text elements
    URL_PATTERN = re.compile(
        r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b'
        r'(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    )
    EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    PHONE_PATTERN = re.compile(r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}')
    
    # Unit patterns
    UNIT_PATTERNS = {
        # Storage units
        r'\b(\d+)\s*KB\b': r'\1 kilobytes',
        r'\b(\d+)\s*MB\b': r'\1 megabytes',
        r'\b(\d+)\s*GB\b': r'\1 gigabytes',
        r'\b(\d+)\s*TB\b': r'\1 terabytes',
        # Time units
        r'\b(\d+)\s*ms\b': r'\1 milliseconds',
        r'\b(\d+)\s*s\b(?![\w])': r'\1 seconds',
        r'\b(\d+)\s*min\b': r'\1 minutes',
        r'\b(\d+)\s*hr?s?\b': r'\1 hours',
        # Common abbreviations
        r'\betc\.': 'et cetera',
        r'\be\.g\.': 'for example',
        r'\bi\.e\.': 'that is',
        r'\bvs\.': 'versus',
        r'\bDr\.': 'Doctor',
        r'\bMr\.': 'Mister',
        r'\bMs\.': 'Miss',
        r'\bMrs\.': 'Missus',
    }
    
    def __init__(self, options: Optional[Dict[str, bool]] = None):
        """
        Initialize normalizer with options.
        
        Args:
            options: Dictionary of normalization options
        """
        self.options = options or {}
        self.normalize = self.options.get('normalize', True)
        self.unit_normalization = self.options.get('unit_normalization', False)
        self.url_normalization = self.options.get('url_normalization', True)
        self.email_normalization = self.options.get('email_normalization', True)
        self.phone_normalization = self.options.get('phone_normalization', True)
        self.optional_pluralization = self.options.get('optional_pluralization_normalization', True)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for TTS processing.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not self.normalize:
            return text
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Apply specific normalizations
        if self.url_normalization:
            text = self._normalize_urls(text)
        
        if self.email_normalization:
            text = self._normalize_emails(text)
        
        if self.phone_normalization:
            text = self._normalize_phone_numbers(text)
        
        if self.unit_normalization:
            text = self._normalize_units(text)
        
        if self.optional_pluralization:
            text = self._normalize_optional_plurals(text)
        
        # Clean up punctuation spacing
        text = self._fix_punctuation_spacing(text)
        
        return text
    
    def _normalize_urls(self, text: str) -> str:
        """Convert URLs to speakable format"""
        def replace_url(match):
            url = match.group(0)
            # Extract domain for simpler speech
            domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', url)
            if domain_match:
                domain = domain_match.group(1)
                # Remove common TLDs for cleaner speech
                domain = re.sub(r'\.(com|org|net|io|co|uk|edu|gov)$', '', domain)
                return f"website {domain.replace('.', ' dot ')}"
            return "website"
        
        return self.URL_PATTERN.sub(replace_url, text)
    
    def _normalize_emails(self, text: str) -> str:
        """Convert emails to speakable format"""
        def replace_email(match):
            email = match.group(0)
            local, domain = email.split('@')
            # Make email more speakable
            local = local.replace('.', ' dot ').replace('_', ' underscore ')
            domain = domain.replace('.', ' dot ')
            return f"{local} at {domain}"
        
        return self.EMAIL_PATTERN.sub(replace_email, text)
    
    def _normalize_phone_numbers(self, text: str) -> str:
        """Convert phone numbers to speakable format"""
        def replace_phone(match):
            phone = match.group(0)
            # Remove all non-digits
            digits = re.sub(r'\D', '', phone)
            # Format as speakable
            if len(digits) == 10:
                return f"{digits[0:3]} {digits[3:6]} {digits[6:]}"
            elif len(digits) == 11 and digits[0] == '1':
                return f"{digits[1:4]} {digits[4:7]} {digits[7:]}"
            else:
                # Just space out the digits
                return ' '.join(digits)
        
        return self.PHONE_PATTERN.sub(replace_phone, text)
    
    def _normalize_units(self, text: str) -> str:
        """Expand unit abbreviations"""
        for pattern, replacement in self.UNIT_PATTERNS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def _normalize_optional_plurals(self, text: str) -> str:
        """Replace (s) with s for better pronunciation"""
        return re.sub(r'\(s\)', 's', text)
    
    def _fix_punctuation_spacing(self, text: str) -> str:
        """Fix spacing around punctuation"""
        # Remove space before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        # Add space after punctuation if missing
        text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)
        # Fix multiple punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        return text


class TextChunker:
    """Text chunking for TTS with token limits"""
    
    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'[.!?]+[\s\n]+|[.!?]+$')
    
    # Approximate tokens per word (conservative estimate)
    TOKENS_PER_WORD = 1.3
    
    def __init__(self, max_tokens: int = 500):
        """
        Initialize text chunker.
        
        Args:
            max_tokens: Maximum tokens per chunk
        """
        self.max_tokens = max_tokens
        self.min_chunk_size = max(10, int(max_tokens * 0.1))  # At least 10% of max
    
    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks suitable for TTS.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        # Split into sentences first
        sentences = self._split_sentences(text)
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # If single sentence exceeds limit, split it
            if sentence_tokens > self.max_tokens:
                # Flush current chunk if any
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        token_count=current_tokens,
                        is_sentence_end=True
                    ))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
            
            # Check if adding sentence exceeds limit
            elif current_tokens + sentence_tokens > self.max_tokens:
                # Flush current chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        token_count=current_tokens,
                        is_sentence_end=True
                    ))
                
                # Start new chunk
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Flush final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                token_count=current_tokens,
                is_sentence_end=True
            ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Use regex to find sentence boundaries
        sentences = []
        last_end = 0
        
        for match in self.SENTENCE_ENDINGS.finditer(text):
            sentence = text[last_end:match.end()].strip()
            if sentence:
                sentences.append(sentence)
            last_end = match.end()
        
        # Add remaining text if any
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sentences.append(remaining)
        
        return sentences
    
    def _split_long_sentence(self, sentence: str) -> List[TextChunk]:
        """Split a long sentence that exceeds token limit"""
        chunks = []
        
        # Try splitting by commas first
        parts = sentence.split(',')
        if len(parts) > 1:
            return self._group_parts(parts, ',')
        
        # Try splitting by semicolons
        parts = sentence.split(';')
        if len(parts) > 1:
            return self._group_parts(parts, ';')
        
        # Try splitting by conjunctions
        conjunctions = [' and ', ' or ', ' but ', ' because ', ' although ', ' while ']
        for conj in conjunctions:
            if conj in sentence:
                parts = sentence.split(conj)
                if len(parts) > 1:
                    # Reconstruct with conjunction
                    reconstructed_parts = []
                    for i, part in enumerate(parts[:-1]):
                        reconstructed_parts.append(part + conj.rstrip())
                    reconstructed_parts.append(parts[-1])
                    return self._group_parts(reconstructed_parts, '')
        
        # Last resort: split by words
        words = sentence.split()
        words_per_chunk = int(self.max_tokens / self.TOKENS_PER_WORD)
        
        for i in range(0, len(words), words_per_chunk):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = ' '.join(chunk_words)
            chunks.append(TextChunk(
                text=chunk_text,
                token_count=self._estimate_tokens(chunk_text),
                is_sentence_end=(i + words_per_chunk >= len(words))
            ))
        
        return chunks
    
    def _group_parts(self, parts: List[str], separator: str) -> List[TextChunk]:
        """Group parts into chunks respecting token limits"""
        chunks = []
        current_parts = []
        current_tokens = 0
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            part_tokens = self._estimate_tokens(part)
            
            if current_tokens + part_tokens > self.max_tokens and current_parts:
                # Create chunk
                chunk_text = (separator + ' ').join(current_parts)
                if separator and not chunk_text.endswith(separator):
                    chunk_text += separator
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    token_count=current_tokens,
                    is_sentence_end=False
                ))
                
                current_parts = [part]
                current_tokens = part_tokens
            else:
                current_parts.append(part)
                current_tokens += part_tokens
        
        # Final chunk
        if current_parts:
            chunk_text = (separator + ' ').join(current_parts)
            chunks.append(TextChunk(
                text=chunk_text,
                token_count=current_tokens,
                is_sentence_end=True
            ))
        
        return chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation based on words
        # Could be replaced with actual tokenizer if available
        word_count = len(text.split())
        return int(word_count * self.TOKENS_PER_WORD)


def detect_language(text: str, voice_code: Optional[str] = None) -> str:
    """
    Detect language from text or voice code.
    
    Args:
        text: Input text
        voice_code: Optional voice code (e.g., 'af' for American Female)
        
    Returns:
        Language code (e.g., 'en', 'ja', 'zh')
    """
    # If voice code provided, use first letter as language hint
    if voice_code and len(voice_code) >= 1:
        lang_prefix = voice_code[0].lower()
        language_map = {
            'a': 'en',  # American English
            'b': 'en',  # British English
            'j': 'ja',  # Japanese
            'z': 'zh',  # Chinese (Mandarin)
            'e': 'es',  # Spanish
            'f': 'fr',  # French
            'h': 'hi',  # Hindi
            'i': 'it',  # Italian
            'p': 'pt',  # Portuguese
            'k': 'ko',  # Korean
            'r': 'ru',  # Russian
            'g': 'de',  # German
        }
        
        if lang_prefix in language_map:
            return language_map[lang_prefix]
    
    # Simple heuristic based on character sets
    # This is a basic implementation - could be enhanced with proper language detection
    
    # Check for CJK characters
    if re.search(r'[\u4e00-\u9fff]', text):  # Chinese characters
        return 'zh'
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):  # Japanese kana
        return 'ja'
    if re.search(r'[\uac00-\ud7af]', text):  # Korean hangul
        return 'ko'
    
    # Check for other scripts
    if re.search(r'[\u0400-\u04ff]', text):  # Cyrillic
        return 'ru'
    if re.search(r'[\u0900-\u097f]', text):  # Devanagari (Hindi)
        return 'hi'
    if re.search(r'[\u0600-\u06ff]', text):  # Arabic
        return 'ar'
    
    # Default to English
    return 'en'


async def process_text_stream(
    text: str,
    normalizer: TextNormalizer,
    chunker: TextChunker,
    language: Optional[str] = None
) -> AsyncGenerator[TextChunk, None]:
    """
    Process text and yield chunks for TTS streaming.
    
    Args:
        text: Input text
        normalizer: Text normalizer instance
        chunker: Text chunker instance
        language: Optional language code
        
    Yields:
        TextChunk objects ready for TTS
    """
    # Normalize entire text first
    normalized_text = normalizer.normalize_text(text)
    
    # Detect language if not provided
    if not language:
        language = detect_language(normalized_text)
    
    # Chunk the text
    chunks = chunker.chunk_text(normalized_text)
    
    # Yield chunks with metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            'language': language,
            'chunk_index': i,
            'total_chunks': len(chunks),
            'is_first': i == 0,
            'is_last': i == len(chunks) - 1
        }
        yield chunk

#
# End of text_processing.py
#######################################################################################################################