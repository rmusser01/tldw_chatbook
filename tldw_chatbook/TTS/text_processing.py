# text_processing.py
# Description: Text processing utilities for TTS including normalization, chunking, and language detection
#
# Imports
import re
from typing import List, Tuple, Optional, AsyncGenerator
from loguru import logger

# Third-party imports
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("nltk not available. Text chunking will use simple splitting.")

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Token counting will use approximation.")

#######################################################################################################################
#
# Functions:

class TextNormalizer:
    """Handles text normalization for TTS processing"""
    
    def __init__(self, options: Optional[dict] = None):
        """
        Initialize normalizer with options.
        
        Args:
            options: Dict with normalization options (normalize, unit_normalization, etc.)
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
        Apply all enabled normalizations to the text.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not self.normalize:
            return text
        
        # Basic normalization - always apply
        text = self._basic_normalization(text)
        
        # Apply specific normalizations based on options
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
        
        return text
    
    def _basic_normalization(self, text: str) -> str:
        """Basic text normalization"""
        # Replace special quotes and punctuation
        text = text.replace("'", "'").replace(""", '"').replace(""", '"')
        text = text.replace("–", "-").replace("—", "-")
        
        # Remove non-printable characters except newlines and spaces
        text = re.sub(r"[^\S \n]", " ", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", " ", text)
        
        return text.strip()
    
    def _normalize_urls(self, text: str) -> str:
        """Convert URLs to speakable format"""
        # Simple URL pattern
        url_pattern = r'https?://(?:www\.)?([a-zA-Z0-9-]+)\.([a-zA-Z]{2,})(?:/[^\s]*)?'
        
        def url_replacer(match):
            domain = match.group(1)
            tld = match.group(2)
            # Convert to speakable format
            domain = domain.replace('-', ' dash ')
            return f"{domain} dot {tld}"
        
        return re.sub(url_pattern, url_replacer, text)
    
    def _normalize_emails(self, text: str) -> str:
        """Convert emails to speakable format"""
        email_pattern = r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+)\.([a-zA-Z]{2,})'
        
        def email_replacer(match):
            user = match.group(1).replace('.', ' dot ').replace('_', ' underscore ')
            domain = match.group(2).replace('-', ' dash ')
            tld = match.group(3)
            return f"{user} at {domain} dot {tld}"
        
        return re.sub(email_pattern, email_replacer, text)
    
    def _normalize_phone_numbers(self, text: str) -> str:
        """Convert phone numbers to speakable format"""
        # US phone number pattern
        us_phone = r'\b(?:1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
        
        def phone_replacer(match):
            # Remove all non-digits
            digits = re.sub(r'\D', '', match.group())
            # Add spaces between digits for better pronunciation
            return ' '.join(digits)
        
        return re.sub(us_phone, phone_replacer, text)
    
    def _normalize_units(self, text: str) -> str:
        """Convert units to full words"""
        replacements = {
            r'\b(\d+)\s*KB\b': r'\1 kilobytes',
            r'\b(\d+)\s*MB\b': r'\1 megabytes',
            r'\b(\d+)\s*GB\b': r'\1 gigabytes',
            r'\b(\d+)\s*TB\b': r'\1 terabytes',
            r'\b(\d+)\s*km\b': r'\1 kilometers',
            r'\b(\d+)\s*m\b': r'\1 meters',
            r'\b(\d+)\s*cm\b': r'\1 centimeters',
            r'\b(\d+)\s*mm\b': r'\1 millimeters',
            r'\b(\d+)\s*kg\b': r'\1 kilograms',
            r'\b(\d+)\s*g\b': r'\1 grams',
            r'\b(\d+)\s*mg\b': r'\1 milligrams',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_optional_plurals(self, text: str) -> str:
        """Replace (s) with s for better pronunciation"""
        return re.sub(r'\(s\)', 's', text)


class TextChunker:
    """Handles text chunking for TTS processing"""
    
    def __init__(self, max_tokens: int = 500, tokenizer_name: Optional[str] = None):
        """
        Initialize text chunker.
        
        Args:
            max_tokens: Maximum tokens per chunk
            tokenizer_name: Name of tokenizer to use (if transformers available)
        """
        self.max_tokens = max_tokens
        self.tokenizer = None
        
        if TRANSFORMERS_AVAILABLE and tokenizer_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks suitable for TTS processing.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if NLTK_AVAILABLE:
            try:
                sentences = nltk.sent_tokenize(text)
                return self._chunk_by_sentences(sentences)
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}")
        
        # Fallback to simple splitting
        return self._simple_chunk(text)
    
    def _chunk_by_sentences(self, sentences: List[str]) -> List[str]:
        """Chunk text by sentences while respecting token limits"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = self._estimate_tokens(sentence)
            
            if current_length + sentence_length > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _simple_chunk(self, text: str) -> List[str]:
        """Simple text chunking by character count"""
        # Approximate 4 characters per token
        max_chars = self.max_tokens * 4
        
        # Split by periods first
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_with_period = sentence + ('. ' if not sentence.endswith('.') else ' ')
            
            if current_length + len(sentence_with_period) > max_chars:
                if current_chunk:
                    chunks.append(''.join(current_chunk).strip())
                current_chunk = [sentence_with_period]
                current_length = len(sentence_with_period)
            else:
                current_chunk.append(sentence_with_period)
                current_length += len(sentence_with_period)
        
        if current_chunk:
            chunks.append(''.join(current_chunk).strip())
        
        return chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            # Rough approximation: 4 characters = 1 token
            return len(text) // 4


async def smart_split(
    text: str,
    lang_code: Optional[str] = None,
    normalization_options: Optional[dict] = None,
    max_tokens: int = 500
) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
    """
    Smart text splitting with normalization for TTS.
    
    Args:
        text: Input text to process
        lang_code: Language code for language-specific processing
        normalization_options: Text normalization options
        max_tokens: Maximum tokens per chunk
        
    Yields:
        Tuples of (normalized_text, phonemes) - phonemes currently None
    """
    # Initialize processors
    normalizer = TextNormalizer(normalization_options)
    chunker = TextChunker(max_tokens)
    
    # Normalize text
    normalized_text = normalizer.normalize_text(text)
    
    # Chunk text
    chunks = chunker.chunk_text(normalized_text)
    
    # Yield chunks
    for chunk in chunks:
        # In the future, could add phoneme generation here
        yield chunk, None


def detect_language(text: str, voice: str) -> str:
    """
    Simple language detection based on voice name.
    
    Args:
        text: Input text (currently unused)
        voice: Voice identifier
        
    Returns:
        Language code (e.g., 'en-us', 'en-gb')
    """
    # Simple heuristic based on voice prefix
    # This should be replaced with proper language detection
    voice_lower = voice.lower()
    
    if voice_lower.startswith('a'):
        return 'en-us'
    elif voice_lower.startswith('b'):
        return 'en-gb'
    else:
        return 'en-us'  # Default


# Ensure NLTK data is downloaded if available
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)

#
# End of text_processing.py
#######################################################################################################################