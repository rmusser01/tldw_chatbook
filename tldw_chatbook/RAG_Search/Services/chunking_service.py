# chunking_service.py
# Description: Service for document chunking and preprocessing
#
# Imports
from typing import List, Dict, Any, Optional, Tuple
import re
from loguru import logger
import hashlib
#
# Local Imports
from ...Utils.optional_deps import DEPENDENCIES_AVAILABLE

logger = logger.bind(module="chunking_service")

# Check dependencies
CHUNKER_AVAILABLE = DEPENDENCIES_AVAILABLE.get('chunker', False)

if CHUNKER_AVAILABLE:
    try:
        from langdetect import detect
        import nltk
        # Ensure punkt tokenizer is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    except ImportError:
        CHUNKER_AVAILABLE = False
        logger.warning("Chunking dependencies import failed")

class ChunkingService:
    """Service for chunking documents into smaller pieces for RAG"""
    
    def __init__(self):
        """Initialize the chunking service"""
        self.default_chunk_size = 400  # words
        self.default_overlap = 100  # words
        
    def chunk_text(
        self,
        text: str,
        chunk_size: int = None,
        overlap: int = None,
        method: str = "words"
    ) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces
        
        Args:
            text: The text to chunk
            chunk_size: Size of each chunk (default: 400 words)
            overlap: Overlap between chunks (default: 100 words)
            method: Chunking method ("words", "sentences", "paragraphs")
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
            
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap
        
        if method == "words":
            return self._chunk_by_words(text, chunk_size, overlap)
        elif method == "sentences":
            return self._chunk_by_sentences(text, chunk_size, overlap)
        elif method == "paragraphs":
            return self._chunk_by_paragraphs(text, chunk_size)
        else:
            logger.warning(f"Unknown chunking method: {method}, using words")
            return self._chunk_by_words(text, chunk_size, overlap)
    
    def _chunk_by_words(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Dict[str, Any]]:
        """Chunk text by word count"""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            # Get chunk words
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Calculate character positions
            start_char = len(" ".join(words[:i])) + (1 if i > 0 else 0)
            end_char = start_char + len(chunk_text)
            
            # Create chunk
            chunk = {
                "text": chunk_text,
                "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:8],
                "start_char": start_char,
                "end_char": end_char,
                "word_count": len(chunk_words),
                "chunk_index": len(chunks)
            }
            chunks.append(chunk)
            
            # Move forward with overlap
            i += chunk_size - overlap
            
        return chunks
    
    def _chunk_by_sentences(
        self,
        text: str,
        target_size: int,
        overlap_sentences: int = 1
    ) -> List[Dict[str, Any]]:
        """Chunk text by sentences"""
        if not CHUNKER_AVAILABLE:
            logger.warning("NLTK not available, falling back to word chunking")
            return self._chunk_by_words(text, target_size, 50)
            
        try:
            sentences = nltk.sent_tokenize(text)
            chunks = []
            
            i = 0
            while i < len(sentences):
                chunk_sentences = []
                word_count = 0
                
                # Add sentences until we reach target size
                j = i
                while j < len(sentences) and word_count < target_size:
                    sentence = sentences[j]
                    sentence_words = len(sentence.split())
                    
                    # Don't exceed target by too much
                    if word_count + sentence_words > target_size * 1.5 and chunk_sentences:
                        break
                        
                    chunk_sentences.append(sentence)
                    word_count += sentence_words
                    j += 1
                
                if chunk_sentences:
                    chunk_text = " ".join(chunk_sentences)
                    
                    # Calculate positions
                    start_pos = text.find(chunk_sentences[0])
                    end_pos = text.find(chunk_sentences[-1]) + len(chunk_sentences[-1])
                    
                    chunk = {
                        "text": chunk_text,
                        "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:8],
                        "start_char": start_pos,
                        "end_char": end_pos,
                        "word_count": word_count,
                        "sentence_count": len(chunk_sentences),
                        "chunk_index": len(chunks)
                    }
                    chunks.append(chunk)
                
                # Move forward with overlap
                i = max(i + 1, j - overlap_sentences)
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error in sentence chunking: {e}")
            return self._chunk_by_words(text, target_size, 50)
    
    def _chunk_by_paragraphs(
        self,
        text: str,
        target_size: int
    ) -> List[Dict[str, Any]]:
        """Chunk text by paragraphs"""
        # Split by double newlines or common paragraph patterns
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        
        current_chunk = []
        current_word_count = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_words = len(para.split())
            
            # If adding this paragraph would exceed target, create chunk
            if current_word_count + para_words > target_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk = {
                    "text": chunk_text,
                    "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:8],
                    "word_count": current_word_count,
                    "paragraph_count": len(current_chunk),
                    "chunk_index": len(chunks)
                }
                chunks.append(chunk)
                
                current_chunk = [para]
                current_word_count = para_words
            else:
                current_chunk.append(para)
                current_word_count += para_words
        
        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk = {
                "text": chunk_text,
                "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:8],
                "word_count": current_word_count,
                "paragraph_count": len(current_chunk),
                "chunk_index": len(chunks)
            }
            chunks.append(chunk)
        
        return chunks
    
    def chunk_document(
        self,
        document: Dict[str, Any],
        chunk_size: int = None,
        overlap: int = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document based on its type
        
        Args:
            document: Document dict with 'content', 'type', etc.
            chunk_size: Target chunk size
            overlap: Chunk overlap
            
        Returns:
            List of chunks with metadata
        """
        content = document.get('content', '')
        doc_type = document.get('type', 'text')
        doc_id = document.get('id', '')
        
        # Determine chunking method based on document type
        if doc_type in ['article', 'blog', 'documentation']:
            method = "paragraphs"
        elif doc_type in ['transcript', 'conversation']:
            method = "sentences"
        else:
            method = "words"
        
        # Chunk the content
        chunks = self.chunk_text(content, chunk_size, overlap, method)
        
        # Add document metadata to each chunk
        for chunk in chunks:
            chunk['document_id'] = doc_id
            chunk['document_type'] = doc_type
            chunk['document_title'] = document.get('title', 'Untitled')
            
        return chunks
    
    def detect_language(self, text: str) -> str:
        """Detect the language of text"""
        if not CHUNKER_AVAILABLE:
            return "unknown"
            
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return "unknown"