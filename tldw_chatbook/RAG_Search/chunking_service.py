"""
Simple chunking service wrapper for the simplified RAG implementation.

This provides a minimal interface to the existing chunking functionality
to satisfy the import requirements of the simplified RAG service.
"""

from typing import List, Dict, Any, Optional
import logging
from tldw_chatbook.Chunking.Chunk_Lib import Chunker, chunk_text

logger = logging.getLogger(__name__)


class ChunkingService:
    """
    Minimal chunking service that wraps the existing Chunk_Lib functionality.
    
    This is a temporary compatibility layer for the simplified RAG implementation.
    """
    
    def __init__(self):
        """Initialize the chunking service."""
        self.chunker = Chunker()
        logger.info("Initialized ChunkingService wrapper")
    
    def chunk_text(self, 
                   content: str, 
                   chunk_size: int = 400, 
                   chunk_overlap: int = 100,
                   method: str = "words") -> List[Dict[str, Any]]:
        """
        Chunk text using the specified method.
        
        Args:
            content: Text to chunk
            chunk_size: Target size of chunks
            chunk_overlap: Overlap between chunks
            method: Chunking method ("words", "sentences", "paragraphs", "tokens")
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        try:
            # Use the global chunk_text function from Chunk_Lib
            chunks = chunk_text(
                text=content,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                language=None,  # Auto-detect
                chunking_method=method
            )
            
            # Convert to expected format
            result_chunks = []
            for i, chunk in enumerate(chunks):
                # Handle both string chunks and dict chunks
                if isinstance(chunk, str):
                    chunk_text_content = chunk
                    start_char = i * (chunk_size - chunk_overlap)  # Approximate
                    end_char = start_char + len(chunk)
                else:
                    # Assume it's a dict with 'text' key
                    chunk_text_content = chunk.get('text', str(chunk))
                    start_char = chunk.get('start_char', i * (chunk_size - chunk_overlap))
                    end_char = chunk.get('end_char', start_char + len(chunk_text_content))
                
                result_chunks.append({
                    'text': chunk_text_content,
                    'start_char': start_char,
                    'end_char': end_char,
                    'word_count': len(chunk_text_content.split()),
                    'chunk_index': i
                })
            
            logger.debug(f"Chunked text into {len(result_chunks)} chunks using method '{method}'")
            return result_chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            # Return single chunk as fallback
            return [{
                'text': content,
                'start_char': 0,
                'end_char': len(content),
                'word_count': len(content.split()),
                'chunk_index': 0
            }]