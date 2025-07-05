"""
Simple chunking service wrapper for the simplified RAG implementation.

This provides a minimal interface to the existing chunking functionality
to satisfy the import requirements of the simplified RAG service.
"""

from typing import List, Dict, Any, Optional
import logging
from tldw_chatbook.Chunking.Chunk_Lib import Chunker

logger = logging.getLogger(__name__)


class ChunkingError(Exception):
    """Base exception for chunking-related errors."""
    pass


class InvalidChunkingMethodError(ChunkingError):
    """Exception raised when an invalid chunking method is specified."""
    pass


class ChunkingService:
    """
    Minimal chunking service that wraps the existing Chunk_Lib functionality.
    
    This is a temporary compatibility layer for the simplified RAG implementation.
    """
    
    def __init__(self):
        """Initialize the chunking service."""
        # Initialize with default options
        self.default_options = {
            'method': 'words',
            'max_size': 400,
            'overlap': 200
        }
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
            # Create options for the chunker
            options = {
                'method': method,
                'max_size': chunk_size,
                'overlap': chunk_overlap
            }
            
            # Create a new Chunker instance with the specific options
            chunker = Chunker(options)
            
            # Use the main chunk_text method
            chunks = chunker.chunk_text(content, method=method)
            
            # Convert to expected format
            result_chunks = []
            current_position = 0
            
            for i, chunk_text in enumerate(chunks):
                # Handle empty chunks
                if not chunk_text:
                    continue
                    
                # Find the position of this chunk in the original text
                start_char = content.find(chunk_text, current_position)
                if start_char == -1:
                    start_char = current_position
                end_char = start_char + len(chunk_text)
                current_position = max(start_char + 1, start_char + len(chunk_text) - chunk_overlap * 5)  # Approximate char overlap
                
                result_chunks.append({
                    'text': chunk_text,
                    'start_char': start_char,
                    'end_char': end_char,
                    'word_count': len(chunk_text.split()),
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


def improved_chunking_process(text: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Wrapper function to provide compatibility with the server's chunking interface.
    
    Args:
        text: The text to chunk
        options: Dictionary containing chunking options:
            - method: The chunking method to use
            - max_size: Maximum size of each chunk
            - overlap: Overlap between chunks
            
    Returns:
        List of chunk dictionaries
        
    Raises:
        InvalidChunkingMethodError: If the chunking method is not supported
        ChunkingError: For other chunking-related errors
    """
    service = ChunkingService()
    
    # Validate method
    valid_methods = ['words', 'sentences', 'paragraphs', 'tokens', 'semantic']
    method = options.get('method', 'words')
    if method not in valid_methods:
        raise InvalidChunkingMethodError(f"Invalid chunking method: {method}. Valid methods are: {valid_methods}")
    
    try:
        return service.chunk_text(text, options)
    except Exception as e:
        raise ChunkingError(f"Error during chunking: {str(e)}") from e