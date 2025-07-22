"""
Context assembler for RAG pipeline with parent document inclusion support.

This module handles the assembly of context from chunks and documents,
including intelligent parent document inclusion based on size constraints.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ParentInclusionStrategy(Enum):
    """Strategy for including parent documents."""
    NEVER = "never"
    ALWAYS = "always"
    SIZE_BASED = "size_based"


@dataclass
class ContextDocument:
    """Represents a document or chunk in the context."""
    id: str
    content: str
    metadata: Dict[str, Any]
    is_chunk: bool
    is_parent: bool
    size: int
    media_id: Optional[int] = None
    chunk_index: Optional[int] = None
    relevance_score: Optional[float] = None


class ContextAssembler:
    """
    Assembles context for RAG pipeline with intelligent parent document inclusion.
    
    Features:
    - Size-aware parent document inclusion
    - Prioritization of matched chunks
    - Context size management
    - Deduplication of content
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the context assembler.
        
        Args:
            config: RAG pipeline configuration containing:
                - include_parent_docs: bool
                - parent_size_threshold: int (characters)
                - parent_inclusion_strategy: str
                - max_context_size: int (characters)
        """
        self.include_parents = config.get("include_parent_docs", False)
        self.parent_threshold = config.get("parent_size_threshold", 5000)
        self.strategy = ParentInclusionStrategy(
            config.get("parent_inclusion_strategy", "size_based")
        )
        self.max_context = config.get("max_context_size", 16000)
        
        logger.info(f"Initialized ContextAssembler with strategy: {self.strategy.value}")
    
    def assemble_context(self,
                        chunks: List[Dict[str, Any]],
                        get_parent_func: Optional[Any] = None) -> List[ContextDocument]:
        """
        Assemble context from chunks with optional parent document inclusion.
        
        Args:
            chunks: List of chunk dictionaries from retrieval
            get_parent_func: Function to retrieve parent document by media_id
            
        Returns:
            List of ContextDocument objects ordered by priority
        """
        context = []
        used_size = 0
        included_media_ids = set()
        parent_candidates = []
        
        # First pass: Add all matched chunks
        for chunk in chunks:
            chunk_doc = self._create_context_doc(chunk, is_chunk=True)
            
            if used_size + chunk_doc.size <= self.max_context:
                context.append(chunk_doc)
                used_size += chunk_doc.size
                
                # Track media IDs for parent lookup
                if chunk_doc.media_id:
                    included_media_ids.add(chunk_doc.media_id)
            else:
                logger.debug(f"Skipping chunk due to size constraints: {chunk_doc.id}")
        
        # Second pass: Consider parent documents if enabled
        if self.include_parents and get_parent_func:
            parent_candidates = self._get_parent_candidates(
                included_media_ids,
                get_parent_func
            )
            
            # Sort parent candidates by size (smaller first)
            parent_candidates.sort(key=lambda p: p.size)
            
            # Try to add parents based on strategy
            for parent in parent_candidates:
                if self._should_include_parent(parent, used_size):
                    context.append(parent)
                    used_size += parent.size
                    logger.info(f"Included parent document {parent.id} (size: {parent.size})")
        
        # Sort final context by priority
        context = self._sort_context(context)
        
        logger.info(f"Assembled context with {len(context)} items, total size: {used_size}")
        return context
    
    def _create_context_doc(self,
                           data: Dict[str, Any],
                           is_chunk: bool = False,
                           is_parent: bool = False) -> ContextDocument:
        """Create a ContextDocument from data."""
        content = data.get("content", data.get("text", ""))
        
        return ContextDocument(
            id=str(data.get("id", "")),
            content=content,
            metadata=data.get("metadata", {}),
            is_chunk=is_chunk,
            is_parent=is_parent,
            size=len(content),
            media_id=data.get("media_id"),
            chunk_index=data.get("chunk_index"),
            relevance_score=data.get("relevance_score", data.get("score"))
        )
    
    def _get_parent_candidates(self,
                             media_ids: Set[int],
                             get_parent_func: Any) -> List[ContextDocument]:
        """Get parent document candidates for the given media IDs."""
        candidates = []
        
        for media_id in media_ids:
            try:
                parent_data = get_parent_func(media_id)
                if parent_data:
                    parent_doc = self._create_context_doc(
                        parent_data,
                        is_chunk=False,
                        is_parent=True
                    )
                    candidates.append(parent_doc)
            except Exception as e:
                logger.error(f"Error retrieving parent for media {media_id}: {e}")
        
        return candidates
    
    def _should_include_parent(self,
                             parent: ContextDocument,
                             current_size: int) -> bool:
        """Determine if a parent document should be included."""
        # Check if we have space
        if current_size + parent.size > self.max_context:
            return False
        
        # Apply strategy
        if self.strategy == ParentInclusionStrategy.ALWAYS:
            return True
        elif self.strategy == ParentInclusionStrategy.NEVER:
            return False
        elif self.strategy == ParentInclusionStrategy.SIZE_BASED:
            return parent.size <= self.parent_threshold
        
        return False
    
    def _sort_context(self, context: List[ContextDocument]) -> List[ContextDocument]:
        """
        Sort context documents by priority.
        
        Priority order:
        1. Matched chunks (by relevance score)
        2. Parent documents (by size)
        3. Other content
        """
        def sort_key(doc: ContextDocument) -> Tuple[int, float, int]:
            # First priority: chunks vs parents
            if doc.is_chunk:
                priority = 0
            elif doc.is_parent:
                priority = 1
            else:
                priority = 2
            
            # Second priority: relevance score (negative for descending)
            score = -(doc.relevance_score or 0)
            
            # Third priority: size (for parents, smaller first)
            size = doc.size if doc.is_parent else 0
            
            return (priority, score, size)
        
        return sorted(context, key=sort_key)
    
    def format_context(self,
                      context_docs: List[ContextDocument],
                      separator: str = "\n\n---\n\n") -> str:
        """
        Format context documents into a single string.
        
        Args:
            context_docs: List of context documents
            separator: Separator between documents
            
        Returns:
            Formatted context string
        """
        parts = []
        
        for doc in context_docs:
            # Add metadata header if available
            header_parts = []
            
            if doc.is_parent:
                header_parts.append("[PARENT DOCUMENT]")
            elif doc.is_chunk:
                header_parts.append(f"[CHUNK {doc.chunk_index}]")
            
            if doc.metadata.get("title"):
                header_parts.append(f"Title: {doc.metadata['title']}")
            
            if doc.metadata.get("source"):
                header_parts.append(f"Source: {doc.metadata['source']}")
            
            # Combine header and content
            if header_parts:
                header = " | ".join(header_parts)
                parts.append(f"{header}\n{doc.content}")
            else:
                parts.append(doc.content)
        
        return separator.join(parts)
    
    def get_context_stats(self, context_docs: List[ContextDocument]) -> Dict[str, Any]:
        """Get statistics about the assembled context."""
        stats = {
            "total_documents": len(context_docs),
            "total_size": sum(doc.size for doc in context_docs),
            "chunks": len([d for d in context_docs if d.is_chunk]),
            "parents": len([d for d in context_docs if d.is_parent]),
            "media_ids": list(set(d.media_id for d in context_docs if d.media_id)),
            "size_utilization": sum(doc.size for doc in context_docs) / self.max_context
        }
        
        return stats


def create_context_assembler(rag_config: Dict[str, Any]) -> ContextAssembler:
    """Factory function to create a context assembler."""
    return ContextAssembler(rag_config)