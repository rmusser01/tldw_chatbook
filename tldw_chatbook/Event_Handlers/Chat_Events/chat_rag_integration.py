# chat_rag_integration.py
# Description: Integration layer between chat events and the new modular RAG service
#
# This module provides functions that use the new RAG service while maintaining
# compatibility with the existing event handler interfaces.

from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple
import asyncio
from pathlib import Path
from loguru import logger

# Local imports
from ...Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Try to import the simplified RAG service
try:
    from tldw_chatbook.RAG_Search.simplified import (
        RAGService, create_config_for_collection, RAGConfig
    )
    RAG_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Simplified RAG service not available: {e}")
    RAG_SERVICE_AVAILABLE = False
    RAGService = None
    create_config_for_collection = None
    RAGConfig = None

if TYPE_CHECKING:
    from ...app import TldwCli

logger = logger.bind(module="chat_rag_integration")

# Global RAG service instance (initialized on first use)
_rag_service: Optional['RAGService'] = None


async def get_rag_service(app: "TldwCli") -> Optional['RAGService']:
    """
    Get or create the RAG service instance.
    
    Args:
        app: The TldwCli app instance
        
    Returns:
        RAGService instance or None if not available
    """
    global _rag_service
    
    if not RAG_SERVICE_AVAILABLE:
        return None
        
    if _rag_service is None:
        try:
            # Create configuration for the RAG service
            config = create_config_for_collection(
                "media",  # Default collection
                persist_dir=Path.home() / ".local" / "share" / "tldw_cli" / "chromadb"
            )
            
            # Create service with simplified interface
            _rag_service = RAGService(config)
            
            logger.info("Simplified RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            _rag_service = None
            
    return _rag_service


async def perform_modular_rag_search(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    top_k: int = 5,
    max_context_length: int = 10000,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank"
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform RAG search using the new modular service.
    
    This function provides the same interface as perform_plain_rag_search
    but uses the new modular RAG service underneath.
    
    Args:
        app: The TldwCli app instance
        query: The search query
        sources: Dict of source types to search (media, conversations, notes)
        top_k: Number of top results to return
        max_context_length: Maximum total character count for context
        enable_rerank: Whether to apply re-ranking
        reranker_model: Which re-ranker to use
        
    Returns:
        Tuple of (results list, formatted context string)
    """
    # Get the RAG service
    rag_service = await get_rag_service(app)
    
    if not rag_service:
        # Fallback to the existing implementation
        logger.warning("RAG service not available, falling back to direct implementation")
        from .chat_rag_events import perform_plain_rag_search
        return await perform_plain_rag_search(
            app, query, sources, top_k, max_context_length,
            enable_rerank, reranker_model
        )
    
    try:
        # Convert sources dict to list of source names
        source_list = []
        if sources.get('media', False):
            source_list.append("MEDIA_DB")
        if sources.get('conversations', False):
            source_list.append("CHAT_HISTORY")
        if sources.get('notes', False):
            source_list.append("NOTES")
        
        # Update config for this request
        rag_service.config.retriever.fts_top_k = top_k
        rag_service.config.processor.max_context_length = max_context_length
        rag_service.config.processor.enable_reranking = enable_rerank
        if reranker_model == "cohere":
            rag_service.config.processor.reranker_provider = "cohere"
        else:
            rag_service.config.processor.reranker_provider = "flashrank"
        
        # Perform search
        results = await rag_service.search(
            query=query,
            sources=source_list,
            media_top_k=top_k,
            chat_top_k=top_k,
            notes_top_k=top_k
        )
        
        # Format results to match expected structure
        formatted_results = []
        for doc in results:
            formatted_results.append({
                'id': doc['id'],
                'title': doc['metadata'].get('title', 'Untitled'),
                'content': doc['content'],
                'source': doc['source'].lower().replace('_db', '').replace('_history', 's'),
                'score': doc['score'],
                'metadata': doc['metadata']
            })
        
        # Build context string
        context_parts = []
        total_chars = 0
        
        for result in formatted_results[:top_k]:
            # Format result
            result_text = f"[{result['source'].upper()} - {result['title']}]\n"
            
            # Add content with character limit check
            remaining_chars = max_context_length - total_chars - len(result_text)
            if remaining_chars <= 0:
                break
                
            content_preview = result['content'][:remaining_chars]
            result_text += content_preview
            
            if len(result['content']) > remaining_chars:
                result_text += "...\n"
            else:
                result_text += "\n"
            
            context_parts.append(result_text)
            total_chars += len(result_text)
            
            if total_chars >= max_context_length:
                break
        
        context_string = "\n---\n".join(context_parts)
        
        logger.info(f"Modular RAG search completed. Found {len(formatted_results)} results")
        return formatted_results, context_string
        
    except Exception as e:
        logger.error(f"Error in modular RAG search: {e}", exc_info=True)
        # Fallback to existing implementation
        from .chat_rag_events import perform_plain_rag_search
        return await perform_plain_rag_search(
            app, query, sources, top_k, max_context_length,
            enable_rerank, reranker_model
        )


async def perform_modular_rag_pipeline(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    top_k: int = 5,
    max_context_length: int = 10000,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank"
) -> Dict[str, Any]:
    """
    Perform full RAG pipeline using the new modular service.
    
    This includes search, processing, and answer generation.
    
    Args:
        app: The TldwCli app instance
        query: The user's question
        sources: Dict of source types to search
        system_prompt: Optional system prompt for generation
        temperature: Generation temperature
        max_tokens: Maximum tokens for generation
        top_k: Number of top results to consider
        max_context_length: Maximum context length
        enable_rerank: Whether to apply re-ranking
        reranker_model: Which re-ranker to use
        
    Returns:
        Dictionary with answer, sources, and metadata
    """
    # Get the RAG service
    rag_service = await get_rag_service(app)
    
    if not rag_service:
        # Fallback to search-only mode
        logger.warning("RAG service not available, returning search results only")
        results, context = await perform_modular_rag_search(
            app, query, sources, top_k, max_context_length,
            enable_rerank, reranker_model
        )
        return {
            "answer": f"Found {len(results)} relevant results. Here's the context:\n\n{context}",
            "sources": results,
            "context": context,
            "metadata": {"mode": "search_only"}
        }
    
    try:
        # Convert sources dict to list
        source_list = []
        if sources.get('media', False):
            source_list.append("MEDIA_DB")
        if sources.get('conversations', False):
            source_list.append("CHAT_HISTORY")
        if sources.get('notes', False):
            source_list.append("NOTES")
        
        # Update config
        rag_service.config.retriever.fts_top_k = top_k
        rag_service.config.processor.max_context_length = max_context_length
        rag_service.config.processor.enable_reranking = enable_rerank
        rag_service.config.generator.temperature = temperature
        rag_service.config.generator.max_tokens = max_tokens
        if system_prompt:
            rag_service.config.generator.system_prompt = system_prompt
        
        # Generate answer
        result = await rag_service.generate_answer(
            query=query,
            sources=source_list,
            stream=False
        )
        
        # Format sources
        formatted_sources = []
        for source in result['sources']:
            formatted_sources.append({
                'id': source['id'],
                'title': source['title'],
                'source': source['source'].lower().replace('_db', '').replace('_history', 's'),
                'score': source['score']
            })
        
        return {
            "answer": result['answer'],
            "sources": formatted_sources,
            "context": result.get('context_preview', ''),
            "metadata": result['metadata']
        }
        
    except Exception as e:
        logger.error(f"Error in modular RAG pipeline: {e}", exc_info=True)
        # Fallback to search-only
        results, context = await perform_modular_rag_search(
            app, query, sources, top_k, max_context_length,
            enable_rerank, reranker_model
        )
        return {
            "answer": f"Error generating answer. Found {len(results)} relevant results:\n\n{context}",
            "sources": results,
            "context": context,
            "metadata": {"mode": "search_only", "error": str(e)}
        }


async def index_documents_modular(
    app: "TldwCli",
    source_type: str,
    documents: List[Dict[str, Any]],
    batch_size: int = 32
) -> bool:
    """
    Index documents using the new modular service.
    
    Args:
        app: The TldwCli app instance
        source_type: Type of source (MEDIA_DB, CHAT_HISTORY, NOTES)
        documents: List of documents to index
        batch_size: Batch size for indexing
        
    Returns:
        True if successful, False otherwise
    """
    rag_service = await get_rag_service(app)
    
    if not rag_service:
        logger.warning("RAG service not available for indexing")
        return False
    
    try:
        await rag_service.embed_documents(source_type, documents)
        logger.info(f"Successfully indexed {len(documents)} documents for {source_type}")
        return True
    except Exception as e:
        logger.error(f"Error indexing documents: {e}", exc_info=True)
        return False


# Migration helpers - these can be used to gradually replace the old functions
async def migrate_to_modular_search(app: "TldwCli", *args, **kwargs):
    """
    Migration wrapper that logs usage of old function and calls new one.
    """
    logger.info("Migrating from perform_plain_rag_search to modular implementation")
    return await perform_modular_rag_search(app, *args, **kwargs)


async def migrate_to_modular_pipeline(app: "TldwCli", *args, **kwargs):
    """
    Migration wrapper for full pipeline.
    """
    logger.info("Migrating from perform_full_rag_pipeline to modular implementation")
    # Extract relevant kwargs
    query = kwargs.get('query', args[0] if args else '')
    sources = kwargs.get('sources', args[1] if len(args) > 1 else {})
    
    return await perform_modular_rag_pipeline(
        app,
        query=query,
        sources=sources,
        **kwargs
    )