# chat_rag_events.py
# Description: Event handlers for RAG functionality in the chat window
#
# Imports
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple
import asyncio
from loguru import logger
from pathlib import Path
import uuid
import os
#
# Local Imports
from ...DB.Client_Media_DB_v2 import MediaDatabase, DatabaseError
from ...DB.ChaChaNotes_DB import CharactersRAGDB
from ...Utils.optional_deps import DEPENDENCIES_AVAILABLE

# Try to import the new modular integration
USE_MODULAR_RAG = os.environ.get('USE_MODULAR_RAG', 'false').lower() in ('true', '1', 'yes')
if USE_MODULAR_RAG:
    try:
        from .chat_rag_integration import (
            perform_modular_rag_search,
            perform_modular_rag_pipeline,
            index_documents_modular
        )
        MODULAR_RAG_AVAILABLE = True
        logger.info("Using new modular RAG implementation")
    except ImportError as e:
        logger.warning(f"Modular RAG integration not available: {e}")
        MODULAR_RAG_AVAILABLE = False
else:
    MODULAR_RAG_AVAILABLE = False

# Conditional imports for RAG services
try:
    from tldw_chatbook.RAG_Search.simplified import (
        RAGService, create_config_for_collection, RAGConfig,
        SearchResult, SearchResultWithCitations
    )
    RAG_SERVICES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Simplified RAG services not available: {e}")
    RAG_SERVICES_AVAILABLE = False
    
    # Create placeholder classes
    class RAGService:
        def __init__(self, *args, **kwargs):
            raise ImportError("RAGService not available. Please install RAG dependencies: pip install tldw_chatbook[embeddings_rag]")
    
    def create_config_for_collection(*args, **kwargs):
        raise ImportError("RAG config not available. Please install RAG dependencies: pip install tldw_chatbook[embeddings_rag]")
    
    class RAGConfig:
        pass
    
    class SearchResult:
        pass
    
    class SearchResultWithCitations:
        pass

if TYPE_CHECKING:
    from ...app import TldwCli

# Configure logger with context
logger = logger.bind(module="chat_rag_events")

# Check if RAG dependencies are available
RAG_AVAILABLE = DEPENDENCIES_AVAILABLE.get('embeddings_rag', False) and RAG_SERVICES_AVAILABLE
RERANK_AVAILABLE = DEPENDENCIES_AVAILABLE.get('flashrank', False)
COHERE_AVAILABLE = DEPENDENCIES_AVAILABLE.get('cohere', False)

async def perform_plain_rag_search(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    top_k: int = 5,
    max_context_length: int = 10000,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank"
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a plain RAG search using BM25 (FTS5) search across selected sources.
    
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
    # Use modular implementation if available and enabled
    if MODULAR_RAG_AVAILABLE:
        logger.info(f"Using modular RAG search for query: '{query}'")
        return await perform_modular_rag_search(
            app, query, sources, top_k, max_context_length, 
            enable_rerank, reranker_model
        )
    
    logger.info(f"Performing plain RAG search for query: '{query}'")
    
    # Note: Caching is now handled internally by the simplified RAG service
    # when perform_full_rag_pipeline is called. For plain RAG search,
    # we don't have caching since it uses direct database queries.
    
    all_results = []
    
    # Search Media Items
    if sources.get('media', False) and app.media_db:
        logger.debug("Searching media items...")
        try:
            media_results = await asyncio.to_thread(
                app.media_db.search_media_db,
                search_query=query,
                search_fields=['title', 'content'],
                page=1,
                results_per_page=top_k * 2,  # Get more results for re-ranking
                include_trash=False
            )
            # Extract just the results list
            if isinstance(media_results, tuple):
                media_items = media_results[0]
            else:
                media_items = media_results
                
            for item in media_items:
                all_results.append({
                    'source': 'media',
                    'id': item.get('id'),
                    'title': item.get('title', 'Untitled'),
                    'content': item.get('content', ''),
                    'score': 1.0,  # Default score for BM25 results
                    'metadata': {
                        'type': item.get('type', 'unknown'),
                        'author': item.get('author', 'Unknown'),
                        'ingestion_date': item.get('ingestion_date', '')
                    }
                })
        except DatabaseError as e:
            logger.error(f"Error searching media DB: {e}")
        except Exception as e:
            logger.error(f"Unexpected error searching media: {e}", exc_info=True)
    
    # Search Conversations
    if sources.get('conversations', False) and app.chachanotes_db:
        logger.debug("Searching conversations...")
        try:
            conv_results = await asyncio.to_thread(
                app.chachanotes_db.search_conversations_by_content,
                search_query=query,
                limit=top_k * 2
            )
            for conv in conv_results:
                # Get messages for context
                messages = await asyncio.to_thread(
                    app.chachanotes_db.get_messages_for_conversation,
                    conversation_id=conv['id'],
                    limit=5  # Get last 5 messages for context
                )
                
                # Combine messages into content
                content_parts = []
                for msg in messages:
                    content_parts.append(f"{msg['sender']}: {msg['content']}")
                
                all_results.append({
                    'source': 'conversation',
                    'id': conv['id'],
                    'title': conv.get('title', 'Untitled Conversation'),
                    'content': "\n".join(content_parts),
                    'score': 1.0,
                    'metadata': {
                        'character_id': conv.get('character_id'),
                        'created_at': conv.get('created_at'),
                        'updated_at': conv.get('updated_at')
                    }
                })
        except Exception as e:
            logger.error(f"Error searching conversations: {e}", exc_info=True)
    
    # Search Notes
    if sources.get('notes', False) and app.notes_service:
        logger.debug("Searching notes...")
        try:
            note_results = await asyncio.to_thread(
                app.notes_service.search_notes,
                user_id=app.notes_user_id,
                search_term=query,
                limit=top_k * 2
            )
            for note in note_results:
                all_results.append({
                    'source': 'note',
                    'id': note['id'],
                    'title': note.get('title', 'Untitled Note'),
                    'content': note.get('content', ''),
                    'score': 1.0,
                    'metadata': {
                        'created_at': note.get('created_at'),
                        'updated_at': note.get('updated_at'),
                        'tags': note.get('tags', [])
                    }
                })
        except Exception as e:
            logger.error(f"Error searching notes: {e}", exc_info=True)
    
    # Apply re-ranking if enabled and available
    if enable_rerank:
        if reranker_model == "flashrank" and RERANK_AVAILABLE:
            logger.debug("Applying FlashRank re-ranking...")
            try:
                from flashrank import Ranker, RerankRequest
                ranker = Ranker()
                
                # Prepare documents for re-ranking
                passages = []
                for result in all_results:
                    # Combine title and content for re-ranking
                    text = f"{result['title']}\n{result['content'][:1000]}"  # Limit content for re-ranking
                    passages.append({"text": text})
                
                if passages:
                    rerank_req = RerankRequest(query=query, passages=passages)
                    ranked_results = ranker.rerank(rerank_req)
                    
                    # Update scores based on re-ranking
                    for i, ranked in enumerate(ranked_results):
                        if i < len(all_results):
                            all_results[ranked['index']]['score'] = ranked['score']
                    
                    # Sort by score
                    all_results.sort(key=lambda x: x['score'], reverse=True)
            except Exception as e:
                logger.error(f"Error during FlashRank re-ranking: {e}", exc_info=True)
                
        elif reranker_model == "cohere" and COHERE_AVAILABLE:
            logger.debug("Applying Cohere re-ranking...")
            try:
                import cohere
                import os
                
                # Get API key from environment or config
                api_key = os.environ.get('COHERE_API_KEY')
                if not api_key and hasattr(app, 'config'):
                    api_key = app.config.get('llm_settings', {}).get('cohere_api_key')
                
                if not api_key:
                    logger.warning("Cohere API key not found. Skipping re-ranking.")
                else:
                    co = cohere.Client(api_key)
                    
                    # Prepare documents for re-ranking
                    documents = []
                    for i, result in enumerate(all_results):
                        # Combine title and content for re-ranking
                        text = f"{result['title']}\n{result['content'][:1000]}"
                        documents.append(text)
                    
                    if documents:
                        # Cohere rerank API
                        response = co.rerank(
                            query=query,
                            documents=documents,
                            top_n=min(len(documents), top_k * 2),  # Get more results than needed
                            model='rerank-english-v2.0'  # Or 'rerank-multilingual-v2.0'
                        )
                        
                        # Create a new list with re-ranked results
                        reranked_results = []
                        for hit in response:
                            idx = hit.index
                            if idx < len(all_results):
                                result = all_results[idx].copy()
                                result['score'] = hit.relevance_score
                                reranked_results.append(result)
                        
                        # Replace with reranked results
                        all_results = reranked_results
                        
            except Exception as e:
                logger.error(f"Error during Cohere re-ranking: {e}", exc_info=True)
    
    # Limit to top_k results
    all_results = all_results[:top_k]
    
    # Build context string with character limit
    context_parts = []
    total_chars = 0
    
    for i, result in enumerate(all_results):
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
    
    logger.info(f"Plain RAG search completed. Found {len(all_results)} results, "
                f"context length: {len(context_string)} chars")
    
    # Note: No caching for plain RAG search as it uses direct database queries
    
    return all_results, context_string


async def perform_full_rag_pipeline(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    top_k: int = 5,
    max_context_length: int = 10000,
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    include_metadata: bool = True,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank"
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform full RAG pipeline with embeddings, vector search, and re-ranking.
    
    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    """
    # Use modular implementation if available and enabled
    if MODULAR_RAG_AVAILABLE:
        logger.info(f"Using modular RAG pipeline for query: '{query}'")
        # The modular pipeline returns a dict, so we need to adapt it
        result = await perform_modular_rag_pipeline(
            app, query, sources, 
            top_k=top_k,
            max_context_length=max_context_length,
            enable_rerank=enable_rerank,
            reranker_model=reranker_model
        )
        # Extract results and context from the dict response
        return result.get('sources', []), result.get('context', '')
    
    logger.info(f"Performing full RAG pipeline for query: '{query}'")
    
    # Check if embeddings are available
    if not DEPENDENCIES_AVAILABLE.get('embeddings_rag', False):
        logger.warning("Embeddings dependencies not available. Falling back to plain RAG.")
        return await perform_plain_rag_search(
            app, query, sources, top_k, max_context_length, 
            enable_rerank=True, reranker_model="flashrank"
        )
    
    # Initialize RAG service
    config = create_config_for_collection(
        "media",  # Default collection
        persist_dir=Path.home() / ".local" / "share" / "tldw_cli" / "chromadb"
    )
    # Configure for the search
    config.search.default_top_k = top_k
    config.search.include_citations = include_metadata
    config.chunking.chunk_size = chunk_size
    config.chunking.chunk_overlap = chunk_overlap
    
    rag_service = RAGService(config)
    
    # Perform semantic search across all sources
    # Build metadata filter based on enabled sources
    source_filters = []
    if sources.get('media', False):
        source_filters.append('media')
    if sources.get('conversations', False):
        source_filters.append('conversations')
    if sources.get('notes', False):
        source_filters.append('notes')
    
    # Search with the simplified RAG service
    search_results = await rag_service.search(
        query=query,
        top_k=top_k * 2,  # Get extra for re-ranking
        search_type="semantic",
        include_citations=include_metadata
    )
    
    # Convert search results to the expected format
    all_results = []
    for result in search_results:
        # Filter by source if needed
        if result.metadata.get('source') in source_filters:
            result_dict = {
                'id': result.id,
                'title': result.metadata.get('title', 'Untitled'),
                'content': result.document,
                'source': result.metadata.get('source', 'unknown'),
                'score': result.score,
                'metadata': result.metadata
            }
            
            # Add citations if available
            if hasattr(result, 'citations') and result.citations:
                result_dict['citations'] = [
                    {
                        'text': cit.text,
                        'document_title': cit.document_title,
                        'confidence': cit.confidence
                    }
                    for cit in result.citations
                ]
            
            all_results.append(result_dict)
    
    # Apply re-ranking if enabled and available
    if enable_rerank and len(all_results) > 0:
        if reranker_model == "flashrank" and RERANK_AVAILABLE:
            try:
                from flashrank import Ranker, RerankRequest
                ranker = Ranker()
                
                passages = []
                for result in all_results:
                    text = f"{result['title']}\n{result['content'][:1000]}"
                    passages.append({"text": text})
                
                if passages:
                    rerank_req = RerankRequest(query=query, passages=passages)
                    ranked_results = ranker.rerank(rerank_req)
                    
                    # Update scores based on re-ranking
                    for i, ranked in enumerate(ranked_results):
                        if i < len(all_results):
                            all_results[ranked['index']]['score'] = ranked['score']
                    
                    # Re-sort by new scores
                    all_results.sort(key=lambda x: x['score'], reverse=True)
            except Exception as e:
                logger.error(f"Error during FlashRank re-ranking: {e}")
                
        elif reranker_model == "cohere" and COHERE_AVAILABLE:
            try:
                import cohere
                import os
                
                # Get API key from environment or config
                api_key = os.environ.get('COHERE_API_KEY')
                if not api_key and hasattr(app, 'config'):
                    api_key = app.config.get('llm_settings', {}).get('cohere_api_key')
                
                if not api_key:
                    logger.warning("Cohere API key not found. Skipping re-ranking.")
                else:
                    co = cohere.Client(api_key)
                    
                    # Prepare documents for re-ranking
                    documents = []
                    for i, result in enumerate(all_results):
                        text = f"{result['title']}\n{result['content'][:1000]}"
                        documents.append(text)
                    
                    if documents:
                        # Cohere rerank API
                        response = co.rerank(
                            query=query,
                            documents=documents,
                            top_n=min(len(documents), top_k * 2),
                            model='rerank-english-v2.0'
                        )
                        
                        # Create a new list with re-ranked results
                        reranked_results = []
                        for hit in response:
                            idx = hit.index
                            if idx < len(all_results):
                                result = all_results[idx].copy()
                                result['score'] = hit.relevance_score
                                reranked_results.append(result)
                        
                        # Replace with reranked results
                        all_results = reranked_results
                        
            except Exception as e:
                logger.error(f"Error during Cohere re-ranking: {e}")
    
    # Limit to top_k results
    all_results = all_results[:top_k]
    
    # Build context string
    context_parts = []
    total_chars = 0
    
    for result in all_results:
        # Format result with metadata if requested
        if include_metadata:
            result_text = f"[{result['source'].upper()} - {result['title']}]\n"
            result_text += f"Score: {result['score']:.3f}\n"
            if result.get('metadata'):
                for key, value in result['metadata'].items():
                    if key not in ['embedding', 'chunk_id']:
                        result_text += f"{key}: {value}\n"
            result_text += "\n"
        else:
            result_text = f"[{result['source'].upper()} - {result['title']}]\n"
        
        # Add content
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
    
    logger.info(f"Full RAG pipeline completed. Found {len(all_results)} results, "
                f"context length: {len(context_string)} chars")
    
    return all_results, context_string


async def _search_media_with_embeddings(
    app: "TldwCli",
    embeddings_service: EmbeddingsService,
    query_embedding: List[float],
    n_results: int
) -> List[Dict[str, Any]]:
    """Search media items using embeddings"""
    results = []
    
    try:
        # Search in ChromaDB media collection
        collection_results = embeddings_service.search_collection(
            collection_name="media_chunks",
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if collection_results and collection_results.get('documents'):
            for i, doc in enumerate(collection_results['documents'][0]):
                metadata = collection_results['metadatas'][0][i]
                distance = collection_results['distances'][0][i]
                
                # Convert distance to similarity score (0-1)
                score = 1 / (1 + distance)
                
                results.append({
                    'source': 'media',
                    'id': metadata.get('media_id'),
                    'title': metadata.get('title', 'Untitled'),
                    'content': doc,
                    'score': score,
                    'metadata': {
                        'type': metadata.get('type', 'unknown'),
                        'author': metadata.get('author', 'Unknown'),
                        'chunk_index': metadata.get('chunk_index', 0)
                    }
                })
    except Exception as e:
        logger.error(f"Error searching media with embeddings: {e}")
    
    return results


async def _search_conversations_with_embeddings(
    app: "TldwCli",
    embeddings_service: EmbeddingsService,
    query_embedding: List[float],
    n_results: int
) -> List[Dict[str, Any]]:
    """Search conversations using embeddings"""
    results = []
    
    try:
        # Search in ChromaDB conversations collection
        collection_results = embeddings_service.search_collection(
            collection_name="conversation_chunks",
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if collection_results and collection_results.get('documents'):
            for i, doc in enumerate(collection_results['documents'][0]):
                metadata = collection_results['metadatas'][0][i]
                distance = collection_results['distances'][0][i]
                
                # Convert distance to similarity score
                score = 1 / (1 + distance)
                
                results.append({
                    'source': 'conversation',
                    'id': metadata.get('conversation_id'),
                    'title': metadata.get('title', 'Untitled Conversation'),
                    'content': doc,
                    'score': score,
                    'metadata': {
                        'character_id': metadata.get('character_id'),
                        'chunk_index': metadata.get('chunk_index', 0)
                    }
                })
    except Exception as e:
        logger.error(f"Error searching conversations with embeddings: {e}")
    
    return results


async def _search_notes_with_embeddings(
    app: "TldwCli",
    embeddings_service: EmbeddingsService,
    query_embedding: List[float],
    n_results: int
) -> List[Dict[str, Any]]:
    """Search notes using embeddings"""
    results = []
    
    try:
        # Search in ChromaDB notes collection
        collection_results = embeddings_service.search_collection(
            collection_name="notes_chunks",
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if collection_results and collection_results.get('documents'):
            for i, doc in enumerate(collection_results['documents'][0]):
                metadata = collection_results['metadatas'][0][i]
                distance = collection_results['distances'][0][i]
                
                # Convert distance to similarity score
                score = 1 / (1 + distance)
                
                results.append({
                    'source': 'note',
                    'id': metadata.get('note_id'),
                    'title': metadata.get('title', 'Untitled Note'),
                    'content': doc,
                    'score': score,
                    'metadata': {
                        'tags': metadata.get('tags', []),
                        'chunk_index': metadata.get('chunk_index', 0)
                    }
                })
    except Exception as e:
        logger.error(f"Error searching notes with embeddings: {e}")
    
    return results


async def perform_hybrid_rag_search(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    top_k: int = 5,
    max_context_length: int = 10000,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank",
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform hybrid RAG search combining BM25 and vector search results.
    
    This approach combines the strengths of:
    - BM25/FTS5: Good for exact keyword matches and rare terms
    - Vector search: Good for semantic similarity and understanding
    
    Args:
        app: The TldwCli app instance
        query: The search query
        sources: Dict of source types to search
        top_k: Number of top results to return
        max_context_length: Maximum total character count for context
        enable_rerank: Whether to apply re-ranking
        reranker_model: Which re-ranker to use
        chunk_size: Size of chunks for vector search
        chunk_overlap: Overlap between chunks
        bm25_weight: Weight for BM25 scores (0-1)
        vector_weight: Weight for vector search scores (0-1)
        
    Returns:
        Tuple of (results list, formatted context string)
    """
    logger.info(f"Performing hybrid RAG search for query: '{query}'")
    
    # Normalize weights
    total_weight = bm25_weight + vector_weight
    if total_weight > 0:
        bm25_weight = bm25_weight / total_weight
        vector_weight = vector_weight / total_weight
    else:
        bm25_weight = vector_weight = 0.5
    
    # Get BM25 results
    logger.debug("Getting BM25 results...")
    bm25_results, _ = await perform_plain_rag_search(
        app, query, sources, top_k * 3, max_context_length,
        enable_rerank=False  # We'll re-rank the combined results
    )
    
    # Get vector search results if available using simplified RAG
    vector_results = []
    if DEPENDENCIES_AVAILABLE.get('embeddings_rag', False) and RAG_SERVICES_AVAILABLE:
        logger.debug("Getting vector search results using simplified RAG...")
        try:
            # Initialize RAG service for hybrid search
            config = create_config_for_collection(
                "media",
                persist_dir=Path.home() / ".local" / "share" / "tldw_cli" / "chromadb"
            )
            config.search.default_top_k = top_k * 3
            config.chunking.chunk_size = chunk_size
            config.chunking.chunk_overlap = chunk_overlap
            
            rag_service = RAGService(config)
            
            # Perform semantic search
            search_results = await rag_service.search(
                query=query,
                top_k=top_k * 3,
                search_type="semantic",
                include_citations=False  # We don't need citations for hybrid
            )
            
            # Convert to expected format and filter by sources
            source_filters = [k for k, v in sources.items() if v]
            for result in search_results:
                if result.metadata.get('source') in source_filters:
                    vector_results.append({
                        'id': result.id,
                        'title': result.metadata.get('title', 'Untitled'),
                        'content': result.document,
                        'source': result.metadata.get('source', 'unknown'),
                        'score': result.score,
                        'metadata': result.metadata
                    })
            
            rag_service.close()
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            logger.warning("Vector search failed - using BM25 results only")
            vector_weight = 0
            bm25_weight = 1.0
    else:
        logger.warning("Embeddings not available, using BM25 only")
        vector_weight = 0
        bm25_weight = 1.0
    
    # Combine and deduplicate results
    combined_results = {}
    
    # Add BM25 results with weighted scores
    for result in bm25_results:
        key = (result['source'], result['id'], result.get('metadata', {}).get('chunk_index', 0))
        if key not in combined_results:
            result_copy = result.copy()
            result_copy['bm25_score'] = result['score']
            result_copy['vector_score'] = 0
            result_copy['score'] = result['score'] * bm25_weight
            combined_results[key] = result_copy
        else:
            combined_results[key]['bm25_score'] = result['score']
            combined_results[key]['score'] += result['score'] * bm25_weight
    
    # Add vector results with weighted scores
    for result in vector_results:
        key = (result['source'], result['id'], result.get('metadata', {}).get('chunk_index', 0))
        if key not in combined_results:
            result_copy = result.copy()
            result_copy['bm25_score'] = 0
            result_copy['vector_score'] = result['score']
            result_copy['score'] = result['score'] * vector_weight
            combined_results[key] = result_copy
        else:
            combined_results[key]['vector_score'] = result['score']
            # Re-calculate combined score
            combined_results[key]['score'] = (
                combined_results[key]['bm25_score'] * bm25_weight +
                result['score'] * vector_weight
            )
    
    # Convert back to list and sort by combined score
    all_results = list(combined_results.values())
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Apply re-ranking if enabled
    if enable_rerank and len(all_results) > 0:
        if reranker_model == "flashrank" and RERANK_AVAILABLE:
            logger.debug("Applying FlashRank re-ranking to hybrid results...")
            try:
                from flashrank import Ranker, RerankRequest
                ranker = Ranker()
                
                passages = []
                for result in all_results:
                    text = f"{result['title']}\n{result['content'][:1000]}"
                    passages.append({"text": text})
                
                if passages:
                    rerank_req = RerankRequest(query=query, passages=passages)
                    ranked_results = ranker.rerank(rerank_req)
                    
                    # Update scores based on re-ranking
                    for i, ranked in enumerate(ranked_results):
                        if i < len(all_results):
                            all_results[ranked['index']]['rerank_score'] = ranked['score']
                            all_results[ranked['index']]['score'] = ranked['score']
                    
                    # Re-sort by new scores
                    all_results.sort(key=lambda x: x['score'], reverse=True)
            except Exception as e:
                logger.error(f"Error during FlashRank re-ranking: {e}")
                
        elif reranker_model == "cohere" and COHERE_AVAILABLE:
            logger.debug("Applying Cohere re-ranking to hybrid results...")
            try:
                import cohere
                import os
                
                api_key = os.environ.get('COHERE_API_KEY')
                if not api_key and hasattr(app, 'config'):
                    api_key = app.config.get('llm_settings', {}).get('cohere_api_key')
                
                if not api_key:
                    logger.warning("Cohere API key not found. Skipping re-ranking.")
                else:
                    co = cohere.Client(api_key)
                    
                    documents = []
                    for result in all_results:
                        text = f"{result['title']}\n{result['content'][:1000]}"
                        documents.append(text)
                    
                    if documents:
                        response = co.rerank(
                            query=query,
                            documents=documents,
                            top_n=min(len(documents), top_k * 2),
                            model='rerank-english-v2.0'
                        )
                        
                        reranked_results = []
                        for hit in response:
                            idx = hit.index
                            if idx < len(all_results):
                                result = all_results[idx].copy()
                                result['rerank_score'] = hit.relevance_score
                                result['score'] = hit.relevance_score
                                reranked_results.append(result)
                        
                        all_results = reranked_results
                        
            except Exception as e:
                logger.error(f"Error during Cohere re-ranking: {e}")
    
    # Limit to top_k results
    all_results = all_results[:top_k]
    
    # Build context string with scores for debugging
    context_parts = []
    total_chars = 0
    
    for i, result in enumerate(all_results):
        # Format result with hybrid scores
        result_text = f"[{result['source'].upper()} - {result['title']}]\n"
        # Log debug information separately
        logger.debug(
            f"Result {i} scores - BM25: {result.get('bm25_score', 0):.3f}, "
            f"Vector: {result.get('vector_score', 0):.3f}, "
            f"Combined: {result['score']:.3f}" + 
            (f", Rerank: {result['rerank_score']:.3f}" if 'rerank_score' in result else "")
        )
        
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
    
    logger.info(f"Hybrid RAG search completed. Found {len(all_results)} results, "
                f"context length: {len(context_string)} chars")
    
    return all_results, context_string


async def get_rag_context_for_chat(app: "TldwCli", user_message: str) -> Optional[str]:
    """
    Get RAG context for a chat message based on current settings.
    
    Returns the context string to be prepended to the user message, or None if RAG is disabled.
    """
    # Check if RAG is enabled
    try:
        rag_enabled = app.query_one("#chat-rag-enable-checkbox").value
        plain_rag_enabled = app.query_one("#chat-rag-plain-enable-checkbox").value
    except:
        logger.debug("RAG checkboxes not found, RAG disabled")
        return None
    
    if not rag_enabled and not plain_rag_enabled:
        logger.debug("RAG is disabled")
        return None
    
    # Get RAG settings
    try:
        sources = {
            'media': app.query_one("#chat-rag-search-media-checkbox").value,
            'conversations': app.query_one("#chat-rag-search-conversations-checkbox").value,
            'notes': app.query_one("#chat-rag-search-notes-checkbox").value
        }
        
        top_k = int(app.query_one("#chat-rag-top-k").value or "5")
        max_context_length = int(app.query_one("#chat-rag-max-context-length").value or "10000")
        
        enable_rerank = app.query_one("#chat-rag-rerank-enable-checkbox").value
        reranker_model = app.query_one("#chat-rag-reranker-model").value
        
        chunk_size = int(app.query_one("#chat-rag-chunk-size").value or "400")
        chunk_overlap = int(app.query_one("#chat-rag-chunk-overlap").value or "100")
        include_metadata = app.query_one("#chat-rag-include-metadata-checkbox").value
        
    except Exception as e:
        logger.error(f"Error reading RAG settings: {e}")
        return None
    
    # Check if any sources are selected
    if not any(sources.values()):
        logger.warning("No RAG sources selected")
        app.notify("Please select at least one RAG source", severity="warning")
        return None
    
    # Perform RAG search
    try:
        if plain_rag_enabled:
            logger.info("Performing plain RAG search")
            results, context = await perform_plain_rag_search(
                app, user_message, sources, top_k, max_context_length,
                enable_rerank, reranker_model
            )
        else:
            logger.info("Performing full RAG pipeline")
            results, context = await perform_full_rag_pipeline(
                app, user_message, sources, top_k, max_context_length,
                chunk_size, chunk_overlap, include_metadata
            )
        
        if context:
            # Format context for inclusion in chat
            rag_context = (
                "### Context from RAG Search:\n"
                f"{context}\n"
                "### End of Context\n\n"
                "Based on the above context, please answer the following question:\n"
            )
            return rag_context
        else:
            logger.warning("No relevant context found")
            return None
            
    except Exception as e:
        logger.error(f"Error performing RAG search: {e}", exc_info=True)
        app.notify(f"RAG search error: {str(e)}", severity="error")
        return None