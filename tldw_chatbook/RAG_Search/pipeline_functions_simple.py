"""
Simple pipeline functions extracted from monolithic implementations.

Each function does one thing well and can be composed into pipelines.
No complex error handling - just let exceptions propagate.
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

from .pipeline_types import SearchResult


# ==============================================================================
# Retrieval Functions
# ==============================================================================

async def search_media_fts5(
    app: Any,
    query: str,
    limit: int = 10,
    keyword_filter: Optional[List[str]] = None
) -> List[SearchResult]:
    """Search media database using FTS5."""
    if not app.media_db:
        return []
    
    logger.debug(f"Searching media for: {query}")
    
    # Search media database
    media_results = await asyncio.to_thread(
        app.media_db.search_media_db,
        search_query=query,
        search_fields=['title', 'content'],
        page=1,
        results_per_page=limit * 2,  # Get extra for filtering
        include_trash=False
    )
    
    # Extract results list
    if isinstance(media_results, tuple):
        media_items = media_results[0]
    else:
        media_items = media_results
    
    # Fetch keywords if needed
    keywords_map = {}
    if media_items and keyword_filter:
        media_ids = [item.get('id') for item in media_items if item.get('id')]
        if media_ids:
            keywords_map = await asyncio.to_thread(
                app.media_db.fetch_keywords_for_media_batch,
                media_ids
            )
    
    # Convert to SearchResult objects
    results = []
    for item in media_items:
        # Apply keyword filter if specified
        if keyword_filter:
            item_keywords = keywords_map.get(item.get('id'), [])
            if not any(kw.lower() in [k.lower() for k in item_keywords] 
                      for kw in keyword_filter):
                continue
        
        results.append(SearchResult(
            source='media',
            id=str(item.get('id', '')),
            title=item.get('title', 'Untitled'),
            content=item.get('content', ''),
            metadata={
                'type': item.get('type', 'unknown'),
                'author': item.get('author', 'Unknown'),
                'ingestion_date': item.get('ingestion_date', ''),
                'keywords': keywords_map.get(item.get('id'), [])
            }
        ))
    
    return results[:limit]


async def search_conversations_fts5(
    app: Any,
    query: str,
    limit: int = 10
) -> List[SearchResult]:
    """Search conversations database using FTS5."""
    if not hasattr(app, 'db_config') or not app.db_config.get('chacha_db_path'):
        return []
    
    logger.debug(f"Searching conversations for: {query}")
    
    from ...DB.ChaChaNotes_DB import CharactersRAGDB
    
    db = CharactersRAGDB(app.db_config['chacha_db_path'])
    conv_results = await asyncio.to_thread(
        db.search_conversations_by_content,
        search_query=query,
        limit=limit * 2
    )
    
    results = []
    for conv in conv_results:
        # Get some messages for context
        messages = await asyncio.to_thread(
            db.get_messages_for_conversation,
            conversation_id=conv['id'],
            limit=5
        )
        
        # Build content from messages
        content_parts = []
        for msg in messages:
            content_parts.append(f"{msg['sender']}: {msg['content']}")
        
        results.append(SearchResult(
            source='conversation',
            id=str(conv['id']),
            title=conv.get('title', 'Untitled Conversation'),
            content='\n'.join(content_parts),
            metadata={
                'character_id': conv.get('character_id'),
                'created_at': conv.get('created_at'),
                'updated_at': conv.get('updated_at')
            }
        ))
    
    return results[:limit]


async def search_notes_fts5(
    app: Any,
    query: str,
    limit: int = 10
) -> List[SearchResult]:
    """Search notes using FTS5."""
    if not hasattr(app, 'db_config') or not app.db_config.get('notes_db_path'):
        return []
    
    logger.debug(f"Searching notes for: {query}")
    
    from ...Notes.DB.Notes_DB import NotesDB
    
    db = NotesDB(app.db_config['notes_db_path'])
    
    # Get user ID if available
    user_id = getattr(app, 'notes_user_id', None)
    if not user_id:
        return []
    
    note_results = await asyncio.to_thread(
        db.search_notes,
        user_id=user_id,
        search_term=query,
        limit=limit
    )
    
    results = []
    for note in note_results:
        results.append(SearchResult(
            source='note',
            id=str(note['id']),
            title=note.get('title', 'Untitled Note'),
            content=note.get('content', ''),
            metadata={
                'created_at': note.get('created_at'),
                'updated_at': note.get('updated_at'),
                'tags': note.get('tags', [])
            }
        ))
    
    return results


async def search_semantic(
    app: Any,
    query: str,
    sources: Dict[str, bool],
    limit: int = 10,
    **kwargs
) -> List[SearchResult]:
    """Search using semantic embeddings."""
    if not hasattr(app, '_rag_service') or not app._rag_service:
        logger.warning("RAG service not initialized")
        return []
    
    logger.debug(f"Performing semantic search for: {query}")
    
    # Use the existing RAG service
    rag_results = await app._rag_service.search(
        query=query,
        search_type="semantic",
        top_k=limit,
        include_citations=True,
        **kwargs
    )
    
    # Convert to our SearchResult format
    results = []
    for result in rag_results:
        results.append(SearchResult(
            source=result.source,
            id=result.id,
            title=result.title,
            content=result.content,
            score=result.score,
            metadata=result.metadata
        ))
    
    return results


# ==============================================================================
# Processing Functions
# ==============================================================================

def deduplicate_results(results: List[SearchResult]) -> List[SearchResult]:
    """Remove duplicate results based on content similarity."""
    seen = {}
    deduped = []
    
    for result in results:
        # Use first 200 chars of content as key
        key = result.content[:200]
        
        if key not in seen or result.score > seen[key].score:
            seen[key] = result
    
    return list(seen.values())


def rerank_results(
    results: List[SearchResult],
    query: str,
    model: str = "flashrank",
    top_k: int = 10
) -> List[SearchResult]:
    """Rerank results using specified model."""
    if not results:
        return results
    
    if model == "flashrank":
        try:
            from flashrank import RerankRequest, Ranker
            
            # Initialize ranker
            ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
            
            # Prepare passages
            passages = []
            for result in results:
                text = f"{result.title}\n{result.content[:1000]}"
                passages.append({"text": text})
            
            # Rerank
            rerank_req = RerankRequest(query=query, passages=passages)
            ranked_results = ranker.rerank(rerank_req)
            
            # Reorder results
            reranked = []
            for ranked in ranked_results[:top_k]:
                idx = ranked.index
                if idx < len(results):
                    result = results[idx]
                    result.score = float(ranked.score)
                    reranked.append(result)
            
            return reranked
            
        except ImportError:
            logger.warning("FlashRank not available, returning original order")
            return results[:top_k]
    
    # Default: just sort by score and limit
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
    return sorted_results[:top_k]


def filter_by_score(results: List[SearchResult], min_score: float = 0.0) -> List[SearchResult]:
    """Filter results by minimum score."""
    return [r for r in results if r.score >= min_score]


# ==============================================================================
# Formatting Functions
# ==============================================================================

def format_as_context(
    results: List[SearchResult],
    max_length: int = 10000,
    include_citations: bool = True
) -> str:
    """Format results as LLM context."""
    if not results:
        return "No relevant context found."
    
    context_parts = []
    total_chars = 0
    
    for i, result in enumerate(results):
        # Format header
        if include_citations:
            header = f"[{result.source.upper()} - {result.title}]"
        else:
            header = ""
        
        # Calculate available space
        remaining = max_length - total_chars - len(header) - 10  # Buffer
        if remaining <= 0:
            break
        
        # Truncate content if needed
        content = result.content
        if len(content) > remaining:
            content = content[:remaining] + "..."
        
        # Build result text
        if header:
            result_text = f"{header}\n{content}"
        else:
            result_text = content
        
        context_parts.append(result_text)
        total_chars += len(result_text) + 10  # Account for separator
        
        if total_chars >= max_length:
            break
    
    return "\n---\n".join(context_parts)


# ==============================================================================
# Combination Functions
# ==============================================================================

async def parallel_search(
    app: Any,
    query: str,
    sources: Dict[str, bool],
    functions: List[Dict[str, Any]]
) -> List[SearchResult]:
    """Execute multiple search functions in parallel."""
    tasks = []
    
    for func_config in functions:
        func_name = func_config['function']
        config = func_config.get('config', {})
        
        if func_name == 'search_fts5':
            # Run FTS5 searches for each enabled source
            if sources.get('media'):
                tasks.append(search_media_fts5(app, query, **config))
            if sources.get('conversations'):
                tasks.append(search_conversations_fts5(app, query, **config))
            if sources.get('notes'):
                tasks.append(search_notes_fts5(app, query, **config))
        elif func_name == 'search_semantic':
            tasks.append(search_semantic(app, query, sources, **config))
    
    # Execute all searches in parallel
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results
    all_results = []
    for results in results_lists:
        if isinstance(results, Exception):
            logger.error(f"Search failed: {results}")
            continue
        all_results.extend(results)
    
    return all_results


def weighted_merge(
    results_lists: List[List[SearchResult]],
    weights: List[float]
) -> List[SearchResult]:
    """Merge multiple result lists with weighted scores."""
    if len(results_lists) != len(weights):
        raise ValueError("Number of result lists must match number of weights")
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        return []
    weights = [w / total_weight for w in weights]
    
    # Merge with weighted scores
    merged = {}
    
    for results, weight in zip(results_lists, weights):
        for result in results:
            key = f"{result.source}:{result.id}"
            
            if key in merged:
                # Update score with weighted average
                merged[key].score += result.score * weight
            else:
                # New result with weighted score
                result.score = result.score * weight
                merged[key] = result
    
    # Sort by final score
    final_results = list(merged.values())
    final_results.sort(key=lambda x: x.score, reverse=True)
    
    return final_results


# ==============================================================================
# Function Registry
# ==============================================================================

RETRIEVAL_FUNCTIONS = {
    'search_media_fts5': search_media_fts5,
    'search_conversations_fts5': search_conversations_fts5,
    'search_notes_fts5': search_notes_fts5,
    'search_semantic': search_semantic,
    'parallel_search': parallel_search,
}

PROCESSING_FUNCTIONS = {
    'deduplicate_results': deduplicate_results,
    'rerank_results': rerank_results,
    'filter_by_score': filter_by_score,
    'weighted_merge': weighted_merge,
}

FORMATTING_FUNCTIONS = {
    'format_as_context': format_as_context,
}