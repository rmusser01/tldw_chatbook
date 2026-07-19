"""
Simple pipeline functions extracted from monolithic implementations.

Each function does one thing well and can be composed into pipelines.
No complex error handling - just let exceptions propagate.
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

from .pipeline_types import SearchResult
from .semantic_availability import (
    record_semantic_empty_index,
    record_semantic_ok,
    record_semantic_unavailable,
    resolve_semantic_rag_service,
    semantic_index_is_empty,
    SEMANTIC_REASON_SEARCH_ERROR,
    SEMANTIC_UNAVAILABLE_MESSAGES,
)


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


def _resolve_chacha_db(app: Any):
    """Resolve the ChaChaNotes DB for pipeline searches.

    Wiring decision (task-295): the app's LIVE instance
    (``TldwCli.chachanotes_db``, thread-local connections already open,
    schema already checked) is preferred; the ``db_config['chacha_db_path']``
    seam stays as a construction fallback for tests and probes. Nothing in
    production populates ``db_config``, and wiring it would add a second
    source of truth for a path the app already resolves at startup.

    Args:
        app: App-like object; either ``chachanotes_db`` (live instance) or
            ``db_config['chacha_db_path']`` (construction path) may be set.

    Returns:
        A CharactersRAGDB, or None when neither seam is available.
    """
    db = getattr(app, "chachanotes_db", None)
    if db is not None:
        return db
    db_config = getattr(app, "db_config", None)
    path = db_config.get("chacha_db_path") if isinstance(db_config, dict) else None
    if not path:
        return None
    from ..DB.ChaChaNotes_DB import CharactersRAGDB

    return CharactersRAGDB(path, client_id="rag_pipeline")


async def search_conversations_fts5(
    app: Any,
    query: str,
    limit: int = 10
) -> List[SearchResult]:
    """Search conversations database using FTS5.

    Args:
        app: App-like object exposing ``db_config['chacha_db_path']``.
        query: FTS search text.
        limit: Maximum number of conversation results to return.

    Returns:
        SearchResult entries in content-relevance order, each carrying a
        snippet built from the conversation's first messages.
    """
    db = _resolve_chacha_db(app)
    if db is None:
        return []
    
    logger.debug(f"Searching conversations for: {query}")
    conv_results = await asyncio.to_thread(
        db.search_conversations_by_content,
        search_query=query,
        limit=limit * 2
    )
    
    # task-260: one batched query for all matched conversations' context
    # messages (was one query per conversation), and text-only snippets
    # never fetch image BLOBs.
    messages_by_conv = await asyncio.to_thread(
        db.get_messages_for_conversations_batch,
        conversation_ids=[conv['id'] for conv in conv_results],
        limit_per_conversation=5,
        include_image_data=False,
    )
    
    results = []
    for conv in conv_results:
        # Build content from messages
        content_parts = []
        for msg in messages_by_conv.get(conv['id'], []):
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
    """Search notes using FTS5.

    task-295: this used to import a ``Notes_DB`` module that no longer
    exists anywhere (and called a ``user_id`` API shape the real store
    never had) -- notes live in ChaChaNotes, so it now routes through the
    same resolved DB as the conversations search.

    Args:
        app: App-like object; see ``_resolve_chacha_db`` for the seams.
        query: FTS search text (matched as a literal phrase).
        limit: Maximum number of note results to return.

    Returns:
        SearchResult entries for matching notes.
    """
    db = _resolve_chacha_db(app)
    if db is None:
        return []
    
    logger.debug(f"Searching notes for: {query}")
    
    note_results = await asyncio.to_thread(
        db.search_notes,
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
                'last_modified': note.get('last_modified')
            }
        ))
    
    return results


async def search_semantic(
    app: Any,
    query: str,
    sources: Dict[str, bool],
    limit: int = 10,
    diagnostics: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[SearchResult]:
    """Search using semantic embeddings, initializing the runtime lazily.

    A missing ``app._rag_service`` no longer means a silent empty result
    (task-250): the shared process-wide RAG service is initialized on first
    use (deps-gated, off the event loop), and when semantic retrieval is
    unavailable or the index is verifiably empty the WHY is recorded into
    ``diagnostics`` so calling surfaces can report it honestly.

    Args:
        app: App-like object carrying (or receiving) ``_rag_service``.
        query: Search query text.
        sources: Enabled sources mapping (currently unused by the vector leg).
        limit: Maximum number of results (``top_k`` for the RAG service).
        diagnostics: Optional dict that receives the semantic-leg state under
            ``SEMANTIC_DIAGNOSTICS_KEY`` (ok / unavailable+reason /
            empty_index).
        **kwargs: Extra kwargs forwarded verbatim to ``rag_service.search``
            (call sites whitelist these; see pipeline_builder_simple).

    Returns:
        SearchResult list; empty when semantic retrieval is unavailable (the
        reason then rides in ``diagnostics``) or genuinely matches nothing.
    """
    rag_service, unavailable_reason = await resolve_semantic_rag_service(app)
    if rag_service is None:
        logger.warning(
            "Semantic search unavailable ({}): {}".format(
                unavailable_reason,
                SEMANTIC_UNAVAILABLE_MESSAGES.get(unavailable_reason, ""),
            )
        )
        record_semantic_unavailable(diagnostics, unavailable_reason)
        return []

    logger.debug(f"Performing semantic search for: {query}")

    # Use the shared RAG service. A raising search must not leave the
    # semantic leg unaccounted for: on the direct semantic/retrieve path an
    # uncaught exception would surface raw error text (or, in gather-based
    # callers, vanish entirely) without ever recording WHY -- so record the
    # search_error state and degrade to the honest-empty outcome instead
    # (PR #692 review).
    try:
        rag_results = await rag_service.search(
            query=query,
            search_type="semantic",
            top_k=limit,
            include_citations=True,
            **kwargs
        )
    except Exception:
        logger.opt(exception=True).error(
            "Semantic search raised; recording search_error state."
        )
        record_semantic_unavailable(diagnostics, SEMANTIC_REASON_SEARCH_ERROR)
        return []

    # Convert to our SearchResult format
    results = []
    for result in rag_results:
        # Check if this result has citations
        if hasattr(result, 'citations') and result.citations:
            # Include citations in metadata
            metadata = result.metadata.copy() if result.metadata else {}
            metadata['_has_citations'] = True
            metadata['_citations'] = [c.to_dict() for c in result.citations]
            
            results.append(SearchResult(
                source=getattr(result, 'source', 'unknown'),
                id=result.id,
                title=getattr(result, 'title', result.document[:50]),
                content=result.document,
                score=result.score,
                metadata=metadata
            ))
        else:
            # Basic result without citations
            results.append(SearchResult(
                source=getattr(result, 'source', 'unknown'),
                id=result.id,
                title=getattr(result, 'title', result.document[:50]),
                content=getattr(result, 'content', result.document),
                score=result.score,
                metadata=result.metadata if hasattr(result, 'metadata') else {}
            ))

    # Distinguish "no matches" from "nothing indexed yet" (task-250): only a
    # trustworthy zero-document count reports the empty-index state.
    if not results and await semantic_index_is_empty(rag_service):
        record_semantic_empty_index(diagnostics)
    else:
        record_semantic_ok(diagnostics, len(results))

    return results


# ==============================================================================
# Processing Functions
# ==============================================================================

def deduplicate_results(results: List[SearchResult]) -> List[SearchResult]:
    """Remove duplicate results based on content similarity."""
    seen = {}
    
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
    functions: List[Dict[str, Any]],
    diagnostics: Optional[Dict[str, Any]] = None
) -> List[SearchResult]:
    """Execute multiple search functions in parallel."""
    tasks = []
    task_func_names = []

    for func_config in functions:
        func_name = func_config['function']
        config = func_config.get('config', {})

        if func_name == 'search_fts5':
            # Run FTS5 searches for each enabled source
            if sources.get('media'):
                tasks.append(search_media_fts5(app, query, **config))
                task_func_names.append('search_media_fts5')
            if sources.get('conversations'):
                tasks.append(search_conversations_fts5(app, query, **config))
                task_func_names.append('search_conversations_fts5')
            if sources.get('notes'):
                tasks.append(search_notes_fts5(app, query, **config))
                task_func_names.append('search_notes_fts5')
        elif func_name == 'search_semantic':
            # Forward only kwargs the RAG service accepts (same fix as the
            # pipeline builder's parallel/retrieve steps, tasks 256/250):
            # splatting the raw config duplicated top_k inside the service
            # call, and gather() below swallowed the resulting TypeError.
            semantic_kwargs = {
                key: config[key]
                for key in ('score_threshold', 'filter_metadata')
                if key in config
            }
            tasks.append(search_semantic(
                app, query, sources,
                limit=config.get('top_k', config.get('limit', 10)),
                diagnostics=diagnostics,
                **semantic_kwargs
            ))
            task_func_names.append('search_semantic')

    # Execute all searches in parallel
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine results
    all_results = []
    for func_name, results in zip(task_func_names, results_lists):
        if isinstance(results, Exception):
            logger.error(f"Search failed ({func_name}): {results}")
            if func_name == 'search_semantic':
                # The vector leg died mid-search: record it so callers can
                # say the results are keyword-only (task-250).
                record_semantic_unavailable(diagnostics, SEMANTIC_REASON_SEARCH_ERROR)
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