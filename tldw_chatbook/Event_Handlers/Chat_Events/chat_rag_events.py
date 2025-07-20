# chat_rag_events_simplified.py
# Description: Simplified event handlers for RAG functionality using pipeline system
#
# Imports
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple
import asyncio
from loguru import logger
import os

# Local Imports
from ...RAG_Search.pipeline_builder_simple import execute_pipeline, BUILTIN_PIPELINES
from ...RAG_Search.pipeline_loader import get_pipeline_loader

if TYPE_CHECKING:
    from ...app import TldwCli

# Configure logger with context
logger = logger.bind(module="chat_rag_events_simplified")

# Check if RAG dependencies are available
try:
    from ...RAG_Search.simplified import create_rag_service, create_config_for_collection
    RAG_SERVICES_AVAILABLE = True
except ImportError:
    logger.warning("RAG services not available")
    RAG_SERVICES_AVAILABLE = False


async def perform_plain_rag_search(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    top_k: int = 5,
    max_context_length: int = 10000,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank",
    keyword_filter_list: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a plain RAG search using the pipeline system.
    """
    logger.info(f"Performing plain RAG search for query: '{query}'")
    
    # Build pipeline configuration
    config = BUILTIN_PIPELINES['plain'].copy()
    config['parameters'] = {
        'top_k': top_k,
        'max_context_length': max_context_length,
        'keyword_filter': keyword_filter_list
    }
    
    # Adjust reranking step if needed
    if not enable_rerank:
        # Remove rerank step
        config['steps'] = [s for s in config['steps'] if s.get('function') != 'rerank_results']
    elif reranker_model != 'flashrank':
        # Update reranker model
        for step in config['steps']:
            if step.get('function') == 'rerank_results':
                step.setdefault('config', {})['model'] = reranker_model
    
    # Execute pipeline
    return await execute_pipeline(config, app, query, sources, top_k=top_k)


async def perform_full_rag_pipeline(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    top_k: int = 10,
    max_context_length: int = 10000,
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    chunk_type: str = "words",
    include_metadata: bool = True,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank",
    keyword_filter_list: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a full semantic RAG pipeline using the pipeline system.
    """
    logger.info(f"Performing semantic RAG search for query: '{query}'")
    
    # Build pipeline configuration
    config = BUILTIN_PIPELINES['semantic'].copy()
    config['parameters'] = {
        'top_k': top_k,
        'max_context_length': max_context_length,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'chunk_type': chunk_type,
        'include_metadata': include_metadata,
        'include_citations': include_metadata
    }
    
    # Adjust reranking step if needed
    if not enable_rerank:
        config['steps'] = [s for s in config['steps'] if s.get('function') != 'rerank_results']
    elif reranker_model != 'flashrank':
        for step in config['steps']:
            if step.get('function') == 'rerank_results':
                step.setdefault('config', {})['model'] = reranker_model
    
    # Execute pipeline
    return await execute_pipeline(config, app, query, sources, top_k=top_k)


async def perform_hybrid_rag_search(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    top_k: int = 10,
    max_context_length: int = 10000,
    enable_rerank: bool = True,
    reranker_model: str = "flashrank",
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    chunk_type: str = "words",
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    keyword_filter_list: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a hybrid RAG search using the pipeline system.
    """
    logger.info(f"Performing hybrid RAG search for query: '{query}'")
    
    # Normalize weights
    total_weight = bm25_weight + vector_weight
    if total_weight > 0:
        bm25_weight = bm25_weight / total_weight
        vector_weight = vector_weight / total_weight
    else:
        bm25_weight = 0.5
        vector_weight = 0.5
    
    # Build pipeline configuration
    config = BUILTIN_PIPELINES['hybrid'].copy()
    
    # Update weights based on sources and user preferences
    # FTS5 gets 3/4 of bm25_weight (split among media, conv, notes)
    # Semantic gets vector_weight
    fts_weight_per_source = bm25_weight / 3
    weights = [fts_weight_per_source, fts_weight_per_source, fts_weight_per_source, vector_weight]
    
    # Update config
    config['parameters'] = {
        'top_k': top_k,
        'max_context_length': max_context_length,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'chunk_type': chunk_type,
        'keyword_filter': keyword_filter_list
    }
    
    # Update weights in parallel step
    for step in config['steps']:
        if step.get('type') == 'parallel':
            step['config']['weights'] = weights
    
    # Adjust reranking step if needed
    if not enable_rerank:
        config['steps'] = [s for s in config['steps'] if s.get('function') != 'rerank_results']
    elif reranker_model != 'flashrank':
        for step in config['steps']:
            if step.get('function') == 'rerank_results':
                step.setdefault('config', {})['model'] = reranker_model
    
    # Execute pipeline
    return await execute_pipeline(config, app, query, sources, top_k=top_k)


async def perform_search_with_pipeline(
    app: "TldwCli",
    query: str,
    sources: Dict[str, bool],
    pipeline_id: str,
    **kwargs
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Perform a search using a specific pipeline ID.
    
    This allows using custom pipelines defined in TOML files.
    """
    logger.info(f"Performing search with pipeline '{pipeline_id}' for query: '{query}'")
    
    # Get pipeline configuration (from built-ins or TOML)
    from ...RAG_Search.pipeline_builder_simple import get_pipeline
    config = get_pipeline(pipeline_id)
    
    if not config:
        logger.error(f"Pipeline '{pipeline_id}' not found")
        return [], f"Pipeline '{pipeline_id}' not found"
    
    # Make a copy to avoid modifying the original
    config = config.copy()
    
    # Merge any pipeline-specific parameters with runtime parameters
    pipeline_params = config.get('parameters', {})
    merged_params = {**pipeline_params, **kwargs}
    
    # Execute pipeline with merged parameters
    return await execute_pipeline(config, app, query, sources, **merged_params)


# Helper function to format results (kept for compatibility)
def format_results_for_llm(
    results: List[Dict[str, Any]], 
    max_chars: int = 10000
) -> str:
    """Format search results as context for LLM."""
    from ...RAG_Search.pipeline_functions_simple import format_as_context
    from ...RAG_Search.pipeline_types import SearchResult
    
    # Convert dicts back to SearchResult objects
    search_results = []
    for r in results:
        search_results.append(SearchResult(
            source=r['source'],
            id=r['id'],
            title=r['title'],
            content=r['content'],
            score=r.get('score', 1.0),
            metadata=r.get('metadata', {})
        ))
    
    return format_as_context(search_results, max_chars)


# Initialize/get RAG service (kept for compatibility)
async def get_or_initialize_rag_service(app: "TldwCli") -> Optional[Any]:
    """Get or initialize the RAG service."""
    if not RAG_SERVICES_AVAILABLE:
        return None
    
    if hasattr(app, '_rag_service') and app._rag_service:
        return app._rag_service
    
    try:
        # Initialize RAG service with profile based on config
        # Default to hybrid_basic for semantic search in pipelines
        profile_name = "hybrid_basic"
        
        # Check if there's a profile preference in config
        if hasattr(app, 'config') and app.config:
            rag_config = app.config.get('rag', {})
            service_config = rag_config.get('service', {})
            profile_name = service_config.get('profile', 'hybrid_basic')
        
        # Create the service
        rag_service = create_rag_service(profile_name=profile_name)
        app._rag_service = rag_service
        return rag_service
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        return None


async def get_rag_context_for_chat(app: "TldwCli", user_message: str) -> Optional[str]:
    """
    Get RAG context for a chat message based on current settings.
    
    Returns the context string to be prepended to the user message, or None if RAG is disabled.
    """
    from textual.css.query import NoMatches
    
    # Check if RAG is enabled
    try:
        rag_enabled = app.query_one("#chat-rag-enable-checkbox").value
        plain_rag_enabled = app.query_one("#chat-rag-plain-enable-checkbox").value
    except NoMatches:
        logger.debug("RAG checkboxes not found, RAG disabled")
        return None
    
    if not rag_enabled and not plain_rag_enabled:
        logger.debug("RAG is disabled")
        return None
    
    # Get search mode from the new dropdown (if it exists)
    search_mode = None
    try:
        search_mode_widget = app.query_one("#chat-rag-search-mode")
        search_mode = search_mode_widget.value
        logger.info(f"RAG search mode from dropdown: {search_mode}")
        
        # If "none" is selected, determine mode from checkboxes or default
        if search_mode == "none":
            # Manual configuration mode - use existing logic
            search_mode = "plain" if plain_rag_enabled else "semantic"
            logger.info(f"Manual config mode: using {search_mode} based on checkboxes")
    except NoMatches:
        # Fallback to checkbox-based detection for backward compatibility
        logger.debug("Search mode dropdown not found, using checkbox-based detection")
        search_mode = "plain" if plain_rag_enabled else "semantic"
    
    # Get RAG settings
    try:
        sources = {
            'media': app.query_one("#chat-rag-search-media-checkbox").value,
            'conversations': app.query_one("#chat-rag-search-conversations-checkbox").value,
            'notes': app.query_one("#chat-rag-search-notes-checkbox").value
        }
        
        # Get keyword filter
        keyword_filter = app.query_one("#chat-rag-keyword-filter").value.strip()
        keyword_filter_list = [kw.strip() for kw in keyword_filter.split(',') if kw.strip()] if keyword_filter else []
        
        if keyword_filter_list:
            logger.info(f"Applying keyword filter: {keyword_filter_list}")
        
        top_k = int(app.query_one("#chat-rag-top-k").value or "5")
        max_context_length = int(app.query_one("#chat-rag-max-context-length").value or "10000")
        
        enable_rerank = app.query_one("#chat-rag-rerank-enable-checkbox").value
        reranker_model = app.query_one("#chat-rag-reranker-model").value
        
        chunk_size = int(app.query_one("#chat-rag-chunk-size").value or "400")
        chunk_overlap = int(app.query_one("#chat-rag-chunk-overlap").value or "100")
        chunk_type = app.query_one("#chat-rag-chunk-type").value or "words"
        include_metadata = app.query_one("#chat-rag-include-metadata-checkbox").value
        
    except Exception as e:
        logger.error(f"Error reading RAG settings: {e}")
        return None
    
    # Check if any sources are selected
    if not any(sources.values()):
        logger.warning("No RAG sources selected")
        app.notify("Please select at least one RAG source", severity="warning")
        return None
    
    # Initialize RAG service if needed for semantic search
    if search_mode == "semantic":
        rag_service = await get_or_initialize_rag_service(app)
        if not rag_service:
            logger.warning("RAG service not available, falling back to plain search")
            search_mode = "plain"
    
    # Perform the search
    try:
        logger.info(f"Performing {search_mode} RAG search for: '{user_message}'")
        
        if search_mode == "plain":
            results, context = await perform_plain_rag_search(
                app, user_message, sources,
                top_k=top_k,
                max_context_length=max_context_length,
                enable_rerank=enable_rerank,
                reranker_model=reranker_model,
                keyword_filter_list=keyword_filter_list
            )
        elif search_mode == "semantic" or search_mode == "full":
            results, context = await perform_full_rag_pipeline(
                app, user_message, sources,
                top_k=top_k,
                max_context_length=max_context_length,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_type=chunk_type,
                include_metadata=include_metadata,
                enable_rerank=enable_rerank,
                reranker_model=reranker_model,
                keyword_filter_list=keyword_filter_list
            )
        elif search_mode == "hybrid":
            results, context = await perform_hybrid_rag_search(
                app, user_message, sources,
                top_k=top_k,
                max_context_length=max_context_length,
                enable_rerank=enable_rerank,
                reranker_model=reranker_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_type=chunk_type,
                keyword_filter_list=keyword_filter_list
            )
        else:
            # Custom pipeline
            results, context = await perform_search_with_pipeline(
                app, user_message, sources, search_mode,
                top_k=top_k,
                max_context_length=max_context_length,
                enable_rerank=enable_rerank,
                reranker_model=reranker_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_type=chunk_type,
                keyword_filter_list=keyword_filter_list
            )
        
        if context and context.strip():
            logger.info(f"RAG context generated: {len(context)} characters")
            return context
        else:
            logger.warning("No relevant RAG context found")
            return None
            
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        app.notify(f"RAG search failed: {str(e)}", severity="error")
        return None