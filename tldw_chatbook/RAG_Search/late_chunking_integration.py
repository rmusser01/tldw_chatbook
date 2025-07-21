"""
Example integration of late chunking and parent document inclusion in RAG pipeline.

This module demonstrates how to integrate the late chunking service and context
assembler into existing RAG pipeline functions.
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

from .late_chunking_service import LateChunkingService, ChunkingConfig
from .context_assembler import ContextAssembler, ContextDocument
from .pipeline_types import SearchResult


async def enhanced_rag_search_with_late_chunking(
    app: Any,
    query: str,
    search_type: str = "hybrid",
    top_k: int = 10,
    rag_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Enhanced RAG search with late chunking and parent document inclusion.
    
    Args:
        app: Application instance with database connections
        query: Search query
        search_type: Type of search ("keyword", "semantic", "hybrid")
        top_k: Number of results to return
        rag_config: RAG configuration with parent inclusion settings
        
    Returns:
        List of search results with context
    """
    # Initialize services
    late_chunker = LateChunkingService(
        db_path=app.db_config.get('media_db_path'),
        cache_size=rag_config.get('late_chunking_cache_size', 100)
    )
    
    context_assembler = ContextAssembler(rag_config or {})
    
    # Step 1: Perform initial search to get matching documents
    logger.info(f"Performing {search_type} search for: {query}")
    
    # Get matching media documents
    from .pipeline_functions_simple import search_media_fts5
    media_results = await search_media_fts5(app, query, limit=top_k * 2)
    
    # Step 2: Perform late chunking on matched documents
    chunked_results = []
    default_chunking_config = ChunkingConfig(
        chunk_size=rag_config.get('chunk_size', 400),
        chunk_overlap=rag_config.get('chunk_overlap', 100),
        method=rag_config.get('chunking_method', 'words')
    )
    
    for result in media_results:
        # Get document-specific chunking config
        doc_config = late_chunker.get_document_config(result.id)
        
        # Perform late chunking
        chunks = late_chunker.get_chunks_for_document(
            media_id=result.id,
            content=result.content,
            doc_config=doc_config,
            default_config=default_chunking_config
        )
        
        # Add chunks to results
        for chunk in chunks:
            chunk_result = {
                'id': f"{result.id}_chunk_{chunk['chunk_index']}",
                'media_id': result.id,
                'content': chunk['text'],
                'text': chunk['text'],  # For compatibility
                'metadata': {
                    **result.metadata,
                    **chunk.get('metadata', {}),
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'chunking_method': chunk.get('chunking_method'),
                    'is_chunk': True
                },
                'relevance_score': result.score
            }
            chunked_results.append(chunk_result)
    
    # Step 3: Assemble context with parent document inclusion
    logger.info(f"Assembling context with parent inclusion strategy: {rag_config.get('parent_inclusion_strategy', 'never')}")
    
    # Function to retrieve parent documents
    async def get_parent_document(media_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve full parent document by media ID."""
        try:
            # Get from database
            from ..DB.Client_Media_DB_v2 import Database
            db = Database(app.db_config['media_db_path'])
            conn = db.get_connection()
            
            cursor = conn.execute(
                "SELECT id, title, content, author, ingestion_date FROM Media WHERE id = ?",
                (media_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': f"parent_{row['id']}",
                    'media_id': row['id'],
                    'content': row['content'],
                    'text': row['content'],
                    'metadata': {
                        'title': row['title'],
                        'author': row['author'],
                        'ingestion_date': row['ingestion_date'],
                        'is_parent': True
                    }
                }
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving parent document {media_id}: {e}")
            return None
    
    # Assemble final context
    context_docs = context_assembler.assemble_context(
        chunks=chunked_results[:top_k],  # Limit to top_k chunks
        get_parent_func=lambda media_id: asyncio.create_task(get_parent_document(media_id))
    )
    
    # Step 4: Format results
    final_results = []
    for doc in context_docs:
        result = {
            'id': doc.id,
            'content': doc.content,
            'metadata': doc.metadata,
            'score': doc.relevance_score or 0,
            'is_chunk': doc.is_chunk,
            'is_parent': doc.is_parent
        }
        final_results.append(result)
    
    # Log statistics
    stats = context_assembler.get_context_stats(context_docs)
    logger.info(f"Context assembly stats: {stats}")
    
    return final_results


def integrate_late_chunking_into_pipeline(pipeline_functions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate late chunking into existing pipeline functions.
    
    This function modifies pipeline functions to use late chunking when enabled.
    
    Args:
        pipeline_functions: Dictionary of pipeline functions
        
    Returns:
        Modified pipeline functions with late chunking support
    """
    original_search = pipeline_functions.get('search_and_retrieve')
    
    async def search_with_late_chunking(app, query, config):
        """Enhanced search function with late chunking support."""
        # Check if late chunking is enabled
        if config.get('chunking', {}).get('enable_late_chunking', False):
            logger.info("Using late chunking for search")
            
            # Prepare config for enhanced search
            rag_config = {
                'chunk_size': config['chunking'].get('chunk_size', 400),
                'chunk_overlap': config['chunking'].get('chunk_overlap', 100),
                'chunking_method': config['chunking'].get('chunking_method', 'words'),
                'late_chunking_cache_size': config['chunking'].get('late_chunking_cache_size', 100),
                'include_parent_docs': config['search'].get('include_parent_docs', False),
                'parent_size_threshold': config['search'].get('parent_size_threshold', 5000),
                'parent_inclusion_strategy': config['search'].get('parent_inclusion_strategy', 'size_based'),
                'max_context_size': config['search'].get('max_context_size', 16000)
            }
            
            return await enhanced_rag_search_with_late_chunking(
                app=app,
                query=query,
                search_type=config['search'].get('default_search_mode', 'hybrid'),
                top_k=config['search'].get('default_top_k', 10),
                rag_config=rag_config
            )
        else:
            # Use original search function
            return await original_search(app, query, config)
    
    # Replace search function
    pipeline_functions['search_and_retrieve'] = search_with_late_chunking
    
    return pipeline_functions


# Example usage in a RAG pipeline
async def example_rag_pipeline_with_features():
    """Example of using the enhanced RAG pipeline with all features."""
    
    # Configuration with all features enabled
    config = {
        'chunking': {
            'enable_late_chunking': True,
            'chunk_size': 400,
            'chunk_overlap': 100,
            'chunking_method': 'hierarchical',
            'late_chunking_cache_size': 200
        },
        'search': {
            'default_search_mode': 'hybrid',
            'default_top_k': 10,
            'include_parent_docs': True,
            'parent_size_threshold': 6000,
            'parent_inclusion_strategy': 'size_based',
            'max_context_size': 20000
        }
    }
    
    # Initialize app with database connections
    app = None  # Your app instance
    
    # Perform search
    query = "What are the key features of the new product?"
    results = await enhanced_rag_search_with_late_chunking(
        app=app,
        query=query,
        search_type=config['search']['default_search_mode'],
        top_k=config['search']['default_top_k'],
        rag_config={**config['chunking'], **config['search']}
    )
    
    # Process results
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Type: {'Parent Document' if result['is_parent'] else 'Chunk'}")
        print(f"Score: {result['score']:.3f}")
        print(f"Content: {result['content'][:200]}...")
        if result['is_chunk']:
            print(f"Chunk {result['metadata']['chunk_index']} of {result['metadata']['total_chunks']}")
    
    return results


# Utility function to update document chunking configuration
async def update_document_chunking_config(
    app: Any,
    media_id: int,
    template_name: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Update the chunking configuration for a specific document.
    
    Args:
        app: Application instance
        media_id: Media document ID
        template_name: Name of chunking template to use
        custom_config: Custom chunking configuration
        
    Returns:
        True if successful
    """
    late_chunker = LateChunkingService(
        db_path=app.db_config.get('media_db_path')
    )
    
    # Create chunking config
    if template_name:
        config = ChunkingConfig(template=template_name)
    elif custom_config:
        config = ChunkingConfig(**custom_config)
    else:
        logger.error("Must provide either template_name or custom_config")
        return False
    
    # Update configuration
    success = late_chunker.update_document_config(media_id, config)
    
    if success:
        logger.info(f"Updated chunking config for media {media_id}")
    else:
        logger.error(f"Failed to update chunking config for media {media_id}")
    
    return success