"""
Simplified RAG search service for MCP integration.

This module provides a simple interface for RAG search functionality
specifically designed for MCP (Model Context Protocol) integration.
"""

from typing import List, Dict, Any, Optional, Union
from loguru import logger

from ...DB.Client_Media_DB_v2 import MediaDatabase
from ...config import load_settings
from .rag_factory import create_rag_service
from .config import RAGConfig


class SimplifiedRAGSearchService:
    """Simplified RAG search service for MCP integration."""
    
    def __init__(self, media_db: MediaDatabase):
        """
        Initialize the search service.
        
        Args:
            media_db: Media database instance
        """
        self.media_db = media_db
        
        # Load RAG configuration
        settings = load_settings()
        rag_config = settings.get('rag_search', {})
        service_config = rag_config.get('service', {})
        
        # Get profile name
        profile_name = service_config.get('profile', 'hybrid_basic')
        
        # Create RAG service with profile
        try:
            self.rag_service = create_rag_service(profile_name=profile_name)
            logger.info(f"Using profile '{profile_name}' for MCP integration")
        except Exception as e:
            logger.error(f"Failed to create RAG service: {e}")
            self.rag_service = None
            logger.info("Falling back to basic search for MCP integration")
        
    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        media_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search across media.
        
        Args:
            query: Search query
            limit: Maximum number of results
            media_types: Optional list of media types to filter
            
        Returns:
            List of search results
        """
        try:
            # Use enhanced RAG service if available
            if self.rag_service and hasattr(self.rag_service, 'search'):
                # Build metadata filter for media types
                filter_metadata = None
                if media_types:
                    filter_metadata = {"media_type": {"$in": media_types}}
                
                # Perform semantic search
                results = await self.rag_service.search(
                    query=query,
                    top_k=limit,
                    search_type="semantic",
                    filter_metadata=filter_metadata
                )
                
                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'id': result.id,
                        'title': result.metadata.get('title', 'Untitled'),
                        'content': result.content,
                        'media_type': result.metadata.get('media_type', 'unknown'),
                        'url': result.metadata.get('url'),
                        'file_path': result.metadata.get('file_path'),
                        'score': result.score,
                        'metadata': result.metadata
                    })
                
                return formatted_results
            else:
                # Fall back to keyword search
                return await self.keyword_search(query, limit, media_types)
        except Exception as e:
            logger.error(f"Error in semantic_search: {e}")
            return []
    
    async def keyword_search(
        self,
        query: str,
        limit: int = 10,
        media_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword search across media.
        
        Args:
            query: Search query
            limit: Maximum number of results
            media_types: Optional list of media types to filter
            
        Returns:
            List of search results
        """
        try:
            # Search using the media database
            results = []
            
            # Search media items
            media_results = self.media_db.search_media(
                query=query,
                limit=limit,
                media_types=media_types
            )
            
            for item in media_results:
                results.append({
                    'id': item.get('id'),
                    'title': item.get('title', 'Untitled'),
                    'content': item.get('content', ''),
                    'media_type': item.get('type', 'unknown'),
                    'url': item.get('url'),
                    'file_path': item.get('local_path'),
                    'score': 1.0,  # Default score for keyword search
                    'metadata': {
                        'author': item.get('author'),
                        'created_at': item.get('created_at'),
                        'ingestion_date': item.get('ingestion_date'),
                        'transcription_model': item.get('transcription_model')
                    }
                })
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in keyword_search: {e}")
            return []