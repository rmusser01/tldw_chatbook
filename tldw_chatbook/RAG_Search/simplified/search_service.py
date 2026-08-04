"""
Simplified RAG search service for MCP integration.

This module provides a simple interface for RAG search functionality
specifically designed for MCP (Model Context Protocol) integration.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from ...DB.Client_Media_DB_v2 import MediaDatabase
from ...config import load_settings
from .rag_factory import create_rag_service


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
        rag_config = settings.get("rag_search", {})
        service_config = rag_config.get("service", {})

        # Get profile name
        profile_name = service_config.get("profile", "hybrid_basic")

        # Create RAG service with profile
        try:
            self.rag_service = create_rag_service(profile_name=profile_name)
            logger.info(f"Using profile '{profile_name}' for MCP integration")
        except Exception as e:
            logger.error(f"Failed to create RAG service: {e}")
            self.rag_service = None
            logger.info("Falling back to basic search for MCP integration")

    async def semantic_search(
        self, query: str, limit: int = 10, media_types: Optional[List[str]] = None
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
            if self.rag_service and hasattr(self.rag_service, "search"):
                # Build metadata filter for media types
                filter_metadata = None
                if media_types:
                    filter_metadata = {"media_type": {"$in": media_types}}

                # Perform semantic search
                results = await self.rag_service.search(
                    query=query,
                    top_k=limit,
                    search_type="semantic",
                    filter_metadata=filter_metadata,
                )

                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append(
                        {
                            "id": result.id,
                            "title": result.metadata.get("title", "Untitled"),
                            # `.document` is the real field on both
                            # SearchResultWithCitations (citations.py) and
                            # SearchResult (vector_store.py) -- neither
                            # dataclass has a `.content` attribute (task-2271
                            # round 2: this crashed with AttributeError on
                            # every semantic search, i.e. the tool's DEFAULT
                            # path, since use_semantic defaults True).
                            "content": result.document,
                            "media_type": result.metadata.get("media_type", "unknown"),
                            "url": result.metadata.get("url"),
                            "file_path": result.metadata.get("file_path"),
                            "score": result.score,
                            "metadata": result.metadata,
                        }
                    )

                return formatted_results
            else:
                # Fall back to keyword search
                return await self.keyword_search(query, limit, media_types)
        except Exception as e:
            # Do NOT swallow this into an empty result (task-2271): a crash
            # here (including one that propagates up from the keyword_search
            # fallback above) must surface as an error, never as a silent
            # "0 results". The caller (MCP/tools.py:perform_rag_search)
            # already catches this into the honest `[{"error": ...}]` shape.
            logger.error(f"Error in semantic_search: {e}")
            raise

    async def keyword_search(
        self, query: str, limit: int = 10, media_types: Optional[List[str]] = None
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
            # Search using the media database. NOTE: the real API is
            # `search_media_db` (there is no `search_media` method) and it
            # returns a (rows, total_matches) tuple -- see
            # Client_Media_DB_v2.MediaDatabase.search_media_db.
            media_results, _total_matches = self.media_db.search_media_db(
                search_query=query,
                media_types=media_types,
                results_per_page=limit,
            )

            # search_media_db's row projection is metadata-only -- it does
            # not select the (potentially large) `content` column, matching
            # the same "second query" pattern used by
            # MediaDatabase.search_media_by_keyword_for_embedding. Batch-fetch
            # content for the matched ids in one query rather than N+1.
            media_ids = [
                item["id"] for item in media_results if item.get("id") is not None
            ]
            content_by_id: Dict[Any, str] = {}
            if media_ids:
                content_by_id = {
                    row["id"]: row.get("content", "") or ""
                    for row in self.media_db.get_media_by_ids_for_embedding(media_ids)
                }

            results = []
            for item in media_results:
                results.append(
                    {
                        "id": item.get("id"),
                        "title": item.get("title", "Untitled"),
                        "content": content_by_id.get(item.get("id"), ""),
                        "media_type": item.get("type", "unknown"),
                        "url": item.get("url"),
                        # Media has no separate local-path column -- `url`
                        # doubles as the source reference for both web and
                        # locally-ingested items (see Client_Media_DB_v2's
                        # Media table schema). Kept as its own key so the
                        # outer result shape (consumed by MCP/tools.py) stays
                        # stable; there is no distinct value to put here.
                        "file_path": None,
                        "score": 1.0,  # Default score for keyword search
                        "metadata": {
                            "author": item.get("author"),
                            "ingestion_date": item.get("ingestion_date"),
                            "transcription_model": item.get("transcription_model"),
                        },
                    }
                )

            return results[:limit]

        except Exception as e:
            # Do NOT swallow this into an empty result (task-2271): a
            # missing/renamed media_db method or any other search failure
            # must surface as an error, never as a silent "0 results".
            logger.error(f"Error in keyword_search: {e}")
            raise
