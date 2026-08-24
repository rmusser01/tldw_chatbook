"""
Simplified RAG search service for MCP integration.

This module provides a simple interface for RAG search functionality
specifically designed for MCP (Model Context Protocol) integration.
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

from ...DB.Client_Media_DB_v2 import MediaDatabase
from ..ingestion_indexing import get_shared_rag_service
from .active_config import resolve_active_rag_search_mode


class SimplifiedRAGSearchService:
    """Simplified RAG search service for MCP integration."""

    def __init__(self, media_db: MediaDatabase):
        """
        Initialize the search service.

        Args:
            media_db: Media database instance
        """
        self.media_db = media_db
        self.rag_service = None

    async def profile_search(
        self, query: str, limit: int = 10, media_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search using the active profile's configured mode."""
        mode = resolve_active_rag_search_mode()
        if mode == "plain":
            return await self.keyword_search(query, limit, media_types)
        return await self._enhanced_search(query, limit, media_types, search_type=mode)

    async def semantic_search(
        self, query: str, limit: int = 10, media_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Perform explicit semantic search across media."""
        return await self._enhanced_search(
            query, limit, media_types, search_type="semantic"
        )

    async def _enhanced_search(
        self,
        query: str,
        limit: int,
        media_types: Optional[List[str]],
        *,
        search_type: str,
    ) -> List[Dict[str, Any]]:
        service = self.rag_service
        if service is None:
            try:
                service = await asyncio.to_thread(get_shared_rag_service)
            except Exception as exc:
                logger.error(f"Failed to acquire shared RAG runtime: {exc}")
                service = None
        if service is None:
            return await self.keyword_search(query, limit, media_types)
        filter_metadata = {"media_type": {"$in": media_types}} if media_types else None
        results = await service.search(
            query=query,
            top_k=limit,
            search_type=search_type,
            filter_metadata=filter_metadata,
            metadata_allowlist={"source_type": ("media",)},
        )
        return self._format_enhanced_results(results)

    @staticmethod
    def _format_enhanced_results(results: Any) -> List[Dict[str, Any]]:
        return [
            {
                "id": result.id,
                "title": result.metadata.get("title", "Untitled"),
                "content": result.document,
                "media_type": result.metadata.get("media_type", "unknown"),
                "url": result.metadata.get("url"),
                "file_path": result.metadata.get("file_path"),
                "score": result.score,
                "metadata": result.metadata,
            }
            for result in results
        ]

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
                        # No score, not a fabricated 1.0: FTS/keyword
                        # relevance is not comparable to a real similarity
                        # score, and a wrong band is worse than no band.
                        # Mirrors the Library's own precedent of nulling the
                        # score at the service boundary for exactly this
                        # reason (see `library_rag_state.py`'s
                        # `library_rag_score_suffix`, which already treats
                        # `None` as "no band" for keyword-mode rows).
                        "score": None,
                        "metadata": {
                            "author": item.get("author"),
                            "ingestion_date": item.get("ingestion_date"),
                            "transcription_model": item.get("transcription_model"),
                        },
                    }
                )

            return results[:limit]

        except Exception:
            # Do NOT swallow this into an empty result (task-2271): a
            # missing/renamed media_db method or any other search failure
            # must surface as an error, never as a silent "0 results".
            logger.error("Keyword search failed.")
            raise
