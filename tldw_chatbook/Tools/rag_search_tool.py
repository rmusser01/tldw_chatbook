"""
RAG Search Tool for LLM function calling.

This tool allows LLMs to search the knowledge base using the RAG system.
"""

import json
from typing import Dict, Any, List, Optional
from loguru import logger

from . import Tool


class RAGSearchTool(Tool):
    """Tool for performing RAG search across the knowledge base."""
    
    @property
    def name(self) -> str:
        return "rag_search"
    
    @property
    def description(self) -> str:
        return "Search the knowledge base using semantic search. Returns relevant documents with citations."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "collections": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["media", "conversations", "notes", "characters"]
                    },
                    "description": "Collections to search in",
                    "default": ["media", "conversations", "notes"]
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                },
                "search_type": {
                    "type": "string",
                    "description": "Type of search to perform",
                    "enum": ["semantic", "hybrid", "keyword"],
                    "default": "semantic"
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute a RAG search.
        
        Args:
            query: The search query
            collections: Collections to search in
            top_k: Number of results
            search_type: Type of search (semantic, hybrid, keyword)
            
        Returns:
            Dictionary with search results or error
        """
        query = kwargs.get("query")
        if not query:
            return {"error": "No search query provided"}
        
        collections = kwargs.get("collections", ["media", "conversations", "notes"])
        top_k = kwargs.get("top_k", 5)
        search_type = kwargs.get("search_type", "semantic")
        
        # Validate top_k
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            top_k = 5
        
        try:
            # Import RAG components dynamically to handle optional dependencies
            try:
                from ..RAG_Search.simplified import RAGService, create_config_for_collection
                from ..Event_Handlers.Chat_Events.chat_rag_integration import perform_modular_rag_search
                RAG_AVAILABLE = True
            except ImportError as e:
                logger.warning(f"RAG dependencies not available: {e}")
                RAG_AVAILABLE = False
            
            if not RAG_AVAILABLE:
                return {
                    "error": "RAG search is not available. Required dependencies are not installed.",
                    "suggestion": "Install with: pip install tldw_chatbook[embeddings_rag]"
                }
            
            logger.info(f"Executing RAG search: '{query}' in {collections} using {search_type}")
            
            # Perform search across requested collections
            all_results = []
            
            for collection in collections:
                try:
                    # Create config for the collection
                    config = create_config_for_collection(collection)
                    
                    # Create RAG service
                    rag_service = RAGService(config=config)
                    
                    # Perform search
                    results = await rag_service.search(
                        query=query,
                        top_k=top_k,
                        search_type=search_type,
                        include_citations=True
                    )
                    
                    # Format results
                    for i, result in enumerate(results):
                        formatted_result = {
                            "collection": collection,
                            "position": i + 1,
                            "score": result.score,
                            "content": result.content[:500],  # Truncate for safety
                            "metadata": result.metadata or {}
                        }
                        
                        # Add citations if available
                        if hasattr(result, 'citations') and result.citations:
                            formatted_result["citations"] = [
                                {
                                    "text": cit.text[:200],
                                    "start": cit.start,
                                    "end": cit.end
                                }
                                for cit in result.citations[:3]  # Limit citations
                            ]
                        
                        all_results.append(formatted_result)
                    
                except Exception as e:
                    logger.warning(f"Error searching collection {collection}: {e}")
                    # Continue with other collections
            
            # Sort all results by score and take top k
            all_results.sort(key=lambda x: x['score'], reverse=True)
            final_results = all_results[:top_k]
            
            return {
                "query": query,
                "collections": collections,
                "search_type": search_type,
                "result_count": len(final_results),
                "results": final_results
            }
            
        except Exception as e:
            logger.error(f"Error performing RAG search: {e}")
            return {
                "query": query,
                "error": f"Search failed: {str(e)}"
            }