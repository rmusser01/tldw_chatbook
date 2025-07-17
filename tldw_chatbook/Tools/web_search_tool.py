"""
Web Search Tool for LLM function calling.

This tool allows LLMs to search the web using various search engines.
"""

import json
from typing import Dict, Any
from loguru import logger

from . import Tool
from ..Web_Scraping.WebSearch_APIs import perform_websearch


class WebSearchTool(Tool):
    """Tool for performing web searches."""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for information using various search engines. Returns a list of relevant search results."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "search_engine": {
                    "type": "string",
                    "description": "Search engine to use",
                    "enum": ["google", "bing", "duckduckgo", "brave", "kagi", "tavily", "searx"],
                    "default": "duckduckgo"
                },
                "result_count": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute a web search.
        
        Args:
            query: The search query
            search_engine: Engine to use (default: duckduckgo)
            result_count: Number of results (default: 5)
            
        Returns:
            Dictionary with search results or error
        """
        query = kwargs.get("query")
        if not query:
            return {
                "error": "No search query provided"
            }
        
        search_engine = kwargs.get("search_engine", "duckduckgo")
        result_count = kwargs.get("result_count", 5)
        
        # Validate result count
        if not isinstance(result_count, int) or result_count < 1 or result_count > 10:
            result_count = 5
        
        try:
            logger.info(f"Executing web search: '{query}' using {search_engine}")
            
            # Call the web search function
            # Using sensible defaults for other parameters
            results = perform_websearch(
                search_engine=search_engine,
                search_query=query,
                content_country="US",
                search_lang="en",
                output_lang="en",
                result_count=result_count,
                date_range=None,
                safesearch="moderate",
                site_blacklist=None,
                exactTerms=None,
                excludeTerms=None,
                filter=None,
                geolocation=None,
                search_result_language=None,
                sort_results_by=None
            )
            
            # Extract and format the results
            if isinstance(results, dict) and "results" in results:
                search_results = results["results"]
                
                # Format results for the LLM
                formatted_results = []
                for i, result in enumerate(search_results[:result_count]):
                    formatted_result = {
                        "position": i + 1,
                        "title": result.get("title", "No title"),
                        "url": result.get("url", ""),
                        "snippet": result.get("snippet", "No description available")
                    }
                    formatted_results.append(formatted_result)
                
                return {
                    "query": query,
                    "engine": search_engine,
                    "result_count": len(formatted_results),
                    "results": formatted_results
                }
            else:
                # Handle different result formats or errors
                return {
                    "query": query,
                    "engine": search_engine,
                    "error": "No results found or unexpected response format",
                    "raw_response": str(results)[:500]  # Truncate for safety
                }
                
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return {
                "query": query,
                "engine": search_engine,
                "error": f"Search failed: {str(e)}"
            }