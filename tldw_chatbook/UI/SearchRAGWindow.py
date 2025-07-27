# SearchRAGWindow.py
# Description: Main RAG search interface - imports modularized components
#
# This file now serves as a compatibility layer, re-exporting the modularized components
# from the Views/RAGSearch package for backward compatibility.

from .Views.RAGSearch import (
    SearchRAGWindow,
    SearchHistoryDropdown,
    SearchResult,
    SavedSearchesPanel
)

# Re-export for backward compatibility
__all__ = [
    "SearchRAGWindow",
    "SearchHistoryDropdown",
    "SearchResult",
    "SavedSearchesPanel"
]

# Import statements preserved for reference
"""
Original imports have been moved to their respective component files:
- SearchHistoryDropdown -> Views/RAGSearch/search_history_dropdown.py
- SearchResult -> Views/RAGSearch/search_result.py  
- SavedSearchesPanel -> Views/RAGSearch/saved_searches_panel.py
- SearchRAGWindow -> Views/RAGSearch/search_rag_window.py
- Constants -> Views/RAGSearch/constants.py
- Event Handlers -> Views/RAGSearch/search_event_handlers.py

The original SearchRAGWindow.py file has been backed up as SearchRAGWindow.py.bak
"""