"""
Search History Dropdown Component

Provides auto-complete functionality with search history
"""

from typing import List
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import ListView, ListItem, Static

from ....DB.search_history_db import SearchHistoryDB


class SearchHistoryDropdown(Container):
    """Dropdown for search history with auto-complete functionality"""
    
    def __init__(self, search_history_db: SearchHistoryDB):
        super().__init__(id="search-history-dropdown", classes="search-history-dropdown hidden")
        self.search_history_db = search_history_db
        self.history_items: List[str] = []
        
    def compose(self) -> ComposeResult:
        yield ListView(id="search-history-list", classes="search-history-list")
            
    async def show_history(self, current_query: str = "") -> None:
        """Show search history filtered by current query"""
        list_view = self.query_one("#search-history-list", ListView)
        await list_view.clear()
        
        # Get recent searches
        history = self.search_history_db.get_search_history(limit=10, days_back=30)
        self.history_items = []
        
        for item in history:
            query = item['query']
            if current_query.lower() in query.lower() or not current_query:
                self.history_items.append(query)
                list_item = ListItem(Static(query, classes="history-item-text"))
                await list_view.append(list_item)
        
        if self.history_items:
            self.remove_class("hidden")
        else:
            self.add_class("hidden")
            
    def hide(self) -> None:
        """Hide the dropdown"""
        self.add_class("hidden")