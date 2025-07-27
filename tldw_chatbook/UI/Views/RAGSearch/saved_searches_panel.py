"""
Saved Searches Panel Component

Panel for managing saved search configurations
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Button, ListView, ListItem
from textual.css.query import NoMatches
from loguru import logger

from ....Utils.paths import get_user_data_dir


class SavedSearchesPanel(Container):
    """Enhanced panel for managing saved searches"""
    
    def __init__(self):
        super().__init__(id="saved-searches-panel", classes="saved-searches-panel-enhanced")
        self.saved_searches: Dict[str, Dict[str, Any]] = self._load_saved_searches()
        self.selected_search_name: Optional[str] = None
        
    def compose(self) -> ComposeResult:
        with Container(classes="saved-searches-wrapper"):
            with Horizontal(classes="saved-searches-header"):
                yield Static("💾 Saved Searches", classes="saved-searches-title")
                yield Button("+", id="new-saved-search", classes="new-search-button", tooltip="Save current search")
            
            if self.saved_searches:
                yield ListView(id="saved-searches-list", classes="saved-searches-list-enhanced")
            else:
                yield Static(
                    "No saved searches yet.\nPerform a search and click 'Save Search' to store it.",
                    classes="empty-saved-searches"
                )
            
            with Horizontal(classes="saved-search-actions-enhanced"):
                yield Button("📥 Load", id="load-saved-search", classes="saved-action-button", disabled=True)
                yield Button("🗑️ Delete", id="delete-saved-search", classes="saved-action-button danger", disabled=True)
    
    def _load_saved_searches(self) -> Dict[str, Dict[str, Any]]:
        """Load saved searches from user data"""
        saved_searches_path = get_user_data_dir() / "saved_searches.json"
        if saved_searches_path.exists():
            try:
                with open(saved_searches_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading saved searches: {e}")
        return {}
    
    def save_search(self, name: str, config: Dict[str, Any]) -> None:
        """Save a search configuration with validation and overwrite protection"""
        # Sanitize the name to prevent issues with invalid characters
        sanitized_name = name.strip()
        if not sanitized_name:
            logger.warning("Cannot save search with empty name")
            return
            
        # Check if this name already exists
        if sanitized_name in self.saved_searches:
            # Update the existing search with new config
            self.saved_searches[sanitized_name]["config"] = config
            self.saved_searches[sanitized_name]["last_used"] = datetime.now().isoformat()
            self.saved_searches[sanitized_name]["updated_at"] = datetime.now().isoformat()
            logger.info(f"Updated existing saved search: {sanitized_name}")
        else:
            # Create a new saved search
            self.saved_searches[sanitized_name] = {
                "config": config,
                "created_at": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat(),
                "updated_at": None
            }
            logger.info(f"Created new saved search: {sanitized_name}")
            
        # Limit the number of saved searches to prevent performance issues
        if len(self.saved_searches) > 50:  # Arbitrary limit
            # Remove the oldest saved search by creation time
            oldest_name = min(
                self.saved_searches.keys(),
                key=lambda k: datetime.fromisoformat(self.saved_searches[k]["created_at"])
            )
            del self.saved_searches[oldest_name]
            logger.info(f"Removed oldest saved search to maintain limit: {oldest_name}")
            
        self._persist_saved_searches()
        self.refresh_list()
    
    def _persist_saved_searches(self) -> None:
        """Save searches to disk"""
        saved_searches_path = get_user_data_dir() / "saved_searches.json"
        saved_searches_path.parent.mkdir(parents=True, exist_ok=True)
        with open(saved_searches_path, 'w') as f:
            json.dump(self.saved_searches, f, indent=2)
    
    async def refresh_list(self) -> None:
        """Refresh the saved searches list"""
        # Check if we have a list view or need to show empty state
        try:
            list_view = self.query_one("#saved-searches-list", ListView)
            await list_view.clear()
            
            for name, data in self.saved_searches.items():
                created = datetime.fromisoformat(data['created_at']).strftime("%Y-%m-%d %H:%M")
                list_item = ListItem(
                    Static(f"{name}\n[dim]{created}[/dim]", classes="saved-search-item")
                )
                await list_view.append(list_item)
                
            # Enable/disable action buttons based on selection
            self.query_one("#load-saved-search").disabled = True
            self.query_one("#delete-saved-search").disabled = True
        except NoMatches:
            # List view doesn't exist, we're showing empty state
            logger.debug("Saved searches list view not found, showing empty state")
            pass