# tldw_chatbook/Widgets/empty_state.py
# Empty state widget for lists and containers
#
# Imports
from typing import Optional, Callable
from textual.app import ComposeResult
from textual.containers import Vertical, Center
from textual.widgets import Static, Button
from textual.widget import Widget

class EmptyState(Widget):
    """Empty state widget to show when lists or containers have no content.
    
    Shows:
    - Icon or emoji
    - Title message
    - Optional description
    - Optional action button
    """
    
    def __init__(
        self,
        icon: str = "📭",
        title: str = "No items found",
        description: Optional[str] = None,
        action_label: Optional[str] = None,
        action_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.icon = icon
        self.title = title
        self.description = description
        self.action_label = action_label
        self.action_id = action_id
        
    def compose(self) -> ComposeResult:
        """Compose the empty state display."""
        with Center(classes="empty-state-container"):
            with Vertical(classes="empty-state-content"):
                # Icon
                yield Static(self.icon, classes="empty-state-icon")
                
                # Title
                yield Static(self.title, classes="empty-state-title")
                
                # Description (optional)
                if self.description:
                    yield Static(self.description, classes="empty-state-description")
                
                # Action button (optional)
                if self.action_label and self.action_id:
                    yield Button(
                        self.action_label,
                        id=self.action_id,
                        classes="empty-state-action",
                        variant="primary"
                    )


# Pre-configured empty states for common scenarios

class ModelsEmptyState(EmptyState):
    """Empty state for model lists."""
    
    def __init__(self, **kwargs):
        super().__init__(
            icon="🤖",
            title="No embedding models found",
            description="Download or configure embedding models to get started",
            action_label="Download Models",
            action_id="empty-state-download-models",
            **kwargs
        )


class CollectionsEmptyState(EmptyState):
    """Empty state for collection lists."""
    
    def __init__(self, **kwargs):
        super().__init__(
            icon="📚",
            title="No collections yet",
            description="Create your first collection to start organizing embeddings",
            action_label="Create Collection",
            action_id="empty-state-create-collection",
            **kwargs
        )


class SearchResultsEmptyState(EmptyState):
    """Empty state for search results."""
    
    def __init__(self, search_term: str = "", **kwargs):
        super().__init__(
            icon="🔍",
            title="No results found",
            description=f"No items match '{search_term}'" if search_term else "Try adjusting your search criteria",
            **kwargs
        )


class FilesEmptyState(EmptyState):
    """Empty state for file selection."""
    
    def __init__(self, **kwargs):
        super().__init__(
            icon="📁",
            title="No files selected",
            description="Select files to create embeddings from",
            action_label="Select Files",
            action_id="empty-state-select-files",
            **kwargs
        )