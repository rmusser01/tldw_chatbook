# tldw_chatbook/Widgets/embeddings_list_items.py
# Enhanced list items for embeddings management
#
# Imports
from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# Third-party imports
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, ListItem, Checkbox
from textual.reactive import reactive
from loguru import logger

# Configure logger
logger = logger.bind(module="embeddings_list_items")


class ModelListItem(ListItem):
    """Enhanced list item for embedding model display.
    
    Shows:
    - Model name with icon
    - Provider (OpenAI, HuggingFace, Local)
    - Download status
    - Memory status
    - Model size (if available)
    """
    
    # Selection state for batch operations
    is_selected: reactive[bool] = reactive(False)
    
    def __init__(
        self,
        model_id: str,
        model_info: Dict[str, Any],
        show_selection: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.model_info = model_info
        self.show_selection = show_selection
        
    def compose(self) -> ComposeResult:
        """Compose the model list item."""
        with Horizontal(classes="model-list-item-container"):
            # Selection checkbox (if enabled)
            if self.show_selection:
                yield Checkbox(
                    value=self.is_selected,
                    id=f"select-model-{self.model_id}",
                    classes="model-select-checkbox"
                )
            
            # Model icon
            yield Static(self._get_model_icon(), classes="model-icon")
            
            # Model info container
            with Vertical(classes="model-info-container"):
                # Model name and provider
                with Horizontal(classes="model-header"):
                    # Add favorite star if model is a favorite
                    name_text = self.model_id
                    if self.model_info.get('is_favorite', False):
                        name_text = "★ " + name_text
                    yield Static(name_text, classes="model-name")
                    yield Static(f"({self._get_provider()})", classes="model-provider")
                
                # Status indicators
                with Horizontal(classes="model-status-row"):
                    yield Static(
                        self._get_download_status(),
                        classes=f"model-download-status {self._get_download_class()}"
                    )
                    yield Static(
                        self._get_memory_status(),
                        classes=f"model-memory-status {self._get_memory_class()}"
                    )
                    if size := self._get_model_size():
                        yield Static(size, classes="model-size")
    
    def _get_model_icon(self) -> str:
        """Get appropriate icon for model type."""
        provider = self.model_info.get('provider', '').lower()
        if provider == 'openai':
            return "🤖"
        elif provider == 'huggingface':
            return "🤗"
        elif self.model_id.startswith('local/'):
            return "📦"
        else:
            return "🧠"
    
    def _get_provider(self) -> str:
        """Get provider display name."""
        provider = self.model_info.get('provider', 'unknown')
        return provider.title()
    
    def _get_download_status(self) -> str:
        """Get download status display."""
        if self.model_info.get('provider') == 'openai':
            return "☁️ Cloud"
        elif self.model_info.get('is_downloaded', False):
            return "✅ Downloaded"
        else:
            return "⬇️ Not downloaded"
    
    def _get_download_class(self) -> str:
        """Get CSS class for download status."""
        if self.model_info.get('provider') == 'openai':
            return "cloud"
        elif self.model_info.get('is_downloaded', False):
            return "downloaded"
        else:
            return "not-downloaded"
    
    def _get_memory_status(self) -> str:
        """Get memory status display."""
        if self.model_info.get('is_loaded', False):
            return "🟢 Loaded"
        else:
            return "⚪ Not loaded"
    
    def _get_memory_class(self) -> str:
        """Get CSS class for memory status."""
        return "loaded" if self.model_info.get('is_loaded', False) else "not-loaded"
    
    def _get_model_size(self) -> Optional[str]:
        """Get model size if available."""
        if size_bytes := self.model_info.get('size_bytes'):
            return self._format_size(size_bytes)
        elif size_str := self.model_info.get('size'):
            return size_str
        return None
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"


class CollectionListItem(ListItem):
    """Enhanced list item for vector collection display.
    
    Shows:
    - Collection name with icon
    - Document count
    - Last modified date
    - Total size estimate
    - Status indicator
    """
    
    # Selection state for batch operations
    is_selected: reactive[bool] = reactive(False)
    
    def __init__(
        self,
        collection_name: str,
        collection_info: Dict[str, Any],
        show_selection: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.collection_name = collection_name
        self.collection_info = collection_info
        self.show_selection = show_selection
        
    def compose(self) -> ComposeResult:
        """Compose the collection list item."""
        with Horizontal(classes="collection-list-item-container"):
            # Selection checkbox (if enabled)
            if self.show_selection:
                yield Checkbox(
                    value=self.is_selected,
                    id=f"select-collection-{self.collection_name}",
                    classes="collection-select-checkbox"
                )
            
            # Collection icon
            yield Static("📚", classes="collection-icon")
            
            # Collection info container
            with Vertical(classes="collection-info-container"):
                # Collection name
                yield Static(self.collection_name, classes="collection-name")
                
                # Metadata row
                with Horizontal(classes="collection-metadata-row"):
                    # Document count
                    doc_count = self.collection_info.get('document_count', 0)
                    yield Static(f"📄 {doc_count} docs", classes="collection-doc-count")
                    
                    # Last modified
                    if modified := self._get_last_modified():
                        yield Static(f"🕐 {modified}", classes="collection-modified")
                    
                    # Size estimate
                    if size := self._get_size_estimate():
                        yield Static(f"💾 {size}", classes="collection-size")
                    
                    # Status
                    yield Static(
                        self._get_status(),
                        classes=f"collection-status {self._get_status_class()}"
                    )
    
    def _get_last_modified(self) -> Optional[str]:
        """Get formatted last modified date."""
        if modified_ts := self.collection_info.get('last_modified'):
            try:
                dt = datetime.fromtimestamp(modified_ts)
                # Show relative time for recent, date for older
                now = datetime.now()
                diff = now - dt
                
                if diff.days == 0:
                    if diff.seconds < 3600:
                        return f"{diff.seconds // 60}m ago"
                    else:
                        return f"{diff.seconds // 3600}h ago"
                elif diff.days == 1:
                    return "Yesterday"
                elif diff.days < 7:
                    return f"{diff.days}d ago"
                else:
                    return dt.strftime("%Y-%m-%d")
            except:
                pass
        return None
    
    def _get_size_estimate(self) -> Optional[str]:
        """Get formatted size estimate."""
        if size_bytes := self.collection_info.get('size_bytes'):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} TB"
        return None
    
    def _get_status(self) -> str:
        """Get collection status."""
        status = self.collection_info.get('status', 'ready')
        if status == 'indexing':
            return "⚡ Indexing"
        elif status == 'error':
            return "❌ Error"
        else:
            return "✅ Ready"
    
    def _get_status_class(self) -> str:
        """Get CSS class for status."""
        status = self.collection_info.get('status', 'ready')
        return status.lower()