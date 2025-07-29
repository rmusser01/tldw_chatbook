# tldw_chatbook/Widgets/HuggingFace/local_models_widget.py
"""
Widget for managing locally downloaded models.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Label, Button, ListView, ListItem, Static
from textual.reactive import reactive
from textual.message import Message
from textual import work
from loguru import logger

if TYPE_CHECKING:
    from ...app import TldwCli


class ModelInfo:
    """Information about a downloaded model."""
    
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.size = path.stat().st_size if path.exists() else 0
        self.modified = datetime.fromtimestamp(path.stat().st_mtime) if path.exists() else None
        
        # Try to extract repo info from parent directory
        parent = path.parent
        if parent.name != "tldw_models":
            self.repo_id = parent.name.replace("_", "/", 1)  # Convert back first underscore
        else:
            self.repo_id = "Unknown"
    
    @property
    def size_str(self) -> str:
        """Get human-readable size string."""
        size = self.size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    @property
    def modified_str(self) -> str:
        """Get human-readable modified date."""
        if not self.modified:
            return "Unknown"
        return self.modified.strftime("%Y-%m-%d %H:%M")


class DeleteConfirmEvent(Message):
    """Event requesting deletion confirmation."""
    
    def __init__(self, model_path: Path) -> None:
        super().__init__()
        self.model_path = model_path


class LocalModelsWidget(Container):
    """Widget for browsing and managing locally downloaded models."""
    
    DEFAULT_CSS = """
    LocalModelsWidget {
        height: 100%;
        layout: vertical;
        background: $surface;
    }
    
    LocalModelsWidget .header {
        height: 3;
        padding: 1;
        background: $boost;
        border-bottom: solid $primary;
        layout: horizontal;
    }
    
    LocalModelsWidget .header-title {
        width: 1fr;
        text-style: bold;
    }
    
    LocalModelsWidget .header Button {
        width: auto;
        min-width: 10;
    }
    
    LocalModelsWidget .model-list {
        height: 1fr;
        overflow-y: auto;
        background: $background;
    }
    
    LocalModelsWidget .model-item {
        padding: 1;
        margin: 1;
        background: $surface;
        border: solid $primary-background-darken-1;
        layout: vertical;
    }
    
    LocalModelsWidget .model-header {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    
    LocalModelsWidget .model-name {
        width: 1fr;
        text-style: bold;
    }
    
    LocalModelsWidget .model-size {
        width: auto;
        margin-left: 1;
        color: $primary;
    }
    
    LocalModelsWidget .model-info {
        color: $text-muted;
        height: 1;
        margin-bottom: 1;
    }
    
    LocalModelsWidget .model-actions {
        layout: horizontal;
        height: 3;
    }
    
    LocalModelsWidget .model-actions Button {
        margin-right: 1;
    }
    
    LocalModelsWidget .summary-bar {
        height: 3;
        padding: 1;
        background: $boost;
        border-top: solid $primary;
    }
    
    LocalModelsWidget .empty-state {
        height: 1fr;
        align: center middle;
        padding: 4;
    }
    
    LocalModelsWidget .empty-state-message {
        text-align: center;
        color: $text-muted;
    }
    
    LocalModelsWidget .delete-confirm-dialog {
        dock: top;
        layer: notification;
        width: 60;
        height: 12;
        max-height: 20;
        background: $boost;
        border: double $error;
        padding: 2;
        margin-top: 4;
        align: center middle;
    }
    
    LocalModelsWidget .delete-confirm-title {
        text-style: bold;
        color: $error;
        margin-bottom: 1;
        height: auto;
    }
    
    LocalModelsWidget .delete-confirm-message {
        margin-bottom: 2;
        height: auto;
        text-align: center;
    }
    
    LocalModelsWidget .delete-confirm-buttons {
        layout: horizontal;
        height: auto;
        align: center middle;
        width: 100%;
    }
    
    LocalModelsWidget .delete-confirm-buttons Button {
        margin: 0 2;
        min-width: 12;
        height: 3;
    }
    """
    
    # Reactive properties
    models: reactive[List[ModelInfo]] = reactive([])
    show_delete_confirm: reactive[bool] = reactive(False)
    pending_delete_path: reactive[Optional[Path]] = reactive(None)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        """Initialize the local models widget."""
        super().__init__(**kwargs)
        self.app_instance = app_instance
        
        # Get download directory from config
        config = self.app_instance.app_config
        self.download_dir = Path(
            config.get("llm_management", {}).get("model_download_dir", "~/Downloads/tldw_models")
        ).expanduser()
    
    def compose(self) -> ComposeResult:
        """Compose the widget UI."""
        # Header
        with Container(classes="header"):
            yield Label("📁 Local Models", classes="header-title")
            yield Button("Refresh", id="refresh-models", variant="primary")
        
        # Model list or empty state
        yield ListView(id="model-list", classes="model-list")
        
        # Summary bar
        with Container(classes="summary-bar"):
            yield Static("", id="summary-text")
        
        # Delete confirmation dialog (hidden by default)
        with Container(classes="delete-confirm-dialog", id="delete-confirm-dialog"):
            with Vertical():
                yield Label("⚠️ Confirm Deletion", classes="delete-confirm-title")
                yield Label("", id="delete-confirm-message", classes="delete-confirm-message")
                with Horizontal(classes="delete-confirm-buttons"):
                    yield Button("Cancel", id="delete-cancel", variant="default")
                    yield Button("Delete", id="delete-confirm", variant="error")
    
    def on_mount(self) -> None:
        """Scan for models on mount."""
        logger.info("LocalModelsWidget mounted, scanning for models")
        self.scan_models()
        
        # Hide delete confirmation dialog initially
        dialog = self.query_one("#delete-confirm-dialog")
        dialog.display = False
    
    def watch_models(self, models: List[ModelInfo]) -> None:
        """Update UI when models change."""
        self.call_later(self._refresh_model_list)
        self._update_summary()
    
    def watch_show_delete_confirm(self, show: bool) -> None:
        """Show/hide delete confirmation dialog."""
        dialog = self.query_one("#delete-confirm-dialog")
        dialog.display = show
    
    @work(thread=True)
    def scan_models(self) -> None:
        """Scan the download directory for model files."""
        logger.info(f"Scanning for models in: {self.download_dir}")
        
        if not self.download_dir.exists():
            logger.warning(f"Download directory does not exist: {self.download_dir}")
            self.models = []
            return
        
        model_files = []
        
        # Common model file extensions
        model_extensions = {'.gguf', '.bin', '.safetensors', '.pt', '.pth', '.onnx'}
        
        # Recursively find all model files
        for root, dirs, files in os.walk(self.download_dir):
            for file in files:
                if any(file.endswith(ext) for ext in model_extensions):
                    file_path = Path(root) / file
                    # Skip small files that might be configs
                    if file_path.stat().st_size > 1024 * 1024:  # > 1MB
                        model_files.append(ModelInfo(file_path))
        
        # Sort by modified date (newest first)
        model_files.sort(key=lambda m: m.modified or datetime.min, reverse=True)
        
        logger.info(f"Found {len(model_files)} model files")
        self.models = model_files
    
    async def _refresh_model_list(self) -> None:
        """Refresh the model list display."""
        model_list = self.query_one("#model-list", ListView)
        await model_list.clear()
        
        if not self.models:
            # Show empty state
            empty_container = Container(
                Label("No models found", classes="empty-state-message"),
                Label("Download models from the 'Download Models' tab", classes="empty-state-message"),
                classes="empty-state"
            )
            await model_list.append(ListItem(empty_container))
        else:
            # Add model items
            for model in self.models:
                item = self._create_model_item(model)
                await model_list.append(item)
    
    def _create_model_item(self, model: ModelInfo) -> ListItem:
        """Create a list item for a model."""
        # Build widgets
        widgets = []
        
        # Header with name and size
        header = Horizontal(
            Static(model.name, classes="model-name"),
            Static(model.size_str, classes="model-size"),
            classes="model-header"
        )
        widgets.append(header)
        
        # Info line
        info_text = f"Repository: {model.repo_id} • Modified: {model.modified_str}"
        widgets.append(Static(info_text, classes="model-info"))
        
        # Path info
        rel_path = model.path.relative_to(self.download_dir)
        widgets.append(Static(f"Path: {rel_path}", classes="model-info"))
        
        # Actions
        delete_btn = Button("Delete", variant="error", classes="delete-button")
        delete_btn.model_path = model.path
        actions = Horizontal(delete_btn, classes="model-actions")
        widgets.append(actions)
        
        # Create container
        container = Vertical(*widgets, classes="model-item")
        
        item = ListItem(container)
        item.model_path = model.path
        return item
    
    def _update_summary(self) -> None:
        """Update the summary text."""
        total_models = len(self.models)
        total_size = sum(m.size for m in self.models)
        
        # Format total size
        size = total_size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                size_str = f"{size:.1f} {unit}"
                break
            size /= 1024.0
        else:
            size_str = f"{size:.1f} PB"
        
        summary = self.query_one("#summary-text", Static)
        if total_models > 0:
            summary.update(f"Total: {total_models} models • Size: {size_str}")
        else:
            summary.update("No models downloaded")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button = event.button
        
        if button.id == "refresh-models":
            # Refresh model list
            logger.info("Refreshing model list")
            self.scan_models()
        
        elif button.id == "delete-cancel":
            # Cancel deletion
            self.show_delete_confirm = False
            self.pending_delete_path = None
        
        elif button.id == "delete-confirm":
            # Confirm deletion
            if self.pending_delete_path:
                self._delete_model(self.pending_delete_path)
            self.show_delete_confirm = False
            self.pending_delete_path = None
        
        elif "delete-button" in button.classes:
            # Request deletion
            if hasattr(button, "model_path"):
                self._show_delete_confirmation(button.model_path)
    
    def _show_delete_confirmation(self, model_path: Path) -> None:
        """Show delete confirmation dialog."""
        self.pending_delete_path = model_path
        
        # Update confirmation message
        message = self.query_one("#delete-confirm-message", Label)
        message.update(f"Are you sure you want to delete:\n{model_path.name}?\n\nThis action cannot be undone.")
        
        self.show_delete_confirm = True
    
    @work(thread=True)
    def _delete_model(self, model_path: Path) -> None:
        """Delete a model file."""
        logger.info(f"Deleting model: {model_path}")
        
        try:
            if model_path.exists():
                if model_path.is_file():
                    model_path.unlink()
                else:
                    shutil.rmtree(model_path)
                
                logger.info(f"Successfully deleted: {model_path}")
                
                # Check if parent directory is empty and remove it
                parent = model_path.parent
                if parent != self.download_dir and parent.exists():
                    try:
                        if not any(parent.iterdir()):
                            parent.rmdir()
                            logger.info(f"Removed empty directory: {parent}")
                    except Exception as e:
                        logger.warning(f"Could not remove parent directory: {e}")
                
                # Refresh model list
                self.scan_models()
                
                # Show success notification
                self.app.call_from_thread(
                    self.notify,
                    f"Deleted: {model_path.name}",
                    severity="information"
                )
            else:
                logger.warning(f"Model file not found: {model_path}")
                self.app.call_from_thread(
                    self.notify,
                    "Model file not found",
                    severity="warning"
                )
        
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            self.app.call_from_thread(
                self.notify,
                f"Error deleting model: {str(e)}",
                severity="error"
            )