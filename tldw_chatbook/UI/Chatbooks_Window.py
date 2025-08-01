# Chatbooks_Window.py
# Description: Enhanced Chatbooks landing page with better UX and wizard integration
#
"""
Enhanced Chatbooks Window
------------------------

Improved UI for the chatbooks feature with better visual hierarchy,
more features, and enhanced user experience with full wizard integration.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import json
import zipfile

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll, Grid
from textual.widgets import Static, Button, Label, Input, ListView, ListItem
from textual.reactive import reactive
from loguru import logger

if TYPE_CHECKING:
    from ..app import TldwCli


class ChatbookCard(Container):
    """Card widget for displaying a chatbook."""
    
    DEFAULT_CSS = """
    ChatbookCard {
        height: 8;
        background: $boost;
        border: round $background-darken-1;
        padding: 1;
        margin: 0 1 1 1;
    }
    
    ChatbookCard:hover {
        background: $primary 10%;
        border: round $primary 50%;
    }
    
    .chatbook-card-title {
        text-style: bold;
        color: $text;
    }
    
    .chatbook-card-description {
        color: $text-muted;
        text-overflow: ellipsis;
    }
    
    .chatbook-card-meta {
        layout: horizontal;
        height: 1;
        margin-top: 1;
    }
    
    .chatbook-card-date {
        color: $text-disabled;
        width: 1fr;
    }
    
    .chatbook-card-size {
        color: $text-disabled;
        text-align: right;
        width: auto;
    }
    
    .chatbook-card-stats {
        color: $primary;
        margin-top: 1;
    }
    """
    
    def __init__(self, chatbook_data: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.chatbook_data = chatbook_data
        
    def compose(self) -> ComposeResult:
        yield Static(self.chatbook_data.get('name', 'Untitled'), classes="chatbook-card-title")
        yield Static(
            self.chatbook_data.get('description', 'No description')[:100] + '...',
            classes="chatbook-card-description"
        )
        
        # Stats line
        stats = self.chatbook_data.get('statistics', {})
        stats_text = f"📚 {stats.get('conversations', 0)} conversations • 📝 {stats.get('notes', 0)} notes • 👤 {stats.get('characters', 0)} characters"
        yield Static(stats_text, classes="chatbook-card-stats")
        
        # Meta information
        with Horizontal(classes="chatbook-card-meta"):
            created = self.chatbook_data.get('created_at', 'Unknown')
            if isinstance(created, str) and created != 'Unknown':
                try:
                    dt = datetime.fromisoformat(created)
                    created = dt.strftime('%Y-%m-%d')
                except:
                    pass
            yield Static(f"📅 {created}", classes="chatbook-card-date")
            
            size = self.chatbook_data.get('size_mb', 0)
            yield Static(f"💾 {size:.1f} MB", classes="chatbook-card-size")


class EmptyStateWidget(Container):
    """Enhanced empty state widget."""
    
    DEFAULT_CSS = """
    EmptyStateWidget {
        align: center middle;
        height: 100%;
        padding: 4;
    }
    
    .empty-state-icon {
        text-align: center;
        color: $primary;
        text-style: bold;
        margin-bottom: 2;
    }
    
    .empty-state-title {
        text-align: center;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    .empty-state-description {
        text-align: center;
        color: $text-muted;
        margin-bottom: 3;
        width: 60;
    }
    
    .empty-state-actions {
        layout: horizontal;
        height: auto;
        align: center middle;
        margin-top: 2;
    }
    
    .empty-state-actions Button {
        margin: 0 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        # ASCII art book icon
        book_art = """
    📚 📖 📚
 ╭─────────╮
 │ CHATBOOK│
 │ LIBRARY │
 ╰─────────╯
        """
        yield Static(book_art, classes="empty-state-icon")
        yield Static("No Chatbooks Yet", classes="empty-state-title")
        yield Static(
            "Chatbooks are portable knowledge packs containing conversations, notes, "
            "characters, and media. Create your first one to get started!",
            classes="empty-state-description"
        )
        
        with Container(classes="empty-state-actions"):
            yield Button("✨ Create New", id="empty-create-btn", variant="primary")
            yield Button("📥 Import", id="empty-import-btn", variant="default")
            yield Button("📋 Browse Templates", id="empty-templates-btn", variant="default")


class ChatbooksWindow(Container):
    """Enhanced Chatbooks management interface."""
    
    BINDINGS = [
        ("c", "create_chatbook", "Create"),
        ("i", "import_chatbook", "Import"),
        ("t", "browse_templates", "Templates"),
        ("m", "manage_exports", "Manage"),
        ("r", "refresh", "Refresh"),
        ("escape", "close", "Close")
    ]
    
    DEFAULT_CSS = """
    ChatbooksWindow {
        layout: vertical;
        background: $background;
    }
    
    .chatbooks-header {
        height: auto;
        background: $boost;
        padding: 2;
        border-bottom: thick $background-darken-1;
    }
    
    .chatbooks-title {
        text-style: bold;
        text-align: center;
        color: $primary;
        margin-bottom: 1;
    }
    
    .chatbooks-subtitle {
        text-align: center;
        color: $text-muted;
    }
    
    .quick-actions {
        height: auto;
        padding: 2;
        background: $panel;
        border-bottom: solid $background-darken-1;
    }
    
    .quick-actions-grid {
        layout: grid;
        grid-size: 4 1;
        grid-gutter: 2;
        height: auto;
        margin-bottom: 1;
    }
    
    .action-card {
        padding: 2;
        background: $boost;
        border: round $background-darken-1;
        align: center middle;
        height: 8;
    }
    
    .action-card:hover {
        background: $primary 20%;
        border: round $primary;
    }
    
    .action-icon {
        text-align: center;
        margin-bottom: 1;
    }
    
    .action-label {
        text-align: center;
        text-style: bold;
    }
    
    .search-bar {
        height: 3;
        padding: 0 2;
        margin-top: 1;
    }
    
    .search-input {
        width: 100%;
    }
    
    .content-area {
        height: 1fr;
        padding: 2;
    }
    
    .section-header {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
        align: left middle;
    }
    
    .section-title {
        text-style: bold;
        color: $primary;
        width: 1fr;
    }
    
    .view-toggles {
        layout: horizontal;
        width: auto;
    }
    
    .view-toggle {
        margin-left: 1;
        min-width: 8;
    }
    
    .chatbooks-grid {
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
        height: auto;
        padding: 1;
    }
    
    .chatbooks-list {
        height: 100%;
        padding: 1;
    }
    
    .stats-bar {
        dock: bottom;
        height: 3;
        background: $panel;
        border-top: solid $background-darken-1;
        padding: 1;
        align: center middle;
    }
    
    .stats-text {
        text-align: center;
        color: $text-muted;
    }
    """
    
    # Reactive properties
    chatbooks = reactive([], recompose=True)
    view_mode = reactive("grid")
    search_query = reactive("")
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance_instance = app_instance
        
        # Get export path from config or use default
        config = self.app_instance_instance.app_config.get("chatbooks", {})
        self._export_path = Path(config.get("export_directory", "~/Documents/Chatbooks")).expanduser()
        self._export_path.mkdir(parents=True, exist_ok=True)
        
    def compose(self) -> ComposeResult:
        # Header
        with Container(classes="chatbooks-header"):
            yield Static("📚 Chatbooks", classes="chatbooks-title")
            yield Static(
                "Create and manage portable knowledge packs",
                classes="chatbooks-subtitle"
            )
        
        # Quick actions
        with Container(classes="quick-actions"):
            with Grid(classes="quick-actions-grid"):
                # Create card
                with Container(classes="action-card", id="create-action"):
                    yield Static("✨", classes="action-icon")
                    yield Static("Create New", classes="action-label")
                
                # Import card
                with Container(classes="action-card", id="import-action"):
                    yield Static("📥", classes="action-icon")
                    yield Static("Import", classes="action-label")
                
                # Templates card
                with Container(classes="action-card", id="templates-action"):
                    yield Static("📋", classes="action-icon")
                    yield Static("Templates", classes="action-label")
                
                # Manage card
                with Container(classes="action-card", id="manage-action"):
                    yield Static("⚙️", classes="action-icon")
                    yield Static("Manage", classes="action-label")
            
            # Search bar
            with Container(classes="search-bar"):
                yield Input(
                    placeholder="🔍 Search chatbooks...",
                    id="chatbook-search",
                    classes="search-input"
                )
        
        # Content area
        with VerticalScroll(classes="content-area"):
            # Section header
            with Container(classes="section-header"):
                yield Static("Recent Chatbooks", classes="section-title", id="section-title")
                
                # View mode toggles
                with Container(classes="view-toggles"):
                    yield Button("▦ Grid", id="view-grid", classes="view-toggle", variant="primary")
                    yield Button("☰ List", id="view-list", classes="view-toggle")
            
            # Content container (will be populated based on chatbooks)
            yield Container(id="chatbooks-container")
        
        # Stats bar
        with Container(classes="stats-bar"):
            yield Static(
                "0 chatbooks • 0 MB total",
                id="stats-text",
                classes="stats-text"
            )
    
    async def on_mount(self) -> None:
        """Called when screen is mounted."""
        await self._refresh_chatbooks()
        
    def watch_chatbooks(self, old_value: List[Dict], new_value: List[Dict]) -> None:
        """React to chatbooks list changes."""
        self._update_content()
        self._update_stats()
        
    def watch_view_mode(self, old_value: str, new_value: str) -> None:
        """React to view mode changes."""
        # Update button states
        grid_btn = self.query_one("#view-grid", Button)
        list_btn = self.query_one("#view-list", Button)
        
        if new_value == "grid":
            grid_btn.variant = "primary"
            list_btn.variant = "default"
        else:
            grid_btn.variant = "default"
            list_btn.variant = "primary"
            
        self._update_content()
        
    def watch_search_query(self, old_value: str, new_value: str) -> None:
        """React to search query changes."""
        self._update_content()
        
    def _update_content(self) -> None:
        """Update the content display."""
        container = self.query_one("#chatbooks-container", Container)
        container.remove_children()
        
        # Filter chatbooks based on search
        filtered = self._filter_chatbooks()
        
        if not filtered:
            # Show empty state
            container.mount(EmptyStateWidget())
            self.query_one("#section-title", Static).update(
                "No chatbooks found" if self.search_query else "Recent Chatbooks"
            )
        else:
            self.query_one("#section-title", Static).update(
                f"Found {len(filtered)} chatbooks" if self.search_query else "Recent Chatbooks"
            )
            
            if self.view_mode == "grid":
                # Grid view
                grid = Grid(classes="chatbooks-grid")
                for cb_data in filtered:
                    card = ChatbookCard(cb_data)
                    grid.mount(card)
                container.mount(grid)
            else:
                # List view
                list_view = ListView(classes="chatbooks-list")
                for cb_data in filtered:
                    item = ListItem(
                        Static(f"📚 {cb_data['name']} - {cb_data.get('description', 'No description')[:50]}...")
                    )
                    list_view.mount(item)
                container.mount(list_view)
                
    def _filter_chatbooks(self) -> List[Dict[str, Any]]:
        """Filter chatbooks based on search query."""
        if not self.search_query:
            return self.chatbooks
            
        query = self.search_query.lower()
        return [
            cb for cb in self.chatbooks
            if query in cb.get('name', '').lower() or
               query in cb.get('description', '').lower() or
               any(query in tag.lower() for tag in cb.get('tags', []))
        ]
        
    def _update_stats(self) -> None:
        """Update statistics display."""
        total_size = sum(cb.get('size_mb', 0) for cb in self.chatbooks)
        stats_text = f"{len(self.chatbooks)} chatbooks • {total_size:.1f} MB total"
        self.query_one("#stats-text", Static).update(stats_text)
        
    async def _refresh_chatbooks(self) -> None:
        """Load chatbooks from export directory."""
        try:
            if not self._export_path.exists():
                self._export_path.mkdir(parents=True, exist_ok=True)
                
            chatbooks = []
            
            # Scan for .zip files
            for zip_file in self._export_path.glob("*.zip"):
                try:
                    # Get basic info
                    cb_info = {
                        'name': zip_file.stem,
                        'path': str(zip_file),
                        'size_mb': zip_file.stat().st_size / (1024 * 1024),
                        'created_at': datetime.fromtimestamp(zip_file.stat().st_ctime).isoformat()
                    }
                    
                    # Try to read manifest for more info
                    try:
                        with zipfile.ZipFile(zip_file, 'r') as zf:
                            if 'manifest.json' in zf.namelist():
                                manifest_data = json.loads(zf.read('manifest.json'))
                                cb_info.update({
                                    'name': manifest_data.get('name', cb_info['name']),
                                    'description': manifest_data.get('description', ''),
                                    'tags': manifest_data.get('tags', []),
                                    'statistics': manifest_data.get('statistics', {
                                        'conversations': 0,
                                        'notes': 0,
                                        'characters': 0
                                    })
                                })
                    except:
                        pass
                        
                    chatbooks.append(cb_info)
                except Exception as e:
                    logger.error(f"Error reading chatbook {zip_file}: {e}")
                    
            # Sort by created date, newest first
            chatbooks.sort(key=lambda x: x['created_at'], reverse=True)
            self.chatbooks = chatbooks
            
        except Exception as e:
            logger.error(f"Error refreshing chatbooks: {e}")
            self.app_instance.notify(f"Error loading chatbooks: {str(e)}", severity="error")
            
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id in ["view-grid", "view-list"]:
            self.view_mode = "grid" if button_id == "view-grid" else "list"
        elif button_id in ["empty-create-btn"]:
            await self.action_create_chatbook()
        elif button_id in ["empty-import-btn"]:
            await self.action_import_chatbook()
        elif button_id in ["empty-templates-btn"]:
            await self.action_browse_templates()
            
    async def on_container_click(self, event) -> None:
        """Handle container clicks for action cards."""
        # Check if clicked element or parent is an action card
        element = event.target
        while element and element.id not in ["create-action", "import-action", "templates-action", "manage-action"]:
            element = element.parent
            
        if element:
            if element.id == "create-action":
                await self.action_create_chatbook()
            elif element.id == "import-action":
                await self.action_import_chatbook()
            elif element.id == "templates-action":
                await self.action_browse_templates()
            elif element.id == "manage-action":
                await self.action_manage_exports()
                
    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "chatbook-search":
            self.search_query = event.value
            
    async def action_create_chatbook(self) -> None:
        """Launch the chatbook creation wizard."""
        from .Wizards.ChatbookCreationWizard import ChatbookCreationWizard
        
        wizard = ChatbookCreationWizard(self.app_instance)
        result = await self.app_instance.push_screen(wizard, wait_for_dismiss=True)
        
        if result and result.get("success"):
            self.app_instance.notify("Chatbook created successfully!", severity="success")
            await self._refresh_chatbooks()
            
    async def action_import_chatbook(self) -> None:
        """Launch the chatbook import wizard."""
        from .Wizards.ChatbookImportWizard import ChatbookImportWizard
        
        wizard = ChatbookImportWizard(self.app_instance)
        result = await self.app_instance.push_screen(wizard, wait_for_dismiss=True)
        
        if result and result.get("success"):
            self.app_instance.notify("Chatbook imported successfully!", severity="success")
            await self._refresh_chatbooks()
            
    async def action_browse_templates(self) -> None:
        """Open the templates browser."""
        from .ChatbookTemplatesWindow import ChatbookTemplatesWindow
        
        templates_window = ChatbookTemplatesWindow(self.app_instance)
        result = await self.app_instance.push_screen(templates_window, wait_for_dismiss=True)
        
        if result:
            # User selected a template, launch creation wizard with pre-filled data
            from .Wizards.ChatbookCreationWizard import ChatbookCreationWizard
            
            # Create wizard with template data
            wizard = ChatbookCreationWizard(self.app_instance)
            # TODO: Pass template data to wizard
            create_result = await self.app_instance.push_screen(wizard, wait_for_dismiss=True)
            
            if create_result and create_result.get("success"):
                self.app_instance.notify("Chatbook created from template!", severity="success")
                await self._refresh_chatbooks()
                
    async def action_manage_exports(self) -> None:
        """Open the export management window."""
        from .ChatbookExportManagementWindow import ChatbookExportManagementWindow
        
        management_window = ChatbookExportManagementWindow(self.app_instance)
        await self.app_instance.push_screen(management_window, wait_for_dismiss=True)
        
        # Refresh after management in case anything was deleted
        await self._refresh_chatbooks()
        
    async def action_refresh(self) -> None:
        """Refresh the chatbooks list."""
        await self._refresh_chatbooks()
        self.app_instance.notify("Chatbooks refreshed", severity="info")
        
    async def action_close(self) -> None:
        """Close the chatbooks window."""
        # This is a container, not a screen, so we don't dismiss
        pass
        
#
# End of Chatbooks_Window.py
#######################################################################################################################