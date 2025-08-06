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
import traceback

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll, Grid
from textual.widgets import Static, Button, Label, Input, ListView, ListItem
from textual.reactive import reactive
from loguru import logger

if TYPE_CHECKING:
    from ..app import TldwCli


class ChatbookCard(Container):
    """Card widget for displaying a chatbook."""
    
    # CSS moved to external file: css/features/_chatbooks.tcss
    
    def __init__(self, chatbook_data: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.chatbook_data = chatbook_data
        
    def compose(self) -> ComposeResult:
        yield Static(self.chatbook_data.get('name', 'Untitled'), classes="chatbook-card-title")
        
        description = self.chatbook_data.get('description', 'No description')
        if len(description) > 100:
            description = description[:100] + '...'
        yield Static(description, classes="chatbook-card-description")
        
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
    """Enhanced empty state widget with better layout."""
    
    # CSS moved to external file: css/features/_chatbooks.tcss
    
    def compose(self) -> ComposeResult:
        with Container(classes="empty-state-container"):
            # Larger ASCII art
            book_art = """
📚 Welcome to Chatbooks 📚

╔════════════════════════╗
║                        ║
║     CHATBOOK           ║
║     LIBRARY            ║
║                        ║
╚════════════════════════╝
            """
            yield Static(book_art, classes="empty-state-icon")
            yield Static("Start Your Knowledge Collection", classes="empty-state-title")
            yield Static(
                "Chatbooks are portable knowledge packs that bundle conversations, notes, "
                "characters, and media into shareable archives. Perfect for research projects, "
                "story development, or knowledge sharing.",
                classes="empty-state-description"
            )
            
            with Container(classes="empty-state-cards"):
                # Primary action - Create New
                yield Button(
                    "✨\nCreate New Chatbook\nStart from scratch with a wizard",
                    id="empty-create-btn",
                    classes="primary-action-button",
                    variant="primary"
                )
                
                # Secondary actions
                with Container(classes="secondary-actions"):
                    yield Button(
                        "📥\nImport\nFrom .zip file",
                        id="empty-import-btn",
                        classes="secondary-action-button"
                    )
                    
                    yield Button(
                        "📋\nTemplates\nPre-made configs",
                        id="empty-templates-btn",
                        classes="secondary-action-button"
                    )


class ChatbooksWindow(Container):
    """Enhanced Chatbooks management interface."""
    
    DEFAULT_CSS = """
    ChatbooksWindow {
        layout: vertical;
        height: 100%;
        width: 100%;
        background: $background;
        display: block !important;
    }
    
    .chatbooks-header {
        height: auto;
        background: $boost;
        padding: 2;
        border-bottom: thick $background-darken-1;
    }
    
    #chatbooks-main-content {
        height: 1fr;
        width: 100%;
        background: $background;
    }
    """
    
    BINDINGS = [
        ("c", "create_chatbook", "Create"),
        ("i", "import_chatbook", "Import"),
        ("t", "browse_templates", "Templates"),
        ("m", "manage_exports", "Manage"),
        ("r", "refresh", "Refresh"),
        ("escape", "close", "Close")
    ]
    
    # Reactive properties
    chatbooks = reactive([], recompose=False)  # Don't recompose, manually update instead
    view_mode = reactive("grid")
    search_query = reactive("")
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._initialized = False  # Track if we've initialized the content
        
        # Get export path from config or use default
        config = self.app_instance.app_config.get("chatbooks", {})
        self._export_path = Path(config.get("export_directory", "~/Documents/Chatbooks")).expanduser()
        self._export_path.mkdir(parents=True, exist_ok=True)
        
        # Log the actual classes being applied
        logger.debug(f"ChatbooksWindow initialized with classes: {self.classes}")
        
    def compose(self) -> ComposeResult:
        logger.debug("ChatbooksWindow.compose() called")
        logger.debug(f"ChatbooksWindow display property: {self.display}")
        logger.debug(f"ChatbooksWindow styles.display: {self.styles.display}")
        
        # Always show header
        with Container(classes="chatbooks-header"):
            yield Static("📚 Chatbooks", classes="chatbooks-title")
            yield Static(
                "Create and manage portable knowledge packs",
                classes="chatbooks-subtitle"
            )
        
        # Main content container that will be updated based on state
        yield Container(id="chatbooks-main-content")
        
        logger.debug(f"Created main_content container with id='chatbooks-main-content'")
    
    async def on_mount(self) -> None:
        """Called when screen is mounted."""
        logger.debug(f"ChatbooksWindow.on_mount() called, display={self.display}")
        # Since we're created with display=False, on_mount might not trigger initialization
        # We'll handle it when the window becomes visible
        if self.display:
            await self._ensure_initialized()
    
    def on_show(self) -> None:
        """Called when widget becomes visible."""
        logger.debug("ChatbooksWindow.on_show() called")
        if not self._initialized:
            self.call_after_refresh(self._ensure_initialized)
    
    async def _ensure_initialized(self) -> None:
        """Ensure the chatbooks are loaded."""
        if not self._initialized:
            logger.debug("Initializing ChatbooksWindow content...")
            self._initialized = True
            await self._refresh_chatbooks()
        
    def watch_chatbooks(self, old_value: List[Dict], new_value: List[Dict]) -> None:
        """React to chatbooks list changes."""
        self._rebuild_layout()
        
    def watch_view_mode(self, old_value: str, new_value: str) -> None:
        """React to view mode changes."""
        if self.chatbooks:  # Only update if we have chatbooks
            self._update_chatbook_display()
        
    def watch_search_query(self, old_value: str, new_value: str) -> None:
        """React to search query changes."""
        if self.chatbooks:  # Only update if we have chatbooks
            self._update_chatbook_display()
        
    def _rebuild_layout(self) -> None:
        """Rebuild the entire layout based on whether chatbooks exist."""
        try:
            logger.debug(f"_rebuild_layout: Starting with {len(self.chatbooks)} chatbooks")
            main_content = self.query_one("#chatbooks-main-content", Container)
            main_content.remove_children()
            
            if not self.chatbooks:
                # Show empty state
                logger.debug("_rebuild_layout: Showing empty state")
                main_content.mount(EmptyStateWidget())
            else:
                # Show full interface with toolbar and content
                logger.debug("_rebuild_layout: Building toolbar")
                
                # Build the complete toolbar structure
                toolbar = Container(classes="chatbooks-toolbar")
                toolbar_row = Container(classes="toolbar-row")
                toolbar_actions = Container(classes="toolbar-actions")
                
                # Create buttons
                create_btn = Button("✨ Create", id="toolbar-create", classes="toolbar-button", variant="primary")
                import_btn = Button("📥 Import", id="toolbar-import", classes="toolbar-button")
                templates_btn = Button("📋 Templates", id="toolbar-templates", classes="toolbar-button")
                manage_btn = Button("⚙️ Manage", id="toolbar-manage", classes="toolbar-button")
                
                # Create search
                search_input = Input(
                    placeholder="🔍 Search chatbooks...",
                    id="chatbook-search",
                    classes="search-input"
                )
                search_container = Container(search_input, classes="search-container")
                spacer = Container(classes="toolbar-spacer")
                
                # Mount toolbar to main_content first, then build its children
                main_content.mount(toolbar)
                toolbar.mount(toolbar_row)
                toolbar_row.mount(toolbar_actions, spacer, search_container)
                toolbar_actions.mount(create_btn, import_btn, templates_btn, manage_btn)
                
                # Build and mount content area
                content_area = VerticalScroll(classes="content-area")
                main_content.mount(content_area)
                
                # Header with view toggles
                header = Container(classes="content-header")
                title_static = Static("Your Chatbooks", classes="content-title", id="content-title")
                toggles = Container(classes="view-toggles")
                grid_btn = Button("▦ Grid", id="view-grid", classes="view-toggle", variant="primary")
                list_btn = Button("☰ List", id="view-list", classes="view-toggle")
                
                content_area.mount(header)
                header.mount(title_static, toggles)
                toggles.mount(grid_btn, list_btn)
                
                # Chatbook display container
                display_container = Container(id="chatbook-display")
                content_area.mount(display_container)
                
                # Stats bar
                stats = Container(classes="stats-bar")
                stats_text = Static("", id="stats-text", classes="stats-text")
                main_content.mount(stats)
                stats.mount(stats_text)
                
                # Update the display
                logger.debug("_rebuild_layout: Updating chatbook display")
                self._update_chatbook_display()
                logger.debug("_rebuild_layout: Updating stats")
                self._update_stats()
                logger.debug("_rebuild_layout: Complete")
        except Exception as e:
            logger.error(f"Error in _rebuild_layout: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
    
    def _update_chatbook_display(self) -> None:
        """Update just the chatbook display area."""
        try:
            logger.debug(f"_update_chatbook_display: Starting with {len(self.chatbooks)} chatbooks")
            display = self.query_one("#chatbook-display", Container)
            display.remove_children()
            
            # Filter chatbooks
            filtered = self._filter_chatbooks()
            logger.debug(f"_update_chatbook_display: After filtering, have {len(filtered)} chatbooks")
            
            if not filtered and self.search_query:
                # No search results
                logger.debug("_update_chatbook_display: No search results")
                display.mount(Static(
                    f"No chatbooks found matching '{self.search_query}'",
                    classes="no-results"
                ))
            else:
                # Update title
                title = self.query_one("#content-title", Static)
                if self.search_query:
                    title.update(f"Search Results ({len(filtered)} found)")
                else:
                    title.update(f"Your Chatbooks ({len(filtered)})")
                
                # Update view toggle states
                grid_btn = self.query_one("#view-grid", Button)
                list_btn = self.query_one("#view-list", Button)
                
                if self.view_mode == "grid":
                    grid_btn.variant = "primary"
                    list_btn.variant = "default"
                else:
                    grid_btn.variant = "default" 
                    list_btn.variant = "primary"
                
                # Display chatbooks
                logger.debug(f"_update_chatbook_display: View mode is {self.view_mode}")
                if self.view_mode == "grid":
                    logger.debug(f"_update_chatbook_display: Creating grid with {len(filtered)} items")
                    grid = Grid(classes="chatbooks-grid")
                    # Mount grid first, then add cards
                    display.mount(grid)
                    for cb_data in filtered:
                        logger.debug(f"_update_chatbook_display: Adding card for {cb_data.get('name', 'Unknown')}")
                        card = ChatbookCard(cb_data)
                        grid.mount(card)
                else:
                    logger.debug(f"_update_chatbook_display: Creating list with {len(filtered)} items")
                    list_view = ListView(classes="chatbooks-list")
                    # Mount list first, then add items
                    display.mount(list_view)
                    for cb_data in filtered:
                        desc = cb_data.get('description', 'No description')
                        if len(desc) > 50:
                            desc = desc[:50] + '...'
                        logger.debug(f"_update_chatbook_display: Adding list item for {cb_data.get('name', 'Unknown')}")
                        item = ListItem(
                            Static(f"📚 {cb_data['name']} - {desc}")
                        )
                        list_view.mount(item)
                logger.debug("_update_chatbook_display: Display updated successfully")
        except Exception as e:
            logger.error(f"Error in _update_chatbook_display: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
                
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
        try:
            total_size = sum(cb.get('size_mb', 0) for cb in self.chatbooks)
            stats_text = f"{len(self.chatbooks)} chatbooks • {total_size:.1f} MB total"
            stats_widget = self.query_one("#stats-text", Static)
            stats_widget.update(stats_text)
        except Exception:
            # Stats element may not exist yet during initial mount
            pass
        
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
            logger.error(f"Error refreshing chatbooks: {e}", exc_info=True)
            self.app.notify(f"Error loading chatbooks: {str(e)}", severity="error")
            
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        # View mode toggles
        if button_id in ["view-grid", "view-list"]:
            self.view_mode = "grid" if button_id == "view-grid" else "list"
            event.stop()
        
        # Empty state buttons
        elif button_id == "empty-create-btn":
            self.action_create_chatbook()
            event.stop()
        elif button_id == "empty-import-btn":
            self.action_import_chatbook()
            event.stop()
        elif button_id == "empty-templates-btn":
            self.action_browse_templates()
            event.stop()
        
        # Toolbar buttons
        elif button_id == "toolbar-create":
            self.action_create_chatbook()
            event.stop()
        elif button_id == "toolbar-import":
            self.action_import_chatbook()
            event.stop()
        elif button_id == "toolbar-templates":
            self.action_browse_templates()
            event.stop()
        elif button_id == "toolbar-manage":
            self.action_manage_exports()
            event.stop()
            
                
    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "chatbook-search":
            self.search_query = event.value
            
    def action_create_chatbook(self) -> None:
        """Launch the chatbook creation wizard."""
        self.run_worker(self._create_chatbook_async())
    
    async def _create_chatbook_async(self) -> None:
        """Async worker for creating chatbook."""
        from .Wizards.ChatbookCreationWizard import ChatbookCreationWizard
        
        wizard = ChatbookCreationWizard(self.app_instance)
        result = await self.app.push_screen(wizard, wait_for_dismiss=True)
        
        if result and result.get("success"):
            self.app.notify("Chatbook created successfully!", severity="success")
            await self._refresh_chatbooks()
            
    def action_import_chatbook(self) -> None:
        """Launch the chatbook import wizard."""
        self.run_worker(self._import_chatbook_async())
    
    async def _import_chatbook_async(self) -> None:
        """Async worker for importing chatbook."""
        from .Wizards.ChatbookImportWizard import ChatbookImportWizard
        
        wizard = ChatbookImportWizard(self.app_instance)
        result = await self.app.push_screen(wizard, wait_for_dismiss=True)
        
        if result and result.get("success"):
            self.app.notify("Chatbook imported successfully!", severity="success")
            await self._refresh_chatbooks()
            
    def action_browse_templates(self) -> None:
        """Open the templates browser."""
        self.run_worker(self._browse_templates_async())
    
    async def _browse_templates_async(self) -> None:
        """Async worker for browsing templates."""
        from .ChatbookTemplatesWindow import ChatbookTemplatesWindow
        
        templates_window = ChatbookTemplatesWindow(self.app_instance)
        result = await self.app.push_screen(templates_window, wait_for_dismiss=True)
        
        if result:
            # User selected a template, launch creation wizard with pre-filled data
            from .Wizards.ChatbookCreationWizard import ChatbookCreationWizard
            
            # Create wizard with template data
            wizard = ChatbookCreationWizard(self.app_instance)
            # TODO: Pass template data to wizard
            create_result = await self.app.push_screen(wizard, wait_for_dismiss=True)
            
            if create_result and create_result.get("success"):
                self.app.notify("Chatbook created from template!", severity="success")
                await self._refresh_chatbooks()
                
    def action_manage_exports(self) -> None:
        """Open the export management window."""
        self.run_worker(self._manage_exports_async())
    
    async def _manage_exports_async(self) -> None:
        """Async worker for managing exports."""
        from .ChatbookExportManagementWindow import ChatbookExportManagementWindow
        
        management_window = ChatbookExportManagementWindow(self.app_instance)
        await self.app.push_screen(management_window, wait_for_dismiss=True)
        
        # Refresh after management in case anything was deleted
        await self._refresh_chatbooks()
        
    async def action_refresh(self) -> None:
        """Refresh the chatbooks list."""
        await self._refresh_chatbooks()
        self.app.notify("Chatbooks refreshed", severity="info")
        
    async def action_close(self) -> None:
        """Close the chatbooks window."""
        # This is a container, not a screen, so we don't dismiss
        pass
        
#
# End of Chatbooks_Window.py
#######################################################################################################################