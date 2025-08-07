# Chatbooks_Window.py
# Description: Chatbooks management interface with proper layout
#
"""
Chatbooks Window
----------------

A proper chatbooks management interface with:
- Left sidebar showing available chatbooks
- Header with action buttons
- Dynamic content area based on selected action
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import json
import zipfile
import traceback

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll, ScrollableContainer
from textual.widgets import Static, Button, Label, ListView, ListItem
from textual.reactive import reactive
from textual.widget import Widget
from loguru import logger

if TYPE_CHECKING:
    from ..app import TldwCli


class ChatbookListItem(ListItem):
    """A list item representing a chatbook."""
    
    def __init__(self, chatbook_data: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.chatbook_data = chatbook_data
        

class ChatbooksWindow(Widget):
    """Chatbooks management interface."""
    
    DEFAULT_CSS = """
    ChatbooksWindow {
        layout: vertical;
        height: 100%;
    }
    """
    
    BINDINGS = [
        ("c", "create_chatbook", "Create"),
        ("l", "load_chatbook", "Load"),
        ("t", "modify_templates", "Templates"),
        ("r", "refresh", "Refresh"),
        ("escape", "close", "Close")
    ]
    
    # Reactive properties
    chatbooks = reactive([], recompose=False)
    selected_chatbook = reactive(None)
    current_view = reactive("welcome")  # welcome, create, load, templates
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        
        # Get export path from config or use default
        config = self.app_instance.app_config.get("chatbooks", {})
        self._export_path = Path(config.get("export_directory", "~/Documents/Chatbooks")).expanduser()
        self._export_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ChatbooksWindow.__init__: Created")
        
    def compose(self) -> ComposeResult:
        """Compose the UI."""
        logger.info(f"ChatbooksWindow.compose() called")
        
        # Main horizontal layout
        yield Horizontal(
            Container(
                Static("📚 Available Chatbooks", classes="sidebar-title"),
                ListView(id="chatbooks-list"),
                id="chatbooks-sidebar"
            ),
            Container(
                Container(
                    Static("📚 Chatbooks Manager", classes="header-title"),
                    Static("Create and manage portable knowledge packs", classes="header-subtitle"),
                    Horizontal(
                        Button("✨ Create New Chatbook", id="btn-create", variant="primary"),
                        Button("📥 Load Existing Chatbook", id="btn-load"),
                        Button("📋 Modify Templates", id="btn-templates"),
                        id="action-buttons"
                    ),
                    id="chatbooks-header"
                ),
                Container(id="dynamic-content"),
                id="content-area"
            ),
            id="main-layout"
        )
        
        logger.info("ChatbooksWindow.compose() completed")
    
    async def on_mount(self) -> None:
        """Called when widget is mounted."""
        logger.info("ChatbooksWindow.on_mount() called")
        await self._refresh_chatbooks()
        self._update_view()
    
    def on_show(self) -> None:
        """Called when widget becomes visible."""
        logger.info("ChatbooksWindow.on_show() called")
        self.call_after_refresh(self._refresh_chatbooks)
    
    def watch_current_view(self, old_value: str, new_value: str) -> None:
        """React to view changes."""
        logger.info(f"View changed from {old_value} to {new_value}")
        self._update_view()
    
    def watch_chatbooks(self, old_value: List[Dict], new_value: List[Dict]) -> None:
        """React to chatbooks list changes."""
        logger.info(f"Chatbooks updated: {len(new_value)} chatbooks")
        self._update_chatbooks_list()
    
    def _update_view(self) -> None:
        """Update the dynamic content area based on current view."""
        try:
            content = self.query_one("#dynamic-content", Container)
            
            # Clear existing content
            content.remove_children()
            
            if self.current_view == "welcome":
                # Welcome view
                welcome = Static(
                    "Welcome to Chatbooks Manager!\n\n"
                    "• Click 'Create New Chatbook' to start a new knowledge pack\n"
                    "• Click 'Load Existing Chatbook' to manage existing chatbooks\n"
                    "• Click 'Modify Templates' to customize chatbook templates\n"
                    "• Select a chatbook from the left sidebar to view it",
                    classes="welcome-text"
                )
                content.mount(welcome)
                
            elif self.current_view == "create":
                # Create view - will launch wizard
                creating = Static("Creating new chatbook...", classes="status-text")
                content.mount(creating)
                self.action_create_chatbook()
                
            elif self.current_view == "load":
                # Load/manage view
                if self.selected_chatbook:
                    # Show chatbook details and management options
                    cb = self.selected_chatbook
                    details = Container(
                        Static(f"📚 {cb.get('name', 'Untitled')}", classes="chatbook-title"),
                        Static(f"Description: {cb.get('description', 'No description')}", classes="chatbook-desc"),
                        Static(f"Size: {cb.get('size_mb', 0):.1f} MB", classes="chatbook-size"),
                        Static(f"Created: {cb.get('created_at', 'Unknown')}", classes="chatbook-date"),
                        Container(
                            Button("Edit Metadata", id="btn-edit-meta"),
                            Button("Add Items", id="btn-add-items"),
                            Button("Remove Items", id="btn-remove-items"),
                            Button("Export", id="btn-export"),
                            Button("Delete", id="btn-delete", variant="error"),
                            classes="management-buttons"
                        ),
                        classes="chatbook-details"
                    )
                    content.mount(details)
                else:
                    no_selection = Static(
                        "Select a chatbook from the left sidebar to manage it",
                        classes="info-text"
                    )
                    content.mount(no_selection)
                    
            elif self.current_view == "templates":
                # Templates view
                self._show_templates_view(content)
                
        except Exception as e:
            logger.error(f"Error updating view: {e}")
    
    def _update_chatbooks_list(self) -> None:
        """Update the chatbooks list in the sidebar."""
        try:
            list_view = self.query_one("#chatbooks-list", ListView)
            list_view.clear()
            
            for i, cb in enumerate(self.chatbooks):
                name = cb.get('name', 'Untitled')
                size = cb.get('size_mb', 0)
                # Use index-based ID to avoid path issues
                safe_id = f"cb_{i}"
                item = ListItem(
                    Static(f"📚 {name} ({size:.1f} MB)"),
                    id=safe_id
                )
                list_view.append(item)
                
        except Exception as e:
            logger.error(f"Error updating chatbooks list: {e}")
    
    async def _refresh_chatbooks(self) -> None:
        """Load chatbooks from export directory."""
        logger.info(f"_refresh_chatbooks: Starting, export_path={self._export_path}")
        try:
            if not self._export_path.exists():
                self._export_path.mkdir(parents=True, exist_ok=True)
                
            chatbooks = []
            
            # Scan for .zip files
            for zip_file in self._export_path.glob("*.zip"):
                try:
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
                                })
                    except:
                        pass
                        
                    chatbooks.append(cb_info)
                except Exception as e:
                    logger.error(f"Error reading chatbook {zip_file}: {e}")
                    
            # Sort by created date, newest first
            chatbooks.sort(key=lambda x: x['created_at'], reverse=True)
            
            logger.info(f"_refresh_chatbooks: Found {len(chatbooks)} chatbooks")
            self.chatbooks = chatbooks
            
        except Exception as e:
            logger.error(f"Error refreshing chatbooks: {e}", exc_info=True)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        logger.info(f"Button pressed: {button_id}")
        
        if button_id == "btn-create":
            self.current_view = "create"
        elif button_id == "btn-load":
            self.current_view = "load"
        elif button_id == "btn-templates":
            self.current_view = "templates"
        elif button_id == "btn-use-template":
            await self._use_selected_template()
        elif button_id == "btn-browse-templates":
            await self._browse_templates()
        elif button_id == "btn-create-template":
            await self._create_custom_template()
        elif button_id == "btn-edit-template":
            await self._edit_template()
        elif button_id == "btn-delete-template":
            await self._delete_template()
        
        event.stop()
    
    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle chatbook selection from list."""
        if event.item.id and event.item.id.startswith("cb_"):
            # Get the index from the ID
            try:
                index = int(event.item.id[3:])  # Remove "cb_" prefix
                if 0 <= index < len(self.chatbooks):
                    self.selected_chatbook = self.chatbooks[index]
                    self.current_view = "load"
            except (ValueError, IndexError):
                logger.error(f"Invalid chatbook selection: {event.item.id}")
    
    def action_create_chatbook(self) -> None:
        """Launch the chatbook creation wizard."""
        self.run_worker(self._create_chatbook_async())
    
    async def _create_chatbook_async(self) -> None:
        """Async worker for creating chatbook."""
        try:
            from .Wizards.ChatbookCreationWizard import ChatbookCreationWizard
            
            wizard = ChatbookCreationWizard(self.app_instance)
            result = await self.app.push_screen(wizard, wait_for_dismiss=True)
            
            if result and result.get("success"):
                self.app.notify("Chatbook created successfully!", severity="success")
                await self._refresh_chatbooks()
                self.current_view = "welcome"
            else:
                self.current_view = "welcome"
        except Exception as e:
            logger.error(f"Error creating chatbook: {e}")
            self.app.notify(f"Error: {str(e)}", severity="error")
            self.current_view = "welcome"
    
    def _show_templates_view(self, content: Container) -> None:
        """Show the templates management view."""
        try:
            # Templates header
            header = Static(
                "📋 Template Management",
                classes="chatbook-title"
            )
            content.mount(header)
            
            # Description
            desc = Static(
                "Chatbook templates help you quickly create new chatbooks with predefined structures.",
                classes="chatbook-desc"
            )
            content.mount(desc)
            
            # Pre-defined templates list
            templates_label = Static("\n📚 Available Templates:", classes="chatbook-title")
            content.mount(templates_label)
            
            # Template list container
            templates_container = Container(classes="templates-list")
            
            # Add pre-defined templates
            templates_data = [
                ("🔬 Research Project", "Organize research conversations, notes, and references"),
                ("✍️ Creative Writing", "Bundle character profiles, story notes, and world-building"),
                ("📚 Learning Journey", "Track learning progress with conversations and study materials"),
                ("📋 Project Documentation", "Document project decisions and technical notes"),
                ("🤖 Personal Assistant", "Export AI assistant conversations and custom prompts"),
                ("🧠 Knowledge Base", "Create a comprehensive knowledge repository")
            ]
            
            for title, description in templates_data:
                template_item = Container(
                    Static(title, classes="template-name"),
                    Static(description, classes="template-description"),
                    classes="template-item"
                )
                templates_container.mount(template_item)
            
            content.mount(templates_container)
            
            # Action buttons for templates
            buttons_container = Container(
                Button("📥 Use Template", id="btn-use-template", variant="primary"),
                Button("🔍 Browse All Templates", id="btn-browse-templates"),
                Button("➕ Create Custom Template", id="btn-create-template"),
                Button("✏️ Edit Template", id="btn-edit-template"),
                Button("🗑️ Delete Template", id="btn-delete-template", variant="error"),
                classes="management-buttons"
            )
            content.mount(buttons_container)
            
        except Exception as e:
            logger.error(f"Error showing templates view: {e}")
    
    async def _use_selected_template(self) -> None:
        """Use the selected template to create a new chatbook."""
        try:
            from .ChatbookTemplatesWindow import ChatbookTemplatesWindow
            
            template_window = ChatbookTemplatesWindow(self.app_instance)
            template = await self.app.push_screen(template_window, wait_for_dismiss=True)
            
            if template:
                self.app.notify(f"Using template: {template.name}", severity="info")
                # Launch creation wizard with template
                await self._create_chatbook_with_template(template)
                
        except Exception as e:
            logger.error(f"Error using template: {e}")
            self.app.notify(f"Error: {str(e)}", severity="error")
    
    async def _browse_templates(self) -> None:
        """Browse all available templates."""
        try:
            from .ChatbookTemplatesWindow import ChatbookTemplatesWindow
            
            template_window = ChatbookTemplatesWindow(self.app_instance)
            template = await self.app.push_screen(template_window, wait_for_dismiss=True)
            
            if template:
                self.app.notify(f"Selected template: {template.name}", severity="info")
                
        except Exception as e:
            logger.error(f"Error browsing templates: {e}")
            self.app.notify(f"Error: {str(e)}", severity="error")
    
    async def _create_custom_template(self) -> None:
        """Create a new custom template."""
        self.app.notify("Custom template creation coming soon!", severity="info")
    
    async def _edit_template(self) -> None:
        """Edit an existing template."""
        self.app.notify("Template editing coming soon!", severity="info")
    
    async def _delete_template(self) -> None:
        """Delete a template."""
        self.app.notify("Template deletion coming soon!", severity="info")
    
    async def _create_chatbook_with_template(self, template) -> None:
        """Create a chatbook using the selected template."""
        try:
            from .Wizards.ChatbookCreationWizard import ChatbookCreationWizard
            
            # Pass template to wizard
            wizard = ChatbookCreationWizard(self.app_instance, template=template)
            result = await self.app.push_screen(wizard, wait_for_dismiss=True)
            
            if result and result.get("success"):
                self.app.notify(f"Chatbook created from template: {template.name}", severity="success")
                await self._refresh_chatbooks()
                self.current_view = "welcome"
                
        except Exception as e:
            logger.error(f"Error creating chatbook with template: {e}")
            self.app.notify(f"Error: {str(e)}", severity="error")
    
    async def action_refresh(self) -> None:
        """Refresh the chatbooks list."""
        await self._refresh_chatbooks()
        self.app.notify("Chatbooks refreshed", severity="info")

#
# End of Chatbooks_Window.py
#