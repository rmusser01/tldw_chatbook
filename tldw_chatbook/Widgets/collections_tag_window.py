# tldw_chatbook/Widgets/collections_tag_window.py
"""
CollectionsTagWindow widget for managing keywords/tags in the media database.
Provides functionality for viewing, editing, merging, and deleting keywords.
"""

from typing import TYPE_CHECKING, List, Dict, Any, Optional
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Button, Label, Input, ListView, ListItem, Markdown
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.message import Message
from loguru import logger

if TYPE_CHECKING:
    from ..app import TldwCli


class KeywordRenameDialog(ModalScreen):
    """Modal dialog for renaming a keyword."""
    
    def __init__(self, keyword: str, keyword_id: int) -> None:
        super().__init__()
        self.keyword = keyword
        self.keyword_id = keyword_id
        
    def compose(self) -> ComposeResult:
        with Container(id="rename-dialog-container"):
            yield Label(f"Rename keyword: {self.keyword}", id="rename-title")
            yield Input(placeholder="New keyword name", id="new-keyword-input", value=self.keyword)
            with Horizontal(id="rename-buttons"):
                yield Button("Cancel", id="cancel-rename", variant="default")
                yield Button("Rename", id="confirm-rename", variant="primary")
                
    @on(Button.Pressed, "#cancel-rename")
    def cancel_rename(self) -> None:
        self.dismiss(None)
        
    @on(Button.Pressed, "#confirm-rename")
    def confirm_rename(self) -> None:
        new_keyword = self.query_one("#new-keyword-input", Input).value.strip()
        if new_keyword and new_keyword != self.keyword:
            self.dismiss((self.keyword_id, new_keyword))
        else:
            self.dismiss(None)


class KeywordMergeDialog(ModalScreen):
    """Modal dialog for merging keywords."""
    
    def __init__(self, source_keywords: List[Dict[str, Any]], all_keywords: List[Dict[str, Any]]) -> None:
        super().__init__()
        self.source_keywords = source_keywords
        self.all_keywords = all_keywords
        
    def compose(self) -> ComposeResult:
        with Container(id="merge-dialog-container"):
            source_names = ", ".join([k['keyword'] for k in self.source_keywords[:5]])
            if len(self.source_keywords) > 5:
                source_names += f" and {len(self.source_keywords) - 5} more"
            yield Label(f"Merge keywords: {source_names}", id="merge-title")
            yield Label("Select target keyword:", id="merge-instruction")
            yield ListView(id="target-keyword-list")
            yield Input(placeholder="Or enter new keyword name", id="new-target-input")
            with Horizontal(id="merge-buttons"):
                yield Button("Cancel", id="cancel-merge", variant="default")
                yield Button("Merge", id="confirm-merge", variant="primary")
                
    def on_mount(self) -> None:
        """Populate the target keyword list."""
        list_view = self.query_one("#target-keyword-list", ListView)
        source_ids = {k['id'] for k in self.source_keywords}
        
        for keyword in self.all_keywords:
            if keyword['id'] not in source_ids:
                item = ListItem(Label(keyword['keyword']))
                item.keyword_data = keyword
                list_view.append(item)
                
    @on(Button.Pressed, "#cancel-merge")
    def cancel_merge(self) -> None:
        self.dismiss(None)
        
    @on(Button.Pressed, "#confirm-merge")
    def confirm_merge(self) -> None:
        # Check if user entered a new keyword
        new_keyword = self.query_one("#new-target-input", Input).value.strip()
        if new_keyword:
            self.dismiss(("new", new_keyword))
            return
            
        # Otherwise get selected keyword from list
        list_view = self.query_one("#target-keyword-list", ListView)
        if list_view.highlighted_child and hasattr(list_view.highlighted_child, 'keyword_data'):
            target = list_view.highlighted_child.keyword_data
            self.dismiss(("existing", target))
        else:
            self.dismiss(None)


class CollectionsTagWindow(Container):
    """
    Window for managing keywords/tags in the media database.
    Provides functionality for viewing, editing, merging, and deleting keywords.
    """
    
    selected_keywords: reactive[List[Dict[str, Any]]] = reactive([])
    keyword_search: reactive[str] = reactive("")
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.all_keywords: List[Dict[str, Any]] = []
        
    def compose(self) -> ComposeResult:
        """Compose the UI structure."""
        with Horizontal(classes="collections-main-container"):
            # Left pane - Keyword list
            with Container(classes="collections-left-pane"):
                yield Label("Keywords/Tags Management", classes="pane-title")
                yield Input(
                    placeholder="Search keywords...",
                    id="keyword-search-input",
                    classes="search-input"
                )
                with Horizontal(classes="keyword-action-buttons"):
                    yield Button("Select All", id="select-all-keywords", classes="small-button")
                    yield Button("Clear Selection", id="clear-selection", classes="small-button")
                yield ListView(id="keyword-list", classes="keyword-list")
                
            # Right pane - Actions and details
            with VerticalScroll(classes="collections-right-pane"):
                yield Label("Keyword Actions", classes="pane-title")
                
                # Selection info
                yield Static("No keywords selected", id="selection-info", classes="selection-info")
                
                # Action buttons
                with Container(classes="action-buttons-container"):
                    yield Button("Rename", id="rename-keyword", disabled=True, variant="primary")
                    yield Button("Merge Selected", id="merge-keywords", disabled=True, variant="primary")
                    yield Button("Delete Selected", id="delete-keywords", disabled=True, variant="error")
                    
                # Keyword details
                yield Label("Keyword Details", classes="section-title")
                yield Markdown("", id="keyword-details", classes="keyword-details")
                
                # Usage statistics
                yield Label("Usage Statistics", classes="section-title")
                yield Markdown("", id="usage-stats", classes="usage-stats")
                
    def on_mount(self) -> None:
        """Initialize the window when mounted."""
        self.load_keywords()
        
    def load_keywords(self) -> None:
        """Load all keywords from the database."""
        try:
            if not self.app_instance.media_db:
                logger.error("Media DB not available")
                return
                
            # Get all keywords with usage counts
            self.all_keywords = self.get_keywords_with_stats()
            self.refresh_keyword_list()
            
        except Exception as e:
            logger.error(f"Error loading keywords: {e}")
            self.app_instance.notify(f"Error loading keywords: {str(e)}", severity="error")
            
    def get_keywords_with_stats(self) -> List[Dict[str, Any]]:
        """Get all keywords with usage statistics."""
        if not self.app_instance.media_db:
            return []
            
        try:
            # Use the new get_keyword_usage_stats method
            return self.app_instance.media_db.get_keyword_usage_stats()
            
        except Exception as e:
            logger.error(f"Error getting keyword stats: {e}")
            return []
            
    def refresh_keyword_list(self) -> None:
        """Refresh the keyword list view."""
        list_view = self.query_one("#keyword-list", ListView)
        list_view.clear()
        
        search_term = self.keyword_search.lower()
        
        for keyword in self.all_keywords:
            if search_term and search_term not in keyword['keyword'].lower():
                continue
                
            # Create list item with keyword and usage count
            item_content = f"{keyword['keyword']} ({keyword['usage_count']} items)"
            item = ListItem(Label(item_content))
            item.keyword_data = keyword
            item.add_class("keyword-item")
            
            # Check if selected
            if any(k['id'] == keyword['id'] for k in self.selected_keywords):
                item.add_class("selected")
                
            list_view.append(item)
            
    @on(Input.Changed, "#keyword-search-input")
    def handle_search_change(self, event: Input.Changed) -> None:
        """Handle keyword search input changes."""
        self.keyword_search = event.value
        self.refresh_keyword_list()
        
    @on(ListView.Selected, "#keyword-list")
    def handle_keyword_selection(self, event: ListView.Selected) -> None:
        """Handle keyword selection from list."""
        if hasattr(event.item, 'keyword_data'):
            keyword = event.item.keyword_data
            
            # Toggle selection
            if any(k['id'] == keyword['id'] for k in self.selected_keywords):
                self.selected_keywords = [k for k in self.selected_keywords if k['id'] != keyword['id']]
                event.item.remove_class("selected")
            else:
                self.selected_keywords.append(keyword)
                event.item.add_class("selected")
                
            self.update_selection_info()
            self.update_action_buttons()
            self.update_keyword_details()
            
    @on(Button.Pressed, "#select-all-keywords")
    def select_all_keywords(self) -> None:
        """Select all visible keywords."""
        list_view = self.query_one("#keyword-list", ListView)
        self.selected_keywords = []
        
        for item in list_view.children:
            if isinstance(item, ListItem) and hasattr(item, 'keyword_data'):
                self.selected_keywords.append(item.keyword_data)
                item.add_class("selected")
                
        self.update_selection_info()
        self.update_action_buttons()
        
    @on(Button.Pressed, "#clear-selection")
    def clear_selection(self) -> None:
        """Clear all keyword selections."""
        self.selected_keywords = []
        list_view = self.query_one("#keyword-list", ListView)
        
        for item in list_view.children:
            if isinstance(item, ListItem):
                item.remove_class("selected")
                
        self.update_selection_info()
        self.update_action_buttons()
        self.update_keyword_details()
        
    def update_selection_info(self) -> None:
        """Update the selection information display."""
        info = self.query_one("#selection-info", Static)
        count = len(self.selected_keywords)
        
        if count == 0:
            info.update("No keywords selected")
        elif count == 1:
            info.update(f"1 keyword selected: {self.selected_keywords[0]['keyword']}")
        else:
            info.update(f"{count} keywords selected")
            
    def update_action_buttons(self) -> None:
        """Enable/disable action buttons based on selection."""
        count = len(self.selected_keywords)
        
        self.query_one("#rename-keyword", Button).disabled = count != 1
        self.query_one("#merge-keywords", Button).disabled = count < 2
        self.query_one("#delete-keywords", Button).disabled = count == 0
        
    def update_keyword_details(self) -> None:
        """Update the keyword details display."""
        details = self.query_one("#keyword-details", Markdown)
        stats = self.query_one("#usage-stats", Markdown)
        
        if len(self.selected_keywords) == 0:
            details.update("*Select a keyword to view details*")
            stats.update("*No keyword selected*")
        elif len(self.selected_keywords) == 1:
            keyword = self.selected_keywords[0]
            details_text = f"""
**Keyword:** {keyword['keyword']}  
**ID:** {keyword['id']}  
**UUID:** {keyword.get('uuid', 'N/A')}  
**Created:** {keyword.get('created_at', 'N/A')}  
**Modified:** {keyword.get('last_modified', 'N/A')}
"""
            details.update(details_text)
            
            # Get media items using this keyword
            if self.app_instance.media_db:
                try:
                    media_query = """
                        SELECT m.id, m.title, m.type, m.ingestion_date
                        FROM Media m
                        JOIN MediaKeywords mk ON m.id = mk.media_id
                        WHERE mk.keyword_id = ?
                        ORDER BY m.ingestion_date DESC
                        LIMIT 10
                    """
                    media_items = self.app_instance.media_db.execute_query(media_query, (keyword['id'],))
                    
                    stats_text = f"**Total items using this keyword:** {keyword['usage_count']}\n\n"
                    if media_items:
                        stats_text += "**Recent items:**\n"
                        for item in media_items:
                            stats_text += f"- {item[1]} ({item[2]})\n"
                        if keyword['usage_count'] > 10:
                            stats_text += f"\n*...and {keyword['usage_count'] - 10} more items*"
                    
                    stats.update(stats_text)
                except Exception as e:
                    stats.update(f"*Error loading usage statistics: {e}*")
        else:
            details.update(f"*{len(self.selected_keywords)} keywords selected*")
            total_usage = sum(k['usage_count'] for k in self.selected_keywords)
            stats.update(f"**Total usage across selected keywords:** {total_usage} items")
            
    @on(Button.Pressed, "#rename-keyword")
    async def handle_rename(self) -> None:
        """Handle keyword rename action."""
        if len(self.selected_keywords) != 1:
            return
            
        keyword = self.selected_keywords[0]
        dialog = KeywordRenameDialog(keyword['keyword'], keyword['id'])
        result = await self.app_instance.push_screen_wait(dialog)
        
        if result:
            keyword_id, new_name = result
            await self.rename_keyword(keyword_id, new_name)
            
    @on(Button.Pressed, "#merge-keywords")
    async def handle_merge(self) -> None:
        """Handle keyword merge action."""
        if len(self.selected_keywords) < 2:
            return
            
        dialog = KeywordMergeDialog(self.selected_keywords, self.all_keywords)
        result = await self.app_instance.push_screen_wait(dialog)
        
        if result:
            merge_type, target = result
            await self.merge_keywords(self.selected_keywords, merge_type, target)
            
    @on(Button.Pressed, "#delete-keywords")
    async def handle_delete(self) -> None:
        """Handle keyword delete action."""
        if not self.selected_keywords:
            return
            
        # Show confirmation dialog
        count = len(self.selected_keywords)
        
        # Create a simple confirmation dialog
        from textual.screen import ModalScreen
        from textual.containers import Vertical, Horizontal
        from textual.widgets import Label, Button
        
        class DeleteConfirmationModal(ModalScreen):
            """Modal for delete confirmation."""
            CSS = """
            #delete-dialog {
                align: center middle;
                width: 60;
                height: 12;
                padding: 1 2;
                background: $surface;
                border: thick $primary;
            }
            """
            
            def __init__(self, keyword_count: int, keywords: List[Dict[str, Any]]):
                super().__init__()
                self.keyword_count = keyword_count
                self.keywords = keywords
                
            def compose(self) -> ComposeResult:
                with Vertical(id="delete-dialog"):
                    keyword_names = ", ".join([k['keyword'] for k in self.keywords[:3]])
                    if len(self.keywords) > 3:
                        keyword_names += f" and {len(self.keywords) - 3} more"
                    
                    yield Label(f"Delete {self.keyword_count} keyword{'s' if self.keyword_count > 1 else ''}?", classes="dialog-title")
                    yield Label(f"Keywords: {keyword_names}", classes="dialog-subtitle")
                    yield Label("This will remove the keywords from all media items.", classes="dialog-message")
                    with Horizontal(classes="dialog-buttons"):
                        yield Button("Cancel", id="cancel-delete", variant="default")
                        yield Button("Delete", id="confirm-delete", variant="error")
                        
            @on(Button.Pressed, "#cancel-delete")
            def handle_cancel(self) -> None:
                self.dismiss(False)
                
            @on(Button.Pressed, "#confirm-delete")
            def handle_confirm(self) -> None:
                self.dismiss(True)
        
        # Show modal and wait for result
        result = await self.app_instance.push_screen_wait(
            DeleteConfirmationModal(count, self.selected_keywords)
        )
        
        if result:
            # User confirmed deletion
            from ..Event_Handlers.collections_tag_events import KeywordDeleteEvent
            keyword_ids = [k['id'] for k in self.selected_keywords]
            self.post_message(KeywordDeleteEvent(keyword_ids))
        
    async def rename_keyword(self, keyword_id: int, new_name: str) -> None:
        """Rename a keyword in the database."""
        from ..Event_Handlers.collections_tag_events import KeywordRenameEvent
        self.post_message(KeywordRenameEvent(keyword_id, new_name))
            
    async def merge_keywords(self, source_keywords: List[Dict[str, Any]], merge_type: str, target: Any) -> None:
        """Merge multiple keywords into one."""
        from ..Event_Handlers.collections_tag_events import KeywordMergeEvent
        
        source_ids = [k['id'] for k in source_keywords]
        
        if merge_type == "new":
            # Create new keyword and merge into it
            self.post_message(KeywordMergeEvent(source_ids, target, create_if_not_exists=True))
        else:
            # Merge into existing keyword
            self.post_message(KeywordMergeEvent(source_ids, target['keyword'], create_if_not_exists=False))