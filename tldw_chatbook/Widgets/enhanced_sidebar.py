"""
Enhanced sidebar widget with improved UX features.

This module provides an enhanced sidebar widget with:
- Better keyboard navigation
- Visual feedback for interactive elements
- Loading states for async operations
- Improved accessibility
"""

from typing import Optional, List, Dict, Any, Callable
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static, Button, Collapsible, LoadingIndicator
from textual.reactive import reactive
from textual.binding import Binding
from textual.message import Message
from textual import work
from loguru import logger


class SidebarSection(Container):
    """A section within the sidebar with enhanced UX."""
    
    # Reactive properties
    is_loading = reactive(False, layout=False)
    is_focused = reactive(False, layout=False)
    
    def __init__(
        self,
        title: str,
        content: Optional[Any] = None,
        collapsible: bool = True,
        collapsed: bool = False,
        **kwargs
    ):
        """Initialize the sidebar section.
        
        Args:
            title: Section title
            content: Content widget or compose function
            collapsible: Whether section can be collapsed
            collapsed: Initial collapsed state
        """
        super().__init__(**kwargs)
        self.title = title
        self.content = content
        self.collapsible = collapsible
        self.collapsed = collapsed
        
    def compose(self) -> ComposeResult:
        """Compose the section UI."""
        if self.collapsible:
            with Collapsible(
                title=self.title,
                collapsed=self.collapsed,
                classes="sidebar-section-collapsible"
            ):
                if self.content:
                    yield self.content
                yield LoadingIndicator(classes="section-loading hidden")
        else:
            yield Static(self.title, classes="section-title")
            if self.content:
                yield self.content
            yield LoadingIndicator(classes="section-loading hidden")
    
    def watch_is_loading(self, is_loading: bool) -> None:
        """Watch loading state changes."""
        loading_indicator = self.query_one(LoadingIndicator)
        if is_loading:
            loading_indicator.remove_class("hidden")
            if self.content:
                self.content.add_class("loading-fade")
        else:
            loading_indicator.add_class("hidden")
            if self.content:
                self.content.remove_class("loading-fade")
    
    @work(exclusive=True)
    async def load_content(self, loader: Callable) -> None:
        """Load content asynchronously with loading indicator.
        
        Args:
            loader: Async function that returns content
        """
        self.is_loading = True
        try:
            result = await loader()
            # Update content based on result
            if result and self.content:
                # Update content widget with result
                pass
        except Exception as e:
            logger.error(f"Error loading section content: {e}")
        finally:
            self.is_loading = False


class EnhancedSidebar(VerticalScroll):
    """Enhanced sidebar with improved UX and keyboard navigation."""
    
    BINDINGS = [
        Binding("j", "focus_next", "Next item", show=False),
        Binding("k", "focus_previous", "Previous item", show=False),
        Binding("enter", "select_focused", "Select", show=False),
        Binding("space", "toggle_focused", "Toggle", show=False),
        Binding("tab", "focus_next_section", "Next section", show=False),
        Binding("shift+tab", "focus_previous_section", "Previous section", show=False),
        Binding("/", "search", "Search", show=False),
        Binding("escape", "clear_focus", "Clear focus", show=False),
    ]
    
    # Reactive properties
    focused_section = reactive(0, layout=False)
    focused_item = reactive(0, layout=False)
    search_active = reactive(False, layout=False)
    
    def __init__(
        self,
        sections: Optional[List[SidebarSection]] = None,
        **kwargs
    ):
        """Initialize the enhanced sidebar.
        
        Args:
            sections: List of sidebar sections
        """
        super().__init__(**kwargs)
        self.sections = sections or []
        self.focusable_items: List[Any] = []
        
    def compose(self) -> ComposeResult:
        """Compose the sidebar UI."""
        for section in self.sections:
            yield section
    
    def on_mount(self) -> None:
        """Handle mount event."""
        # Build list of focusable items
        self._update_focusable_items()
        
    def _update_focusable_items(self) -> None:
        """Update the list of focusable items."""
        self.focusable_items = []
        for widget in self.walk_children():
            if widget.focusable:
                self.focusable_items.append(widget)
    
    def action_focus_next(self) -> None:
        """Focus next item in the sidebar."""
        if not self.focusable_items:
            return
            
        self.focused_item = (self.focused_item + 1) % len(self.focusable_items)
        self.focusable_items[self.focused_item].focus()
        self._scroll_to_focused()
    
    def action_focus_previous(self) -> None:
        """Focus previous item in the sidebar."""
        if not self.focusable_items:
            return
            
        self.focused_item = (self.focused_item - 1) % len(self.focusable_items)
        self.focusable_items[self.focused_item].focus()
        self._scroll_to_focused()
    
    def action_focus_next_section(self) -> None:
        """Focus next section in the sidebar."""
        if not self.sections:
            return
            
        self.focused_section = (self.focused_section + 1) % len(self.sections)
        section = self.sections[self.focused_section]
        
        # Find first focusable item in section
        for widget in section.walk_children():
            if widget.focusable:
                widget.focus()
                self._scroll_to_focused()
                break
    
    def action_focus_previous_section(self) -> None:
        """Focus previous section in the sidebar."""
        if not self.sections:
            return
            
        self.focused_section = (self.focused_section - 1) % len(self.sections)
        section = self.sections[self.focused_section]
        
        # Find first focusable item in section
        for widget in section.walk_children():
            if widget.focusable:
                widget.focus()
                self._scroll_to_focused()
                break
    
    def action_select_focused(self) -> None:
        """Select the currently focused item."""
        if self.focused:
            # Simulate click on focused widget
            focused_widget = self.app.focused
            if focused_widget and hasattr(focused_widget, 'action_press'):
                focused_widget.action_press()
    
    def action_toggle_focused(self) -> None:
        """Toggle the currently focused item (for collapsibles)."""
        focused_widget = self.app.focused
        if focused_widget and isinstance(focused_widget, Collapsible):
            focused_widget.collapsed = not focused_widget.collapsed
    
    def action_search(self) -> None:
        """Activate search mode."""
        self.search_active = True
        # Post message to open search widget
        self.post_message(SearchActivated())
    
    def action_clear_focus(self) -> None:
        """Clear focus from all items."""
        self.app.set_focus(None)
    
    def _scroll_to_focused(self) -> None:
        """Scroll to make the focused item visible."""
        focused_widget = self.app.focused
        if focused_widget:
            self.scroll_to_widget(focused_widget, animate=True)
    
    def add_section(self, section: SidebarSection) -> None:
        """Add a new section to the sidebar.
        
        Args:
            section: Section to add
        """
        self.sections.append(section)
        self.mount(section)
        self._update_focusable_items()
    
    def remove_section(self, section: SidebarSection) -> None:
        """Remove a section from the sidebar.
        
        Args:
            section: Section to remove
        """
        if section in self.sections:
            self.sections.remove(section)
            section.remove()
            self._update_focusable_items()
    
    def get_section(self, title: str) -> Optional[SidebarSection]:
        """Get a section by title.
        
        Args:
            title: Section title
            
        Returns:
            Section if found, None otherwise
        """
        for section in self.sections:
            if section.title == title:
                return section
        return None


class SearchActivated(Message):
    """Message sent when search is activated."""
    pass


class SidebarItemSelected(Message):
    """Message sent when a sidebar item is selected."""
    
    def __init__(self, item: Any, section: str):
        super().__init__()
        self.item = item
        self.section = section