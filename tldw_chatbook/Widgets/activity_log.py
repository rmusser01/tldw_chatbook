# tldw_chatbook/Widgets/activity_log.py
# Activity log widget for tracking operations
#
# Imports
from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from collections import deque
from dataclasses import dataclass
import json

# Third-party imports
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Label, Button, DataTable, Input
from textual.widget import Widget
from textual.reactive import reactive
from textual.timer import Timer
from loguru import logger

# Configure logger
logger = logger.bind(module="activity_log")

# Type definitions
LogLevel = Literal["info", "success", "warning", "error", "debug"]


@dataclass
class ActivityEntry:
    """Single activity log entry."""
    timestamp: datetime
    level: LogLevel
    category: str
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActivityEntry':
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=data["level"],
            category=data["category"],
            message=data["message"],
            details=data.get("details")
        )


class ActivityLogWidget(Widget):
    """Widget for displaying and managing activity logs.
    
    Features:
    - Real-time activity tracking
    - Filtering by level and category
    - Search functionality
    - Export capabilities
    - Automatic cleanup of old entries
    """
    
    DEFAULT_CLASSES = "activity-log-widget"
    
    # Maximum entries to keep in memory
    MAX_ENTRIES = 1000
    
    # Reactive properties
    filter_level: reactive[Optional[LogLevel]] = reactive(None)
    filter_category: reactive[Optional[str]] = reactive(None)
    search_query: reactive[str] = reactive("")
    auto_scroll: reactive[bool] = reactive(True)
    
    def __init__(
        self,
        show_filters: bool = True,
        show_search: bool = True,
        show_actions: bool = True,
        max_entries: int = MAX_ENTRIES,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.show_filters = show_filters
        self.show_search = show_search
        self.show_actions = show_actions
        self.max_entries = max_entries
        
        # Storage
        self.entries: deque[ActivityEntry] = deque(maxlen=max_entries)
        self.categories: set[str] = set()
        
        # UI state
        self._update_timer: Optional[Timer] = None
        
    def compose(self) -> ComposeResult:
        """Compose the activity log widget."""
        with Container(classes="activity-log-container"):
            # Header with controls
            with Horizontal(classes="activity-log-header"):
                yield Label("Activity Log", classes="activity-log-title")
                
                if self.show_actions:
                    with Horizontal(classes="activity-log-actions"):
                        yield Button("Clear", id="clear-log", classes="small-button")
                        yield Button("Export", id="export-log", classes="small-button")
                        yield Button(
                            "⬇ Auto-scroll" if self.auto_scroll else "⬇ Auto-scroll (off)",
                            id="toggle-auto-scroll",
                            classes="small-button"
                        )
            
            # Filters and search
            if self.show_filters or self.show_search:
                with Horizontal(classes="activity-log-controls"):
                    if self.show_filters:
                        # Level filter buttons
                        with Horizontal(classes="level-filters"):
                            yield Button("All", id="filter-all", classes="filter-button active")
                            yield Button("Info", id="filter-info", classes="filter-button")
                            yield Button("Success", id="filter-success", classes="filter-button")
                            yield Button("Warning", id="filter-warning", classes="filter-button")
                            yield Button("Error", id="filter-error", classes="filter-button")
                    
                    if self.show_search:
                        yield Input(
                            placeholder="Search logs...",
                            id="search-logs",
                            classes="log-search-input"
                        )
            
            # Log display area
            with VerticalScroll(id="log-scroll", classes="activity-log-scroll"):
                yield Container(id="log-entries", classes="log-entries-container")
    
    def on_mount(self) -> None:
        """Initialize the log display."""
        self._update_display()
        # Start periodic update timer (for relative timestamps)
        self._update_timer = self.set_interval(60, self._update_timestamps)
    
    def add_entry(
        self,
        message: str,
        level: LogLevel = "info",
        category: str = "general",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new activity entry.
        
        Args:
            message: Log message
            level: Log level (info, success, warning, error, debug)
            category: Category for filtering
            details: Additional details
        """
        entry = ActivityEntry(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            details=details
        )
        
        self.entries.append(entry)
        self.categories.add(category)
        
        # Update display
        self._add_entry_to_display(entry)
        
        # Auto-scroll if enabled
        if self.auto_scroll:
            self._scroll_to_bottom()
    
    def log_info(self, message: str, category: str = "general", **details) -> None:
        """Log an info message."""
        self.add_entry(message, "info", category, details if details else None)
    
    def log_success(self, message: str, category: str = "general", **details) -> None:
        """Log a success message."""
        self.add_entry(message, "success", category, details if details else None)
    
    def log_warning(self, message: str, category: str = "general", **details) -> None:
        """Log a warning message."""
        self.add_entry(message, "warning", category, details if details else None)
    
    def log_error(self, message: str, category: str = "general", **details) -> None:
        """Log an error message."""
        self.add_entry(message, "error", category, details if details else None)
    
    def log_debug(self, message: str, category: str = "general", **details) -> None:
        """Log a debug message."""
        self.add_entry(message, "debug", category, details if details else None)
    
    def _add_entry_to_display(self, entry: ActivityEntry) -> None:
        """Add a single entry to the display."""
        # Check if entry should be displayed based on filters
        if not self._should_display_entry(entry):
            return
        
        # Create entry widget
        entry_widget = self._create_entry_widget(entry)
        
        # Add to container
        try:
            container = self.query_one("#log-entries", Container)
            container.mount(entry_widget)
        except Exception as e:
            logger.error(f"Error adding log entry: {e}")
    
    def _create_entry_widget(self, entry: ActivityEntry) -> Widget:
        """Create a widget for a log entry."""
        # Format timestamp
        time_str = entry.timestamp.strftime("%H:%M:%S")
        relative_time = self._get_relative_time(entry.timestamp)
        
        # Get icon and style for level
        icon, level_class = self._get_level_style(entry.level)
        
        with Horizontal(classes=f"log-entry log-{entry.level}") as container:
            # Timestamp
            yield Static(
                f"{time_str}\n{relative_time}",
                classes="log-timestamp"
            )
            
            # Level icon
            yield Static(icon, classes=f"log-icon {level_class}")
            
            # Category
            yield Static(f"[{entry.category}]", classes="log-category")
            
            # Message
            message_text = entry.message
            if entry.details:
                # Add details indicator
                message_text += " [+]"
            yield Static(message_text, classes="log-message")
        
        return container
    
    def _get_level_style(self, level: LogLevel) -> tuple[str, str]:
        """Get icon and CSS class for log level."""
        styles = {
            "info": ("ℹ️", "level-info"),
            "success": ("✅", "level-success"),
            "warning": ("⚠️", "level-warning"),
            "error": ("❌", "level-error"),
            "debug": ("🐛", "level-debug")
        }
        return styles.get(level, ("•", "level-default"))
    
    def _get_relative_time(self, timestamp: datetime) -> str:
        """Get relative time string (e.g., '2m ago')."""
        now = datetime.now()
        delta = now - timestamp
        
        if delta.total_seconds() < 60:
            return "just now"
        elif delta.total_seconds() < 3600:
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes}m ago"
        elif delta.total_seconds() < 86400:
            hours = int(delta.total_seconds() / 3600)
            return f"{hours}h ago"
        else:
            days = int(delta.total_seconds() / 86400)
            return f"{days}d ago"
    
    def _should_display_entry(self, entry: ActivityEntry) -> bool:
        """Check if entry should be displayed based on current filters."""
        # Level filter
        if self.filter_level and entry.level != self.filter_level:
            return False
        
        # Category filter
        if self.filter_category and entry.category != self.filter_category:
            return False
        
        # Search filter
        if self.search_query:
            query = self.search_query.lower()
            if (query not in entry.message.lower() and
                query not in entry.category.lower()):
                return False
        
        return True
    
    def _update_display(self) -> None:
        """Update the entire log display."""
        try:
            container = self.query_one("#log-entries", Container)
            # Remove all children from the container
            for child in list(container.children):
                child.remove()
            
            # Add filtered entries
            for entry in self.entries:
                if self._should_display_entry(entry):
                    entry_widget = self._create_entry_widget(entry)
                    container.mount(entry_widget)
            
        except Exception as e:
            logger.error(f"Error updating log display: {e}")
    
    def _update_timestamps(self) -> None:
        """Update relative timestamps periodically."""
        # This would update all visible timestamps
        # For now, just trigger a display update
        self._update_display()
    
    def _scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the log."""
        try:
            scroll = self.query_one("#log-scroll", VerticalScroll)
            scroll.scroll_end(animate=False)
        except:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "clear-log":
            self.clear_log()
        elif button_id == "export-log":
            self.export_log()
        elif button_id == "toggle-auto-scroll":
            self.auto_scroll = not self.auto_scroll
            event.button.label = "⬇ Auto-scroll" if self.auto_scroll else "⬇ Auto-scroll (off)"
        elif button_id and button_id.startswith("filter-"):
            self._handle_filter_button(button_id)
    
    def _handle_filter_button(self, button_id: str) -> None:
        """Handle filter button clicks."""
        # Remove active class from all filter buttons
        for button in self.query(".filter-button"):
            button.remove_class("active")
        
        # Set new filter and mark button as active
        button = self.query_one(f"#{button_id}", Button)
        button.add_class("active")
        
        if button_id == "filter-all":
            self.filter_level = None
        else:
            level = button_id.replace("filter-", "")
            self.filter_level = level
        
        # Update display
        self._update_display()
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "search-logs":
            self.search_query = event.value
            self._update_display()
    
    def clear_log(self) -> None:
        """Clear all log entries."""
        self.entries.clear()
        self.categories.clear()
        self._update_display()
        self.log_info("Log cleared", "system")
    
    def export_log(self, filepath: Optional[str] = None) -> None:
        """Export log to file."""
        if not filepath:
            # Generate default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"activity_log_{timestamp}.json"
        
        try:
            # Convert entries to JSON
            data = {
                "exported_at": datetime.now().isoformat(),
                "entries": [entry.to_dict() for entry in self.entries]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.log_success(f"Log exported to {filepath}", "system")
            
        except Exception as e:
            logger.error(f"Failed to export log: {e}")
            self.log_error(f"Export failed: {str(e)}", "system")
    
    def get_entries(
        self,
        level: Optional[LogLevel] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ActivityEntry]:
        """Get filtered log entries.
        
        Args:
            level: Filter by level
            category: Filter by category
            limit: Maximum entries to return
            
        Returns:
            List of matching entries
        """
        entries = []
        
        for entry in reversed(self.entries):  # Most recent first
            if level and entry.level != level:
                continue
            if category and entry.category != category:
                continue
            
            entries.append(entry)
            
            if limit and len(entries) >= limit:
                break
        
        return entries