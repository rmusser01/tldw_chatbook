# tldw_chatbook/Widgets/activity_log.py
# Activity log widget for tracking operations
#
# Imports
from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import csv
import json

# Third-party imports
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Label, Button, Input, Select
from textual.widget import Widget
from textual.reactive import reactive
from textual.timer import Timer
from textual.message import Message
from loguru import logger

# Configure logger
logger = logger.bind(module="activity_log")

# Type definitions
LogLevel = Literal["info", "success", "warning", "error", "debug"]


# Event messages
class ActivityLogCleared(Message):
    """Event sent when the activity log is cleared."""

    def __init__(self) -> None:
        super().__init__()


class ActivityLogExported(Message):
    """Event sent when the activity log is exported."""
    
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        super().__init__()


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

    def to_string(self) -> str:
        """Return a human-readable string representation."""
        base = (
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{self.level.upper()} [{self.category}] {self.message}"
        )
        if not self.details:
            return base
        return f"{base} | details={json.dumps(self.details, sort_keys=True)}"


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
        show_export: Optional[bool] = None,
        auto_scroll: bool = True,
        max_entries: int = MAX_ENTRIES,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.show_filters = show_filters
        self.show_search = show_search
        self.show_actions = show_actions
        self.show_export = show_actions if show_export is None else show_export
        self.max_entries = max_entries
        self.auto_scroll = auto_scroll

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
                        yield Button("Clear Log", id="clear-log", classes="small-button")
                        if self.show_export:
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
                        yield Select(
                            [
                                ("Info", "info"),
                                ("Success", "success"),
                                ("Warning", "warning"),
                                ("Error", "error"),
                                ("Debug", "debug"),
                            ],
                            prompt="All Levels",
                            allow_blank=True,
                            value=Select.NULL,
                            id="log-level-filter",
                            classes="log-filter-select",
                        )
                        yield Select(
                            [],
                            prompt="All Categories",
                            allow_blank=True,
                            value=Select.NULL,
                            id="log-category-filter",
                            classes="log-filter-select",
                        )

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

        self.entries.appendleft(entry)
        self.categories.add(category)

        if self.is_mounted:
            if self.show_filters:
                self._update_category_filter()
            if self._has_active_filters():
                self._update_display()
            else:
                self._add_entry_to_display(entry)
                self._trim_display_entries()

        # Auto-scroll if enabled
        if self.auto_scroll and self.is_mounted:
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
            before_target = next(iter(container.children), None)
            container.mount(entry_widget, before=before_target)
        except Exception as e:
            logger.error(f"Error adding log entry: {e}")

    def _create_entry_widget(self, entry: ActivityEntry) -> Widget:
        """Create a widget for a log entry."""
        # Format timestamp
        time_str = entry.timestamp.strftime("%H:%M:%S")
        relative_time = self._get_relative_time(entry.timestamp)
        
        # Get icon and style for level
        icon, level_class = self._get_level_style(entry.level)

        message_text = entry.message
        if entry.details:
            message_text += f" [+] {json.dumps(entry.details, sort_keys=True)}"

        return Horizontal(
            Static(
                f"{time_str}\n{relative_time}",
                classes="log-timestamp"
            ),
            Static(icon, classes=f"log-icon {level_class}"),
            Static(f"[{entry.category}]", classes="log-category"),
            Static(message_text, classes="log-message"),
            classes=f"log-entry log-{entry.level}",
        )
    
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

    def _filter_entry(self, entry: ActivityEntry) -> bool:
        """Backwards-compatible filter helper for tests and callers."""
        return self._should_display_entry(entry)

    def _has_active_filters(self) -> bool:
        """Return True when the display is constrained by any filter control."""
        return bool(self.filter_level or self.filter_category or self.search_query)

    def _get_filtered_entries(self) -> List[ActivityEntry]:
        """Return entries visible under the current filter state."""
        return [entry for entry in self.entries if self._should_display_entry(entry)]

    def _trim_display_entries(self) -> None:
        """Keep mounted entry widgets aligned with the backing deque length."""
        try:
            container = self.query_one("#log-entries", Container)
        except Exception:
            return

        visible_count = len(self._get_filtered_entries()) if self._has_active_filters() else len(self.entries)
        children = list(container.children)
        while len(children) > visible_count:
            children[-1].remove()
            children.pop()

    def _apply_filters(self) -> None:
        """Rebuild the log display using the current filter state."""
        if not self.is_mounted:
            return
        self._update_display()

    def _update_display(self) -> None:
        """Update the entire log display."""
        try:
            container = self.query_one("#log-entries", Container)
            for child in list(container.children):
                child.remove()

            # Add filtered entries
            for entry in self._get_filtered_entries():
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
        except Exception:
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

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        if event.input.id == "search-logs":
            self.search_query = event.value
            self._update_display()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle filter dropdown changes."""
        if event.select.id == "log-level-filter":
            self.filter_level = None if event.value == Select.NULL else event.value
            self._apply_filters()
        elif event.select.id == "log-category-filter":
            self.filter_category = None if event.value == Select.NULL else event.value
            self._apply_filters()

    def _update_category_filter(self) -> None:
        """Refresh category filter options to match available entry categories."""
        if not self.is_mounted or not self.show_filters:
            return

        category_filter = self.query_one("#log-category-filter", Select)
        options = [(category, category) for category in sorted(self.categories)]
        category_filter.set_options(options)

        if self.filter_category and self.filter_category not in self.categories:
            self.filter_category = None
            category_filter.clear()

    def clear_log(self) -> None:
        """Clear all log entries."""
        self.entries.clear()
        self.categories.clear()
        self._update_display()
        if self.show_filters and self.is_mounted:
            self._update_category_filter()
        self.post_message(ActivityLogCleared())

    def export_log(self, target: Optional[str] = None, filtered_only: bool = False) -> None:
        """Export log to a file.

        Args:
            target: Either a format name (`json`, `csv`, `text`) or a file path.
            filtered_only: Export only entries visible under current filters.
        """
        export_format, filepath = self._resolve_export_target(target)
        entries = self._get_filtered_entries() if filtered_only else list(self.entries)

        try:
            with open(filepath, "w", newline="") as handle:
                if export_format == "json":
                    self._write_json_export(handle, entries)
                elif export_format == "csv":
                    self._write_csv_export(handle, entries)
                else:
                    self._write_text_export(handle, entries)

            self.log_success(f"Log exported to {filepath}", "system")
            self.post_message(ActivityLogExported(filepath))

        except Exception as e:
            logger.error(f"Failed to export log: {e}")
            self.log_error(f"Export failed: {str(e)}", "system")

    def _resolve_export_target(self, target: Optional[str]) -> tuple[str, str]:
        """Resolve an export format and destination path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_format = "json"

        if not target:
            return export_format, f"activity_log_{timestamp}.json"

        normalized_target = target.lower()
        if normalized_target in {"json", "csv", "text", "txt"}:
            export_format = "text" if normalized_target in {"text", "txt"} else normalized_target
            extension = "txt" if export_format == "text" else export_format
            return export_format, f"activity_log_{timestamp}.{extension}"

        path = Path(target)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            export_format = "csv"
        elif suffix in {".txt", ".text"}:
            export_format = "text"

        return export_format, str(path)

    def _write_json_export(self, handle, entries: List[ActivityEntry]) -> None:
        """Write entries as JSON."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "entries": [entry.to_dict() for entry in entries],
        }
        json.dump(data, handle, indent=2)

    def _write_csv_export(self, handle, entries: List[ActivityEntry]) -> None:
        """Write entries as CSV."""
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "level", "category", "message", "details"],
        )
        writer.writeheader()
        for entry in entries:
            row = entry.to_dict()
            row["details"] = json.dumps(entry.details, sort_keys=True) if entry.details else ""
            writer.writerow(row)

    def _write_text_export(self, handle, entries: List[ActivityEntry]) -> None:
        """Write entries as plain text."""
        handle.write(f"Exported at: {datetime.now().isoformat()}\n")
        handle.write("=" * 80 + "\n")
        for entry in entries:
            handle.write(entry.to_string() + "\n")
    
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

        for entry in self.entries:  # Most recent first
            if level and entry.level != level:
                continue
            if category and entry.category != category:
                continue
            
            entries.append(entry)
            
            if limit and len(entries) >= limit:
                break
        
        return entries
