# tldw_chatbook/Widgets/status_widget.py
"""
Enhanced status widget with color-coded messages and Rich formatting.
"""

from typing import List, Optional, Literal
from datetime import datetime
from textual.app import RenderResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static
from rich.console import RenderableType
from rich.text import Text
from rich.panel import Panel
from rich.table import Table


StatusLevel = Literal["info", "success", "warning", "error", "debug"]


class StatusMessage:
    """A single status message with metadata."""
    
    def __init__(self, message: str, level: StatusLevel = "info", timestamp: Optional[datetime] = None):
        self.message = message
        self.level = level
        self.timestamp = timestamp or datetime.now()
    
    def format(self) -> Text:
        """Format the message with color based on level."""
        level_colors = {
            "info": "bright_blue",
            "success": "bright_green",
            "warning": "bright_yellow",
            "error": "bright_red",
            "debug": "dim white"
        }
        
        level_symbols = {
            "info": "ℹ",
            "success": "✓",
            "warning": "⚠",
            "error": "✗",
            "debug": "•"
        }
        
        color = level_colors.get(self.level, "white")
        symbol = level_symbols.get(self.level, "•")
        
        # Format: [HH:MM:SS] ✓ Message
        time_str = self.timestamp.strftime("%H:%M:%S")
        
        text = Text()
        text.append(f"[{time_str}] ", style="dim")
        text.append(f"{symbol} ", style=f"bold {color}")
        text.append(self.message, style=color)
        
        return text


class EnhancedStatusWidget(Widget):
    """Enhanced status widget with color-coded messages and scrolling."""
    
    DEFAULT_CSS = """
    EnhancedStatusWidget {
        width: 100%;
        height: auto;
        min-height: 5;
        max-height: 20;
        background: $surface;
        border: round $border;
        padding: 1;
        overflow-y: auto;
    }
    """
    
    messages: reactive[List[StatusMessage]] = reactive([], recompose=True)
    max_messages: reactive[int] = reactive(100)
    show_timestamp: reactive[bool] = reactive(True)
    title: reactive[str] = reactive("Status")
    
    def __init__(
        self,
        title: str = "Status",
        max_messages: int = 100,
        show_timestamp: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.max_messages = max_messages
        self.show_timestamp = show_timestamp
        self._messages: List[StatusMessage] = []
    
    def add_message(self, message: str, level: StatusLevel = "info") -> None:
        """Add a new status message."""
        new_message = StatusMessage(message, level)
        self._messages.append(new_message)
        
        # Trim to max messages
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
        
        # Update reactive to trigger recompose
        self.messages = self._messages.copy()
        
        # Scroll to bottom to show new message
        self.scroll_end()
    
    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self.messages = []
    
    def add_info(self, message: str) -> None:
        """Add an info message."""
        self.add_message(message, "info")
    
    def add_success(self, message: str) -> None:
        """Add a success message."""
        self.add_message(message, "success")
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.add_message(message, "warning")
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.add_message(message, "error")
    
    def add_debug(self, message: str) -> None:
        """Add a debug message."""
        self.add_message(message, "debug")
    
    def render(self) -> RenderableType:
        """Render the status messages."""
        if not self._messages:
            return Panel(
                Text("No status messages", style="dim italic"),
                title=self.title,
                border_style="dim"
            )
        
        # Create a text object with all messages
        text = Text()
        for i, msg in enumerate(self._messages):
            if i > 0:
                text.append("\n")
            text.append(msg.format())
        
        return Panel(
            text,
            title=self.title,
            border_style="bright_blue"
        )
    
    def get_summary(self) -> Table:
        """Get a summary table of message counts by level."""
        table = Table(title="Status Summary")
        table.add_column("Level", style="bold")
        table.add_column("Count", justify="right")
        
        counts = {
            "info": 0,
            "success": 0,
            "warning": 0,
            "error": 0,
            "debug": 0
        }
        
        for msg in self._messages:
            counts[msg.level] += 1
        
        for level, count in counts.items():
            if count > 0:
                level_colors = {
                    "info": "bright_blue",
                    "success": "bright_green",
                    "warning": "bright_yellow",
                    "error": "bright_red",
                    "debug": "dim white"
                }
                table.add_row(
                    Text(level.capitalize(), style=level_colors[level]),
                    str(count)
                )
        
        return table