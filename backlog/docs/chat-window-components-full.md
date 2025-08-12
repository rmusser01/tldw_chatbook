# Chat Window Components - Complete Code Documentation

## Supporting Widgets and Components

### 1. Chat Message Enhanced Widget

This is the main message display component used in the chat log:

```python
# chat_message_enhanced.py
from typing import Optional, Dict, Any, List
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, TextArea, ListView, ListItem
from textual.reactive import reactive
from textual.message import Message
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
import json
import base64

class ChatMessageEnhanced(Container):
    """Enhanced chat message widget with rich content support."""
    
    DEFAULT_CSS = """
    ChatMessageEnhanced {
        layout: vertical;
        margin: 1 0;
        padding: 1;
        background: $surface;
        border: round $primary;
    }
    
    .message-header {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
    }
    
    .message-role {
        width: 10;
        text-style: bold;
    }
    
    .message-timestamp {
        width: 1fr;
        text-align: right;
        color: $text-muted;
    }
    
    .message-content {
        padding: 1;
    }
    
    .message-actions {
        layout: horizontal;
        height: 3;
        margin-top: 1;
    }
    
    .action-button {
        width: auto;
        margin: 0 1;
    }
    
    .editing-area {
        display: none;
    }
    
    .editing-area.active {
        display: block;
    }
    
    .image-container {
        margin: 1 0;
        padding: 1;
        border: round $primary;
    }
    
    .code-block {
        margin: 1 0;
        padding: 1;
        background: $surface-darken-1;
        border: round $secondary;
    }
    """
    
    # Message display modes
    DISPLAY_MODES = {
        "text": "plain text",
        "markdown": "markdown rendering",
        "code": "syntax highlighted code",
        "json": "formatted JSON",
        "image": "image display"
    }
    
    def __init__(
        self,
        message: Dict[str, Any],
        message_id: Optional[str] = None,
        editable: bool = True,
        show_actions: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.message = message
        self.message_id = message_id or self._generate_id()
        self.editable = editable
        self.show_actions = show_actions
        self.is_editing = False
        self.display_mode = self._detect_display_mode()
        
    def _generate_id(self) -> str:
        """Generate unique message ID."""
        import uuid
        return f"msg-{uuid.uuid4().hex[:8]}"
    
    def _detect_display_mode(self) -> str:
        """Detect the appropriate display mode for the message content."""
        content = self.message.get("content", "")
        
        # Check for images
        if "image" in self.message or self._has_base64_image(content):
            return "image"
        
        # Check for code blocks
        if "```" in content:
            return "code"
        
        # Check for JSON
        if self._is_json(content):
            return "json"
        
        # Check for markdown indicators
        if any(marker in content for marker in ["#", "*", "_", "[", "]"]):
            return "markdown"
        
        return "text"
    
    def _has_base64_image(self, content: str) -> bool:
        """Check if content contains base64 encoded image."""
        return "data:image" in content or "base64," in content
    
    def _is_json(self, content: str) -> bool:
        """Check if content is valid JSON."""
        try:
            json.loads(content)
            return True
        except:
            return False
    
    def compose(self) -> ComposeResult:
        """Compose the message widget."""
        # Message header
        with Horizontal(classes="message-header"):
            role = self.message.get("role", "unknown")
            timestamp = self.message.get("timestamp", "")
            
            yield Static(
                role.capitalize(),
                classes=f"message-role role-{role}"
            )
            yield Static(timestamp, classes="message-timestamp")
        
        # Message content
        with Container(classes="message-content"):
            content = self.message.get("content", "")
            
            if self.display_mode == "image":
                yield self._render_image_content(content)
            elif self.display_mode == "code":
                yield self._render_code_content(content)
            elif self.display_mode == "json":
                yield self._render_json_content(content)
            elif self.display_mode == "markdown":
                yield self._render_markdown_content(content)
            else:
                yield Static(content, classes="message-text")
        
        # Editing area (hidden by default)
        with Container(classes="editing-area"):
            yield TextArea(
                self.message.get("content", ""),
                id=f"edit-{self.message_id}",
                classes="edit-textarea"
            )
            with Horizontal(classes="edit-actions"):
                yield Button("Save", id=f"save-{self.message_id}", variant="success")
                yield Button("Cancel", id=f"cancel-{self.message_id}", variant="error")
        
        # Action buttons
        if self.show_actions:
            with Horizontal(classes="message-actions"):
                if self.editable:
                    yield Button("Edit", classes="action-button edit-button")
                yield Button("Copy", classes="action-button copy-button")
                yield Button("Delete", classes="action-button delete-button")
                
                # Special actions based on content
                if self.display_mode == "code":
                    yield Button("Run", classes="action-button run-button")
                if self.display_mode == "image":
                    yield Button("Save Image", classes="action-button save-image-button")
    
    def _render_image_content(self, content: str) -> Container:
        """Render image content."""
        with Container(classes="image-container"):
            if "data:image" in content:
                # Extract base64 data
                image_data = content.split(",")[1] if "," in content else content
                yield Static(f"[Image: {len(image_data)} bytes]", classes="image-placeholder")
            else:
                yield Static(content, classes="message-text")
    
    def _render_code_content(self, content: str) -> Container:
        """Render code content with syntax highlighting."""
        with Container(classes="code-block"):
            # Extract code blocks
            import re
            code_pattern = r"```(\w+)?\n(.*?)```"
            matches = re.findall(code_pattern, content, re.DOTALL)
            
            if matches:
                for lang, code in matches:
                    lang = lang or "python"
                    # Use Rich's Syntax for highlighting
                    syntax = Syntax(code, lang, theme="monokai", line_numbers=True)
                    yield Static(syntax, classes="code-syntax")
            else:
                yield Static(content, classes="message-text")
    
    def _render_json_content(self, content: str) -> Container:
        """Render JSON content with formatting."""
        with Container(classes="json-block"):
            try:
                data = json.loads(content)
                formatted = json.dumps(data, indent=2)
                syntax = Syntax(formatted, "json", theme="monokai")
                yield Static(syntax, classes="json-syntax")
            except:
                yield Static(content, classes="message-text")
    
    def _render_markdown_content(self, content: str) -> Container:
        """Render markdown content."""
        with Container(classes="markdown-container"):
            # Use Rich's Markdown renderer
            md = Markdown(content)
            yield Static(md, classes="markdown-content")
    
    async def action_edit_message(self) -> None:
        """Enter edit mode for the message."""
        self.is_editing = True
        edit_area = self.query_one(".editing-area")
        edit_area.add_class("active")
        
        # Hide the content area
        content_area = self.query_one(".message-content")
        content_area.add_class("hidden")
        
        # Focus the edit textarea
        textarea = self.query_one(f"#edit-{self.message_id}", TextArea)
        textarea.focus()
    
    async def action_save_edit(self) -> None:
        """Save edited message."""
        textarea = self.query_one(f"#edit-{self.message_id}", TextArea)
        new_content = textarea.text
        
        # Update message
        self.message["content"] = new_content
        self.display_mode = self._detect_display_mode()
        
        # Exit edit mode
        self.is_editing = False
        edit_area = self.query_one(".editing-area")
        edit_area.remove_class("active")
        
        # Show content area
        content_area = self.query_one(".message-content")
        content_area.remove_class("hidden")
        
        # Refresh the widget
        self.refresh()
        
        # Emit event
        self.post_message(self.MessageEdited(self.message_id, new_content))
    
    async def action_cancel_edit(self) -> None:
        """Cancel editing."""
        self.is_editing = False
        edit_area = self.query_one(".editing-area")
        edit_area.remove_class("active")
        
        # Show content area
        content_area = self.query_one(".message-content")
        content_area.remove_class("hidden")
    
    async def action_copy_message(self) -> None:
        """Copy message content to clipboard."""
        import pyperclip
        content = self.message.get("content", "")
        pyperclip.copy(content)
        self.app.notify("Message copied to clipboard")
    
    async def action_delete_message(self) -> None:
        """Delete this message."""
        self.post_message(self.MessageDeleted(self.message_id))
        await self.remove()
    
    # Custom events
    class MessageEdited(Message):
        """Event emitted when a message is edited."""
        def __init__(self, message_id: str, new_content: str):
            self.message_id = message_id
            self.new_content = new_content
            super().__init__()
    
    class MessageDeleted(Message):
        """Event emitted when a message is deleted."""
        def __init__(self, message_id: str):
            self.message_id = message_id
            super().__init__()
```

### 2. Chat Tab Container

For managing multiple chat sessions:

```python
# chat_tab_container.py
from typing import Dict, Optional, List
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import TabbedContent, TabPane, Button
from textual.reactive import reactive
import uuid

class ChatSession:
    """Represents a single chat session."""
    
    def __init__(self, session_id: str = None, title: str = "New Chat"):
        self.session_id = session_id or str(uuid.uuid4())
        self.title = title
        self.messages: List[Dict] = []
        self.metadata = {
            "created_at": None,
            "updated_at": None,
            "character": None,
            "system_prompt": None,
            "keywords": [],
        }
        self.is_temp = True
        self.is_modified = False
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to the session."""
        message = {
            "role": role,
            "content": content,
            "timestamp": self._get_timestamp(),
            **kwargs
        }
        self.messages.append(message)
        self.is_modified = True
        return message
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

class ChatTabContainer(Container):
    """Container for managing multiple chat tabs."""
    
    DEFAULT_CSS = """
    ChatTabContainer {
        height: 100%;
    }
    
    .tab-bar {
        height: 3;
        layout: horizontal;
        background: $boost;
        border-bottom: solid $primary;
    }
    
    .tab-button {
        width: auto;
        min-width: 15;
        max-width: 30;
        margin: 0 1;
    }
    
    .tab-button.active {
        background: $primary;
        text-style: bold;
    }
    
    .new-tab-button {
        width: 3;
        margin-left: 1;
    }
    
    .close-tab-button {
        width: 3;
        margin-left: auto;
    }
    """
    
    # Maximum number of tabs
    MAX_TABS = 10
    
    def __init__(self, app_instance, **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.sessions: Dict[str, ChatSession] = {}
        self.active_session_id: Optional[str] = None
        self.enhanced_mode = False  # Flag for enhanced features
        
        # Create initial session
        self._create_new_session()
    
    def _create_new_session(self, title: str = "New Chat") -> ChatSession:
        """Create a new chat session."""
        if len(self.sessions) >= self.MAX_TABS:
            self.app_instance.notify(
                f"Maximum {self.MAX_TABS} tabs reached",
                severity="warning"
            )
            return None
        
        session = ChatSession(title=title)
        self.sessions[session.session_id] = session
        self.active_session_id = session.session_id
        return session
    
    def compose(self) -> ComposeResult:
        """Compose the tab container."""
        with TabbedContent(id="chat-tabs"):
            for session_id, session in self.sessions.items():
                # Create tab for each session
                tab_id = f"tab-{session_id}"
                with TabPane(session.title, id=tab_id):
                    yield self._create_session_content(session)
    
    def _create_session_content(self, session: ChatSession) -> Container:
        """Create content for a session tab."""
        from textual.containers import VerticalScroll, Horizontal
        from textual.widgets import TextArea, Button, Static
        
        with Container(classes="session-content"):
            # Message area
            with VerticalScroll(id=f"chat-log-{session.session_id}"):
                # Render existing messages
                for message in session.messages:
                    yield ChatMessageEnhanced(
                        message,
                        message_id=f"msg-{session.session_id}-{len(session.messages)}"
                    )
            
            # Input area
            with Horizontal(id=f"input-area-{session.session_id}"):
                yield TextArea(
                    id=f"chat-input-{session.session_id}",
                    classes="chat-input"
                )
                yield Button(
                    "Send",
                    id=f"send-stop-chat-{session.session_id}",
                    classes="send-button"
                )
    
    def get_active_session(self) -> Optional[ChatSession]:
        """Get the currently active session."""
        if self.active_session_id:
            return self.sessions.get(self.active_session_id)
        return None
    
    def close_session(self, session_id: str) -> bool:
        """Close a chat session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Check if session has unsaved changes
        if session.is_modified and not session.is_temp:
            # Prompt user to save
            self.app_instance.push_screen(
                ConfirmDialog(
                    "Save changes?",
                    f"Session '{session.title}' has unsaved changes."
                ),
                lambda result: self._handle_close_confirmation(session_id, result)
            )
            return False
        
        # Remove session
        del self.sessions[session_id]
        
        # Switch to another session if this was active
        if session_id == self.active_session_id:
            if self.sessions:
                self.active_session_id = list(self.sessions.keys())[0]
            else:
                # Create new session if no sessions left
                self._create_new_session()
        
        self.refresh()
        return True
    
    def _handle_close_confirmation(self, session_id: str, save: bool):
        """Handle close confirmation result."""
        if save:
            # Save session before closing
            self.save_session(session_id)
        
        # Now close it
        del self.sessions[session_id]
        self.refresh()
    
    def save_session(self, session_id: str) -> bool:
        """Save a chat session."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        # Implement actual save logic here
        # This would typically save to database
        session.is_modified = False
        session.is_temp = False
        return True
```

### 3. Tool Message Widgets

For displaying tool calls and results:

```python
# tool_message_widgets.py
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Button, Collapsible
from textual.reactive import reactive
import json

class ToolCallMessage(Container):
    """Widget for displaying tool call messages."""
    
    DEFAULT_CSS = """
    ToolCallMessage {
        margin: 1 0;
        padding: 1;
        background: $surface;
        border: round $warning;
    }
    
    .tool-header {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
    }
    
    .tool-icon {
        width: 3;
        text-align: center;
    }
    
    .tool-name {
        width: 1fr;
        text-style: bold;
        color: $warning;
    }
    
    .tool-status {
        width: auto;
        padding: 0 1;
        background: $surface-darken-1;
        border: round $primary;
    }
    
    .tool-parameters {
        padding: 1;
        background: $surface-darken-1;
        border: round $primary;
        margin: 1 0;
    }
    
    .parameter-item {
        margin: 0 0 1 0;
    }
    
    .parameter-name {
        text-style: bold;
        color: $text-muted;
    }
    
    .parameter-value {
        margin-left: 2;
        color: $text;
    }
    """
    
    TOOL_ICONS = {
        "search": "🔍",
        "calculator": "🧮",
        "datetime": "🕐",
        "weather": "🌤️",
        "database": "🗄️",
        "api": "🔌",
        "file": "📁",
        "default": "🔧"
    }
    
    TOOL_STATUS = {
        "pending": ("Pending", "surface"),
        "running": ("Running...", "warning"),
        "success": ("Success", "success"),
        "error": ("Error", "error"),
        "cancelled": ("Cancelled", "surface-darken-1")
    }
    
    def __init__(
        self,
        tool_name: str,
        tool_id: str,
        parameters: Dict[str, Any],
        status: str = "pending",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_id = tool_id
        self.parameters = parameters
        self.status = reactive(status)
        self.result = None
        self.error = None
        self.collapsed = False
    
    def compose(self) -> ComposeResult:
        """Compose the tool call widget."""
        # Header
        with Horizontal(classes="tool-header"):
            # Tool icon
            icon = self.TOOL_ICONS.get(
                self.tool_name.lower(),
                self.TOOL_ICONS["default"]
            )
            yield Static(icon, classes="tool-icon")
            
            # Tool name
            yield Static(
                f"Tool: {self.tool_name}",
                classes="tool-name"
            )
            
            # Status indicator
            status_text, status_color = self.TOOL_STATUS[self.status]
            yield Static(
                status_text,
                classes=f"tool-status status-{self.status}"
            )
        
        # Parameters (collapsible)
        with Collapsible(title="Parameters", collapsed=self.collapsed):
            with Container(classes="tool-parameters"):
                if self.parameters:
                    for param_name, param_value in self.parameters.items():
                        with Container(classes="parameter-item"):
                            yield Static(
                                f"{param_name}:",
                                classes="parameter-name"
                            )
                            yield Static(
                                self._format_value(param_value),
                                classes="parameter-value"
                            )
                else:
                    yield Static("No parameters", classes="no-parameters")
        
        # Result or error (if available)
        if self.result is not None:
            with Container(classes="tool-result"):
                yield Static("Result:", classes="result-label")
                yield Static(
                    self._format_value(self.result),
                    classes="result-value"
                )
        
        if self.error is not None:
            with Container(classes="tool-error"):
                yield Static("Error:", classes="error-label")
                yield Static(
                    str(self.error),
                    classes="error-value"
                )
    
    def _format_value(self, value: Any) -> str:
        """Format a parameter or result value for display."""
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        elif isinstance(value, bool):
            return "✓" if value else "✗"
        elif value is None:
            return "null"
        else:
            return str(value)
    
    def update_status(self, status: str, result: Any = None, error: Any = None):
        """Update the tool call status."""
        self.status = status
        self.result = result
        self.error = error
        self.refresh()
    
    def watch_status(self, old_status: str, new_status: str):
        """React to status changes."""
        # Update status display
        status_widget = self.query_one(".tool-status", Static)
        status_text, _ = self.TOOL_STATUS[new_status]
        status_widget.update(status_text)
        
        # Update styling
        status_widget.remove_class(f"status-{old_status}")
        status_widget.add_class(f"status-{new_status}")


class ToolResultMessage(Container):
    """Widget for displaying tool results."""
    
    DEFAULT_CSS = """
    ToolResultMessage {
        margin: 1 0;
        padding: 1;
        background: $surface;
        border: round $success;
    }
    
    .result-header {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
        color: $success;
    }
    
    .result-content {
        padding: 1;
        background: $surface-darken-1;
        border: round $primary;
    }
    
    .result-actions {
        layout: horizontal;
        margin-top: 1;
    }
    """
    
    def __init__(
        self,
        tool_name: str,
        tool_id: str,
        result: Any,
        execution_time: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_id = tool_id
        self.result = result
        self.execution_time = execution_time
    
    def compose(self) -> ComposeResult:
        """Compose the tool result widget."""
        # Header
        with Horizontal(classes="result-header"):
            yield Static(
                f"✓ {self.tool_name} Result",
                classes="result-title"
            )
            if self.execution_time:
                yield Static(
                    f"({self.execution_time:.2f}s)",
                    classes="execution-time"
                )
        
        # Result content
        with Container(classes="result-content"):
            yield Static(
                self._format_result(self.result),
                classes="result-text"
            )
        
        # Actions
        with Horizontal(classes="result-actions"):
            yield Button("Copy", classes="copy-button action-button")
            yield Button("Use in Chat", classes="use-button action-button")
    
    def _format_result(self, result: Any) -> str:
        """Format the result for display."""
        if isinstance(result, str):
            return result
        elif isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        else:
            return str(result)
```

### 4. Voice Input Widget

```python
# voice_input_widget.py
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Button, ProgressBar
from textual.reactive import reactive
from textual.message import Message
from typing import Optional, Callable
import asyncio

class VoiceInputWidget(Container):
    """Widget for voice input with visual feedback."""
    
    DEFAULT_CSS = """
    VoiceInputWidget {
        layout: vertical;
        height: 10;
        padding: 1;
        background: $surface;
        border: round $error;
        margin: 1 0;
    }
    
    .voice-header {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
    }
    
    .voice-status {
        width: 1fr;
        text-style: bold;
    }
    
    .voice-close {
        width: 3;
    }
    
    .voice-visualization {
        height: 3;
        background: $surface-darken-1;
        border: round $primary;
        margin: 1 0;
    }
    
    .voice-level-bar {
        height: 1;
        background: $success;
    }
    
    .voice-transcript {
        padding: 1;
        background: $surface-darken-1;
        border: round $primary;
        min-height: 3;
    }
    
    .recording {
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    """
    
    def __init__(
        self,
        on_transcript: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.is_recording = reactive(False)
        self.audio_level = reactive(0.0)
        self.transcript = reactive("")
        self.partial_transcript = reactive("")
        self.on_transcript = on_transcript
        self.on_error = on_error
        self.recording_task = None
    
    def compose(self) -> ComposeResult:
        """Compose the voice input widget."""
        # Header
        with Horizontal(classes="voice-header"):
            yield Static(
                "🎤 Voice Input",
                classes="voice-status"
            )
            yield Button("✕", classes="voice-close", id="close-voice")
        
        # Audio level visualization
        with Container(classes="voice-visualization"):
            yield ProgressBar(
                total=100,
                show_eta=False,
                classes="voice-level-bar"
            )
        
        # Transcript display
        with Container(classes="voice-transcript"):
            yield Static(
                self.partial_transcript or "Listening...",
                id="transcript-display"
            )
        
        # Control buttons
        with Horizontal(classes="voice-controls"):
            yield Button(
                "Start" if not self.is_recording else "Stop",
                id="toggle-recording",
                variant="error" if self.is_recording else "success"
            )
            yield Button("Clear", id="clear-transcript")
            yield Button("Send", id="send-transcript", variant="primary")
    
    async def start_recording(self):
        """Start voice recording."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.add_class("recording")
        
        # Update button
        toggle_btn = self.query_one("#toggle-recording", Button)
        toggle_btn.label = "Stop"
        toggle_btn.variant = "error"
        
        # Start recording task
        self.recording_task = asyncio.create_task(self._recording_loop())
        
        # Notify
        self.post_message(self.RecordingStarted())
    
    async def stop_recording(self):
        """Stop voice recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.remove_class("recording")
        
        # Update button
        toggle_btn = self.query_one("#toggle-recording", Button)
        toggle_btn.label = "Start"
        toggle_btn.variant = "success"
        
        # Cancel recording task
        if self.recording_task:
            self.recording_task.cancel()
            self.recording_task = None
        
        # Finalize transcript
        if self.partial_transcript:
            self.transcript = self.partial_transcript
            if self.on_transcript:
                self.on_transcript(self.transcript)
        
        # Notify
        self.post_message(self.RecordingStopped(self.transcript))
    
    async def _recording_loop(self):
        """Main recording loop."""
        try:
            # This would interface with actual audio recording
            # For now, it's a placeholder
            while self.is_recording:
                # Simulate audio level changes
                import random
                self.audio_level = random.random() * 100
                
                # Update visualization
                level_bar = self.query_one(".voice-level-bar", ProgressBar)
                level_bar.advance(self.audio_level - level_bar.progress)
                
                # Simulate transcript updates
                # In real implementation, this would come from speech recognition
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            self.post_message(self.RecordingError(str(e)))
    
    def update_transcript(self, text: str, is_final: bool = False):
        """Update the transcript display."""
        if is_final:
            self.transcript = text
            self.partial_transcript = ""
        else:
            self.partial_transcript = text
        
        # Update display
        transcript_display = self.query_one("#transcript-display", Static)
        transcript_display.update(text)
    
    # Custom messages
    class RecordingStarted(Message):
        """Emitted when recording starts."""
        pass
    
    class RecordingStopped(Message):
        """Emitted when recording stops."""
        def __init__(self, transcript: str):
            self.transcript = transcript
            super().__init__()
    
    class RecordingError(Message):
        """Emitted when recording encounters an error."""
        def __init__(self, error: str):
            self.error = error
            super().__init__()

class VoiceInputMessage(Message):
    """Message for voice input events."""
    def __init__(self, text: str, is_final: bool = False):
        self.text = text
        self.is_final = is_final
        super().__init__()
```

## CSS Files Structure

### Main CSS (_chat.tcss)

[Already included in previous document]

### Unified Sidebar CSS (_unified_sidebar.tcss)

[Already included in previous document]

## Event Handler Modules

### chat_events.py Structure

```python
# Event_Handlers/Chat_Events/chat_events.py

class ChatEvents:
    """Central chat event handler."""
    
    async def handle_chat_send_button_pressed(self, app_instance, event):
        """Handle send button press."""
        # Get chat input
        chat_input = app_instance.query_one("#chat-input", TextArea)
        message = chat_input.text
        
        if not message:
            return
        
        # Check for attachments
        chat_window = app_instance.query_one("ChatWindowEnhanced")
        attachment = chat_window.get_pending_attachment()
        
        # Create message with attachment if present
        # ... implementation
    
    async def handle_stop_chat_generation_pressed(self, app_instance, event):
        """Stop ongoing generation."""
        # Cancel current worker
        if hasattr(app_instance, 'current_chat_worker'):
            if app_instance.current_chat_worker:
                app_instance.current_chat_worker.cancel()
    
    # ... 30+ more handlers

chat_events = ChatEvents()
```

## Configuration Files

### config.toml Structure

```toml
[chat_defaults]
enable_tabs = false
default_provider = "openai"
default_model = "gpt-4"
temperature = 0.7
max_tokens = 2000

[chat.images]
show_attach_button = true
max_image_size = 10485760  # 10MB
supported_formats = ["png", "jpg", "jpeg", "gif", "webp"]

[chat.voice]
show_mic_button = true
transcription_provider = "whisper"
language = "en"

[chat_sidebar]
width = 30
position = "left"
active_tab = "session"
advanced_mode = false

[rag]
enabled = false
pipeline = "none"
chunk_size = 1000
overlap = 200
```

## Issues Summary

### Critical Architecture Problems

1. **Monolithic Design**: 1000+ line files doing everything
2. **Event Handler Chaos**: 40+ button mappings in central dispatcher
3. **State Management**: No single source of truth
4. **Tight Coupling**: Components directly reference each other
5. **Testing Nightmare**: Can't test components in isolation
6. **Legacy Debt**: Multiple systems for same functionality

### Performance Issues

1. **Memory Leaks**: Event handlers not properly cleaned up
2. **Unnecessary Re-renders**: No optimization for reactive updates
3. **Large File Handling**: No streaming for attachments
4. **Voice Input**: Always loaded even when not used

### UX Problems

1. **Inconsistent Patterns**: Different interaction models
2. **No Keyboard Navigation**: Mouse-only for many features
3. **Poor Error Handling**: Generic error messages
4. **No Progress Indication**: Long operations block UI

This complete documentation provides the foundation for understanding the current implementation and planning a proper rebuild.