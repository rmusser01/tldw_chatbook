# Chat Window Rebuild Strategy - Textual Framework Aligned

## Executive Summary

Based on the official Textual documentation, we need to rebuild the chat window following Textual's patterns: reactive attributes with proper typing, correct widget composition, CSS-driven layouts, and proper event handling via messages. The current implementation fights against Textual instead of working with it.

## Documentation References
- [Textual App Basics](https://textual.textualize.io/guide/app/)
- [Reactive Attributes](https://textual.textualize.io/guide/reactivity/)
- [CSS in Textual](https://textual.textualize.io/guide/CSS/)
- [Workers](https://textual.textualize.io/guide/workers/)
- [Testing](https://textual.textualize.io/guide/testing/)

## Key Textual Principles to Follow

### 1. **Reactive Programming First**
- Use `reactive()` attributes for state management
- Leverage `watch_` methods for state changes
- Use `validate_` methods for data validation
- Avoid manual DOM manipulation

### 2. **Message-Based Communication**
- Custom messages for component communication
- Proper event bubbling and handling
- Use `@on` decorators for clean event binding

### 3. **CSS-Driven Layouts**
- All styling in CSS, not Python
- Use proper selectors and pseudo-classes
- Leverage Textual's layout systems (grid, dock, flex)

### 4. **Proper Widget Composition**
- `compose()` method for widget tree
- Avoid dynamic widget creation when possible
- Use containers for layout structure

## Revised Architecture

### Core Structure (Aligned with Textual)

```
tldw_chatbook/
├── chat/
│   ├── app.py                 # Main ChatApp class
│   ├── screens/
│   │   ├── chat_screen.py     # Main chat screen
│   │   └── modals.py          # Modal dialogs
│   ├── widgets/
│   │   ├── message_list.py    # Message display widget
│   │   ├── message_item.py    # Individual message
│   │   ├── chat_input.py      # Input area widget
│   │   └── chat_sidebar.py    # Sidebar widget
│   ├── messages.py            # Custom Textual messages
│   ├── models.py              # Data models (Pydantic)
│   └── styles/
│       ├── chat.tcss          # Main styles
│       └── themes/            # Theme variations
```

## Implementation Plan

### Phase 1: Core Chat Application

#### 1.1 Main App Class (Corrected per Official Docs)

```python
# chat/app.py
from textual.app import App
from textual.binding import Binding
from textual.reactive import reactive
from typing import Optional
from .screens.chat_screen import ChatScreen
from .models import ChatSession, Settings

class ChatApp(App):
    """Main chat application following Textual patterns.
    
    References:
    - App basics: https://textual.textualize.io/guide/app/
    - Reactive attributes: https://textual.textualize.io/guide/reactivity/#reactive-attributes
    """
    
    # CSS configuration - per https://textual.textualize.io/guide/CSS/#loading-css
    # CSS_PATH must be in same directory or use CSS string
    CSS = """
    ChatApp {
        background: $surface;
    }
    """
    
    TITLE = "Chat Interface"
    
    # Key bindings - per https://textual.textualize.io/guide/input/#bindings
    BINDINGS = [
        Binding("ctrl+n", "new_session", "New Chat"),
        Binding("ctrl+s", "save_session", "Save"),
        Binding("ctrl+o", "open_session", "Open"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+\\", "toggle_sidebar", "Toggle Sidebar"),
    ]
    
    # Reactive state with proper typing - per https://textual.textualize.io/guide/reactivity/#reactive-descriptors
    current_session: reactive[Optional[ChatSession]] = reactive(None, init=False)
    settings: reactive[Settings] = reactive(Settings)
    sidebar_visible: reactive[bool] = reactive(True)
    
    def on_mount(self):
        """Initialize app after mounting.
        Per https://textual.textualize.io/guide/app/#mounting
        Apps push screens, not compose them."""
        # Push the main screen
        self.push_screen(ChatScreen())
        
        # Create initial session
        self.current_session = ChatSession()
        
    def watch_current_session(self, old_session: Optional[ChatSession], new_session: Optional[ChatSession]):
        """React to session changes.
        Per https://textual.textualize.io/guide/reactivity/#watch-methods
        Watchers receive both old and new values."""
        # Update title
        self.title = f"Chat - {new_session.title if new_session else 'No Session'}"
        
        # Post message to current screen
        if self.screen:
            self.screen.post_message(SessionChanged(new_session))
    
    def action_new_session(self):
        """Create new chat session.
        Per https://textual.textualize.io/guide/actions/"""
        self.current_session = ChatSession()
    
    def action_toggle_sidebar(self):
        """Toggle sidebar visibility."""
        self.sidebar_visible = not self.sidebar_visible
```

#### 1.2 Custom Messages (Textual Pattern)

```python
# chat/messages.py
from textual.message import Message
from typing import Optional
from .models import ChatMessage, ChatSession

class SessionChanged(Message):
    """Posted when active session changes."""
    def __init__(self, session: Optional[ChatSession]):
        super().__init__()
        self.session = session

class MessageSent(Message):
    """Posted when user sends a message."""
    def __init__(self, content: str, attachments: list = None):
        super().__init__()
        self.content = content
        self.attachments = attachments or []

class MessageReceived(Message):
    """Posted when LLM responds."""
    def __init__(self, message: ChatMessage):
        super().__init__()
        self.message = message

class StreamingUpdate(Message):
    """Posted during streaming responses."""
    def __init__(self, content: str, done: bool = False):
        super().__init__()
        self.content = content
        self.done = done
```

### Phase 2: Main Screen Implementation

#### 2.1 Chat Screen (Corrected Screen Pattern)

```python
# chat/screens/chat_screen.py
from textual.screen import Screen
from textual.containers import Container, Horizontal
from textual import on, work
from ..widgets import MessageList, ChatInput, ChatSidebar
from ..messages import MessageSent, SessionChanged

class ChatScreen(Screen):
    """Main chat screen following Textual patterns.
    
    References:
    - Screens: https://textual.textualize.io/guide/screens/
    - CSS: https://textual.textualize.io/guide/CSS/#css-files
    """
    
    # CSS as string per https://textual.textualize.io/guide/CSS/#inline-css
    CSS = """
    ChatScreen {
        layout: horizontal;
    }
    
    #chat-container {
        width: 1fr;
        height: 100%;
    }
    
    #sidebar {
        dock: left;
        width: 30;
        transition: offset 200ms in_out_cubic;
    }
    
    #sidebar.-hidden {
        offset-x: -100%;
    }
    """
    
    def compose(self):
        """Compose the screen layout.
        Per https://textual.textualize.io/guide/screens/#composing-screens"""
        # Sidebar (can be hidden via CSS)
        yield ChatSidebar(id="sidebar")
        
        # Main chat area
        with Container(id="chat-container"):
            yield MessageList(id="message-list")
            yield ChatInput(id="chat-input")
    
    def on_mount(self):
        """Set up screen after mounting.
        Per https://textual.textualize.io/events/mount/"""
        # Check app state for sidebar visibility
        self.update_sidebar_visibility()
    
    def update_sidebar_visibility(self):
        """Update sidebar visibility via CSS classes.
        Per https://textual.textualize.io/guide/CSS/#setting-classes"""
        sidebar = self.query_one("#sidebar")
        if self.app.sidebar_visible:
            sidebar.remove_class("-hidden")
        else:
            sidebar.add_class("-hidden")
    
    @on(MessageSent)
    async def handle_message_sent(self, event: MessageSent):
        """Handle message sent from input widget.
        Per https://textual.textualize.io/guide/events/#custom-messages"""
        # Add user message
        message_list = self.query_one("#message-list", MessageList)
        message_list.add_user_message(event.content)
        
        # Process with LLM using worker
        self.process_message(event.content)
    
    @work(exclusive=True)
    async def process_message(self, content: str):
        """Process message with LLM.
        Per https://textual.textualize.io/guide/workers/#thread-workers
        Workers don't return values - use callbacks."""
        try:
            # Simulate LLM call
            response = await self.call_llm(content)
            # Update UI from worker
            self.call_from_thread(self.add_assistant_response, response)
        except Exception as e:
            self.call_from_thread(self.notify, f"Error: {e}", severity="error")
    
    def add_assistant_response(self, response: str):
        """Add assistant response to message list."""
        message_list = self.query_one("#message-list", MessageList)
        message_list.add_assistant_message(response)
    
    @on(SessionChanged)
    def handle_session_changed(self, event: SessionChanged):
        """Handle session change from app."""
        # Update message list
        message_list = self.query_one("#message-list", MessageList)
        message_list.load_session(event.session)
```

### Phase 3: Widget Implementation (Textual Patterns)

#### 3.1 Message List Widget (Corrected)

```python
# chat/widgets/message_list.py
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual import on
from typing import List, Optional
from .message_item import MessageItem
from ..messages import MessageReceived, StreamingUpdate
from ..models import ChatSession, ChatMessage

class MessageList(VerticalScroll):
    """Message list following Textual reactive patterns.
    
    References:
    - Reactive attributes: https://textual.textualize.io/guide/reactivity/#reactive-attributes
    - Containers: https://textual.textualize.io/widgets/verticalscroll/
    """
    
    CSS = """
    MessageList {
        height: 1fr;
        padding: 1;
        background: $surface;
    }
    
    MessageList:focus {
        border: solid $accent;
    }
    """
    
    # Reactive state with proper typing per https://textual.textualize.io/guide/reactivity/#reactive-descriptors
    session: reactive[Optional[ChatSession]] = reactive(None, init=False)
    messages: reactive[List[ChatMessage]] = reactive(list, recompose=True)
    is_streaming: reactive[bool] = reactive(False)
    
    def compose(self):
        """Initial composition - called when recompose=True triggers.
        Per https://textual.textualize.io/guide/reactivity/#recompose"""
        for message in self.messages:
            yield MessageItem(message)
    
    def watch_session(self, old_session: Optional[ChatSession], new_session: Optional[ChatSession]):
        """React to session changes.
        Per https://textual.textualize.io/guide/reactivity/#watch-methods"""
        if new_session:
            self.messages = new_session.messages
        else:
            self.messages = []
    
    def watch_messages(self):
        """React to message list changes.
        With recompose=True, compose() is called automatically.
        Per https://textual.textualize.io/guide/reactivity/#recompose"""
        # Scroll to bottom after recompose
        self.call_after_refresh(self.scroll_end)
    
    def add_user_message(self, content: str):
        """Add a user message to the list."""
        message = ChatMessage(role="user", content=content)
        # Create new list to trigger reactive update
        self.messages = [*self.messages, message]
    
    def add_assistant_message(self, content: str):
        """Add an assistant message."""
        message = ChatMessage(role="assistant", content=content)
        self.messages = [*self.messages, message]
    
    @on(MessageReceived)
    def handle_message_received(self, event: MessageReceived):
        """Handle new message event.
        Per https://textual.textualize.io/guide/events/#handler-methods"""
        # Update reactive list - triggers recompose
        self.messages = [*self.messages, event.message]
    
    @on(StreamingUpdate)
    def handle_streaming_update(self, event: StreamingUpdate):
        """Handle streaming updates for progressive display."""
        if not self.is_streaming:
            # Start new streaming message
            self.is_streaming = True
            streaming_msg = ChatMessage(role="assistant", content="")
            self.messages = [*self.messages, streaming_msg]
        
        # Update last message widget directly for performance
        if self.messages and self.children:
            last_widget = self.children[-1]
            if isinstance(last_widget, MessageItem):
                last_widget.update_content(event.content)
        
        if event.done:
            self.is_streaming = False
```

#### 3.2 Chat Input Widget (Proper Textual Input)

```python
# chat/widgets/chat_input.py
from textual.containers import Horizontal
from textual.widgets import TextArea, Button
from textual.reactive import reactive
from textual import on, work
from textual.validation import Validator
from ..messages import MessageSent

class ChatInput(Horizontal):
    """Chat input following Textual patterns."""
    
    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        max-height: 12;
        padding: 1;
        dock: bottom;
        background: $panel;
    }
    
    #input-area {
        width: 1fr;
        min-height: 3;
        max-height: 10;
    }
    
    .input-button {
        width: auto;
        margin: 0 1;
    }
    
    #send-button:disabled {
        opacity: 0.5;
    }
    """
    
    # Reactive state
    is_valid = reactive(True)
    is_sending = reactive(False)
    has_attachment = reactive(False)
    
    def compose(self):
        """Compose the input area."""
        yield TextArea(
            id="input-area",
            placeholder="Type a message...",
            tab_behavior="focus"
        )
        
        yield Button(
            "📎",
            id="attach-button",
            classes="input-button",
            tooltip="Attach file"
        )
        
        yield Button(
            "Send",
            id="send-button",
            classes="input-button",
            variant="primary"
        )
    
    def on_mount(self):
        """Set up input area."""
        # Focus input on mount
        self.query_one("#input-area").focus()
    
    @on(TextArea.Changed, "#input-area")
    def validate_input(self, event: TextArea.Changed):
        """Validate input as user types."""
        content = event.text_area.text.strip()
        self.is_valid = bool(content)
        
        # Update send button state
        send_button = self.query_one("#send-button", Button)
        send_button.disabled = not self.is_valid or self.is_sending
    
    @on(Button.Pressed, "#send-button")
    async def send_message(self):
        """Send the message."""
        if not self.is_valid or self.is_sending:
            return
        
        input_area = self.query_one("#input-area", TextArea)
        content = input_area.text.strip()
        
        if not content:
            return
        
        # Update state
        self.is_sending = True
        
        # Post message event - let app handle the logic
        attachments = self.get_attachments() if self.has_attachment else []
        self.post_message(MessageSent(content, attachments))
        
        # Clear input
        input_area.clear()
        self.has_attachment = False
        
        # Reset state
        self.is_sending = False
    
    @on(Button.Pressed, "#attach-button")
    async def handle_attachment(self):
        """Handle file attachment."""
        # Push a file picker screen
        from ..screens.file_picker import FilePickerScreen
        
        result = await self.app.push_screen_wait(FilePickerScreen())
        if result:
            self.add_attachment(result)
    
    def add_attachment(self, file_path: str):
        """Add file attachment."""
        self.has_attachment = True
        # Update button to show attachment
        attach_button = self.query_one("#attach-button", Button)
        attach_button.label = "📎✓"
```

#### 3.3 Sidebar Widget (Proper Container Pattern)

```python
# chat/widgets/chat_sidebar.py
from textual.containers import Container, VerticalScroll
from textual.widgets import Static, Button, Input, Select, TabbedContent, TabPane
from textual.reactive import reactive
from textual import on

class ChatSidebar(Container):
    """Sidebar using Textual's tabbed interface."""
    
    DEFAULT_CSS = """
    ChatSidebar {
        background: $panel;
        border-right: solid $primary;
    }
    
    .sidebar-section {
        padding: 1;
        margin-bottom: 1;
    }
    
    .sidebar-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .sidebar-input {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }
    """
    
    def compose(self):
        """Compose sidebar with tabs."""
        with TabbedContent():
            with TabPane("Session", id="session-tab"):
                yield self.compose_session_tab()
            
            with TabPane("Settings", id="settings-tab"):
                yield self.compose_settings_tab()
            
            with TabPane("History", id="history-tab"):
                yield self.compose_history_tab()
    
    def compose_session_tab(self):
        """Compose session management tab."""
        with VerticalScroll():
            with Container(classes="sidebar-section"):
                yield Static("Current Session", classes="sidebar-title")
                yield Input(
                    placeholder="Session title...",
                    id="session-title",
                    classes="sidebar-input"
                )
                
                yield Button("New Session", id="new-session", variant="primary")
                yield Button("Save Session", id="save-session")
                yield Button("Load Session", id="load-session")
    
    def compose_settings_tab(self):
        """Compose settings tab."""
        with VerticalScroll():
            with Container(classes="sidebar-section"):
                yield Static("LLM Settings", classes="sidebar-title")
                
                # Provider selection
                providers = [("openai", "OpenAI"), ("anthropic", "Anthropic")]
                yield Select(
                    options=providers,
                    id="provider-select",
                    value="openai"
                )
                
                # Model selection
                yield Select(
                    options=[],
                    id="model-select"
                )
                
                # Temperature slider would go here
                yield Input(
                    value="0.7",
                    id="temperature",
                    placeholder="Temperature (0-1)",
                    classes="sidebar-input"
                )
    
    def compose_history_tab(self):
        """Compose history tab."""
        with VerticalScroll():
            yield Static("Recent Conversations", classes="sidebar-title")
            # History items would be loaded here
    
    @on(Select.Changed, "#provider-select")
    def handle_provider_change(self, event: Select.Changed):
        """Update models when provider changes."""
        # This would load models for the selected provider
        model_select = self.query_one("#model-select", Select)
        
        # Example model loading
        if event.value == "openai":
            models = [("gpt-4", "GPT-4"), ("gpt-3.5", "GPT-3.5")]
        else:
            models = [("claude-3", "Claude 3")]
        
        model_select.set_options(models)
```

### Phase 4: CSS Styling (Textual Way)

```css
/* chat/styles/chat.tcss */

/* Use Textual's design system */
$primary: #0066cc;
$surface: #1e1e1e;
$panel: #2a2a2a;
$text: #ffffff;
$text-muted: #888888;

/* App-level styling */
ChatApp {
    background: $surface;
}

/* Screen layouts using Textual's layout system */
ChatScreen {
    layout: horizontal;
}

/* Message styling with proper selectors */
MessageItem {
    margin: 1 0;
    padding: 1;
    background: $panel;
    border: round $primary;
}

MessageItem.user {
    align: right;
    background: $primary 20%;
}

MessageItem.assistant {
    align: left;
}

MessageItem:hover {
    background: $panel-lighten-1;
}

/* Input area with proper height management */
ChatInput {
    height: auto;
    min-height: 5;
    max-height: 15;
}

#input-area {
    width: 1fr;
    border: solid $primary;
}

#input-area:focus {
    border: solid $accent;
}

/* Responsive design using Textual's units */
@media (max-width: 100) {
    #sidebar {
        width: 25;
    }
}

@media (max-width: 80) {
    #sidebar {
        display: none;
    }
}
```

### Phase 5: Worker Pattern for Async Operations (Corrected)

```python
# Using Textual's worker pattern for LLM calls
# Per https://textual.textualize.io/guide/workers/
from textual.worker import work, Worker, WorkerState

class ChatScreen(Screen):
    
    @work(exclusive=True)
    async def send_to_llm(self, message: str):
        """Send message to LLM in background worker.
        Per https://textual.textualize.io/guide/workers/#thread-workers
        Workers should not return values - use callbacks instead."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": message}]},
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                result = response.json()["choices"][0]["message"]["content"]
                
                # Use call_from_thread to update UI from worker
                # Per https://textual.textualize.io/guide/workers/#thread-safe-updates
                self.call_from_thread(self.add_assistant_message, result)
        except Exception as e:
            self.call_from_thread(self.notify, f"Error: {str(e)}", severity="error")
    
    @on(MessageSent)
    def handle_message_sent(self, event: MessageSent):
        """Handle message sent - use worker for LLM call."""
        # Add user message immediately
        self.add_user_message(event.content)
        
        # Start worker for LLM response
        self.send_to_llm(event.content)
    
    def add_assistant_message(self, content: str):
        """Add assistant message - called from worker thread.
        This method is thread-safe when called via call_from_thread."""
        message_list = self.query_one("#message-list", MessageList)
        message_list.add_assistant_message(content)
```

### Phase 6: Testing (Corrected for Textual Testing Framework)

```python
# tests/test_chat.py
# Per https://textual.textualize.io/guide/testing/
import pytest
from chat.app import ChatApp

@pytest.mark.asyncio
async def test_send_message():
    """Test sending a message using Textual's testing framework.
    Per https://textual.textualize.io/guide/testing/#testing-apps"""
    app = ChatApp()
    async with app.run_test() as pilot:
        # App pushes screen on mount, wait for it
        await pilot.pause()
        
        # Type in input area
        await pilot.click("#text-input")
        await pilot.press(*"Hello, world!")
        
        # Verify send button is enabled
        send_btn = app.query_one("#send-btn")
        assert not send_btn.disabled
        
        # Send message
        await pilot.click("#send-btn")
        
        # Wait for reactive update
        await pilot.pause()
        
        # Verify message appears
        messages = app.query("MessageItem")
        assert len(messages) >= 1
        
        # Verify input was cleared
        text_input = app.query_one("#text-input")
        assert text_input.text == ""

@pytest.mark.asyncio 
async def test_sidebar_toggle():
    """Test sidebar visibility toggle.
    Per https://textual.textualize.io/guide/testing/#pilot-object"""
    app = ChatApp()
    async with app.run_test() as pilot:
        await pilot.pause()  # Let screen mount
        
        # Get sidebar
        sidebar = app.query_one("#sidebar")
        
        # Check initial state (visible)
        assert "-hidden" not in sidebar.classes
        
        # Toggle sidebar with keyboard
        await pilot.press("ctrl+\\")
        await pilot.pause()
        
        # Check sidebar is hidden
        assert "-hidden" in sidebar.classes
        
        # Toggle again
        await pilot.press("ctrl+\\") 
        await pilot.pause()
        
        # Check sidebar is visible again
        assert "-hidden" not in sidebar.classes
```

## Key Corrections and Documentation References

### 1. **CSS Loading** 
Per [CSS Guide](https://textual.textualize.io/guide/CSS/#loading-css):
- ✅ Use `CSS` string attribute or `CSS_PATH` in same directory
- ❌ Avoid nested paths in `CSS_PATH`
- ✅ Use inline CSS for widget-specific styles

### 2. **Reactive Attributes**
Per [Reactivity Guide](https://textual.textualize.io/guide/reactivity/#reactive-descriptors):
- ✅ Include type hints: `reactive[Type]`
- ✅ Use `init=False` to prevent initial watcher call
- ✅ Watchers receive old and new values (except with `recompose=True`)
- ❌ Don't use `reactive(None)` without proper typing

### 3. **Worker Pattern**
Per [Workers Guide](https://textual.textualize.io/guide/workers/#thread-workers):
- ✅ Use `call_from_thread()` to update UI from workers
- ✅ Workers are fire-and-forget or use callbacks
- ❌ Workers don't return values directly

### 4. **Screen Management**
Per [Screens Guide](https://textual.textualize.io/guide/screens/):
- ✅ App uses `push_screen()`, `pop_screen()`, `switch_screen()`
- ✅ Screens compose widgets, not other screens
- ❌ Don't yield screens in `App.compose()`

### 5. **Message Handling**
Per [Events Guide](https://textual.textualize.io/guide/events/#custom-messages):
- ✅ Messages bubble up by default
- ✅ Use `event.stop()` to prevent bubbling
- ✅ Use `@on()` decorator for clean handling

## Migration Strategy

### Step 1: Create New Structure Alongside Old
```bash
tldw_chatbook/
├── UI/                    # Old code
│   └── Chat_Window_Enhanced.py
└── chat/                  # New code
    ├── app.py
    └── widgets/
```

### Step 2: Feature Flag Switch
```python
# In main app
if get_cli_setting("use_new_chat", False):
    from chat.app import ChatApp
    self.chat_widget = ChatApp()
else:
    from UI.Chat_Window_Enhanced import ChatWindowEnhanced
    self.chat_widget = ChatWindowEnhanced()
```

### Step 3: Incremental Migration
1. Start with MessageItem widget
2. Move to MessageList
3. Then ChatInput
4. Finally the main container

### Step 4: Data Migration
- Keep same database schema
- Update data access patterns
- Maintain backward compatibility

## Success Metrics

### Code Quality
- Follow Textual patterns consistently
- No widget > 200 lines
- All styling in CSS
- Proper use of reactive attributes

### Performance
- Smooth scrolling at 60fps
- No UI blocking during LLM calls
- Efficient reactive updates
- Memory usage < 50MB

### User Experience
- All features keyboard accessible
- Proper focus management
- Clear visual feedback
- Responsive to terminal resize

## Summary of Critical Changes

### Must Fix from Original Implementation:
1. **App doesn't compose screens** - Use `push_screen()` in `on_mount()`
2. **Reactive attributes need typing** - Use `reactive[Type]` syntax
3. **Workers don't return values** - Use `call_from_thread()` for UI updates
4. **CSS paths must be local** - Use inline CSS or file in same directory
5. **Watchers receive two parameters** - old and new values (except with `recompose=True`)

### Official Documentation Links for Implementation:
- [App Structure](https://textual.textualize.io/guide/app/)
- [Reactive Attributes](https://textual.textualize.io/guide/reactivity/)
- [CSS Styling](https://textual.textualize.io/guide/CSS/)
- [Workers](https://textual.textualize.io/guide/workers/)
- [Screens](https://textual.textualize.io/guide/screens/)
- [Events and Messages](https://textual.textualize.io/guide/events/)
- [Testing](https://textual.textualize.io/guide/testing/)
- [Widget Reference](https://textual.textualize.io/widgets/)

## Conclusion

This corrected strategy aligns precisely with Textual's official documentation. By following these patterns - properly typed reactive attributes, correct screen management, worker callbacks, and inline CSS - we create a maintainable chat interface that leverages Textual's strengths rather than fighting against them.