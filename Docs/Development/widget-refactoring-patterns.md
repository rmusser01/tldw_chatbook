# Widget Refactoring Patterns
## From Direct Manipulation to Reactive Programming

**Date:** August 15, 2025  
**Purpose:** Guide for refactoring widgets from direct manipulation to Textual's reactive patterns

---

## Anti-Pattern: Direct Widget Manipulation

The codebase currently has 6,149 instances of direct widget manipulation. This violates Textual's reactive programming model and causes:
- Race conditions
- Stale UI states
- Complex debugging
- Poor testability
- Memory leaks

---

## Pattern 1: Query → Reactive Attribute

### ❌ BEFORE (Direct Manipulation)
```python
class ChatWindow(Container):
    def update_send_button(self, is_streaming: bool):
        # Direct query and manipulation
        button = self.query_one("#send-stop-chat", Button)
        if is_streaming:
            button.label = "Stop"
            button.variant = "error"
        else:
            button.label = "Send"
            button.variant = "primary"
```

### ✅ AFTER (Reactive Pattern)
```python
class ChatWindow(Container):
    # Reactive state
    is_streaming = reactive(False)
    
    def compose(self) -> ComposeResult:
        # Button reacts to state changes
        yield Button(
            "Send",
            id="send-stop-chat",
            variant="primary"
        )
    
    def watch_is_streaming(self, is_streaming: bool) -> None:
        """Automatically called when is_streaming changes."""
        button = self.query_one("#send-stop-chat", Button)
        if is_streaming:
            button.label = "Stop"
            button.variant = "error"
        else:
            button.label = "Send"
            button.variant = "primary"
    
    def start_streaming(self):
        # Just change the reactive attribute
        self.is_streaming = True
    
    def stop_streaming(self):
        self.is_streaming = False
```

---

## Pattern 2: Computed Properties for Derived State

### ❌ BEFORE (Manual Updates)
```python
class ChatWindow(Container):
    def update_attachment_indicator(self):
        indicator = self.query_one("#image-attachment-indicator")
        attach_button = self.query_one("#attach-image")
        
        if self.pending_image:
            indicator.update(f"📎 {self.pending_image.name}")
            indicator.add_class("visible")
            attach_button.variant = "success"
        else:
            indicator.update("")
            indicator.remove_class("visible")
            attach_button.variant = "default"
```

### ✅ AFTER (Computed Reactive)
```python
class ChatWindow(Container):
    pending_image = reactive(None)
    
    @property
    def attachment_text(self) -> str:
        """Computed property for attachment display."""
        if self.pending_image:
            return f"📎 {self.pending_image.name}"
        return ""
    
    @property
    def has_attachment(self) -> bool:
        """Computed property for attachment state."""
        return self.pending_image is not None
    
    def compose(self) -> ComposeResult:
        yield Static(
            "",
            id="image-attachment-indicator",
            classes="hidden"  # Initially hidden
        )
        yield Button(
            "Attach",
            id="attach-image",
            variant="default"
        )
    
    def watch_pending_image(self, image) -> None:
        """React to attachment changes."""
        indicator = self.query_one("#image-attachment-indicator", Static)
        attach_button = self.query_one("#attach-image", Button)
        
        indicator.update(self.attachment_text)
        indicator.set_class(not self.has_attachment, "hidden")
        attach_button.variant = "success" if self.has_attachment else "default"
```

---

## Pattern 3: Message-Based Communication

### ❌ BEFORE (Direct Cross-Widget Access)
```python
class ChatWindow(Container):
    def send_message(self):
        # Directly accessing other widgets
        chat_log = self.app_instance.query_one("#chat-log", VerticalScroll)
        provider = self.app_instance.query_one("#chat-api-provider", Select)
        model = self.app_instance.query_one("#chat-api-model", Select)
        
        message = ChatMessage(
            content=self.get_input_text(),
            provider=provider.value,
            model=model.value
        )
        chat_log.mount(message)
```

### ✅ AFTER (Message-Based)
```python
from textual.message import Message

class SendChatMessage(Message):
    """Message to send chat content."""
    def __init__(self, content: str, provider: str, model: str):
        super().__init__()
        self.content = content
        self.provider = provider
        self.model = model

class ChatWindow(Container):
    # Local state only
    current_provider = reactive("openai")
    current_model = reactive("gpt-4")
    
    def send_message(self):
        # Post message instead of direct manipulation
        self.post_message(SendChatMessage(
            content=self.get_input_text(),
            provider=self.current_provider,
            model=self.current_model
        ))

class ChatLog(VerticalScroll):
    @on(SendChatMessage)
    def handle_new_message(self, message: SendChatMessage):
        """React to new chat messages."""
        chat_message = ChatMessage(
            content=message.content,
            provider=message.provider,
            model=message.model
        )
        self.mount(chat_message)
```

---

## Pattern 4: Recompose for Dynamic UI

### ❌ BEFORE (Manual DOM Manipulation)
```python
class ChatWindow(Container):
    def toggle_sidebar(self):
        sidebar = self.query_one("#chat-sidebar")
        if sidebar.has_class("hidden"):
            sidebar.remove_class("hidden")
            sidebar.display = True
        else:
            sidebar.add_class("hidden")
            sidebar.display = False
```

### ✅ AFTER (Recompose)
```python
class ChatWindow(Container):
    show_sidebar = reactive(True, recompose=True)
    
    def compose(self) -> ComposeResult:
        # Conditionally compose based on state
        if self.show_sidebar:
            yield Container(id="chat-sidebar")
        
        yield Container(id="chat-main")
    
    def toggle_sidebar(self):
        # Just toggle the reactive attribute
        self.show_sidebar = not self.show_sidebar
        # UI automatically recomposes
```

---

## Pattern 5: Worker Pattern for Async Operations

### ❌ BEFORE (Blocking UI)
```python
class ChatWindow(Container):
    async def load_conversation(self, conv_id: int):
        # UI freezes during database access
        messages = await self.db.get_messages(conv_id)
        
        chat_log = self.query_one("#chat-log")
        chat_log.clear()
        
        for msg in messages:
            widget = ChatMessage(msg)
            chat_log.mount(widget)
```

### ✅ AFTER (Worker Pattern)
```python
from textual.worker import work

class ChatWindow(Container):
    messages = reactive([], recompose=True)
    is_loading = reactive(False)
    
    @work(thread=True)
    def load_conversation(self, conv_id: int):
        """Load conversation in background."""
        # This runs in a thread, won't block UI
        messages = self.db.get_messages(conv_id)  # Blocking DB call OK here
        
        # Update UI from thread
        self.call_from_thread(self.update_messages, messages)
    
    def update_messages(self, messages):
        """Update reactive attribute from main thread."""
        self.messages = messages
        self.is_loading = False
    
    def compose(self) -> ComposeResult:
        if self.is_loading:
            yield Static("Loading...")
        else:
            for msg in self.messages:
                yield ChatMessage(msg)
    
    def start_load(self, conv_id: int):
        self.is_loading = True
        self.load_conversation(conv_id)  # Start worker
```

---

## Real Example: ChatWindowEnhanced Refactoring

### Current Issues in Chat_Window_Enhanced.py

```python
# Line 218: Direct manipulation
attach_button = self.query_one("#attach-image")
indicator = self.query_one("#image-attachment-indicator")
if self.pending_attachment:
    attach_button.variant = "success"
    indicator.add_class("visible")
```

### Refactored Version

```python
class ChatWindowEnhanced(Container):
    # Single source of truth
    pending_attachment = reactive(None)
    is_streaming = reactive(False)
    
    def compose(self) -> ComposeResult:
        """Compose based on reactive state."""
        yield Button(
            "Attach",
            id="attach-image",
            variant=self._attachment_variant
        )
        yield Static(
            self._attachment_text,
            id="image-attachment-indicator",
            classes=self._indicator_classes
        )
    
    @property
    def _attachment_variant(self) -> str:
        return "success" if self.pending_attachment else "default"
    
    @property
    def _attachment_text(self) -> str:
        if self.pending_attachment:
            return f"📎 {self.pending_attachment.name}"
        return ""
    
    @property
    def _indicator_classes(self) -> str:
        return "visible" if self.pending_attachment else "hidden"
    
    def watch_pending_attachment(self, attachment):
        """React to attachment changes."""
        # Update only what's necessary
        self.query_one("#attach-image", Button).variant = self._attachment_variant
        
        indicator = self.query_one("#image-attachment-indicator", Static)
        indicator.update(self._attachment_text)
        indicator.set_class(not attachment, "hidden")
```

---

## Migration Strategy

### Phase 1: Identify Patterns (Week 1)
1. Catalog all `query_one` and `query` calls
2. Group by widget and operation type
3. Identify state dependencies

### Phase 2: Add Reactive Attributes (Week 2)
1. Create reactive attributes for all mutable state
2. Add watchers for state changes
3. Keep both patterns temporarily

### Phase 3: Replace Queries (Week 3-4)
1. Replace direct queries with reactive updates
2. Convert to message-based communication
3. Add workers for async operations

### Phase 4: Clean Up (Week 5)
1. Remove redundant code
2. Consolidate state management
3. Add comprehensive tests

---

## Testing Patterns

### Testing Reactive Widgets

```python
import pytest
from textual.app import App
from textual.testing import AppTest

@pytest.mark.asyncio
async def test_chat_window_streaming_state():
    """Test that streaming state updates UI correctly."""
    
    class TestApp(App):
        def compose(self):
            yield ChatWindowEnhanced()
    
    async with TestApp().run_test() as pilot:
        # Get the chat window
        chat_window = pilot.app.query_one(ChatWindowEnhanced)
        
        # Initial state
        assert chat_window.is_streaming == False
        button = pilot.app.query_one("#send-stop-chat", Button)
        assert button.label == "Send"
        
        # Change state
        chat_window.is_streaming = True
        await pilot.pause()  # Let reactive update happen
        
        # Verify UI updated
        assert button.label == "Stop"
        assert button.variant == "error"
```

---

## Common Pitfalls to Avoid

### 1. Mixing Patterns
❌ Don't mix reactive and direct manipulation in the same widget

### 2. Over-Recomposing
❌ Don't use `recompose=True` for simple property changes
✅ Use watchers for efficient updates

### 3. Circular Dependencies
❌ Avoid reactive attributes that depend on each other
✅ Use computed properties for derived state

### 4. Thread Safety
❌ Never update UI from worker threads directly
✅ Always use `call_from_thread`

### 5. Message Storms
❌ Don't post messages in watchers that trigger more watchers
✅ Use debouncing or flags to prevent loops

---

## Metrics for Success

| Metric | Current | Target |
|--------|---------|--------|
| Direct queries per widget | 10-50 | 0-3 |
| Reactive attributes | 0-2 | 5-10 |
| Message handlers | 0-1 | 3-5 |
| Test coverage | ~20% | >80% |
| UI responsiveness | Variable | Consistent |

---

## Next Steps

1. **Pick a pilot widget** - Start with `ChatWindowEnhanced`
2. **Apply patterns** - Use this guide
3. **Measure improvement** - Track metrics
4. **Document learnings** - Update this guide
5. **Scale to other widgets** - Apply lessons learned

---

*This is a living document. Update with new patterns and learnings as the refactoring progresses.*