# Chat Event Handlers Migration Guide

## Overview

This guide explains how to migrate from the old imperative event handlers to the new Textual-compliant reactive handlers.

## Key Changes

### 1. **No More Direct Widget Manipulation**

❌ **OLD WAY** (Bad):
```python
# Direct widget queries and manipulation
chat_container = app.query_one("#chat-log", VerticalScroll)
await chat_container.mount(ChatMessage(...))
text_area = app.query_one("#chat-input", TextArea)
text_area.clear()
```

✅ **NEW WAY** (Good):
```python
# Use messages and reactive attributes
app.post_message(UserMessageSent(content))
# Widget updates itself through reactive attributes
self.messages = [*self.messages, new_message]  # Triggers UI update
```

### 2. **Message-Based Communication**

❌ **OLD WAY**:
```python
async def handle_chat_send_button_pressed(app, event):
    # 500+ lines of imperative code
    # Direct manipulation everywhere
```

✅ **NEW WAY**:
```python
@on(UserMessageSent)
async def handle_user_message(self, event: UserMessageSent):
    # Post messages for actions
    # Let widgets handle their own updates
```

### 3. **Proper Worker Usage**

❌ **OLD WAY**:
```python
# Blocking operations in handlers
response = chat_api_call(...)  # Blocks UI
await db.save_message(...)  # Blocks UI
```

✅ **NEW WAY**:
```python
@work(exclusive=True)
async def process_message(self, content: str):
    # Runs in worker, doesn't block UI
    response = await asyncio.to_thread(chat_api_call, ...)
    self.call_from_thread(self.update_ui, response)
```

### 4. **Reactive State Management**

❌ **OLD WAY**:
```python
# State scattered everywhere
app.current_conversation_id = "xxx"
app.is_streaming = True
widget.some_state = value
```

✅ **NEW WAY**:
```python
# Reactive attributes with watchers
class ChatWidget(Widget):
    session_id: reactive[str] = reactive("")
    is_streaming: reactive[bool] = reactive(False)
    
    def watch_is_streaming(self, old, new):
        # React to state changes automatically
```

## Migration Steps

### Step 1: Install New Files

1. Add `chat_messages.py` - Message definitions
2. Add `chat_events_refactored.py` - Refactored handlers
3. Add `chat_streaming_refactored.py` - Streaming handlers

### Step 2: Update Widgets to Use Messages

Update your chat widgets to handle messages:

```python
class ChatWidget(Widget):
    # Add reactive state
    messages: reactive[List[ChatMessage]] = reactive([])
    
    # Handle messages
    @on(UserMessageSent)
    def on_user_message(self, event: UserMessageSent):
        # Update reactive state
        self.messages = [*self.messages, ChatMessage(event.content, "user")]
    
    @on(LLMResponseCompleted)
    def on_llm_response(self, event: LLMResponseCompleted):
        # Update reactive state
        self.messages = [*self.messages, ChatMessage(event.full_response, "assistant")]
```

### Step 3: Replace Button Handlers

❌ **OLD**:
```python
button_handlers = {
    "send-stop-chat": chat_events.handle_chat_send_button_pressed,
    ...
}
```

✅ **NEW**:
```python
# In button handler
if button_id == "send-stop-chat":
    content = self.get_input_content()  # Simple getter
    self.post_message(UserMessageSent(content))
```

### Step 4: Update Streaming

❌ **OLD**:
```python
# Direct manipulation in streaming
widget = app.query_one("#ai-message")
widget.content += chunk
widget.refresh()
```

✅ **NEW**:
```python
# Reactive streaming
@on(LLMResponseChunk)
def on_chunk(self, event: LLMResponseChunk):
    self.streaming_content = self.streaming_content + event.chunk
    # UI updates automatically!
```

## Benefits of Migration

1. **Performance**: No UI blocking, smooth 60fps
2. **Maintainability**: Clear separation of concerns
3. **Testability**: Easy to test with message mocking
4. **Reliability**: No race conditions or state conflicts
5. **Textual Compliance**: Works properly with Textual's architecture

## Gradual Migration Strategy

### Phase 1: Add Message Definitions
- Keep old handlers working
- Add new message classes
- Start posting messages alongside old code

### Phase 2: Update Widgets
- Add reactive attributes to widgets
- Add message handlers
- Keep old direct manipulation as fallback

### Phase 3: Replace Handlers
- Switch to new handlers one by one
- Test each replacement
- Remove old handler when confirmed working

### Phase 4: Cleanup
- Remove all `query_one` calls
- Remove all `mount` calls
- Remove all direct state manipulation
- Celebrate! 🎉

## Common Patterns

### Getting Input Value
```python
# Still need one query for input
text_area = self.query_one("#chat-input", TextArea)
content = text_area.text
# But then use messages
self.post_message(UserMessageSent(content))
text_area.clear()
```

### Showing Errors
```python
# Don't mount error widgets
# Post error messages
self.post_message(ChatError("Something went wrong"))
```

### Updating Display
```python
# Don't refresh/update widgets
# Update reactive attributes
self.message_count = len(self.messages)  # Triggers UI update
```

## Testing

Test the refactored handlers:

```python
# Easy to test with messages
async def test_send_message():
    app = ChatApp()
    async with app.run_test() as pilot:
        # Post a message
        app.post_message(UserMessageSent("Hello"))
        
        # Check reactive state updated
        assert len(app.messages) == 1
        assert app.messages[0].content == "Hello"
```

## Rollback Plan

If issues arise:
1. Keep old handlers in `chat_events.py`
2. New handlers in `chat_events_refactored.py`
3. Switch between them with a flag
4. Gradual migration per handler

## Conclusion

This migration makes the chat system:
- Properly reactive
- Non-blocking
- Textual-compliant
- More maintainable
- More performant

The effort is worth it for a properly architected chat system that works with Textual, not against it.