# Implementation Plan: Multiple Response Selection with Conversation Forking

## Overview
Enable users to request multiple AI responses for a single query, browse through them, and select which response to use for continuing the conversation. This will be implemented using conversation forking to maintain conversation history integrity.

## Key Components to Modify/Create

### 1. UI Components

#### New Widget: `ResponseSelectorWidget` (`Widgets/Chat_Widgets/response_selector.py`)
- Display multiple AI responses in a carousel/tabbed interface
- Show response indicators (1/3, 2/3, etc.)
- Navigation buttons (Previous/Next)
- Selection button ("Use This Response")
- Visual differentiation for selected response

#### Settings Sidebar Updates (`Widgets/settings_sidebar.py`)
- Add checkbox: "Enable Multiple Responses"
- Add input field: "Number of Responses" (default: 1, max: 5)
- Validation to ensure n > 1 only when multiple responses enabled

#### Chat Message Widget Enhancement (`Widgets/Chat_Widgets/chat_message_enhanced.py`)
- Add response navigation controls when multiple responses exist
- Visual indicator for forked messages
- Display response variant number (e.g., "Response 2 of 3")

### 2. Event Handlers

#### New: `chat_multi_response_events.py` (`Event_Handlers/Chat_Events/`)
- `handle_response_navigation()` - Navigate between responses
- `handle_response_selection()` - Select a response and create fork
- `handle_regenerate_all_variants()` - Regenerate all response variants

#### Update: `chat_events.py`
- Modify `handle_chat_send_button_pressed()` to check multiple response settings
- Handle array of responses instead of single response
- Create temporary response storage for selection

#### Update: `chat_streaming_events.py`
- Support streaming multiple responses sequentially
- Track which response variant is being streamed
- Update UI indicators during streaming

### 3. Backend Logic

#### New: `conversation_forking.py` (`Chat/`)
```python
def create_conversation_fork(db, parent_message_id, selected_response):
    """Create a new conversation branch from a message."""
    pass

def get_fork_history(db, conversation_id):
    """Retrieve fork tree for a conversation."""
    pass

def merge_conversation_branches(db, branch1_id, branch2_id):
    """Optional: merge branches."""
    pass

def get_alternative_responses(db, message_id):
    """Get all response variants for a message."""
    pass
```

#### Update: `Chat_Functions.py`
- Modify `chat_api_call()` to handle multiple responses
- Store all responses, not just the first
- Add response metadata (variant number, selected status)

#### Update: `worker_events.py`
- Handle multiple response streaming
- Emit events for each response variant
- Track completion of all variants

### 4. Database Updates

#### New Table: `response_variants`
```sql
CREATE TABLE response_variants (
  id TEXT PRIMARY KEY,
  message_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
  variant_number INTEGER NOT NULL,
  content TEXT NOT NULL,
  is_selected BOOLEAN DEFAULT FALSE,
  created_at TEXT NOT NULL,
  metadata TEXT,
  UNIQUE(message_id, variant_number)
);

CREATE INDEX idx_response_variants_message ON response_variants(message_id);
CREATE INDEX idx_response_variants_selected ON response_variants(message_id, is_selected);
```

#### Update: `ChaChaNotes_DB.py`
- Add methods for storing/retrieving response variants
- Update conversation creation to support forking
- Add indexes for efficient fork queries

### 5. Configuration

#### Update: `config.py`
```toml
[chat_defaults]
enable_multiple_responses = false
default_response_count = 1
max_response_count = 5
auto_select_first_response = true
show_response_navigation = true
preserve_unselected_responses = true
```

## Implementation Steps

### Phase 1: Backend Foundation
1. Create response_variants table and migration
2. Implement conversation forking logic
3. Update Chat_Functions to handle multiple responses
4. Test API calls with n > 1

### Phase 2: Core UI Components
1. Create ResponseSelectorWidget
2. Update settings sidebar with new controls
3. Add response navigation to chat messages
4. Implement basic response switching

### Phase 3: Event Handling
1. Create multi-response event handlers
2. Update existing event handlers for compatibility
3. Implement response selection and forking
4. Handle streaming for multiple responses

### Phase 4: Polish & Testing
1. Add visual indicators for forked conversations
2. Implement keyboard shortcuts (Tab/Shift+Tab for navigation)
3. Add response comparison view
4. Create comprehensive tests
5. Update documentation

## User Experience Flow

1. User enables "Multiple Responses" in settings
2. User sets desired number of responses (e.g., 3)
3. User sends a message
4. System generates 3 responses sequentially (with progress indicator)
5. ResponseSelector appears with all 3 responses
6. User can:
   - Navigate between responses (← →)
   - Compare responses side-by-side
   - Select preferred response ("Use This")
   - Regenerate all variants
7. Selected response becomes part of conversation
8. Non-selected responses are stored as variants
9. User can later view/switch to alternative branches

## Technical Considerations

- **Performance**: Stream responses sequentially to avoid API rate limits
- **Storage**: Store all variants but only display selected in main chat
- **UI State**: Maintain response selection state across tab switches
- **Backwards Compatibility**: Ensure single-response mode still works
- **Memory**: Limit max responses to prevent excessive memory usage
- **Cost**: Warn users about increased API costs with multiple responses

## Success Criteria

- Users can generate 1-5 responses per request
- All responses are displayed clearly
- Selection creates proper conversation fork
- Fork history is maintained in database
- UI clearly indicates current branch/variant
- Feature can be enabled/disabled without breaking existing chats
- Performance remains acceptable with multiple responses

## Detailed Implementation Guide

### ResponseSelectorWidget Implementation

```python
# Widgets/Chat_Widgets/response_selector.py
from textual.widget import Widget
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, Static
from textual.reactive import reactive

class ResponseSelectorWidget(Widget):
    """Widget for selecting between multiple AI responses."""
    
    current_index = reactive(0)
    responses = reactive([])
    
    def compose(self):
        with Vertical():
            # Header with response indicator
            yield Label(f"Response {self.current_index + 1} of {len(self.responses)}", 
                       id="response-indicator")
            
            # Response content area
            yield Static("", id="response-content")
            
            # Navigation controls
            with Horizontal(id="response-nav"):
                yield Button("← Previous", id="prev-response", disabled=True)
                yield Button("Use This Response", id="select-response", variant="primary")
                yield Button("Next →", id="next-response")
    
    def watch_current_index(self, index: int):
        """Update display when index changes."""
        self.update_response_display()
        self.update_navigation_buttons()
    
    def update_response_display(self):
        """Update the displayed response content."""
        if self.responses:
            content_widget = self.query_one("#response-content", Static)
            content_widget.update(self.responses[self.current_index].content)
            
            indicator = self.query_one("#response-indicator", Label)
            indicator.update(f"Response {self.current_index + 1} of {len(self.responses)}")
    
    def update_navigation_buttons(self):
        """Enable/disable navigation buttons based on current position."""
        prev_btn = self.query_one("#prev-response", Button)
        next_btn = self.query_one("#next-response", Button)
        
        prev_btn.disabled = self.current_index == 0
        next_btn.disabled = self.current_index >= len(self.responses) - 1
```

### Database Migration

```sql
-- Migration: Add response variants support
-- Version: 8

-- Create response_variants table
CREATE TABLE IF NOT EXISTS response_variants (
  id TEXT PRIMARY KEY,
  message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  variant_number INTEGER NOT NULL,
  content TEXT NOT NULL,
  is_selected BOOLEAN DEFAULT FALSE,
  created_at TEXT NOT NULL,
  metadata TEXT,
  UNIQUE(message_id, variant_number)
);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_response_variants_message 
  ON response_variants(message_id);
CREATE INDEX IF NOT EXISTS idx_response_variants_selected 
  ON response_variants(message_id, is_selected);

-- Add fork tracking to conversations
ALTER TABLE conversations 
  ADD COLUMN fork_depth INTEGER DEFAULT 0;
ALTER TABLE conversations 
  ADD COLUMN fork_path TEXT;

-- Add variant tracking to messages
ALTER TABLE messages 
  ADD COLUMN has_variants BOOLEAN DEFAULT FALSE;
ALTER TABLE messages 
  ADD COLUMN selected_variant_id TEXT REFERENCES response_variants(id);
```

### Event Handler Updates

```python
# Event_Handlers/Chat_Events/chat_multi_response_events.py
from typing import List, Optional
from loguru import logger

async def handle_multiple_responses(app, responses: List[str], original_message_id: str):
    """Handle multiple AI responses for selection."""
    logger.info(f"Received {len(responses)} responses for message {original_message_id}")
    
    # Store all responses as variants
    variants = []
    for i, response in enumerate(responses):
        variant_id = await store_response_variant(
            app.db,
            message_id=original_message_id,
            variant_number=i + 1,
            content=response,
            is_selected=(i == 0)  # Auto-select first if configured
        )
        variants.append(variant_id)
    
    # Create and mount the response selector widget
    selector = ResponseSelectorWidget()
    selector.responses = responses
    selector.current_index = 0
    
    # Mount in the chat UI
    chat_container = app.query_one("#chat-log")
    await chat_container.mount(selector)
    
    return variants

async def handle_response_selection(app, variant_id: str, message_id: str):
    """Handle user selection of a response variant."""
    logger.info(f"User selected variant {variant_id} for message {message_id}")
    
    # Update database to mark selected variant
    await app.db.execute("""
        UPDATE response_variants 
        SET is_selected = (id = ?)
        WHERE message_id = ?
    """, (variant_id, message_id))
    
    # Create conversation fork if not the original response
    original_variant = await app.db.fetch_one("""
        SELECT id FROM response_variants 
        WHERE message_id = ? AND variant_number = 1
    """, (message_id,))
    
    if variant_id != original_variant['id']:
        # This is a fork point
        await create_conversation_fork(
            app.db,
            parent_message_id=message_id,
            selected_variant_id=variant_id
        )
    
    # Update UI to show selected response
    await update_chat_display_with_selection(app, variant_id)
```

### API Call Modifications

```python
# Updates to Chat_Functions.py
def chat_api_call_multiple(
    api_endpoint: str,
    messages_payload: List[Dict[str, Any]],
    n_responses: int = 1,
    **kwargs
) -> List[str]:
    """
    Modified chat_api_call to handle multiple responses.
    
    Args:
        n_responses: Number of responses to generate
        
    Returns:
        List of response strings
    """
    responses = []
    
    # Check if API supports n parameter
    if api_endpoint in ['openai', 'anthropic', 'groq']:
        # These APIs support n parameter directly
        kwargs['n'] = n_responses
        result = chat_api_call(api_endpoint, messages_payload, **kwargs)
        
        if 'choices' in result:
            responses = [choice['message']['content'] for choice in result['choices']]
        else:
            responses = [result]
    else:
        # For APIs that don't support n, make multiple calls
        for i in range(n_responses):
            # Add slight variation to avoid identical responses
            if i > 0:
                kwargs['seed'] = kwargs.get('seed', 42) + i
                
            result = chat_api_call(api_endpoint, messages_payload, **kwargs)
            responses.append(extract_response_content(result))
    
    return responses
```

### Streaming Support for Multiple Responses

```python
# Updates to chat_streaming_events.py
class MultiResponseStreamHandler:
    """Handler for streaming multiple responses."""
    
    def __init__(self, app, n_responses: int):
        self.app = app
        self.n_responses = n_responses
        self.current_response_index = 0
        self.responses = []
        self.current_content = ""
    
    async def handle_chunk(self, event: StreamingChunk):
        """Handle incoming chunk for current response."""
        self.current_content += event.text_chunk
        
        # Update UI with current response being streamed
        await self.update_streaming_display(
            response_index=self.current_response_index,
            content=self.current_content,
            is_complete=False
        )
    
    async def handle_response_complete(self):
        """Handle completion of current response."""
        self.responses.append(self.current_content)
        self.current_content = ""
        self.current_response_index += 1
        
        if self.current_response_index < self.n_responses:
            # Start next response
            await self.start_next_response()
        else:
            # All responses complete
            await self.finalize_responses()
    
    async def finalize_responses(self):
        """Finalize and display all responses for selection."""
        await handle_multiple_responses(
            self.app,
            self.responses,
            self.original_message_id
        )
```

## Next Steps

1. **Implement Phase 1**: Create database schema and basic forking logic
2. **Build ResponseSelectorWidget**: Complete the UI component
3. **Integrate with existing chat flow**: Update event handlers
4. **Add configuration options**: Implement settings UI
5. **Test with different providers**: Ensure compatibility
6. **Add visual polish**: Icons, animations, keyboard shortcuts
7. **Document the feature**: User guide and API documentation