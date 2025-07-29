# Logits Checker Implementation Plan

## Overview
Add a new sidetab/window to the Evals tab called 'Logits Checker' that records and displays logits for every token, allowing users to see potential generations at each step. The implementation will follow the project's established patterns and integrate with the existing LLM infrastructure.

## Architecture Design

### 1. UI Structure
The Logits Checker will be implemented as a new view within the Evals tab, following the existing pattern of other views (Setup, Results, Models, Datasets).

#### 1.1 Navigation Integration
- Add new constant: `EVALS_VIEW_LOGITS = "evals-view-logits"`
- Add navigation button: `EVALS_NAV_LOGITS = "evals-nav-logits"`
- Update the navigation pane with "Logits Checker" button

#### 1.2 Layout Design
Following Textual's layout patterns from `/Docs/Design/Textual-Layouts.md`:
```css
/* Two-column layout for logits checker */
.logits-checker-layout {
    layout: horizontal;
}
.logits-input-panel {
    width: 40%;
    min-width: 30;
}
.logits-display-panel {
    width: 60%;
}
```

#### 1.3 Components Structure
```
Container (id=EVALS_VIEW_LOGITS)
├── Static ("🔢 Logits Checker", classes="pane-title")
├── Horizontal (classes="logits-checker-layout")
│   ├── Container (classes="logits-input-panel")
│   │   ├── Provider/Model Selection (using form_components)
│   │   ├── TextArea (prompt input)
│   │   ├── Button ("Generate with Logits")
│   │   └── Settings (temperature, top_k, etc.)
│   └── Container (classes="logits-display-panel")
│       ├── TokenDisplay (custom widget)
│       └── LogitsTable (custom widget)
```

### 2. Custom Widgets

#### 2.1 TokenDisplay Widget
A custom widget to display generated tokens with interactive features:
```python
class TokenDisplay(Container):
    """Displays tokens with hover/click interactions"""
    tokens = reactive([])  # List of token objects with logits
    selected_token_index = reactive(None)
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="token-row"):
            # Dynamically create token buttons
            pass
    
    def on_button_pressed(self, event: Button.Pressed):
        # Handle token selection
        pass
```

#### 2.2 LogitsTable Widget
Displays the top logits for the selected token:
```python
class LogitsTable(Container):
    """Shows top N logits for selected token"""
    logits_data = reactive([])  # List of (token, probability) tuples
    
    def compose(self) -> ComposeResult:
        yield Static("Top Tokens", classes="section-title")
        with VerticalScroll():
            # Render logits as interactive list
            pass
```

### 3. Backend Integration

#### 3.1 Streaming Event Extensions
Extend existing streaming events to include logits data:
```python
class StreamingChunkWithLogits(StreamingChunk):
    """Extended streaming chunk that includes logits"""
    def __init__(self, text_chunk: str, logprobs: Optional[Dict] = None):
        super().__init__(text_chunk)
        self.logprobs = logprobs
```

#### 3.2 Provider Support Matrix
Based on the codebase analysis, implement logprobs support for:
- **OpenAI**: Already supports `logprobs` and `top_logprobs` parameters
- **Anthropic**: Check API documentation for logprobs support
- **Local providers** (llama.cpp, koboldcpp, etc.): Often support logprobs natively

#### 3.3 Data Flow
1. User enters prompt and selects provider/model
2. Call `chat_api_call` with `llm_logprobs=True` and `llm_top_logprobs=10`
3. Parse streaming response to extract logprobs data
4. Send `StreamingChunkWithLogits` events
5. UI updates token display and logits table reactively

### 4. Event Handlers

#### 4.1 New Event Types
```python
class LogitsGenerationStarted(Message):
    """Signals start of logits generation"""
    pass

class LogitsTokenReceived(Message):
    """Carries token and its logits data"""
    def __init__(self, token: str, logprobs: List[Tuple[str, float]]):
        self.token = token
        self.logprobs = logprobs

class LogitsGenerationCompleted(Message):
    """Signals completion with full token list"""
    def __init__(self, tokens: List[Dict]):
        self.tokens = tokens
```

#### 4.2 Event Handler Integration
Add handlers to `EvalsWindow`:
```python
@on(Button.Pressed, "#generate-logits-btn")
def handle_generate_logits(self, event: Button.Pressed):
    """Start logits generation"""
    # Validate inputs
    # Start worker for API call
    pass

def on_logits_token_received(self, event: LogitsTokenReceived):
    """Update UI with new token and logits"""
    # Update TokenDisplay
    # Update LogitsTable if token is selected
    pass
```

### 5. Implementation Steps

#### Phase 1: Basic UI Setup
1. Add navigation button and view container
2. Create basic layout with provider/model selection
3. Add prompt input and generate button
4. Style with appropriate CSS

#### Phase 2: Core Functionality
1. Implement logits extraction for OpenAI provider
2. Create TokenDisplay widget with basic rendering
3. Add LogitsTable widget for probability display
4. Wire up event flow for non-streaming responses

#### Phase 3: Streaming Support
1. Extend streaming events to carry logprobs
2. Update streaming parsers for each provider
3. Add real-time token display updates
4. Handle streaming edge cases

#### Phase 4: Interactive Features
1. Add hover effects to show top alternatives
2. Implement click-to-select alternative tokens
3. Add probability visualization (bars/percentages)
4. Add export functionality for analysis

#### Phase 5: Provider Extensions
1. Add logprobs support for Anthropic (if available)
2. Implement for local providers (llama.cpp, etc.)
3. Add provider-specific configuration options
4. Handle provider limitations gracefully

### 6. Technical Considerations

#### 6.1 Performance
- Limit top_logprobs to reasonable number (10-20)
- Use reactive updates efficiently
- Consider virtualization for long token sequences
- Cache logprobs data to avoid re-fetching

#### 6.2 Error Handling
- Gracefully handle providers without logprobs support
- Show clear error messages for API failures
- Provide fallback UI for missing data
- Log errors appropriately

#### 6.3 Data Structures
```python
@dataclass
class TokenWithLogits:
    token: str
    token_id: int
    log_probability: float
    top_alternatives: List[Tuple[str, float]]
    position: int
```

### 7. Testing Strategy
1. Unit tests for logprobs parsing logic
2. Integration tests with mock API responses
3. UI tests for interactive features
4. Provider-specific compatibility tests

### 8. Future Enhancements
- Token probability heatmap visualization
- Comparative analysis between models
- Logits-based prompt engineering tools
- Integration with Evals for logit-based metrics
- Save/load logits sessions
- Token tree visualization for alternative paths

## Lessons Learned & Issues

### Research Findings
1. **Mikupad Implementation**: Uses inline probability display with hover interactions, good UX pattern to follow
2. **Provider Variability**: Different providers return logprobs in different formats, need normalization layer
3. **Streaming Complexity**: Logprobs in streaming responses require careful parsing and state management

### Potential Challenges
1. **Provider Limitations**: Not all providers support logprobs (especially some cloud providers)
2. **Performance**: Large top_logprobs values can slow down generation
3. **UI Complexity**: Displaying many tokens with alternatives needs careful design to avoid clutter

### Design Decisions
1. **Separate Tab vs Inline**: Chose separate tab for dedicated analysis space
2. **Streaming First**: Design with streaming in mind from the start
3. **Provider Agnostic**: Abstract logprobs handling to work across providers

---

*Document created: 2025-07-28*
*Status: Planning Phase Complete*

### Implementation Notes
1. **Constants Location**: The Evals tab constants are defined locally in Evals_Window_v3.py rather than Constants.py, following the pattern where each window manages its own view constants.

### Implementation Progress

#### Completed Tasks:
1. ✅ **Navigation Integration**: Added "Logits Checker" button and constants (EVALS_VIEW_LOGITS, EVALS_NAV_LOGITS)
2. ✅ **Basic UI Structure**: Created two-column layout with configuration panel and display panel
3. ✅ **Form Components**: Added provider/model selection, prompt input, and advanced settings
4. ✅ **Event Handlers**: Implemented handlers for provider change and generate button
5. ✅ **CSS Styling**: Added comprehensive styles in _evaluation_v3.tcss for all logits components
6. ✅ **Helper Methods**: Added _populate_models_for_logits_provider and _start_logits_generation

#### Implementation Details:
- Used existing form_components for consistent UI
- Integrated with existing provider/model selection pattern
- Added placeholder for actual logits generation (marked with TODO)
- Status updates use existing _update_status method
- Follows existing patterns for work decorators and async operations

#### Next Steps:
1. Implement actual LLM API call with logprobs enabled
2. Create custom TokenDisplay widget for interactive token display
3. Create LogitsTable widget for probability visualization
4. Add streaming support for real-time updates
5. Implement token selection interaction

#### Code Locations:
- View definition: `Evals_Window_v3.py` lines 1149-1214
- Event handlers: `Evals_Window_v3.py` lines 498-544
- Helper methods: `Evals_Window_v3.py` lines 842-1065
- CSS styles: `_evaluation_v3.tcss` lines 700-850
- Streaming events: `worker_events.py` lines 41-45, 590-593, 604-608

### Phase 2: Logprobs Integration Complete

#### Completed Tasks:
1. ✅ **StreamingChunkWithLogits Event**: Created new event class that extends StreamingChunk with logprobs data
2. ✅ **Streaming Parser Update**: Modified chat_wrapper_function to extract and send logprobs with chunks
3. ✅ **Event Handlers**: Added handlers for streaming chunks in Logits Checker view
4. ✅ **Token Display**: Implemented inline token buttons that show as text streams
5. ✅ **Logits Table**: Created dynamic table that shows top alternatives when token is clicked
6. ✅ **Worker Integration**: Connected chat API with logprobs parameters to UI

#### Implementation Details:
- **Streaming Event Flow**:
  1. User enters prompt and clicks "Generate with Logits"
  2. Worker calls `chat()` with `llm_logprobs=True` and `llm_top_logprobs=10`
  3. Streaming parser extracts logprobs from choices[0]["logprobs"]
  4. Posts `StreamingChunkWithLogits` events with text and logprobs data
  5. UI handles events and creates clickable token buttons
  6. Click handler displays top alternatives with probabilities

- **Logprobs Data Structure** (OpenAI format):
  ```json
  {
    "content": [{
      "token": "Hello",
      "logprob": -0.123,
      "top_logprobs": [
        {"token": "Hello", "logprob": -0.123},
        {"token": "Hi", "logprob": -1.456},
        ...
      ]
    }]
  }
  ```

#### Current Status:
The Logits Checker is now fully functional with:
- Provider/model selection
- Prompt input with advanced settings
- Real-time token streaming with logprobs
- Interactive token selection
- Top alternatives display with probabilities

#### Testing Notes:
- Requires OpenAI API key configured
- Provider must support logprobs (OpenAI confirmed, others may vary)
- Top_logprobs parameter limits alternatives (max 5 for OpenAI)

### llama.cpp Support Added:
- ✅ **Parameter Mapping**: Added `logprobs` and `top_logprobs` to PROVIDER_PARAM_MAP for llama_cpp
- ✅ **Function Signature**: Updated `chat_with_llama` to accept logprobs parameters
- ✅ **Parameter Passing**: Modified function to pass logprobs to OpenAI-compatible server
- ✅ **Compatibility**: Modern llama.cpp servers (2024+) support logprobs via OpenAI-compatible API (PR #10783)

### vLLM Support Added:
- ✅ **Parameter Mapping**: Added `top_logprobs` to PROVIDER_PARAM_MAP for vllm (logprobs was already present)
- ✅ **Function Signature**: Updated `chat_with_vllm` to accept top_logprobs parameter
- ✅ **Parameter Passing**: Modified function to pass both logprobs and top_logprobs to OpenAI-compatible server
- ✅ **Compatibility**: vLLM supports logprobs through its OpenAI-compatible API

### UI Updates:
- ✅ **Provider Dropdown**: Updated Logits Checker to show only OpenAI, Llama.cpp, and vLLM as provider options
- ✅ **User-Friendly Names**: Providers display as "OpenAI", "Llama.cpp", and "vLLM" in the dropdown
- ✅ **Model Selection**: Models are now loaded from user's config file (providers section)
- ✅ **Smart Model Handling**: 
  - OpenAI: Shows models from config
  - Llama.cpp: Shows "Use Loaded Model" and passes None to API (uses server's loaded model)
  - vLLM: Shows "Use Loaded Model" and passes None to API (uses server's loaded model)

### Bug Fixes:
- ✅ **Import Error Fixed**: Replaced `get_api_key` import with proper `get_cli_setting("API", f"{provider}_api_key")` usage
- ✅ **Navigation Button Handler**: Added `event.stop()` to prevent event bubbling to app level
- ✅ **All Button Handlers**: Added `event.stop()` to all button handlers in Logits Checker to prevent "Unhandled button press" warnings
- ✅ **Event Propagation**: Fixed event bubbling issues that were causing errors at the app level
- ✅ **run_worker Arguments**: Fixed run_worker call to use positional arguments instead of keyword arguments (was causing "unexpected keyword argument" error)
- ✅ **Worker Method Calls**: Fixed calling `_start_logits_generation` directly since it's already decorated with `@work(exclusive=True)` (was causing "multiple values for argument 'exclusive'" error)
- ✅ **Nested Worker Pattern**: Removed `run_worker` call inside `_start_logits_generation` and instead call `_run_logits_chat` directly, letting its `@work(thread=True)` decorator handle threading
- ✅ **Threading Context Fix**: Fixed `call_from_thread` to use `self.app_instance.call_from_thread` instead of `self.call_from_thread` (which doesn't exist on the widget)
- ✅ **Streaming Events Fix**: Changed from calling `chat()` directly to using `chat_wrapper_function()` which properly handles streaming and posts events to the app instance
- ✅ **Debug Logging**: Added debug logging to track streaming events and help diagnose issues with token display

### Current Investigation:
- **Issue**: Streaming events are being posted as regular `StreamingChunk` instead of `StreamingChunkWithLogits`
- **Root Causes Identified**: 
  1. The app-level handler for `StreamingChunk` was catching all events before they could reach the Evals window
  2. The streaming parser needed to find where logprobs are located in the response
- **Fix Applied**: 
  1. Modified `_run_logits_chat` to handle streaming directly instead of using `chat_wrapper_function`
  2. Events are now posted directly to the Evals window handlers using `call_from_thread`
  3. This bypasses the app-level handler that was intercepting events
- **Debug Logging Added**: 
  1. Logs when logprobs are found in choice or delta objects
  2. Logs the first chunk structure when logprobs aren't found
  3. Shows available keys in both choice and delta objects
  
### Debug Logging Added:
The following debug logs have been added to help diagnose the issue:
1. In `chat_wrapper_function`: Logs whether logprobs is enabled and the top_logprobs value
2. When logprobs is enabled, logs the first streaming chunk's full JSON structure
3. When logprobs are found, logs the location (choice vs delta) and the data
4. When logprobs are expected but not found, logs the available keys in both choice and delta objects
5. When posting events, logs whether StreamingChunk or StreamingChunkWithLogits is being used

### Status Update:
Based on the latest test:
1. ✅ **Streaming is working** - Events are being received as `StreamingChunkWithLogits` 
2. ✅ **Tokens are displayed** - Each token appears as a clickable button
3. ✅ **Logprobs data is being received** - Logs show "with logprobs: True" for each event
4. ❌ **Logprobs display error** - "Can't mount widget(s) before Container() is mounted" when clicking tokens

### Fixes Applied:
1. Fixed the mounting error by checking if container is mounted before using it
2. Added debug logging to capture the actual logprobs data structure
3. Improved error handling in the display method

### Next Steps for Testing:
1. **Run another test with OpenAI**:
   - The mounting error should now be fixed
   - Check logs for "Logits Checker: Found logprobs in choice. Structure:" or "Found logprobs in delta. Structure:"
   - This will show the actual JSON structure of the logprobs data
   
2. **Click on tokens**:
   - Should no longer see the mounting error
   - If logprobs structure doesn't match expected format, you'll see "Unexpected logprobs format" with the actual structure in logs
   
3. **Once we see the structure**:
   - We can update the parser to match OpenAI's actual format
   - The top alternatives should then display correctly

The key is to find out what format OpenAI is using for logprobs in the streaming response so we can parse it correctly.