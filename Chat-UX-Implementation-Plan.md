# Chat Screen UX Implementation Plan

## Summary of Updates to Chat-UX-8.md

The document has been updated to follow Textual best practices based on official documentation review:

### CSS Corrections Made:
1. **Removed unsupported CSS properties**:
   - `font-size`, `letter-spacing`, `font-weight` → Use `text-style: bold/italic`
   - `transform`, `transition`, `animation`, `@keyframes` → Not supported
   - `::before`, `::after` pseudo-elements → Not supported
   - `position: absolute` → Use `dock` or `layer` system
   - CSS variables (`--var-name`) → Not supported, use values directly
   - `z-index` → Use `layer` system instead

2. **Fixed CSS syntax for Textual**:
   - Opacity values: Use percentages (50%) or decimals (0.5)
   - Borders: Use Textual's border syntax (solid, thick, round, dashed)
   - Pseudo-classes: Only use supported ones (:hover, :focus, :dark, :blur, :disabled, etc.)
   - Background with alpha: Use `$color XX%` syntax
   - Scrollbar: Use `scrollbar-size` (not `scrollbar-width`), `scrollbar-gutter: stable`

3. **Adapted patterns to Textual**:
   - Animations → Use reactive properties and text styling changes
   - Floating panels → Use `layer: above` or `layer: modal`
   - Focus management → Use `outline` property (supported)

## Implementation Priorities for Current Chat Screen

Based on the current state of the Chat Screen and Textual best practices, here are the priorities:

### Phase 1: Core Visual Improvements (Week 1)
**High Impact, Low Complexity**

1. **✅ Enhanced Collapsible Styling** (COMPLETED)
   - Already implemented improved collapsible sections
   - Added hover and focus states
   - Fixed height issues

2. **✅ Visual Feedback States** (COMPLETED)
   - Button hover/focus states implemented
   - Loading indicators created
   - Error states defined

3. **Semantic Grouping** (PRIORITY 1)
   - **Files to modify**: 
     - `tldw_chatbook/UI/Screens/chat_screen.py`
     - `tldw_chatbook/UI/Chat_Window_Enhanced.py`
   - **Implementation**:
     ```python
     # Group settings into logical sections
     with Container(classes="settings-group primary-group"):
         yield Static("ESSENTIAL", classes="group-header")
         # Model config, conversation controls
     
     with Container(classes="settings-group secondary-group"):
         yield Static("FEATURES", classes="group-header")
         # Character, RAG, etc.
     ```

### Phase 2: State Management (Week 1-2)
**High Impact, Medium Complexity**

1. **Collapsible State Persistence (Using Textual Best Practices)**
   - **Files to modify**: 
     - `tldw_chatbook/UI/Screens/chat_screen.py`
     - `tldw_chatbook/state/ui_state.py` (already exists)
   - **Implementation using reactive properties**:
     ```python
     from textual.reactive import reactive
     import toml
     from pathlib import Path
     
     class ChatScreen(BaseAppScreen):
         # Reactive property with watcher for auto-save
         sidebar_state = reactive({}, layout=False)
         
         def watch_sidebar_state(self, new_state: dict) -> None:
             """Auto-save when state changes."""
             self._save_sidebar_state(new_state)
         
         def _save_sidebar_state(self, state: dict) -> None:
             """Save to TOML config file."""
             config_path = Path.home() / ".config" / "tldw_cli" / "ui_state.toml"
             config_path.parent.mkdir(parents=True, exist_ok=True)
             
             try:
                 with open(config_path, 'w') as f:
                     toml.dump({"sidebar": state}, f)
             except Exception as e:
                 logger.error(f"Failed to save sidebar state: {e}")
     ```

2. **Bulk Collapse/Expand Actions**
   - Add buttons for "Expand All" / "Collapse All"
   - Keep essential sections always visible
   - Use `mutate_reactive()` when modifying dict/list state

### Phase 3: Search & Filter System (Week 2)
**High Impact, Medium Complexity**

1. **Real-time Settings Search (Using Workers)**
   - **Files to create/modify**: 
     - `tldw_chatbook/Widgets/sidebar_search.py`
     - Use existing `enhanced_sidebar.py` as base
   - **Implementation with Workers**:
     ```python
     from textual import work
     from textual.worker import Worker, WorkerState
     
     class SidebarSearch(Input):
         search_results = reactive([], recompose=False)
         
         @on(Input.Changed)
         def handle_search_change(self, event: Input.Changed) -> None:
             """Trigger search with debouncing."""
             # Cancel previous search if still running
             if hasattr(self, '_search_worker') and self._search_worker:
                 self._search_worker.cancel()
             
             # Start new search
             self._search_worker = self.search_settings(event.value)
         
         @work(exclusive=True, thread=True)
         def search_settings(self, query: str) -> None:
             """Search in background thread."""
             if not query:
                 self.call_from_thread(self.clear_results)
                 return
             
             # Check for cancellation using get_current_worker
             from textual.worker import get_current_worker
             worker = get_current_worker()
             if worker.is_cancelled:
                 return
             
             # Perform search
             results = self._fuzzy_search(query)
             
             # Update UI from thread
             self.call_from_thread(self.update_results, results)
         
         def update_results(self, results: list) -> None:
             """Update UI with search results."""
             self.search_results = results
     ```

### Phase 4: Quick Actions Bar (Week 2-3)
**Medium Impact, Low Complexity**

1. **Persistent Quick Actions**
   - **Location**: Top of sidebar
   - **Actions**: New Chat, Save, Export, Reset
   - **Implementation**:
     ```python
     with Horizontal(classes="quick-actions-bar"):
         yield Button("🆕", id="quick-new-chat", tooltip="New Chat")
         yield Button("💾", id="quick-save", tooltip="Save")
     ```

### Phase 5: Keyboard Navigation (Week 3)
**Medium Impact, Medium Complexity**

1. **Enhanced Keyboard Support**
   - **Already partially implemented** in enhanced_sidebar.py
   - Add section jumping (Alt+1, Alt+2, etc.)
   - Quick setting access (Ctrl+T for temperature)

### Phase 6: Form Validation (Week 3-4)
**Medium Impact, High Complexity**

1. **Real-time Input Validation (Using Textual Validators)**
   - **Files to create**: 
     - `tldw_chatbook/Widgets/validated_input.py`
   - **Implementation with Textual's validation system**:
     ```python
     from textual.validation import Number, Length, Regex, ValidationResult, Validator
     from textual.widgets import Input
     from textual.reactive import reactive
     
     class TemperatureValidator(Validator):
         """Custom validator for temperature values."""
         
         def validate(self, value: str) -> ValidationResult:
             """Validate temperature is between 0 and 2."""
             if not value:
                 return self.failure("Temperature is required")
             
             try:
                 temp = float(value)
                 if temp < 0:
                     return self.failure("Temperature must be >= 0")
                 if temp > 2:
                     return self.failure("Temperature must be <= 2")
                 if temp > 1.5:
                     # Warning but still valid
                     return self.success(warning="High temperature may produce inconsistent results")
                 return self.success()
             except ValueError:
                 return self.failure("Temperature must be a number")
     
     class ValidatedTemperatureInput(Input):
         """Temperature input with built-in validation."""
         
         def __init__(self, **kwargs):
             super().__init__(
                 validators=[TemperatureValidator()],
                 **kwargs
             )
         
         @on(Input.Submitted)
         def handle_submit(self, event: Input.Submitted) -> None:
             """Only process if valid."""
             if not self.is_valid:
                 event.stop()
                 self.add_class("input-error")
                 # Access validation errors properly
                 if self.validation_errors:
                     self.notify(str(self.validation_errors[0]), severity="error")
     ```

### Phase 7: Responsive Sidebar (Week 4)
**Low Impact, Medium Complexity**

1. **Width Adjustment**
   - Already at 35% width
   - Add resize handle for manual adjustment
   - Save preference

## Immediate Next Steps (This Week)

### 1. Implement Semantic Grouping
```python
# In chat_screen.py, modify create_chat_sidebar()
def create_chat_sidebar(self) -> ComposeResult:
    with VerticalScroll(id="chat-left-sidebar", classes="sidebar"):
        # Quick actions
        with Horizontal(classes="quick-actions-bar"):
            yield Button("➕", id="expand-all", tooltip="Expand all")
            yield Button("➖", id="collapse-all", tooltip="Collapse all")
        
        # Essential group
        with Container(classes="settings-group primary-group"):
            yield Static("ESSENTIAL", classes="group-header")
            # Existing model config collapsible
            # Existing conversation collapsible
        
        # Features group
        with Container(classes="settings-group secondary-group"):
            yield Static("FEATURES", classes="group-header")
            # Character settings
            # RAG settings
```

### 2. Add State Persistence
```python
# In chat_screen.py
from textual.css.query import QueryError
from textual.widgets import Collapsible

@on(Collapsible.Toggled)
def handle_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
    """Save collapsible state using reactive property."""
    # Update reactive dict - must create new dict for reactive to trigger
    new_state = dict(self.sidebar_state)
    new_state[event.collapsible.id] = event.collapsible.collapsed
    self.sidebar_state = new_state  # Triggers watcher

def on_mount(self) -> None:
    """Restore saved states."""
    super().on_mount()
    
    # Load saved state
    saved_state = self._load_sidebar_state()
    if saved_state:
        self.sidebar_state = saved_state
    
    # Apply to collapsibles after mount
    self.set_timer(0.1, self._restore_collapsible_states)

def _restore_collapsible_states(self) -> None:
    """Restore collapsible states from saved state."""
    for coll_id, collapsed in self.sidebar_state.items():
        try:
            collapsible = self.query_one(f"#{coll_id}", Collapsible)
            collapsible.collapsed = collapsed
        except QueryError:
            # Collapsible might not exist in current mode
            pass
```

### 3. Implement Bulk Actions
```python
@on(Button.Pressed, "#expand-all")
def expand_all(self) -> None:
    """Expand all collapsible sections."""
    for collapsible in self.query(Collapsible):
        collapsible.collapsed = False

@on(Button.Pressed, "#collapse-all")  
def collapse_all(self) -> None:
    """Collapse all non-priority sections."""
    for collapsible in self.query(Collapsible):
        if "priority-high" not in collapsible.classes:
            collapsible.collapsed = True
```

### 4. Add Search Input
```python
# Add to sidebar
yield Input(
    placeholder="🔍 Search settings...",
    id="settings-search",
    classes="sidebar-search"
)

@on(Input.Changed, "#settings-search")
def filter_settings(self, event: Input.Changed) -> None:
    """Filter settings based on search query."""
    query = event.value.lower()
    # Implementation from enhanced_sidebar.py
```

## Testing Checklist

- [ ] Collapsible sections maintain state across screen changes
- [ ] Search filters settings correctly
- [ ] Keyboard navigation works (Tab, Shift+Tab, arrow keys)
- [ ] Visual feedback on all interactive elements
- [ ] Loading states display correctly
- [ ] Form validation provides clear feedback
- [ ] Sidebar width persists across sessions
- [ ] Quick actions work as expected

## Files to Reference

1. **`tldw_chatbook/Widgets/enhanced_sidebar.py`** - Contains keyboard navigation patterns
2. **`tldw_chatbook/Widgets/loading_states.py`** - Loading state widgets (if exists)
3. **`tldw_chatbook/css/layout/_sidebars.tcss`** - Current sidebar CSS
4. **`tldw_chatbook/state/ui_state.py`** - Existing UIState class for state management

## Success Metrics

- Reduced time to find settings (target: <3 seconds)
- Fewer mis-clicks due to better visual hierarchy
- Positive user feedback on organization
- No CSS errors in console
- Smooth performance with no lag

## Important Textual Best Practices Applied

1. **Event Handlers**: Use synchronous methods (not async) for most @on handlers
2. **Workers**: Use @work decorator with thread=True for background tasks
3. **Reactive Properties**: Create new objects (not mutate) to trigger watchers
4. **CSS**: Only use Textual-supported properties (no web CSS)
5. **Validation**: Use Textual's Validator classes for input validation
6. **State Access**: Use query_one() with QueryError handling for safe widget access
7. **Thread Safety**: Use call_from_thread() to update UI from workers
8. **Timers**: Use set_timer() for delayed operations after mount