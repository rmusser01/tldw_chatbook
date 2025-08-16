# State Decomposition Analysis
## App Class Reactive Attributes Refactoring

**Date:** August 15, 2025  
**Current State:** 65 reactive attributes in main app class  
**Target State:** < 20 attributes with proper state containers

---

## Current State Analysis

The `TldwCli` app class has 65 reactive attributes managing state for 17 different features. This violates:
- **Single Responsibility Principle** - App class manages all state
- **Separation of Concerns** - Mixing UI, business logic, and data
- **Encapsulation** - All state is globally accessible
- **Maintainability** - 6000+ line file with tangled dependencies

---

## Reactive Attributes by Category

### 1. Navigation State (3 attributes)
```python
current_tab: reactive[str] = reactive("")
splash_screen_active: reactive[bool] = reactive(False)
media_active_view: reactive[Optional[str]] = reactive(None)
```
**Owner:** Should be in NavigationState container

### 2. Chat State (15 attributes)
```python
chat_api_provider_value: reactive[Optional[str]]
current_chat_is_ephemeral: reactive[bool]
current_chat_conversation_id: reactive[Optional[str]]
current_chat_active_character_data: reactive[Optional[Dict]]
active_chat_tab_id: reactive[Optional[str]]
chat_sessions: reactive[Dict[str, Dict[str, Any]]]
chat_sidebar_collapsed: reactive[bool]
chat_right_sidebar_collapsed: reactive[bool]
chat_right_sidebar_width: reactive[int]
chat_sidebar_selected_prompt_id: reactive[Optional[int]]
chat_sidebar_selected_prompt_system: reactive[Optional[str]]
chat_sidebar_selected_prompt_user: reactive[Optional[str]]
chat_sidebar_loaded_prompt_id: reactive[Optional[Union[int, str]]]
chat_sidebar_loaded_prompt_title_text: reactive[str]
chat_sidebar_loaded_prompt_system_text: reactive[str]
```
**Owner:** Should be in ChatState container

### 3. Notes State (12 attributes)
```python
current_selected_note_id: reactive[Optional[str]]
current_selected_note_version: reactive[Optional[int]]
current_selected_note_title: reactive[Optional[str]]
current_selected_note_content: reactive[Optional[str]]
notes_unsaved_changes: reactive[bool]
notes_sort_by: reactive[str]
notes_sort_ascending: reactive[bool]
notes_preview_mode: reactive[bool]
notes_auto_save_enabled: reactive[bool]
notes_auto_save_timer: reactive[Optional[Timer]]
notes_last_save_time: reactive[Optional[float]]
notes_auto_save_status: reactive[str]
```
**Owner:** Should be in NotesState container

### 4. Conv/Char State (6 attributes)
```python
ccp_active_view: reactive[str]
ccp_api_provider_value: reactive[Optional[str]]
current_editing_character_id: reactive[Optional[str]]
current_editing_character_data: reactive[Optional[Dict]]
current_conv_char_tab_conversation_id: reactive[Optional[str]]
current_ccp_character_details: reactive[Optional[Dict]]
```
**Owner:** Should be in ConvCharState container

### 5. Sidebar States (5 attributes)
```python
notes_sidebar_left_collapsed: reactive[bool]
notes_sidebar_right_collapsed: reactive[bool]
conv_char_sidebar_left_collapsed: reactive[bool]
conv_char_sidebar_right_collapsed: reactive[bool]
evals_sidebar_collapsed: reactive[bool]
```
**Owner:** Should be in UILayoutState container

### 6. Other States (24 attributes)
- RAG/Search states
- Media states
- Prompt management states
- UI preferences
- etc.

---

## Proposed State Container Architecture

### 1. NavigationState
```python
@dataclass
class NavigationState:
    """Manages app-wide navigation state."""
    current_tab: str = ""
    current_screen: Optional[str] = None
    navigation_history: List[str] = field(default_factory=list)
    splash_active: bool = False
    
    def navigate_to(self, destination: str) -> None:
        """Navigate to a tab or screen."""
        self.navigation_history.append(self.current_tab)
        self.current_tab = destination
    
    def go_back(self) -> Optional[str]:
        """Navigate to previous location."""
        if self.navigation_history:
            return self.navigation_history.pop()
        return None
```

### 2. ChatState
```python
@dataclass
class ChatSession:
    """Single chat session state."""
    id: str
    conversation_id: Optional[int] = None
    is_ephemeral: bool = True
    character_data: Optional[Dict] = None
    messages: List[Dict] = field(default_factory=list)
    
@dataclass
class ChatState:
    """Manages all chat-related state."""
    active_session_id: Optional[str] = None
    sessions: Dict[str, ChatSession] = field(default_factory=dict)
    provider: str = "openai"
    model: str = "gpt-4"
    
    # Sidebar state
    sidebar_collapsed: bool = False
    right_sidebar_collapsed: bool = False
    selected_prompt_id: Optional[int] = None
    
    def create_session(self, tab_id: str) -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(id=tab_id)
        self.sessions[tab_id] = session
        return session
    
    def get_active_session(self) -> Optional[ChatSession]:
        """Get the currently active session."""
        if self.active_session_id:
            return self.sessions.get(self.active_session_id)
        return None
```

### 3. NotesState
```python
@dataclass
class Note:
    """Single note data."""
    id: str
    title: str
    content: str
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

@dataclass
class NotesState:
    """Manages notes-related state."""
    selected_note_id: Optional[str] = None
    notes: Dict[str, Note] = field(default_factory=dict)
    
    # Editor state
    unsaved_changes: bool = False
    preview_mode: bool = False
    
    # Auto-save state
    auto_save_enabled: bool = True
    last_save_time: Optional[float] = None
    auto_save_status: str = ""
    
    # View state
    sort_by: str = "date_created"
    sort_ascending: bool = False
    
    def get_selected_note(self) -> Optional[Note]:
        """Get currently selected note."""
        if self.selected_note_id:
            return self.notes.get(self.selected_note_id)
        return None
    
    def mark_unsaved(self) -> None:
        """Mark current note as having unsaved changes."""
        self.unsaved_changes = True
        self.auto_save_status = "pending"
```

### 4. UILayoutState
```python
@dataclass
class UILayoutState:
    """Manages UI layout preferences."""
    sidebars: Dict[str, bool] = field(default_factory=lambda: {
        "chat_left": False,
        "chat_right": False,
        "notes_left": False,
        "notes_right": False,
        "conv_char_left": False,
        "conv_char_right": False,
        "evals": False
    })
    
    sidebar_widths: Dict[str, int] = field(default_factory=dict)
    
    def toggle_sidebar(self, sidebar_id: str) -> bool:
        """Toggle a sidebar's visibility."""
        current = self.sidebars.get(sidebar_id, False)
        self.sidebars[sidebar_id] = not current
        return self.sidebars[sidebar_id]
```

---

## Implementation Strategy

### Phase 1: Create State Containers (Week 1)
```python
# app_state.py
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class AppState:
    """Root state container for the application."""
    navigation: NavigationState = field(default_factory=NavigationState)
    chat: ChatState = field(default_factory=ChatState)
    notes: NotesState = field(default_factory=NotesState)
    conv_char: ConvCharState = field(default_factory=ConvCharState)
    ui_layout: UILayoutState = field(default_factory=UILayoutState)
```

### Phase 2: Make State Containers Reactive (Week 2)
```python
from textual.reactive import reactive

class TldwCli(App):
    # Single reactive root state
    state = reactive(AppState())
    
    def watch_state(self, old_state: AppState, new_state: AppState):
        """React to any state change."""
        # Dispatch updates to relevant components
        if old_state.navigation.current_tab != new_state.navigation.current_tab:
            self.handle_tab_change(new_state.navigation.current_tab)
```

### Phase 3: Migrate Existing Code (Week 3-4)
```python
# BEFORE: Direct attribute access
self.current_chat_conversation_id = conversation_id
self.chat_sidebar_collapsed = True

# AFTER: State container access
self.state.chat.active_session.conversation_id = conversation_id
self.state.ui_layout.toggle_sidebar("chat_left")
```

### Phase 4: Add State Persistence (Week 5)
```python
import json
from pathlib import Path

class StateManager:
    """Manages state persistence and recovery."""
    
    def save_state(self, state: AppState, path: Path):
        """Persist state to disk."""
        state_dict = asdict(state)
        path.write_text(json.dumps(state_dict))
    
    def load_state(self, path: Path) -> AppState:
        """Load state from disk."""
        if path.exists():
            state_dict = json.loads(path.read_text())
            return AppState(**state_dict)
        return AppState()
```

---

## Migration Path

### Step 1: Parallel Implementation
1. Create new state containers alongside existing attributes
2. Mirror updates to both systems
3. Verify functionality remains identical

### Step 2: Gradual Migration
1. Migrate one feature at a time (start with Notes)
2. Update all references in that feature's code
3. Test thoroughly before moving to next feature

### Step 3: Cleanup
1. Remove old reactive attributes
2. Delete compatibility shims
3. Update all tests

---

## Benefits After Refactoring

| Aspect | Current | After Refactoring |
|--------|---------|-------------------|
| App class lines | 6000+ | < 1000 |
| Reactive attributes | 65 | < 10 |
| State access | Global | Scoped |
| Testing | Complex mocking | Simple state injection |
| Debugging | Trace through app | Isolated state containers |
| Memory usage | All state in memory | Lazy loading possible |
| Persistence | Custom per feature | Unified state saving |

---

## Example: Refactored App Class

```python
class TldwCli(App):
    """Main application with minimal state."""
    
    # Only essential app-level state
    state = reactive(AppState())
    theme = reactive("default")
    is_loading = reactive(False)
    
    def compose(self) -> ComposeResult:
        """Compose UI based on navigation mode."""
        if self.state.navigation.use_screens:
            # Screen-based navigation
            yield Container(id="screen-container")
        else:
            # Tab-based navigation
            yield TabBar()
            yield Container(id="tab-container")
    
    def on_mount(self):
        """Initialize app with loaded state."""
        # Load persisted state
        self.state = StateManager().load_state()
        
        # Set up auto-save
        self.set_interval(30, self.auto_save_state)
    
    def auto_save_state(self):
        """Periodically persist state."""
        StateManager().save_state(self.state)
```

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| App class attributes | 65 | < 20 | Count reactive attrs |
| State container classes | 0 | 5-7 | Count dataclasses |
| Lines in app.py | 6000+ | < 1000 | wc -l app.py |
| Test complexity | High | Low | Cyclomatic complexity |
| State bugs | Frequent | Rare | Bug tracker |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|---------|------------|
| Breaking changes | High | Parallel implementation |
| Performance regression | Medium | Profile before/after |
| Lost functionality | High | Comprehensive tests |
| Team resistance | Medium | Gradual migration |

---

## Next Steps

1. **Review and approve** this design with team
2. **Create state containers** in new module
3. **Start with NotesState** as pilot
4. **Measure improvements** 
5. **Apply learnings** to other states

---

*This refactoring will transform the monolithic app class into a clean, maintainable architecture with proper separation of concerns.*