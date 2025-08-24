# App.py Refactoring Plan
## Following Textual Best Practices

**Current State:** 5,857 lines, 176 methods, 65 reactive attributes  
**Target State:** < 500 lines, < 20 methods, < 10 reactive attributes

---

## Critical Issues in Current app.py

### 1. Violations of Textual Best Practices
- ❌ **Monolithic App Class** - Everything in one file
- ❌ **Direct Widget Manipulation** - Using query_one extensively  
- ❌ **Mixed Responsibilities** - UI, business logic, data access all mixed
- ❌ **Improper Event Handling** - 50+ event handlers in app class
- ❌ **State Management** - 65 reactive attributes in app class
- ❌ **No Message-Based Communication** - Direct method calls between components

### 2. General Code Quality Issues
- ❌ **God Object Anti-Pattern** - App class does everything
- ❌ **No Separation of Concerns** - UI mixed with business logic
- ❌ **Poor Error Handling** - Try/except blocks everywhere
- ❌ **Code Duplication** - Similar patterns repeated
- ❌ **Long Methods** - Some methods > 200 lines
- ❌ **Magic Numbers** - Hardcoded values throughout

---

## Refactoring Strategy

### Phase 1: State Extraction (Week 1)

#### 1.1 Create State Management Module
```python
# tldw_chatbook/state/__init__.py
from .app_state import AppState
from .chat_state import ChatState
from .notes_state import NotesState
from .navigation_state import NavigationState

# tldw_chatbook/state/app_state.py
from dataclasses import dataclass, field
from textual.reactive import reactive

@dataclass
class AppState:
    """Root application state container."""
    navigation: NavigationState = field(default_factory=NavigationState)
    chat: ChatState = field(default_factory=ChatState)
    notes: NotesState = field(default_factory=NotesState)
    ui: UIState = field(default_factory=UIState)
    
    # Only truly app-level state
    theme: str = "default"
    is_loading: bool = False
    encryption_enabled: bool = False
```

#### 1.2 Move Reactive Attributes
```python
# BEFORE: In app.py
class TldwCli(App):
    current_tab: reactive[str] = reactive("")
    chat_api_provider_value: reactive[Optional[str]] = reactive(None)
    # ... 63 more attributes

# AFTER: In state containers
class NavigationState:
    current_screen: str = "chat"
    history: List[str] = field(default_factory=list)

class ChatState:
    provider: str = "openai"
    model: str = "gpt-4"
    sessions: Dict[str, ChatSession] = field(default_factory=dict)
```

### Phase 2: Event Handler Extraction (Week 2)

#### 2.1 Create Handler Registry
```python
# tldw_chatbook/handlers/__init__.py
from .handler_registry import HandlerRegistry

# tldw_chatbook/handlers/handler_registry.py
class HandlerRegistry:
    """Central registry for all event handlers."""
    
    def __init__(self, app: App):
        self.app = app
        self._handlers = {}
    
    def register(self, event_type, handler):
        """Register a handler for an event type."""
        self._handlers[event_type] = handler
    
    def handle(self, event):
        """Route event to appropriate handler."""
        handler = self._handlers.get(type(event))
        if handler:
            return handler(self.app, event)
```

#### 2.2 Extract Event Handlers
```python
# BEFORE: In app.py
@on(Button.Pressed)
async def on_button_pressed(self, event):
    # 200 lines of if/elif logic
    
# AFTER: In handlers/button_handler.py
class ButtonHandler:
    """Handles all button press events."""
    
    def __init__(self, app: App):
        self.app = app
    
    async def handle(self, event: Button.Pressed):
        """Route button press to specific handler."""
        handlers = {
            "save-button": self._handle_save,
            "cancel-button": self._handle_cancel,
            # ...
        }
        
        handler = handlers.get(event.button.id)
        if handler:
            await handler(event)
```

### Phase 3: Screen Navigation Cleanup (Week 3)

#### 3.1 Create Navigation Manager
```python
# tldw_chatbook/navigation/navigation_manager.py
from typing import Dict, Type
from textual.screen import Screen

class NavigationManager:
    """Manages screen navigation and history."""
    
    def __init__(self, app: App):
        self.app = app
        self.screen_map = self._build_screen_map()
        self.history = []
    
    def _build_screen_map(self) -> Dict[str, Type[Screen]]:
        """Build map of screen names to classes."""
        return {
            'chat': ChatScreen,
            'notes': NotesScreen,
            # ... all 17 screens
        }
    
    async def navigate_to(self, screen_name: str):
        """Navigate to a screen by name."""
        screen_class = self.screen_map.get(screen_name)
        if not screen_class:
            raise ValueError(f"Unknown screen: {screen_name}")
        
        # Track history
        self.history.append(screen_name)
        
        # Switch screen
        await self.app.switch_screen(screen_class(self.app))
    
    async def go_back(self):
        """Navigate to previous screen."""
        if len(self.history) > 1:
            self.history.pop()
            await self.navigate_to(self.history[-1])
```

#### 3.2 Simplify App Navigation
```python
# BEFORE: In app.py
@on(NavigateToScreen)
async def handle_screen_navigation(self, message):
    # 50 lines of screen mapping logic

# AFTER: In app.py
def __init__(self):
    super().__init__()
    self.nav_manager = NavigationManager(self)

@on(NavigateToScreen)
async def handle_screen_navigation(self, message):
    await self.nav_manager.navigate_to(message.screen_name)
```

### Phase 4: Message-Based Architecture (Week 4)

#### 4.1 Define Application Messages
```python
# tldw_chatbook/messages/__init__.py
from textual.message import Message

class StateChanged(Message):
    """Emitted when application state changes."""
    def __init__(self, state_path: str, old_value, new_value):
        super().__init__()
        self.state_path = state_path
        self.old_value = old_value
        self.new_value = new_value

class SaveRequested(Message):
    """Request to save current data."""
    pass

class LoadCompleted(Message):
    """Data loading completed."""
    def __init__(self, data):
        super().__init__()
        self.data = data
```

#### 4.2 Replace Direct Calls
```python
# BEFORE: Direct manipulation
self.query_one("#status").update("Loading...")
db_result = self.db.load_data()
self.query_one("#content").update(db_result)

# AFTER: Message-based
self.post_message(StatusUpdate("Loading..."))
self.post_message(LoadDataRequest())

@on(LoadCompleted)
def handle_load_completed(self, message):
    self.post_message(ContentUpdate(message.data))
```

### Phase 5: Initialization Cleanup (Week 5)

#### 5.1 Create App Builder
```python
# tldw_chatbook/core/app_builder.py
class AppBuilder:
    """Builds and configures the application."""
    
    def __init__(self):
        self.config = {}
        self.handlers = []
        self.state = AppState()
    
    def with_config(self, config_path: str):
        """Load configuration."""
        self.config = load_config(config_path)
        return self
    
    def with_handlers(self, *handlers):
        """Register event handlers."""
        self.handlers.extend(handlers)
        return self
    
    def build(self) -> TldwCli:
        """Build configured application."""
        app = TldwCli()
        app.state = self.state
        app.config = self.config
        
        for handler in self.handlers:
            app.register_handler(handler)
        
        return app
```

#### 5.2 Simplify App Class
```python
# BEFORE: Complex __init__ with 200+ lines
class TldwCli(App):
    def __init__(self):
        super().__init__()
        # 200+ lines of initialization
        
# AFTER: Clean initialization
class TldwCli(App):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.nav_manager = NavigationManager(self)
        self.handler_registry = HandlerRegistry(self)
```

### Phase 6: Remove Obsolete Code (Week 6)

#### 6.1 Remove Tab-Based Code
- Delete all tab switching logic
- Remove PlaceholderWindow class
- Clean up compose method
- Remove tab-related reactive attributes

#### 6.2 Remove Direct Widget Queries
- Replace all query_one calls with messages
- Remove widget caching
- Clean up widget references

---

## Refactored App.py Structure

```python
# tldw_chatbook/app.py (< 500 lines)
from textual.app import App, ComposeResult
from textual import on
from textual.reactive import reactive

from .state import AppState
from .navigation import NavigationManager
from .handlers import HandlerRegistry
from .messages import NavigateToScreen

class TldwCli(App):
    """Main application following Textual best practices."""
    
    # Minimal reactive state
    state = reactive(AppState())
    
    # CSS and bindings
    CSS_PATH = "css/tldw_cli_modular.tcss"
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+b", "toggle_sidebar", "Toggle Sidebar"),
    ]
    
    def __init__(self):
        super().__init__()
        self.nav_manager = NavigationManager(self)
        self.handler_registry = HandlerRegistry(self)
        self._setup_handlers()
    
    def compose(self) -> ComposeResult:
        """Compose the UI."""
        from .UI.titlebar import TitleBar
        from .UI.Tab_Links import TabLinks
        from .UI.footer import AppFooterStatus
        
        yield TitleBar()
        yield TabLinks(ALL_TABS, "chat")
        yield Container(id="screen-container")
        yield AppFooterStatus()
    
    async def on_mount(self):
        """Initialize application."""
        # Load initial screen
        await self.nav_manager.navigate_to("chat")
        
        # Set up auto-save
        self.set_interval(30, self._auto_save)
    
    @on(NavigateToScreen)
    async def handle_navigation(self, message: NavigateToScreen):
        """Handle screen navigation."""
        await self.nav_manager.navigate_to(message.screen_name)
    
    def _setup_handlers(self):
        """Register all event handlers."""
        from .handlers import (
            ButtonHandler,
            KeyboardHandler,
            StateHandler
        )
        
        self.handler_registry.register(Button.Pressed, ButtonHandler(self))
        self.handler_registry.register(Key.Pressed, KeyboardHandler(self))
        self.handler_registry.register(StateChanged, StateHandler(self))
    
    async def _auto_save(self):
        """Auto-save application state."""
        from .persistence import save_state
        await save_state(self.state)
```

---

## Implementation Plan

### Week 1: State Extraction
- [ ] Create state module structure
- [ ] Define state containers
- [ ] Migrate reactive attributes
- [ ] Update references

### Week 2: Event Handlers
- [ ] Create handler registry
- [ ] Extract button handlers
- [ ] Extract keyboard handlers
- [ ] Extract custom event handlers

### Week 3: Navigation
- [ ] Create navigation manager
- [ ] Clean up screen mapping
- [ ] Implement history tracking
- [ ] Add navigation shortcuts

### Week 4: Messages
- [ ] Define application messages
- [ ] Replace direct widget calls
- [ ] Implement message routing
- [ ] Add message logging

### Week 5: Initialization
- [ ] Create app builder
- [ ] Clean up __init__
- [ ] Extract configuration
- [ ] Simplify mounting

### Week 6: Cleanup
- [ ] Remove tab code
- [ ] Delete unused methods
- [ ] Clean up imports
- [ ] Add documentation

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| File size (lines) | 5,857 | < 500 |
| Methods | 176 | < 20 |
| Reactive attributes | 65 | < 10 |
| Cyclomatic complexity | High | Low |
| Test coverage | ~20% | > 80% |
| Load time | 3-5s | < 1s |

---

## Benefits After Refactoring

1. **Maintainability** - Easy to understand and modify
2. **Testability** - Each component can be tested in isolation
3. **Performance** - Faster startup and runtime
4. **Extensibility** - Easy to add new features
5. **Debugging** - Clear separation of concerns
6. **Team Collaboration** - Multiple developers can work on different parts

---

## Next Steps

1. Review and approve this plan
2. Create feature branch for refactoring
3. Start with Phase 1 (State Extraction)
4. Test after each phase
5. Document changes

This refactoring will transform app.py from a monolithic god object into a clean, maintainable application following Textual best practices.