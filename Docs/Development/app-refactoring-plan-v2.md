# App.py Refactoring Plan v2.0
## Following Textual Best Practices - Corrected Version

**Current State:** 5,857 lines, 176 methods, 65 reactive attributes  
**Target State:** < 500 lines, < 20 methods, properly managed reactive state

---

## Critical Corrections from v1.0

### ✅ Fixed Issues:
1. **State Management** - Use individual reactive attributes, not complex objects
2. **Screen Construction** - Proper parameter handling for different screen types
3. **Resource Paths** - Absolute paths for CSS and resources
4. **Error Handling** - Comprehensive error handling throughout
5. **Compatibility Layer** - Gradual migration support

---

## Refactoring Strategy (Revised)

### Phase 1: Reactive State Architecture (Week 1)

#### 1.1 Correct State Management Approach

**❌ WRONG (from v1.0):**
```python
# This won't work - reactive() can't handle complex dataclasses
state = reactive(AppState())
```

**✅ CORRECT Approach:**
```python
# tldw_chatbook/app_refactored.py
from textual.reactive import reactive

class TldwCliRefactored(App):
    """App with properly managed reactive state."""
    
    # Individual reactive attributes for each state domain
    current_screen = reactive("chat")
    is_loading = reactive(False)
    theme = reactive("default")
    
    # Use reactive dictionaries for complex state
    chat_state = reactive({
        "provider": "openai",
        "model": "gpt-4",
        "is_streaming": False,
        "sidebar_collapsed": False
    })
    
    notes_state = reactive({
        "selected_note_id": None,
        "unsaved_changes": False,
        "preview_mode": False
    })
    
    ui_state = reactive({
        "sidebars": {},
        "modal_open": False,
        "dark_mode": True
    })
```

#### 1.2 State Container Pattern (Non-Reactive)

```python
# tldw_chatbook/state/state_manager.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
from datetime import datetime

class StateManager:
    """
    Manages application state without reactivity.
    State changes trigger reactive updates in the app.
    """
    
    def __init__(self, app: App):
        self.app = app
        self._state = {
            "navigation": NavigationState(),
            "chat": ChatState(),
            "notes": NotesState(),
            "ui": UIState()
        }
    
    def update_chat_provider(self, provider: str, model: str):
        """Update chat provider and trigger reactive update."""
        self._state["chat"].provider = provider
        self._state["chat"].model = model
        
        # Update reactive attribute to trigger UI update
        self.app.chat_state = {
            **self.app.chat_state,
            "provider": provider,
            "model": model
        }
    
    def save_state(self, path: Path):
        """Save state with proper serialization."""
        state_dict = {}
        for key, value in self._state.items():
            if hasattr(value, 'to_dict'):
                state_dict[key] = value.to_dict()
            else:
                state_dict[key] = str(value)
        
        # Custom JSON encoder for datetime
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
        
        path.write_text(json.dumps(state_dict, cls=DateTimeEncoder, indent=2))
```

### Phase 2: Screen Navigation with Compatibility (Week 2)

#### 2.1 Enhanced Navigation Manager

```python
# tldw_chatbook/navigation/navigation_manager.py
from typing import Optional, Type, Dict
from textual.screen import Screen
from loguru import logger
import inspect

class NavigationManager:
    """Navigation manager with error handling and compatibility."""
    
    def __init__(self, app: App):
        self.app = app
        self.screen_cache: Dict[str, Screen] = {}
        self.screen_registry = self._build_registry()
        
    def _build_registry(self) -> Dict[str, Type[Screen]]:
        """Build screen registry with fallbacks."""
        registry = {}
        
        # Try new screen locations first, fallback to old
        screens_to_load = [
            ("chat", "UI.Screens.chat_screen", "ChatScreen", 
             "UI.Chat_Window_Enhanced", "ChatWindowEnhanced"),
            ("notes", "UI.Screens.notes_screen", "NotesScreen",
             "UI.Notes_Window", "NotesWindow"),
            # ... other screens
        ]
        
        for screen_name, new_module, new_class, old_module, old_class in screens_to_load:
            try:
                # Try new location
                module = __import__(f"tldw_chatbook.{new_module}", fromlist=[new_class])
                registry[screen_name] = getattr(module, new_class)
                logger.debug(f"Loaded {screen_name} from new location")
            except (ImportError, AttributeError):
                try:
                    # Fallback to old location
                    module = __import__(f"tldw_chatbook.{old_module}", fromlist=[old_class])
                    registry[screen_name] = getattr(module, old_class)
                    logger.warning(f"Using legacy location for {screen_name}")
                except (ImportError, AttributeError) as e:
                    logger.error(f"Failed to load screen {screen_name}: {e}")
        
        return registry
    
    def _create_screen(self, screen_name: str, screen_class: Type[Screen]) -> Optional[Screen]:
        """Create screen with proper parameter handling."""
        try:
            # Check what parameters the screen expects
            sig = inspect.signature(screen_class.__init__)
            params = list(sig.parameters.keys())
            
            # Remove 'self' from parameters
            if 'self' in params:
                params.remove('self')
            
            # Determine how to construct the screen
            if not params:
                # No parameters needed
                return screen_class()
            elif 'app' in params or 'app_instance' in params:
                # Expects app parameter
                return screen_class(self.app)
            else:
                # Try with no parameters as fallback
                return screen_class()
                
        except Exception as e:
            logger.error(f"Failed to create screen {screen_name}: {e}")
            return None
    
    async def navigate_to(self, screen_name: str) -> bool:
        """Navigate with error handling and recovery."""
        try:
            # Check current screen
            if self.app.current_screen == screen_name:
                logger.debug(f"Already on screen: {screen_name}")
                return True
            
            # Get screen class
            screen_class = self.screen_registry.get(screen_name)
            if not screen_class:
                logger.error(f"Unknown screen: {screen_name}")
                self.app.notify(f"Screen '{screen_name}' not found", severity="error")
                return False
            
            # Create or get cached screen
            if screen_name in self.screen_cache:
                screen = self.screen_cache[screen_name]
            else:
                screen = self._create_screen(screen_name, screen_class)
                if not screen:
                    self.app.notify(f"Failed to create screen '{screen_name}'", severity="error")
                    return False
                
                # Cache for reuse (optional)
                if self._should_cache(screen_name):
                    self.screen_cache[screen_name] = screen
            
            # Update loading state
            self.app.is_loading = True
            
            # Switch screen
            await self.app.switch_screen(screen)
            
            # Update state
            self.app.current_screen = screen_name
            self.app.is_loading = False
            
            logger.info(f"Navigated to: {screen_name}")
            return True
            
        except Exception as e:
            logger.error(f"Navigation failed: {e}", exc_info=True)
            self.app.is_loading = False
            self.app.notify("Navigation failed", severity="error")
            return False
    
    def _should_cache(self, screen_name: str) -> bool:
        """Determine if screen should be cached."""
        # Cache frequently used screens
        return screen_name in ["chat", "notes", "media"]
    
    def clear_cache(self, screen_name: Optional[str] = None):
        """Clear screen cache."""
        if screen_name:
            self.screen_cache.pop(screen_name, None)
        else:
            self.screen_cache.clear()
```

### Phase 3: Message-Based Architecture (Week 3)

#### 3.1 Application Messages

```python
# tldw_chatbook/messages.py
from textual.message import Message
from typing import Any, Optional, Dict

class StateUpdateMessage(Message):
    """Message for state updates."""
    def __init__(self, domain: str, key: str, value: Any):
        super().__init__()
        self.domain = domain  # e.g., "chat", "notes"
        self.key = key
        self.value = value

class NavigationMessage(Message):
    """Enhanced navigation message."""
    def __init__(self, screen: str, params: Optional[Dict] = None):
        super().__init__()
        self.screen = screen
        self.params = params or {}

class ErrorMessage(Message):
    """Message for error handling."""
    def __init__(self, error: str, severity: str = "error"):
        super().__init__()
        self.error = error
        self.severity = severity

class SaveStateMessage(Message):
    """Request to save application state."""
    pass

class LoadStateMessage(Message):
    """Request to load application state."""
    pass
```

#### 3.2 Message Handlers

```python
# tldw_chatbook/handlers/message_handler.py
from textual import on

class MessageHandler:
    """Centralized message handling."""
    
    def __init__(self, app: App):
        self.app = app
    
    @on(StateUpdateMessage)
    async def handle_state_update(self, message: StateUpdateMessage):
        """Handle state update messages."""
        domain = message.domain
        key = message.key
        value = message.value
        
        # Update the appropriate reactive state
        if domain == "chat":
            self.app.chat_state = {
                **self.app.chat_state,
                key: value
            }
        elif domain == "notes":
            self.app.notes_state = {
                **self.app.notes_state,
                key: value
            }
        elif domain == "ui":
            self.app.ui_state = {
                **self.app.ui_state,
                key: value
            }
    
    @on(ErrorMessage)
    async def handle_error(self, message: ErrorMessage):
        """Handle error messages."""
        self.app.notify(message.error, severity=message.severity)
        logger.error(f"Error: {message.error}")
```

### Phase 4: Proper App Implementation (Week 4)

#### 4.1 Corrected App Class

```python
# tldw_chatbook/app_refactored.py
import os
from pathlib import Path
from typing import Optional, Dict, Any

from loguru import logger
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive, Reactive
from textual.widgets import Button

# Proper imports with error handling
try:
    from .state.state_manager import StateManager
    from .navigation.navigation_manager import NavigationManager
    from .handlers.message_handler import MessageHandler
except ImportError as e:
    logger.error(f"Import error - using fallbacks: {e}")
    StateManager = None
    NavigationManager = None
    MessageHandler = None


class TldwCliRefactored(App):
    """Refactored app following Textual best practices."""
    
    # Proper CSS path
    CSS_PATH = Path(__file__).parent / "css" / "tldw_cli_modular.tcss"
    
    # Key bindings
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+s", "save", "Save"),
        ("escape", "back", "Back"),
    ]
    
    # Reactive attributes (simple types only)
    current_screen: Reactive[str] = reactive("chat")
    is_loading: Reactive[bool] = reactive(False)
    theme: Reactive[str] = reactive("default")
    
    # Reactive dictionaries for complex state
    chat_state: Reactive[Dict[str, Any]] = reactive({
        "provider": "openai",
        "model": "gpt-4",
        "is_streaming": False
    })
    
    notes_state: Reactive[Dict[str, Any]] = reactive({
        "selected_note_id": None,
        "unsaved_changes": False
    })
    
    ui_state: Reactive[Dict[str, Any]] = reactive({
        "sidebars": {},
        "dark_mode": True
    })
    
    def __init__(self):
        """Initialize with proper error handling."""
        super().__init__()
        
        # Initialize managers with error handling
        try:
            self.state_manager = StateManager(self) if StateManager else None
            self.nav_manager = NavigationManager(self) if NavigationManager else None
            self.message_handler = MessageHandler(self) if MessageHandler else None
        except Exception as e:
            logger.error(f"Failed to initialize managers: {e}")
            self.state_manager = None
            self.nav_manager = None
            self.message_handler = None
        
        # Load configuration safely
        self._load_config()
    
    def _load_config(self):
        """Load configuration with error handling."""
        try:
            from .config import load_cli_config_and_ensure_existence
            load_cli_config_and_ensure_existence()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    def compose(self) -> ComposeResult:
        """Compose UI with error handling."""
        try:
            # Check for splash screen
            from .config import get_cli_setting
            if get_cli_setting("splash_screen", "enabled", False):
                from .Widgets.splash_screen import SplashScreen
                yield SplashScreen(id="splash")
                return
        except Exception as e:
            logger.error(f"Splash screen error: {e}")
        
        # Compose main UI
        yield from self._compose_main_ui()
    
    def _compose_main_ui(self) -> ComposeResult:
        """Compose main UI components."""
        try:
            from .UI.titlebar import TitleBar
            yield TitleBar()
        except ImportError:
            yield Container()  # Fallback
        
        try:
            from .UI.Tab_Links import TabLinks
            from .Constants import ALL_TABS
            yield TabLinks(tab_ids=ALL_TABS, initial_active_tab="chat")
        except ImportError:
            yield Container()  # Fallback
        
        # Screen container
        yield Container(id="screen-container")
        
        try:
            from .Widgets.AppFooterStatus import AppFooterStatus
            yield AppFooterStatus()
        except ImportError:
            yield Container()  # Fallback
    
    async def on_mount(self):
        """Mount with error handling."""
        try:
            # Navigate to initial screen
            if self.nav_manager:
                await self.nav_manager.navigate_to(self.current_screen)
            else:
                logger.error("Navigation manager not available")
                
            # Load saved state
            await self._load_state()
            
        except Exception as e:
            logger.error(f"Mount error: {e}")
            self.notify("Failed to initialize", severity="error")
    
    # Reactive watchers
    
    def watch_current_screen(self, old_screen: str, new_screen: str):
        """React to screen changes."""
        logger.info(f"Screen changed: {old_screen} -> {new_screen}")
        
        # Update any dependent state
        if self.state_manager:
            self.state_manager.on_screen_change(new_screen)
    
    def watch_theme(self, old_theme: str, new_theme: str):
        """React to theme changes."""
        # Apply theme changes
        logger.info(f"Theme changed: {old_theme} -> {new_theme}")
    
    # Event handlers
    
    @on(NavigateToScreen)
    async def handle_navigation(self, message: NavigateToScreen):
        """Handle navigation with error recovery."""
        if self.nav_manager:
            success = await self.nav_manager.navigate_to(message.screen_name)
            if not success:
                # Try to go home as fallback
                await self.nav_manager.navigate_to("chat")
        else:
            logger.error("Navigation manager not available")
    
    @on(Button.Pressed)
    async def handle_button(self, event: Button.Pressed):
        """Handle button presses with compatibility."""
        button_id = event.button.id
        
        if not button_id:
            return
        
        # Compatibility layer for old tab buttons
        if button_id.startswith("tab-"):
            screen_name = button_id[4:]
            await self.handle_navigation(NavigateToScreen(screen_name=screen_name))
        
        # Handle other buttons
        elif button_id == "save":
            await self.action_save()
        elif button_id == "quit":
            self.exit()
    
    # Actions
    
    async def action_save(self):
        """Save action with error handling."""
        try:
            await self._save_state()
            self.notify("Saved")
        except Exception as e:
            logger.error(f"Save failed: {e}")
            self.notify("Save failed", severity="error")
    
    async def action_quit(self):
        """Quit with cleanup."""
        try:
            await self._save_state()
        except:
            pass  # Don't block quit
        finally:
            self.exit()
    
    async def action_back(self):
        """Go back with error handling."""
        if self.nav_manager:
            await self.nav_manager.go_back()
    
    # State persistence
    
    async def _save_state(self):
        """Save state with proper error handling."""
        try:
            state_path = Path.home() / ".config" / "tldw_cli" / "state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.state_manager:
                self.state_manager.save_state(state_path)
            else:
                # Fallback: save reactive state directly
                import json
                state = {
                    "current_screen": self.current_screen,
                    "theme": self.theme,
                    "chat_state": dict(self.chat_state),
                    "notes_state": dict(self.notes_state),
                    "ui_state": dict(self.ui_state)
                }
                state_path.write_text(json.dumps(state, indent=2))
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def _load_state(self):
        """Load state with error handling."""
        try:
            state_path = Path.home() / ".config" / "tldw_cli" / "state.json"
            if not state_path.exists():
                return
            
            import json
            state = json.loads(state_path.read_text())
            
            # Update reactive attributes
            if "current_screen" in state:
                self.current_screen = state["current_screen"]
            if "theme" in state:
                self.theme = state["theme"]
            if "chat_state" in state:
                self.chat_state = state["chat_state"]
            if "notes_state" in state:
                self.notes_state = state["notes_state"]
            if "ui_state" in state:
                self.ui_state = state["ui_state"]
                
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
```

---

## Implementation Plan (Revised)

### Week 1: Foundation
- [ ] Implement proper reactive state architecture
- [ ] Create StateManager with serialization
- [ ] Add comprehensive error handling
- [ ] Write unit tests for state management

### Week 2: Navigation
- [ ] Implement NavigationManager with compatibility
- [ ] Add screen registry with fallbacks
- [ ] Test all 17 screens load correctly
- [ ] Add navigation error recovery

### Week 3: Messages
- [ ] Define application messages
- [ ] Implement message handlers
- [ ] Convert direct calls to messages
- [ ] Add message logging

### Week 4: Integration
- [ ] Integrate all components
- [ ] Add compatibility layer
- [ ] Test with existing screens
- [ ] Performance profiling

### Week 5: Migration
- [ ] Update screens one by one
- [ ] Maintain backward compatibility
- [ ] Run parallel testing
- [ ] Document migration steps

### Week 6: Cleanup
- [ ] Remove obsolete code
- [ ] Update documentation
- [ ] Final testing
- [ ] Deploy

---

## Testing Strategy

### 1. Unit Tests
```python
def test_reactive_state():
    """Test reactive attributes work correctly."""
    app = TldwCliRefactored()
    
    # Test simple reactive
    app.current_screen = "notes"
    assert app.current_screen == "notes"
    
    # Test dict reactive
    app.chat_state = {**app.chat_state, "provider": "anthropic"}
    assert app.chat_state["provider"] == "anthropic"

def test_navigation_manager():
    """Test navigation with error handling."""
    app = TldwCliRefactored()
    nav = NavigationManager(app)
    
    # Test successful navigation
    assert asyncio.run(nav.navigate_to("chat"))
    
    # Test invalid screen
    assert not asyncio.run(nav.navigate_to("invalid"))
```

### 2. Integration Tests
```python
@pytest.mark.asyncio
async def test_full_navigation_flow():
    """Test complete navigation flow."""
    app = TldwCliRefactored()
    async with app.run_test() as pilot:
        # Test initial state
        assert app.current_screen == "chat"
        
        # Navigate to notes
        app.post_message(NavigateToScreen("notes"))
        await pilot.pause()
        assert app.current_screen == "notes"
        
        # Test error recovery
        app.post_message(NavigateToScreen("invalid"))
        await pilot.pause()
        # Should still be on notes or fallback to chat
        assert app.current_screen in ["notes", "chat"]
```

---

## Migration Checklist

### Pre-Migration
- [ ] Full backup of current app
- [ ] Document all custom modifications
- [ ] Test suite passing on old app
- [ ] Performance baseline recorded

### During Migration
- [ ] Run both apps in parallel
- [ ] Test each screen individually
- [ ] Verify state persistence
- [ ] Check error handling
- [ ] Monitor performance

### Post-Migration
- [ ] All tests passing
- [ ] Performance improved
- [ ] Documentation updated
- [ ] Team trained on new architecture

---

## Success Metrics

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Lines of Code | 5,857 | < 500 | `wc -l` |
| Startup Time | 3-5s | < 1s | Timer |
| Memory Usage | 500MB | < 200MB | Memory profiler |
| Test Coverage | ~20% | > 80% | pytest-cov |
| Error Rate | High | < 1% | Error logs |
| Code Complexity | Very High | Low | Cyclomatic complexity |

---

This revised plan addresses all the issues found and provides a robust, error-resistant implementation that follows Textual best practices.