# Application-Wide Screen Navigation Migration Plan

## Overview
This document outlines how to migrate the entire tldw_chatbook application from tab-based navigation to screen-based navigation, following the pattern established in the Media Ingestion module.

## Current Architecture (Tab-Based)
- All windows are loaded as containers within tabs
- Navigation happens via TabBar/TabLinks/TabDropdown
- All tabs remain in memory once loaded
- State management is handled within each window

## Proposed Architecture (Screen-Based)
- Each major feature becomes a Screen subclass
- Top-level navigation bar with links
- Only active screen is loaded in memory
- Centralized state management with ScreenManager

## Migration Steps

### Phase 1: Core Infrastructure
1. **Extend ScreenManager**
   ```python
   class GlobalScreenManager(ScreenManager):
       """Global screen manager for the entire application."""
       
       def __init__(self, app: App):
           super().__init__(app)
           self.register_screens()
       
       def register_screens(self):
           """Register all application screens."""
           self.screens = {
               'chat': ChatScreen,
               'conversations': ConversationsScreen,
               'notes': NotesScreen,
               'media': MediaScreen,
               'search': SearchScreen,
               'ingest': MediaIngestDispatcher,
               'tools': ToolsSettingsScreen,
               'llm': LLMManagementScreen,
               'customize': CustomizeScreen,
               'logs': LogsScreen,
               'coding': CodingScreen,
               'stats': StatsScreen,
               'evals': EvalsScreen,
           }
   ```

2. **Create AppNavigationBar**
   ```python
   class AppNavigationBar(Container):
       """Main application navigation bar."""
       
       def compose(self) -> ComposeResult:
           with Horizontal(classes="app-nav"):
               yield Button("Chat", id="nav-chat")
               yield Static("|")
               yield Button("Conversations", id="nav-conversations")
               # ... etc
   ```

### Phase 2: Convert Windows to Screens

#### Example: ChatWindow → ChatScreen
```python
# Before (Window/Container)
class ChatWindow(Container):
    def compose(self) -> ComposeResult:
        # UI components
        
# After (Screen)
class ChatScreen(Screen):
    def compose(self) -> ComposeResult:
        yield AppNavigationBar(active="chat")
        with Container(id="chat-content"):
            # Original UI components
    
    def save_state(self) -> dict:
        """Save chat state."""
        return {
            'conversation_id': self.current_conversation_id,
            'messages': self.messages,
            'input_text': self.query_one("#chat-input").value
        }
    
    def restore_state(self, state: dict) -> None:
        """Restore chat state."""
        self.current_conversation_id = state.get('conversation_id')
        self.messages = state.get('messages', [])
        # Restore input text, etc.
```

### Phase 3: Update Main App

```python
class TldwCli(App[None]):
    def __init__(self):
        super().__init__()
        self.screen_manager = GlobalScreenManager(self)
        self.current_screen_name = "chat"
    
    def compose(self) -> ComposeResult:
        """Compose just shows initial screen."""
        yield ChatScreen(self)
    
    @on(NavigateToScreen)
    async def handle_navigation(self, message: NavigateToScreen):
        """Handle screen navigation."""
        await self.switch_to_screen(message.screen_name)
    
    async def switch_to_screen(self, screen_name: str):
        """Switch to a different screen."""
        # Save current state
        self.screen_manager.save_current_state()
        
        # Get screen class
        screen_class = self.screen_manager.screens.get(screen_name)
        if not screen_class:
            return
        
        # Create and push new screen
        new_screen = screen_class(self)
        await self.push_screen(new_screen)
        
        # Update tracking
        self.current_screen_name = screen_name
        
        # Restore state if available
        self.screen_manager.restore_screen_state(new_screen)
```

### Phase 4: State Persistence

1. **Add persistent storage**
   ```python
   class StatePersistence:
       """Persist screen states across sessions."""
       
       def __init__(self, app_id: str):
           self.state_file = Path.home() / f".{app_id}_state.json"
       
       def save(self, states: dict):
           """Save all screen states to disk."""
           with open(self.state_file, 'w') as f:
               json.dump(states, f)
       
       def load(self) -> dict:
           """Load screen states from disk."""
           if self.state_file.exists():
               with open(self.state_file) as f:
                   return json.load(f)
           return {}
   ```

2. **Auto-save on exit**
   ```python
   def on_unmount(self):
       """Save all states on app exit."""
       self.screen_manager.save_all_states()
       self.state_persistence.save(self.screen_manager.get_all_states())
   ```

## Benefits

### Performance
- **Memory**: Only active screen loaded (~80% reduction in memory usage)
- **Startup**: Faster initial load (load only first screen)
- **Navigation**: Instant switching for previously loaded screens

### User Experience
- **State Preservation**: Forms and scroll positions maintained
- **Deep Linking**: Direct navigation to specific screens
- **Keyboard Shortcuts**: Global shortcuts for screen switching
- **History**: Back/forward navigation support

### Development
- **Testing**: Each screen can be tested independently
- **Modularity**: Screens are self-contained units
- **Debugging**: Easier to trace issues to specific screens
- **Hot Reload**: Can reload individual screens during development

## Migration Timeline

### Week 1-2: Infrastructure
- Implement GlobalScreenManager
- Create AppNavigationBar
- Set up state persistence
- Create base screen classes

### Week 3-4: Core Screens
- Convert ChatWindow → ChatScreen
- Convert NotesWindow → NotesScreen
- Convert MediaWindow → MediaScreen
- Test state management

### Week 5-6: Remaining Screens
- Convert all remaining windows
- Implement navigation shortcuts
- Add history support
- Performance testing

### Week 7-8: Polish & Testing
- User acceptance testing
- Performance optimization
- Documentation updates
- Migration guide for plugins

## Rollback Plan

If issues arise, the migration can be rolled back by:
1. Keeping the old tab-based code in a separate branch
2. Using a feature flag to switch between implementations
3. Gradual rollout (some features use screens, others use tabs)

## Configuration

Add to config.toml:
```toml
[navigation]
type = "screen"  # or "tab" for legacy
save_state = true
state_file = "~/.tldw_cli_state.json"
animation = "slide"  # screen transition animation
history_size = 10

[screens]
default = "chat"
preload = ["chat", "notes"]  # screens to preload
lazy_load_delay = 500  # ms before loading other screens
```

## Testing Strategy

1. **Unit Tests**: Each screen tested independently
2. **Integration Tests**: Navigation flow testing
3. **State Tests**: Verify state save/restore
4. **Performance Tests**: Memory and speed benchmarks
5. **User Tests**: A/B testing with select users

## Conclusion

The screen-based navigation approach offers significant advantages in terms of performance, maintainability, and user experience. The migration can be done incrementally, allowing for thorough testing at each phase. The Media Ingestion module serves as a proven example of this architecture working effectively.