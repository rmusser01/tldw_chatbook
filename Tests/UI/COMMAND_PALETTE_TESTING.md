# Command Palette Testing Guide

This document describes the comprehensive command palette implementation and testing approach for tldw_chatbook.

## Implementation Summary

### ✅ Implemented Command Palette Providers

1. **ThemeProvider** - Theme switching functionality
2. **TabNavigationProvider** - Tab navigation commands  
3. **LLMProviderProvider** - LLM provider management
4. **QuickActionsProvider** - Quick action shortcuts
5. **SettingsProvider** - Settings and preferences
6. **CharacterProvider** - Character/persona management
7. **MediaProvider** - Media and content management
8. **DeveloperProvider** - Developer and debug commands

### ✅ Configuration Integration

- **Default theme on startup** from `~/.config/tldw_cli/config.toml`
- **Theme selection persistence** - selections are saved to config
- **Configurable theme limit** in command palette

### ✅ Key Bindings

- **Ctrl+P** - Opens command palette
- **Ctrl+Q** - Quit application

## Manual Testing Procedures

### 1. Theme Provider Testing

```bash
# Test basic theme functionality
python -m tldw_chatbook.app

# In the app:
1. Press Ctrl+P
2. Should see "Theme: Change Theme" option
3. Press Enter on it -> Shows helpful message
4. Type "theme" -> Should show all available themes
5. Type "dark" -> Should show dark themes
6. Select a theme -> Should apply and save to config
```

**Expected Results:**
- Single "Change Theme" entry in default palette view
- All 60+ themes available when searching "theme"
- Theme changes are applied immediately
- Theme preference is saved to config file
- Helpful instruction message when selecting main theme command

### 2. Tab Navigation Testing

```bash
# Test tab navigation
python -m tldw_chatbook.app

# In the app:
1. Press Ctrl+P
2. Type "tab" or "switch"
3. Should see all 12 tab options
4. Select any tab -> Should switch immediately
```

**Expected Results:**
- All tabs available: Chat, Character Chat, Notes, Media, Search, Ingest, Tools & Settings, LLM Management, Logs, Stats, Evaluations, Coding
- Immediate tab switching with success notification
- Popular tabs shown in discovery mode

### 3. Quick Actions Testing

```bash
# Test quick actions
python -m tldw_chatbook.app

# In the app:
1. Press Ctrl+P  
2. Type "new" or "quick"
3. Should see action options
4. Try "New Chat", "New Note", etc.
```

**Expected Results:**
- Quick access to common actions
- Appropriate tab switching for actions
- Success notifications

### 4. Settings Provider Testing

```bash
# Test settings commands
python -m tldw_chatbook.app

# In the app:
1. Press Ctrl+P
2. Type "settings" or "config"
3. Should see settings options
4. Try "Open Config File" -> Should show path
5. Try "Open Settings Tab" -> Should switch to Tools & Settings
```

**Expected Results:**
- Settings commands available
- Config file path displayed
- Tab switching to settings works
- Temperature setting commands available

### 5. Integration Testing

```bash
# Test overall command palette functionality
python -m tldw_chatbook.app

# In the app:
1. Press Ctrl+P
2. Should see clean, organized command list
3. Type different queries:
   - "chat" -> Should show chat-related commands
   - "new" -> Should show creation commands  
   - "switch" -> Should show switching commands
   - "theme dark" -> Should show dark themes
4. Commands should execute without errors
5. All providers should be responsive
```

**Expected Results:**
- Command palette opens quickly
- Search is responsive with fuzzy matching
- Commands execute successfully
- Error handling works (no crashes)
- All 8 provider categories functional

## Automated Test Coverage

### Working Tests (`test_command_palette_basic.py`)

✅ **Import Tests**
- Command palette providers can be imported
- Constants are available
- App configuration is correct

✅ **Basic Structure Tests**  
- All providers have required methods
- App has COMMANDS registered
- Ctrl+P keybinding is registered
- Provider classes can be instantiated

✅ **Method Verification Tests**
- Providers have search() and discover() methods
- Theme provider has switch_theme() method
- Tab provider has switch_tab() method
- Methods are callable

### Test Limitations

❌ **Runtime Behavior Tests**
- Provider.app property is read-only (Textual framework limitation)
- Cannot mock app context for full integration testing
- Async method testing requires complex Textual app setup

❌ **UI Integration Tests**
- Textual's command palette testing requires AppTest which has version compatibility issues
- Full end-to-end testing requires running actual app

## Test Commands

```bash
# Run basic structural tests
python -m pytest Tests/UI/test_command_palette_basic.py -v

# Run import verification
python -c "from tldw_chatbook.app import ThemeProvider; print('✅ Imports work')"

# Check app configuration
python -c "
from tldw_chatbook.app import TldwCli
print('COMMANDS:', len(TldwCli.COMMANDS))
print('BINDINGS:', [str(b.key) for b in TldwCli.BINDINGS])
"
```

## Configuration Testing

### Test Default Theme Loading

```bash
# Check current config
cat ~/.config/tldw_cli/config.toml | grep default_theme

# Test theme persistence
python -m tldw_chatbook.app
# Change theme via Ctrl+P
# Exit and check config file was updated
```

### Test Theme Limit Configuration

```bash
# Edit config file
echo "palette_theme_limit = 5" >> ~/.config/tldw_cli/config.toml

# Test that theme search shows limited results
python -m tldw_chatbook.app
# Press Ctrl+P, type "theme" - should show fewer themes
```

## Error Handling Testing

### Test Invalid Theme

```bash
# Manually edit config to invalid theme
echo "default_theme = 'invalid-theme-name'" >> ~/.config/tldw_cli/config.toml

# App should start with fallback theme and log warning
python -m tldw_chatbook.app
```

### Test Provider Errors

All providers include try/catch blocks that:
- Show user-friendly error messages via app.notify()
- Don't crash the application
- Log errors appropriately

## Performance Testing

### Command Palette Responsiveness

```bash
# Time palette opening
python -m tldw_chatbook.app
# Press Ctrl+P multiple times - should be instant
# Type queries - should be responsive
# Select commands - should execute quickly
```

**Expected Performance:**
- Palette opens in <100ms
- Search results appear as you type
- Command execution is immediate
- No noticeable lag with 60+ themes

## Development Testing

For developers making changes to command palette providers:

1. **Add new commands** - Follow existing provider patterns
2. **Test new providers** - Add to COMMANDS set in app.py
3. **Verify error handling** - All methods should have try/catch
4. **Check help text** - All commands should have descriptive help
5. **Test search terms** - Ensure commands are discoverable

## Known Issues and Limitations

1. **Testing Framework**: Textual command palette testing has framework limitations
2. **App Context**: Providers require full app context for complete testing
3. **Async Testing**: Complex async behavior needs app integration
4. **Theme Loading**: Some themes may fail to load due to missing dependencies

## Success Criteria

✅ All 8 provider categories implemented  
✅ Theme persistence working  
✅ Clean command palette interface  
✅ Fuzzy search functionality  
✅ Error handling implemented  
✅ Configuration integration  
✅ Key bindings registered  
✅ Basic structural tests passing  

The command palette implementation provides a comprehensive, user-friendly interface for accessing all application functionality through a single keyboard shortcut (Ctrl+P).