# Chat Tabs Feature Guide

## Overview

The Chat Tabs feature allows you to have multiple concurrent chat sessions within the tldw_chatbook application. Each tab maintains its own independent conversation history, character assignment, and settings.

## Enabling Chat Tabs

Chat tabs are disabled by default. To enable them:

1. Edit your config file at `~/.config/tldw_cli/config.toml`
2. Find the `[chat_defaults]` section
3. Set `enable_tabs = true`
4. Optionally adjust `max_tabs` (default is 10)

```toml
[chat_defaults]
# ... other settings ...
enable_tabs = true  # Enable tabbed chat interface
max_tabs = 10      # Maximum number of chat tabs allowed
```

## Using Chat Tabs

### Creating a New Tab
- Click the "+" button in the tab bar
- Or use the keyboard shortcut: `Ctrl+T`

### Switching Between Tabs
- Click on a tab to switch to it
- Use keyboard shortcuts:
  - `Ctrl+Tab` - Next tab
  - `Ctrl+Shift+Tab` - Previous tab
  - `Ctrl+1` through `Ctrl+9` - Jump to specific tab (not yet implemented)

### Closing Tabs
- Click the "×" button on a tab
- Or use `Ctrl+W` to close the current tab
- Non-ephemeral chats will prompt for confirmation before closing

### Tab Features

Each tab maintains:
- **Independent conversation** - Messages are isolated per tab
- **Character assignment** - Each tab can have a different character
- **Conversation state** - Ephemeral or saved conversations
- **Streaming state** - Multiple tabs can stream responses simultaneously

### Tab Titles
- New tabs start with generic titles like "Chat 1", "Chat 2"
- When you save a conversation, the tab title updates to the conversation name
- Character assignments show an icon (👤) in the tab

## Technical Details

### Architecture
The implementation follows a modular pattern:
- `ChatTabContainer` - Manages multiple tabs and sessions
- `ChatTabBar` - Provides the tab navigation UI
- `ChatSession` - Individual chat session widget
- `ChatSessionData` - Data model for session state

### Limitations
- Maximum number of tabs is configurable (default: 10)
- Tab state is not persisted between app restarts (planned feature)
- All tabs share the same provider/model settings from the sidebar

### Compatibility
- Works with both basic and enhanced chat windows
- Compatible with all existing chat features (RAG, images, etc.)
- Event handlers need updates to support tab context (in progress)

## Future Enhancements

Planned improvements:
- Tab persistence across app restarts
- Tab-specific settings (model, temperature, etc.)
- Drag and drop tab reordering
- Tab duplication
- Export/import tab sessions
- Visual indicators for tabs with unread messages

## Troubleshooting

### Tabs not appearing
- Check that `enable_tabs = true` in your config
- Restart the application after changing the config

### Performance issues with many tabs
- Reduce `max_tabs` in config
- Close unused tabs to free memory

### Tab content not updating
- This might indicate event handler issues
- Check the logs for errors
- Report issues with specific reproduction steps