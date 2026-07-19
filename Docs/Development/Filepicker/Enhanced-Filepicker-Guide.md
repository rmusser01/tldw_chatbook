# Enhanced File Picker Guide

## Overview

The enhanced file picker in tldw_chatbook provides a feature-rich file selection experience with persistent settings, bookmarks, recent files, and context-aware defaults. It subclasses the vendored `textual_fspicker` base dialog and adds persistence and search without forking the third-party code. This guide covers the implemented features and how to use them effectively.

## Key Features

### 1. **Persistent Recent Files** 📋
- Automatically tracks the last 20 files/directories you've accessed
- Persists between sessions in the config file
- Context-aware: Different recent lists for different use cases
- Quick access with `Ctrl+R`

### 2. **Bookmarks System** ⭐
- Default bookmarks for common directories (Home, Desktop, Documents, Downloads)
- Add custom bookmarks with `Ctrl+D`
- Remove bookmarks by pressing `Ctrl+D` on a bookmarked directory
- Quick access with `Ctrl+B`
- Jump to bookmarks 1-9 with number keys

### 3. **Context-Aware Last Directory** 📁
- Remembers the last directory used for each context
- Different contexts for:
  - Character imports
  - Note attachments
  - Chat images
  - Document ingestion
  - LLM model files
  - And more...

### 4. **Enhanced Navigation** 🧭
- **Breadcrumb navigation**: Click any part of the path to jump there
- **Search functionality**: `Ctrl+F` to search files in current directory
- **Quick path editing**: `Ctrl+L` to edit the path directly
- **Hidden files toggle**: `Ctrl+H` to show/hide hidden files
- **Directory refresh**: `F5` to refresh the current directory

### 5. **Visual Improvements** 🎨
- Icons for different file types and bookmarks
- Clear visual indicators for attached files
- Highlighted search results
- Better spacing and organization

## Keyboard Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+H` | Toggle Hidden | Show/hide hidden files |
| `Ctrl+L` | Edit Path | Focus path input for direct editing |
| `Ctrl+R` | Recent Files | Show/hide recent files panel |
| `Ctrl+B` | Bookmarks | Show/hide bookmarks panel |
| `Ctrl+D` | Bookmark Current | Add/remove current directory bookmark |
| `F5` | Refresh | Refresh current directory listing |
| `Ctrl+F` | Search | Focus search box |
| `1-9` | Quick Jump | Jump to bookmark 1-9 (only when an input is not focused) |
| `Escape` | Cancel | Close file picker |

## Usage Examples

### Basic File Open
```python
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

# Basic usage - will use last directory from this context
file_dialog = EnhancedFileOpen(
    title="Select a file",
    context="my_feature"  # Unique context for persistent settings
)
```

### With File Filters
```python
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters

# Create filters
image_filters = Filters(
    ("Images", lambda p: p.suffix.lower() in (".png", ".jpg", ".jpeg")),
    ("PNG files", lambda p: p.suffix.lower() == ".png"),
    ("All files", lambda p: True)
)

# Use with filters
file_dialog = EnhancedFileOpen(
    title="Select Image",
    filters=image_filters,
    context="image_selection"
)
```

### File Save Dialog
```python
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave

# Save dialog with default filename
save_dialog = EnhancedFileSave(
    title="Save Document",
    default_filename="document.txt",
    context="document_save"
)
```

## Configuration

The enhanced file picker stores its settings in your TOML config file under the `[filepicker]` section:

```toml
[filepicker]
# Recent files for different contexts
recent_character_import = [
    {path = "/home/user/characters/alice.json", name = "alice.json", type = "file", timestamp = "2024-01-15T10:30:00"},
    {path = "/home/user/characters/bob.png", name = "bob.png", type = "file", timestamp = "2024-01-15T10:25:00"}
]

# Last used directories
last_dir_character_import = "/home/user/characters"
last_dir_notes = "/home/user/documents/notes"

# Bookmarks for different contexts
bookmarks_default = [
    {name = "Home", path = "/home/user", icon = "🏠"},
    {name = "Characters", path = "/home/user/characters", icon = "📁", custom = true}
]
```

## Context Names

The following contexts are pre-configured with the migration:

- `character_import` - Character card imports
- `notes` - Note file operations
- `chat_images` - Chat image attachments
- `note_ingest` - Note ingestion
- `prompt_ingest` - Prompt imports
- `character_ingest` - Character ingestion
- `embeddings` - Embeddings file selection
- `llm_models` - LLM model file selection
- `ollama_models` - Ollama-specific models
- `vllm_models` - vLLM model files
- `transformers_models` - Transformers models
- `onnx_models` - ONNX model files
- `mlx_models` - MLX model files
- `file_open` - Generic file open (default)
- `file_save` - Generic file save (default)
- `eval_file` - Generic evaluation file selection
- `eval_task` - Evaluation task file selection
- `eval_dataset` - Evaluation dataset file selection
- `eval_export` - Evaluation result export destination

## Advanced Features

### Custom Bookmarks
Users can add their own bookmarks by navigating to a directory and pressing `Ctrl+D`. Custom bookmarks are marked with `custom: true` in the config.

### Search Functionality
Press `Ctrl+F` to focus the search box. The directory list filters in real-time as you type, matching partial filenames case-insensitively. Press the **Clear** button or `Ctrl+F` again to reset the filter.

### Multiple Contexts
Different features of the app use different contexts, so your character import directory won't interfere with your document directory, etc.

## Migration from Basic File Picker

All existing code has been automatically migrated to use the enhanced file picker. The migration:
1. Replaced imports from `textual_fspicker` with `enhanced_file_picker`
2. Added appropriate context parameters to all file dialogs
3. Maintained backward compatibility with existing code

## Troubleshooting

### Recent Files Not Persisting
- Check that the config file is writable
- Verify the `[filepicker]` section exists in your config
- Check logs for any save errors

### Bookmarks Not Showing
- Press `Ctrl+B` to toggle the bookmarks panel
- Ensure bookmarks are saved in the config
- Try resetting to defaults if corrupted

### Last Directory Not Remembered
- Ensure you're using consistent context names
- Check that the directory still exists
- Verify config file permissions

## Future Enhancements

Planned improvements include:
- Drag and drop support
- Multi-file selection
- File preview pane
- Grid view for images
- Favorites filters
- Path auto-completion
- Advanced sorting options

## Testing

Run the focused test suite to verify the enhanced file picker:

```bash
python3 -m pytest Tests/UI/test_file_picker_filters_callable.py \
    Tests/UI/test_file_picker_bookmarks_lazy.py \
    Tests/UI/test_file_picker_action_tooltips.py \
    Tests/UI/test_enhanced_file_dialog_mount.py -q
```

For interactive exploration, you can still run `Tests/test_enhanced_filepicker.py` as a standalone Textual app.