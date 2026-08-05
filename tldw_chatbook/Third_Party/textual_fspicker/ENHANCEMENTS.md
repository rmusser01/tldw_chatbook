# File Picker Enhancements

This document describes the enhancements made to the textual-fspicker module to improve user experience and functionality.

## New Features

### 1. Enhanced Keyboard Shortcuts
- **Ctrl+H** - Toggle hidden files (in addition to existing '.')
- **Ctrl+L** - Toggle path input field for direct path entry (shows/hides a text field where you can type absolute or relative paths)
- **Ctrl+R** - Toggle recent locations panel
- **Ctrl+F** - Toggle search mode and focus search input
- **F5** - Refresh current directory
- **Ctrl+D** - Bookmark current directory (placeholder for future implementation)

### 2. Recent Locations Panel
- Shows recently accessed files and directories
- Toggle visibility with Ctrl+R
- Click to quickly navigate to recent locations
- Placeholder for persistent storage (needs implementation)

### 3. Breadcrumb Navigation
- Visual path display with clickable components
- Click any path segment to navigate directly
- Better understanding of current location in filesystem hierarchy

### 4. Search Within Directory
- Real-time search filtering of directory contents
- Toggle with Ctrl+F
- Filters files and folders by name
- Clear button to reset search

### 5. Direct Path Input
- Toggle with Ctrl+L to show/hide a path input field
- Enter absolute paths (e.g., `/home/user/documents`) or relative paths (e.g., `../folder`)
- Supports home directory expansion (e.g., `~/Documents`)
- Press Enter or click "Go" to navigate
- Automatically navigates to parent directory if a file path is entered
- Shows error notification if path doesn't exist

### 5. Improved Visual Feedback
- Notifications for keyboard actions
- Better error messages
- Search active indicator

## Implementation Details

### Modified Files

1. **base_dialog.py**
   - Added new keyboard bindings
   - Added reactive properties for UI state
   - Implemented breadcrumb navigation
   - Added recent locations panel
   - Added search container
   - Implemented all keyboard action handlers

2. **parts/directory_navigation.py**
   - Added `search_filter` reactive variable
   - Modified `hide()` method to respect search filter
   - Added `_watch_search_filter()` method
   - Modified `_repopulate_display()` to handle search state

### Code Structure

The enhancements maintain backward compatibility while adding new optional features. The UI components are hidden by default and can be toggled via keyboard shortcuts.

### CSS Additions

New CSS rules were added for:
- Breadcrumb navigation styling
- Recent locations panel
- Search container
- Visibility toggles

## Usage Example

```python
from textual_fspicker import FileOpen

# Use the enhanced file picker
file_dialog = FileOpen(
    title="Select a file",
    filters=Filters(
        ("Python Files", "*.py"),
        ("All Files", "*.*")
    )
)

# All new features are available via keyboard shortcuts
```

## Future Improvements

1. **Persistent Storage for Recent Locations**
   - Save to user config directory
   - Load on startup
   - Configurable max items

2. **Bookmarks System**
   - Save frequently used directories
   - Manage bookmarks UI
   - Persistent storage

3. **Advanced Search**
   - Regex support
   - Case sensitivity toggle
   - Search in subdirectories option

4. **File Preview**
   - Preview pane for text files
   - Image thumbnails
   - File metadata display

## Testing

The enhancements have been tested with:
- Various directory structures
- Hidden files toggle
- Search functionality
- Breadcrumb navigation
- Keyboard shortcuts

### 6. Usable Filename Input for Save Dialogs (task-1479)

Live UAT of a keyboard-only export flow (Evals results-grid export, at a
235x52 terminal) found the `FileSave`/`FileOpen` input bar unusable:

- The filename `Input`'s rendered width could collapse to a handful of
  columns because the file-type filter `Select` next to it was set to
  `width: 1fr` -- flexible, not fixed, so it competed with the Input for
  space and, once the app's own `Select { width: 100%; }` bundle rule
  (`components/_dialogs.tcss` documents this in full) wins the CSS-origin
  battle against this package's `DEFAULT_CSS` regardless of source order,
  the Select claimed the entire row. The filter `Select` now gets a fixed
  `width: 24` in `file_dialog.py`'s `DEFAULT_CSS`, and the app bundle pins
  the same width with a selector specific enough to win there too.
- `FileSave` (not `FileOpen`) now focuses its filename `Input` on mount
  instead of the directory listing (`FileSystemPickerScreen._focus_initial_widget`,
  overridden in `file_save.py`), so a keyboard user can press Enter right
  away to confirm the seeded default filename, instead of Enter activating
  the highlighted directory row (usually `..`).
- `Select` posts its own `Changed` message as a side effect of mounting
  with an explicit initial `value=` -- not from a user picking a filter.
  `BaseFileDialog._change_filter` used to unconditionally move focus back
  to the directory listing on every `Select.Changed`, including that first,
  synthetic one, which raced with (and usually beat) the new mount-time
  focus above. It now ignores focus-stealing for the first event only,
  tracked via `BaseFileDialog._filter_select_changed_by_user`.
- `_select_file`/`_confirm_file` (`file_dialog.py`) read the filename back
  via `self.query_one(Input)`, which is ambiguous: the screen also carries
  a hidden `#path-input` (Ctrl+L) and `#search-input` (Ctrl+F), both
  mounted before the input bar's own filename `Input`. An unscoped
  `query_one(Input)` silently grabbed one of those instead, so even once
  focus and width were fixed, pressing Enter on the (correctly focused,
  correctly filled) filename field read back an empty, unrelated Input and
  rejected with "A file must be chosen". Both call sites now query through
  `InputBar` first (`self.query_one(InputBar).query_one(Input)`), which is
  unambiguous since InputBar's own children are exactly one `Input` (the
  filename) and, if filters were supplied, one `Select`.

None of this touches `FileOpen`'s own default focus behaviour (still the
directory listing) or the separate `EnhancedFileDialog` picker in
`Widgets/enhanced_file_picker.py`, which composes its own, differently-`id`d
Input/Select and is unaffected.

## Contributing Upstream

These enhancements are designed to be contributed back to the original textual-fspicker project. They:
- Maintain backward compatibility
- Follow the existing code style
- Add optional features that don't change default behavior
- Include proper documentation

To contribute:
1. Fork the original repository
2. Apply these changes
3. Add tests for new features
4. Submit a pull request with this documentation