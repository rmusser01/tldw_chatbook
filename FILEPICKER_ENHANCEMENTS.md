# File Picker Enhancements

## Overview
The enhanced file picker now includes the following features:

### 1. Show/Hide Hidden Files
- **Keyboard shortcut**: `Ctrl+H`
- **UI Control**: Click the "Hidden Files" button in the toolbar
- Hidden files (those starting with `.`) can now be toggled on/off

### 2. File Sorting Options
The file picker now supports multiple sorting methods:
- **Name** (default): Alphabetical order
- **Size**: Sort by file size
- **Type**: Sort by file extension
- **Date**: Sort by modification time (newest first)

Use the "Sort by" dropdown in the toolbar to change sorting.

### 3. Search Functionality
- Real-time search filtering as you type
- Case-insensitive search
- Searches in filenames only
- Clear button to reset search

### 4. Additional Features
- **Recent Locations** (`Ctrl+R`): Access recently used files/folders
- **Breadcrumb Navigation**: Click on any parent directory in the path
- **Direct Path Input** (`Ctrl+L`): Type/edit the path directly
- **Refresh** (`F5`): Refresh the current directory listing

## Implementation Details

### Modified Files:
1. `/tldw_chatbook/Third_Party/textual_fspicker/parts/directory_navigation.py`
   - Added `sort_by` reactive variable for sorting method
   - Added `search_filter` reactive variable for search text
   - Enhanced `_sort()` method to support multiple sorting options
   - Modified `hide()` method to include search filtering

2. `/tldw_chatbook/Widgets/enhanced_file_picker.py`
   - Created `FilePickerToolbar` widget with all controls
   - Added event handlers for sorting, searching, and hidden files toggle
   - Connected UI controls to the underlying DirectoryNavigation widget

### Testing
Run the test script to see all features in action:
```bash
python test_enhanced_filepicker.py
```

## Usage Example
```python
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters

# Create filters
filters = Filters(
    ("Python Files", "*.py"),
    ("All Files", "*.*"),
)

# Show dialog
file_path = await app.push_screen(
    EnhancedFileOpen(
        location=Path.home(),
        title="Select a File",
        filters=filters
    )
)
```