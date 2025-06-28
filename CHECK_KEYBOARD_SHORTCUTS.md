# File Picker Keyboard Shortcuts Check

The enhanced file picker should support these keyboard shortcuts:

1. **Ctrl+H** - Toggle hidden files
2. **Ctrl+F** - Focus search input
3. **Ctrl+L** - Edit path directly (from base implementation)
4. **Ctrl+R** - Show recent files (from base implementation)
5. **F5** - Refresh directory
6. **Escape** - Cancel

## What's Actually Implemented

Looking at the code structure:
- The keyboard shortcuts are defined in `EnhancedFileDialog` BINDINGS
- But `EnhancedFileOpen` inherits from `FileOpen`, not `EnhancedFileDialog`
- So the bindings from `EnhancedFileDialog` are not being used

## The Real Issue

The problem is that:
1. `EnhancedFileDialog` has all the enhanced features but is not being used
2. `EnhancedFileOpen` and `EnhancedFileSave` inherit from the base classes
3. They override `compose()` to add the toolbar, but the toolbar might not be showing properly

## Current Status

The toolbar functionality IS implemented in the underlying `DirectoryNavigation` widget:
- `show_hidden` property exists and works
- `sort_by` property exists and works  
- `search_filter` property exists and works
- `toggle_hidden()` method exists

So even without the visible toolbar, you can:
- Press **Ctrl+H** to toggle hidden files (if the binding is active)
- The sorting and search would work if we could access the controls

## Next Steps

To verify if the features work, try:
1. Opening the import notes dialog
2. Pressing Ctrl+H to see if hidden files toggle
3. Check if any toolbar appears at the top of the dialog