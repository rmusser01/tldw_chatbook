# MediaWindow CSS Grid Conversion Plan

## Overview
Convert the MediaWindow component from using positional CSS (dock, width percentages) to modern CSS Grid layout for better maintainability and flexibility.

## Current Structure Analysis

### Main Layout
```
MediaWindow (Container) - layout: horizontal
├── media-nav-pane (VerticalScroll) - dock: left, width: 25%
│   ├── Static ("Media Types")
│   └── Button[] (media type buttons)
└── media-content-pane (Container) - width: 1fr
    ├── media-sidebar-toggle-button - dock: top
    └── media-view-area[] (Horizontal containers)
        ├── media-content-left-pane (Container) - width: 35%
        │   ├── Label (title)
        │   ├── Input (search)
        │   ├── Input (keyword filter)
        │   ├── Checkbox (show deleted)
        │   ├── ListView (media items)
        │   └── Horizontal (pagination)
        └── media-content-right-pane (VerticalScroll) - width: 65%
            └── MediaDetailsWidget
```

### Current CSS Properties
1. **Main container**: `layout: horizontal`
2. **Nav pane**: `dock: left`, `width: auto`, `min-width: 25`, `max-width: 60`
3. **Content pane**: `width: 1fr`, `layout: vertical`
4. **Toggle button**: `dock: top`
5. **View areas**: `layout: horizontal`
6. **Left/Right panes**: `width: 35%` / `width: 65%`

## Grid Conversion Strategy

### IMPORTANT: Textual Grid System Notes
- Textual does NOT support `grid-column` or `grid-row` properties
- Widgets are placed in cells automatically in order of composition
- Use `column-span` and `row-span` to make widgets span multiple cells
- The order of widgets in compose() determines their grid placement

### 1. Main MediaWindow Grid
```css
#media-window {
    layout: grid;
    grid-size: 2 1;  /* 2 columns, 1 row */
    grid-columns: 25% 1fr;
    height: 100%;
    width: 100%;
}

#media-window.sidebar-collapsed {
    grid-columns: 0 1fr;
}
```

### 2. Navigation Pane
```css
.media-nav-pane {
    /* First child - automatically goes in first cell */
    /* Remove dock: left */
    /* Keep other properties */
}
```

### 3. Content Pane
```css
.media-content-pane {
    /* Second child - automatically goes in second cell */
    /* Remove width: 1fr */
    height: 100%;
}
```

### 4. View Area Grid
```css
.media-view-area > Grid {
    layout: grid;
    grid-size: 2 1;  /* 2 columns, 1 row */
    grid-columns: 35% 65%;
    grid-gutter: 1;
    height: 100%;
    width: 100%;
}
```

### 5. Left/Right Panes
```css
.media-content-left-pane {
    /* First child - automatically goes in first cell */
    /* Remove width: 35% */
}

.media-content-right-pane {
    /* Second child - automatically goes in second cell */
    /* Remove width: 65% */
}
```

## Python Code Changes

### MediaWindow.py Modifications

1. **Import Grid**:
   ```python
   from textual.containers import Container, VerticalScroll, Horizontal, Vertical, Grid
   ```

2. **Update compose() method**:
   - Note: MediaWindow itself is already a Container, so we don't need to wrap it in Grid
   - The MediaWindow class should have `id="media-window"` set in its initialization
   - Only update the internal Horizontal containers to Grid
   
   ```python
   def compose(self) -> ComposeResult:
       # Navigation pane stays the same
       with VerticalScroll(classes="media-nav-pane", id="media-nav-pane"):
           # ... existing nav content ...
       
       # Content pane stays the same
       with Container(classes="media-content-pane", id="media-content-pane"):
           # ... toggle button ...
           
           # For each media view - change Horizontal to Grid
           with Grid(id=view_id, classes="media-view-area"):
               # Left pane
               with Container(classes="media-content-left-pane"):
                   # ... existing left content ...
               
               # Right pane  
               with VerticalScroll(classes="media-content-right-pane"):
                   # ... existing right content ...
   ```

3. **Update MediaWindow initialization**:
   ```python
   # NO CHANGES NEEDED - id is already set by app.py when creating the widget
   # MediaWindow is instantiated with id="media-window" in app.py
   ```

4. **Update sidebar collapse logic**:
   ```python
   def watch_media_sidebar_collapsed(self, collapsed: bool) -> None:
       try:
           # Add class to self (MediaWindow) for grid column adjustment
           self.set_class(collapsed, "sidebar-collapsed")
           
           # Keep existing logic for nav pane and toggle button
           nav_pane = self.query_one("#media-nav-pane")
           toggle_button = self.query_one("#media-sidebar-toggle-button")
           nav_pane.set_class(collapsed, "collapsed")
           toggle_button.set_class(collapsed, "collapsed")
       except QueryError as e:
           self.log.warning(f"UI component not found during media sidebar collapse: {e}")
   ```

## Implementation Steps

### Phase 1: CSS Updates
1. Update `_media.tcss` with Grid layout rules
2. Remove positional CSS (dock, width percentages)
3. Add grid-specific classes and rules
4. Test CSS changes with existing Python code

### Phase 2: Python Updates
1. Import Grid container
2. Update compose() method structure
3. Replace Horizontal containers with Grid where needed
4. Update reactive watchers for grid-aware behavior
5. Test functionality

### Phase 3: Testing & Refinement
1. Test sidebar collapse/expand
2. Verify media type switching
3. Check responsive behavior
4. Test all media views (collections, multi-item, etc.)
5. Verify scroll behavior is maintained

## Benefits of Grid Layout

1. **Cleaner Structure**: No mixing of positional and layout properties
2. **Better Responsiveness**: Grid naturally handles size changes
3. **Easier Maintenance**: Clear parent-child relationships
4. **Modern Approach**: Aligns with current CSS best practices
5. **Flexibility**: Easy to add new columns or rearrange layout

## Potential Issues & Solutions

### Issue 1: Sidebar Toggle Button Positioning
- Current: Uses `dock: top`
- Solution: Keep dock for now, or use absolute positioning within grid cell

### Issue 2: Dynamic View Switching
- Current: Sets display: none/block
- Solution: No change needed, this works with Grid

### Issue 3: Collapsed Sidebar Animation
- Current: Sets width to 0
- Solution: Use grid-columns transition or display: none

### Issue 4: Special Views (Collections, Multi-Item)
- Current: Have different internal structure
- Solution: Apply Grid only to standard media views

## Rollback Plan

1. Keep backup of original files
2. Test incrementally
3. Use version control for easy reversion
4. Document any unexpected behaviors

## Success Criteria

1. ✓ All media views display correctly
2. ✓ Sidebar collapse/expand works
3. ✓ No visual regressions
4. ✓ Performance is maintained or improved
5. ✓ Code is cleaner and more maintainable