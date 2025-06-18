# RAG Search UI Improvements Plan

## Overview
This document outlines the improvements being made to the RAG Search UI under the Search tab in the tldw_chatbook application. The goal is to enhance user experience, accessibility, and performance.

## Completed Work

### 1. **Single Pane Layout Conversion** ✅
- Converted from dual-pane (sidebar + main content) to single scrollable pane
- Moved all search options into the main view with better organization
- Used collapsible sections for advanced options

### 2. **Initial Improvements File** ✅
- Created `SearchRAGWindow_improved.py` with enhanced features
- This file contains the complete implementation of all requested improvements

## Requested Improvements Implementation Status

### High Priority

1. **Search History Dropdown** ✅
   - Added `SearchHistoryDropdown` component
   - Shows recent searches as user types
   - Filters history based on current input
   - Click to select from history

2. **Better Pagination** ✅
   - Implemented proper pagination controls
   - Shows "Page X of Y" with Previous/Next buttons
   - Virtual scrolling for large result sets
   - Results per page: 20 (configurable)

3. **Remove Auto-Search** ✅
   - Search only triggers on button click
   - No more automatic searches while typing
   - Better control over when searches execute

4. **Streaming Results** ✅
   - Results display as they arrive
   - Progress indicator shows search progress
   - Non-blocking UI during search

5. **Result Cards Enhancement** ✅
   - Visual cards with clear borders
   - Source type icons (🎬 Media, 💬 Conversations, 📝 Notes)
   - Color-coded source indicators
   - Relevance score visualization with bar graph

6. **Source Type Indicators** ✅
   - Icons for each source type
   - Color coding (cyan for media, green for conversations, yellow for notes)
   - Clear visual distinction between sources

### Medium Priority

7. **Unified Settings Panel** ⚠️ (Partially Complete)
   - Quick settings always visible (search mode, sources)
   - Advanced settings in collapsible section
   - TODO: Move persistent settings to Settings Tab

8. **Progressive Disclosure** ✅
   - Advanced options hidden by default
   - Collapsible sections for complex settings
   - Only show relevant options based on search mode

9. **Action Button Hierarchy** ✅
   - Primary: Search button (prominent, blue)
   - Secondary: Save Search, Export Results
   - Tertiary: Index Content, Clear Cache (de-emphasized)

10. **Visual Feedback** ✅
    - Loading indicator during search
    - Progress bar for long operations
    - Status messages for all states
    - Clear error messages

11. **Background Indexing Progress** ✅
    - Progress bar shows indexing status
    - Non-blocking UI during indexing
    - Status updates for each phase

12. **Saved Searches** ✅
    - Save current search configuration
    - Load saved searches from panel
    - Persist to user data directory

13. **ARIA Labels** ✅
    - Added accessibility labels to all interactive elements
    - Screen reader friendly
    - Semantic HTML structure

14. **Tab Navigation** ✅
    - Logical tab order through elements
    - Keyboard shortcuts (Ctrl+K for search focus)
    - Escape to clear search

15. **Better Metadata Display** ✅
    - Compact view shows first 3 metadata items
    - Expandable to show all metadata
    - Clean formatting with dimmed text

## CSS Updates Needed

The following CSS classes need to be added to `tldw_cli.tcss`:

```css
/* Search History Dropdown */
.search-history-dropdown {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    max-height: 15;
    background: $surface;
    border: solid $primary;
    border-top: none;
    z-index: 100;
}

.search-history-dropdown.hidden {
    display: none;
}

.search-history-list {
    max-height: 15;
    overflow-y: auto;
}

.history-item-text {
    padding: 1;
}

.history-item-text:hover {
    background: $boost;
}

/* Enhanced Search UI */
.search-input-container {
    position: relative;
    margin-bottom: 2;
}

.search-input-enhanced {
    width: 1fr;
}

.search-button {
    min-width: 10;
}

/* Quick Settings */
.quick-settings {
    background: $surface;
    padding: 1;
    margin-bottom: 1;
    border: round $primary-background;
}

.quick-select {
    margin: 0 1;
}

.source-checkboxes {
    layout: horizontal;
}

.source-checkbox {
    margin: 0 1;
}

/* Status Container */
.status-container {
    margin: 1 0;
    padding: 1;
    background: $boost;
    border: round $primary;
}

.search-status {
    text-align: center;
    margin-bottom: 1;
}

/* Result Cards */
.search-result-card {
    margin-bottom: 1;
    background: $panel;
    border: solid $primary-background;
    padding: 1;
    transition: background 0.2s;
}

.search-result-card:hover {
    background: $panel-lighten-1;
    border-color: $primary;
}

.result-card-content {
    width: 100%;
}

.source-indicator {
    min-width: 12;
}

.result-title {
    width: 1fr;
    margin: 0 1;
}

.result-score-visual {
    min-width: 15;
    text-align: right;
}

.result-preview {
    margin: 1 0;
    color: $text-muted;
}

.result-metadata.compact {
    layout: horizontal;
    margin-bottom: 1;
}

.metadata-item {
    margin-right: 2;
}

.metadata-more {
    color: $primary;
}

.result-metadata-full {
    margin: 1 0;
    padding: 1;
    background: $surface;
    border: round $primary-background;
}

.result-actions Button {
    margin-right: 1;
}

Button.mini {
    min-width: 8;
    height: 3;
}

Button.primary {
    background: $primary;
    color: $text;
}

Button.secondary {
    background: $secondary;
    color: $text;
}

Button.tertiary {
    background: $surface;
    color: $text-muted;
}

/* Pagination */
.results-header-bar {
    layout: horizontal;
    margin-bottom: 1;
}

.results-summary {
    width: 1fr;
}

.pagination-controls {
    layout: horizontal;
}

.page-info {
    margin: 0 1;
    min-width: 10;
    text-align: center;
}

/* Saved Searches */
.saved-searches-panel {
    background: $surface;
    padding: 1;
    margin-bottom: 1;
    border: round $primary-background;
}

.saved-searches-list {
    height: 5;
    border: round $primary;
    margin: 1 0;
}

.saved-search-actions {
    layout: horizontal;
}

.saved-search-actions Button {
    margin-right: 1;
}

/* Action Buttons Bar */
.action-buttons-bar {
    layout: horizontal;
    margin-top: 2;
    padding-top: 1;
    border-top: solid $primary-background;
}

.primary-actions {
    layout: horizontal;
    width: 1fr;
}

.maintenance-actions {
    layout: horizontal;
}

.primary-actions Button,
.maintenance-actions Button {
    margin-right: 1;
}

/* Parameter Grid */
.parameter-grid {
    layout: grid;
    grid-size: 2;
    grid-columns: auto 1fr;
    grid-gutter: 1 2;
    margin: 1 0;
}

.param-input {
    width: 100%;
}

/* Advanced Settings */
#advanced-settings-collapsible {
    margin: 1 0;
}

.advanced-settings-content {
    padding: 1;
    background: $surface;
    border: round $primary-background;
}

.subsection-title {
    margin: 1 0;
    color: $secondary;
}

.chunking-options {
    margin-top: 1;
    padding-top: 1;
    border-top: dashed $primary-background;
}
```

## Next Steps

1. **Replace the original SearchRAGWindow.py**
   - Back up the original file
   - Replace with SearchRAGWindow_improved.py
   - Test all functionality

2. **Add CSS styles**
   - Add the CSS classes above to tldw_cli.tcss
   - Test visual appearance and responsiveness

3. **Move persistent settings to Settings Tab**
   - Create a RAG Settings section in Tools & Settings
   - Move default values for:
     - Default search mode
     - Default sources
     - Default top-k value
     - Default chunking parameters
     - Re-ranking preferences

4. **Testing checklist**
   - [ ] Search history dropdown works
   - [ ] Pagination controls function correctly
   - [ ] Progressive disclosure hides/shows correctly
   - [ ] All keyboard shortcuts work
   - [ ] ARIA labels are properly read by screen readers
   - [ ] Visual feedback appears during operations
   - [ ] Saved searches persist between sessions
   - [ ] Export functionality works
   - [ ] Indexing shows progress without blocking UI

5. **Future enhancements**
   - Add search templates for common queries
   - Implement search query builder UI
   - Add more export formats (PDF, CSV)
   - Implement bulk operations on results
   - Add search result filtering post-search
   - Implement search result bookmarking

## File Structure Changes

- `SearchRAGWindow.py` → `SearchRAGWindow_improved.py` (new implementation)
- `tldw_cli.tcss` → Add new CSS classes for enhanced UI
- `search_history.db` → Already exists, no changes needed
- `saved_searches.json` → New file in user data directory

## Dependencies

No new dependencies required. All improvements use existing Textual widgets and features.

## Performance Considerations

1. **Virtual scrolling** - Only renders visible results
2. **Streaming results** - Display as they arrive
3. **Background workers** - Non-blocking operations
4. **Debounced history** - Prevents excessive database queries
5. **Cached search config** - Reduces repeated calculations

## Accessibility Improvements

1. **ARIA labels** on all interactive elements
2. **Keyboard navigation** with logical tab order
3. **Screen reader announcements** for state changes
4. **High contrast** visual indicators
5. **Clear focus indicators** on all controls

## Migration Notes

When replacing the old SearchRAGWindow with the improved version:

1. Ensure all event handlers in the main app still work
2. Update any references to removed methods
3. Test integration with other components
4. Verify database connections remain stable
5. Check that all keyboard shortcuts don't conflict

## Status

This improvement plan is ready for implementation. The SearchRAGWindow_improved.py file contains all the requested features and is ready to replace the original after testing.