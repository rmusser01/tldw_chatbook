# Multiple Analyses per Media Item - Feature Summary

## Overview
Added support for storing and navigating multiple analyses for a single media item using the existing DocumentVersions table in the database schema.

## Database Schema
The existing schema already supports multiple analyses through the `DocumentVersions` table:
- Each media item can have multiple document versions
- Each version has an `analysis_content` field
- Versions are tracked with `version_number` and timestamps

## UI Changes

### 1. Navigation Controls
Added analysis navigation controls in `media_viewer_panel.py`:
- Previous button (`◀`) - Navigate to older analyses
- Indicator showing "Analysis X/Y" - Current position and total count
- Next button (`▶`) - Navigate to newer analyses

### 2. Reactive Properties
Added to track multiple analyses:
```python
all_analyses: reactive[List[Dict[str, Any]]] = reactive([])
current_analysis_index: reactive[int] = reactive(0)
```

### 3. CSS Styling
Added styling for the navigation controls:
```css
MediaViewerPanel .analysis-navigation {
    layout: horizontal;
    height: 3;
    margin-top: 1;
    margin-bottom: 1;
    align: center middle;
}
```

## Functionality

### 1. Loading Analyses
When a media item is loaded:
- `load_all_analyses()` fetches all document versions with analysis content
- Analyses are sorted by version number (newest first)
- The most recent analysis is displayed by default

### 2. Navigation
Users can navigate between analyses using:
- Previous/Next buttons to move through the analysis history
- Indicator shows current position (e.g., "Analysis 1/3")
- Buttons are disabled appropriately at boundaries

### 3. Saving New Analyses
When a new analysis is generated and saved:
- Creates a new document version
- Reloads the analyses list to include the new entry
- Updates navigation to show the new total count

### 4. Integration with Existing Features
- Save button creates a new analysis version
- Overwrite updates the current version
- Edit functionality works with the currently displayed analysis

## Code Changes

### Files Modified:
1. **`tldw_chatbook/Widgets/Media/media_viewer_panel.py`**:
   - Added reactive properties for multiple analyses
   - Added navigation UI components
   - Implemented `load_all_analyses()` method
   - Added navigation button handlers
   - Updated CSS for navigation controls

2. **`tldw_chatbook/UI/MediaWindow_v2.py`**:
   - Updated save/overwrite handlers to reload analyses after changes
   - Maintains compatibility with existing analysis generation

## User Experience
1. When viewing a media item with multiple analyses:
   - See "Analysis 1/3" indicating 3 total analyses
   - Use arrow buttons to browse through different analyses
   - Most recent analysis shown first

2. When generating a new analysis:
   - New analysis is saved as a new version
   - Navigation updates to show the new total
   - Can still browse older analyses

3. Navigation is intuitive:
   - Disabled buttons at boundaries prevent errors
   - Clear indicator of current position
   - Analyses sorted chronologically (newest first)