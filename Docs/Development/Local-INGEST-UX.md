# Local Media Ingestion UX Design Documentation

## Overview
This document tracks all design decisions, reasoning, and implementation details for the tabbed "Ingest Media (Local)" interface in tldw_chatbook.

## Design Philosophy
- **Consistency**: Follow existing UI patterns from the application
- **Clarity**: Each media type gets its own dedicated tab with relevant options
- **Flexibility**: Support various media formats while maintaining a unified interface
- **Progressive Disclosure**: Show essential options by default, advanced options in collapsibles

## Decision Log

### 2025-01-30: Initial Design Decisions

#### Tab Structure Decision
**Decision**: Create 8 tabs for different media types: Video, Audio, Document, PDF, Ebook, Web Article, XML, Plaintext

**Reasoning**:
- Each media type has unique processing requirements and options
- Separate tabs prevent UI clutter and confusion
- Users can quickly navigate to their specific media type
- Allows for future expansion of media-specific features

#### Tab Order Decision
**Decision**: Video → Audio → Document → PDF → Ebook → Web Article → XML → Plaintext

**Reasoning**:
1. **Video/Audio first**: These are the most complex media types with transcription needs
2. **Document/PDF next**: Common business/academic use cases
3. **Ebook follows**: Similar to documents but with specific metadata needs
4. **Web Article**: Unique as it uses URLs instead of files
5. **XML/Plaintext last**: More technical/specialized use cases

#### UI Pattern Decision
**Decision**: Each tab follows a consistent 4-section structure:
1. File Selection Section
2. Processing Options Section
3. Metadata Section
4. Action Section

**Reasoning**:
- Predictable layout reduces cognitive load
- Users learn the pattern once and can apply it across all tabs
- Logical flow from selection → configuration → metadata → action
- Matches the existing pattern in the tldw API forms

#### State Management Decision
**Decision**: Extend the existing `selected_local_files` dictionary to track files per media type

**Reasoning**:
- Reuses existing infrastructure
- Maintains separation between different media types
- Allows users to prepare multiple media types before processing
- Consistent with how the tldw API forms handle state

#### Method Organization Decision
**Decision**: Create separate `compose_local_[media_type]_tab()` methods instead of a generic method

**Reasoning**:
- Each media type has unique fields and options
- Easier to maintain and modify individual tabs
- Better code organization and readability
- Allows for media-specific logic without complex conditionals

## UI/UX Patterns

### File Selection Pattern
- "Select Files" button opens file picker with appropriate filters
- Selected files displayed in a ListView
- "Clear Selection" button to reset
- File count indicator for user feedback

### Options Organization Pattern
- Common options (chunking, analysis) in main view
- Media-specific options prominently displayed
- Advanced options in collapsibles
- Sensible defaults pre-populated

### Visual Feedback Pattern
- Loading indicators during processing
- Status messages in dedicated area
- Success/error notifications
- Progress indication where applicable

## Integration Points

### File Picker Integration
- Reuse existing `FileOpen` dialog from textual_fspicker
- Media-specific file filters per tab
- Multiple file selection support

### Database Integration
- Local processing results stored in Media DB
- Metadata properly indexed
- Chunk storage for large files

### Processing Pipeline Integration
- Hook into existing media processing functions
- Support for background processing via workers
- Progress reporting through existing mechanisms

## Future Enhancement Considerations

### Planned Enhancements
1. Drag-and-drop file support
2. Batch processing with queue management
3. Processing profiles/presets
4. Preview functionality for selected files
5. Integration with external tools (ffmpeg, pandoc, etc.)

### Extensibility Points
- Easy to add new media types as new tabs
- Options sections can be extended without breaking layout
- Processing pipeline can be enhanced per media type
- Metadata fields can be customized per media type

## Implementation Notes

### CSS Considerations
- Using existing ingest window styles as base
- Adding specific styles for tabbed content
- Maintaining visual consistency with rest of application
- Ensuring responsive behavior

### Event Handling
- Each tab's buttons get unique IDs following pattern: `ingest-local-[media_type]-[action]`
- Event handlers follow existing naming conventions
- State updates trigger appropriate UI refreshes

### Error Handling
- File validation before processing
- Clear error messages for user
- Graceful fallbacks for unsupported formats
- Recovery options where applicable

## Implementation Progress

### 2025-01-30: Initial Implementation

#### Step 1: Import Updates ✓
- Added `TabbedContent` and `TabPane` to imports in `Ingest_Window.py`
- Decision: Keep imports organized with other textual.widgets imports

#### Step 2: Tab Composition Methods ✓
- Created 8 separate `compose_local_[media_type]_tab()` methods
- Each method returns a complete tab interface with:
  - File selection section
  - Processing options section
  - Metadata section
  - Action section

**Key Decisions Made:**
1. **Consistent ID Pattern**: All element IDs follow pattern `ingest-local-[media_type]-[element]`
   - Reasoning: Makes event handling straightforward and predictable
2. **Section Classes**: Used consistent class names across all tabs
   - Reasoning: Allows unified CSS styling while maintaining flexibility
3. **Default Values**: Populated sensible defaults for common fields
   - Reasoning: Reduces user friction, allows quick processing
4. **Progressive Disclosure**: Advanced options in collapsibles
   - Reasoning: Clean interface for basic users, power features for advanced users

#### Step 3: Replace Placeholder ✓
- Replaced static placeholder with `TabbedContent` widget
- Each tab uses `TabPane` with descriptive labels
- Used `yield from` pattern to compose tab content
- Decision: Added title "Local Media Ingestion" above tabs for clarity

#### Step 4: CSS Styling ✓
- Added comprehensive styles to `_ingest.tcss`
- Created consistent visual hierarchy with borders and backgrounds
- Ensured responsive layout with proper overflow handling

**CSS Design Decisions:**
1. **Section Styling**: Each section has rounded borders and surface background
   - Reasoning: Visual separation helps users understand different option groups
2. **Tab Content Scrolling**: Made tab content scrollable
   - Reasoning: Accommodates varying amounts of options per media type
3. **Form Layout**: Two-column layout for metadata fields
   - Reasoning: Efficient use of horizontal space
4. **Consistent Spacing**: Standardized margins and padding
   - Reasoning: Professional appearance, reduces visual noise

### Next Steps

#### Event Handler Implementation
Need to create handlers for:
1. File selection buttons (8 media types) ✓
2. Clear selection buttons (8 media types) ✓
3. Process buttons (8 media types) - TODO
4. Special handling for web article URLs (not file-based) ✓

#### State Management Updates
1. Extend `selected_local_files` to track files for new media types ✓
2. Add URL tracking for web article tab - TODO (using TextArea instead)
3. Consider processing state per tab - TODO

#### Integration Points
1. Connect to existing file picker infrastructure ✓
2. Hook into media processing pipelines - TODO
3. Implement progress tracking - TODO
4. Add error handling and notifications - TODO

### Step 5: Button Handling Implementation ✓

#### File Selection Handling
- Added handling for `ingest-local-[media_type]-select-files` buttons
- Each media type opens file picker with appropriate filters
- Media type prefixed with "local_" to distinguish from API selections

**Key Decisions:**
1. **Media Type Prefix**: Used `local_` prefix for local media file tracking
   - Reasoning: Prevents conflicts with tldw API file selections
   - Allows reuse of existing `selected_local_files` dictionary
2. **File Filters**: Created media-specific file filters
   - Reasoning: Better user experience, prevents selecting wrong file types
   - Includes "All Files" option for flexibility

#### Clear Selection Handling
- Added handling for `ingest-local-[media_type]-clear-files` buttons
- Special handling for web article URL clearing
- Updates ListView UI after clearing

#### File Picker Integration
- Created `_get_file_filters_for_media_type()` helper method
- Returns appropriate `Filters` object for each media type
- Comprehensive file extension coverage per media type

**File Extension Mappings:**
- **Video**: .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm, .m4v, .mpg, .mpeg
- **Audio**: .mp3, .wav, .flac, .aac, .ogg, .wma, .m4a, .opus, .aiff
- **Document**: .docx, .doc, .odt, .rtf, .txt
- **PDF**: .pdf
- **Ebook**: .epub, .mobi, .azw, .azw3, .fb2
- **XML**: .xml, .xsd, .xsl
- **Plaintext**: .txt, .md, .text, .log, .csv

#### ListView Update Logic
- Modified `handle_file_picker_dismissed` to detect local vs API files
- Uses different ListView ID patterns:
  - Local: `#ingest-local-{media_type}-files-list`
  - API: `#tldw-api-selected-local-files-list-{media_type}`
- Maintains backward compatibility with existing API functionality

### Remaining Implementation Tasks

#### Process Button Handlers
Need to implement handlers for `ingest-local-[media_type]-process` buttons:
1. Collect form data from UI elements
2. Validate selected files/URLs
3. Create processing request
4. Handle progress updates
5. Display results/errors

#### Integration with Processing Pipeline
1. Connect to existing media processing functions
2. Map UI options to processing parameters
3. Handle different processing flows per media type
4. Implement error recovery

### Technical Notes

#### Tab Implementation Pattern
```python
with TabPane("Label", id="unique-id"):
    yield from self.compose_method()
```
This pattern allows clean separation of tab content composition.

#### ID Naming Convention
- Navigation: `ingest-nav-[view]`
- Local media elements: `ingest-local-[media_type]-[element]`
- API elements: `tldw-api-[element]-[media_type]`

This ensures no ID conflicts between different ingestion methods.

#### CSS Class Strategy
- Container classes: `ingest-[section]-section`
- Common elements: `ingest-[element]`
- Tab-specific: `ingest-media-tab-content`

Allows both global and specific styling control.

## Implementation Summary

### Completed Features (2025-01-30)

1. **Tabbed Interface Structure** ✓
   - Replaced placeholder with `TabbedContent` widget
   - 8 tabs for different media types
   - Clean, organized layout

2. **UI Components per Tab** ✓
   - File selection section (except Web Article which uses URL input)
   - Processing options section with media-specific settings
   - Metadata section for overrides
   - Action section with process button and status area

3. **File Selection Functionality** ✓
   - Select files button with media-specific filters
   - Clear selection button
   - Selected files display in ListView
   - Web article URL input handling

4. **CSS Styling** ✓
   - Comprehensive styles for all sections
   - Consistent visual hierarchy
   - Responsive layout

5. **State Management** ✓
   - Extended `selected_local_files` dictionary
   - Proper tracking with "local_" prefix
   - UI updates on file selection/clearing

### Ready for Next Phase

The tabbed "Ingest Media (Local)" interface is now ready for:
1. Process button handler implementation
2. Integration with media processing pipelines
3. Progress tracking and error handling
4. Testing with actual media files

### Key Achievement

Successfully transformed a placeholder "Coming Soon" message into a fully functional, professionally styled tabbed interface with 8 different media type tabs, each with appropriate options and file selection capabilities. The implementation maintains consistency with the existing application patterns while providing a clean, intuitive user experience.

## Troubleshooting History

### Issue: tldw API tabs showing nothing (2025-01-30)
**Problem**: User reported "None of the tabs in the `Ingest Content via tldw API` display anything"

**Root Causes Identified**:
1. CSS display issues - Textual only supports "block" or "none" for display property, not "flex"
2. Duplicate compose_tldw_api_form method in Ingest_Window.py
3. Widget inheritance - IngestTldwApiTabbedWindow needed to inherit from Vertical not Container
4. CSS height specification - Changed from "1fr" to "100%"

**Solutions Applied**:
1. Removed duplicate compose_tldw_api_form method from Ingest_Window.py (lines 97-276)
2. Changed IngestTldwApiTabbedWindow inheritance from Container to Vertical
3. Fixed CSS to use only "block" display property
4. Updated CSS height from "1fr" to "100%" for proper sizing
5. Added explicit display:block to #tldw-api-tabbed-window
6. Fixed .hidden class to properly use display:none
7. Rebuilt modular CSS to include all changes

**Key Learning**: Textual has specific CSS limitations compared to web CSS. Always use supported properties and test visibility issues systematically.