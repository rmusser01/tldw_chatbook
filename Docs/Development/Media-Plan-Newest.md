# Media UI Rebuild Plan - Version 88

## Executive Summary

Complete architectural rebuild of the Media UI following Textual framework best practices, creating a modular, responsive, and maintainable media management interface. The primary focus is on the **Detailed Media View** with secondary views (Analysis Review, Multi-Item Review, Collections) as future enhancements.

## Architecture Overview

### Core Design Principles
1. **Composition over Inheritance**: Use modular widgets that can be composed
2. **Reactive State Management**: Leverage Textual's reactive properties for automatic UI updates
3. **Event-Driven Communication**: Decouple components through custom events
4. **Responsive Layout**: Use fractional units and flexible containers
5. **Progressive Enhancement**: Start with core functionality, layer on features

### Component Hierarchy
```
MediaWindowV88 (Main Container)
├── MediaNavigationColumn (Left Column - 20% width)
│   ├── MediaTypeSelector (Dropdown)
│   └── MediaItemList (Paged List)
│       ├── MediaListItem (Individual Items)
│       └── PaginationControls
├── MediaContentArea (Right Area - 80% width)
│   ├── MediaSearchBar (Collapsible)
│   │   ├── QuickSearch
│   │   └── AdvancedFilters (expandable)
│   ├── MediaMetadataPanel (4-row layout)
│   │   ├── MetadataDisplay
│   │   └── ActionButtons (Edit, Delete)
│   └── MediaViewerTabs
│       ├── ContentTab (Media content viewer)
│       └── AnalysisTab (Analysis viewer/generator)
```

## Implementation Plan

### Phase 1: Foundation Components

#### 1.1 Base Media Window Structure
**File**: `tldw_chatbook/UI/MediaWindowV88.py`

```python
class MediaWindowV88(Container):
    """
    Main orchestrator for the Media UI.
    Uses horizontal layout with left navigation column and right content area.
    """
    
    DEFAULT_CSS = """
    MediaWindowV88 {
        layout: horizontal;
        height: 100%;
    }
    
    #media-nav-column {
        width: 20%;
        min-width: 25;
        border-right: solid $primary;
    }
    
    #media-content-area {
        width: 1fr;
        layout: vertical;
    }
    """
```

#### 1.2 Navigation Column
**File**: `tldw_chatbook/Widgets/MediaV88/navigation_column.py`

Features:
- Dropdown for media type selection
- Scrollable list of media items
- Pagination at bottom
- Reactive updates on selection

Key Methods:
- `set_media_type(type_slug: str)`: Change active media type
- `load_items(items: List[Dict], page: int, total: int)`: Update list
- `handle_item_selection(item_id: int)`: Emit selection event

#### 1.3 Search Bar Component
**File**: `tldw_chatbook/Widgets/MediaV88/search_bar.py`

Features:
- Collapsible design with toggle button
- Quick search input
- Advanced filters in collapsible section
- Keyword tags input
- Date range filters
- Sort options

State Management:
```python
search_term: reactive[str] = reactive("")
keywords: reactive[List[str]] = reactive([])
collapsed: reactive[bool] = reactive(False)
show_advanced: reactive[bool] = reactive(False)
```

### Phase 2: Content Display Components

#### 2.1 Metadata Panel
**File**: `tldw_chatbook/Widgets/MediaV88/metadata_panel.py`

Layout:
```
Row 1: Title, Type, Date Created
Row 2: Author, URL/Source, Date Modified  
Row 3: Keywords/Tags (scrollable horizontal)
Row 4: Description/Summary
Bottom: [Edit] [Delete] buttons
```

Features:
- Read-only display mode
- Inline edit mode with validation
- Optimistic locking for concurrent edits
- Auto-save with debouncing

#### 2.2 Content Viewer Tabs
**File**: `tldw_chatbook/Widgets/MediaV88/content_viewer_tabs.py`

Tab Structure:
1. **Content Tab**:
   - Markdown/Text renderer
   - Search within content
   - Zoom controls
   - Copy functionality

2. **Analysis Tab**:
   - Analysis display (Markdown)
   - Generate new analysis button
   - Provider/Model selection
   - Save/Export options
   - Version history

### Phase 3: Data Flow & Event System

#### 3.1 Custom Events
**File**: `tldw_chatbook/Event_Handlers/media_v88_events.py`

```python
class MediaItemSelectedEventV88(Message):
    """Fired when user selects a media item"""
    media_id: int
    media_data: Dict[str, Any]

class MediaSearchEventV88(Message):
    """Fired when search parameters change"""
    search_term: str
    keywords: List[str]
    filters: Dict[str, Any]

class MediaUpdateEventV88(Message):
    """Fired when media metadata is updated"""
    media_id: int
    changes: Dict[str, Any]
    
class MediaDeleteEventV88(Message):
    """Request media deletion"""
    media_id: int
    soft_delete: bool = True
```

#### 3.2 Data Service Layer
**File**: `tldw_chatbook/Services/media_service_v88.py`

Responsibilities:
- Abstract database operations
- Handle caching
- Manage pagination
- Coordinate with sync engine

Key Methods:
```python
async def search_media(
    query: str = None,
    media_type: str = None,
    keywords: List[str] = None,
    page: int = 1,
    per_page: int = 20
) -> Tuple[List[Dict], int]:
    """Search media with filters"""

async def get_media_details(media_id: int) -> Dict[str, Any]:
    """Get full media item with content"""

async def update_media(media_id: int, updates: Dict) -> bool:
    """Update media metadata"""

async def generate_analysis(media_id: int, params: Dict) -> str:
    """Generate AI analysis"""
```

### Phase 4: State Management

#### 4.1 Media Store
**File**: `tldw_chatbook/Stores/media_store_v88.py`

```python
class MediaStoreV88:
    """
    Centralized state management for Media UI.
    Uses reactive properties for automatic UI updates.
    """
    
    # Current view state
    active_media_type: reactive[str] = reactive("all-media")
    selected_media_id: reactive[Optional[int]] = reactive(None)
    
    # Search state
    search_params: reactive[Dict] = reactive({})
    search_results: reactive[List[Dict]] = reactive([])
    
    # UI state
    navigation_collapsed: reactive[bool] = reactive(False)
    search_collapsed: reactive[bool] = reactive(False)
    
    # Cache
    _media_cache: Dict[int, Dict] = {}
    _analysis_cache: Dict[int, List[Dict]] = {}
```

### Phase 5: CSS Architecture

#### 5.1 Modular CSS System
**File**: `tldw_chatbook/css/components/_media_v88.tcss`

Structure:
```css
/* Base layout */
MediaWindowV88 { }

/* Navigation column */
.media-nav-column { }
.media-type-selector { }
.media-item-list { }

/* Content area */
.media-content-area { }
.media-search-bar { }
.media-metadata-panel { }
.media-viewer-tabs { }

/* States */
.collapsed { display: none; }
.selected { background: $accent; }
.loading { opacity: 0.6; }
```

#### 5.2 Theme Variables
```css
/* Media UI specific theme variables */
--media-nav-width: 20%;
--media-nav-min-width: 25;
--media-list-item-height: 5;
--media-metadata-rows: 4;
--media-search-height: auto;
```

## Testing Strategy

### Unit Tests
**File**: `Tests/UI/test_media_window_v88.py`

Coverage areas:
1. Component initialization
2. Event propagation
3. State updates
4. Data binding
5. Error handling

### Integration Tests
**File**: `Tests/Integration/test_media_flow_v88.py`

Test scenarios:
1. Search → Select → View flow
2. Edit → Save → Refresh flow
3. Generate analysis → Save flow
4. Pagination with filters
5. Concurrent edit handling

## Migration Strategy

### Gradual Rollout
1. Implement MediaWindowV88 alongside existing MediaWindow_v2
2. Add feature flag in config: `use_new_media_ui: false`
3. Test with subset of users
4. Migrate data and remove old implementation

### Backward Compatibility
- Reuse existing database layer
- Maintain event compatibility where possible
- Preserve keyboard shortcuts
- Keep same URL/navigation structure

## Performance Optimizations

### Lazy Loading
- Load media content only when selected
- Virtualize long lists
- Defer analysis loading

### Caching Strategy
- LRU cache for media items (size: 100)
- Cache search results for 5 minutes
- Invalidate on updates

### Debouncing
- Search input: 300ms
- Metadata save: 1000ms
- Resize events: 100ms

## Accessibility Features

### Keyboard Navigation
- Tab through all interactive elements
- Arrow keys for list navigation
- Escape to close modals/collapse panels
- Enter to select/confirm

### Screen Reader Support
- Semantic HTML roles
- ARIA labels for icons
- Status announcements for async operations

## Future Enhancements (Lower Priority)

### Analysis Review View
- Side-by-side analysis comparison
- Version history timeline
- Diff view for changes
- Bulk analysis operations

### Multi-Item Review
- Card-based layout
- Batch operations toolbar
- Quick navigation between items
- Export to various formats

### Collections View
- Tag cloud visualization
- Drag-and-drop organization
- Smart collections (auto-filter)
- Sharing/collaboration features

## Implementation Timeline

### Week 1: Foundation
- [ ] Base window structure
- [ ] Navigation column with dropdown and list
- [ ] Basic event system

### Week 2: Core Features
- [ ] Search bar with collapse
- [ ] Metadata panel with display
- [ ] Content viewer tabs

### Week 3: Interactivity
- [ ] Edit functionality
- [ ] Delete with confirmation
- [ ] Analysis generation

### Week 4: Polish
- [ ] Performance optimization
- [ ] Error handling
- [ ] Testing and refinement

## Success Metrics

1. **Performance**: Page load < 500ms, search response < 200ms
2. **Usability**: 90% task completion rate
3. **Maintainability**: 80% code coverage, < 10 cyclomatic complexity
4. **Accessibility**: WCAG 2.1 AA compliance

## Risk Mitigation

### Technical Risks
- **Database performance**: Add indexes, implement pagination
- **Memory leaks**: Proper cleanup in unmount, weak references
- **State sync issues**: Single source of truth, immutable updates

### User Experience Risks
- **Learning curve**: Provide tooltips, maintain familiar patterns
- **Data loss**: Auto-save, confirmation dialogs, undo functionality
- **Performance degradation**: Progressive loading, virtualization

## Conclusion

This plan provides a comprehensive blueprint for rebuilding the Media UI with a focus on maintainability, performance, and user experience. The modular architecture allows for incremental development and testing while maintaining backward compatibility. The primary Detailed Media View will serve as the foundation for future enhancements.