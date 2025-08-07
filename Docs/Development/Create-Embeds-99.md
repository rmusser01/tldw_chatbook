# New Search Embeddings Window Design Plan

## Overview
Create a new embeddings creation interface following the current Chatbook layout pattern, replacing the existing wizard-based approach with a more streamlined, single-window design.

## Design Layout

### Top Section (Header)
- **Title**: "Search Embeddings Window" 
- **Action**: `Launch Wizard` button (fallback to existing wizard if needed)

### Content Type Selection (Row 2)
- Single horizontal row with checkboxes:
  - ☑️ Chats | ☑️ Character Chats | ☑️ Notes | ☑️ Media
- Multiple selection allowed (users can combine content types)

### Content Tree & Settings (Row 3)
**Left Side (60% width):**
- **Title**: "Content"
- **Filter Box**: Keyword search for content discovery
- **Tree Widget**: SmartContentTree showing hierarchical content from selected categories
  - Persistent selection across searches (search "keyword1" → select items → search "keyword2" → select more items → both remain selected)
  - Support for selecting individual items, groups, or entire categories
  - Visual indicators for selected items

**Right Side (40% width):**
- **Model Dropdown**: Select embedding model
- **Advanced Options**: Collapsible section with:
  - Chunk size settings
  - Overlap settings  
  - Storage backend options
  - Collection naming

### Bottom Section - Split View
**Left Side:**
- **Title**: "Create Embeddings"
- **Settings**: Collection name input, final options
- **Actions**: [Cancel] button (left), [Create Embeddings] button (right)

**Right Side:**
- **Title**: "Embedding Results" 
- **Progress Display**: Real-time status tracking, progress bars, logs

## Technical Implementation Plan

### Phase 1: New Window Structure
1. **Create**: `SearchEmbeddingsWindow.py` - new main window class
2. **Integrate**: Enhanced content selection using existing `SmartContentTree.py`
3. **Layout**: Use Horizontal/Vertical containers to match described layout
4. **Styling**: Follow existing Chatbook CSS patterns from `Chat_Window_Enhanced.py`

### Phase 2: Content Integration
1. **Content Sources**: Integrate with existing DBs (CharactersRAGDB, MediaDatabase)
2. **Tree Population**: Load content based on checkbox selections
3. **Search Functionality**: Real-time filtering with persistent selection
4. **Multi-Selection**: Cross-search selection preservation

### Phase 3: Settings & Controls
1. **Model Management**: Integration with existing EmbeddingFactory
2. **Advanced Options**: Collapsible panel with embeddings config
3. **Form Validation**: Real-time validation of inputs
4. **State Management**: Reactive attributes for UI updates

### Phase 4: Processing Integration  
1. **Progress Tracking**: Real-time embedding creation status
2. **Background Processing**: Use existing embeddings creation logic
3. **Result Display**: Status logs, progress bars, completion feedback
4. **Error Handling**: Graceful error display and recovery

### Phase 5: Navigation Integration
1. **Replace**: Current SearchWindow embeddings views 
2. **Update**: Navigation to use new SearchEmbeddingsWindow
3. **Migration**: Preserve existing wizard as fallback option
4. **Testing**: Ensure smooth integration with existing flows

## Files to Create/Modify

### New Files:
- `tldw_chatbook/UI/SearchEmbeddingsWindow.py` - Main window class
- `Create-Embeds-99.md` - This design document

### Files to Modify:
- `tldw_chatbook/UI/SearchWindow.py` - Update embeddings integration
- Existing CSS files - Add styles for new layout
- Navigation handlers - Update to use new window

### Dependencies:
- Leverage existing: `SmartContentTree`, `EmbeddingFactory`, `ChromaDBManager`
- Enhance: Multi-selection persistence, real-time filtering
- Integrate: Progress tracking, background processing

## Key Features

### Enhanced UX:
- Single-window workflow (no multi-step wizard)
- Real-time content preview and selection
- Persistent selection across searches
- Immediate feedback and validation

### Technical Benefits:
- Simplified codebase (single window vs multi-step wizard)
- Better performance (no step transitions)
- Enhanced state management
- More intuitive user flow

This plan transforms the current multi-step wizard into a streamlined, single-window interface that matches the existing Chatbook design patterns while providing enhanced functionality for content selection and embeddings creation.

## Textual Framework Integration Guidelines

### Core Textual Patterns to Follow

Based on the Textual-LLM-Use-1.md reference guide, we will implement the following patterns:

#### 1. Widget Architecture
```python
from textual.widget import Widget
from textual.reactive import reactive
from textual.containers import Container, Horizontal, Vertical
from textual import on

class SearchEmbeddingsWindow(Container):
    """Main window following Textual best practices."""
    
    # Reactive state management
    selected_content_types = reactive(set())
    selected_items = reactive(set())
    is_processing = reactive(False)
    
    # Lifecycle methods
    def on_mount(self) -> None:
        """Initialize after mounting."""
        self.load_initial_data()
    
    def compose(self) -> ComposeResult:
        """Define widget structure."""
        # Implementation follows
```

#### 2. Event Handling with @on Decorator
```python
@on(Checkbox.Changed)
def handle_content_type_selection(self, event: Checkbox.Changed) -> None:
    """Handle content type checkbox changes."""
    pass

@on(Button.Pressed, "#create-embeddings")
def handle_create_embeddings(self) -> None:
    """Handle embedding creation."""
    pass
```

#### 3. Reactive Programming
- Use `reactive` attributes for state that triggers UI updates
- Implement `watch_*` methods for state change reactions
- Use `recompose=True` for major UI rebuilds, `layout=True` for layout updates

#### 4. Worker Pattern for Background Tasks
```python
from textual.worker import work

@work(thread=True, exclusive=True)
def process_embeddings(self) -> None:
    """Process embeddings in background thread."""
    # Heavy processing logic
    self.call_from_thread(self.update_progress, progress)
```

#### 5. CSS Styling Strategy
- Use DEFAULT_CSS class attribute for widget-specific styles
- Follow semantic naming conventions: `.content-tree`, `.settings-panel`
- Leverage CSS variables for theming consistency
- Use CSS Grid for complex layouts

### Performance Considerations

1. **Lazy Loading**: Load content tree data only when content types are selected
2. **Batch Updates**: Update multiple reactive attributes together to minimize redraws  
3. **Worker Usage**: Use workers for I/O operations and heavy processing
4. **Minimal Recomposition**: Prefer `refresh()` over `recompose=True` when possible

### Error Handling Pattern
```python
@work
async def load_content_data(self, content_type: str):
    try:
        data = await self.fetch_content(content_type)
        self.populate_tree(data)
    except Exception as e:
        self.notify(f"Error loading {content_type}: {e}", severity="error")
        self.log.error(f"Content loading failed: {e}")
```

## Architectural Decision Records (ADRs)

### ADR-001: Single Window vs Multi-Step Wizard

**Status**: Accepted
**Date**: 2025-08-06

**Context**: Current embeddings creation uses multi-step wizard which requires navigation between steps and complex state management.

**Decision**: Replace multi-step wizard with single-window interface.

**Rationale**:
- Reduces cognitive load - users see all options at once
- Eliminates state management complexity between steps  
- Matches existing Chatbook UI patterns
- Allows real-time preview of selections
- Improves workflow efficiency

**Consequences**:
- More complex single-window layout
- Need robust form validation
- Requires careful information hierarchy design

### ADR-002: Content Selection Strategy

**Status**: Accepted  
**Date**: 2025-08-06

**Context**: Need to allow users to select content from multiple sources (Chats, Notes, Media, etc.) with filtering and search.

**Decision**: Use checkbox-based content type selection with SmartContentTree for item selection.

**Rationale**:
- Checkboxes allow multiple content type selection
- Tree view provides familiar hierarchical navigation
- Persistent selection across searches improves UX
- Leverages existing SmartContentTree component

**Consequences**:
- Need to enhance SmartContentTree for persistent selection
- Complex state management for cross-search selections
- Requires efficient tree population logic

### ADR-003: Reactive State Management

**Status**: Accepted
**Date**: 2025-08-06

**Context**: Need to coordinate UI state between content selection, settings, and progress display.

**Decision**: Use Textual reactive attributes for all stateful UI elements.

**Rationale**:
- Reactive programming ensures UI consistency
- Automatic re-rendering on state changes
- Clean separation of state and presentation
- Built-in validation support

**Consequences**:
- Need to design reactive attribute hierarchy carefully
- Watch methods must be efficient to avoid performance issues
- Complex state dependencies require careful management

### ADR-004: Background Processing Architecture  

**Status**: Accepted
**Date**: 2025-08-06

**Context**: Embedding creation is CPU/I-O intensive and should not block UI.

**Decision**: Use Textual worker pattern with thread-based processing for embedding creation.

**Rationale**:
- Maintains responsive UI during processing
- Built-in worker lifecycle management
- Clean separation of UI and processing logic
- Progress reporting capabilities

**Consequences**:
- Need thread-safe communication with UI
- Error handling across thread boundaries
- Worker cleanup on window close

### ADR-005: CSS Architecture

**Status**: Accepted
**Date**: 2025-08-06

**Context**: Need consistent styling that matches existing Chatbook patterns.

**Decision**: Use component-based CSS with DEFAULT_CSS attributes and semantic class names.

**Rationale**:
- Component encapsulation improves maintainability
- Semantic names improve readability
- Matches existing Textual best practices
- Easy to customize and theme

**Consequences**:
- Need to establish CSS naming conventions
- Potential style duplication across components
- CSS specificity management required

### ADR-006: Integration Strategy

**Status**: Accepted
**Date**: 2025-08-06

**Context**: New window must integrate with existing SearchWindow navigation.

**Decision**: Replace existing embeddings creation views in SearchWindow with new SearchEmbeddingsWindow.

**Rationale**:
- Maintains existing navigation patterns
- Preserves user familiarity
- Allows gradual migration
- Keeps wizard as fallback option

**Consequences**:
- Need to update SearchWindow navigation handlers
- Temporary code duplication during transition
- Testing required for navigation integration

## Implementation Notes

### Key Dependencies
- Existing: `SmartContentTree`, `EmbeddingFactory`, `ChromaDBManager`
- Database: `CharactersRAGDB`, `MediaDatabase` 
- UI: Textual framework components

### Testing Strategy  
- Unit tests for reactive state management
- Integration tests for content loading
- UI tests using Textual's testing framework
- Performance tests for large content sets

### Migration Path
1. Implement new SearchEmbeddingsWindow
2. Update SearchWindow integration
3. Test with existing data
4. Deploy with wizard fallback
5. Gather user feedback
6. Remove wizard once stable