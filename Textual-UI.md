# Textual UI Analysis: Ingestion Pages

## Executive Summary

This document provides a comprehensive analysis of the Textual UI implementation for the Ingestion pages in the tldw_chatbook application. The analysis identifies several critical layout and styling issues that prevent the UI from displaying as intended, along with actionable recommendations for fixes.

## Current Architecture Overview

### Layout Structure
The Ingestion window uses a two-pane horizontal layout:
- **Left Navigation Pane** (`ingest-nav-pane`): 25% width with navigation buttons
- **Right Content Pane** (`ingest-content-pane`): Flexible width (`1fr`) containing view areas

### View Management
- Multiple view areas are shown/hidden using `display: none/block`
- Each media type has its own dedicated view container
- Views include: Prompts, Characters, Notes, Local Media (8 types), and TLDW API (8 types)

## Critical Issues Identified

### 1. Fixed Width Navigation Pane
**Problem**: The navigation pane has a fixed 25% width with min/max constraints:
```css
.ingest-nav-pane {
    width: 25%;
    min-width: 25;
    max-width: 60;
}
```
**Impact**: On narrow terminals, 25% might be too wide, causing content overflow. The min-width of 25 characters can push content off-screen.

### 2. Height Management Issues
**Problem**: Several containers use problematic height settings:
```css
.ingest-selected-files-list {
    height: 5;  /* Fixed height */
}
.ingest-status-area {
    height: 8;  /* Fixed height */
}
```
**Impact**: Fixed heights prevent content from expanding when needed, causing text to be cut off or hidden.

### 3. Nested Scrolling Conflicts
**Problem**: Multiple levels of scrollable containers:
- Main window container
- Navigation pane with `overflow-y: auto`
- Content pane with `overflow-y: auto`  
- Individual view areas with `VerticalScroll` widgets
- Preview areas with additional `VerticalScroll`

**Impact**: Nested scrolling creates unpredictable behavior and can trap scroll events, preventing users from accessing content.

### 4. Display Toggle Anti-Pattern
**Problem**: View switching uses `display: none/block`:
```python
child.styles.display = "none"  # Hidden
child.styles.display = "block"  # Shown
```
**Impact**: This causes layout recalculation and potential flashing. Content may not render properly when made visible.

### 5. Container Overflow Issues
**Problem**: The content pane lacks proper overflow handling:
```css
.ingest-content-pane {
    width: 1fr;
    height: 100%;
    overflow-y: auto;  /* May conflict with child scrollables */
}
```
**Impact**: Child elements may overflow the container, especially with fixed heights on sub-elements.

### 6. Missing Responsive Design
**Problem**: No media queries or responsive breakpoints exist. The layout assumes a minimum terminal width.
**Impact**: UI breaks on narrow terminals (< 80 characters wide).

## Textual Framework Considerations

### CSS Layout Rules
1. **Fractional Units**: Use `1fr` for flexible sizing instead of percentages
2. **Height Inheritance**: Parent containers must have defined heights for percentage-based children
3. **Overflow Handling**: Use `overflow: auto` sparingly; prefer `VerticalScroll` widgets
4. **Display Property**: Use `visibility: hidden` instead of `display: none` for smoother transitions

### Container Best Practices
1. **VerticalScroll**: Should be immediate parent of scrollable content
2. **Height Management**: Use `height: 100%` or `height: 1fr` for fill behavior
3. **Nested Containers**: Avoid multiple levels of scrollable containers

## Specific Component Analysis

### IngestWindow (Main Container)
- Uses `layout: horizontal` correctly
- Missing proper height constraints for child containers
- Should implement responsive width switching

### Navigation Pane
- Fixed width causes issues on narrow terminals
- Should use collapsible design or overlay on small screens
- Overflow handling conflicts with button interaction

### Content Views
- Each view properly uses `Vertical` containers
- Missing consistent height management
- Preview areas use nested `VerticalScroll` unnecessarily

### TLDW API Windows
- Proper use of `Collapsible` for advanced options
- Form layout uses appropriate `Horizontal` grouping
- Status areas need flexible height instead of fixed

## Recommended Fixes

### 1. Responsive Navigation
```css
.ingest-nav-pane {
    width: 25%;
    min-width: 15;  /* Reduced minimum */
    max-width: 40;  /* Reduced maximum */
}

/* Add media query equivalent using conditional styling */
@media (max-width: 80) {
    .ingest-nav-pane {
        width: 100%;
        position: overlay;  /* Textual-specific */
    }
}
```

### 2. Flexible Height System
```css
.ingest-selected-files-list {
    height: auto;
    min-height: 5;
    max-height: 10;
}

.ingest-preview-area {
    height: 1fr;  /* Fill available space */
    min-height: 10;
}

.ingest-status-area {
    height: auto;
    min-height: 5;
    max-height: 15;
}
```

### 3. Proper Scrolling Hierarchy
- Remove `overflow-y: auto` from parent containers
- Use single `VerticalScroll` per view area
- Ensure proper height inheritance chain

### 4. View Switching Improvement
```python
# Instead of display manipulation
child.styles.visibility = "hidden"  # Hidden but maintains layout
child.styles.visibility = "visible"  # Shown

# Or use Textual's built-in methods
child.visible = False/True
```

### 5. Container Structure Refactor
```python
# Recommended structure
with Container(id="ingest-window"):
    with VerticalScroll(id="nav-pane"):
        # Navigation buttons
    with Container(id="content-pane"):
        with VerticalScroll(id="active-view"):
            # Current view content
```

## Action Plan

### Phase 1: Critical Fixes (Immediate)
1. **Fix Height Constraints**
   - Replace all fixed heights with flexible units
   - Add min/max height constraints where needed
   - Ensure proper height inheritance

2. **Resolve Scrolling Conflicts**
   - Remove nested `overflow` properties
   - Consolidate to single scroll container per view
   - Test scroll behavior across all views

3. **Improve View Switching**
   - Replace `display` toggling with `visibility`
   - Add transition states for smoother UX
   - Implement proper view lifecycle

### Phase 2: Layout Improvements (Short-term)
1. **Responsive Navigation**
   - Implement collapsible sidebar
   - Add hamburger menu for narrow screens
   - Create overlay mode for mobile-like terminals

2. **Form Layout Standardization**
   - Create reusable form components
   - Implement consistent spacing system
   - Add proper label-input associations

3. **Status Area Enhancement**
   - Replace TextArea with custom status widget
   - Add progress indicators
   - Implement color-coded status messages

### Phase 3: UX Enhancements (Long-term)
1. **Progressive Disclosure**
   - Move advanced options to secondary screens
   - Implement wizard-style flow for complex ingestions
   - Add context-sensitive help

2. **Performance Optimization**
   - Lazy load view content
   - Implement virtual scrolling for large lists
   - Add debouncing for responsive updates

3. **Accessibility Improvements**
   - Add keyboard navigation shortcuts
   - Implement focus management
   - Add screen reader annotations

## Testing Recommendations

1. **Terminal Size Testing**
   - Test with minimum size (80x24)
   - Test with various aspect ratios
   - Verify responsive behavior

2. **Content Overflow Testing**
   - Test with maximum content in each view
   - Verify scroll behavior with long lists
   - Check status area with extended output

3. **Performance Testing**
   - Measure view switching speed
   - Profile memory usage with large datasets
   - Test with slow terminal emulators

## Conclusion

The Ingestion UI has a solid foundation but requires significant adjustments to work properly within Textual's layout system. The primary issues stem from:
1. Fixed dimensions preventing proper content flow
2. Conflicting overflow handling between nested containers
3. Lack of responsive design considerations
4. Suboptimal view management patterns

By implementing the recommended fixes in phases, the UI can be transformed into a robust, responsive interface that works well across all terminal sizes and content scenarios.

---

## Document Review & Deficiencies

Upon review, this document could be enhanced with:

1. **Code Examples**: More specific code snippets showing exact implementations
2. **Visual Diagrams**: ASCII diagrams showing current vs. proposed layouts
3. **Performance Metrics**: Specific measurements of current issues
4. **Textual Version Specifics**: Consider Textual version differences
5. **Alternative Solutions**: Multiple approaches for each issue

## Deep Research Findings & Implementation Guide

### Critical Discovery: Missing Navigation Logic
The IngestWindow's navigation buttons (`ingest-nav-*`) have **no handler implementation**. The `on_button_pressed` method in IngestWindow doesn't handle these buttons, explaining why view switching is broken.

### Phase 1: Detailed Implementation Analysis

#### 1. Fix Height Constraints - Potential Issues & Solutions

**Issues Found:**
- Fixed heights (`height: 5`, `height: 8`) throughout `_ingest.tcss`
- Parent containers lack proper height definitions, breaking percentage-based children
- Textual's `height: auto` behaves differently than web CSS

**Concrete Solutions:**
```css
/* REPLACE these problematic fixed heights */
.ingest-selected-files-list {
    height: 5;  /* BAD: Fixed height */
}

/* WITH flexible Textual-appropriate units */
.ingest-selected-files-list {
    height: 1fr;
    min-height: 5;
    max-height: 15;
}

/* Ensure parent containers enable height inheritance */
.ingest-view-area {
    height: 100%;  /* Must be explicit for children */
}
```

**Risk Mitigation:**
- Test each height change individually
- Use `min-height` to prevent collapse
- Verify with `textual console` for layout debugging

#### 2. Resolve Scrolling Conflicts - Complete Strategy

**Critical Issues:**
- Navigation pane has `overflow-y: auto` AND contains VerticalScroll
- Multiple nested VerticalScroll widgets compete for scroll events
- Content pane's overflow conflicts with child scrollables

**Implementation Plan:**
```python
# WRONG: Nested scrolling
with VerticalScroll():  # Parent scroll
    with VerticalScroll():  # Child scroll - CONFLICTS!
        content()

# CORRECT: Single scroll hierarchy
with Container():  # No scroll
    with VerticalScroll():  # Only scroll container
        content()
```

**CSS Fixes Required:**
```css
/* REMOVE all overflow properties */
.ingest-nav-pane {
    /* overflow-y: auto; */  /* DELETE THIS */
    /* overflow-x: hidden; */ /* DELETE THIS */
}

.ingest-content-pane {
    /* overflow-y: auto; */  /* DELETE THIS */
}

/* Let VerticalScroll handle ALL scrolling */
```

#### 3. Implement Proper View Switching

**Current Problem:**
```python
# Current implementation is MISSING!
def on_button_pressed(self, event):
    button_id = event.button.id
    # No handling for ingest-nav-* buttons!
```

**Complete Implementation:**
```python
async def on_button_pressed(self, event: Button.Pressed) -> None:
    button_id = event.button.id
    
    # Handle navigation buttons
    if button_id.startswith("ingest-nav-"):
        view_id = button_id.replace("ingest-nav-", "ingest-view-")
        await self.switch_view(view_id)
        event.stop()

async def switch_view(self, new_view_id: str) -> None:
    """Properly switch views with lifecycle management."""
    # 1. Save scroll position of current view
    current_view = self.query_one(".ingest-view-area:visible")
    if current_view:
        self._scroll_positions[current_view.id] = current_view.scroll_y
    
    # 2. Hide all views
    for view in self.query(".ingest-view-area"):
        view.display = False  # Use Textual's display property
    
    # 3. Show new view
    new_view = self.query_one(f"#{new_view_id}")
    new_view.display = True
    
    # 4. Restore scroll position
    if new_view_id in self._scroll_positions:
        new_view.scroll_to(y=self._scroll_positions[new_view_id])
    
    # 5. Update active button styling
    for btn in self.query(".ingest-nav-button"):
        btn.remove_class("active")
    self.query_one(f"#{new_view_id.replace('view', 'nav')}").add_class("active")
```

### Phase 2: Enhanced Implementation Details

#### 1. Responsive Navigation - Working Pattern

**Copy from CodingWindow's proven approach:**
```python
class IngestWindow(Container):
    sidebar_collapsed = reactive(False)
    
    def compose(self):
        # Add collapse button to nav pane
        with VerticalScroll(id="ingest-nav-pane") as nav:
            yield Button("◀", id="collapse-nav", classes="nav-toggle")
            # ... rest of navigation
    
    def watch_sidebar_collapsed(self, collapsed: bool):
        nav = self.query_one("#ingest-nav-pane")
        if collapsed:
            nav.add_class("collapsed")
        else:
            nav.remove_class("collapsed")
```

**CSS for responsive behavior:**
```css
.ingest-nav-pane {
    width: 25%;
    transition: width 200ms;
}

.ingest-nav-pane.collapsed {
    width: 3;  /* Just show toggle button */
}

/* Responsive breakpoint simulation */
@container (max-width: 80) {  /* Textual 0.47+ */
    .ingest-nav-pane {
        position: absolute;
        z-index: 10;
    }
}
```

#### 2. Form Layout Standardization - Extraction Pattern

**Create reusable form builders:**
```python
def create_form_field(label: str, field_id: str, 
                     field_type="input", **kwargs) -> ComposeResult:
    """Standardized form field creation."""
    yield Label(f"{label}:", classes="form-label")
    if field_type == "input":
        yield Input(id=field_id, classes="form-input", **kwargs)
    elif field_type == "textarea":
        yield TextArea(id=field_id, classes="form-textarea", **kwargs)
    elif field_type == "select":
        yield Select(id=field_id, classes="form-select", **kwargs)
```

#### 3. Status Area Enhancement - Rich Integration

**Enhanced status with Rich formatting:**
```python
class StatusArea(Widget):
    """Custom status widget with color coding."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._messages = []
    
    def add_status(self, message: str, level: str = "info"):
        colors = {
            "info": "blue",
            "success": "green", 
            "warning": "yellow",
            "error": "red"
        }
        formatted = f"[{colors.get(level, 'white')}]{message}[/]"
        self._messages.append(formatted)
        self.refresh()
    
    def render(self) -> str:
        return "\n".join(self._messages[-10:])  # Show last 10
```

### Testing Strategy for Each Fix

1. **Height Testing:**
   ```bash
   # Test with minimal content
   # Test with overflow content
   # Test terminal resize
   ```

2. **Scroll Testing:**
   ```python
   # Unit test for single scroll container
   assert len(view.query("VerticalScroll")) == 1
   ```

3. **View Switch Testing:**
   ```python
   # Test rapid switching
   # Test scroll position preservation
   # Test async operation cancellation
   ```

## Next Steps Action Plan (Revised)

### Immediate Actions (This Week)
1. **Fix missing navigation handler** - Critical blocker
2. **Remove ALL overflow CSS properties** - Conflicts with VerticalScroll
3. **Implement proper view switching** with lifecycle management
4. Create test cases for each identified issue
5. Set up terminal size testing environment

### Short-term Actions (Next 2 Weeks)
1. **Extract working patterns** from CodingWindow for collapsible sidebar
2. **Replace fixed heights** systematically with testing
3. Create reusable form component library
4. **Implement scroll position preservation**

### Long-term Actions (Next Month)
1. Add responsive breakpoints using Textual's container queries
2. Implement Rich-enhanced status areas
3. Create comprehensive UI component documentation
4. Profile and optimize performance bottlenecks

### Success Metrics
- View navigation works without missing handler errors
- No nested scrolling conflicts
- All content visible without clipping
- Scroll positions preserved between view switches
- Height inheritance works properly throughout
- Responsive sidebar collapses on narrow terminals

## Critical Warnings & Textual-Specific Pitfalls

### 1. **Height Unit Gotchas**
- `height: auto` in Textual doesn't work like CSS - it collapses to 0
- `height: 100%` requires ALL parent containers to have explicit heights
- `height: 1fr` only works in containers with `layout: vertical/horizontal`

### 2. **VerticalScroll Limitations**
- Cannot nest VerticalScroll widgets - inner ones will break
- Parent must have explicit height for VerticalScroll to work
- Don't mix CSS overflow with VerticalScroll - choose one approach

### 3. **Display vs Visible Property**
- `display = False` removes from layout (like `display: none`)
- `visible = False` hides but maintains space (like `visibility: hidden`)
- For view switching, use `display` to avoid layout gaps

### 4. **Reactive State Timing**
- Reactive watchers fire immediately on mount - check for initialization
- Use `call_after_refresh()` for DOM-dependent operations
- Async operations in watchers need careful cancellation

### 5. **Terminal Compatibility**
- Not all terminals support all colors - test with basic 16-color mode
- Some terminals don't report size changes - poll if needed
- Unicode characters (arrows, icons) may not render correctly

### 6. **Performance Considerations**
- Large TextAreas (>1000 lines) can slow down rendering
- Multiple simultaneous CSS class changes cause layout thrashing
- Use `batch_update()` context manager for multiple DOM changes

### 7. **Focus Management**
- Textual doesn't automatically manage focus on view switches
- Hidden elements can still have focus - explicitly blur them
- Tab order needs manual management in dynamic layouts

## Final Implementation Checklist

- [x] Remove ALL CSS overflow properties ✅ (Phase 1)
- [x] Implement missing navigation button handlers ✅ (Phase 1)
- [x] Replace fixed heights with flexible units ✅ (Phase 1)
- [x] Ensure single VerticalScroll per view ✅ (Phase 1)
- [ ] Add scroll position preservation (Future enhancement)
- [ ] Test with 80x24 terminal minimum (Needs testing)
- [x] Verify no nested scrollable containers ✅ (Phase 1)
- [ ] Implement proper focus management (Future enhancement)
- [ ] Add loading states for async operations (Partially done with EnhancedStatusWidget)
- [ ] Test with screen readers (if applicable)

## Implementation Status

### ✅ Phase 1: Critical Fixes (COMPLETED)
1. **Fixed Height Constraints** - All fixed heights replaced with min/max/auto
2. **Resolved Scrolling Conflicts** - Removed all CSS overflow properties
3. **Implemented View Switching** - Added missing navigation handlers

### ✅ Phase 2: Layout Improvements (COMPLETED)
1. **Responsive Navigation** - Collapsible sidebar implemented
2. **Form Layout Standardization** - Created reusable form components
3. **Status Area Enhancement** - Created EnhancedStatusWidget with color coding

### 📄 Documentation Created
- `Textual-UI-Implementation-Summary.md` - Detailed summary of all changes
- `form_components.py` - Reusable form building utilities
- `status_widget.py` - Enhanced status display widget
- `Ingest_Window_Example.py` - Example implementation using new components

See `Textual-UI-Implementation-Summary.md` for complete implementation details.

## Phase 3: UX Enhancements - Detailed Implementation Plan

### Overview
Phase 3 focuses on creating a more intuitive, performant, and accessible user experience through progressive disclosure, performance optimizations, and comprehensive accessibility improvements.

### 1. Progressive Disclosure Implementation

#### 1.1 Wizard-Style Flow for Complex Ingestions
**Goal**: Guide users through complex ingestion processes step-by-step, reducing cognitive load.

**Implementation Strategy**:
```python
class IngestionWizard(Container):
    """Multi-step wizard for guided ingestion."""
    
    current_step = reactive(0)
    total_steps = reactive(4)
    
    STEPS = [
        "Select Files",
        "Configure Processing", 
        "Set Metadata",
        "Review & Process"
    ]
    
    def compose(self):
        # Progress indicator
        yield WizardProgress(self.current_step, self.total_steps)
        
        # Step content container
        with Container(id="wizard-content"):
            yield from self.render_current_step()
        
        # Navigation buttons
        yield WizardNavigation(
            on_previous=self.previous_step,
            on_next=self.next_step,
            on_finish=self.complete_wizard
        )
```

**Key Features**:
- Step-by-step progression with visual progress indicator
- Validation before advancing to next step
- Ability to go back and modify previous selections
- Context preservation across steps
- Smart defaults based on media type

#### 1.2 Advanced Options Toggle
**Implementation**:
```python
class AdvancedOptionsToggle(Widget):
    """Toggle between basic and advanced mode."""
    
    show_advanced = reactive(False)
    
    def compose(self):
        yield Checkbox(
            "Show Advanced Options",
            value=self.show_advanced,
            id="advanced-toggle"
        )
        
        with Container(id="advanced-container") as container:
            container.display = self.show_advanced
            yield from self.advanced_options()
    
    def watch_show_advanced(self, show: bool):
        container = self.query_one("#advanced-container")
        container.display = show
        # Save preference
        self.app.save_preference("show_advanced", show)
```

#### 1.3 Context-Sensitive Help System
**Components**:
1. **Inline Help Icons**: Small (?) icons next to complex fields
2. **Tooltip System**: Rich tooltips with examples
3. **Help Panel**: Collapsible help section with detailed guides
4. **Interactive Examples**: Show example inputs for different scenarios

```python
def create_help_tooltip(field_id: str, help_text: str) -> Widget:
    """Create a help icon with tooltip."""
    return Button(
        "?",
        id=f"{field_id}-help",
        classes="help-icon",
        tooltip=help_text
    )
```

### 2. Performance Optimization Strategy

#### 2.1 Lazy Loading for View Content
**Problem**: All views are composed at startup, consuming memory and slowing initial load.

**Solution**: Implement lazy view composition
```python
class LazyView(Container):
    """Base class for lazily-loaded views."""
    
    _composed = False
    
    def on_mount(self):
        if not self._composed:
            self.compose_content()
            self._composed = True
    
    def compose_content(self):
        """Override to provide actual content."""
        raise NotImplementedError
```

#### 2.2 Virtual Scrolling for Large Lists
**Implementation**: Create VirtualListView widget
```python
class VirtualListView(VerticalScroll):
    """ListView that only renders visible items."""
    
    items = reactive([])
    item_height = reactive(3)
    visible_range = reactive((0, 20))
    
    def __init__(self, items: List[Any], item_renderer: Callable):
        super().__init__()
        self.items = items
        self.item_renderer = item_renderer
        self._item_cache = {}
    
    def on_scroll(self, event):
        # Calculate visible range based on scroll position
        start_idx = int(self.scroll_y / self.item_height)
        end_idx = start_idx + int(self.height / self.item_height) + 1
        
        if (start_idx, end_idx) != self.visible_range:
            self.visible_range = (start_idx, end_idx)
            self.refresh_visible_items()
    
    def refresh_visible_items(self):
        # Only render items in visible range
        start, end = self.visible_range
        for idx in range(start, min(end, len(self.items))):
            if idx not in self._item_cache:
                self._item_cache[idx] = self.item_renderer(self.items[idx])
```

**Benefits**:
- Handles 10,000+ items smoothly
- Constant memory usage regardless of list size
- Smooth scrolling performance

#### 2.3 Debounced Updates
**Implementation**: Debounce utility for responsive updates
```python
from asyncio import create_task, sleep, CancelledError

class DebouncedInput(Input):
    """Input with debounced change events."""
    
    debounce_time = 0.3  # seconds
    
    def __init__(self, *args, debounce_time: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.debounce_time = debounce_time
        self._debounce_task = None
    
    async def _on_change(self, event):
        # Cancel previous debounce task
        if self._debounce_task:
            self._debounce_task.cancel()
        
        # Create new debounce task
        self._debounce_task = create_task(self._debounced_change(event))
    
    async def _debounced_change(self, event):
        try:
            await sleep(self.debounce_time)
            self.post_message(self.DebouncedChange(self, event.value))
        except CancelledError:
            pass
```

### 3. Accessibility Improvements

#### 3.1 Comprehensive Keyboard Navigation
**Implementation Plan**:

1. **Global Shortcuts** (app.py):
```python
KEYBOARD_SHORTCUTS = {
    "ctrl+n": "new_ingestion",
    "ctrl+o": "open_file",
    "ctrl+s": "save_current",
    "ctrl+/": "toggle_help",
    "alt+1-9": "switch_to_tab_n",
    "ctrl+tab": "next_tab",
    "ctrl+shift+tab": "previous_tab",
    "escape": "close_dialog_or_cancel",
    "f1": "context_help",
    "f6": "focus_navigation",
    "f10": "focus_menu"
}
```

2. **Form Navigation**:
   - Tab: Next field
   - Shift+Tab: Previous field
   - Enter: Submit form (when in last field)
   - Escape: Cancel/close form
   - Space: Toggle checkboxes/expand collapsibles

3. **List Navigation**:
   - Arrow keys: Navigate items
   - Space: Select/deselect item
   - Ctrl+A: Select all
   - Delete: Remove selected items

#### 3.2 Focus Management System
**Implementation**:
```python
class FocusManager:
    """Manages focus flow and trapping."""
    
    def __init__(self, app):
        self.app = app
        self.focus_stack = []
    
    def push_focus_context(self, container: Widget):
        """Save current focus and trap focus within container."""
        current_focus = self.app.focused
        self.focus_stack.append(current_focus)
        
        # Find first focusable element in container
        first_focusable = container.query("Input, Button, Select, TextArea").first()
        if first_focusable:
            first_focusable.focus()
        
        # Trap focus within container
        self._trap_focus(container)
    
    def pop_focus_context(self):
        """Restore previous focus context."""
        if self.focus_stack:
            previous_focus = self.focus_stack.pop()
            if previous_focus:
                previous_focus.focus()
```

#### 3.3 Screen Reader Support
**Implementation Strategy**:

1. **Semantic Structure**:
```python
# Use proper ARIA roles
yield Static(
    "Processing Options",
    classes="section-header",
    id="processing-header"
)
yield Input(
    placeholder="Enter title",
    id="title-input",
    classes="form-input",
    aria_labelledby="processing-header"
)
```

2. **Live Regions** for status updates:
```python
class AccessibleStatus(Static):
    """Status widget with screen reader announcements."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_class("aria-live-polite")
    
    def update_status(self, message: str, urgent: bool = False):
        self.update(message)
        if urgent:
            self.add_class("aria-live-assertive")
        else:
            self.add_class("aria-live-polite")
```

3. **Form Validation Announcements**:
```python
def announce_validation_error(field: Widget, error: str):
    """Announce validation errors to screen readers."""
    field.add_class("aria-invalid")
    field.set_attribute("aria-describedby", f"{field.id}-error")
    
    # Create error message element
    error_elem = Static(
        error,
        id=f"{field.id}-error",
        classes="validation-error aria-live-assertive"
    )
    field.parent.mount(error_elem, after=field)
```

### 4. Implementation Priorities and Timeline

#### Phase 3.1 (Weeks 1-2): Foundation
1. Create base components for progressive disclosure
2. Implement lazy loading infrastructure
3. Set up keyboard navigation framework

#### Phase 3.2 (Weeks 3-4): Core Features
1. Build wizard component for complex ingestions
2. Implement virtual scrolling for file lists
3. Add debouncing to all input fields

#### Phase 3.3 (Weeks 5-6): Accessibility
1. Comprehensive keyboard shortcut system
2. Focus management implementation
3. Screen reader testing and fixes

#### Phase 3.4 (Week 7): Polish and Testing
1. Performance profiling and optimization
2. Accessibility audit
3. User testing with keyboard-only navigation
4. Documentation updates

### 5. Success Metrics

1. **Performance Metrics**:
   - Initial load time < 500ms
   - View switch time < 100ms
   - Smooth scrolling with 10,000+ items
   - Memory usage stable under 200MB

2. **Accessibility Metrics**:
   - All features accessible via keyboard
   - WCAG 2.1 AA compliance
   - Screen reader compatibility verified
   - Focus indicators always visible

3. **Usability Metrics**:
   - 90% task completion rate in wizard mode
   - 50% reduction in time to complete complex ingestions
   - Positive feedback on progressive disclosure
   - Reduced support requests for complex features

### 6. Testing Strategy

1. **Automated Testing**:
   - Unit tests for all new components
   - Performance benchmarks for virtual scrolling
   - Keyboard navigation test suite

2. **Manual Testing**:
   - Screen reader testing with NVDA/JAWS
   - Keyboard-only navigation testing
   - Large dataset performance testing

3. **User Testing**:
   - A/B testing wizard vs traditional flow
   - Accessibility testing with users with disabilities
   - Performance perception studies