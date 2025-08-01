# Tab Bar to Dropdown Implementation Plan

## Overview

This document outlines the implementation plan for converting the horizontal tab bar navigation system to a dropdown-based navigation docked to the upper left of the application. This change will improve screen real estate usage while maintaining all existing functionality.

## Current State Analysis

### Existing Architecture

1. **TabBar Widget** (`UI/Tab_Bar.py`)
   - Extends `Horizontal` container
   - Uses `HorizontalScroll` for overflow handling
   - Generates `Button` widgets for each tab
   - Maps tab IDs to display labels
   - Tracks active tab state via CSS classes

2. **Integration Points**
   - `app.py` composes TabBar at line 1496
   - Tab switching handled via `@on(Button.Pressed)` events
   - CSS styling in `layout/_tabs.tcss` and `layout/_containers.tcss`
   - Constants defined in `Constants.py` (TAB_CHAT, TAB_CODING, etc.)

3. **Current Behavior**
   - Horizontal tab row docked at top (height: 3)
   - Scrollable when tabs exceed viewport width
   - Visual feedback for active tab (bold, accent color)
   - Click-based navigation between tabs

## Proposed Architecture

### TabDropdown Widget Design

```python
class TabDropdown(Container):
    """Dropdown navigation widget to replace horizontal tab bar"""
    
    def __init__(self, tab_ids: List[str], initial_active_tab: str, **kwargs):
        super().__init__(**kwargs)
        self.tab_ids = tab_ids
        self.current_tab = initial_active_tab
        self.id = "tab-dropdown-container"
        
    def compose(self) -> ComposeResult:
        # Create Select widget with tab options
        options = [(self._get_tab_label(tab_id), tab_id) for tab_id in self.tab_ids]
        yield Select(
            options=options,
            value=self.current_tab,
            id="tab-dropdown-select",
            prompt="Navigate to..."
        )
```

### Key Design Decisions

1. **Widget Choice**: Use Textual's `Select` widget for native dropdown behavior
2. **Positioning**: Dock to upper-left with minimal width (auto-size to content)
3. **State Management**: Maintain current tab state within widget
4. **Event Flow**: Convert Select.Changed events to existing tab switching logic

## Implementation Steps

### Phase 1: Create TabDropdown Widget

1. **Create `UI/Tab_Dropdown.py`**
   ```python
   from typing import TYPE_CHECKING, List, Tuple
   from textual.app import ComposeResult
   from textual.containers import Container
   from textual.widgets import Select
   from textual.events import Event
   from textual.message import Message
   
   if TYPE_CHECKING:
       from ..app import TldwCli
   
   from ..Constants import (TAB_CCP, TAB_TOOLS_SETTINGS, TAB_INGEST, 
                            TAB_LLM, TAB_EVALS, TAB_CODING, TAB_STTS, 
                            TAB_STUDY, TAB_CHATBOOKS)
   
   class TabChanged(Message):
       """Message emitted when dropdown selection changes"""
       def __init__(self, tab_id: str) -> None:
           self.tab_id = tab_id
           super().__init__()
   
   class TabDropdown(Container):
       # Implementation as outlined above
   ```

2. **Label Mapping Logic**
   - Reuse existing label mapping from TabBar
   - Consider extracting to shared utility function

### Phase 2: Update Application Integration

1. **Modify `app.py`**
   ```python
   # Import change
   from .UI.Tab_Dropdown import TabDropdown  # Instead of TabBar
   
   # In _create_main_ui_widgets() around line 1496
   widgets.append(TabDropdown(
       tab_ids=ALL_TABS, 
       initial_active_tab=self._initial_tab_value
   ))
   ```

2. **Event Handler Updates**
   ```python
   @on(Select.Changed, "#tab-dropdown-select")
   def on_dropdown_tab_change(self, event: Select.Changed) -> None:
       """Handle dropdown selection changes"""
       if event.value:
           self.switch_tab(event.value)
   ```

### Phase 3: CSS Styling

1. **Create `css/features/_tab_dropdown.tcss`**
   ```css
   /* Tab Dropdown Container */
   #tab-dropdown-container {
       dock: left;
       align: left top;
       width: auto;
       min-width: 15;
       max-width: 30;
       height: 3;
       padding: 0 1;
       background: $panel;
       border-right: solid $border;
       border-bottom: solid $border;
   }
   
   /* Dropdown Select Widget */
   #tab-dropdown-select {
       width: 100%;
       background: $panel;
       color: $text;
   }
   
   #tab-dropdown-select:hover {
       background: $panel-lighten-1;
   }
   
   #tab-dropdown-select:focus {
       border: solid $accent;
   }
   
   /* Dropdown Options */
   #tab-dropdown-select SelectOverlay {
       background: $background;
       border: solid $border;
       max-height: 20;
   }
   
   #tab-dropdown-select Option {
       padding: 0 1;
       background: $panel;
   }
   
   #tab-dropdown-select Option:hover {
       background: $accent;
       color: $text;
   }
   ```

2. **Update CSS build system**
   - Add new file to `css/build_css.py` imports
   - Ensure proper module ordering

### Phase 4: Migration Path

1. **Feature Flag Approach**
   ```python
   # In config.toml
   [ui]
   use_dropdown_navigation = false  # Default to false for backward compatibility
   
   # In app.py
   use_dropdown = get_cli_setting("ui", "use_dropdown_navigation", False)
   if use_dropdown:
       widgets.append(TabDropdown(...))
   else:
       widgets.append(TabBar(...))
   ```

2. **Gradual Rollout**
   - Test with feature flag in development
   - Gather user feedback
   - Make default after stability confirmed

## Technical Considerations

### Performance

1. **Lazy Loading**: Maintain existing window lazy-loading behavior
2. **Event Efficiency**: Single event dispatch for tab changes
3. **Render Performance**: Dropdown renders fewer DOM elements than tab bar

### Accessibility

1. **Keyboard Navigation**: Select widget provides native keyboard support
2. **Screen Readers**: Ensure proper ARIA labels
3. **Visual Indicators**: Clear active tab indication in dropdown

### Edge Cases

1. **Many Tabs**: Dropdown handles overflow better than horizontal scroll
2. **Dynamic Tabs**: Support adding/removing tabs at runtime
3. **Tab Validation**: Prevent selection of disabled/hidden tabs

## Testing Strategy

### Unit Tests

1. **Widget Tests** (`Tests/UI/test_tab_dropdown.py`)
   - Initialization with various tab configurations
   - Label mapping correctness
   - Event emission on selection change
   - State synchronization

2. **Integration Tests**
   - Tab switching functionality
   - Window visibility management
   - Event handler connections
   - CSS class application

### Manual Testing Checklist

- [ ] Dropdown appears in correct position
- [ ] All tabs visible in dropdown
- [ ] Selection changes active window
- [ ] Visual feedback for current tab
- [ ] Keyboard navigation works
- [ ] No regression in tab functionality
- [ ] Performance comparable to tab bar
- [ ] Works with all themes

## Rollback Plan

1. **Feature Flag**: Disable via config
2. **Code Preservation**: Keep TabBar widget until deprecation
3. **Quick Revert**: Single line change in app.py
4. **Data Migration**: No persistent state changes required

## Success Metrics

1. **Screen Real Estate**: Measure additional vertical space gained
2. **User Feedback**: Survey on navigation preference
3. **Performance**: Compare render times and memory usage
4. **Bug Reports**: Track issues specific to dropdown navigation

## Timeline Estimate

- Phase 1 (Widget Creation): 2-3 hours
- Phase 2 (Integration): 1-2 hours
- Phase 3 (Styling): 1-2 hours
- Phase 4 (Testing & Polish): 2-3 hours
- **Total**: 6-10 hours of development time

## Future Enhancements

1. **Search Integration**: Add fuzzy search in dropdown
2. **Recent Tabs**: Show recently used tabs at top
3. **Grouping**: Organize tabs by category
4. **Icons**: Add visual icons to dropdown options
5. **Customization**: User-defined tab order/visibility