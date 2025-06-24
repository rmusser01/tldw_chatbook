# CSS Modularization Master Plan

## Project Overview
Modularizing the CSS/TCSS code for tldw_chatbook from a monolithic 3,809-line file into a maintainable, scalable module system.

## Current State
- **Main File**: `tldw_chatbook/css/tldw_cli.tcss` (3,809 lines)
- **Embedded CSS**: 12+ Python files with `DEFAULT_CSS` strings
- **Theme System**: Separate theme files in `css/Themes/`

## Directory Structure
```
css/
├── base/
│   ├── reset.tcss        ✅ Created - Base resets and defaults
│   ├── variables.tcss    ✅ Created - CSS variables/design tokens
│   └── typography.tcss   ✅ Created - Font and text styles
├── components/
│   ├── buttons.tcss      ✅ Created - Button styles
│   ├── inputs.tcss       ✅ Created - Input, TextArea, Select styles
│   ├── sidebars.tcss     ✅ Created - Sidebar patterns
│   ├── messages.tcss     ⏳ Pending - Chat message components
│   ├── containers.tcss   ⏳ Pending - Layout containers
│   ├── lists.tcss        ✅ Created - ListView, ListItem styles
│   └── modals.tcss       ⏳ Pending - Modal/dialog styles
├── layouts/
│   ├── app-layout.tcss   ✅ Created - Main app structure
│   ├── window-base.tcss  ⏳ Pending - Base window styles
│   └── grid-layouts.tcss ⏳ Pending - Grid and flex patterns
├── windows/
│   ├── chat.tcss         ⏳ Pending - Chat window specific
│   ├── media.tcss        ⏳ Pending - Media window specific
│   ├── notes.tcss        ⏳ Pending - Notes window specific
│   ├── conv-char.tcss    ⏳ Pending - Conversations/Characters window
│   ├── tools.tcss        ⏳ Pending - Tools & Settings window
│   ├── llm-mgmt.tcss     ⏳ Pending - LLM Management window
│   ├── evals.tcss        ⏳ Pending - Evals window specific
│   ├── coding.tcss       ⏳ Pending - Coding window specific
│   └── logs.tcss         ⏳ Pending - Logs window specific
├── widgets/
│   ├── chat-message.tcss ⏳ Pending - Extract from chat_message.py
│   └── [other widgets]   ⏳ Pending - Extract from Python files
├── utilities/
│   ├── spacing.tcss      ⏳ Pending - Margin/padding utilities
│   ├── visibility.tcss   ⏳ Pending - Show/hide utilities
│   └── animations.tcss   ⏳ Pending - Transition/animation utilities
└── main.tcss             ⏳ Pending - Import aggregator
```

## Migration Progress Tracker

### Phase 1: Setup Structure ✅
- [x] Create directory structure
- [x] Create variables.tcss with design tokens
- [x] Create reset.tcss with base resets
- [ ] Create main.tcss import file

### Phase 2: Extract Core Components ✅
- [x] Typography styles
- [x] Button components
- [x] Input components (Input, TextArea, Select)
- [x] Sidebar patterns
- [x] List components
- [x] Message components
- [ ] Container/layout components
- [ ] Modal/dialog components

### Phase 3: Window-Specific Styles
- [ ] Extract Chat window styles (lines ~97-198)
- [ ] Extract Conv/Char/Prompts window styles (lines ~202-416)
- [ ] Extract Logs window styles (lines ~421-449)
- [ ] Extract Notes window styles (lines ~568-591)
- [ ] Extract Tools & Settings styles (lines ~653-1033)
- [ ] Extract LLM Management styles (lines ~1034-1366)
- [ ] Extract Media window styles (lines ~1371-1546)
- [ ] Extract Evals window styles (lines ~1555-1606)
- [ ] Extract Coding window styles (lines ~1611-1661)

### Phase 4: Widget CSS Migration
Python files with DEFAULT_CSS to extract:
- [ ] `Widgets/chat_message.py`
- [ ] `Widgets/chat_message_enhanced.py`
- [ ] `Widgets/notes_sidebar_left.py`
- [ ] `Widgets/notes_sidebar_right.py`
- [ ] `Widgets/notes_sync_widget.py`
- [ ] `Widgets/enhanced_file_picker.py`
- [ ] `UI/Notes_Window.py`
- [ ] `UI/Tools_Settings_Window.py`
- [ ] `UI/Chat_Window_Enhanced.py`
- [ ] `UI/Logs_Window.py`
- [ ] `Third_Party/textual_fspicker/base_dialog.py`
- [ ] `Third_Party/textual_fspicker/file_dialog.py`

### Phase 5: Optimization & Cleanup
- [ ] Remove duplicate styles
- [ ] Standardize naming conventions
- [ ] Add CSS documentation
- [ ] Create migration guide
- [ ] Update app.py to use new CSS structure
- [ ] Test all functionality
- [ ] Archive old tldw_cli.tcss

## Common Patterns Identified

### Sidebar Pattern
Used in: Chat, Conv/Char/Prompts, Notes windows
```css
.sidebar {
    dock: left/right;
    width: 25%;
    min-width: 20;
    max-width: 80;
    background: $boost;
    padding: 1 2;
    border-[right/left]: thick $background-darken-1;
    height: 100%;
    overflow-y: auto;
    overflow-x: hidden;
}
```

### Input Patterns
Common styles for inputs across all windows:
```css
width: 100%;
margin-bottom: 1;
border: round $surface/$primary-lighten-2;
```

### Button Patterns
Common button styles:
```css
width: 100%/1fr;
margin-bottom: 1;
height: 3;
```

### Collapsible Pattern
Used for expandable sections in sidebars

### Message/Card Pattern
Used for chat messages, notes cards, etc.

## Naming Convention Guidelines

### BEM-like Structure
- Block: `.chat-message`
- Element: `.chat-message__header`
- Modifier: `.chat-message--user`

### Prefixes
- `u-` for utilities: `.u-hidden`, `.u-margin-top-1`
- `l-` for layout: `.l-sidebar`, `.l-grid`
- `c-` for components: `.c-button`, `.c-input`
- `w-` for window-specific: `.w-chat-sidebar`

### Semantic Names
- Use role-based names: `.sidebar-primary` not `.left-sidebar`
- Use state names: `.is-active`, `.is-collapsed`
- Use purpose names: `.action-buttons` not `.bottom-buttons`

## CSS Variable Naming
- Colors: `$color-{name}-{modifier}`
- Spacing: `$spacing-{size}`
- Dimensions: `${component}-{property}`
- Z-index: `$z-{level}`

## Implementation Notes

### Import Order in main.tcss
1. Variables and tokens
2. Reset and base styles
3. Typography
4. Layout systems
5. Components (smallest to largest)
6. Window-specific overrides
7. Utilities
8. Theme overrides

### Testing Strategy
1. Visual regression testing
2. Component isolation testing
3. Cross-window compatibility
4. Theme switching validation
5. Performance benchmarking

### Backward Compatibility
- Keep original file during migration
- Use feature flags for gradual rollout
- Maintain CSS selector compatibility
- Document breaking changes

## Benefits Tracking
- [ ] Reduced file size per module
- [ ] Faster development cycles
- [ ] Easier debugging
- [ ] Better team collaboration
- [ ] Improved performance
- [ ] Enhanced maintainability

## Known Issues/Challenges
1. Textual's CSS parser may have limitations with @import
2. Theme system integration needs careful handling
3. Some selectors are deeply nested and specific
4. Dynamic CSS generation in Python needs refactoring
5. Testing CSS modules in isolation

## Current Status Summary

### Completed ✅
1. **Directory Structure**: Full module structure created
2. **Base Files**: 
   - variables.tcss - Design tokens
   - reset.tcss - Base resets
   - typography.tcss - Text styles
3. **Core Components**:
   - buttons.tcss - All button patterns
   - inputs.tcss - Form elements
   - sidebars.tcss - Sidebar patterns
   - lists.tcss - List components
   - messages.tcss - Chat messages
4. **Layouts**:
   - app-layout.tcss - Main app structure
5. **Utilities**:
   - spacing.tcss - Margin/padding utilities
   - visibility.tcss - Show/hide utilities
6. **Windows** (Started):
   - chat.tcss - Chat window example
7. **Documentation**:
   - main.tcss - Import aggregator
   - CSS_MIGRATION_GUIDE.md - Migration instructions

### Next Steps
1. **Complete Window Extractions**:
   - Extract remaining window-specific styles
   - Create window CSS files for each tab
   
2. **Extract Widget CSS**:
   - Move DEFAULT_CSS from Python files
   - Create widget-specific CSS modules
   
3. **Testing & Integration**:
   - Test with Textual's CSS parser
   - Verify @import functionality
   - Update app.py to use main.tcss
   
4. **Optimization**:
   - Remove duplicates
   - Optimize import order
   - Add CSS compression

5. **Documentation**:
   - Component usage examples
   - Naming convention guide
   - Best practices document

## Implementation Notes

### Key Decisions Made
1. **Import Strategy**: Using @import in main.tcss with fallback to original
2. **Naming Convention**: BEM-like with utility prefixes (u-, l-, c-, w-)
3. **Organization**: Grouped by type (base, components, layouts, windows, utilities)
4. **Migration Path**: Gradual with original CSS as fallback

### Challenges Encountered
1. **File Size**: Original CSS too large to read in one go
2. **Complexity**: Deep nesting and specific selectors
3. **Dependencies**: Some styles depend on Textual internals

### Benefits Realized
1. **Modularity**: Clear separation of concerns
2. **Reusability**: Common patterns extracted
3. **Maintainability**: Easy to find and update styles
4. **Scalability**: New features can be added easily

---
Last Updated: [Current Date]
Status: Phase 2 Complete, Phase 3 Ready to Begin