# CSS Modularization Master Plan

## Project Overview
Modularizing the CSS/TCSS code for tldw_chatbook from a monolithic 3,809-line file into a maintainable, scalable module system.

## ⚠️ IMPORTANT UPDATES (2024-06-23)

1. **Textual does not support @import directives** - We implemented a custom CSS loader (`css_loader.py`) that concatenates modular files at runtime.

2. **Many CSS properties are not supported by Textual** - During migration, we discovered and fixed numerous compatibility issues (see CSS Fixes Applied section).

3. **Application is now running successfully** with the modular CSS system, using the custom loader and with all CSS compatibility issues resolved.

4. **Progress**: Phase 2 partially complete - 4 of 9 window styles extracted, all core components completed.

## Current State
- **Main File**: `tldw_chatbook/css/tldw_cli.tcss` (3,809 lines)
- **Embedded CSS**: 12+ Python files with `DEFAULT_CSS` strings
- **Theme System**: Separate theme files in `css/Themes/`

## Directory Structure
```
css/
├── css_loader.py         ✅ Created - Custom CSS loader (concatenates files)
├── base/
│   ├── reset.tcss        ✅ Created - Base resets and defaults
│   ├── variables.tcss    ✅ Created - CSS variables/design tokens
│   └── typography.tcss   ✅ Created - Font and text styles
├── components/
│   ├── buttons.tcss      ✅ Created - Button styles
│   ├── inputs.tcss       ✅ Created - Input, TextArea, Select styles
│   ├── sidebars.tcss     ✅ Created - Sidebar patterns
│   ├── messages.tcss     ✅ Created - Chat message components
│   ├── containers.tcss   ✅ Created - Layout containers
│   ├── lists.tcss        ✅ Created - ListView, ListItem styles
│   └── modals.tcss       ✅ Created - Modal/dialog styles
├── layouts/
│   ├── app-layout.tcss   ✅ Created - Main app structure
│   ├── window-base.tcss  ⏳ Pending - Base window styles
│   └── grid-layouts.tcss ⏳ Pending - Grid and flex patterns
├── windows/
│   ├── chat.tcss         ✅ Created - Chat window specific
│   ├── media.tcss        ✅ Created - Media window specific
│   ├── notes.tcss        ✅ Created - Notes window specific
│   ├── conv-char.tcss    ✅ Created - Conversations/Characters window
│   ├── tools.tcss        ✅ Created - Tools & Settings window
│   ├── llm-mgmt.tcss     ✅ Created - LLM Management window
│   ├── evals.tcss        ✅ Created - Evals window specific
│   ├── coding.tcss       ✅ Created - Coding window specific
│   └── logs.tcss         ✅ Created - Logs window specific
├── widgets/
│   ├── chat-message.tcss ⏳ Pending - Extract from chat_message.py
│   └── [other widgets]   ⏳ Pending - Extract from Python files
├── utilities/
│   ├── spacing.tcss      ✅ Created - Margin/padding utilities
│   ├── visibility.tcss   ✅ Created - Show/hide utilities
│   └── animations.tcss   ⏳ Pending - Transition/animation utilities
├── main.tcss             🚫 Not Used - @import not supported by Textual
└── tldw_cli.tcss         📦 Original - Monolithic CSS file (fallback)
```

## Migration Progress Tracker

### Phase 1: Setup Structure ✅
- [x] Create directory structure
- [x] Create variables.tcss with design tokens
- [x] Create reset.tcss with base resets
- [x] ~~Create main.tcss import file~~ Created CSS loader instead

### Phase 2: Extract Core Components ✅ COMPLETED
- [x] Typography styles
- [x] Button components
- [x] Input components (Input, TextArea, Select) - with CSS fixes
- [x] Sidebar patterns - with CSS fixes
- [x] List components
- [x] Message components - with CSS fixes
- [x] Container/layout components
- [x] Modal/dialog components

### Phase 3: Window-Specific Styles ✅ COMPLETED
- [x] Extract Chat window styles (lines ~97-198) - with CSS fixes
- [x] Extract Conv/Char/Prompts window styles (lines ~202-416)
- [x] Extract Logs window styles (lines ~421-449)
- [x] Extract Notes window styles (lines ~568-591)
- [x] Extract Tools & Settings styles (lines ~653-1033)
- [x] Extract LLM Management styles (lines ~1034-1366)
- [x] Extract Media window styles (lines ~1371-1546)
- [x] Extract Evals window styles (lines ~1555-1606 and ~1667-1715)
- [x] Extract Coding window styles (lines ~1611-1661)

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
1. ~~Textual's CSS parser may have limitations with @import~~ **CONFIRMED: Textual does not support @import**
2. Theme system integration needs careful handling
3. Some selectors are deeply nested and specific
4. Dynamic CSS generation in Python needs refactoring
5. Testing CSS modules in isolation
6. **NEW: Many standard CSS properties are not supported by Textual**

## Current Status Summary

### Completed ✅
1. **CSS Loader Implementation**: Created `css_loader.py` to concatenate modular CSS files
2. **Base Files**: 
   - variables.tcss - Design tokens
   - reset.tcss - Base resets
   - typography.tcss - Text styles
3. **Core Components** (ALL COMPLETED - Phase 2 ✅):
   - buttons.tcss - All button patterns
   - inputs.tcss - Form elements (with CSS fixes)
   - sidebars.tcss - Sidebar patterns (with CSS fixes)
   - lists.tcss - List components
   - messages.tcss - Chat messages (with CSS fixes)
   - containers.tcss - General-purpose layout containers
   - modals.tcss - Modal/dialog components
4. **Layouts**:
   - app-layout.tcss - Main app structure
5. **Utilities**:
   - spacing.tcss - Margin/padding utilities (with CSS fixes)
   - visibility.tcss - Show/hide utilities (with CSS fixes)
6. **Windows** (ALL COMPLETED - Phase 3 ✅):
   - chat.tcss - Chat window styles (with CSS fixes)
   - media.tcss - Media window styles
   - notes.tcss - Notes window styles
   - conv-char.tcss - Conversations/Characters/Prompts window styles
   - logs.tcss - Logs window styles
   - tools.tcss - Tools & Settings window styles (including Ingest)
   - llm-mgmt.tcss - LLM Management window styles
   - evals.tcss - Evaluations window styles (both old and new implementations)
   - coding.tcss - Coding window styles
7. **App Integration**: 
   - Updated app.py to use CSS loader instead of CSS_PATH
   - Application now runs successfully with modular CSS

### Next Steps
1. **Extract Widget CSS** (Phase 4):
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
1. **Import Strategy**: ~~Using @import in main.tcss~~ **CHANGED: Using custom CSS loader to concatenate files**
2. **Naming Convention**: BEM-like with utility prefixes (u-, l-, c-, w-)
3. **Organization**: Grouped by type (base, components, layouts, windows, utilities)
4. **Migration Path**: Gradual with original CSS as fallback
5. **CSS Compatibility**: Comment out unsupported CSS properties rather than delete

### Challenges Encountered
1. **File Size**: Original CSS too large to read in one go
2. **Complexity**: Deep nesting and specific selectors
3. **Dependencies**: Some styles depend on Textual internals
4. **Textual Limitations**: Many CSS properties not supported, requiring workarounds
5. **No @import Support**: Required building custom CSS loader
6. **CSS Syntax Differences**: Textual CSS is not standard CSS

### Benefits Realized
1. **Modularity**: Clear separation of concerns
2. **Reusability**: Common patterns extracted
3. **Maintainability**: Easy to find and update styles
4. **Scalability**: New features can be added easily

## CSS Fixes Applied

During migration, the following CSS properties had to be modified for Textual compatibility:

### Properties Removed/Commented Out:
- `position: absolute/relative` → Use `dock` instead
- `z-index` → Textual handles layering automatically
- `pointer-events: none` → Not supported
- `white-space: nowrap` → Not supported
- `font-family: monospace` → Not supported
- `font-size: 90%` → Not supported
- `gap: 1` → Use margin on children instead
- `clip: rect()` → Not supported
- `margin: auto` → Use `align: center middle` on parent
- All CSS animations and transitions
- `@media` queries
- `px` units → Use integer values

### Syntax Changes:
- `Input[type="number"]` → `Input.number`
- `border-color: $warning` → `border: solid $warning`
- Negative margins → Not supported

---
Last Updated: 2024-06-23
Status: Phase 2 & 3 Complete (All core components and windows extracted), App Running Successfully, Ready for Phase 4