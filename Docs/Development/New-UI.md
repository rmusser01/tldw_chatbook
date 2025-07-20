# ASCII-Style UI Design for tldw_chatbook

## Overview

This document outlines the transformation of tldw_chatbook's UI from modern flat/blocky buttons to a terminal-inspired ASCII aesthetic. The goal is to create a cohesive, retro-computing feel that enhances the terminal experience while maintaining excellent usability.

## Design Principles

### 1. **Meaningful Symbolism**
- Every ASCII character should convey meaning
- Consistent use of symbols across the application
- Clear visual hierarchy through character weight

### 2. **Terminal Authenticity**
- Embrace the constraints of text-based interfaces
- Use box-drawing characters for structure
- Leverage ASCII art for visual interest

### 3. **Functional Beauty**
- ASCII elements should enhance, not hinder, usability
- Clear state indicators (hover, active, disabled)
- Maintain accessibility standards

### 4. **Color as Enhancement**
- Use color to complement ASCII structure
- Ensure UI is readable in monochrome
- Consistent color coding for states and roles

## ASCII Character Reference

### Box-Drawing Characters

#### Single-Line Boxes
```
┌─────┬─────┐
│     │     │
├─────┼─────┤
│     │     │
└─────┴─────┘
```
Characters: `┌ ┐ └ ┘ ─ │ ├ ┤ ┬ ┴ ┼`

#### Double-Line Boxes
```
╔═════╦═════╗
║     ║     ║
╠═════╬═════╣
║     ║     ║
╚═════╩═════╝
```
Characters: `╔ ╗ ╚ ╝ ═ ║ ╠ ╣ ╦ ╩ ╬`

#### Rounded Boxes
```
╭─────┬─────╮
│     │     │
├─────┼─────┤
│     │     │
╰─────┴─────╯
```
Characters: `╭ ╮ ╰ ╯`

### Status and Progress Indicators

#### Checkboxes and Radio Buttons
```
[ ] Unchecked     [x] Checked      [■] Selected
( ) Unselected    (•) Selected     (○) Alternative
```

#### Progress Indicators
```
Static:  [████████░░] 80%
Spinner: ⣾⣽⣻⢿⡿⣟⣯⣷ (animated)
Blocks:  ▏▎▍▌▋▊▉█
```

#### Status Symbols
```
✓ Success/Complete    ✗ Error/Failed      ⚠ Warning
ℹ Information        ⚡ Action            ◆ Active
▸ Collapsed          ▾ Expanded          → Navigation
```

### Decorative Elements

#### Separators
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  Heavy
────────────────────────────────  Normal
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  Dashed
································  Dotted
```

#### Arrows and Pointers
```
← ↑ → ↓  Basic arrows
◀ ▲ ▶ ▼  Filled arrows
« ‹ › »  Chevrons
⟨ ⟩      Angle brackets
```

## Component Redesigns

### 1. Buttons

#### Current (Blocky)
```css
Button {
    background: $primary;
    border: none;
}
```

#### New (ASCII)
```
Normal:    ┌─────────┐     Hover:     ┌═════════┐
           │  Click  │                │  Click  │
           └─────────┘                └═════════┘

Pressed:   ╔═════════╗     Disabled:  ┌╌╌╌╌╌╌╌╌╌┐
           ║  Click  ║                │  Click  │
           ╚═════════╝                └╌╌╌╌╌╌╌╌╌┘
```

### 2. Input Fields

```
Label:
┌─────────────────────────┐
│ Enter text here...      │
└─────────────────────────┘

Focused:
╔═════════════════════════╗
║ Enter text here...█     ║
╚═════════════════════════╝
```

### 3. Tab Bar

```
┌─────────┬───────────┬──────────┬───────────┐
│  Chat   │  Notes    │ Settings │   Help    │
├─────────┴───────────┴──────────┴───────────┤
│                                             │
│  Tab Content Area                           │
│                                             │
└─────────────────────────────────────────────┘

Active Tab:
╔═════════╗─────────┬──────────┬───────────┐
║  Chat   ║  Notes  │ Settings │   Help    │
╚═════════╩─────────┴──────────┴───────────┘
```

### 4. Message Bubbles

```
User Message:
┌─[You]────────────────────┐
│ Hello, how are you?      │
└──────────────────────────┘

Assistant Message:
╔═[Assistant]══════════════╗
║ I'm doing well, thanks!  ║
╚══════════════════════════╝
```

### 5. Lists and Trees

```
File Tree:
├── Documents/
│   ├── report.pdf
│   └── notes.txt
├── Images/
│   ├── photo1.jpg
│   └── photo2.png
└── README.md

Menu List:
┌─── Main Menu ───┐
│ ▸ New Chat      │
│ ▸ Load Chat     │
│ ▸ Settings      │
│ ──────────────  │
│ ▸ Exit          │
└─────────────────┘
```

### 6. Progress Bars

```
Simple:
[████████████████████░░░░░░░░░░] 67%

Detailed:
┌─ Processing ────────────────────────┐
│ [████████████████░░░░░░░░░░░░] 60% │
│ 600/1000 items                      │
└─────────────────────────────────────┘
```

### 7. Modals/Dialogs

```
╔═══════════════════════════════════╗
║          Confirm Action           ║
╠═══════════════════════════════════╣
║                                   ║
║  Are you sure you want to delete  ║
║  this conversation?               ║
║                                   ║
║  ┌─────────┐    ┌─────────┐      ║
║  │ Cancel  │    │   OK    │      ║
║  └─────────┘    └─────────┘      ║
╚═══════════════════════════════════╝
```

## Implementation Steps

### Phase 1: Foundation (Week 1)
1. **Create ASCII utility module** (`tldw_chatbook/Utils/ascii_ui.py`)
   - Box drawing helper functions
   - Character constants
   - Style mappings

2. **Update base CSS** 
   - Add ASCII-specific CSS variables
   - Create border style classes
   - Define hover/active states

3. **Create ASCII button widget**
   - Extend existing Button class
   - Implement state rendering
   - Add hover effects

### Phase 2: Core Components (Week 2)
1. **Convert primary buttons**
   - Chat input submit button
   - Navigation buttons
   - Action buttons

2. **Update input fields**
   - Text inputs with ASCII borders
   - Text areas with ASCII frames
   - Search boxes

3. **Redesign tab bar**
   - ASCII tab separators
   - Active tab indicators
   - Tab switching animations

### Phase 3: Content Areas (Week 3)
1. **Message display**
   - ASCII message bubbles
   - Role-based styling
   - Timestamp formatting

2. **List components**
   - File trees with ASCII branches
   - Menu items with indicators
   - Selection highlighting

3. **Progress indicators**
   - Loading bars
   - Spinners
   - Status messages

### Phase 4: Advanced Features (Week 4)
1. **Modal dialogs**
   - ASCII window frames
   - Shadow effects using characters
   - Focus management

2. **Data displays**
   - Tables with ASCII borders
   - Charts using ASCII art
   - Statistics panels

3. **Special effects**
   - ASCII animations
   - Transition effects
   - Loading states

### Phase 5: Polish & Themes (Week 5)
1. **Theme system**
   - Classic terminal theme
   - Modern ASCII theme
   - High contrast theme

2. **Accessibility**
   - Screen reader compatibility
   - Keyboard navigation
   - Focus indicators

3. **Performance optimization**
   - Character caching
   - Render optimization
   - Smooth transitions

## Code Examples

### ASCII Button Widget

```python
# tldw_chatbook/Widgets/ascii_button.py
from textual.widgets import Button
from textual.reactive import reactive

class ASCIIButton(Button):
    """Button with ASCII box-drawing borders."""
    
    pressed = reactive(False)
    
    def __init__(self, label: str, variant: str = "single", **kwargs):
        super().__init__(label, **kwargs)
        self.variant = variant  # "single", "double", "rounded"
        
    def render_border(self) -> str:
        """Render ASCII border based on state."""
        if self.pressed:
            return "double"
        elif self.mouse_over:
            return "thick"
        else:
            return self.variant
```

### CSS for ASCII Buttons

```css
/* tldw_chatbook/css/components/_ascii_buttons.tcss */

ASCIIButton {
    border: ascii;
    padding: 0 2;
    height: 3;
    min-width: 10;
    text-align: center;
}

ASCIIButton:hover {
    border: thick;
    color: $primary-lighten-1;
}

ASCIIButton:focus {
    border: double;
    color: $primary;
}

ASCIIButton.-pressed {
    border: double;
    color: $primary-darken-1;
}

ASCIIButton.-disabled {
    border: ascii;
    color: $text-disabled;
    text-style: dim;
}
```

### ASCII Progress Bar

```python
# tldw_chatbook/Widgets/ascii_progress.py
class ASCIIProgressBar(Widget):
    """Progress bar using ASCII characters."""
    
    progress = reactive(0.0)  # 0.0 to 1.0
    
    def render(self) -> str:
        width = self.size.width - 2  # Account for borders
        filled = int(width * self.progress)
        empty = width - filled
        
        bar = "█" * filled + "░" * empty
        percentage = f"{int(self.progress * 100)}%"
        
        return f"[{bar}] {percentage}"
```

## Theme Variations

### 1. Classic Terminal
- Single-line boxes only
- Monochrome with limited colors
- Simple ASCII characters (no Unicode)

### 2. Modern Terminal
- Mix of single and double-line boxes
- Full color palette
- Extended Unicode characters

### 3. Retro Computing
- Double-line boxes prominently
- CGA/EGA color schemes
- DOS-style interface elements

### 4. Minimalist
- Sparse use of borders
- Focus on whitespace
- Subtle ASCII accents

## Migration Strategy

### 1. Gradual Rollout
- Start with non-critical components
- A/B test with users
- Gather feedback iteratively

### 2. Feature Flags
```python
# config.toml
[ui]
ascii_mode = true
ascii_theme = "modern"
```

### 3. Compatibility Mode
- Support both UI styles initially
- Allow users to switch
- Deprecate old style over time

### 4. Documentation Updates
- Update screenshots
- Create ASCII UI guide
- Update keyboard shortcuts

## Testing Considerations

### 1. Terminal Compatibility
- Test on various terminal emulators
- Verify character rendering
- Check color support

### 2. Performance Testing
- Measure render times
- Profile memory usage
- Test with large datasets

### 3. Accessibility Testing
- Screen reader compatibility
- Keyboard navigation
- High contrast modes

## Future Enhancements

### 1. ASCII Art Integration
- Splash screens with ASCII art
- Decorative headers
- Custom logos

### 2. Animation Library
- Text-based animations
- Transition effects
- Loading animations

### 3. Custom Widgets
- ASCII charts/graphs
- Terminal games
- Interactive diagrams

## Conclusion

This ASCII-style UI transformation will give tldw_chatbook a distinctive, terminal-native aesthetic that sets it apart from typical TUI applications. By carefully implementing these changes in phases, we can ensure a smooth transition while maintaining the application's functionality and user experience.

The key is to balance nostalgia with usability, creating an interface that feels both familiar to terminal users and fresh in its execution.