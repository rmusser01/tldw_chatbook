# CSS Modularization Migration Guide

## Overview
This guide provides step-by-step instructions for migrating from the monolithic `tldw_cli.tcss` to the new modular CSS architecture.

**IMPORTANT UPDATE**: Textual does not support CSS @import directives. We've implemented a custom CSS loader that concatenates modular files at runtime.

## What's Changed

### Old Structure
```
css/
├── tldw_cli.tcss (3,809 lines - all styles)
└── Themes/
    └── [theme files]
```

### New Structure
```
css/
├── css_loader.py (Custom loader for concatenating CSS files)
├── main.tcss (DEPRECATED - not used due to lack of @import support)
├── base/
│   ├── reset.tcss
│   ├── variables.tcss
│   └── typography.tcss
├── components/
│   ├── buttons.tcss
│   ├── inputs.tcss
│   ├── sidebars.tcss
│   ├── messages.tcss
│   └── lists.tcss
├── layouts/
│   └── app-layout.tcss
├── windows/
│   ├── chat.tcss
│   ├── media.tcss
│   ├── notes.tcss
│   └── conv-char.tcss
├── utilities/
│   ├── spacing.tcss
│   └── visibility.tcss
└── Themes/
    └── [theme files]
```

## Migration Steps

### 1. Update Application CSS Loading

Due to Textual's lack of @import support, we use a custom CSS loader. In `app.py`:

```python
# Old approach (doesn't work with Textual)
CSS_PATH = "css/main.tcss"

# New approach - using CSS loader
from tldw_chatbook.css.css_loader import load_modular_css
from pathlib import Path

class TldwCli(App[None]):
    # Load modular CSS at class definition time
    CSS = load_modular_css(Path(__file__).parent / "css", use_fallback=True)
```

### 2. Update Widget CSS References

For widgets with embedded CSS, gradually extract to separate files:

```python
# Old approach in widget.py
class MyWidget(Widget):
    DEFAULT_CSS = """
    MyWidget {
        width: 100%;
        height: auto;
    }
    """

# New approach
class MyWidget(Widget):
    # CSS now in css/widgets/my-widget.tcss
    pass
```

### 3. Update Custom Styles

If you have custom styles, migrate them to appropriate modules:

- **Component overrides** → Add to specific component file
- **Window-specific styles** → Add to window file
- **Utility classes** → Add to utilities
- **New components** → Create new component file

### 4. Theme Compatibility

Themes remain compatible through Textual's built-in theme system. Themes do not need to import the main CSS as they are applied as overrides automatically by Textual.

## CSS Loader Implementation

The `css_loader.py` module provides the functionality to concatenate modular CSS files:

```python
from tldw_chatbook.css.css_loader import CSSLoader
from pathlib import Path

# Create loader instance
loader = CSSLoader(Path("css"))

# Load all CSS files in correct order
css_content = loader.load_css_files(use_fallback=True)

# Or save combined CSS for debugging
loader.save_combined_css(Path("combined.tcss"))
```

The loader:
- Maintains correct import order (variables → base → components → windows → utilities)
- Includes helpful comments showing which file each section comes from
- Optionally includes the original CSS as fallback
- Logs which files are loaded successfully

## Benefits of Migration

1. **Faster Development**
   - Find styles quickly
   - Avoid conflicts
   - Clear organization

2. **Better Performance**
   - Smaller file sizes per module
   - Only load what's needed
   - Easier to optimize

3. **Team Collaboration**
   - Work on different modules
   - Reduced merge conflicts
   - Clear ownership

4. **Maintainability**
   - Logical organization
   - Easy to update
   - Clear dependencies
   - Easier to identify and fix Textual compatibility issues

## Common Patterns

### Using Component Classes

```css
/* Instead of repeating styles */
#my-sidebar {
    dock: left;
    width: 25%;
    min-width: 20;
    /* ... etc ... */
}

/* Use component class */
#my-sidebar {
    /* Inherits from .sidebar in sidebars.tcss */
    /* Add only specific overrides */
}
```

### Combining Utilities

```css
/* HTML/Textual */
<Button class="button-primary u-mb-2 u-px-4">
    Save Changes
</Button>
```

### Window-Specific Overrides

```css
/* In windows/my-window.tcss */
#my-window .sidebar {
    /* Inherits base sidebar styles */
    width: 30%; /* Window-specific override */
}
```

## Testing Your Migration

1. **Visual Regression Testing**
   - Screenshot before migration
   - Screenshot after migration
   - Compare for differences

2. **Component Testing**
   - Test each component in isolation
   - Verify inheritance works
   - Check specificity

3. **Theme Testing**
   - Switch between themes
   - Verify overrides work
   - Check color inheritance

## Rollback Plan

The CSS loader includes the original CSS as a fallback by default:

```python
# In css_loader.py
css_content = loader.load_css_files(use_fallback=True)
```

To rollback:
1. Change app.py back to use CSS_PATH:
   ```python
   CSS_PATH = str(Path(__file__).parent / "css/tldw_cli.tcss")
   ```
2. Remove the CSS loader import and usage
3. Revert any widget CSS extractions

## FAQ

### Q: Will this break my custom themes?
A: No, themes will continue to work. You may want to update them to use the modular structure for better organization.

### Q: Can I migrate gradually?
A: Yes! The main.tcss includes the original CSS as a fallback, allowing gradual migration.

### Q: How do I add new styles?
A: 
1. Identify the type (component, window, utility)
2. Add to appropriate file
3. Follow naming conventions
4. Document complex styles

### Q: What about performance?
A: Initial load may be slightly slower due to multiple imports, but overall performance improves due to better organization and potential for optimization.

## Support

For questions or issues during migration:
1. Check this guide
2. Review the CSS_MODULARIZATION_PLAN.md
3. Look at existing modules for examples
4. Test in isolation first

## Textual CSS Limitations

During migration, we discovered several CSS features that Textual does not support:

### Unsupported CSS Properties
- `position` (absolute, relative, fixed)
- `z-index` 
- `pointer-events`
- `white-space`
- `font-family`
- `font-size`
- `gap` (use margin on children instead)
- `clip`
- `border-color` (use full `border` property)

### Unsupported CSS Features
- `@import` directives
- `@keyframes` and animations
- `@media` queries
- CSS transitions
- Pseudo-elements (::before, ::after)
- Attribute selectors (`Input[type="number"]`)
- Function notation in values (`rect()`, `calc()`)
- `px` units (use integers for character-based units)
- `margin: auto` (use `align` on parent container)
- Negative margins

### Workarounds
- **For @import**: Use our custom CSS loader
- **For animations**: Handle in Python code
- **For attribute selectors**: Use CSS classes (`Input.number`)
- **For border-color**: Use `border: solid $color`
- **For centering**: Use `align: center middle` on parent

## Next Steps

After migration:
1. Set `use_fallback=False` in CSS loader once all styles are migrated
2. Delete or archive tldw_cli.tcss
3. Update documentation
4. Complete extraction of remaining window styles
5. Extract widget CSS from Python files