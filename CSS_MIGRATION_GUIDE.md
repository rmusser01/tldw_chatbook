# CSS Modularization Migration Guide

## Overview
This guide provides step-by-step instructions for migrating from the monolithic `tldw_cli.tcss` to the new modular CSS architecture.

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
├── main.tcss (import aggregator)
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
│   └── chat.tcss
├── utilities/
│   ├── spacing.tcss
│   └── visibility.tcss
└── Themes/
    └── [theme files]
```

## Migration Steps

### 1. Update Application CSS Loading

In `app.py`, update the CSS loading to use the new main.tcss:

```python
# Old
CSS_PATH = "css/tldw_cli.tcss"

# New
CSS_PATH = "css/main.tcss"
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

Themes remain compatible. Update theme files to override modular styles:

```css
/* In theme file */
@import "../main.tcss";

/* Theme overrides */
Button {
    background: $my-theme-button-color;
}
```

## Benefits of Migration

1. **Faster Development**
   - Find styles quickly
   - Avoid conflicts
   - Clear organization

2. **Better Performance**
   - Smaller file sizes
   - Potential for lazy loading
   - Easier caching

3. **Team Collaboration**
   - Work on different modules
   - Reduced merge conflicts
   - Clear ownership

4. **Maintainability**
   - Logical organization
   - Easy to update
   - Clear dependencies

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

The migration includes the original CSS as a fallback:

```css
/* In main.tcss */
@import "./tldw_cli.tcss"; /* Temporary during migration */
```

To rollback:
1. Change CSS_PATH back to "css/tldw_cli.tcss"
2. Remove modular CSS imports
3. Revert widget changes

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

## Next Steps

After migration:
1. Remove old tldw_cli.tcss import from main.tcss
2. Delete or archive tldw_cli.tcss
3. Update documentation
4. Train team on new structure
5. Set up CSS linting rules