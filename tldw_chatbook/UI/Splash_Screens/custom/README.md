# Custom Splash Screen Effects

This directory is for user-contributed splash screen effects. To add your own effect:

## Quick Start

1. Create a new Python file in this directory (e.g., `my_awesome_effect.py`)
2. Import the base class and decorator:
   ```python
   from ..base_effect import BaseEffect, register_effect
   ```
3. Create your effect class:
   ```python
   @register_effect("my_awesome_effect")
   class MyAwesomeEffect(BaseEffect):
       def __init__(self, parent_widget, **kwargs):
           super().__init__(parent_widget, **kwargs)
           self.width = kwargs.get('width', 80)
           self.height = kwargs.get('height', 24)
       
       def update(self) -> Optional[str]:
           # Your animation logic here
           return "Your animated content"
   ```

## Effect Guidelines

- Effect names should be lowercase with underscores (e.g., `my_awesome_effect`)
- Classes should end with `Effect` (e.g., `MyAwesomeEffect`)
- Use the parent widget's width/height from kwargs
- Return styled text using Rich markup syntax
- Keep frame rates reasonable (update called ~20-30 times per second)

## Available Utilities

The `BaseEffect` class provides:
- `_grid_to_string(grid, style_grid)`: Convert 2D arrays to styled text
- `_add_centered_text(grid, style_grid, text, y, style)`: Add centered text
- `frame_count`: Current frame number
- `start_time`: Animation start time

## Example Template

```python
"""My awesome splash screen effect."""

import random
from typing import Optional
from ..base_effect import BaseEffect, register_effect


@register_effect("my_awesome_effect")
class MyAwesomeEffect(BaseEffect):
    """Brief description of your effect."""
    
    def __init__(self, parent_widget, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.title = kwargs.get('title', 'TLDW Chatbook')
        
    def update(self) -> Optional[str]:
        """Generate the next frame of animation."""
        # Create grids for characters and styles
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Your animation logic here
        # ...
        
        # Add title
        self._add_centered_text(grid, styles, self.title, self.height // 2, "bold cyan")
        
        # Convert to string and return
        return self._grid_to_string(grid, styles)
```

## Testing Your Effect

1. Add your effect name to your config file's `active_cards` list
2. Run the app with your effect: `splash_screen.card_selection = "my_awesome_effect"`
3. Or use the splash screen viewer in the app to preview it

## Contributing Back

If you'd like to share your effect:
1. Ensure it follows the guidelines above
2. Test it thoroughly
3. Consider moving it to an appropriate category folder
4. Submit a pull request!