# Splash Screen Developer Guide

This guide explains the new modular splash screen system for developers and contributors.

## Overview

The splash screen system has been refactored from a monolithic file (`splash_animations.py`) into a modular, extensible structure. Each effect is now its own file, organized by category, making it easier to:

- Find specific effects
- Add new effects
- Maintain existing effects
- Contribute to the project

## Directory Structure

```
Utils/Splash_Screens/
├── __init__.py              # Auto-discovery and registration system
├── base_effect.py           # Base class and registration decorator
├── card_definitions.py      # All card configurations
├── classic/                 # Classic effects (matrix, glitch, typewriter, etc.)
│   ├── __init__.py
│   ├── matrix_rain.py
│   ├── glitch.py
│   └── ...
├── environmental/           # Nature/environment effects
│   ├── __init__.py
│   ├── starfield.py
│   ├── raindrops.py
│   └── ...
├── tech/                    # Technology/sci-fi effects
│   ├── __init__.py
│   ├── digital_rain.py
│   ├── terminal_boot.py
│   └── ...
├── gaming/                  # Gaming-inspired effects
│   ├── __init__.py
│   ├── pacman.py
│   ├── tetris.py
│   └── ...
├── psychedelic/            # Trippy/psychedelic effects
│   ├── __init__.py
│   ├── lava_lamp.py
│   ├── kaleidoscope.py
│   └── ...
└── custom/                 # User-contributed effects
    ├── __init__.py
    └── README.md           # Instructions for contributors
```

## Creating a New Effect

### 1. Choose the Right Category

Select the category that best fits your effect:
- **classic**: Traditional terminal effects, basic animations
- **environmental**: Nature, weather, physics simulations
- **tech**: Hacking, sci-fi, computer-themed
- **gaming**: Game references, achievements, retro gaming
- **psychedelic**: Trippy, colorful, mind-bending effects
- **custom**: For experimental or user-specific effects

### 2. Create Your Effect File

Create a new Python file in the appropriate category directory:

```python
"""Brief description of your effect."""

import random
import time
from typing import Optional, Any, List

from ..base_effect import BaseEffect, register_effect


@register_effect("your_effect_name")
class YourEffectNameEffect(BaseEffect):
    """Detailed description of what your effect does."""
    
    def __init__(
        self,
        parent_widget: Any,
        width: int = 80,
        height: int = 24,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', width)
        self.height = kwargs.get('height', height)
        
        # Initialize your effect's state here
        self.my_state = []
        
    def update(self) -> Optional[str]:
        """Generate the next frame of animation.
        
        Returns:
            Rich-formatted string for the current frame,
            or None if no update needed.
        """
        # Create your animation frame
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Your animation logic here
        # ...
        
        # Convert to string and return
        return self._grid_to_string(grid, styles)
```

### 3. Register Your Effect

The `@register_effect` decorator automatically registers your effect when the module is imported. The name you provide (e.g., "your_effect_name") is what users will reference in their configuration.

### 4. Add Card Definition

Add an entry to `card_definitions.py`:

```python
"your_effect_name": {
    "type": "animated",
    "effect": "your_effect_name",
    "title": "Your Effect Title",
    "subtitle": "Optional subtitle",
    "style": "white on black",
    "animation_speed": 0.05,
    # Add any custom parameters your effect uses
    "custom_param": "value"
}
```

## BaseEffect Utilities

The `BaseEffect` class provides helpful utilities:

### `_grid_to_string(grid, style_grid)`
Converts 2D character and style arrays into Rich-formatted text:
```python
grid = [['H', 'i'], [' ', '!']]
styles = [['bold red', 'bold blue'], [None, 'green']]
return self._grid_to_string(grid, styles)
```

### `_add_centered_text(grid, style_grid, text, y, style)`
Adds centered text to a grid at the specified y position:
```python
self._add_centered_text(grid, styles, "GAME OVER", 10, "bold red")
```

### Properties
- `frame_count`: Number of frames since effect started
- `start_time`: Timestamp when effect was created

## Rich Markup

Use Rich markup syntax for styling:
- Colors: `"red"`, `"green"`, `"rgb(255,128,0)"`, `"#ff8000"`
- Styles: `"bold"`, `"italic"`, `"underline"`, `"dim"`
- Combined: `"bold red on blue"`, `"italic cyan"`

Important: Escape square brackets in content:
```python
ESCAPED_OPEN_BRACKET = r'\['
ESCAPED_CLOSE_BRACKET = r'\]'
```

## Testing Your Effect

### 1. Quick Test Script

Create a test script to verify your effect works:

```python
from tldw_chatbook.Utils.Splash_Screens import load_all_effects, get_effect_class

# Load all effects
load_all_effects()

# Get your effect
EffectClass = get_effect_class("your_effect_name")

# Create instance
effect = EffectClass(None, width=80, height=24)

# Test a few frames
for i in range(10):
    frame = effect.update()
    if frame:
        print(frame)
        print("-" * 80)
    time.sleep(0.1)
```

### 2. Test in Splash Screen Viewer

1. Launch the app
2. Go to the splash screen viewer
3. Find your effect in the list
4. Preview it

### 3. Test as Default Splash

Add to your config file:
```toml
[splash_screen]
card_selection = "your_effect_name"
```

## Best Practices

1. **Performance**: Keep `update()` fast (<50ms) for smooth animation
2. **Memory**: Don't store large frame histories
3. **Terminal Size**: Handle various terminal sizes gracefully
4. **Colors**: Test on both light and dark terminals
5. **Frame Rate**: Respect the `animation_speed` parameter
6. **Rich Markup**: Always escape user content to prevent markup conflicts

## Common Patterns

### Time-Based Animation
```python
def update(self):
    elapsed = time.time() - self.start_time
    phase = elapsed * self.speed
    # Use phase for smooth animations
```

### Grid-Based Effects
```python
def update(self):
    grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
    styles = [[None for _ in range(self.width)] for _ in range(self.height)]
    
    # Modify grid
    for y in range(self.height):
        for x in range(self.width):
            grid[y][x] = self.calculate_char(x, y)
            styles[y][x] = self.calculate_style(x, y)
    
    return self._grid_to_string(grid, styles)
```

### Particle Systems
```python
def update(self):
    # Update particles
    for particle in self.particles:
        particle.update()
    
    # Remove dead particles
    self.particles = [p for p in self.particles if p.alive]
    
    # Spawn new particles
    if random.random() < self.spawn_rate:
        self.particles.append(self.create_particle())
    
    # Render
    return self.render_particles()
```

## Contributing

1. Fork the repository
2. Create your effect following this guide
3. Test thoroughly
4. Submit a pull request

Place experimental effects in `custom/` initially. After review and testing, they may be moved to an appropriate category.

## Migration from Old System

If you have an effect in the old `splash_animations.py`:

1. Extract the class definition
2. Create a new file in the appropriate category
3. Add imports and the `@register_effect` decorator
4. Update any relative imports
5. Test the migrated effect

## Troubleshooting

### Effect Not Found
- Ensure the file is in a recognized category directory
- Check that `@register_effect("name")` matches the name you're using
- Verify no import errors in your effect file

### Import Errors
- Use relative imports: `from ..base_effect import BaseEffect`
- Check all required dependencies are imported

### Style Not Applying
- Verify Rich markup syntax is correct
- Check for unescaped square brackets
- Test with simple colors first

## Examples

See these effects for reference implementations:
- **Simple**: `classic/typewriter.py` - Basic text reveal
- **Grid-based**: `classic/matrix_rain.py` - Falling characters
- **Time-based**: `classic/pulse.py` - Smooth color transitions
- **Complex**: `gaming/tetris.py` - Game simulation
- **Particle**: `environmental/fireworks.py` - Particle system

Happy animating! 🎨