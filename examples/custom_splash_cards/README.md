# Custom Splash Card Examples

This directory contains example configurations and code for creating custom splash screens in TLDW CLI.

## Files

### Configuration Examples

- **`cyberpunk_card.toml`** - A glitch-effect cyberpunk themed splash card
- **`minimalist_card.toml`** - Clean, simple design with typewriter effect  
- **`gaming_card.toml`** - Achievement-style gaming splash screen
- **`config_examples.toml`** - Multiple complete configuration examples for different use cases

### Code Examples

- **`custom_animation_effect.py`** - Python code showing how to create custom animation effects:
  - `FireEffect` - ASCII fire animation
  - `RainbowWaveEffect` - Rainbow colored wave animation
  - `ParticleEffect` - Particle system around content

## How to Use

### Using Configuration Files

1. **For individual splash cards**: Copy the `.toml` files to `~/.config/tldw_cli/splash_cards/`
   ```bash
   mkdir -p ~/.config/tldw_cli/splash_cards
   cp cyberpunk_card.toml ~/.config/tldw_cli/splash_cards/
   ```

2. **For config.toml settings**: Copy the desired section from `config_examples.toml` to your main config:
   ```bash
   # Edit ~/.config/tldw_cli/config.toml
   # Copy one of the [splash_screen_*] sections and rename to [splash_screen]
   ```

3. **Add to active cards**: Update your config.toml to include the new card:
   ```toml
   [splash_screen]
   active_cards = ["default", "matrix", "cyberpunk_card", "minimalist_card"]
   ```

### Using Custom Animation Effects

1. Copy the animation effect code to your project
2. Import it in `tldw_chatbook/Utils/splash_animations.py`
3. Add the effect to your splash card configuration
4. Update `_start_card_animation()` in `splash_screen.py` to handle the new effect

Example:
```python
# In splash_animations.py
from examples.custom_splash_cards.custom_animation_effect import FireEffect

# In splash_screen.py, add to _start_card_animation():
elif effect_type == "fire":
    self.effect_handler = FireEffect(self, content=content, **effect_params)
```

## Creating Your Own

### Basic Static Card
```toml
type = "static"
content = """
Your ASCII art here
"""
style = "bold white on blue"
```

### Animated Card
```toml
type = "animated"
effect = "typewriter"  # or "matrix_rain", "glitch", "retro_terminal"
content = "Your content"
animation_speed = 0.05
```

### Custom Effect Class Template
```python
class MyEffect(BaseEffect):
    def __init__(self, parent_widget, content, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.content = content
        # Initialize your effect state
    
    def update(self) -> Optional[str]:
        # Update animation state
        # Return the current frame as a string
        return self.render_frame()
```

## Tips

1. **Performance**: Keep animations simple - terminal rendering has limits
2. **Colors**: Use Textual color names or RGB values: `rgb(255,128,0)`
3. **Size**: Design for 80x24 terminal minimum
4. **Testing**: Test with different terminal sizes and themes
5. **Accessibility**: Provide static alternatives for animated cards

## Sharing Your Creations

Created an awesome splash card? Consider:
1. Submitting a PR to add it to the built-in cards
2. Sharing in the discussions/issues
3. Creating a splash card pack repository

Happy splashing! 🎨