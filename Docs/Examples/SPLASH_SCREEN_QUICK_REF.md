# TLDW Chatbook Splash Screen Quick Reference

## Quick Configuration Snippets

### Enable/Disable Splash Screen
```toml
[splash_screen]
enabled = false  # Completely disable splash screen
```

### Set Specific Splash Screen
```toml
[splash_screen]
card_selection = "matrix"  # Always use matrix effect
```

### Random Selection from Favorites
```toml
[splash_screen]
card_selection = "random"
active_cards = ["neural_network", "starfield", "circuit_trace", "plasma_field"]
```

### Fast Loading (Minimal Splash)
```toml
[splash_screen]
duration = 0.5
card_selection = "minimal"
fade_in_duration = 0.1
fade_out_duration = 0.1
```

### Extended Demo Mode
```toml
[splash_screen]
duration = 5.0
skip_on_keypress = false  # Force full duration
card_selection = "fractal_zoom"
```

## Common Customization Scenarios

### 1. Professional/Corporate Setup
```toml
[splash_screen]
card_selection = "random"
active_cards = ["blueprint", "terminal_boot", "loading_bar", "circuit_trace"]
duration = 2.0
show_progress = true

[splash_screen.effects]
fade_in_duration = 0.2
fade_out_duration = 0.2
```

### 2. Hacker/Security Theme
```toml
[splash_screen]
card_selection = "random"
active_cards = ["matrix", "glitch", "hacker_terminal", "binary_matrix", "digital_rain"]
duration = 2.5

# For matrix effect customization
[splash_screen.matrix]
style = "bold green on black"
animation_speed = 0.03
```

### 3. Scientific/Research
```toml
[splash_screen]
card_selection = "random"
active_cards = ["neural_network", "quantum_particles", "dna_sequence", "constellation_map", "fractal_zoom"]
duration = 3.0
```

### 4. Creative/Artistic
```toml
[splash_screen]
card_selection = "random"
active_cards = ["plasma_field", "ascii_fire", "old_film", "text_explosion", "pixel_zoom"]
duration = 2.5
```

### 5. Minimal/Fast
```toml
[splash_screen]
card_selection = "random"
active_cards = ["minimal", "minimal_fade", "ascii_spinner"]
duration = 1.0
show_progress = false
```

## Tips for Creating ASCII Art

### Size Guidelines
- Width: 60-80 characters (to fit most terminals)
- Height: 15-20 lines (leave room for progress bar)
- Center your art for best appearance

### Character Sets for Different Styles

**Box Drawing:**
```
┌─┬─┐  ╔═╦═╗  ╭─╮
│ │ │  ║ ║ ║  │ │
├─┼─┤  ╠═╬═╣  ├─┤
└─┴─┘  ╚═╩═╝  ╰─╯
```

**Shading:**
```
░░░ (light)
▒▒▒ (medium)
▓▓▓ (dark)
███ (solid)
```

**Decorative:**
```
◆ ◇ ◈ ○ ● ◯ ◉ ◊ ◦ • ∙
★ ☆ ✦ ✧ ✩ ✪ ✫ ✬ ✭ ✮ ✯
▲ △ ▼ ▽ ◀ ▶ ◁ ▷ ◄ ►
```

### ASCII Art Tools
1. **Online Generators**: 
   - patorjk.com/software/taag/ (Text to ASCII)
   - asciiart.eu (Large collection)
   - ascii-art-generator.org

2. **Manual Creation Tips**:
   - Use monospace font in your editor
   - Enable column selection for alignment
   - Test in terminal to check appearance

### Example: Creating a Logo
```
Step 1: Basic Text
TLDW

Step 2: Add Style
╔════╗
║TLDW║
╚════╝

Step 3: Enhance
╔═══════════╗
║  ╔═╦═╗    ║
║  ║ ║ ║    ║
║  ╩ ╩ ╩    ║
╚═══════════╝
```

## Performance Optimization

### For Slower Systems
```toml
[splash_screen]
# Use simple effects
active_cards = ["minimal", "loading_bar", "typewriter"]
# Reduce animation speed
animation_speed = 0.5
# Shorter duration
duration = 1.0
```

### For Remote/SSH Sessions
```toml
[splash_screen]
# Avoid complex animations
active_cards = ["default", "blueprint", "minimal"]
# Or disable entirely
enabled = false
```

## Debugging Splash Screens

### View Current Configuration
```bash
# Check your config file
cat ~/.config/tldw_cli/config.toml | grep -A 20 "\[splash_screen\]"
```

### Test Specific Card
```toml
[splash_screen]
card_selection = "test_card_name"  # Replace with card to test
duration = 10.0  # Longer duration for testing
skip_on_keypress = true  # Allow early exit
```

### Common Issues

1. **Splash screen not showing**
   - Check `enabled = true`
   - Verify card name exists
   - Check terminal size (minimum 80x24)

2. **Animation too fast/slow**
   - Adjust `animation_speed`
   - Modify effect-specific speed settings

3. **Colors not displaying**
   - Ensure terminal supports 256 colors
   - Try simpler color schemes

## Custom Card Installation

1. Create TOML file in `~/.config/tldw_cli/splash_cards/`
2. Add to active_cards list:
   ```toml
   active_cards = ["default", "matrix", "my_custom_card"]
   ```
3. Restart application to test

## Quick Copy-Paste Templates

### Static Card Template
```toml
[card]
name = "my_static"
type = "static"
content = """
YOUR ASCII ART HERE
"""
style = "bold white on black"
```

### Animated Card Template
```toml
[card]
name = "my_animated"
type = "animated"
effect = "typewriter"  # or any effect name
content = "Your text here"
animation_speed = 0.05
style = "bold cyan on black"
```