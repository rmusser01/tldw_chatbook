# TLDW Chatbook Splash Screens Catalog

This document provides a comprehensive catalog of all available splash screens in TLDW Chatbook, including their configuration options and visual characteristics.

## Table of Contents
1. [Overview](#overview)
2. [Configuration Basics](#configuration-basics)
3. [Static Splash Screens](#static-splash-screens)
4. [Classic Animated Effects](#classic-animated-effects)
5. [Visual Effects](#visual-effects)
6. [Interactive Animations](#interactive-animations)
7. [Tech-Themed Animations](#tech-themed-animations)
8. [Creative Effects](#creative-effects)
9. [New Animated Effects](#new-animated-effects)

## Overview

TLDW Chatbook includes over 40 built-in splash screens that can be displayed during application startup. These range from simple static ASCII art to complex animated effects inspired by classic computer interfaces, sci-fi movies, and modern UI patterns.

## Configuration Basics

To configure splash screens, edit your `~/.config/tldw_cli/config.toml`:

```toml
[splash_screen]
enabled = true
duration = 2.5  # seconds
card_selection = "random"  # or specific card name
show_progress = true
skip_on_keypress = true

# List of cards to randomly select from
active_cards = ["matrix", "glitch", "starfield", "neural_network"]
```

## Static Splash Screens

### 1. default
- **Type**: Static
- **Description**: Classic TLDW ASCII art logo in a box frame
- **Style**: Bold white on black
- **Best for**: Traditional, professional appearance

### 2. blueprint
- **Type**: Static
- **Description**: Technical blueprint-style diagram showing system architecture
- **Style**: Cyan on dark blue background
- **Best for**: Technical documentation, engineering themes

### 3. classic
- **Type**: Static (from Splash.py)
- **Description**: Original TLDW ASCII art with "too long; didn't watch" spelled out
- **Best for**: Nostalgic, retro computing feel

### 4. compact
- **Type**: Static (from Splash.py)
- **Description**: Compact box-style logo with Unicode characters
- **Best for**: Clean, modern terminals

### 5. minimal
- **Type**: Static (from Splash.py)
- **Description**: Minimalist design using simple line characters
- **Best for**: Fast loading, minimal visual impact

## Classic Animated Effects

### 6. matrix
- **Type**: Animated
- **Effect**: matrix_rain
- **Description**: Matrix-style falling green characters
- **Configuration**:
  ```toml
  title = "tldw chatbook"
  subtitle = "Loading user interface..."
  style = "bold green on black"
  animation_speed = 0.05
  ```
- **Best for**: Sci-fi themes, hacker aesthetic

### 7. glitch
- **Type**: Animated
- **Effect**: glitch
- **Description**: Glitching text effect with random character corruption
- **Configuration**:
  ```toml
  glitch_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
  animation_speed = 0.1
  ```
- **Best for**: Cyberpunk themes, system errors

### 8. retro
- **Type**: Animated
- **Effect**: retro_terminal
- **Description**: Old CRT monitor effect with scanlines and phosphor glow
- **Configuration**:
  ```toml
  scanline_speed = 0.02
  phosphor_glow = true
  ```
- **Best for**: Retro computing, nostalgia

### 9. typewriter
- **Type**: Animated
- **Effect**: typewriter
- **Description**: Text appears character by character like typing
- **Configuration**:
  ```toml
  animation_speed = 0.08  # Controls typing speed
  ```
- **Best for**: Minimal fade-in effects, storytelling

## Visual Effects

### 10. tech_pulse
- **Type**: Animated
- **Effect**: pulse
- **Description**: Pulsing brightness effect on tech-themed ASCII art
- **Configuration**:
  ```toml
  pulse_speed = 0.5  # Cycles per second
  min_brightness = 80
  max_brightness = 200
  color = [100, 180, 255]  # Light blue
  ```

### 11. code_scroll
- **Type**: Animated
- **Effect**: code_scroll
- **Description**: Scrolling code lines with highlighted title
- **Configuration**:
  ```toml
  scroll_speed = 0.1
  num_code_lines = 18
  code_line_style = "dim blue"
  title_style = "bold yellow"
  ```

### 12. minimal_fade
- **Type**: Animated
- **Effect**: typewriter
- **Description**: Slow reveal of minimal text
- **Best for**: Clean, professional applications

### 13. arcade_high_score
- **Type**: Animated
- **Effect**: blink
- **Description**: Arcade-style display with blinking "INSERT COIN" text
- **Configuration**:
  ```toml
  blink_speed = 0.5
  blink_targets = ["LOADING...", "PRESS ANY KEY TO START!"]
  blink_style_off = "dim"
  ```

### 14. digital_rain
- **Type**: Animated
- **Effect**: digital_rain
- **Description**: Enhanced Matrix effect with mixed character sets
- **Configuration**:
  ```toml
  base_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
  highlight_chars = "!@#$%^*()-+=[]{};:,.<>/?"
  highlight_chance = 0.05
  ```

### 15. loading_bar
- **Type**: Animated
- **Effect**: loading_bar
- **Description**: Classic progress bar with percentage
- **Configuration**:
  ```toml
  fill_char = "█"
  bar_style = "bold green"
  text_above = "SYSTEM INITIALIZATION SEQUENCE"
  ```

### 16. starfield
- **Type**: Animated
- **Effect**: starfield
- **Description**: Star Wars-style hyperspace effect
- **Configuration**:
  ```toml
  num_stars = 200
  warp_factor = 0.25
  max_depth = 40.0
  star_chars = ["·", ".", "*", "+"]
  ```

### 17. terminal_boot
- **Type**: Animated
- **Effect**: terminal_boot
- **Description**: Simulated system boot sequence
- **Configuration**:
  ```toml
  cursor = "▋"
  boot_sequence = [
    {"text": "BIOS initializing...", "type_speed": 0.02},
    {"text": "Memory Test: OK", "pause_after": 0.2}
  ]
  ```

## Interactive Animations

### 18. glitch_reveal
- **Type**: Animated
- **Effect**: glitch_reveal
- **Description**: Logo gradually revealed through decreasing glitch
- **Configuration**:
  ```toml
  duration = 2.5
  start_intensity = 0.9
  end_intensity = 0.0
  ```

### 19. ascii_morph
- **Type**: Animated
- **Effect**: ascii_morph
- **Description**: Morphs between two ASCII art pieces
- **Configuration**:
  ```toml
  duration = 3.0
  start_art_name = "morph_art_start"
  end_art_name = "morph_art_end"
  morph_style = "dissolve"  # or "random_pixel", "wipe_left_to_right"
  ```

### 20. game_of_life
- **Type**: Animated
- **Effect**: game_of_life
- **Description**: Conway's Game of Life cellular automaton
- **Configuration**:
  ```toml
  grid_width = 50
  grid_height = 18
  gol_update_interval = 0.15
  initial_pattern = "glider"  # or "random"
  ```

### 21. scrolling_credits
- **Type**: Animated
- **Effect**: scrolling_credits
- **Description**: Movie-style scrolling credits
- **Configuration**:
  ```toml
  scroll_speed = 2.0  # Lines per second
  credits_list = [
    {"role": "Lead Developer", "name": "Your Name"},
    {"line": "Special Thanks To:"}
  ]
  ```

### 22. spotlight_reveal
- **Type**: Animated
- **Effect**: spotlight
- **Description**: Moving spotlight reveals hidden content
- **Configuration**:
  ```toml
  spotlight_radius = 7
  movement_speed = 15.0
  path_type = "lissajous"  # or "random_walk", "circle"
  ```

### 23. sound_bars
- **Type**: Animated
- **Effect**: sound_bars
- **Description**: Audio equalizer visualization
- **Configuration**:
  ```toml
  num_bars = 25
  bar_char_filled = "┃"
  update_speed = 0.05
  ```

## Creative Effects

### 24. raindrops_pond
- **Type**: Animated
- **Effect**: raindrops
- **Description**: Ripples on water surface
- **Configuration**:
  ```toml
  spawn_rate = 2.0  # Drops per second
  ripple_chars = ["·", "o", "O", "()"]
  max_concurrent_ripples = 20
  ```

### 25. pixel_zoom
- **Type**: Animated
- **Effect**: pixel_zoom
- **Description**: Pixelation zoom in/out effect
- **Configuration**:
  ```toml
  duration = 3.0
  max_pixel_size = 10
  effect_type = "resolve"  # or "pixelate"
  ```

### 26. text_explosion
- **Type**: Animated
- **Effect**: text_explosion
- **Description**: Text particles explode or implode
- **Configuration**:
  ```toml
  text_to_animate = "T . L . D . W"
  effect_direction = "implode"  # or "explode"
  particle_spread = 40.0
  ```

### 27. old_film
- **Type**: Animated
- **Effect**: old_film
- **Description**: Old film projector effect with grain and shake
- **Configuration**:
  ```toml
  frame_duration = 0.8
  shake_intensity = 1
  grain_density = 0.07
  ```

### 28. maze_generator
- **Type**: Animated
- **Effect**: maze_generator
- **Description**: Procedural maze generation visualization
- **Configuration**:
  ```toml
  maze_width = 79  # Must be odd
  maze_height = 21  # Must be odd
  wall_char = "▓"
  generation_speed = 0.005
  ```

### 29. dwarf_fortress
- **Type**: Animated
- **Effect**: mining
- **Description**: Dwarf Fortress-style mining animation
- **Configuration**:
  ```toml
  dig_speed = 0.6
  style = "rgb(139,69,19) on black"  # Brown stone
  ```

## Tech-Themed Animations

### 30. neural_network
- **Type**: Animated
- **Effect**: neural_network
- **Description**: Animated neural network with signal propagation
- **Best for**: AI/ML applications

### 31. quantum_particles
- **Type**: Animated
- **Effect**: quantum_particles
- **Description**: Quantum particle visualization with uncertainty
- **Best for**: Scientific computing, physics

### 32. ascii_wave
- **Type**: Animated
- **Effect**: ascii_wave
- **Description**: Sine wave animation across screen
- **Best for**: Audio applications, signal processing

### 33. binary_matrix
- **Type**: Animated
- **Effect**: binary_matrix
- **Description**: Binary code falling like Matrix rain
- **Best for**: Low-level systems, binary data

### 34. constellation_map
- **Type**: Animated
- **Effect**: constellation_map
- **Description**: Connected star constellation visualization
- **Best for**: Navigation, astronomy apps

### 35. typewriter_news
- **Type**: Animated
- **Effect**: typewriter_news
- **Description**: News ticker typewriter effect
- **Best for**: News readers, information displays

### 36. dna_sequence
- **Type**: Animated
- **Effect**: dna_sequence
- **Description**: DNA double helix visualization
- **Best for**: Bioinformatics, genetics

### 37. circuit_trace
- **Type**: Animated
- **Effect**: circuit_trace
- **Description**: Electronic circuit path tracing
- **Best for**: Electronics, hardware interfaces

### 38. plasma_field
- **Type**: Animated
- **Effect**: plasma_field
- **Description**: Plasma effect using sine waves
- **Best for**: Retro demos, visual effects

### 39. ascii_fire
- **Type**: Animated
- **Effect**: ascii_fire
- **Description**: Animated fire effect in ASCII
- **Best for**: Intense, dramatic entrances

### 40. rubiks_cube
- **Type**: Animated
- **Effect**: rubiks_cube
- **Description**: 3D Rubik's cube rotation
- **Best for**: Puzzle games, problem-solving apps

### 41. data_stream
- **Type**: Animated
- **Effect**: data_stream
- **Description**: Streaming data visualization
- **Best for**: Data processing, analytics

### 42. fractal_zoom
- **Type**: Animated
- **Effect**: fractal_zoom
- **Description**: Mandelbrot set fractal zoom
- **Best for**: Mathematics, complex visualizations

### 43. ascii_spinner
- **Type**: Animated
- **Effect**: ascii_spinner
- **Description**: Various ASCII loading spinners
- **Best for**: Simple loading indication

### 44. hacker_terminal
- **Type**: Animated
- **Effect**: hacker_terminal
- **Description**: Hollywood-style hacking interface
- **Best for**: Security tools, penetration testing

## Customization Tips

1. **Performance**: For slower systems, use simpler effects like `minimal_fade` or static screens
2. **Branding**: Create custom ASCII art and use with effects like `glitch_reveal` or `typewriter`
3. **Themes**: Match splash screen to your application's purpose (e.g., `circuit_trace` for hardware tools)
4. **Duration**: Adjust based on typical load time - longer for heavy initialization
5. **Colors**: Most effects support custom color schemes via style parameters

## Psychedelic Effects

### 45. psychedelic_mandala
- **Type**: Animated
- **Effect**: psychedelic_mandala
- **Description**: A rotating, colorful mandala that expands from the center.
- **Best for**: Meditative or creative applications.

### 46. lava_lamp
- **Type**: Animated
- **Effect**: lava_lamp
- **Description**: Morphing, colored blobs that rise and fall.
- **Best for**: Groovy, retro themes.

### 47. kaleidoscope
- **Type**: Animated
- **Effect**: kaleidoscope
- **Description**: Symmetrical, mirrored patterns that shift and rotate.
- **Best for**: Visual-heavy, artistic applications.

### 48. deep_dream
- **Type**: Animated
- **Effect**: deep_dream
- **Description**: ASCII art with recursive, dream-like patterns.
- **Best for**: AI-themed, surreal applications.

### 49. trippy_tunnel
- **Type**: Animated
- **Effect**: trippy_tunnel
- **Description**: A perspective tunnel with shifting colors.
- **Best for**: Sci-fi, futuristic themes.

### 50. melting_screen
- **Type**: Animated
- **Effect**: melting_screen
- **Description**: Screen content appears to melt and drip downwards.
- **Best for**: Surreal, artistic, or error-themed screens.

### 51. color_pulse
- **Type**: Animated
- **Effect**: color_pulse
- **Description**: Screen pulses through a psychedelic color palette.
- **Best for**: Simple, yet vibrant loading screens.

### 52. shroom_vision
- **Type**: Animated
- **Effect**: shroom_vision
- **Description**: Simulates a "mushroom vision" effect with distorted visuals.
- **Best for**: Fun, playful, or creative applications.

### 53. hypno_swirl
- **Type**: Animated
- **Effect**: hypno_swirl
- **Description**: A hypnotic, swirling pattern.
- **Best for**: Mesmerizing, focus-themed applications.

### 54. electric_sheep
- **Type**: Animated
- **Effect**: electric_sheep
- **Description**: Abstract, evolving patterns.
- **Best for**: Abstract, generative art-style screens.

## Creating Custom Cards

Place custom splash card TOML files in `~/.config/tldw_cli/splash_cards/`. See the examples directory for templates.