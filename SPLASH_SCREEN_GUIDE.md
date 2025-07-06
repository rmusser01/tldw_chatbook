# TLDW CLI Splash Screen Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Creating Custom Splash Screens](#creating-custom-splash-screens)
5. [Animation System](#animation-system)
6. [CSS Styling](#css-styling)
7. [Integration and Event Flow](#integration-and-event-flow)
8. [Troubleshooting](#troubleshooting)
9. [Examples](#examples)

## Overview

The TLDW CLI splash screen system provides a customizable, animated startup experience inspired by Call of Duty's calling cards. It supports multiple animation effects, progress tracking, and user-defined splash cards that can be randomly selected or specifically configured.

### Key Features
- Multiple built-in animation effects (Matrix rain, glitch, retro terminal)
- Progress bar with status messages
- Skip functionality with keypress
- Configuration-based customization
- Support for custom splash cards
- Smooth fade in/out transitions
- Memory-efficient lazy loading of main UI

## Architecture

### Component Structure
```
tldw_chatbook/
├── Widgets/
│   └── splash_screen.py         # Main SplashScreen widget
├── Utils/
│   ├── Splash.py               # ASCII art and card configs
│   └── splash_animations.py    # Animation effect classes
├── css/features/
│   └── _splash.tcss           # Splash screen styles
└── app.py                     # Integration point
```

### Lifecycle Flow
1. **App Startup**: `compose()` checks if splash screen is enabled
2. **Splash Display**: Only splash screen widget is yielded, app returns early
3. **Animation**: Splash screen runs its configured animation
4. **Progress Updates**: App can update progress during initialization
5. **Close Event**: After duration or keypress, `SplashScreenClosed` event fires
6. **UI Mount**: Main UI is created and mounted only after splash closes
7. **Cleanup**: Splash screen is removed from DOM

## Configuration

### Basic Configuration (config.toml)
```toml
[splash_screen]
# Enable/disable the splash screen
enabled = true

# Duration in seconds (float)
duration = 1.5

# Allow users to skip with any keypress
skip_on_keypress = true

# Card selection mode: "random", "sequential", or specific card name
card_selection = "random"

# Show progress bar and status text
show_progress = true

# List of active cards to choose from
active_cards = ["default", "matrix", "glitch", "retro", "classic", "compact", "minimal"]

[splash_screen.effects]
# Fade transition timings
fade_in_duration = 0.3
fade_out_duration = 0.2

# Animation speed multiplier (1.0 = normal speed)
animation_speed = 1.0
```

### Configuration Options Explained

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Master switch for splash screen |
| `duration` | float | 1.5 | Display time in seconds |
| `skip_on_keypress` | bool | true | Allow keyboard skip |
| `card_selection` | string | "random" | How to select splash card |
| `show_progress` | bool | true | Display progress bar |
| `active_cards` | list | [...] | Available splash cards |
| `fade_in_duration` | float | 0.3 | Fade in animation time |
| `fade_out_duration` | float | 0.2 | Fade out animation time |
| `animation_speed` | float | 1.0 | Animation playback speed |

## Creating Custom Splash Screens

### Method 1: Built-in Card Configuration

Add a new card to the `_load_card()` method in `splash_screen.py`:

```python
"custom_card": {
    "type": "animated",  # or "static"
    "effect": "matrix_rain",  # animation effect to use
    "title": "My Custom App",
    "subtitle": "Awesome Subtitle",
    "style": "bold cyan on black",
    "animation_speed": 0.08,
    # Effect-specific parameters
    "matrix_chars": "01",  # For matrix effect
    "glitch_intensity": 0.5,  # For glitch effect
}
```

### Method 2: Custom Card Files (Future)

Place TOML files in `~/.config/tldw_cli/splash_cards/`:

```toml
# ~/.config/tldw_cli/splash_cards/my_card.toml
type = "animated"
effect = "typewriter"
content = """
╭─────────────────────────╮
│   Welcome to My App     │
│   Version 2.0           │
╰─────────────────────────╯
"""
style = "bold blue on black"
typewriter_speed = 0.05
typewriter_sound = true
```

### Method 3: ASCII Art Integration

Use the existing ASCII art from `Utils/Splash.py`:

```python
# In Splash.py
MY_CUSTOM_ASCII = r"""
 __  __         _            
|  \/  |_   _  | |_ _____  __
| |\/| | | | | | __/ _ \ \/ /
| |  | | |_| | | ||  __/>  < 
|_|  |_|\__, |  \__\___/_/\_\
        |___/                
"""

# Add to get_ascii_art() function
"my_custom": MY_CUSTOM_ASCII,

# Add card config
def get_splash_card_config(name: str) -> Dict[str, Any]:
    if name == "my_custom":
        return {
            "type": "static",
            "content": get_ascii_art("my_custom"),
            "style": "bold magenta on black",
            "effect": None
        }
```

## Animation System

### Base Animation Class

All animations inherit from `BaseEffect`:

```python
class BaseEffect:
    def __init__(self, parent_widget: Any, **kwargs):
        self.parent = parent_widget
        self.frame_count = 0
        self.start_time = time.time()
        
    def update(self) -> Optional[str]:
        """Return the next frame of animation."""
        self.frame_count += 1
        return None
    
    def reset(self) -> None:
        """Reset animation to initial state."""
        self.frame_count = 0
        self.start_time = time.time()
```

### Built-in Effects

#### 1. Matrix Rain Effect
- Falling green characters like The Matrix
- Gradually reveals title and subtitle
- Customizable character set and fall speed

```python
MatrixRainEffect(
    parent_widget=self,
    title="TLDW CLI",
    subtitle="Terminal LLM Dialog Writer",
    speed=0.05,
    width=80,
    height=24
)
```

#### 2. Glitch Effect
- Random character corruption
- Multiple color variations
- Adjustable intensity

```python
GlitchEffect(
    parent_widget=self,
    content=ascii_art,
    glitch_chars="!@#$%^&*",
    speed=0.1,
    intensity=0.3
)
```

#### 3. Retro Terminal Effect
- CRT monitor simulation
- Phosphor glow effect
- Scanline animation

```python
RetroTerminalEffect(
    parent_widget=self,
    content=ascii_art,
    scanline_speed=0.02,
    phosphor_glow=True,
    flicker_intensity=0.1
)
```

### Creating Custom Animation Effects

```python
# In splash_animations.py
class WaveEffect(BaseEffect):
    """Creates a wave animation across text."""
    
    def __init__(self, parent_widget: Any, content: str, wave_speed: float = 0.1, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.content = content
        self.wave_speed = wave_speed
        self.lines = content.split('\n')
        
    def update(self) -> Optional[str]:
        """Create wave effect by offsetting characters."""
        output_lines = []
        time_offset = time.time() - self.start_time
        
        for y, line in enumerate(self.lines):
            offset = int(math.sin(y * 0.5 + time_offset * 2) * 3)
            padded_line = " " * max(0, offset) + line
            output_lines.append(padded_line)
            
        return '\n'.join(output_lines)
```

## CSS Styling

### Required CSS Structure

The splash screen requires specific CSS classes and IDs:

```css
/* Main container - must cover entire screen */
SplashScreen, .splash-screen {
    layer: overlay;      /* Display above other content */
    width: 100%;
    height: 100%;
    dock: top;          /* Fill entire area */
    background: $background;
}

/* Content containers */
#splash-center {         /* Centers content */
    align: center middle;
}

#splash-content {        /* Content wrapper */
    width: 80;          /* Textual grid units */
    padding: 2 4;
}

#splash-main {          /* Main display area */
    min-height: 24;     /* Ensure visibility */
    text-align: center;
}

/* Progress elements */
#splash-progress-bar {
    width: 60;
    align: center middle;
}
```

### Animation-Specific Styles

```css
/* Matrix rain effect */
.matrix-rain {
    color: $success;     /* Green color */
    text-style: bold;
}

/* Glitch effect variations */
.glitch-effect-error { color: $error; }
.glitch-effect-warning { color: $warning; }
.glitch-effect-success { color: $success; }

/* Retro terminal */
.retro-terminal {
    color: $success;
    background: rgba(0, 0, 0, 0.9);
}

.retro-scanline {
    background: rgba(0, 255, 0, 0.1);
}
```

### Custom Styling

Add custom styles for your splash cards:

```css
/* Custom splash card style */
.my-custom-splash {
    color: rgb(128, 128, 255);
    text-style: bold italic;
    background: linear-gradient(0deg, $primary 0%, $background 100%);
}

/* Animation keyframes (note: Textual doesn't support @keyframes) */
/* Use Textual's animation API instead */
.my-custom-splash.animating {
    border: heavy $primary;
}
```

## Integration and Event Flow

### 1. App Startup (app.py)

```python
def compose(self) -> ComposeResult:
    # Check if splash screen is enabled
    splash_enabled = get_cli_setting("splash_screen", "enabled", True)
    
    if splash_enabled:
        # Create splash screen with config
        self._splash_screen_widget = SplashScreen(
            card_name=card_selection,
            duration=duration,
            skip_on_keypress=skip,
            show_progress=progress
        )
        self.splash_screen_active = True
        yield self._splash_screen_widget
        
        # IMPORTANT: Return early to prevent main UI from loading
        return
    
    # If disabled, load main UI immediately
    yield from self._compose_main_ui()
```

### 2. Progress Updates

During initialization, update the splash screen:

```python
def on_mount(self) -> None:
    if self.splash_screen_active and self._splash_screen_widget:
        self._splash_screen_widget.update_progress(0.3, "Loading configuration...")
        
    # Do initialization work...
    
    if self.splash_screen_active and self._splash_screen_widget:
        self._splash_screen_widget.update_progress(0.7, "Connecting to services...")
```

### 3. Splash Close Event

```python
@on(SplashScreenClosed)
async def on_splash_screen_closed(self, event: SplashScreenClosed) -> None:
    """Handle splash screen closing."""
    self.splash_screen_active = False
    
    # Remove splash screen
    if self._splash_screen_widget:
        await self._splash_screen_widget.remove()
        self._splash_screen_widget = None
    
    # NOW create and mount the main UI
    main_ui_widgets = self._create_main_ui_widgets()
    await self.mount(*main_ui_widgets)
    
    # Schedule post-mount setup
    self.call_after_refresh(self._post_mount_setup)
```

### Event Timeline

```
[App Start]
    |
    v
[compose() called]
    |
    ├─> [Splash Enabled?]
    |      |
    |      ├─> Yes: Yield SplashScreen only, return
    |      |         |
    |      |         v
    |      |    [SplashScreen.on_mount()]
    |      |         |
    |      |         ├─> Start animations
    |      |         ├─> Set close timer
    |      |         └─> Listen for keypresses
    |      |
    |      └─> No: Yield main UI immediately
    |
    v
[App.on_mount()]
    |
    ├─> Update splash progress (if active)
    └─> Setup logging, themes, etc.
    
[Time passes / Key pressed]
    |
    v
[SplashScreen._request_close()]
    |
    ├─> Stop animations
    ├─> Fade out
    └─> Post SplashScreenClosed event
        |
        v
[App.on_splash_screen_closed()]
    |
    ├─> Remove splash widget
    ├─> Create main UI widgets
    ├─> Mount main UI
    └─> Continue initialization
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Splash Screen Not Showing

**Symptoms**: App starts directly without splash screen

**Causes & Solutions**:
- **Configuration**: Ensure `enabled = true` in config.toml
- **Return Statement**: Verify `compose()` returns after yielding splash
- **CSS Loading**: Check that _splash.tcss is included in build_css.py
- **Debug**: Add logging to verify splash screen creation

```python
# Debug in compose()
logger.info(f"Splash enabled: {splash_enabled}")
if splash_enabled:
    logger.info("Creating splash screen widget")
    # ...
```

#### 2. Splash Screen Shows But No Animation

**Symptoms**: Static display, no movement

**Causes & Solutions**:
- **Timer Not Set**: Check animation timer is created in `_start_card_animation()`
- **Effect Handler**: Verify effect handler is assigned
- **Update Method**: Ensure `_update_animation()` is being called

```python
# Debug animation
def _update_animation(self) -> None:
    logger.debug(f"Animation update, frame: {self.current_frame}")
    if self.effect_handler:
        content = self.effect_handler.update()
        # ...
```

#### 3. Main UI Not Loading After Splash

**Symptoms**: Splash closes but app appears frozen

**Causes & Solutions**:
- **Event Handler**: Verify `on_splash_screen_closed()` is decorated with `@on()`
- **Widget Creation**: Check `_create_main_ui_widgets()` returns valid widgets
- **Mount Error**: Look for exceptions in mount operation

```python
# Add error handling
try:
    main_ui_widgets = self._create_main_ui_widgets()
    await self.mount(*main_ui_widgets)
except Exception as e:
    logger.error(f"Failed to mount main UI: {e}")
```

#### 4. Progress Bar Not Updating

**Symptoms**: Progress stays at 0% or doesn't reflect updates

**Causes & Solutions**:
- **Widget Query**: Progress bar widget might not be found
- **Value Range**: Ensure progress values are between 0.0 and 1.0
- **Timing**: Updates might happen before widget is ready

```python
def update_progress(self, value: float, text: Optional[str] = None) -> None:
    # Clamp value to valid range
    self.progress = max(0.0, min(1.0, value))
    
    if self.show_progress:
        try:
            progress_bar = self.query_one("#splash-progress-bar", ProgressBar)
            progress_bar.update(progress=self.progress * 100)
        except Exception as e:
            logger.warning(f"Could not update progress: {e}")
```

### Debug Mode

Enable detailed splash screen debugging:

```python
# In splash_screen.py __init__
self.debug = get_cli_setting("splash_screen", "debug", False)

# Throughout the code
if self.debug:
    logger.debug(f"Splash state: active={self.is_active}, progress={self.progress}")
```

## Examples

### Example 1: Minimal Configuration

```toml
# Minimal splash screen - just enable with defaults
[splash_screen]
enabled = true
```

### Example 2: Fast Loading Screen

```toml
# Quick splash with just branding
[splash_screen]
enabled = true
duration = 0.8
show_progress = false
card_selection = "minimal"
active_cards = ["minimal"]

[splash_screen.effects]
fade_in_duration = 0.1
fade_out_duration = 0.1
```

### Example 3: Full Gaming Experience

```toml
# Full Call of Duty style experience
[splash_screen]
enabled = true
duration = 3.0
skip_on_keypress = true
card_selection = "random"
show_progress = true
active_cards = ["matrix", "glitch", "retro", "neon", "hologram"]

[splash_screen.effects]
fade_in_duration = 0.5
fade_out_duration = 0.3
animation_speed = 1.2
```

### Example 4: Custom Branded Splash

```python
# Add to splash_screen.py built_in_cards
"company_brand": {
    "type": "animated",
    "effect": "typewriter",
    "content": """
    ╔═══════════════════════════════════╗
    ║                                   ║
    ║      ACME CORPORATION             ║
    ║      AI Assistant v3.0            ║
    ║                                   ║
    ║      Initializing Systems...      ║
    ║                                   ║
    ╚═══════════════════════════════════╝
    """,
    "style": "bold white on rgb(0,32,64)",
    "typewriter_speed": 0.02,
    "typewriter_reveal_line": True
}
```

### Example 5: Seasonal Splash Cards

```python
# Add seasonal detection
import datetime

def _select_card(self) -> str:
    """Select card with seasonal awareness."""
    month = datetime.datetime.now().month
    
    # Holiday seasons
    if month == 12:  # December
        return "winter_holiday"
    elif month == 10:  # October
        return "spooky"
    elif month == 7:  # July
        return "summer"
    
    # Default behavior
    return super()._select_card()
```

## Best Practices

### Performance
1. Keep animations lightweight - avoid complex calculations
2. Limit animation frame rates (use intervals > 0.05s)
3. Pre-calculate animation data when possible
4. Clean up timers properly on close

### User Experience
1. Always provide skip option for accessibility
2. Keep total duration under 3 seconds
3. Show meaningful progress messages
4. Ensure text is readable with sufficient contrast

### Development
1. Test with splash disabled to ensure app works without it
2. Add new cards to active_cards list explicitly
3. Use configuration for all timing values
4. Log important state changes for debugging

### Accessibility
1. Respect user's reduced motion preferences
2. Avoid flashing or strobing effects
3. Ensure splash can be completely disabled
4. Provide text alternatives for ASCII art

## Future Enhancements

### Planned Features
1. **Sound Effects**: Optional audio for animations
2. **Custom Card Loader**: Load cards from user directory
3. **Transition Effects**: More elaborate transitions between splash and main UI
4. **Interactive Elements**: Mini-games or interactive animations
5. **Theming Support**: Adapt to active terminal theme
6. **Plugin System**: Allow external animation effect modules
7. **Statistics**: Track which cards users see most/skip most
8. **A/B Testing**: Test different splash configurations

### Community Contributions
To contribute new splash cards or effects:
1. Follow the existing animation class structure
2. Ensure all timings are configurable
3. Add appropriate CSS classes
4. Include example configuration
5. Test with various terminal sizes
6. Document any new dependencies