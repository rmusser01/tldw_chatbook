# Custom Splash Card Examples

This directory contains example configurations and code for creating custom splash screens in TLDW CLI.

## Files

### Configuration Examples

#### Original Cards
- **`cyberpunk_card.toml`** - A glitch-effect cyberpunk themed splash card
- **`minimalist_card.toml`** - Clean, simple design with typewriter effect  
- **`gaming_card.toml`** - Achievement-style gaming splash screen
- **`config_examples.toml`** - Multiple complete configuration examples for different use cases

#### Retro Computing Collection
- **`commodore64_card.toml`** - Commodore 64 boot screen with BASIC prompt
- **`msdos_boot_card.toml`** - MS-DOS startup sequence with memory check
- **`apple2_card.toml`** - Apple II green phosphor monitor effect
- **`bbs_dialing_card.toml`** - Classic BBS dial-up connection screen
- **`punchcard_card.toml`** - IBM System/360 punch card reader

#### Philosophical AI Collection
- **`consciousness_bootstrap_card.toml`** - AI consciousness initialization
- **`dream_state_card.toml`** - Electric sheep and digital dreams
- **`existential_query_card.toml`** - SQL queries for meaning in the universe
- **`turing_test_card.toml`** - Interactive Turing test dialogue
- **`digital_meditation_card.toml`** - Zen meditation in code

#### Developer Humor Collection
- **`git_conflict_card.toml`** - Git merge conflict chaos
- **`dependency_hell_card.toml`** - npm install gone wrong
- **`stackoverflow_card.toml`** - Stack Overflow driven development
- **`code_review_card.toml`** - Traumatic code review experience
- **`regex_madness_card.toml`** - Complex regex patterns and regret

#### Sci-Fi Adventure Collection
- **`starship_boot_card.toml`** - Starship systems initialization
- **`time_machine_card.toml`** - Temporal navigation interface
- **`alien_contact_card.toml`** - First contact with alien programmers
- **`netrunner_card.toml`** - Cyberpunk netrunner interface
- **`mars_colony_card.toml`** - Mars colony terminal status

#### Nature & Organic Collection
- **`digital_garden_card.toml`** - Code growing like a garden
- **`ocean_waves_card.toml`** - Deep learning ocean with data waves
- **`mountain_peak_card.toml`** - Reaching the summit of 10,000 commits
- **`forest_neural_card.toml`** - Random forest classifier visualization
- **`butterfly_effect_card.toml`** - Chaos theory in version control

#### Music & Audio Production Collection
- **`synthesizer_boot_card.toml`** - Analog synthesizer startup sequence
- **`daw_loading_card.toml`** - Digital Audio Workstation loading project
- **`mixing_console_card.toml`** - SSL mixing console with VU meters
- **`waveform_visualizer_card.toml`** - Audio spectrum analyzer display
- **`midi_sequencer_card.toml`** - TR-808 style drum pattern sequencer

#### Mythology & Ancient Legends Collection
- **`greek_oracle_card.toml`** - Oracle of Delphi with divine prophecies
- **`egyptian_hieroglyphs_card.toml`** - Book of the Digital Dead with hieroglyphs
- **`norse_runes_card.toml`** - Norse mythology terminal with Yggdrasil
- **`celtic_druid_card.toml`** - Celtic stone circle and druid wisdom
- **`aztec_calendar_card.toml`** - Aztec calendar with Quetzalcoatl blessing

#### Culinary Code Kitchen Collection
- **`recipe_compiler_card.toml`** - Recipe as code with baking compilation
- **`chefs_terminal_card.toml`** - Kitchen OS with chef command line
- **`molecular_gastronomy_card.toml`** - Scientific cooking lab interface
- **`baking_algorithm_card.toml`** - Sourdough fermentation process
- **`sushi_assembly_card.toml`** - Sushi assembly line pipeline

#### Horror & Gothic Terminal Collection
- **`haunted_terminal_card.toml`** - Possessed terminal with ghost processes
- **`vampire_process_card.toml`** - Vampire process manager draining resources
- **`zombie_apocalypse_card.toml`** - System infected with zombie processes
- **`lovecraftian_compiler_card.toml`** - Eldritch horror compiler with madness
- **`gothic_cathedral_card.toml`** - Gothic cathedral OS with prayers

#### Social Media Simulator Collection
- **`influencer_dashboard_card.toml`** - Influencer analytics and metrics
- **`tweet_storm_card.toml`** - Twitter terminal with timeline feed
- **`instagram_filter_card.toml`** - Instagram reality filter loader
- **`tiktok_algorithm_card.toml`** - TikTok endless scroll algorithm
- **`linkedin_networking_card.toml`** - LinkedIn corporate synergy terminal

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

4. **Use themed collections**: Choose a specific theme for consistent experience:
   ```toml
   # For retro computing theme
   [splash_screen]
   active_cards = ["commodore64_card", "msdos_boot_card", "apple2_card"]
   
   # For developer humor theme
   [splash_screen]
   active_cards = ["git_conflict_card", "dependency_hell_card", "stackoverflow_card"]
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
6. **Theming**: Group related cards for consistent user experience
7. **Loading Messages**: Each card category includes thematic loading messages

## Sharing Your Creations

Created an awesome splash card? Consider:
1. Submitting a PR to add it to the built-in cards
2. Sharing in the discussions/issues
3. Creating a splash card pack repository

Happy splashing! 🎨