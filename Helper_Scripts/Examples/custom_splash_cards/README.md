# Custom Splash Cards Examples

This directory contains example TOML configuration files for creating custom splash screens in TLDW Chatbook.

## Available Examples

1. **simple_static.toml** - A basic static ASCII art splash screen
2. **animated_matrix.toml** - Customized Matrix rain effect with red theme
3. **corporate_boot.toml** - Professional boot sequence for corporate environments
4. **holiday_theme.toml** - Seasonal splash screen with snow effect

## How to Use

1. Copy any of these example files to your splash cards directory:
   ```bash
   cp simple_static.toml ~/.config/tldw_cli/splash_cards/
   ```

2. Edit the file to customize for your needs

3. Add your custom card to the active cards list in your config:
   ```toml
   [splash_screen]
   active_cards = ["default", "matrix", "simple_static"]  # Added custom card
   ```

4. Or set it as the only splash screen:
   ```toml
   [splash_screen]
   card_selection = "simple_static"  # Always use this card
   ```

## Creating Your Own

Use these examples as templates. Key elements:

- `[card]` section defines the card properties
- `type` can be "static" or "animated"
- `effect` specifies which animation to use (for animated cards)
- `content` holds ASCII art (for static cards)
- `style` uses Rich text styling format
- Additional parameters depend on the effect chosen

## Tips

- Test your ASCII art in a monospace font editor
- Keep width under 80 characters for compatibility
- Leave room for progress bar (keep height under 20 lines)
- Use the catalog documentation to find all available effects and their parameters

## Resources

- Full splash screen catalog: `/Docs/Examples/SPLASH_SCREENS_CATALOG.md`
- Quick reference guide: `/Docs/Examples/SPLASH_SCREEN_QUICK_REF.md`
- ASCII art generators: patorjk.com/software/taag/