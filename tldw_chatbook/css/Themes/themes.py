# themes.py
from textual.theme import Theme
from textual.color import Color


# Helper to convert string dicts to Theme objects
def create_theme_from_dict(name: str, theme_dict: dict) -> Theme:
    theme_args = {"name": name}
    for key, value in theme_dict.items():
        if key == "dark":
            theme_args[key] = bool(value)
        # All other color keys are assumed to be color strings
        elif key in [
            "primary", "secondary", "accent", "warning", "error", "success",
            "background", "surface", "panel", "foreground",
        ]:
            try:
                # Ensure value is a string before parsing, though it should be from the dict
                theme_args[key] = Color.parse(str(value))
            except Exception as e:
                print(f"Warning: Could not parse color '{value}' for key '{key}' in theme '{name}'. Error: {e}")
                # Fallback to a default color or skip if parsing fails
                # For example, Color.parse("red") or continue
        else:  # For any other variables Textual's Theme constructor might support (e.g., 'variables' dict)
            theme_args[key] = value
    return Theme(**theme_args)


RAW_THEMES_DATA = {
    # It's good practice to ensure your custom theme names are unique
    # and don't clash with potential future built-in Textual themes.
    "my_custom_dark": {
        "primary": "#004578",
        "secondary": "#006FB3",
        "accent": "#0099FF",
        "warning": "#FFD700",
        "error": "#FF0000",
        "success": "#008000",
        "background": "#1E1E1E",
        "surface": "#2C2C2C",
        "panel": "#252525",
        "foreground": "#FFFFFF",
        "dark": True,
    },
    "solarized_dark_inspired": {
        "background": "#002b36",
        "surface": "#073642",
        "panel": "#073642",
        "primary": "#268bd2",
        "secondary": "#2aa198",
        "accent": "#b58900",
        "foreground": "#839496",
        "warning": "#cb4b16",
        "error": "#dc322f",
        "success": "#859900",
        "dark": True,
    },
    "paper_light_inspired": {
        "background": "#EEEEEE",
        "surface": "#FFFFFF",
        "panel": "#FFFFFF",
        "primary": "#AF8700",
        "secondary": "#D75F00",
        "accent": "#005F87",
        "foreground": "#444444",
        "warning": "#D75F00",
        "error": "#D70000",
        "success": "#008700",
        "dark": False,
    },
    # Add more themes here
}

MY_THEMES = {
    name: create_theme_from_dict(name, data)
    for name, data in RAW_THEMES_DATA.items()
}


# 1. Theme: "Classic Terminal" (Retro Green)
classic_terminal_green_theme = Theme(
    name="classic_terminal_green",
    primary="#33FF33",       # Bright Green (main interactive elements)
    secondary="#228B22",     # Forest Green (secondary elements, e.g., button backgrounds)
    accent="#55FF55",        # Lighter Green (borders, focus highlights)
    foreground="#33FF33",    # Bright Green (main text)
    background="#0A0A0A",    # Very dark gray (main background)
    surface="#1A1A1A",       # Darker gray (input backgrounds, component surfaces)
    panel="#0A0A0A",         # Very dark gray (panel backgrounds, same as main)
    success="#55FF55",       # Brighter Green
    warning="#FFBF00",       # Amber
    error="#FF3333",         # Classic Red
    dark=True,
    variables={
        "footer-key-foreground": "#55FF55",
        "input-selection-background": "#228B22 50%", # Forest Green with 50% alpha
    },
)

# 2. Theme: "Modern Dark" (Inspired by Dracula)
modern_dark_dracula_theme = Theme(
    name="modern_dark_dracula",
    primary="#ff79c6",       # Pink (primary actions)
    secondary="#bd93f9",     # Purple (secondary actions, input borders)
    accent="#8be9fd",        # Cyan (focus, highlights)
    foreground="#f8f8f2",    # Light gray/off-white
    background="#282a36",    # Dark purplish-blue
    surface="#44475a",       # Lighter background shade (buttons)
    panel="#20222b",         # Even darker (sidebars, data tables)
    success="#50fa7b",       # Green
    warning="#f1fa8c",       # Yellow
    error="#ff5555",         # Red (Dracula's red)
    dark=True,
    variables={
        "footer-key-foreground": "#8be9fd",   # Cyan
        "input-selection-background": "#bd93f9 40%", # Purple with 40% alpha
        "text-muted": "#6272a4",             # Lighter purple/gray for comments/dimmed text
    },
)

# 3. Theme: "Paper Light"
paper_light_theme = Theme(
    name="paper_light",
    primary="#5B4636",       # Sepia (primary actions)
    secondary="#AACCFF",     # Soft blue (secondary actions, hover states)
    accent="#AACCFF",        # Soft blue (focus, highlights)
    foreground="#333333",    # Dark gray (main text)
    background="#FEFEFA",    # Off-white/very light beige
    surface="#E0E0E0",       # Light gray (buttons)
    panel="#F8F8F4",         # Slightly off-white (panels, distinct from pure white inputs)
    success="#A8D8B0",       # Muted green
    warning="#EBCB8B",       # Standard warning yellow, fits light theme
    error="#BF616A",         # Standard error red, muted enough
    dark=False,
    variables={
        "footer-key-foreground": "#5B4636",   # Sepia
        "input-selection-background": "#AACCFF 50%", # Soft blue with 50% alpha
        "text-muted": "#777777",             # Lighter gray for subtle text
    },
)

# 4. Theme: "High Contrast Accessibility" (Yellow on Black)
high_contrast_yellow_black_theme = Theme(
    name="high_contrast_yellow_black",
    primary="#FFFFFF",       # White (major interactive elements, borders)
    secondary="#FFFF00",     # Bright Yellow (button text, secondary highlight)
    accent="#FFFFFF",        # White (focus outline, borders)
    foreground="#FFFF00",    # Bright Yellow
    background="#000000",    # Black
    surface="#333333",       # Dark Gray (button background)
    panel="#1A1A1A",         # Very Dark Gray (input background, panels)
    success="#00FF00",       # Bright Green
    warning="#FFFF00",       # Bright Yellow (main theme color for warning)
    error="#FF0000",         # Bright Red
    dark=True,
    variables={
        "footer-key-foreground": "#FFFFFF",   # White
        "input-selection-background": "#FFFFFF 30%", # White with 30% alpha
    },
)

# 5. Theme: "Ocean Depths"
ocean_depths_theme = Theme(
    name="ocean_depths",
    primary="#FF7F50",       # Coral (accent buttons, primary actions)
    secondary="#75E6DA",     # Light teal/aqua (hover, input focus)
    accent="#189AB4",        # Brighter teal (borders)
    foreground="#E0FFFF",    # Light Cyan/Off-White
    background="#0A2342",    # Deep dark blue
    surface="#005073",       # Medium dark blue/teal (buttons)
    panel="#001B33",         # Very dark blue (inputs, panels)
    success="#66CDAA",       # Medium Aquamarine (fits theme)
    warning="#F0E68C",       # Sandy Beige
    error="#FF6347",         # Tomato (reddish coral)
    dark=True,
    variables={
        "footer-key-foreground": "#75E6DA",         # Light teal/aqua
        "input-selection-background": "#FF7F50 40%", # Coral with 40% alpha
        "text-muted": "#ADD8E6",                   # Light blue
        "statusbar-background": "#F0E68C",         # Sandy beige (for specific components like status bar)
        "statusbar-foreground": "#0A2342",         # Deep dark blue (text on status bar)
    },
)

# 6. Theme: "Solarized Dark"
solarized_dark_theme = Theme(
    name="solarized_dark",
    primary="#268bd2",       # blue
    secondary="#6c71c4",     # violet
    accent="#2aa198",        # cyan
    foreground="#839496",    # base0
    background="#002b36",    # base03
    surface="#073642",       # base02 (buttons, inputs)
    panel="#073642",         # base02 (larger panels)
    success="#859900",       # green
    warning="#b58900",       # yellow
    error="#dc322f",         # red
    dark=True,
    variables={
        "footer-key-foreground": "#2aa198",         # cyan
        "input-selection-background": "#268bd2 40%", # blue with 40% alpha
        "text-muted": "#586e75",                   # base01 (dimmed text)
        "text-highlight": "#93a1a1",               # base1 (more important text)
    },
)

# 6. Theme: "Solarized Light"
solarized_light_theme = Theme(
    name="solarized_light",
    primary="#268bd2",       # blue
    secondary="#6c71c4",     # violet
    accent="#2aa198",        # cyan
    foreground="#657b83",    # base00
    background="#fdf6e3",    # base3
    surface="#eee8d5",       # base2 (buttons, inputs)
    panel="#eee8d5",         # base2 (larger panels)
    success="#859900",       # green
    warning="#b58900",       # yellow
    error="#dc322f",         # red
    dark=False,
    variables={
        "footer-key-foreground": "#2aa198",         # cyan
        "input-selection-background": "#268bd2 40%", # blue with 40% alpha
        "text-muted": "#93a1a1",                   # base1 (dimmed text for light theme)
        "text-highlight": "#586e75",               # base01 (more important text for light theme)
    },
)

# 7. Theme: "Monokai Pro"
monokai_pro_theme = Theme(
    name="monokai_pro",
    primary="#A6E22E",       # Green (action buttons)
    secondary="#FF6188",     # Pink (special buttons)
    accent="#AE81FF",        # Purple (default button border, hover)
    foreground="#FCFCFA",    # Off-white
    background="#1E1D20",    # Very dark gray/black
    surface="#4A454D",       # Darker mid-tone (buttons)
    panel="#2D2A2E",         # Slightly lighter dark (inputs, panels)
    success="#A6E22E",       # Green
    warning="#FD971F",       # Orange
    error="#FF6188",         # Pink (often used for errors in Monokai)
    dark=True,
    variables={
        "footer-key-foreground": "#66D9EF",         # Blue
        "input-selection-background": "#AE81FF 40%", # Purple with 40% alpha
        "text-muted": "#75715E",                   # Gray (comments)
        "log-view-background": "#272822",          # Custom for specific widget example
        "log-view-foreground": "#E6DB74",          # Custom for specific widget example
    },
)

# 8. Theme: "Gruvbox Dark"
gruvbox_dark_theme = Theme(
    name="gruvbox_dark",
    primary="#fabd2f",       # Bright Yellow (prominent accent, hover)
    secondary="#b8bb26",     # Bright Green (confirm buttons)
    accent="#83a598",        # Bright Blue (input borders)
    foreground="#fbf1c7",    # fg0 (pale yellow)
    background="#1d2021",    # bg0_h (dark gray)
    surface="#3c3836",       # Darker bg shade (buttons)
    panel="#282828",         # Slightly lighter bg (inputs, panels)
    success="#b8bb26",       # Bright Green
    warning="#fabd2f",       # Bright Yellow
    error="#fb4934",         # Bright Red
    dark=True,
    variables={
        "footer-key-foreground": "#83a598",         # Bright Blue
        "input-selection-background": "#fabd2f 40%", # Bright Yellow with 40% alpha
        "text-muted": "#a89984",                   # Gray
    },
)

# 8. Theme: "Gruvbox Light"
gruvbox_light_theme = Theme(
    name="gruvbox_light",
    primary="#d65d0e",       # Dark Orange (prominent accent)
    secondary="#458588",     # Dark Blue
    accent="#b16286",        # Dark Purple
    foreground="#3c3836",    # Dark gray text
    background="#fbf1c7",    # Pale yellow background
    surface="#ebdbb2",       # Lighter bg shade (buttons)
    panel="#f9f5d7",         # Even lighter or same as background (panels)
    success="#98971a",       # Dark Green
    warning="#d79921",       # Dark Yellow
    error="#cc241d",         # Dark Red
    dark=False,
    variables={
        "footer-key-foreground": "#458588",         # Dark Blue
        "input-selection-background": "#d65d0e 40%", # Dark Orange with 40% alpha
        "text-muted": "#7c6f64",                   # Gray for light background
    },
)

# 9. Theme: "Cyberpunk Neon"
cyberpunk_neon_theme = Theme(
    name="cyberpunk_neon",
    primary="#FF00FF",       # Hot Pink (button color, border)
    secondary="#39FF14",     # Neon Green (secondary actions, input focus)
    accent="#FEFE22",        # Laser Lemon (highlights, contrast border)
    foreground="#00FFFF",    # Electric Blue
    background="#0D0221",    # Deep space blue/purple
    surface="#240046",       # Dark Purple (button background)
    panel="#101010",         # Dark Gray/Black (inputs, panels)
    success="#39FF14",       # Neon Green
    warning="#FEFE22",       # Laser Lemon
    error="#FF1D58",         # Neon Red/Pink (distinct error color)
    dark=True,
    variables={
        "footer-key-foreground": "#FEFE22",         # Laser Lemon
        "input-selection-background": "#FF00FF 40%", # Hot Pink with 40% alpha
        "text-highlight": "#FEFE22",               # Laser Lemon for highlighted text
    },
)

# 10. Theme: "Earthy Nature Tones"
earthy_nature_theme = Theme(
    name="earthy_nature",
    primary="#CD853F",       # Peru (Muted Orange/Brown, action buttons)
    secondary="#808000",     # Olive (button hover)
    accent="#E2725B",        # Terracotta (input focus)
    foreground="#F5F5DC",    # Beige (Off-White)
    background="#2F4F2F",    # Forest Green
    surface="#556B2F",       # Dark Olive Green (button background)
    panel="#4A4034",         # Dark Taupe (inputs, panels)
    success="#8FBC8F",       # Dark Sea Green
    warning="#DAA520",       # Goldenrod
    error="#B22222",         # Firebrick
    dark=True,
    variables={
        "footer-key-foreground": "#E2725B",         # Terracotta
        "input-selection-background": "#CD853F 40%", # Peru with 40% alpha
        "text-muted": "#BDB76B",                   # Dark Khaki
        "tree-control-background": "#3A322A",      # Darker Brown for tree view specific (example)
    },
)

# 11. Theme: "Pastel Dreams"
pastel_dreams_theme = Theme(
    name="pastel_dreams",
    primary="#FFDFD3",       # Baby Pink (default button background)
    secondary="#D4F0E0",     # Mint Green (confirm button background)
    accent="#A8DFFF",        # Brighter Sky Blue (input focus)
    foreground="#A094B7",    # Soft Gray-Purple
    background="#F5F0FF",    # Light Lavender White
    surface="#FFFFFF",       # White (input backgrounds)
    panel="#E6DFF2",         # Slightly darker lavender (sidebars, panels)
    success="#D4F0E0",       # Mint Green
    warning="#FFE6B3",       # Soft Pastel Yellow
    error="#FFB6C1",         # Light Pink (for errors)
    dark=False,
    variables={
        "text-title": "#796A93",             # Darker Soft Purple for titles
        "input-border-default": "#C9EBFB",   # Light Sky Blue for input border
        "button-default-foreground": "#BF8A7E", # Muted Rose for default button text
        "button-confirm-foreground": "#7BAA8F", # Muted Mint for confirm button text
    },
)

# 12. Theme: "Sweet Sorbet"
sweet_sorbet_theme = Theme(
    name="sweet_sorbet",
    primary="#FFFACD",       # Pastel Lemon (default button)
    secondary="#FFB6C1",     # Raspberry Pink (special action button)
    accent="#CFF0C0",        # Slightly darker lime (input focus)
    foreground="#7D7068",    # Warm Gray
    background="#FFF0E5",    # Peachy Cream
    surface="#FFFFFF",       # White (input backgrounds)
    panel="#FFF0E5",         # Peachy Cream (panels, same as background or slightly off)
    success="#E0FFD1",       # Light Lime
    warning="#FFEEAA",       # Soft Lemon Yellow
    error="#FFB6C1",         # Raspberry Pink (can serve as error too)
    dark=False,
    variables={
        "input-border-default": "#E0FFD1",      # Light Lime for input border
        "button-default-foreground": "#B8A26B", # Muted Gold for default button text
        "button-special-foreground": "#996570", # Muted Dark Pink for special button text
        "progressbar-background": "#FFE0CC",    # Light peach track for progress bar
        "progressbar-color": "#FFB6C1",         # Raspberry fill for progress bar
    },
)

# 13. Theme: "Cloudy Day"
cloudy_day_theme = Theme(
    name="cloudy_day",
    primary="#B0C4DE",       # Powder Blue (navigation buttons, input focus)
    secondary="#C5D0E6",     # Light Periwinkle (hover states)
    accent="#B0C4DE",        # Powder Blue (matches primary for focus)
    foreground="#5A6470",    # Cool Dark Gray
    background="#E0E8F0",    # Light Blue-Gray
    surface="#FAFAFA",       # Soft White (default button backgrounds, component surfaces)
    panel="#F0F4F8",         # Slightly lighter blue-gray for panels
    success="#B2D8B2",       # Soft Green
    warning="#F0E6A2",       # Soft Yellow
    error="#DEB0B0",         # Soft Red
    dark=False,
    variables={
        "text-muted": "#778899",               # Light Slate Gray for info text
        "input-background": "#FFFFFF",         # White for input backgrounds
        "input-border-default": "#D0D8E0",     # Default input border
        "button-default-background": "#FAFAFA",# Soft white for default buttons
        "button-default-foreground": "#5A6470",# Cool dark gray for default button text
        "button-navigation-foreground": "#42505E", # Text for navigation buttons
    },
)

# 14. Theme: "Kawaii Candy"
kawaii_candy_theme = Theme(
    name="kawaii_candy",
    primary="#E6E6FA",       # Lavender (default buttons)
    secondary="#FF69B4",     # Hot Pink (super important buttons)
    accent="#FFFFE0",        # Sunny Yellow (input focus)
    foreground="#754C59",    # Dark Magenta-Brown
    background="#FFEFF2",    # Light Pink
    surface="#FFFFFF",       # White (input backgrounds)
    panel="#FFF5F7",         # Very Light Pink (panels)
    success="#7FFFD4",       # Aquamarine (can be success)
    warning="#FFFFE0",       # Sunny Yellow
    error="#FF69B4",         # Hot Pink (can be error)
    dark=False,
    variables={
        "input-border-default": "#7FFFD4",      # Aquamarine for input border
        "button-default-foreground": "#5C5C7D", # Default button text color
        "markdown-h1-color": "#FF69B4",
        "markdown-h2-color": "#754C59",
        "markdown-link-color": "#7FFFD4",
    },
)

# 15. Theme: "Bunny Fluff"
bunny_fluff_theme = Theme(
    name="bunny_fluff",
    primary="#F5F5DC",       # Light Beige (default buttons)
    secondary="#FCEEED",     # Very Light Rose (hover, focus accent)
    accent="#FCEEED",        # Very Light Rose (input focus)
    foreground="#8D8580",    # Soft Warm Gray
    background="#FAF7F5",    # Off-White
    surface="#FFFFFF",       # White (input backgrounds)
    panel="#F0EBE8",         # Slightly darker for footer/panels
    success="#E8F5E8",       # Very Light Green
    warning="#FFFDE8",       # Very Light Yellow
    error="#FDE8E8",         # Very Light Red/Rose
    dark=False,
    variables={
        "input-border-default": "#EAE2DC",      # Very light warm gray for input border
        "button-default-foreground": "#7A736E", # Default button text color
        "footer-background": "#F0EBE8",
        "footer-foreground": "#8D8580",
    },
)

# 16. Theme: "Neon Sunset Drive"
neon_sunset_drive_theme = Theme(
    name="neon_sunset_drive",
    primary="#008080",       # Teal (action buttons)
    secondary="#FF8C00",     # Sunset Orange (secondary interactive elements, borders)
    accent="#FFD700",        # Golden Yellow (focus, highlights)
    foreground="#FF00FF",    # Electric Pink
    background="#2A0A4A",    # Deep Indigo/Purple
    surface="#4B0082",       # Indigo (default button background)
    panel="#1A0433",         # Darker Purple (input backgrounds)
    success="#00CC66",       # Bright Neon Green
    warning="#FFD700",       # Golden Yellow
    error="#FF3333",         # Bright Neon Red
    dark=True,
    variables={
        "text-secondary-info": "#00FFFF",      # Cyan for secondary info text
        "input-border-default": "#00FFFF",     # Cyan for input border
        "button-default-foreground": "#FFD700",# Golden Yellow for default button text
        "button-action-foreground": "#FFFFFF", # White for action button text
    },
)

# 17. Theme: "Palm Mall"
palm_mall_theme = Theme(
    name="palm_mall",
    primary="#FF69B4",       # Neon Pink (prominent interactive, focus)
    secondary="#FFFFAA",     # Light Yellow (alternative accent)
    accent="#FF69B4",        # Neon Pink (input focus, matches primary)
    foreground="#4B0082",    # Dark Purple
    background="#B4E1E7",    # Washed-out Teal/Blue
    surface="#FFDAE9",       # Light Pink (default button background)
    panel="#FFFFFF",         # White (input backgrounds)
    success="#A7F0A7",       # Pastel Green
    warning="#FFFFAA",       # Light Yellow
    error="#FF80A0",         # Slightly deeper pink for errors
    dark=False,
    variables={
        "input-border-default": "#FFFFAA",      # Light Yellow for input border
        "button-default-foreground": "#C71585", # Medium Violet Red for default button text
        "header-background": "#FFB6C1",
        "header-foreground": "#4B0082",
        "statusbar-background": "#4B0082",
        "statusbar-foreground": "#B0E0E6",      # Soft Cyan for status bar text
    },
)

# 18. Theme: "Glitch Grid"
glitch_grid_theme = Theme(
    name="glitch_grid",
    primary="#007FFF",       # Electric Blue (default buttons)
    secondary="#9400D3",     # Glitchy Purple (system critical buttons)
    accent="#FF1493",        # Hot Pink (input focus, error/warning)
    foreground="#00FF00",    # Bright Green
    background="#000000",    # True Black
    surface="#1A1A1A",       # Dark Gray (default button background)
    panel="#0A0A0A",         # Almost Black (input backgrounds)
    success="#00DD00",       # Slightly different Bright Green
    warning="#FF1493",       # Hot Pink
    error="#FF1493",         # Hot Pink
    dark=True,
    variables={
        "input-border-default": "#9400D3",             # Glitchy Purple for input border
        "button-system-critical-foreground": "#00FF00",# Green text for critical buttons
        "button-system-critical-border": "#FF1493",   # Pink border for critical buttons
    },
)

# 19. Theme: "Paradise Virtua"
paradise_virtua_theme = Theme(
    name="paradise_virtua",
    primary="#FF007F",       # Magenta Rose (prominent interactive, borders)
    secondary="#8A8AFF",     # Lavender Blue (select buttons)
    accent="#FFA500",        # Orange Soda (input focus, button text)
    foreground="#AFEEEE",    # Light Aqua
    background="#005060",    # Dark Teal
    surface="#207080",       # Mid Teal (default button background)
    panel="#003040",         # Darker Teal (input backgrounds)
    success="#76D7C4",       # Aqua Green
    warning="#FFA500",       # Orange Soda
    error="#FF4D4D",         # Bright Red/Pink
    dark=True,
    variables={
        "button-default-foreground": "#FFA500", # Orange Soda for default button text
        "button-select-foreground": "#002030",  # Dark text for select buttons
        "titlebar-background": "#FF007F",       # Magenta Rose for title bar
        "titlebar-foreground": "#FFFFFF",       # White text for title bar
    },
)

# 20. Theme: "Lost Artifacts (Atari)"
lost_artifacts_atari_theme = Theme(
    name="lost_artifacts_atari",
    primary="#A08050",       # Muted Gold (prominent interactive, focus)
    secondary="#706080",     # Faded Purple (secondary interactive elements, borders)
    accent="#408080",        # Atari Teal (input border)
    foreground="#D0D0D0",    # Off-White
    background="#262A33",    # Very Dark Blue-Gray
    surface="#363A43",       # Darker Gray (default button background)
    panel="#1E222B",         # Even Darker Blue-Gray (input backgrounds)
    success="#609060",       # Muted Atari Green
    warning="#A08050",       # Muted Gold (can be warning too)
    error="#B07070",         # Dusty Rose
    dark=True,
    variables={
        "text-muted": "#70A0A0",                # Muted Cyan for descriptions
        "button-critical-background": "#B07070",# Dusty Rose for critical buttons
        "button-critical-foreground": "#FFFFFF",# White text for critical buttons
    },
)

# 21. Theme: "Green Phosphor Terminal"
green_phosphor_terminal_theme = Theme(
    name="green_phosphor_terminal",
    primary="#33FF33",       # Bright Green (interactive elements)
    secondary="#00AA00",     # Darker Green (borders, less important)
    accent="#66FF66",        # Lighter Bright Green (highlights, focus)
    foreground="#00FF00",    # Bright Green (main text)
    background="#000000",    # True Black
    surface="#001A00",       # Very Dark Green (button background)
    panel="#050505",         # Almost Black (input background)
    success="#33FF33",       # Bright Green
    warning="#FFFF00",       # Classic Terminal Yellow
    error="#FF3333",         # Classic Terminal Red
    dark=True,
    variables={
        "text-muted": "#00AA00",
        "text-highlight": "#66FF66",
        "log-output-background": "#020202",
        "log-output-foreground": "#00CF00",
    },
)

# 22. Theme: "Zero Day Stealth"
zero_day_stealth_theme = Theme(
    name="zero_day_stealth",
    primary="#3A6A8A",       # Muted Action Blue
    secondary="#4A5A7A",     # Desaturated Blue (hover, focus, borders)
    accent="#4A5A7A",        # Desaturated Blue (input focus, matches secondary)
    foreground="#A0B0C0",    # Light Gray-Blue
    background="#101015",    # Very Dark Charcoal
    surface="#1A1A20",       # Slightly Lighter Dark (default button background)
    panel="#0A0A0D",         # Even Darker (input background)
    success="#5A8A5A",       # Muted Green
    warning="#B0A070",       # Muted Yellow/Amber
    error="#8A4A4A",         # Muted Red
    dark=True,
    variables={
        "text-highlight": "#C0D0E0",
        "command-palette-background": "#15151A",
        "command-palette-border": "#4A5A7A",
    },
)

# 23. Theme: "Red Team Ops"
red_team_ops_theme = Theme(
    name="red_team_ops",
    primary="#FF3333",       # Bright Red
    secondary="#AA0000",     # Darker Red
    accent="#FF3333",        # Bright Red (input focus, matches primary)
    foreground="#CCCCCC",    # Light Gray
    background="#181818",    # Dark Gray, almost Black
    surface="#282828",       # Darker Gray (default button background)
    panel="#101010",         # Very Dark Gray (input background)
    success="#33CC33",       # Contrasting Bright Green
    warning="#FF3333",       # Bright Red
    error="#FF0000",         # Even brighter/pure Red for distinct error
    dark=True,
    variables={
        "input-border-default": "#882222", # Muted Red
        "progressbar-color": "#FF3333",
        "button-critical-action-foreground": "#FFFFFF",
    },
)

# 24. Theme: "Deep Dive Cyberspace"
deep_dive_cyberspace_theme = Theme(
    name="deep_dive_cyberspace",
    primary="#33CCFF",       # Lighter Electric Blue
    secondary="#0077CC",     # Medium Blue (hover, borders)
    accent="#33CCFF",        # Lighter Electric Blue (input focus, matches primary)
    foreground="#00FFFF",    # Bright Cyan
    background="#051025",    # Very Dark Blue
    surface="#002040",       # Dark Blue/Teal (default button background)
    panel="#000515",         # Even Darker Blue (input background)
    success="#33FF99",       # Contrasting Bright Mint Green
    warning="#FFFF33",       # Contrasting Bright Yellow
    error="#FF33AACC",       # Contrasting Bright Pink/Magenta
    dark=True,
    variables={
        "text-info": "#33CCFF",
        "input-border-default": "#0077CC",
        "button-execute-foreground": "#000020", # Dark text on light blue button
        "network-graph-background": "#0A1530",
        "network-graph-border": "#005577",
    },
)

# 25. Theme: "Ghost in the Shell" (Monochrome & Minimal)
ghost_in_the_shell_theme = Theme(
    name="ghost_in_the_shell",
    primary="#708090",       # SlateGray (focus accent, main interactive color)
    secondary="#555555",     # Medium Gray (borders, less active elements)
    accent="#708090",        # SlateGray (input focus, matches primary)
    foreground="#D0D0D0",    # Light Gray
    background="#222222",    # Dark Gray
    surface="#333333",       # Slightly Lighter Dark Gray (default button background)
    panel="#111111",         # Very Dark Gray (input background, status bar)
    success="#709080",       # Very subtle desaturated green
    warning="#A09070",       # Very subtle desaturated yellow/amber
    error="#907070",         # Very subtle desaturated red
    dark=True,
    variables={
        "text-muted": "#888888",
        "input-border-default": "#555555",
        "statusbar-background": "#111111",
        "statusbar-foreground": "#A0A0A0",
        "statusbar-border-top": "#555555", # Assuming this would map to a TCSS property
    },
)

from textual.theme import Theme

# 1. Theme: "Retro Mint Chip" (Light)
# A refreshing light theme with minty greens, chocolate browns, and creamy off-whites.
retro_mint_chip_theme = Theme(
    name="retro_mint_chip",
    primary="#68B0AB",       # Mint Green (for interactive elements, focus)
    secondary="#A0937D",     # Lighter Brown/Taupe (secondary elements, borders)
    accent="#3D8B83",        # Darker Teal/Mint (input highlights, accents)
    foreground="#5E454B",    # Dark Chocolate Brown (main text)
    background="#F5EFE6",    # Creamy Off-White (main background)
    surface="#E8DFD8",       # Slightly darker cream (default button background, widget surface)
    panel="#F0E6DB",         # Lighter panel background (input backgrounds, distinct panels)
    success="#5A9C78",       # Muted Jade Green (success messages)
    warning="#FFA07A",       # Light Salmon/Peach (warning messages)
    error="#D2691E",         # Chocolate Brown/Sienna (error messages)
    dark=False,
    variables={
        "text-muted": "#8C7853",              # Muted brown for less important text
        "input-placeholder": "#BEB2A7",       # Lighter placeholder text
        "border-subtle": "#D1C7B7",           # Subtle border color
        "scrollbar-color": "#68B0AB",         # Primary color for scrollbar
    },
)

# 2. Theme: "Blueprint Tech" (Dark)
# A dark theme inspired by technical blueprints, featuring deep blues, cyans, and yellow accents.
blueprint_tech_theme = Theme(
    name="blueprint_tech",
    primary="#00BFFF",       # Bright Cyan (DeepSkyBlue)
    secondary="#4682B4",     # Medium Blue (SteelBlue)
    accent="#FFD700",        # Yellow (Gold)
    foreground="#E0FFFF",    # Light Cyan (main text)
    background="#000080",    # Deep Blue (Navy - main background)
    surface="#0000CD",       # Medium Blue (button background, widget surface)
    panel="#000050",         # Even Darker Blue (input backgrounds, panels)
    success="#32CD32",       # Lime Green
    warning="#FFA500",       # Orange
    error="#FF4500",         # OrangeRed
    dark=True,
    variables={
        "text-highlight": "#FFFFFF",           # White for important highlights
        "grid-line-major": "#3030A0",        # For schematic-like grids
        "grid-line-minor": "#181860",        # Fainter grid lines
        "code-background": "#000060",         # Background for code blocks
    },
)

# 3. Theme: "Vintage Scroll" (Light)
# A light theme evoking aged paper, rich inks, and classic bookbinding.
vintage_scroll_theme = Theme(
    name="vintage_scroll",
    primary="#8B0000",       # Burgundy Red (DarkRed - for interactive elements)
    secondary="#B8860B",     # Muted Gold (DarkGoldenrod - borders, secondary elements)
    accent="#556B2F",        # Dark Olive Green (accents, input highlights)
    foreground="#5D4037",    # Dark Sepia Brown (main text)
    background="#F5F5DC",    # Aged Paper (Beige - main background)
    surface="#EEE8AA",       # Pale Goldenrod (button background, widget surface)
    panel="#FAF0E6",         # Linen (input backgrounds, panels)
    success="#228B22",       # Forest Green
    warning="#DAA520",       # Goldenrod
    error="#A52A2A",         # Brown (for error messages)
    dark=False,
    variables={
        "text-link": "#0000CD",               # Medium Blue for hyperlinks
        "blockquote-background": "#F0EAD6",  # Slightly different bg for blockquotes
        "blockquote-border": "#D2B48C",       # Tan border for blockquotes
        "header-color": "#8B0000",            # Primary color for headers
    },
)

# 4. Theme: "Twilight Lavender Fields" (Dark)
# A calming dark theme with shades of lavender, purple, and soft pink accents.
twilight_lavender_fields_theme = Theme(
    name="twilight_lavender_fields",
    primary="#9370DB",       # Medium Lavender (MediumPurple - interactive elements)
    secondary="#6A5ACD",     # Muted Purple (SlateBlue - borders, secondary)
    accent="#FFB6C1",        # Soft Pink (LightPink - highlights, input focus)
    foreground="#E6E6FA",    # Light Lavender (main text)
    background="#301934",    # Deep Purple/Indigo (main background)
    surface="#483D8B",       # Dark Slate Blue (button background, widget surface)
    panel="#200020",         # Very Dark Purple (input backgrounds, panels)
    success="#3CB371",       # Medium Sea Green
    warning="#FFDAB9",       # PeachPuff (Pale Yellowish-Pink)
    error="#DB7093",         # PaleVioletRed (Dusky Rose/Muted Magenta)
    dark=True,
    variables={
        "text-muted": "#B0A4C4",              # Muted light purple for less important text
        "focus-border": "#FFB6C1",            # Accent color for focus borders
        "progress-bar-color": "#9370DB",      # Primary color for progress bars
        "tooltip-background": "#483D8B",      # Surface color for tooltips
    },
)

# 5. Theme: "Golden Hour Desert" (Light)
# A warm light theme inspired by desert landscapes at sunset, with sandy tones, burnt orange, and sky blue.
golden_hour_desert_theme = Theme(
    name="golden_hour_desert",
    primary="#CC5500",       # Burnt Orange
    secondary="#D4A017",     # Muted Yellow/Ochre (Golden Ochre)
    accent="#87CEEB",        # Sky Blue
    foreground="#4A3B31",    # Dark Brown/Charcoal (main text)
    background="#F0E68C",    # Sandy Beige (Khaki - main background)
    surface="#E6D8AD",       # Lighter, slightly desaturated sand (button background)
    panel="#FAF3D2",         # Very light sand (input backgrounds)
    success="#556B2F",       # Olive Green (DarkOliveGreen)
    warning="#FF8C00",       # Deep Orange (DarkOrange)
    error="#B22222",         # Rusty Red (Firebrick)
    dark=False,
    variables={
        "text-muted": "#8B7D6B",                # Muted sandy brown
        "button-primary-hover-background": "#E06000", # Darker orange for hover
        "link-color": "#007BA7",              # Cerulean blue for links
        "border-strong": "#A0522D",           # Sienna for stronger borders
    },
)

# 6. Theme: "Industrial Gearworks" (Dark)
# A stark, functional dark theme with various grays, steel blue, and metallic bronze/copper accents.
industrial_gearworks_theme = Theme(
    name="industrial_gearworks",
    primary="#5F7D8E",       # Muted Steel Blue
    secondary="#708090",     # Medium Gray (SlateGray)
    accent="#B87333",        # Bronze/Copper
    foreground="#D1D0CE",    # Light Steel Gray (main text)
    background="#2B2B2B",    # Dark Charcoal Gray (main background)
    surface="#3C3C3C",       # Slightly lighter dark gray (button background)
    panel="#1E1E1E",         # Even Darker Gray (input backgrounds)
    success="#38761D",       # Dark Green (like coated machinery)
    warning="#FFCC00",       # Industrial Yellow (bright, like hazard tape)
    error="#CC3333",         # Industrial Red
    dark=True,
    variables={
        "text-highlight": "#FFFFFF",             # White for critical highlights
        "border-heavy": "#A9A9A9",             # DarkGray for heavier borders
        "widget-border": "#505050",             # Standard widget border
        "tooltip-background": "#4A4A4A",        # Tooltip background
    },
)

# 7. Theme: "Coral Bloom" (Light)
# A bright and vibrant light theme inspired by coral reefs, with coral pink, seafoam green, and sunny yellow.
coral_bloom_theme = Theme(
    name="coral_bloom",
    primary="#FF7F50",       # Bright Coral Pink
    secondary="#40E0D0",     # Seafoam Green (Turquoise)
    accent="#FFEB3B",        # Sunny Yellow (Material Yellow A200)
    foreground="#2F4F4F",    # Dark Slate Gray (main text)
    background="#F0FFFF",    # Very Light Aqua (Azure - main background)
    surface="#E0FFFF",       # Light Cyan (button background)
    panel="#FAFFFF",         # Almost white panel (input backgrounds)
    success="#32CD32",       # Lime Green
    warning="#FFA500",       # Orange
    error="#FF4081",         # Bright Pink (Material Pink A200)
    dark=False,
    variables={
        "text-muted": "#778899",                 # LightSlateGray
        "list-item-active-background": "#FFEBCD", # BlanchedAlmond (warm highlight)
        "highlight-primary": "#FF7F50 30%",      # Primary color with alpha for selections
        "badge-background": "#40E0D0",           # Secondary for badges
        "badge-foreground": "#2F4F4F",           # Dark text on badge
    },
)

# 8. Theme: "Autumn Embers" (Dark)
# A warm and cozy dark theme with fiery oranges, deep reds, golden yellows, and rich browns.
autumn_embers_theme = Theme(
    name="autumn_embers",
    primary="#E67E22",       # Fiery Orange (Carrot)
    secondary="#C04000",     # Muted Red (Maroon variant)
    accent="#F1C40F",        # Golden Yellow (Sunflower)
    foreground="#FDF5E6",    # Creamy Beige (OldLace - main text)
    background="#362419",    # Deep Brown/Charcoal (main background)
    surface="#5C3D2E",       # Rich Brown (button background)
    panel="#2A1B10",         # Even Darker Brown (input backgrounds)
    success="#2E8B57",       # Forest Green (SeaGreen)
    warning="#E69B00",       # Amber (Dark Goldenrod variant)
    error="#9B1B1B",         # Deep Burgundy/Crimson
    dark=True,
    variables={
        "text-highlight": "#FFF0A5",           # Light yellow highlight
        "log-date-foreground": "#F1C40F",      # Golden yellow for dates in logs
        "code-comment-color": "#A0522D",       # Sienna for code comments
        "button-hover-text": "#FFFFFF",
    },
)

# 9. Theme: "Zen Garden" (Light)
# A calm and minimalist light theme with stone grays, muted jade green, sandy beige, and a touch of water blue.
zen_garden_theme = Theme(
    name="zen_garden",
    primary="#8FBC8F",       # Muted Jade Green (DarkSeaGreen)
    secondary="#D2B48C",     # Sandy Beige (Tan)
    accent="#ADD8E6",        # Water Blue (LightBlue)
    foreground="#36454F",    # Dark Charcoal Gray (main text)
    background="#F5F5F5",    # Light Stone Gray (WhiteSmoke - main background)
    surface="#E9E9E9",       # Slightly darker light gray (button background)
    panel="#FFFFFF",         # White (input backgrounds, panels for clean look)
    success="#90EE90",       # Soft Green (LightGreen)
    warning="#F0E68C",       # Pale Yellow (Khaki)
    error="#CD5C5C",         # Terracotta/Muted Red (IndianRed)
    dark=False,
    variables={
        "text-muted": "#707070",                  # Medium gray for muted text
        "border-zen": "#C0C0C0",                # Silver for subtle borders
        "input-focus-border": "#8FBC8F",        # Primary color for input focus border
        "container-background-alt": "#ECECEC",   # Alternative background for containers
    },
)

# 10. Theme: "Nebula Dreams" (Dark)
# A cosmic dark theme featuring very dark purples, with vibrant magenta, teal, and electric blue accents like distant nebulae.
nebula_dreams_theme = Theme(
    name="nebula_dreams",
    primary="#C71585",       # Electric Magenta (MediumVioletRed)
    secondary="#008080",     # Deep Teal
    accent="#1E90FF",        # Bright Cosmic Blue (DodgerBlue)
    foreground="#AFEEEE",    # Pale Turquoise/Silver (main text)
    background="#0B0014",    # Very Dark Purple/Almost Black (main background)
    surface="#1A0033",       # Dark Purple (button background)
    panel="#050008",         # Extremely Dark Purple/Black (input backgrounds)
    success="#39FF14",       # Luminous Green (Neon Green)
    warning="#FFFF33",       # Pulsar Yellow (Canary Yellow)
    error="#FF004F",         # Red Giant (Neon Red/Strong Pink)
    dark=True,
    variables={
        "star-highlight": "#FFFFFF",                   # Bright white for star-like highlights
        "text-accent-glow": "#1E90FF 40%",           # Accent color with transparency for a glow effect
        "scrollbar-thumb-color": "#C71585",          # Primary color for scrollbar thumb
        "command-palette-selected-background": "#400080", # Darker purple for selections
    },
)

# 1. Theme: "Volcanic Ash & Lava" (Dark)
# A dramatic theme with molten oranges and reds against dark, ashen grays.
volcanic_ash_lava_theme = Theme(
    name="volcanic_ash_lava",
    primary="#FF5A00",       # Molten Orange (interactive elements, focus)
    secondary="#B22222",     # Firebrick Red (secondary elements, borders)
    accent="#FFD700",        # Gold/Sulphur Yellow (highlights, input focus)
    foreground="#E0E0E0",    # Light Ash Grey (main text)
    background="#1A1A1A",    # Dark Ash (main background)
    surface="#2C2C2C",       # Slightly Lighter Ash (button background, widget surface)
    panel="#101010",         # Obsidian Black (input backgrounds, distinct panels)
    success="#6AB04C",       # Muted Green (like new growth after eruption)
    warning="#F0AD4E",       # Sulphur Yellow/Orange (warning messages)
    error="#D9534F",         # Smoldering Red (error messages)
    dark=True,
    variables={
        "tooltip-background": "#3D3D3D",          # Darker tooltip background
        "border-accent": "#FF5A00",             # Use primary for prominent borders
        "text-muted": "#777777",                  # Muted grey for less important text
        "scrollbar-color-hover": "#FF8C00",     # DarkOrange for scrollbar hover
    },
)

# 2. Theme: "Spring Meadowburst" (Light)
# A vibrant light theme bursting with fresh greens, sky blues, and floral accents.
spring_meadowburst_theme = Theme(
    name="spring_meadowburst",
    primary="#7FFF00",       # Chartreuse/Vibrant Green (interactive elements)
    secondary="#87CEFA",     # Light Sky Blue (borders, secondary elements)
    accent="#FF69B4",        # Hot Pink/Floral Accent (highlights, input focus)
    foreground="#2E4020",    # Dark Forest Green (main text for contrast)
    background="#F0FFF0",    # Honeydew (very light green, main background)
    surface="#E0FEE0",       # Lighter Mint Cream (button background, widget surface)
    panel="#FAFFAF",         # Pale Yellow/Sunlight (input backgrounds, panels)
    success="#4CAF50",       # Healthy Green (success messages)
    warning="#FFC107",       # Sunny Orange/Amber (warning messages)
    error="#C00000",         # Deep Berry Red (error messages)
    dark=False,
    variables={
        "link-color": "#1976D2",                # Standard blue for links
        "text-highlight-bg": "#FFFFB3",         # Pale yellow for text selection background
        "input-border-active": "#7FFF00",       # Primary color for active input border
        "footer-background": "#D0E0D0",         # Light, earthy green for footer
    },
)

# 3. Theme: "80s Arcade Carpet" (Dark)
# A retro dark theme with neon pinks, electric blues, and bright yellows on a deep purple base, like an old arcade floor.
eighties_arcade_carpet_theme = Theme(
    name="80s_arcade_carpet",
    primary="#FF00FF",       # Fuchsia/Neon Pink (interactive elements)
    secondary="#00FFFF",     # Aqua/Electric Blue (borders, secondary elements)
    accent="#FFFF00",        # Bright Yellow (highlights, input focus)
    foreground="#EAEAEA",    # Light Grey (main text)
    background="#1A001A",    # Deep Purple (main background)
    surface="#2A102A",       # Darker Purple (button background, widget surface)
    panel="#0D000D",         # Almost Black Purple (input backgrounds, panels)
    success="#00DD00",       # Neon Green (success messages)
    warning="#FFAB00",       # Neon Orange (warning messages)
    error="#FF3333",         # Bright Error Red (error messages)
    dark=True,
    variables={
        "scrollbar-color-active": "#FFFF00",    # Accent yellow for active scrollbar
        "command-palette-border": "#FF00FF",    # Primary pink for command palette border
        "grid-lines": "#330033",                # Subtle dark purple grid lines
        "text-glow": "#00FFFF 50%",             # Secondary color with transparency for a glow
    },
)

# 4. Theme: "Ancient Papyrus" (Light)
# A warm, light theme evoking aged papyrus, with sepia tones, faded inks, and a touch of lapis lazuli.
ancient_papyrus_theme = Theme(
    name="ancient_papyrus",
    primary="#0047AB",       # Cobalt Blue (Lapis Lazuli inspired, for important interactive elements)
    secondary="#8B4513",     # Saddle Brown (faded ink, borders, secondary elements)
    accent="#B8860B",        # DarkGoldenrod (gold details, highlights)
    foreground="#5D4037",    # Dark Sepia (main text)
    background="#FDF5E6",    # Old Lace/Papyrus (main background)
    surface="#FAF0E6",       # Linen/Lighter Papyrus (button background, widget surface)
    panel="#F5EFE0",         # Even Lighter Papyrus/Scroll Edge (input backgrounds, panels)
    success="#556B2F",       # Dark Olive Green (preserved inscription color)
    warning="#CD853F",       # Peru/Light Brown (faded warning on papyrus)
    error="#A52A2A",         # Brown/Terracotta (damaged section color)
    dark=False,
    variables={
        "blockquote-text": "#4A2E20",          # Darker brown for blockquote text
        "header-underline-color": "#0047AB",  # Primary color for header underlines
        "text-muted": "#A08C78",                # Muted beige/brown for less important text
        "border-subtle": "#D2B48C",             # Tan for subtle borders
    },
)

# 5. Theme: "Urban Stealth Camo" (Muted Dark)
# A muted dark theme with desaturated urban grays, olive drab, and concrete accents.
urban_stealth_camo_theme = Theme(
    name="urban_stealth_camo",
    primary="#5A6D7C",       # Muted Cadet Blue (main interactive)
    secondary="#4B5340",     # Dark Olive Drab (borders, secondary)
    accent="#787269",        # Desaturated Stone Grey/Concrete (highlights, input focus)
    foreground="#A0A098",    # Light Grey/Off-White (main text)
    background="#282C34",    # Dark Slate Grey/Asphalt (main background)
    surface="#3A3F4B",       # Slightly Lighter Dark Slate (button background)
    panel="#1E2127",         # Very Dark Slate (input backgrounds)
    success="#4F7942",       # Muted Olive Green (success messages)
    warning="#B08D57",       # Desaturated Bronze/Khaki (warning messages)
    error="#9E4848",         # Dusky Muted Red (error messages)
    dark=True,
    variables={
        "text-dim": "#6C7A89",                  # Dimmed text color
        "button-outline-focus": "#787269",      # Accent for button focus outline
        "panel-border": "#404552",              # Border for panels
        "placeholder-text": "#525860",          # Darker placeholder text
    },
)

# 6. Theme: "Confectionery Bliss" (Playful Light)
# A sweet and playful light theme with pastel pinks, blues, yellows, and a contrasting deep lavender text.
confectionery_bliss_theme = Theme(
    name="confectionery_bliss",
    primary="#FFB6C1",       # Light Pink/Bubblegum (interactive elements)
    secondary="#AFEEEE",     # Pale Turquoise/Mint Frosting (borders, secondary)
    accent="#FFFACD",        # Lemon Chiffon/Yellow Candy (highlights, input focus)
    foreground="#6A4C9C",    # Deep Lavender/Grape (main text for readability)
    background="#F0F8FF",    # AliceBlue (very pale, main background)
    surface="#FFF0F5",       # LavenderBlush (pale pink surface for buttons)
    panel="#F5FFFA",         # MintCream (pale mint panel for inputs)
    success="#90EE90",       # Light Green/Pistachio (success messages)
    warning="#FFD700",       # Gold/Butterscotch (warning messages)
    error="#F08080",         # Light Coral/Cherry Red (error messages)
    dark=False,
    variables={
        "input-background-hover": "#FFFFFF",    # White for input hover
        "text-flavor-berry": "#800080",       # Purple for special flavor text
        "highlight-secondary": "#AFEEEE 40%",  # Secondary color with alpha for selection
        "button-text-primary": "#483263",     # Darker purple for text on primary buttons
    },
)

# 7. Theme: "Mystic Redwood Grove" (Earthy Dark)
# A deep, earthy dark theme inspired by ancient forests, with dark greens, rich browns, and bioluminescent accents.
mystic_redwood_grove_theme = Theme(
    name="mystic_redwood_grove",
    primary="#3AAFA9",       # Muted Teal/Glowing Moss (interactive)
    secondary="#4A3F35",     # Dark Bark Brown (borders, secondary)
    accent="#F2A900",        # Golden Chanterelle/Muted Orange Glow (highlights)
    foreground="#C5D1C0",    # Pale Lichen Green/Mist (main text)
    background="#1B262C",    # Very Dark Blue-Green/Forest Night (main background)
    surface="#2D3A40",       # Darker Forest Floor (button background)
    panel="#11181C",         # Deepest Shadow (input backgrounds)
    success="#5FAD56",       # Fern Green (success messages)
    warning="#E9A800",       # Golden/Amber Glow (warning messages)
    error="#C1440E",         # Russet/Decay (error messages)
    dark=True,
    variables={
        "tree-guide-line": "#4A3F35",           # Brown for tree widget guides
        "log-level-debug": "#3AAFA9",           # Primary color for debug logs
        "code-background": "#243137",           # Background for code blocks
        "text-ephemeral": "#88B0A4",            # Muted teal for less important text
    },
)

# 8. Theme: "Art Deco Metropolis" (Elegant Dark)
# An elegant dark theme featuring rich golds, silvers, and jewel tones against black or charcoal.
art_deco_metropolis_theme = Theme(
    name="art_deco_metropolis",
    primary="#CAA472",       # Muted Gold/Brass (interactive elements)
    secondary="#607D8B",     # Blue Grey/Steel (borders, secondary elements)
    accent="#006A4E",        # Dark Emerald Green (highlights, input focus)
    foreground="#E1D9D1",    # Alabaster/Off-White (main text)
    background="#121212",    # Onyx Black (main background)
    surface="#282828",       # Charcoal Grey (button background, widget surface)
    panel="#0A0A0A",         # Pitch Black (input backgrounds, panels)
    success="#2E7D32",       # Subtle Dark Green (success messages)
    warning="#B7950B",       # Antiqued Gold (warning messages)
    error="#981E19",         # Deep Crimson (error messages)
    dark=True,
    variables={
        "border-metallic": "#8D99AE",           # Silver/metallic border color
        "text-title": "#CAA472",                # Primary color for titles
        "widget-highlight-border": "#006A4E",   # Accent color for special widget borders
        "input-selection": "#455A64 50%",      # BlueGrey with alpha for input selection
    },
)

# 9. Theme: "Desert Oasis Mirage" (Light/Contrast)
# A light theme contrasting arid desert sands with vibrant oasis teals and sun-baked terracotta.
desert_oasis_mirage_theme = Theme(
    name="desert_oasis_mirage",
    primary="#00A2CA",       # Vibrant Cerulean/Deep Oasis Water (interactive)
    secondary="#E07A5F",     # Terracotta/Clay (borders, secondary)
    accent="#F4D35E",        # Pale Sun Yellow/Mirage Glow (highlights)
    foreground="#5C3D2E",    # Dark Sun-baked Earth (main text)
    background="#FCF5E5",    # Pale Sand/Ivory (main background)
    surface="#F8EDD8",       # Light Dune (button background)
    panel="#FFFBF0",         # Whitewashed Sand (input backgrounds)
    success="#50C878",       # Emerald Green/Lush Growth (success messages)
    warning="#FFBF00",       # Amber/Strong Sun (warning messages)
    error="#C12E2E",         # Sunburnt Red/Danger (error messages)
    dark=False,
    variables={
        "water-text": "#00A2CA",                # Primary color for water-themed text
        "sand-dune-highlight": "#FAF0C8",       # Very light sand for highlights
        "text-muted": "#8B7965",                # Muted sandy brown
        "border-strong": "#A0522D",             # Sienna for stronger borders
    },
)

# 10. Theme: "Starlight Cinema Noir" (Monochromatic + Accent, Dark)
# A high-contrast dark theme inspired by classic film noir, using deep blacks, bright whites, and a striking cinematic red accent.
starlight_cinema_noir_theme = Theme(
    name="starlight_cinema_noir",
    primary="#E50914",       # Cinematic Red (e.g., Netflix red, for main interactions)
    secondary="#666666",     # Mid-Dark Grey (supporting elements, borders)
    accent="#B0B0B0",        # Silver Screen Grey (subtle highlights, input focus)
    foreground="#F5F5F5",    # Bright White/Projector Light (main text)
    background="#0D0D0D",    # Deep Black/Velvet Curtain (main background)
    surface="#222222",       # Dark Stage Grey (button background, widget surface)
    panel="#000000",         # Pure Black/Void (input backgrounds, panels)
    success="#3DD13D",       # Green Screen Green (playful success)
    warning="#FFAA00",       # Amber Warning Light (classic warning)
    error="#D40000",         # Darker, Intense Red (error messages)
    dark=True,
    variables={
        "focus-outline": "#E50914",             # Primary red for focus outlines
        "text-subtle-contrast": "#CCCCCC",      # Lighter grey for subtitles or less critical text
        "button-critical-background": "#990000",# Darker red for critical action buttons
        "dialog-border": "#444444",             # Border for dialogs
    },
)

# 1. Theme: "Eva Unit-01 Protocol" (Dark)
# Inspired by Evangelion Unit-01: deep purples, vibrant greens, stark blacks, and alert oranges.
eva_unit_01_protocol_theme = Theme(
    name="eva_unit_01_protocol",
    primary="#7CFC00",       # Luminous Bio-Green (interactive, focus)
    secondary="#6A0DAD",     # Deep Armor Purple (secondary elements, borders)
    accent="#FF8C00",        # Alert Orange/Core (highlights, input focus)
    foreground="#D0E0D0",    # Pale Sync Interface Grey (main text)
    background="#100C14",    # NERV Command Black (main background)
    surface="#2A1F36",       # Dark Purple Bio-Plating (button background)
    panel="#08060A",         # Deepest Cockpit Shadow (input backgrounds)
    success="#32CD32",       # A.T. Field Green (success messages)
    warning="#FFA500",       # Warning Signal Orange (warning messages)
    error="#DC143C",         # Angel Core Crimson (error messages)
    dark=True,
    variables={
        "sync-ratio-text": "#98FB98",            # Pale green for status readouts
        "plugsuit-highlight": "#FFA500 30%",   # Orange with alpha for selection
        "berserk-mode-indicator": "#FF0000",   # Pure Red for critical alerts
        "lclinstrumentation-text": "#87CEFA",   # Light Sky Blue for LCL-like elements
    },
)

# 2. Theme: "Ghibli's Hidden Forest" (Light/Earthy)
# Soft, natural tones inspired by Studio Ghibli's lush landscapes.
ghiblis_hidden_forest_theme = Theme(
    name="ghiblis_hidden_forest",
    primary="#6B8E23",       # Mossy Olive Green (interactive elements)
    secondary="#87CEEB",     # Clear Sky Blue (secondary elements, borders)
    accent="#D2B48C",        # Aged Wood Tan (highlights, input focus)
    foreground="#4A3B31",    # Rich Earth Brown (main text)
    background="#F5F5DC",    # Sun-dappled Beige (main background)
    surface="#E0EEDD",       # Pale Leaf Green (button background)
    panel="#FAF0E6",         # Linen/Light Cloud (input backgrounds)
    success="#3CB371",       # Healthy Forest Green (success messages)
    warning="#F4A460",       # Sandy Brown Warning (warning messages)
    error="#CD5C5C",         # Faded Berry Red (error messages)
    dark=False,
    variables={
        "river-flow-text": "#5F9EA0",           # Cadet blue for water elements
        "soot-sprite-dot": "#2F4F4F",           # Dark slate gray for tiny details
        "kodama-white": "#FFFAFA",              # Snow white for spirit-like accents
        "sunbeam-yellow": "#FFFACD",            # Lemon chiffon for light effects
    },
)

# 3. Theme: "Magical Girl Transform!" (Bright/Pastel Light)
# Vibrant pastels of pink, blue, and yellow, full of sparkle and energy.
magical_girl_transform_theme = Theme(
    name="magical_girl_transform",
    primary="#FF69B4",       # Bright Transformation Pink (interactive elements)
    secondary="#ADD8E6",     # Cosmic Sky Blue (borders, secondary elements)
    accent="#FFD700",        # Star Power Gold (highlights, input focus)
    foreground="#800080",    # Royal Purple (for contrasty text)
    background="#FFF0F5",    # Lavender Blush/Cloud White (main background)
    surface="#FFC0CB",       # Lighter Pink Ribbon (button background)
    panel="#E6E6FA",         # Pale Lavender Dream (input backgrounds)
    success="#98FB98",       # Minty Fresh Green (success messages)
    warning="#FFA07A",       # Light Salmon/Peach (warning messages)
    error="#DB7093",         # Rose Quartz Red (error messages)
    dark=False,
    variables={
        "sparkle-effect-color": "#FFFFFF",       # Pure white for sparkles
        "friendship-aura-bg": "#FFB6C1 30%",   # Primary pink with alpha for highlights
        "compact-mirror-border": "#C0C0C0",     # Silver for item borders
        "mascot-guide-text": "#4682B4",         # Steel blue for helper text
    },
)

# 4. Theme: "Jujutsu Sorcery Night" (Dark)
# Deep indigos, cursed energy purples, and blood-red accents of Jujutsu Kaisen.
jujutsu_sorcery_night_theme = Theme(
    name="jujutsu_sorcery_night",
    primary="#C71585",       # Cursed Energy Magenta (interactive)
    secondary="#2A0D45",     # Deepest Indigo Veil (borders, secondary)
    accent="#B22222",        # Blood Pact Red (highlights, input focus)
    foreground="#D8BFD8",    # Pale Thistle/Spirit Light (main text)
    background="#0B071A",    # Tokyo Night Black (main background)
    surface="#1E1030",       # Shadow Technique Purple (button background)
    panel="#05030D",         # Void Black (input backgrounds)
    success="#00A86B",       # Jade Vine Green (positive outcome)
    warning="#E09200",       # Cursed Seal Amber (warning messages)
    error="#8B0000",         # Malevolent Shrine Dark Red (error messages)
    dark=True,
    variables={
        "domain-expansion-field": "#4B0082 20%", # Indigo with alpha for fields
        "cursed-tool-sheen": "#A9A9A9",          # Dark grey for metallic tools
        "shikigami-outline": "#708090",         # Slate grey for summoned outlines
        "sukuna-tattoo-red": "#800000",         # Darker red for specific accents
    },
)

# 5. Theme: "Akira Neo-Tokyo Grit" (Dark)
# The iconic crimson red, concrete grays, and neon highlights of Akira's dystopian world.
akira_neo_tokyo_grit_theme = Theme(
    name="akira_neo_tokyo_grit",
    primary="#D70000",       # Iconic Akira Crimson (interactive elements)
    secondary="#4A4A4A",     # Urban Concrete Grey (borders, secondary)
    accent="#00E5FF",        # Capsule Gang Neon Cyan (highlights, input focus)
    foreground="#E0E0E0",    # Dusty Billboard White (main text)
    background="#1C1C1C",    # Asphalt Black (main background)
    surface="#333333",       # Rubble Grey (button background)
    panel="#0F0F0F",         # Deep Shadow Alley (input backgrounds)
    success="#39FF14",       # Glitch Green (success, digital feel)
    warning="#FFBF00",       # Hazard Yellow (warning messages)
    error="#B20000",         # Destructive Red (error messages)
    dark=True,
    variables={
        "kanedas-bike-red": "#FF0000",             # Brighter red for key highlights
        "espers-power-glow": "#8A2BE2 40%",      # BlueViolet with alpha
        "government-data-blue": "#3366CC",       # Official blue for specific text
        "city-lights-yellow": "#FFFF66",         # Pale yellow for distant lights
    },
)

# 6. Theme: "Scouting Legionnaire" (Muted/Desaturated Dark)
# The somber, utilitarian colors of Attack on Titan's Survey Corps.
scouting_legionnaire_theme = Theme(
    name="scouting_legionnaire",
    primary="#556B2F",       # Survey Corps Dark Olive (interactive)
    secondary="#8B4513",     # Worn Leather Brown (borders, secondary)
    accent="#A0522D",        # Faded Sienna/Terracotta (highlights, input focus)
    foreground="#D2D2C0",    # Old Parchment/Report Paper (main text)
    background="#2F2F28",    # Wall Sina Stone Grey (main background)
    surface="#3E3E35",       # Barracks Wood Brown (button background)
    panel="#1A1A17",         # Deep Forest Shadow (input backgrounds)
    success="#6B8E23",       # Hopeful Green Sprout (success messages)
    warning="#BDB76B",       # Cautionary Dark Khaki (warning messages)
    error="#800000",         # Titan Blood Maroon (error messages)
    dark=True,
    variables={
        "map-lines-color": "#A0522D",           # Sienna for map details
        "uniform-strap-color": "#704214",       # Darker brown for details
        "titan-steam-white": "#F5F5F5 30%",     # Off-white with alpha for steam
        "flare-gun-green": "#2E8B57",           # SeaGreen for signals
    },
)

# 7. Theme: "Saiyan Power Orange" (Vibrant/Energetic Light)
# The iconic orange and blue of Goku's gi, bursting with Dragon Ball Z energy.
saiyan_power_orange_theme = Theme(
    name="saiyan_power_orange",
    primary="#FF8C00",       # Goku's Gi Orange (interactive)
    secondary="#1E90FF",     # Capsule Corp Blue (borders, secondary)
    accent="#FFD700",        # Super Saiyan Gold (highlights, input focus)
    foreground="#000080",    # Deep Training Blue/Black (main text for contrast)
    background="#F0F8FF",    # Clear Sky AliceBlue (main background)
    surface="#FFA500",       # Lighter Training Orange (button background)
    panel="#FFFFFF",         # Nimbus Cloud White (input backgrounds)
    success="#32CD32",       # Senzu Bean Green (success messages)
    warning="#FFFF00",       # Ki Blast Yellow (warning messages)
    error="#FF0000",         # Battle Damage Red (error messages)
    dark=False,
    variables={
        "dragonball-star-red": "#FF4500",        # OrangeRed for Dragon Ball stars
        "scouter-display-green": "#00FF00",      # Bright green for scouter text
        "kame-house-roof": "#B22222",            # Firebrick for Namekian details
        "training-ground-earth": "#DEB887",      # BurlyWood for environment colors
    },
)

# 8. Theme: "Psycho-Pass Dominator" (Cool/Futuristic Dark)
# Clinical blues, cyans, and grays with the distinct Dominator mode accents from Psycho-Pass.
psycho_pass_dominator_theme = Theme(
    name="psycho_pass_dominator",
    primary="#00BFFF",       # Sibyl System Cyan (interactive, non-lethal)
    secondary="#4682B4",     # Enforcer Steel Blue (borders, secondary)
    accent="#FF4500",        # Lethal Eliminator Orange-Red (critical highlights, input focus)
    foreground="#E0FFFF",    # Holographic Display Light Cyan (main text)
    background="#101820",    # MWPSB Dark Blue-Grey (main background)
    surface="#1C2833",       # Dominator Casing Grey (button background)
    panel="#0A0F14",         # Crime Coefficient Black (input backgrounds)
    success="#2ECC71",       # System Normal Green (success messages)
    warning="#F39C12",       # Paralyzer Mode Yellow (warning messages)
    error="#E74C3C",         # Crime Index Red (error messages)
    dark=True,
    variables={
        "crime-coefficient-value": "#FF4500",   # Accent color for critical numbers
        "latent-criminal-hue": "#B0E0E6 30%",  # PowderBlue with alpha for subtle highlights
        "inspector-badge-gold": "#FFD700",      # Gold for official symbols
        "area-scan-lines": "#008080 50%",      # Teal with alpha for scan effects
    },
)

# 9. Theme: "Sakura Viewing Picnic" (Pastel Light)
# Delicate cherry blossom pinks, fresh greens, and soft sky blues.
sakura_viewing_picnic_theme = Theme(
    name="sakura_viewing_picnic",
    primary="#FFB6C1",       # Sakura Blossom Pink (interactive)
    secondary="#90EE90",     # Fresh Sprout Green (borders, secondary)
    accent="#ADD8E6",        # Spring Sky Light Blue (highlights, input focus)
    foreground="#556B2F",    # Dark Moss Green (main text for readability)
    background="#FFF5EE",    # Seashell White/Pale Petal (main background)
    surface="#FFE4E1",       # Misty Rose/Lighter Pink (button background)
    panel="#F0FFF0",         # Honeydew Green/Picnic Blanket (input backgrounds)
    success="#3CB371",       # Healthy Leaf Green (success messages)
    warning="#FFDEAD",       # Dango NavahoWhite (warning messages)
    error="#F08080",         # Umeboshi Light Coral (error messages)
    dark=False,
    variables={
        "hanami-lantern-paper": "#FAFAD2",      # LightGoldenrodYellow for lanterns
        "tatami-mat-beige": "#EEE8AA",          # PaleGoldenrod for surfaces
        "koi-pond-blue": "#87CEFA",             # LightSkyBlue for decorative elements
        "falling-petal-accent": "#FFC0CB 60%",  # Pink with alpha for subtle effects
    },
)

# 10. Theme: "80s Anime OVA Sunset" (Dark/Nostalgic Neon)
# Inspired by the aesthetic of 80s OVAs: deep purples, magenta, teal, and sunset oranges.
eighties_anime_ova_sunset_theme = Theme(
    name="80s_anime_ova_sunset",
    primary="#F92C86",       # Retro Magenta (interactive)
    secondary="#03DAC5",     # Cassette Teal (borders, secondary)
    accent="#FFA500",        # Tokyo Tower Sunset Orange (highlights, input focus)
    foreground="#EAD7FE",    # Lavender Mist (main text)
    background="#241E4E",    # Midnight Drive Purple (main background)
    surface="#3B306A",       # Twilight Cityscape Indigo (button background)
    panel="#1A1438",         # VHS Tape Dark Purple (input backgrounds)
    success="#4CAF50",       # Arcade Screen Green (success messages)
    warning="#FFEB3B",       # Headlight Yellow (warning messages)
    error="#EF5350",         # Laser Disc Red (error messages)
    dark=True,
    variables={
        "city-skyline-glow": "#673AB7 30%",    # DeepPurple with alpha for background glow
        "boombox-chrome": "#B0BEC5",           # BlueGrey for metallic accents
        "lens-flare-white": "#FFFFFF 70%",     # White with alpha for highlights
        "character-outline-key": "#FF4081",    # Bright pink for strong outlines if needed
    },
)

agentic_terminal_theme = Theme(
    name="agentic_terminal",
    primary="#4CC9F0",
    secondary="#38BDF8",
    accent="#22D3EE",
    foreground="#D7E8F7",
    background="#08111F",
    surface="#101A2A",
    panel="#0D1626",
    success="#39D98A",
    warning="#FBBF24",
    error="#FF5A6A",
    dark=True,
    variables={
        "ds-surface-panel": "#0D1626",
        "ds-surface-raised": "#142033",
        "ds-surface-inspector": "#101A2A",
        "ds-grid-line": "#26364D",
        "ds-text-primary": "#D7E8F7",
        "ds-text-muted": "#7C8EA6",
        "ds-action-focus": "#4CC9F0",
        "ds-focus-fg": "#D7E8F7",
        "ds-focus-bg": "#142033",
        "ds-focus-accent": "#4CC9F0",
        "ds-input-focus-border": "#4CC9F0",
        "ds-input-focus-bg": "#142033",
        "ds-input-focus-accent": "#4CC9F0",
        "ds-status-ready": "#39D98A",
        "ds-status-running": "#4CC9F0",
        "ds-status-info": "#38BDF8",
        "ds-status-warning": "#FBBF24",
        "ds-status-approval-required": "#F59E0B",
        "ds-status-blocked": "#FF5A6A",
        "ds-status-error": "#FF5A6A",
        "ds-status-paused": "#C084FC",
        "ds-status-unsaved": "#FBBF24",
        "ds-status-recovered": "#34D399",
        "ds-authority-local": "#39D98A",
        "ds-authority-server": "#4CC9F0",
        "ds-authority-workspace": "#A78BFA",
        "ds-authority-remote-only": "#FBBF24",
        "ds-source-role-context": "#7C8EA6",
        "ds-source-role-evidence": "#22D3EE",
        "ds-source-role-editable-target": "#FBBF24",
        "ds-source-role-output-seed": "#A78BFA",
        "footer-key-foreground": "#4CC9F0",
        "input-selection-background": "#4CC9F0 35%",
        "text-muted": "#7C8EA6",
    },
)


ALL_THEMES = [
    agentic_terminal_theme,
    classic_terminal_green_theme,
    modern_dark_dracula_theme,
    paper_light_theme,
    high_contrast_yellow_black_theme,
    ocean_depths_theme,
    solarized_dark_theme,
    solarized_light_theme,
    monokai_pro_theme,
    gruvbox_dark_theme,
    gruvbox_light_theme,
    cyberpunk_neon_theme,
    earthy_nature_theme,
    pastel_dreams_theme,
    sweet_sorbet_theme,
    cloudy_day_theme,
    kawaii_candy_theme,
    bunny_fluff_theme,
    neon_sunset_drive_theme,
    palm_mall_theme,
    glitch_grid_theme,
    paradise_virtua_theme,
    lost_artifacts_atari_theme,
    green_phosphor_terminal_theme,
    zero_day_stealth_theme,
    red_team_ops_theme,
    deep_dive_cyberspace_theme,
    ghost_in_the_shell_theme,
    retro_mint_chip_theme,
    blueprint_tech_theme,
    vintage_scroll_theme,
    twilight_lavender_fields_theme,
    golden_hour_desert_theme,
    industrial_gearworks_theme,
    coral_bloom_theme,
    autumn_embers_theme,
    zen_garden_theme,
    nebula_dreams_theme,
    volcanic_ash_lava_theme,
    spring_meadowburst_theme,
    eighties_arcade_carpet_theme,
    ancient_papyrus_theme,
    urban_stealth_camo_theme,
    confectionery_bliss_theme,
    mystic_redwood_grove_theme,
    art_deco_metropolis_theme,
    desert_oasis_mirage_theme,
    starlight_cinema_noir_theme,
    eva_unit_01_protocol_theme,
    ghiblis_hidden_forest_theme,
    magical_girl_transform_theme,
    jujutsu_sorcery_night_theme,
    akira_neo_tokyo_grit_theme,
    scouting_legionnaire_theme,
    saiyan_power_orange_theme,
    psycho_pass_dominator_theme,
    sakura_viewing_picnic_theme,
    eighties_anime_ova_sunset_theme,
]

# Example of a theme with the 'variables' attribute as shown in Textual docs:
# MY_THEMES["arctic_example"] = Theme(
#     name="arctic_example",
#     primary=Color.parse("#88C0D0"),
#     secondary=Color.parse("#81A1C1"),
#     accent=Color.parse("#B48EAD"),
#     foreground=Color.parse("#D8DEE9"),
#     background=Color.parse("#2E3440"),
#     success=Color.parse("#A3BE8C"),
#     warning=Color.parse("#EBCB8B"),
#     error=Color.parse("#BF616A"),
#     surface=Color.parse("#3B4252"),
#     panel=Color.parse("#434C5E"),
#     dark=True,
#     variables={
#         "block-cursor-text-style": "none", # Needs to be a Style object if not a simple string
#         "footer-key-foreground": Color.parse("#88C0D0"),
#         "input-selection-background": Color.parse("#81a1c1").with_alpha(0.35),
#     },
# )

# Example of how you might collect them all for registration
# ALL_THEMES = [
#     classic_terminal_green_theme,
#     modern_dark_dracula_theme,
#     paper_light_theme,
#     high_contrast_yellow_black_theme,
#     ocean_depths_theme,
#     solarized_dark_theme,
#     solarized_light_theme,
#     monokai_pro_theme,
#     gruvbox_dark_theme,
#     gruvbox_light_theme,
#     cyberpunk_neon_theme,
#     earthy_nature_theme,
# ]
